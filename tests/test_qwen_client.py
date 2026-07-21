import json
from pathlib import Path

import pytest

from awareliquid.adapter.qwen_client import (
    DEFAULT_TOKEN_BUDGET,
    MockChatClient,
    QwenChatClient,
    TokenUsage,
    UsageLedger,
    build_chat_client,
    formal_network_denied_probe,
)


class _Response:
    def __init__(self, body):
        self.body = body

    def read(self):
        return json.dumps(self.body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False


def _response(model="qwen-plus", usage=None):
    return {
        "model": model,
        "choices": [{"message": {"content": "A"}}],
        "usage": usage or {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _formal_client(tmp_path, **kwargs):
    ledger_path = tmp_path / "usage.json"
    return QwenChatClient(
        api_key="test-key",
        competition_mode=True,
        formal_run_id=f"test-run-{tmp_path.name}",
        formal_ledger_path=str(ledger_path),
        **kwargs,
    )


def test_token_usage_add():
    a = TokenUsage(10, 2, 12)
    b = TokenUsage(5, 1, 6)
    c = a + b
    assert c.as_dict() == {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}


def test_mock_client_reports_usage_and_answers():
    client = MockChatClient()
    res = client.chat([{"role": "user", "content": "Options:\nA) yes\nB) no\nAnswer:"}])
    assert res.text == "A"
    assert res.usage.total_tokens > 0
    assert res.usage.total_tokens == res.usage.prompt_tokens + res.usage.completion_tokens


def test_missing_key_fails_closed_without_mock_fallback(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("transport called"))
    monkeypatch.delenv("AWARELIQUID_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("AWARELIQUID_LLM_BACKEND", "qwen")
    with pytest.raises(RuntimeError, match="API_KEY"):
        build_chat_client()


def test_build_respects_explicit_mock(monkeypatch):
    monkeypatch.setenv("AWARELIQUID_LLM_BACKEND", "mock")
    with pytest.raises(RuntimeError, match="explicit test mode"):
        build_chat_client()
    assert isinstance(build_chat_client(test_mode=True), MockChatClient)


def test_formal_network_requires_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("AWARELIQUID_ALLOW_FORMAL_NETWORK", "1")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(_response()),
    )

    result = _formal_client(tmp_path).chat(
        [{"role": "user", "content": "Answer A"}], max_tokens=8
    )

    assert result.text == "A"
    assert result.usage.total_tokens == 5


def test_usage_ledger_reserves_reconciles_and_persists(tmp_path):
    path = tmp_path / "usage.json"
    ledger = UsageLedger(max_tokens=100, path=str(path))
    reserved = ledger.reserve(30)
    ledger.reconcile(reserved, TokenUsage(10, 5, 15))
    assert ledger.snapshot() == {
        "used_tokens": 15,
        "reserved_tokens": 0,
        "remaining_tokens": 85,
        "max_tokens": 100,
    }
    reopened = UsageLedger(max_tokens=100, path=str(path))
    assert reopened.snapshot()["used_tokens"] == 15


def test_shared_persistent_ledger_reconciles_stale_clients_atomically(tmp_path):
    path = tmp_path / "shared-usage.json"
    # Both ledgers start from the same empty file. The second therefore has a
    # deliberately stale in-memory balance when the first commits its usage.
    first = UsageLedger(max_tokens=DEFAULT_TOKEN_BUDGET, path=str(path))
    second = UsageLedger(max_tokens=DEFAULT_TOKEN_BUDGET, path=str(path))

    first_reservation = first.reserve(0)
    second_reservation = second.reserve(0)
    first.reconcile(first_reservation, TokenUsage(2_000_000, 500_000, 2_500_000))
    second.reconcile(second_reservation, TokenUsage(2_000_000, 500_000, 2_500_000))

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "max_tokens": DEFAULT_TOKEN_BUDGET,
        "used_tokens": DEFAULT_TOKEN_BUDGET,
    }
    assert first.snapshot()["used_tokens"] == DEFAULT_TOKEN_BUDGET

    # A third stale instance must refresh under the same lock and reject an
    # over-budget aggregate rather than overwrite the committed 5M balance.
    third = UsageLedger(max_tokens=DEFAULT_TOKEN_BUDGET, path=str(path))
    third_reservation = third.reserve(0)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        third.reconcile(third_reservation, TokenUsage(1, 0, 1))
    third.release(third_reservation)
    assert json.loads(path.read_text(encoding="utf-8"))["used_tokens"] == DEFAULT_TOKEN_BUDGET


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "qwen-unapproved"}, "allowlisted"),
        ({"base_url": "http://dashscope.aliyuncs.com/compatible-mode/v1"}, "HTTPS"),
    ],
)
def test_formal_config_denials_happen_before_transport(tmp_path, monkeypatch, kwargs, message):
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("transport called"))
    with pytest.raises(ValueError, match=message):
        _formal_client(tmp_path, **kwargs)


def test_formal_config_accepts_third_party_https_qwen_endpoint(tmp_path):
    client = _formal_client(
        tmp_path,
        provider="third-party-gateway",
        base_url="https://gateway.example.invalid/openai/v1",
    )
    assert client.provider == "third-party-gateway"
    assert client.base_url == "https://gateway.example.invalid/openai/v1"


def test_formal_config_accepts_official_qwen36_plus_model(tmp_path):
    client = _formal_client(tmp_path, model="qwen3.6-plus")
    assert client.model == "qwen3.6-plus"


def test_formal_mode_requires_persistent_ledger():
    with pytest.raises(ValueError, match="formal_ledger_path"):
        QwenChatClient(api_key="test-key", competition_mode=True)


@pytest.mark.parametrize(
    "ledger_budget",
    [DEFAULT_TOKEN_BUDGET - 1, DEFAULT_TOKEN_BUDGET + 1],
)
def test_formal_mode_rejects_prebuilt_ledger_with_noncanonical_budget(tmp_path, ledger_budget):
    ledger = UsageLedger(max_tokens=ledger_budget, path=str(tmp_path / "usage.json"))
    with pytest.raises(ValueError, match="exact 5000000 token budget"):
        QwenChatClient(
            api_key="test-key",
            competition_mode=True,
            usage_ledger=ledger,
            formal_run_id=f"test-run-{tmp_path.name}",
            formal_ledger_path=str(tmp_path / "usage.json"),
        )


def test_formal_mode_rejects_noncanonical_token_budget(tmp_path):
    with pytest.raises(ValueError, match="exact 5000000 token budget"):
        QwenChatClient(
            api_key="test-key",
            competition_mode=True,
            token_budget=DEFAULT_TOKEN_BUDGET - 1,
            formal_run_id=f"test-run-{tmp_path.name}",
            formal_ledger_path=str(tmp_path / "usage.json"),
        )


def test_formal_clients_share_one_explicit_run_ledger_and_aggregate(tmp_path):
    run_id = f"aggregate-{tmp_path.name}"
    ledger_path = str(tmp_path / "run-usage.json")
    first = QwenChatClient(
        api_key="test-key",
        competition_mode=True,
        formal_run_id=run_id,
        formal_ledger_path=ledger_path,
    )
    second = QwenChatClient(
        api_key="test-key",
        competition_mode=True,
        formal_run_id=run_id,
        formal_ledger_path=ledger_path,
    )
    first_reservation = first.usage_ledger.reserve(0)
    second_reservation = second.usage_ledger.reserve(0)
    first.usage_ledger.reconcile(first_reservation, TokenUsage(2_000_000, 500_000, 2_500_000))
    second.usage_ledger.reconcile(second_reservation, TokenUsage(2_000_000, 500_000, 2_500_000))
    assert second.usage_ledger.snapshot()["used_tokens"] == DEFAULT_TOKEN_BUDGET

    third = QwenChatClient(
        api_key="test-key",
        competition_mode=True,
        formal_run_id=run_id,
        formal_ledger_path=ledger_path,
    )
    reservation = third.usage_ledger.reserve(0)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        third.usage_ledger.reconcile(reservation, TokenUsage(1, 0, 1))
    third.usage_ledger.release(reservation)


def test_formal_run_rejects_an_alternate_ledger_path(tmp_path):
    run_id = f"alternate-{tmp_path.name}"
    QwenChatClient(
        api_key="test-key",
        competition_mode=True,
        formal_run_id=run_id,
        formal_ledger_path=str(tmp_path / "first.json"),
    )
    with pytest.raises(ValueError, match="already bound"):
        QwenChatClient(
            api_key="test-key",
            competition_mode=True,
            formal_run_id=run_id,
            formal_ledger_path=str(tmp_path / "alternate.json"),
        )


def test_formal_network_probe_denies_without_urlopen_or_persistent_io(tmp_path, monkeypatch):
    def sentinel(*_args, **_kwargs):
        pytest.fail("formal network denial probe reached urllib")

    monkeypatch.setattr("urllib.request.urlopen", sentinel)
    monkeypatch.setattr("awareliquid.adapter.qwen_client.UsageLedger", lambda *args: pytest.fail("ledger initialized"))
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    formal_network_denied_probe()
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before


def test_formal_call_is_denied_before_transport(tmp_path, monkeypatch):
    client = _formal_client(tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("transport called"))
    monkeypatch.setattr("urllib.request.Request", lambda *_args, **_kwargs: pytest.fail("request constructed"))
    with pytest.raises(RuntimeError, match="network transport is denied"):
        client.chat([{"role": "user", "content": "offline only"}], max_tokens=1)
    assert client.usage_ledger.snapshot()["used_tokens"] == 0


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"prompt_tokens": 3, "completion_tokens": 2},
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 4},
        {"prompt_tokens": "3", "completion_tokens": 2, "total_tokens": 5},
    ],
)
def test_qwen_usage_must_be_known_complete_and_exact(tmp_path, monkeypatch, usage):
    client = QwenChatClient(
        api_key="test-key",
        usage_ledger_path=str(tmp_path / "usage.json"),
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        body = _response(usage=usage)
        if usage is None:
            body.pop("usage")
        return _Response(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="usage"):
        client.chat([{"role": "user", "content": "offline only"}], max_tokens=1)
    assert calls == ["https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"]
    assert client.usage_ledger.snapshot()["used_tokens"] == 0


def test_qwen_call_persists_exact_usage(tmp_path, monkeypatch):
    client = QwenChatClient(
        api_key="test-key",
        usage_ledger_path=str(tmp_path / "usage.json"),
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _Response(_response()))
    result = client.chat([{"role": "user", "content": "offline only"}], max_tokens=1)
    assert result.usage == TokenUsage(3, 2, 5)
    assert json.loads(Path(client.usage_ledger.path).read_text(encoding="utf-8")) == {
        "max_tokens": 5_000_000,
        "used_tokens": 5,
    }


def test_qwen_budget_denies_before_transport(tmp_path, monkeypatch):
    ledger = UsageLedger(max_tokens=5, path=str(tmp_path / "budget.json"))
    reserved = ledger.reserve(0)
    ledger.reconcile(reserved, TokenUsage(3, 2, 5))
    spent = QwenChatClient(api_key="test-key", usage_ledger=ledger)
    monkeypatch.setattr("urllib.request.urlopen", lambda *unused: pytest.fail("transport called"))
    with pytest.raises(RuntimeError, match="budget exhausted"):
        spent.chat([{"role": "user", "content": "offline only"}], max_tokens=1)


def test_billed_usage_is_recorded_even_when_the_response_is_rejected(tmp_path, monkeypatch):
    """A served-but-rejected response was still billed, so it must be counted.

    Releasing the reservation instead would silently under-report real spend
    against the 5M budget — the ledger would show room that no longer exists.
    """
    monkeypatch.setenv("AWARELIQUID_ALLOW_FORMAL_NETWORK", "1")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(_response(model="qwen-turbo")),
    )
    client = _formal_client(tmp_path)  # requests qwen-plus, API answers qwen-turbo

    with pytest.raises(RuntimeError, match="wrong or missing model"):
        client.chat([{"role": "user", "content": "Answer A"}], max_tokens=8)

    # The 5 tokens the API billed are committed, not discarded.
    assert client.usage_ledger.snapshot()["used_tokens"] == 5
    assert json.loads(Path(client.usage_ledger.path).read_text(encoding="utf-8")) == {
        "max_tokens": 5_000_000,
        "used_tokens": 5,
    }
