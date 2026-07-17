import os

from awareliquid.adapter.qwen_client import (
    MockChatClient,
    TokenUsage,
    build_chat_client,
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


def test_build_falls_back_to_mock_without_key(monkeypatch):
    monkeypatch.delenv("AWARELIQUID_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("AWARELIQUID_LLM_BACKEND", "qwen")
    client = build_chat_client()
    assert isinstance(client, MockChatClient)


def test_build_respects_explicit_mock(monkeypatch):
    monkeypatch.setenv("AWARELIQUID_LLM_BACKEND", "mock")
    assert isinstance(build_chat_client(), MockChatClient)
