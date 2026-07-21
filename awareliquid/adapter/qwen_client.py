"""Qwen chat client with per-call token accounting.

The adapter never modifies base-model weights: generation is delegated to the
official Qwen API (OpenAI-compatible endpoint, e.g. Alibaba Cloud DashScope).
This module is the *only* place that talks to that API, so token usage is
measured in exactly one spot and can be summed across a run.

Configuration is entirely by environment variable so nothing is hard-coded:

    AWARELIQUID_LLM_BACKEND   "qwen" | "mock"      (default "qwen")
    AWARELIQUID_LLM_API_KEY   secret token         (falls back to DASHSCOPE_API_KEY)
    AWARELIQUID_LLM_BASE_URL  OpenAI-compatible base URL
                              (default DashScope compatible-mode endpoint)
    AWARELIQUID_LLM_MODEL     model id             (default "qwen-plus")
    AWARELIQUID_ALLOW_FORMAL_NETWORK=1 explicitly enables formal Qwen egress;
                              the default is deny.

Mock is available only by explicit test-only selection. Formal execution never
falls back to a mock client or a different transport.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

# Advisory cross-process file locking. `fcntl` is POSIX-only and `msvcrt` is
# Windows-only, so import whichever exists rather than hard-failing at import
# time -- a top-level `import fcntl` makes this whole package unimportable on
# Windows, including for offline/mock use.
try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]
try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - platform dependent
    msvcrt = None  # type: ignore[assignment]

# Upper bound on how long we wait for a peer to release the ledger lock.
_LOCK_TIMEOUT_SECONDS = 60.0


def _acquire_file_lock(lock_file) -> None:
    """Take an exclusive advisory lock on *lock_file*, blocking until granted."""
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return
    if msvcrt is not None:
        # msvcrt locks a byte range starting at the current position. LK_LOCK
        # blocks but gives up after ~10s, so retry until our own deadline.
        lock_file.seek(0)
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise
    # No advisory locking on this platform: the in-process RLock still applies,
    # but concurrent processes are not serialised.


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry so a completed rename survives a crash.

    POSIX-only: Windows has no directory handle to sync (opening one raises),
    and NTFS journals the rename itself, so this is a no-op there.
    """
    if os.name == "nt":
        return
    try:
        directory_fd = os.open(str(directory), os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _release_file_lock(lock_file) -> None:
    """Release the lock taken by :func:`_acquire_file_lock`."""
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return
    if msvcrt is not None:
        lock_file.seek(0)
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            # Already released (or never held) -- nothing to undo.
            pass

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_PROVIDER = "qwen"
DEFAULT_TOKEN_BUDGET = 5_000_000

# Formal runs must agree on one durable aggregate.  This registry deliberately
# lives in this transport module, next to the sole formal client, so a second
# client cannot silently point the same run at an alternate ledger.
_FORMAL_LEDGER_PATHS: Dict[str, str] = {}
_FORMAL_LEDGER_PATHS_LOCK = threading.RLock()

# Formal transport is deny-by-default. A real execution environment must opt in
# explicitly; all provider, model, endpoint, and token-ledger checks still run.
_FORMAL_NETWORK_ENV = "AWARELIQUID_ALLOW_FORMAL_NETWORK"
_ALLOWED_QWEN_MODELS = {
    "qwen-plus", "qwen-turbo", "qwen-max", "qwen-long",
    "qwen3.6-plus",
    "qwen3-235b-a22b", "qwen3-32b", "qwen3-30b-a3b", "qwen3-14b",
    "qwen3-8b", "qwen3-4b", "qwen3-1.7b", "qwen3-0.6b",
}
@dataclass
class TokenUsage:
    """Token counts for a single call (mirrors the API ``usage`` object)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class UsageLedger:
    """Fail-closed, process-safe global token-budget ledger.

    ``path`` is optional. When supplied, committed usage is persisted after
    every reconciliation so a crashed submission cannot silently lose its
    accounting record. Each operation refreshes committed usage while holding a
    per-ledger file lock; reconciliation then atomically replaces the ledger
    file before releasing that lock. This prevents independent clients from
    overwriting each other's observed usage. Reservations remain in-memory and
    are released when a request fails before a usage object is received.
    """

    def __init__(self, max_tokens: int = DEFAULT_TOKEN_BUDGET, path: Optional[str] = None):
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens
        self.path = Path(path) if path else None
        self.used_tokens = 0
        self._reserved_tokens = 0
        self._lock = threading.RLock()
        self._persistence_error: Optional[RuntimeError] = None
        self._load()

    @property
    def is_persistent(self) -> bool:
        return self.path is not None

    def reserve(self, estimated_tokens: int) -> int:
        """Reserve an upper bound before an API call and return that amount."""
        if isinstance(estimated_tokens, bool) or not isinstance(estimated_tokens, int):
            raise ValueError("estimated_tokens must be an integer")
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens cannot be negative")
        amount = estimated_tokens
        with self._lock:
            self._raise_if_persistence_failed()
            with self._persistent_lock():
                self._refresh_persisted_usage_locked()
                if self.used_tokens + self._reserved_tokens + amount > self.max_tokens:
                    raise RuntimeError(
                        "Qwen token budget exhausted: "
                        f"used={self.used_tokens}, reserved={self._reserved_tokens}, "
                        f"requested={amount}, limit={self.max_tokens}"
                    )
                self._reserved_tokens += amount
                return amount

    def release(self, reserved_tokens: int) -> None:
        with self._lock:
            if (
                isinstance(reserved_tokens, bool)
                or not isinstance(reserved_tokens, int)
                or reserved_tokens < 0
                or reserved_tokens > self._reserved_tokens
            ):
                raise RuntimeError("invalid Qwen token reservation release")
            self._reserved_tokens -= reserved_tokens

    def reconcile(self, reserved_tokens: int, usage: TokenUsage) -> None:
        with self._lock:
            self._raise_if_persistence_failed()
            if not isinstance(usage, TokenUsage):
                raise TypeError("usage must be TokenUsage")
            if (
                isinstance(reserved_tokens, bool)
                or not isinstance(reserved_tokens, int)
                or reserved_tokens < 0
                or reserved_tokens > self._reserved_tokens
            ):
                raise RuntimeError("invalid Qwen token reservation reconciliation")
            counts = usage.as_dict()
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
                raise ValueError("usage fields must be non-negative integers")
            if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
                raise ValueError("usage.total_tokens must equal prompt_tokens + completion_tokens")
            actual = usage.total_tokens
            with self._persistent_lock():
                # A different client or process may have reconciled since this
                # instance reserved its estimate. Refresh while holding the
                # shared lock, then use that committed aggregate for the gate.
                self._refresh_persisted_usage_locked()
                if self.used_tokens + self._reserved_tokens - reserved_tokens + actual > self.max_tokens:
                    raise RuntimeError(
                        "Qwen token budget exceeded by API usage: "
                        f"used={self.used_tokens}, reserved={self._reserved_tokens - reserved_tokens}, "
                        f"actual={actual}, limit={self.max_tokens}"
                    )
                updated_used_tokens = self.used_tokens + actual
                try:
                    self._persist_locked(updated_used_tokens)
                except Exception as exc:
                    # The usage result is known but cannot be durably counted.
                    # Refuse future calls from this ledger rather than risk
                    # spending against an unverifiable aggregate.
                    self._persistence_error = RuntimeError(
                        "usage ledger persistence failed; refusing further calls"
                    )
                    raise self._persistence_error from exc
                self._reserved_tokens -= reserved_tokens
                self.used_tokens = updated_used_tokens

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            self._raise_if_persistence_failed()
            with self._persistent_lock():
                self._refresh_persisted_usage_locked()
            return {
                "used_tokens": self.used_tokens,
                "reserved_tokens": self._reserved_tokens,
                "remaining_tokens": max(
                    0, self.max_tokens - self.used_tokens - self._reserved_tokens
                ),
                "max_tokens": self.max_tokens,
            }

    def _load(self) -> None:
        with self._lock:
            with self._persistent_lock():
                self._refresh_persisted_usage_locked()

    def _refresh_persisted_usage_locked(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if set(data) != {"max_tokens", "used_tokens"}:
                raise ValueError("usage ledger schema is invalid")
            stored_limit = data["max_tokens"]
            if isinstance(stored_limit, bool) or not isinstance(stored_limit, int):
                raise ValueError("usage ledger max_tokens is invalid")
            if stored_limit != self.max_tokens:
                raise ValueError("usage ledger max_tokens does not match current budget")
            used_tokens = data["used_tokens"]
            if isinstance(used_tokens, bool) or not isinstance(used_tokens, int):
                raise ValueError("usage ledger used_tokens is invalid")
            if used_tokens < 0 or used_tokens > self.max_tokens:
                raise ValueError("usage ledger used_tokens is out of range")
            self.used_tokens = used_tokens
        except FileNotFoundError:
            self.used_tokens = 0
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid usage ledger: {self.path}") from exc

    @contextmanager
    def _persistent_lock(self):
        """Hold an advisory cross-process lock for the persistent ledger."""
        if self.path is None:
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        try:
            with lock_path.open("a+b") as lock_file:
                _acquire_file_lock(lock_file)
                try:
                    yield
                finally:
                    _release_file_lock(lock_file)
        except OSError as exc:
            raise RuntimeError(f"unable to lock usage ledger: {self.path}") from exc

    def _persist_locked(self, used_tokens: int) -> None:
        """Atomically persist committed usage while ``_persistent_lock`` is held."""
        if self.path is None:
            return
        payload = {"max_tokens": self.max_tokens, "used_tokens": used_tokens}
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
                json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
            _fsync_directory(self.path.parent)
        except OSError:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def _raise_if_persistence_failed(self) -> None:
        if self._persistence_error is not None:
            raise self._persistence_error


@dataclass
class ChatResult:
    """Uniform return shape from every chat backend."""

    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""


class QwenChatClient:
    """OpenAI-compatible chat client for the Qwen model family.

    Uses only the standard library (``urllib``) so the package has no hard HTTP
    dependency. Retries transient failures a small, bounded number of times.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        timeout: float = 60.0,
        max_retries: int = 2,
        competition_mode: bool = False,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        usage_ledger: Optional[UsageLedger] = None,
        usage_ledger_path: Optional[str] = None,
        formal_run_id: Optional[str] = None,
        formal_ledger_path: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("QwenChatClient requires a non-empty api_key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.competition_mode = bool(competition_mode)
        self.formal_run_id = formal_run_id
        self.formal_ledger_path = (
            _canonical_ledger_path(formal_ledger_path)
            if formal_ledger_path is not None
            else None
        )
        if usage_ledger is not None and not isinstance(usage_ledger, UsageLedger):
            raise TypeError("usage_ledger must be a UsageLedger")
        if self.competition_mode:
            self._configure_formal_ledger(
                token_budget=token_budget,
                usage_ledger=usage_ledger,
                usage_ledger_path=usage_ledger_path,
            )
        else:
            self.usage_ledger = usage_ledger if usage_ledger is not None else UsageLedger(
                token_budget, usage_ledger_path
            )
        if self.competition_mode:
            self.assert_formal_configuration()

    def _configure_formal_ledger(
        self,
        *,
        token_budget: int,
        usage_ledger: Optional[UsageLedger],
        usage_ledger_path: Optional[str],
    ) -> None:
        """Build the one ledger permitted for this explicitly named run."""
        if self.formal_ledger_path is None:
            raise ValueError("formal mode requires an explicit formal_ledger_path")
        if not isinstance(self.formal_run_id, str) or not self.formal_run_id.strip():
            raise ValueError("formal mode requires a non-empty formal_run_id")
        self.formal_run_id = self.formal_run_id.strip()
        if usage_ledger_path is not None:
            raise ValueError(
                "formal mode accepts ledger configuration only through formal_ledger_path"
            )
        if usage_ledger is not None:
            if (
                usage_ledger.path is None
                or _canonical_ledger_path(str(usage_ledger.path))
                != self.formal_ledger_path
            ):
                raise ValueError(
                    "formal usage_ledger path must exactly match formal_ledger_path"
                )
            self.usage_ledger = usage_ledger
        else:
            _bind_formal_ledger_path(self.formal_run_id, self.formal_ledger_path)
            self.usage_ledger = UsageLedger(token_budget, self.formal_ledger_path)
        if usage_ledger is not None:
            _bind_formal_ledger_path(self.formal_run_id, self.formal_ledger_path)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> ChatResult:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if self.competition_mode:
            self.assert_formal_configuration()
            _deny_formal_network_transport()
        estimated_tokens = _estimate_tokens(
            json.dumps(messages, ensure_ascii=False)
        ) + int(max_tokens)
        reserved_tokens = self.usage_ledger.reserve(estimated_tokens)
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "enable_thinking": False,
            }
        ).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        # True once `reserved_tokens` has been settled by a successful
        # reconcile, so the error paths below know not to release it twice.
        settled = False
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                result = self._parse(body, require_usage=True)
                if self.competition_mode and result.model != self.model:
                    # The API already served and BILLED this call, so its tokens
                    # must be counted even though we reject the response. Simply
                    # releasing the reservation here would silently under-report
                    # real spend against the budget.
                    self.usage_ledger.reconcile(reserved_tokens, result.usage)
                    settled = True
                    raise RuntimeError("Qwen API returned a wrong or missing model")
                self.usage_ledger.reconcile(reserved_tokens, result.usage)
                settled = True
                return result
            except urllib.error.HTTPError as exc:  # 4xx/5xx
                detail = exc.read().decode("utf-8", "ignore")
                last_err = RuntimeError(f"Qwen API HTTP {exc.code}: {detail[:300]}")
                # 4xx is a caller error (bad key/model) -- do not retry.
                if 400 <= exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_err = exc
            except Exception:
                # Response validation errors must not strand a reservation.
                # They are not transport failures and should not be retried
                # blindly. A reservation already settled by reconcile (billed
                # usage recorded) must NOT be released again.
                if not settled:
                    self.usage_ledger.release(reserved_tokens)
                raise
        self.usage_ledger.release(reserved_tokens)
        raise RuntimeError(f"Qwen chat failed after {self.max_retries + 1} attempts: {last_err}")

    @staticmethod
    def _parse(body: Dict, require_usage: bool = False) -> ChatResult:
        choices = body.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        usage_raw = body.get("usage")
        if require_usage and not isinstance(usage_raw, dict):
            raise RuntimeError("Qwen API response is missing usage accounting")
        usage_raw = usage_raw or {}
        required_usage_fields = {"prompt_tokens", "completion_tokens", "total_tokens"}
        if require_usage and not required_usage_fields.issubset(usage_raw):
            raise RuntimeError("Qwen API response has incomplete usage accounting")
        if require_usage:
            values = [usage_raw[name] for name in ("prompt_tokens", "completion_tokens", "total_tokens")]
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise RuntimeError("Qwen API response has invalid usage accounting")
            usage = TokenUsage(*values)
            if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
                raise RuntimeError("Qwen API response has inconsistent usage accounting")
        else:
            usage = TokenUsage()
        return ChatResult(text=text.strip(), usage=usage, model=str(body.get("model", "")))

    def _validate_competition_provider(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("competition mode requires a non-empty provider")

    def _validate_competition_endpoint(self) -> None:
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.port not in (None, 443)
            or parsed.params or parsed.query or parsed.fragment
            or parsed.username or parsed.password
        ):
            raise ValueError(
                "competition mode requires a valid HTTPS OpenAI-compatible endpoint"
            )

    def _validate_competition_model(self) -> None:
        if self.model not in _ALLOWED_QWEN_MODELS:
            raise ValueError("competition mode only permits allowlisted Qwen models")

    def assert_formal_configuration(self) -> None:
        """Validate the complete formal-mode boundary without using transport.

        This method is intentionally public for the QA agent's dependency gate:
        an injected object is accepted only when it is the exact formal Qwen
        client and this complete validation succeeds.
        """
        if not self.competition_mode:
            raise ValueError("formal mode requires competition_mode=True")
        self._validate_competition_provider()
        self._validate_competition_endpoint()
        self._validate_competition_model()
        if self.usage_ledger.max_tokens != DEFAULT_TOKEN_BUDGET:
            raise ValueError(
                f"formal mode requires an exact {DEFAULT_TOKEN_BUDGET} token budget"
            )
        if not self.usage_ledger.is_persistent:
            raise ValueError("competition mode requires a persistent usage ledger")
        if not isinstance(self.formal_run_id, str) or not self.formal_run_id:
            raise ValueError("formal mode requires a non-empty formal_run_id")
        if self.formal_ledger_path is None:
            raise ValueError("formal mode requires an explicit formal_ledger_path")
        if self.usage_ledger.path is None or (
            _canonical_ledger_path(str(self.usage_ledger.path))
            != self.formal_ledger_path
        ):
            raise ValueError(
                "formal usage_ledger path must exactly match formal_ledger_path"
            )
        _bind_formal_ledger_path(self.formal_run_id, self.formal_ledger_path)


def _canonical_ledger_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("formal_ledger_path must be a non-empty path")
    return str(Path(path).expanduser().resolve(strict=False))


def _bind_formal_ledger_path(run_id: str, ledger_path: str) -> None:
    """Reject a different ledger path for an already configured formal run."""
    with _FORMAL_LEDGER_PATHS_LOCK:
        configured_path = _FORMAL_LEDGER_PATHS.setdefault(run_id, ledger_path)
        if configured_path != ledger_path:
            raise ValueError(
                "formal run is already bound to a different formal_ledger_path"
            )


def formal_network_enabled() -> bool:
    """Whether formal (competition-mode) network egress is currently permitted.

    Formal runs deny egress by default; the operator must arm it deliberately.
    Callers use this to fail fast at start-up instead of dying on the first
    question, after documents have already been ingested.
    """
    return os.environ.get(_FORMAL_NETWORK_ENV, "0") == "1"


FORMAL_NETWORK_ENV = _FORMAL_NETWORK_ENV  # public alias for operators/tooling


def _deny_formal_network_transport() -> None:
    """Fail before payload, authorization header, or urllib transport exists."""
    if not formal_network_enabled():
        raise RuntimeError("formal Qwen network transport is denied in this offline phase")


def formal_network_denied_probe() -> None:
    """Exercise only the formal deny gate, without configuration or I/O.

    In particular, this probe does not construct a client, inspect credentials,
    create a ledger, acquire a lock, build a request, or call ``urlopen``.
    """
    try:
        _deny_formal_network_transport()
    except RuntimeError as exc:
        if "network transport is denied" not in str(exc):
            raise AssertionError("formal network probe failed for the wrong reason") from exc
    else:
        raise AssertionError("formal network probe unexpectedly permitted a call")


class MockChatClient:
    """Deterministic offline stand-in.

    Estimates token usage with a whitespace/character heuristic so downstream
    token-budget logic has realistic numbers to work with, and answers
    multiple-choice prompts by echoing the first option letter it can find.
    """

    model = "mock"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> ChatResult:
        prompt = "\n".join(msg.get("content", "") for msg in messages)
        # Answer from the LAST user message only (ignore the system prompt, whose
        # examples like "ACD)" would otherwise be mistaken for an option label).
        user_msgs = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
        target = user_msgs[-1] if user_msgs else prompt
        # Pick the first option label at a line start: "A) ...", "A. ...", "A、...".
        match = re.search(r"(?m)^\s*([A-Z])[\).、]", target)
        answer = match.group(1) if match else "A"
        usage = TokenUsage(
            prompt_tokens=_estimate_tokens(prompt),
            completion_tokens=_estimate_tokens(answer),
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return ChatResult(text=answer, usage=usage, model=self.model)


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~1 token per CJK char, ~1 per 4 Latin chars.

    Only used by the mock backend; the real client reports exact usage.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    latin = len(text) - cjk
    return cjk + max(1, latin // 4)


def build_chat_client(
    competition_mode: Optional[bool] = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    usage_ledger_path: Optional[str] = None,
    formal_run_id: Optional[str] = None,
    formal_ledger_path: Optional[str] = None,
    *,
    test_mode: Optional[bool] = None,
):
    """Return a chat client based on environment configuration.

    Mock is test-only and must be explicitly selected. Missing credentials,
    invalid providers/models/endpoints, and construction errors always raise.
    """
    backend = os.environ.get("AWARELIQUID_LLM_BACKEND", "qwen").lower()
    if competition_mode is None:
        competition_mode = os.environ.get("AWARELIQUID_COMPETITION_MODE", "0") == "1"
    competition_mode = bool(competition_mode)
    if test_mode is None:
        test_mode = os.environ.get("AWARELIQUID_TEST_MODE", "0") == "1"
    if backend == "mock":
        if competition_mode:
            raise RuntimeError("mock backend is forbidden in competition mode")
        if not test_mode:
            raise RuntimeError("mock backend requires explicit test mode")
        return MockChatClient()
    if backend != "qwen":
        raise RuntimeError("only the qwen backend is supported")

    api_key = os.environ.get("AWARELIQUID_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("Qwen backend requires AWARELIQUID_LLM_API_KEY")

    base_url = os.environ.get("AWARELIQUID_LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("AWARELIQUID_LLM_MODEL", DEFAULT_MODEL)
    if competition_mode:
        if usage_ledger_path is not None:
            raise ValueError(
                "formal mode accepts ledger configuration only through formal_ledger_path"
            )
        formal_run_id = formal_run_id or os.environ.get("AWARELIQUID_FORMAL_RUN_ID")
        formal_ledger_path = formal_ledger_path or os.environ.get(
            "AWARELIQUID_FORMAL_LEDGER_PATH"
        )
    return QwenChatClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        provider=os.environ.get("AWARELIQUID_LLM_PROVIDER", DEFAULT_PROVIDER),
        competition_mode=competition_mode,
        token_budget=token_budget,
        usage_ledger_path=(
            usage_ledger_path or os.environ.get("AWARELIQUID_USAGE_LEDGER_PATH")
        ) if not competition_mode else None,
        formal_run_id=formal_run_id,
        formal_ledger_path=formal_ledger_path,
    )
