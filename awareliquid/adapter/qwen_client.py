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

When no API key is present the client transparently falls back to a deterministic
``MockChatClient`` so demos, unit tests and offline development keep working.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


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
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        if not api_key:
            raise ValueError("QwenChatClient requires a non-empty api_key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> ChatResult:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return self._parse(body)
            except urllib.error.HTTPError as exc:  # 4xx/5xx
                detail = exc.read().decode("utf-8", "ignore")
                last_err = RuntimeError(f"Qwen API HTTP {exc.code}: {detail[:300]}")
                # 4xx is a caller error (bad key/model) -- do not retry.
                if 400 <= exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_err = exc
        raise RuntimeError(f"Qwen chat failed after {self.max_retries + 1} attempts: {last_err}")

    @staticmethod
    def _parse(body: Dict) -> ChatResult:
        choices = body.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        usage_raw = body.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
            completion_tokens=int(usage_raw.get("completion_tokens", 0)),
            total_tokens=int(usage_raw.get("total_tokens", 0)),
        )
        # Some gateways omit total_tokens; reconstruct so accounting stays sound.
        if usage.total_tokens == 0:
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return ChatResult(text=text.strip(), usage=usage, model=str(body.get("model", "")))


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


def build_chat_client():
    """Return a chat client based on environment configuration.

    Never raises for a missing key: if the ``qwen`` backend is requested but no
    API key is available, falls back to :class:`MockChatClient` so the pipeline
    still runs (offline-first invariant).
    """
    backend = os.environ.get("AWARELIQUID_LLM_BACKEND", "qwen").lower()
    if backend == "mock":
        return MockChatClient()

    api_key = os.environ.get("AWARELIQUID_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        return MockChatClient()

    base_url = os.environ.get("AWARELIQUID_LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("AWARELIQUID_LLM_MODEL", DEFAULT_MODEL)
    try:
        return QwenChatClient(api_key=api_key, base_url=base_url, model=model)
    except Exception:
        return MockChatClient()
