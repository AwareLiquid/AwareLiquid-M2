"""Answer schemas and robust parsing for choice-style questions.

The adapter answers three question shapes:

* ``mcq``   single choice -> one letter, e.g. "B"
* ``tf``    true/false      -> one letter (A/B), treated as a 2-option mcq
* ``multi`` multiple choice -> sorted, de-duplicated letters, e.g. "ABD"

A generation model rarely emits a bare letter; it says "The answer is B." or
"应选 A、C". :func:`parse_answer` normalises whatever the model returns into the
canonical form for each type so scoring is deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .qwen_client import TokenUsage

VALID_TYPES = ("mcq", "tf", "multi")
_LETTER = re.compile(r"[A-Z]")


@dataclass
class AnswerResult:
    """One answered question, ready to be written to a submission row."""

    qid: str
    answer: str
    qtype: str
    usage: TokenUsage = field(default_factory=TokenUsage)


def parse_answer(raw: str, qtype: str, num_options: int = 4) -> str:
    """Normalise a model's free-text reply to the canonical answer for *qtype*.

    Extraction is case-preserving on purpose: only letters within
    ``A .. A+num_options-1`` count, and a capital that merely begins a prose word
    (``Answer``, ``Definitely``) is skipped, so an option is recognised only when
    it stands alone or inside an all-caps run (``ABD``). A lowercase reply
    (``"the answer is b"``) is handled by a boundary-anchored fallback.
    """
    if qtype not in VALID_TYPES:
        raise ValueError(f"unknown qtype {qtype!r}, expected one of {VALID_TYPES}")

    text = raw or ""
    if qtype == "tf":
        # Providers occasionally ignore the letter-only instruction and emit
        # the semantic boolean.  Normalize that safe, unambiguous form before
        # scanning prose for option letters (where the ``A`` in ``Answer``
        # would otherwise be a tempting false positive).
        boolean = text.strip().lower()
        if re.fullmatch(r"(?:true|yes|正确|对)", boolean):
            return "A"
        if re.fullmatch(r"(?:false|no|错误|错)", boolean):
            return "B"
    # Defence in depth (the prompt already asks for letters only): if the model
    # still prefixes reasoning, narrow to the span after an explicit answer marker
    # or, failing that, the last non-empty line -- so acronyms in the reasoning
    # ("EBITDA", "ROE") cannot leak into the parsed answer.
    lowered = text.lower()
    for marker in ("答案", "正确选项", "应选", "answer:", "answer is", "answer"):
        idx = lowered.rfind(marker)
        if idx != -1:
            text = text[idx + len(marker):]
            break
    else:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[-1]
    max_ord = ord("A") + max(1, num_options) - 1
    letters: List[str] = []
    for m in _LETTER.finditer(text):
        ch = m.group()
        if ord(ch) > max_ord:
            continue
        nxt = text[m.end()] if m.end() < len(text) else ""
        # A capital followed by a lowercase letter is the start of a word, not an
        # answer choice ("Answer", "Because"); genuine option letters stand alone
        # or sit in an all-caps run ("ABD").
        if nxt.isascii() and nxt.islower():
            continue
        letters.append(ch)

    if not letters:
        # Fallback: an isolated lowercase option letter, e.g. "the answer is b".
        max_lower = chr(ord("a") + max(1, num_options) - 1)
        pat = rf"(?<![a-z])[a-{max_lower}](?![a-z])"
        letters = [c.upper() for c in re.findall(pat, text)]

    if qtype == "multi":
        # De-duplicate then sort -> "ABD".
        return "".join(sorted(set(letters)))
    # mcq / tf: the first valid letter is the choice.
    return letters[0] if letters else ""


def summarize_usage(results: List[AnswerResult]) -> TokenUsage:
    """Aggregate token usage across a batch of answers."""
    total = TokenUsage()
    for r in results:
        total = total + r.usage
    return total
