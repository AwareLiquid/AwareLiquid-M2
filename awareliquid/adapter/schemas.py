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
