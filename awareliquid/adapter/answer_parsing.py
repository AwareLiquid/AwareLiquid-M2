"""Turning model output into a canonical answer.

Covers three concerns that are easy to get subtly wrong: reading the
structured per-option judgement, checking an answer is canonical for its
format, and aggregating several independent samples into one answer.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from .schemas import parse_answer


_OPTION_STATE = re.compile(
    r"^\s*([A-Z])\s*=\s*(SUPPORTED|REFUTED|INSUFFICIENT)\b",
    re.IGNORECASE,
)
_OPTION_RELATION_STATE = re.compile(
    r"^\s*([A-Z])\s*:\s*.*?\bstate\s*=\s*"
    r"(SUPPORTED|REFUTED|INSUFFICIENT)\b",
    re.IGNORECASE,
)
_ANSWER_LINE = re.compile(r"(?:^|\n)\s*ANSWER\s*:\s*([A-Z]+)\b", re.IGNORECASE)


def _parse_structured_answer(raw: str, qtype: str, num_options: int) -> str:
    """Parse one-call option states, with a safe fallback for nonconforming output."""
    text = raw or ""
    state_pairs = []
    for line in text.splitlines():
        match = _OPTION_RELATION_STATE.search(line) or _OPTION_STATE.search(line)
        if match:
            state_pairs.append((match.group(1), match.group(2)))
    states = {
        label.upper(): state.upper()
        for label, state in state_pairs
        if ord("A") <= ord(label.upper()) < ord("A") + max(1, num_options)
    }
    if states:
        supported = "".join(
            label for label in sorted(states) if states[label] == "SUPPORTED"
        )
        if qtype == "multi":
            return supported
        if supported:
            return supported[0]
    answer_line = _ANSWER_LINE.search(text)
    if answer_line:
        return parse_answer(answer_line.group(1), qtype, num_options=num_options)
    return parse_answer(text, qtype, num_options=num_options)


def _is_canonical_answer(answer: str, qtype: str, num_options: int) -> bool:
    allowed = {chr(ord("A") + i) for i in range(max(1, num_options))}
    if qtype in {"mcq", "tf"}:
        return answer in allowed and (qtype != "tf" or answer in {"A", "B"})
    return bool(answer) and set(answer) <= allowed and answer == "".join(sorted(set(answer)))


def _majority_vote(answers: Sequence[str], qtype: str, num_options: int) -> str:
    """Aggregate independent answer samples into one answer.

    Voting happens per LETTER, not on the whole answer string. That distinction
    is what makes this useful for multi-select: a spurious letter the model only
    emits in a minority of samples is dropped, while a letter it consistently
    supports survives. Whole-string voting could not separate the two, because
    "AB" and "ABD" are simply different strings.

    Single-answer formats take the most frequent letter. Ill-formed samples are
    ignored; if every sample is ill-formed the first one is returned unchanged so
    the caller's existing canonicality retry still applies.
    """
    valid = [a for a in answers if _is_canonical_answer(a, qtype, num_options)]
    if not valid:
        return answers[0] if answers else ""

    if qtype in ("mcq", "tf"):
        return Counter(valid).most_common(1)[0][0]

    # multi: keep a letter only when a strict majority of samples carries it.
    threshold = len(valid) / 2
    letter_votes = Counter(letter for answer in valid for letter in set(answer))
    kept = sorted(letter for letter, votes in letter_votes.items() if votes > threshold)
    if kept:
        return "".join(kept)
    # Nothing reached a majority (e.g. two samples disagreeing completely) --
    # fall back to the most frequent complete answer rather than inventing one.
    return Counter(valid).most_common(1)[0][0]
