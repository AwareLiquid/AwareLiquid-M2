"""Fail-closed AFAC question and submission CSV contract.

This module is deliberately pure: it performs no model invocation, network access,
or file I/O.  The official CSV is defined here rather than inferred from legacy
submission examples.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


CSV_HEADER = ("qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens")
OPTION_KEYS = ("A", "B", "C", "D")
ANSWER_FORMATS = ("mcq", "multi", "tf")
TOKEN_BUDGET = 5_000_000
_MISSING = object()


class ContractError(ValueError):
    """Structured error raised instead of repairing malformed formal data."""

    def __init__(self, code: str, field: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.message = message

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "field": self.field, "message": self.message}}


@dataclass(frozen=True)
class ParsedQuestion:
    qid: str
    domain: str
    split: str
    question: str
    options: Mapping[str, str]
    answer_format: str
    type_hint: str
    doc_ids: Optional[Tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "domain": self.domain,
            "split": self.split,
            "question": self.question,
            "options": dict(self.options),
            "answer_format": self.answer_format,
            "type": self.type_hint,
            "doc_ids": list(self.doc_ids) if self.doc_ids is not None else None,
        }


@dataclass(frozen=True)
class SubmissionAnswer:
    """One answer and its complete, known model usage."""

    qid: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.qid, str) or not self.qid.strip() or self.qid == "summary":
            raise ContractError("INVALID_QID", "qid", "qid must be a non-empty non-summary string")
        if not isinstance(self.answer, str):
            raise ContractError("INVALID_FORMAL_ANSWER", "answer", "answer must be a string")
        _validate_usage(self.prompt_tokens, self.completion_tokens, self.total_tokens, "tokens")


@dataclass(frozen=True)
class SubmissionSummary:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        _validate_usage(
            self.prompt_tokens, self.completion_tokens, self.total_tokens, "summary"
        )
        if self.total_tokens > TOKEN_BUDGET:
            raise ContractError(
                "TOKEN_BUDGET_EXCEEDED",
                "summary.total_tokens",
                f"total tokens exceed the {TOKEN_BUDGET:,} token budget",
            )


def _validate_usage(prompt_tokens: Any, completion_tokens: Any, total_tokens: Any, field: str) -> None:
    values = (prompt_tokens, completion_tokens, total_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ContractError("UNKNOWN_OR_INVALID_TOKEN_USAGE", field, "token usage must be known non-negative integers")
    if total_tokens != prompt_tokens + completion_tokens:
        raise ContractError("INCONSISTENT_TOKEN_TOTAL", field, "total_tokens must equal prompt_tokens plus completion_tokens")


def _text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError("INVALID_REQUIRED_FIELD", field, f"{field} must be a non-empty string")
    return value


def _options(value: Any, answer_format: str) -> dict[str, str]:
    allowed_keys = {"A", "B"} if answer_format == "tf" else set(OPTION_KEYS)
    if not isinstance(value, Mapping) or set(value) != allowed_keys:
        expected = "A and B" if answer_format == "tf" else "A, B, C, and D"
        raise ContractError("INVALID_OPTIONS", "options", f"options must contain exactly {expected}")
    keys = ("A", "B") if answer_format == "tf" else OPTION_KEYS
    if any(not isinstance(value[key], str) or not value[key].strip() for key in keys):
        raise ContractError("INVALID_OPTIONS", "options", "option values must be non-empty strings")
    return {key: value[key] for key in keys}


def _doc_ids(payload: Mapping[str, Any], split: str) -> Optional[Tuple[str, ...]]:
    value = payload.get("doc_ids", _MISSING)
    if split == "A":
        if value is _MISSING or value is None:
            raise ContractError("DOC_IDS_REQUIRED_FOR_A", "doc_ids", "split A requires doc_ids")
        if not isinstance(value, list) or not value:
            raise ContractError("INVALID_DOC_IDS", "doc_ids", "doc_ids must be a non-empty list of strings")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ContractError("INVALID_DOC_IDS", "doc_ids", "doc_ids must be a non-empty list of strings")
        return tuple(value)

    if value is _MISSING or value is None:
        return None
    raise ContractError("SPLIT_DOC_IDS_CONFLICT", "doc_ids", "split B must not include doc_ids")


def parse_question(payload: Mapping[str, Any]) -> ParsedQuestion:
    """Validate one official question; split and answer format are never inferred."""

    if not isinstance(payload, Mapping):
        raise ContractError("INVALID_QUESTION", "question", "question must be an object")
    split_value = payload.get("split")
    if not isinstance(split_value, str) or split_value.upper() not in {"A", "B"}:
        raise ContractError("INVALID_SPLIT", "split", "split must be A or B")
    answer_format = payload.get("answer_format")
    if answer_format not in ANSWER_FORMATS:
        raise ContractError("INVALID_ANSWER_FORMAT", "answer_format", "answer_format must be mcq, multi, or tf")
    split = split_value.upper()
    return ParsedQuestion(
        qid=_text(payload, "qid"),
        domain=_text(payload, "domain"),
        split=split,
        question=_text(payload, "question"),
        options=_options(payload.get("options"), answer_format),
        answer_format=answer_format,
        type_hint=_text(payload, "type"),
        doc_ids=_doc_ids(payload, split),
    )


def parse_questions(payloads: Iterable[Mapping[str, Any]]) -> tuple[ParsedQuestion, ...]:
    """Validate a complete question collection and reject duplicate qids."""

    if isinstance(payloads, (str, bytes)):
        raise ContractError("INVALID_QUESTIONS", "questions", "questions must be an iterable of objects")
    try:
        questions = tuple(parse_question(payload) for payload in payloads)
    except TypeError as error:
        raise ContractError("INVALID_QUESTIONS", "questions", "questions must be an iterable of objects") from error
    qids = [question.qid for question in questions]
    if len(qids) != len(set(qids)):
        raise ContractError("DUPLICATE_QID", "qid", "question qids must be unique")
    return questions


def _validate_answer_for_question(answer: SubmissionAnswer, question: ParsedQuestion) -> None:
    value = answer.answer
    if question.answer_format == "tf":
        if value not in {"A", "B"}:
            raise ContractError("INVALID_FORMAL_ANSWER", "answer", "tf answer must be A or B")
        return
    if question.answer_format == "mcq":
        if value not in question.options:
            raise ContractError("INVALID_FORMAL_ANSWER", "answer", "answer must be an uppercase provided option letter")
        return
    if (
        not value
        or any(letter not in question.options for letter in value)
        or value != "".join(sorted(set(value)))
    ):
        raise ContractError(
            "INVALID_FORMAL_ANSWER",
            "answer",
            "multi answer must be sorted, deduplicated uppercase option letters without separators",
        )


def validate_submission(
    questions: Sequence[ParsedQuestion], answers: Sequence[SubmissionAnswer]
) -> SubmissionSummary:
    """Validate completeness, formal answers, aggregate token equality, and budget."""

    question_qids = [question.qid for question in questions]
    answer_qids = [answer.qid for answer in answers]
    if len(question_qids) != len(set(question_qids)):
        raise ContractError("DUPLICATE_QID", "qid", "question qids must be unique")
    if len(answer_qids) != len(set(answer_qids)):
        raise ContractError("DUPLICATE_QID", "qid", "output qids must be unique")
    expected = set(question_qids)
    actual = set(answer_qids)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        detail = "missing " + repr(missing) if missing else "unknown " + repr(unknown)
        raise ContractError("INCOMPLETE_OR_UNKNOWN_QIDS", "qid", f"output qids must exactly match inputs: {detail}")

    by_qid = {question.qid: question for question in questions}
    for answer in answers:
        _validate_answer_for_question(answer, by_qid[answer.qid])

    return SubmissionSummary(
        prompt_tokens=sum(answer.prompt_tokens for answer in answers),
        completion_tokens=sum(answer.completion_tokens for answer in answers),
        total_tokens=sum(answer.total_tokens for answer in answers),
    )


def render_submission_csv(
    questions: Sequence[ParsedQuestion], answers: Sequence[SubmissionAnswer]
) -> str:
    """Return the official UTF-8-text five-column CSV after full validation."""

    summary = validate_submission(questions, answers)
    rows: list[tuple[Any, ...]] = [CSV_HEADER]
    rows.append(("summary", "", summary.prompt_tokens, summary.completion_tokens, summary.total_tokens))
    rows.extend(
        (answer.qid, answer.answer, answer.prompt_tokens, answer.completion_tokens, answer.total_tokens)
        for answer in answers
    )
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue()
