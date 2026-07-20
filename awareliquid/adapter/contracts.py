"""Frozen AFAC input, internal, and formal-output contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


AFAC_TOKEN_BUDGET = 5_000_000
ANSWER_FORMATS = ("mcq", "multi", "tf")
OPTION_KEYS = ("A", "B", "C", "D")
TF_OUTPUTS = ("T", "F")
_MISSING = object()


class ContractError(ValueError):
    """Machine-readable fail-closed contract violation."""

    def __init__(
        self,
        code: str,
        field: str,
        detail: str,
        *,
        split: Optional[str] = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.field = field
        self.detail = detail
        self.split = split

    def as_dict(self) -> dict[str, Optional[str]]:
        return {
            "code": self.code,
            "field": self.field,
            "detail": self.detail,
            "split": self.split,
        }


@dataclass(frozen=True)
class CompetitionQuestion:
    """Internal standard object after input-contract validation."""

    qid: str
    domain: str
    split: str
    question: str
    options: Mapping[str, str]
    answer_format: str
    type_hint: str
    doc_ids: Optional[Tuple[str, ...]]


@dataclass(frozen=True)
class SubmissionTokenUsage:
    """Per-question model-interface token columns in formal CSV order."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ContractError(
                "INVALID_TOKEN_USAGE", "tokens", "token values must be integers"
            )
        if any(value < 0 for value in values):
            raise ContractError(
                "INVALID_TOKEN_USAGE", "tokens", "token values cannot be negative"
            )
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ContractError(
                "INVALID_TOKEN_USAGE",
                "total_tokens",
                "total_tokens must equal prompt_tokens + completion_tokens",
            )


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(
            "INVALID_REQUIRED_FIELD", field, f"{field} must be a non-empty string"
        )
    return value


def _validate_options(value: Any, answer_format: str) -> dict[str, str]:
    # Audited real TF inputs use A/B labels; mcq and multi remain four-option.
    allowed_keys = {"A", "B"} if answer_format == "tf" else set(OPTION_KEYS)
    value_keys = set(value) if isinstance(value, Mapping) else None
    if value_keys is None or (value_keys != allowed_keys and value_keys != set(OPTION_KEYS)):
        expected = "A and B (or A, B, C, and D)" if answer_format == "tf" else "A, B, C, and D"
        raise ContractError(
            "INVALID_OPTIONS",
            "options",
            f"options must be an object with exactly {expected} keys",
        )
    keys = OPTION_KEYS if value_keys == set(OPTION_KEYS) else ("A", "B")
    if any(not isinstance(value[key], str) for key in keys):
        raise ContractError(
            "INVALID_OPTIONS", "options", "every option value must be a string"
        )
    return {key: value[key] for key in keys}


def _validate_doc_ids(payload: Mapping[str, Any], split: str) -> Optional[Tuple[str, ...]]:
    doc_ids = payload.get("doc_ids", _MISSING)
    if split == "A":
        is_valid = (
            isinstance(doc_ids, list)
            and bool(doc_ids)
            and all(isinstance(doc_id, str) and bool(doc_id.strip()) for doc_id in doc_ids)
        )
        if not is_valid:
            raise ContractError(
                "INVALID_DOC_IDS_FOR_SPLIT_A",
                "doc_ids",
                "split A requires a non-empty list of non-empty document IDs",
                split=split,
            )
        return tuple(doc_ids)

    if doc_ids is _MISSING or doc_ids is None:
        return None
    if isinstance(doc_ids, list):
        raise ContractError(
            "DOC_IDS_SPLIT_CONFLICT",
            "doc_ids",
            "split B does not accept a doc_ids list, including an empty list",
            split=split,
        )
    raise ContractError(
        "INVALID_DOC_IDS_FOR_SPLIT_B",
        "doc_ids",
        "split B accepts only a missing or null doc_ids field",
        split=split,
    )


def validate_question(payload: Mapping[str, Any]) -> CompetitionQuestion:
    """Validate one official input object without inferring its split."""

    if not isinstance(payload, Mapping):
        raise ContractError("INVALID_QUESTION", "question", "question must be an object")
    if "split" not in payload:
        raise ContractError("MISSING_SPLIT", "split", "split is required and is not inferred")
    split_value = payload["split"]
    if not isinstance(split_value, str) or split_value.upper() not in {"A", "B"}:
        raise ContractError("INVALID_SPLIT", "split", "split must be A or B")
    split = split_value.upper()

    answer_format = payload.get("answer_format")
    if answer_format not in ANSWER_FORMATS:
        raise ContractError(
            "INVALID_ANSWER_FORMAT",
            "answer_format",
            "answer_format must be mcq, multi, or tf",
            split=split,
        )

    return CompetitionQuestion(
        qid=_required_text(payload, "qid"),
        domain=_required_text(payload, "domain"),
        split=split,
        question=_required_text(payload, "question"),
        options=_validate_options(payload.get("options"), answer_format),
        answer_format=answer_format,
        type_hint=_required_text(payload, "type"),
        doc_ids=_validate_doc_ids(payload, split),
    )


def validate_formal_answer(answer_format: str, answer: str) -> str:
    """Validate the answer representation used in the formal CSV."""

    is_valid = False
    if answer_format == "mcq":
        is_valid = answer in OPTION_KEYS
    elif answer_format == "tf":
        is_valid = answer in TF_OUTPUTS
    elif answer_format == "multi" and isinstance(answer, str):
        is_valid = bool(answer) and all(char in OPTION_KEYS for char in answer)
        is_valid = is_valid and answer == "".join(sorted(set(answer)))
    if not is_valid:
        raise ContractError(
            "INVALID_FORMAL_ANSWER",
            "answer",
            f"invalid {answer_format!r} formal answer {answer!r}",
        )
    return answer


def build_summary_row(
    budget_tokens: int,
    used_tokens: int,
    unused_tokens: Optional[int] = None,
) -> list[Any]:
    """Build a structural summary row without deriving its unused field."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (budget_tokens, used_tokens)
    ) or used_tokens > budget_tokens:
        raise ContractError(
            "INVALID_TOKEN_USAGE",
            "used_tokens",
            "used_tokens and token_budget must be non-negative integers with used <= budget",
        )
    if unused_tokens is None:
        raise ContractError(
            "MISSING_UNUSED_TOKENS",
            "unused_tokens",
            "unused_tokens must be supplied explicitly before rendering",
        )
    if isinstance(unused_tokens, bool) or not isinstance(unused_tokens, int) or unused_tokens < 0:
        raise ContractError(
            "INVALID_TOKEN_USAGE",
            "unused_tokens",
            "unused_tokens must be a non-negative integer",
        )
    return ["summary", budget_tokens, used_tokens, unused_tokens]


def build_question_row(
    qid: str, answer: str, usage: SubmissionTokenUsage
) -> list[Any]:
    """Build one formal per-question row after its answer has been validated."""

    if not isinstance(qid, str) or not qid:
        raise ContractError("INVALID_REQUIRED_FIELD", "qid", "qid cannot be empty")
    if not isinstance(answer, str) or not answer:
        raise ContractError("INVALID_FORMAL_ANSWER", "answer", "answer cannot be empty")
    return [
        qid,
        answer,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    ]
