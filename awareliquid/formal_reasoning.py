"""Offline structural evidence handling for the A4 contract."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple, Union

from .evidence import Evidence


DecimalInput = Union[Decimal, int, str]
_ALLOWED_ARITHMETIC = {"add", "subtract", "multiply", "divide"}
_ALLOWED_COMPARISONS = {
    "equal", "not_equal", "greater_than", "greater_or_equal", "less_than", "less_or_equal"
}
_TABLE_FIELD_NAMES = {"table_id", "row_id", "column_ids", "unit", "footnote"}


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _text_tuple(value: Sequence[str], name: str, *, required: bool = False) -> Tuple[str, ...]:
    if isinstance(value, str) or value is None:
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(value)
    if required and not result:
        raise ValueError(f"{name} must not be empty")
    if not all(isinstance(item, str) and item for item in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


def _decimal(value: DecimalInput) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("decimal values must be Decimal, int, or string")
    if not isinstance(value, (Decimal, int, str)):
        raise TypeError("decimal values must be Decimal, int, or string")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError("invalid decimal value") from error
    if not result.is_finite():
        raise ValueError("decimal value must be finite")
    return result


def _merge_ids(values: Iterable["EvidenceDecimal"]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(identifier for value in values for identifier in value.evidence_ids))


@dataclass(frozen=True)
class EvidenceContext:
    """An ordered, non-transforming collection of source Evidence records."""

    records: Tuple[Evidence, ...]

    def __init__(self, records: Sequence[Evidence]):
        result = tuple(records)
        if not all(isinstance(record, Evidence) for record in result):
            raise TypeError("records must contain Evidence values")
        identifiers = tuple(record.evidence_id for record in result)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("records must have unique evidence IDs")
        object.__setattr__(self, "records", result)

    @property
    def evidence_ids(self) -> Tuple[str, ...]:
        return tuple(record.evidence_id for record in self.records)

    def require_ids(self, evidence_ids: Sequence[str]) -> Tuple[str, ...]:
        identifiers = _text_tuple(evidence_ids, "evidence_ids", required=True)
        identifiers = tuple(dict.fromkeys(identifiers))
        known = set(self.evidence_ids)
        unknown = [identifier for identifier in identifiers if identifier not in known]
        if unknown:
            raise ValueError(f"unknown evidence IDs: {', '.join(unknown)}")
        return identifiers

    def select(self, evidence_ids: Sequence[str]) -> "EvidenceContext":
        wanted = set(self.require_ids(evidence_ids))
        return EvidenceContext(tuple(record for record in self.records if record.evidence_id in wanted))

    def to_dicts(self) -> Tuple[dict, ...]:
        return tuple(record.to_dict() for record in self.records)


@dataclass(frozen=True)
class EvidenceDecimal:
    """One finite Decimal value explicitly tied to source Evidence IDs."""

    value: Decimal
    evidence_ids: Tuple[str, ...]
    context: EvidenceContext
    unit: Optional[str] = None

    def __init__(
        self,
        value: DecimalInput,
        evidence_ids: Sequence[str],
        context: EvidenceContext,
        unit: Optional[str] = None,
    ):
        if not isinstance(context, EvidenceContext):
            raise TypeError("context must be an EvidenceContext")
        if unit is not None and (not isinstance(unit, str) or not unit):
            raise ValueError("unit must be a non-empty string or None")
        object.__setattr__(self, "value", _decimal(value))
        object.__setattr__(self, "evidence_ids", context.require_ids(evidence_ids))
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "unit", unit)


def structural_arithmetic(operation: str, operands: Sequence[EvidenceDecimal]) -> EvidenceDecimal:
    """Apply one whitelisted Decimal operation while retaining all provenance."""

    if operation not in _ALLOWED_ARITHMETIC:
        raise ValueError("unknown structural arithmetic operation")
    values = tuple(operands)
    if not values or not all(isinstance(value, EvidenceDecimal) for value in values):
        raise TypeError("operands must be a non-empty sequence of EvidenceDecimal values")
    if operation in {"subtract", "divide"} and len(values) != 2:
        raise ValueError(f"{operation} requires exactly two operands")
    if operation == "add":
        result = sum((value.value for value in values), Decimal("0"))
    elif operation == "multiply":
        result = Decimal("1")
        for value in values:
            result *= value.value
    elif operation == "subtract":
        result = values[0].value - values[1].value
    else:
        if values[1].value == 0:
            raise ZeroDivisionError("division by zero")
        result = values[0].value / values[1].value
    unit = values[0].unit if all(value.unit == values[0].unit for value in values) else None
    records = {record.evidence_id: record for value in values for record in value.context.records}
    return EvidenceDecimal(result, _merge_ids(values), EvidenceContext(tuple(records.values())), unit)


class OptionState(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class OptionDecision:
    option_id: str
    state: OptionState
    evidence_ids: Tuple[str, ...]
    comparison: Optional[str]


def _compare(actual: Decimal, expected: Decimal, comparison: str) -> bool:
    return {
        "equal": actual == expected,
        "not_equal": actual != expected,
        "greater_than": actual > expected,
        "greater_or_equal": actual >= expected,
        "less_than": actual < expected,
        "less_or_equal": actual <= expected,
    }[comparison]


def evaluate_option(
    option_id: str,
    actual: Optional[EvidenceDecimal],
    expected: Optional[EvidenceDecimal],
    comparison: str = "equal",
    *,
    observed_text: Optional[str] = None,
    required_text: Optional[str] = None,
) -> OptionDecision:
    """Return a deterministic state for a bound numerical and exact-text claim."""

    _nonempty_text(option_id, "option_id")
    if comparison not in _ALLOWED_COMPARISONS:
        raise ValueError("unknown comparison")
    values = tuple(value for value in (actual, expected) if value is not None)
    if not all(isinstance(value, EvidenceDecimal) for value in values):
        raise TypeError("actual and expected must be EvidenceDecimal values or None")
    evidence_ids = _merge_ids(values)
    if actual is None or expected is None or actual.unit != expected.unit:
        return OptionDecision(option_id, OptionState.INSUFFICIENT, evidence_ids, comparison)
    if (observed_text is None) != (required_text is None):
        return OptionDecision(option_id, OptionState.INSUFFICIENT, evidence_ids, comparison)
    if observed_text is not None and (not isinstance(observed_text, str) or not isinstance(required_text, str)):
        raise TypeError("text boundaries must be strings or None")
    matches = _compare(actual.value, expected.value, comparison)
    matches = matches and (observed_text is None or observed_text == required_text)
    state = OptionState.SUPPORTED if matches else OptionState.REFUTED
    return OptionDecision(option_id, state, evidence_ids, comparison)


@dataclass(frozen=True)
class ExactQuery:
    """Only exact structural fields accepted by ExactLocator."""

    split: str
    doc_ids: Tuple[str, ...] = ()
    official_b_data: bool = False
    doc_id: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None
    clause: Optional[str] = None
    phrase: Optional[str] = None
    numbers: Tuple[str, ...] = ()
    dates: Tuple[str, ...] = ()
    percentages: Tuple[str, ...] = ()
    currencies: Tuple[str, ...] = ()
    ratings: Tuple[str, ...] = ()
    table_fields: Tuple[Tuple[str, str], ...] = ()
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    def __post_init__(self) -> None:
        if self.split not in {"A", "B"}:
            raise ValueError("split must be A or B")
        doc_ids = _text_tuple(self.doc_ids, "doc_ids")
        if self.split == "A" and not doc_ids:
            raise ValueError("A queries require non-empty doc_ids")
        if self.split == "B" and not self.official_b_data:
            raise ValueError("B queries require official B data")
        if self.split == "B" and doc_ids:
            raise ValueError("B queries must not supply doc_ids")
        object.__setattr__(self, "doc_ids", tuple(dict.fromkeys(doc_ids)))
        for name in ("doc_id", "title", "section", "clause", "phrase"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty_text(value, name))
        for name in ("numbers", "dates", "percentages", "currencies", "ratings"):
            object.__setattr__(self, name, _text_tuple(getattr(self, name), name))
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be supplied together")
        if self.char_start is not None:
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.char_start, self.char_end)):
                raise TypeError("character bounds must be integers")
            if self.char_start < 0 or self.char_end < self.char_start:
                raise ValueError("character bounds must satisfy 0 <= start <= end")
        fields = tuple(self.table_fields)
        for name, value in fields:
            if name not in _TABLE_FIELD_NAMES or not isinstance(value, str) or not value:
                raise ValueError("table_fields must contain allowed exact field/value pairs")
        object.__setattr__(self, "table_fields", fields)


class ExactLocator:
    """Locate records by conjunction of exact allowed field checks only."""

    def __init__(self, context: EvidenceContext):
        if not isinstance(context, EvidenceContext):
            raise TypeError("context must be an EvidenceContext")
        self._context = context

    def locate(self, query: ExactQuery) -> Tuple[str, ...]:
        if not isinstance(query, ExactQuery):
            raise TypeError("query must be an ExactQuery")
        matches = [record for record in self._context.records if self._matches(record, query)]
        matches.sort(key=lambda record: (record.doc_id, record.page, record.char_start, record.char_end, record.evidence_id))
        return tuple(record.evidence_id for record in matches)

    @staticmethod
    def _matches(record: Evidence, query: ExactQuery) -> bool:
        if query.split == "A" and record.doc_id not in query.doc_ids:
            return False
        if query.doc_id is not None and record.doc_id != query.doc_id:
            return False
        if query.title is not None and record.title != query.title:
            return False
        if query.section is not None and record.section != query.section:
            return False
        if query.clause is not None and record.section != query.clause:
            return False
        if query.phrase is not None and query.phrase not in record.content:
            return False
        tokens = (*query.numbers, *query.dates, *query.percentages, *query.currencies, *query.ratings)
        if any(token not in record.content for token in tokens):
            return False
        if query.char_start is not None and (record.char_start != query.char_start or record.char_end != query.char_end):
            return False
        return all(ExactLocator._table_value_matches(record, name, value) for name, value in query.table_fields)

    @staticmethod
    def _table_value_matches(record: Evidence, name: str, value: str) -> bool:
        field_value = getattr(record, name)
        return value in field_value if name == "column_ids" else field_value == value
