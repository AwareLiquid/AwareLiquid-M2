"""Deterministic, offline provenance records for parsed source material.

This module deliberately records source boundaries only.  It does not retrieve,
rank, summarize, or infer from source text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping, Optional, Sequence, Tuple


PARSER_VERSION = "a3-evidence-v1"


REQUIRED_PROVENANCE_FIELDS = (
    "domain",
    "doc_id",
    "page",
    "source_path",
    "char_start",
    "char_end",
    "content",
    "section",
    "title",
    "table_id",
    "row_id",
    "column_ids",
    "unit",
    "footnote",
    "parent_evidence_id",
    "neighbor_evidence_ids",
    "parse_warning",
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _optional_text(value: Optional[str], name: str) -> Optional[str]:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return value


def _text_tuple(value: Sequence[str], name: str) -> Tuple[str, ...]:
    if value is None:
        raise TypeError(f"{name} must be a sequence of strings")
    if isinstance(value, str):
        raise TypeError(f"{name} must be a sequence of strings, not a string")
    result = tuple(value)
    if not all(isinstance(item, str) for item in result):
        raise TypeError(f"{name} must contain only strings")
    return result


@dataclass(frozen=True)
class Evidence:
    """A non-semantic record of an extracted source span and its boundaries.

    ``content`` is the exact text emitted by a parser for this record.  The
    offsets remain offsets in the original parser source, rather than offsets
    into ``content``.  Consequently, a record can preserve a table row or a
    page fragment without changing its original location.
    """

    domain: str
    doc_id: str
    page: int
    source_path: str
    char_start: int
    char_end: int
    content: str
    section: Optional[str] = None
    title: Optional[str] = None
    table_id: Optional[str] = None
    row_id: Optional[str] = None
    column_ids: Tuple[str, ...] = ()
    unit: Optional[str] = None
    footnote: Optional[str] = None
    parent_evidence_id: Optional[str] = None
    neighbor_evidence_ids: Tuple[str, ...] = ()
    parse_warning: Tuple[str, ...] = ()
    parser_version: str = PARSER_VERSION
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("domain", "doc_id", "source_path", "parser_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1:
            raise ValueError("page must be a positive integer")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.char_start, self.char_end)):
            raise TypeError("char_start and char_end must be integers")
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError("offsets must satisfy 0 <= char_start <= char_end")

        for name in ("section", "title", "table_id", "row_id", "unit", "footnote", "parent_evidence_id"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        for name in ("column_ids", "neighbor_evidence_ids", "parse_warning"):
            object.__setattr__(self, name, _text_tuple(getattr(self, name), name))

        object.__setattr__(self, "evidence_id", self.stable_id())

    @property
    def content_sha256(self) -> str:
        """SHA-256 of the exact extracted content, encoded as UTF-8."""

        return sha256(self.content.encode("utf-8")).hexdigest()

    def stable_id(self) -> str:
        """Return the ID derived only from version, source, position, and text."""

        identity = {
            "char_end": self.char_end,
            "char_start": self.char_start,
            "content_sha256": self.content_sha256,
            "doc_id": self.doc_id,
            "page": self.page,
            "parser_version": self.parser_version,
            "source_path": self.source_path,
        }
        return "evidence-" + sha256(_canonical_json(identity)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize every provenance and boundary field without omission."""

        return {
            "evidence_id": self.evidence_id,
            "parser_version": self.parser_version,
            "domain": self.domain,
            "doc_id": self.doc_id,
            "page": self.page,
            "source_path": self.source_path,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "content": self.content,
            "content_sha256": self.content_sha256,
            "section": self.section,
            "title": self.title,
            "table_id": self.table_id,
            "row_id": self.row_id,
            "column_ids": list(self.column_ids),
            "unit": self.unit,
            "footnote": self.footnote,
            "parent_evidence_id": self.parent_evidence_id,
            "neighbor_evidence_ids": list(self.neighbor_evidence_ids),
            "parse_warning": list(self.parse_warning),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Evidence":
        """Normalize a parsed record while rejecting absent required provenance."""

        missing = [name for name in REQUIRED_PROVENANCE_FIELDS if name not in record]
        if missing:
            raise ValueError(f"missing required evidence fields: {', '.join(missing)}")
        return cls(
            domain=record["domain"],
            doc_id=record["doc_id"],
            page=record["page"],
            source_path=record["source_path"],
            char_start=record["char_start"],
            char_end=record["char_end"],
            content=record["content"],
            section=record.get("section"),
            title=record.get("title"),
            table_id=record.get("table_id"),
            row_id=record.get("row_id"),
            column_ids=record.get("column_ids"),
            unit=record.get("unit"),
            footnote=record.get("footnote"),
            parent_evidence_id=record.get("parent_evidence_id"),
            neighbor_evidence_ids=record.get("neighbor_evidence_ids"),
            parse_warning=record.get("parse_warning"),
            parser_version=record.get("parser_version", PARSER_VERSION),
        )


EvidenceRecord = Evidence


def normalize_evidence(record: Mapping[str, Any]) -> Evidence:
    """Build one deterministic :class:`Evidence` value from parser output."""

    return Evidence.from_dict(record)
