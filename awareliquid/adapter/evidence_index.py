"""Deterministic document structure for competition-safe retrieval.

The index stores navigation metadata and exact source text only.  It does not
produce semantic summaries, embeddings, or answer-bearing conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Tuple

from .chunker import chunk_document


_ANCHOR = re.compile(
    r"(?:19|20)\d{2}\s*年?"
    r"|\d+(?:,\d{3})*(?:\.\d+)?\s*"
    r"(?:个百分点|亿元|万元|美元|港元|bp|BP|%|％|亿|万|元)"
    r"|第\s*[一二三四五六七八九十百\d]+\s*"
    r"(?:条|章|节|款|项|个保单年度|保单年度)"
    r"|AAA|AA\+?"
    r"|除外|例外|不适用|特殊情形|免责|等待期"
)
_CLAUSE = re.compile(
    r"第\s*[一二三四五六七八九十百\d]+\s*(?:条|章|节|款|项)"
    r"|(?<![\d.])\d+(?:\.\d+){2,3}\s*(?=[、.]|[一-鿿])",
    re.MULTILINE,
)
_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s+|第[一二三四五六七八九十百\d]+[章节条款项]|"
    r"[一二三四五六七八九十百]+、|\d+(?:\.\d+){0,3}[、.])\s*.*$"
)


def _unique(values: List[str]) -> Tuple[str, ...]:
    seen = set()
    result = []
    for value in values:
        normalized = re.sub(r"\s+", "", value).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(value.strip())
    return tuple(result)


def _heading(text: str) -> Optional[str]:
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if candidate and _HEADING.match(candidate) and len(candidate) <= 160:
            return candidate
    return None


def _table_fields(content: str) -> Tuple[Optional[str], Tuple[str, ...]]:
    rows = [
        [part.strip() for part in line.split("|") if part.strip()]
        for line in (content or "").splitlines()
        if "|" in line
    ]
    if not rows:
        return None, ()
    row_labels = [row[0] for row in rows if row]
    row_id = " ".join(row_labels) if row_labels else None
    column_ids = tuple(rows[0][1:]) if len(rows) > 1 else ()
    return row_id, column_ids


def _page_chunks(doc_id: str, text: str, max_chars: int, overlap_chars: int):
    pages = (text or "").split("\f")
    chunks = []
    for page, page_text in enumerate(pages, start=1):
        for chunk in chunk_document(
            doc_id, page_text, max_chars=max_chars, overlap_chars=overlap_chars
        ):
            chunks.append((page, page_text, chunk.text))
    return chunks


@dataclass(frozen=True)
class EvidenceNode:
    """One exact source span plus deterministic navigation fields."""

    doc_id: str
    node_id: str
    chunk_idx: int
    page: int
    content: str
    section: Optional[str]
    clause_id: Optional[str]
    table_id: Optional[str]
    anchors: Tuple[str, ...]
    neighbor_chunk_idxs: Tuple[int, ...]
    row_id: Optional[str] = None
    column_ids: Tuple[str, ...] = ()

    def as_meta(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "node_id": self.node_id,
            "chunk_idx": self.chunk_idx,
            "page": self.page,
            "section": self.section,
            "clause_id": self.clause_id,
            "table_id": self.table_id,
            "row_id": self.row_id,
            "column_ids": list(self.column_ids),
            "anchors": list(self.anchors),
            "neighbor_chunk_idxs": list(self.neighbor_chunk_idxs),
            "search_terms": " ".join(
                value
                for value in (self.section, self.clause_id, *self.anchors)
                if value
            ),
        }


def build_evidence_nodes(
    doc_id: str,
    text: str,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> List[EvidenceNode]:
    """Build a source-linked, non-semantic index from document text."""
    raw_chunks = _page_chunks(doc_id, text, max_chars, overlap_chars)
    nodes = []
    for chunk_idx, (page, page_text, content) in enumerate(raw_chunks):
        anchors = _unique(_ANCHOR.findall(content))
        clauses = _CLAUSE.findall(content)
        section = _heading(content) or _heading(page_text)
        clause_id = clauses[0].strip() if clauses else None
        table_id = f"{doc_id}:page-{page}" if "|" in content else None
        row_id, column_ids = _table_fields(content)
        neighbors = tuple(
            index
            for index in (chunk_idx - 1, chunk_idx + 1)
            if 0 <= index < len(raw_chunks)
        )
        nodes.append(
            EvidenceNode(
                doc_id=str(doc_id),
                node_id=f"{doc_id}:node-{chunk_idx}",
                chunk_idx=chunk_idx,
                page=page,
                content=content,
                section=section,
                clause_id=clause_id,
                table_id=table_id,
                row_id=row_id,
                column_ids=column_ids,
                anchors=anchors,
                neighbor_chunk_idxs=neighbors,
            )
        )
    return nodes
