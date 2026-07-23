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
# Polarity markers. Clause documents turn on these: "酒后驾驶属于责任免除" and
# "酒后驾驶不属于责任免除" differ by one character yet invert the answer. The
# sentence survives retrieval and compression intact, but a reader matching on
# the topic can skim straight past the negation -- so negated statements are
# lifted out and surfaced separately instead of being left inline.
_NEGATION = re.compile(
    r"不属于|不承担|不负责|不负赔偿|不适用|不予|不得|不构成|不视为|不再|不受|"
    r"无需|并非|免收|不收取|不计入|不包括|不包含|均不"
)
# "除外" / "但……除外" is an EXCEPTION to the surrounding rule, not a negation of
# it: "感染艾滋病……但因输血导致的除外" means the transfusion case is carved OUT
# of the exclusion, i.e. it IS covered. Labelling that as a negation tells the
# reader the opposite of what the clause says, so exceptions are tracked and
# announced separately from plain negations.
_EXCEPTION = re.compile(r"除外|但书|另有约定|不在此限|但.{0,12}除外")
# Latin sentence enders only split when followed by whitespace/end, so a decimal
# such as "3.45" is never treated as a boundary.
_SENTENCE = re.compile(r"(?<=[。！？；])\s*|(?<=[.!?;])(?=\s|$)")


def _clip(sentence: str, max_chars: int) -> str:
    return sentence if len(sentence) <= max_chars else sentence[:max_chars] + "…"


def _polarity_statements(
    text: str, limit: int = 3, max_chars: int = 120
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Split *text* into (negated, excepted) sentences.

    The two are kept apart on purpose. A negation reverses a claim ("酒后驾驶
    不属于责任免除"); an exception carves a case OUT of the surrounding rule
    ("……但因输血导致的除外"), which usually means the carved-out case is
    treated the opposite way from the rule it sits in. Collapsing them into one
    "negated" label misreports the second kind.
    """
    negated: List[str] = []
    excepted: List[str] = []
    for raw in _SENTENCE.split(text or ""):
        sentence = raw.strip()
        if not sentence:
            continue
        if _EXCEPTION.search(sentence) and len(excepted) < limit:
            excepted.append(_clip(sentence, max_chars))
        elif _NEGATION.search(sentence) and len(negated) < limit:
            negated.append(_clip(sentence, max_chars))
    return tuple(negated), tuple(excepted)


def _negated_statements(text: str, limit: int = 3, max_chars: int = 120) -> Tuple[str, ...]:
    """Sentences that explicitly negate something (exceptions excluded)."""
    return _polarity_statements(text, limit, max_chars)[0]


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
    # Sentences in this span that explicitly negate something. Polarity is a
    # first-class structural field here for the same reason clause_id is: it is
    # deterministic, extractable offline, and answer-bearing in clause text.
    negations: Tuple[str, ...] = ()
    # Sentences carving a case OUT of the surrounding rule ("……但……除外").
    exceptions: Tuple[str, ...] = ()

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
            "negations": list(self.negations),
            "exceptions": list(self.exceptions),
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
                negations=_polarity_statements(content)[0],
                exceptions=_polarity_statements(content)[1],
                neighbor_chunk_idxs=neighbors,
            )
        )
    return nodes
