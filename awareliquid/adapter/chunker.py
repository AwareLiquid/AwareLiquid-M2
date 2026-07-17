"""Document ingestion and chunking for long financial text.

Long reports do not fit in a single context window and, more importantly for a
token-metered setting, most of a report is irrelevant to any one question. The
adapter therefore splits each document into overlapping, boundary-aware chunks
so that retrieval can pull in only the few passages a question needs.

The splitter is language-agnostic (works on Chinese, which has no spaces) and
tries to break on natural boundaries -- blank lines, then sentence punctuation
-- before falling back to a hard character cut, so a chunk rarely severs a
number from its unit or a clause from its subject.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# Sentence-ending punctuation for both Latin and CJK scripts. Latin enders only
# break when followed by whitespace/end, so a decimal point ("124.5") is never
# treated as a sentence boundary.
_SENT_END = re.compile(r"(?<=[。！？；])\s*|(?<=[.!?;])(?=\s|$)")
_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    """One retrievable passage of a source document."""

    doc_id: str
    chunk_idx: int
    text: str

    def as_meta(self) -> dict:
        return {"doc_id": self.doc_id, "chunk_idx": self.chunk_idx}


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_document(
    doc_id: str,
    text: str,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> List[Chunk]:
    """Split *text* into overlapping :class:`Chunk` objects.

    Parameters
    ----------
    max_chars:
        Soft upper bound on chunk length. Chunks grow by whole sentences until
        adding the next one would exceed this, so boundaries land on sentence
        ends whenever possible.
    overlap_chars:
        Number of trailing characters carried into the next chunk so a fact that
        straddles a boundary is still fully present in at least one chunk.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < max_chars")

    text = (text or "").strip()
    if not text:
        return []

    # Sentence stream, respecting paragraph boundaries first.
    sentences: List[str] = []
    for para in _PARA_SPLIT.split(text):
        sentences.extend(_split_sentences(para))

    chunks: List[Chunk] = []
    buf = ""
    for sent in sentences:
        # A single sentence longer than the budget is hard-split on its own.
        if len(sent) > max_chars:
            if buf:
                chunks.append(Chunk(doc_id, len(chunks), buf))
                buf = ""
            for i in range(0, len(sent), max_chars):
                chunks.append(Chunk(doc_id, len(chunks), sent[i : i + max_chars]))
            continue
        if buf and len(buf) + 1 + len(sent) > max_chars:
            chunks.append(Chunk(doc_id, len(chunks), buf))
            # Seed the next buffer with an overlap tail from the one just emitted.
            buf = buf[-overlap_chars:] if overlap_chars else ""
            buf = (buf + " " + sent).strip() if buf else sent
        else:
            buf = (buf + " " + sent).strip() if buf else sent

    if buf.strip():
        chunks.append(Chunk(doc_id, len(chunks), buf.strip()))
    return chunks
