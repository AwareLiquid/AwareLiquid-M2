"""Dynamic context compression for token-metered question answering.

Retrieval narrows a long document down to a handful of chunks, but those chunks
are still verbose relative to what a single question needs. The compressor is
the token-saving stage: it keeps only the sentences within each retrieved chunk
that actually bear on the question, so the prompt sent to the generation model
is a fraction of the raw passages while still carrying every load-bearing fact.

The default strategy is *extractive* and runs entirely locally, so compression
itself costs zero generation tokens -- important when the score rewards token
efficiency. A sentence is scored by:

* lexical overlap with the question (shared content terms), and
* a salience bonus for numbers, percentages, currency and dates, which carry the
  facts financial questions turn on.

Sentences are then greedily packed into a character budget, highest score first,
and re-emitted in their original document order so local reasoning context is
preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# Numbers, percentages, currency amounts, dates -- the tokens financial answers
# hinge on. A sentence carrying these is worth keeping even on weak lexical match.
_SALIENT = re.compile(
    r"\d[\d,\.]*\s*(?:%|％|亿|万|元|美元|港元|bp|BP|个百分点)?"
    r"|\d{4}\s*年|\d+\s*月|\d+\s*日"
)
# Split on CJK sentence enders always, and on Latin enders only when followed by
# whitespace/end -- so the '.' inside a decimal like "124.5" never splits.
_SENT_SPLIT = re.compile(r"(?<=[。！？；])\s*|(?<=[.!?;])(?=\s|$)")
# Content-term tokenisation: CJK bigrams + Latin words, minus trivial stopwords.
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_CJK = re.compile(r"[一-鿿]")
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "with", "as", "by", "at", "this", "that", "which",
    "的", "了", "是", "在", "和", "与", "及", "对", "为", "也", "都", "而",
}


def _terms(text: str) -> set:
    """Bag of content terms: lowercased Latin words + CJK character bigrams."""
    text = text or ""
    out = {w.lower() for w in _LATIN_WORD.findall(text) if w.lower() not in _STOP}
    chars = _CJK.findall(text)
    out.update(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return {t for t in out if t not in _STOP}


@dataclass
class CompressedContext:
    """Result of compressing retrieved passages for one question."""

    text: str
    kept_sentences: int
    source_sentences: int


class ExtractiveCompressor:
    """Local, LLM-free sentence selection under a character budget."""

    def __init__(self, budget_chars: int = 1200, salience_weight: float = 1.5):
        if budget_chars <= 0:
            raise ValueError(f"budget_chars must be positive, got {budget_chars}")
        self.budget_chars = int(budget_chars)
        self.salience_weight = float(salience_weight)

    def _score_sentence(self, sent: str, q_terms: set) -> float:
        s_terms = _terms(sent)
        if not s_terms:
            return 0.0
        overlap = len(s_terms & q_terms)
        salience = self.salience_weight if _SALIENT.search(sent) else 0.0
        # Normalise overlap by sentence length so a long sentence cannot win on
        # sheer size; add the flat salience bonus for fact-bearing sentences.
        return overlap / (1.0 + len(s_terms) ** 0.5) + salience

    def compress(self, question: str, passages: Sequence[str]) -> CompressedContext:
        """Select the most relevant sentences from *passages* under the budget."""
        q_terms = _terms(question)
        # (original_index, sentence, score) across all passages, in reading order.
        indexed: List[Tuple[int, str, float]] = []
        for passage in passages:
            for sent in (p.strip() for p in _SENT_SPLIT.split(passage or "") if p.strip()):
                idx = len(indexed)
                indexed.append((idx, sent, self._score_sentence(sent, q_terms)))

        if not indexed:
            return CompressedContext(text="", kept_sentences=0, source_sentences=0)

        # Greedily take the highest-scoring sentences until the budget is spent.
        ranked = sorted(indexed, key=lambda t: t[2], reverse=True)
        chosen, used = [], 0
        for idx, sent, score in ranked:
            if score <= 0 and chosen:
                break  # nothing relevant left to add
            if used + len(sent) > self.budget_chars and chosen:
                continue
            chosen.append((idx, sent))
            used += len(sent)
            if used >= self.budget_chars:
                break

        # Re-emit in original order so adjacent context reads coherently.
        chosen.sort(key=lambda t: t[0])
        text = "\n".join(f"- {s}" for _, s in chosen)
        return CompressedContext(
            text=text, kept_sentences=len(chosen), source_sentences=len(indexed)
        )
