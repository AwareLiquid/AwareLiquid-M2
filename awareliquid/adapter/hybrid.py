"""Reciprocal Rank Fusion (RRF) of dense and lexical retrieval.

Dense (cosine over e5 vectors) and lexical (BM25 over an FTS5 index) retrieve on
different signals: the dense channel captures meaning and paraphrase, the lexical
channel captures the exact tokens financial questions turn on (rates like
``3.45%``, rating codes like ``AAA``, line-item names, ``FY2023``). Fusing them
recovers answers either channel alone would miss.

RRF fuses **rank positions, not raw scores**, so the two incomparable scales
(bounded cosine vs. unbounded BM25) never need calibration:

    score(d) = Σ_channel  weight_channel / (k + rank_channel(d))

A small ``k`` gives the rank-1 item a much steeper contribution (``1/(k+1)``),
which sharpens the head of the list — what matters when only the top few chunks
are fed to the model. A document found by only one channel still scores from that
channel alone (it simply contributes nothing from the channel that missed it).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

# A ranked hit is (id, content, meta), best-first.
Hit = Tuple[int, str, Optional[Any]]


def rrf_fuse(
    dense: Sequence[Hit],
    sparse: Sequence[Hit],
    k: int = 10,
    w_dense: float = 0.7,
    w_sparse: float = 0.3,
    top_k: int = 6,
) -> List[Hit]:
    """Fuse two ranked hit lists into one, best-first, keeping *top_k*.

    ``dense`` and ``sparse`` are each ordered best-first. ``k`` is the RRF
    constant (smaller = sharper head; ~10–20 suits small per-question pools).
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")
    scores: dict = {}
    payload: dict = {}

    def _accumulate(hits: Sequence[Hit], weight: float) -> None:
        for rank, (rid, content, meta) in enumerate(hits):
            # rank is 0-based here, so (k + rank + 1) == k + 1-based-rank.
            scores[rid] = scores.get(rid, 0.0) + weight / (k + rank + 1)
            payload.setdefault(rid, (content, meta))

    _accumulate(dense, w_dense)
    _accumulate(sparse, w_sparse)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Hit] = []
    for rid, _score in ordered[:top_k]:
        content, meta = payload[rid]
        out.append((rid, content, meta))
    return out
