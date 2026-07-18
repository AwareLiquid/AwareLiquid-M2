"""End-to-end memory-augmented QA over long documents.

``MemoryQAAgent`` wires the pieces together into the external adapter loop:

    ingest   document text -> chunks -> embeddings -> vector store
    answer    question -> retrieve chunks -> compress -> compact prompt
              -> Qwen chat API -> parsed answer + token usage

The base model is never modified: it is reached only through the injected chat
client. All retrieval and compression run locally, so they add nothing to the
generation-token bill -- the only tokens spent are the compact prompt and the
short answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..memory.encoder import SentenceEncoder
from ..memory.knowledge_store import PersistentKnowledgeMemory
from .chunker import chunk_document
from .compressor import ExtractiveCompressor
from .hybrid import rrf_fuse
from .qwen_client import ChatResult, build_chat_client
from .schemas import AnswerResult, parse_answer

# Temporal anchors that split a comparison/computation question into operands:
# a year ("2023" / "2023 年"), or a quarter ("Q3", "第三季度", "三季度"/"上半年").
_YEAR = re.compile(r"(?:19|20)\d{2}")
_QUARTER = re.compile(r"Q[1-4]|第[一二三四1-4]季度|[上下]半年|[一二三四1-4]季度")


def _detect_anchors(question: str) -> List[str]:
    """Distinct temporal operands in *question*, used to build per-operand
    sub-queries. Returns [] when there is at most one, so a single-target
    question is not needlessly fanned out."""
    anchors: List[str] = []
    seen: set = set()
    for m in list(_YEAR.finditer(question)) + list(_QUARTER.finditer(question)):
        tok = m.group()
        if tok not in seen:
            seen.add(tok)
            anchors.append(tok)
    return anchors if len(anchors) >= 2 else []


_SYSTEM_PROMPT = (
    "You are a precise financial document analyst. Answer the multiple-choice "
    "question using ONLY the provided context. Reply with the option letter(s) "
    "and NOTHING else -- no words, no reasoning, no punctuation. For single-choice "
    "or true/false output exactly one letter (e.g. B). For multiple-choice output "
    "every correct letter with no separators (e.g. ACD). If the context is "
    "insufficient, output the single best-supported letter."
)


@dataclass
class RetrievalConfig:
    # max_chars stays at or below the encoder's token window (e5: 512) so a whole
    # chunk is embedded rather than silently truncated.
    max_chars: int = 450
    overlap_chars: int = 80
    top_k: int = 6
    compression_budget: int = 1200
    max_answer_tokens: int = 16
    # -- hybrid retrieval (dense cosine + lexical BM25, fused with RRF) --
    hybrid: bool = True          # fuse BM25 with cosine when FTS5 is available
    rrf_pool: int = 20           # candidates per channel before fusion
    rrf_k: int = 10              # RRF constant (small = sharper head)
    w_dense: float = 0.7         # dense (semantic) fusion weight
    w_sparse: float = 0.3        # lexical (BM25) fusion weight
    center: bool = True          # anisotropy-robust centered cosine for the dense channel
    # -- multi-query retrieval (per-option sub-queries, union) --
    multi_query: bool = False    # retrieve the question + each option separately, then union
    multi_query_cap: int = 8     # max chunks kept after the union


class MemoryQAAgent:
    """Retrieve-compress-answer agent backed by a local vector store."""

    def __init__(
        self,
        encoder: Optional[SentenceEncoder] = None,
        store: Optional[PersistentKnowledgeMemory] = None,
        chat_client=None,
        compressor: Optional[ExtractiveCompressor] = None,
        config: Optional[RetrievalConfig] = None,
        db_path: str = ":memory:",
    ):
        self.encoder = encoder or SentenceEncoder()
        self.config = config or RetrievalConfig()
        # The store's key_dim must equal the encoder's output dim; probe once.
        self.store = store or PersistentKnowledgeMemory(
            key_dim=self.encoder.dim, db_path=db_path
        )
        self.chat_client = chat_client or build_chat_client()
        self.compressor = compressor or ExtractiveCompressor(
            budget_chars=self.config.compression_budget
        )
        self._ingested_docs: set = set()

    # -- ingest -----------------------------------------------------------
    def ingest_document(self, doc_id: str, text: str) -> int:
        """Chunk, embed and store one document. Returns the number of chunks."""
        chunks = chunk_document(
            doc_id,
            text,
            max_chars=self.config.max_chars,
            overlap_chars=self.config.overlap_chars,
        )
        for ch in chunks:
            key = self.encoder.encode(ch.text, is_query=False)
            self.store.write(key, ch.text, meta=ch.as_meta())
        self._ingested_docs.add(str(doc_id))
        return len(chunks)

    def ingest_documents(self, docs: Dict[str, str]) -> int:
        """Ingest a ``{doc_id: text}`` mapping. Returns total chunks stored."""
        return sum(self.ingest_document(did, txt) for did, txt in docs.items())

    # -- retrieve ---------------------------------------------------------
    @staticmethod
    def _enrich(question: str, options: Optional[Sequence[str]]) -> str:
        """Query string used for retrieval AND compression.

        The answer-bearing sentence often shares its wording with an *option*
        ("net margin fell to 12.4%") rather than the question ("how did
        profitability change?"), so the option text is folded in as extra terms.
        """
        if options:
            return question + " " + " ".join(str(o) for o in options)
        return question

    def _hybrid_hits(
        self, query_text: str, allowed: Optional[List[str]], limit: int
    ) -> List[Tuple[int, str]]:
        """One retrieval pass for *query_text*: dense (cosine) and lexical (BM25)
        channels fused with RRF, returning ``(id, content)`` best-first. Falls
        back to dense-only when hybrid is off or FTS5 is unavailable."""
        q_key = self.encoder.encode(query_text, is_query=True)
        use_hybrid = self.config.hybrid and self.store.fts_enabled
        pool = max(limit, self.config.rrf_pool) if use_hybrid else limit
        dense = self.store.query(
            q_key, top_k=pool, center=self.config.center, return_ids=True, doc_ids=allowed
        )
        dense_hits = [(h[0], h[1], h[3]) for h in dense]  # (id, content, meta)
        if not use_hybrid:
            return [(rid, c) for rid, c, _m in dense_hits[:limit]]
        sparse_hits = self.store.search_bm25(query_text, top_k=pool, doc_ids=allowed)
        fused = rrf_fuse(
            dense_hits, sparse_hits, k=self.config.rrf_k,
            w_dense=self.config.w_dense, w_sparse=self.config.w_sparse, top_k=limit,
        )
        return [(rid, c) for rid, c, _m in fused]

    def retrieve(
        self,
        question: str,
        doc_ids: Optional[Sequence[str]] = None,
        options: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Return the top chunk texts for *question*, restricted to *doc_ids*.

        Single-pass (default): one hybrid retrieval over the question enriched
        with the option text. The dense + BM25 channels each retrieve a wider
        pool, fused with RRF; the doc filter is applied inside each channel's scan
        so ranking happens *within* the allowed set. When ``doc_ids`` is None all
        ingested docs are eligible.

        Multi-query (``config.multi_query``): retrieve the bare question AND each
        option as a separate sub-query, then union the results (keeping each
        chunk's best rank across sub-queries). Comparison / formula questions need
        evidence for *several* targets at once — one averaged query embedding
        tends to retrieve only one side, so per-target sub-queries surface every
        operand. Retrieval is local, so this costs no generation tokens; the extra
        chunks are still bounded by ``multi_query_cap`` and the compression budget.
        """
        if len(self.store) == 0:
            return []
        allowed = [str(d) for d in doc_ids] if doc_ids else None

        if self.config.multi_query and (options or _detect_anchors(question)):
            # Sub-queries: the bare question, one anchor-boosted query per temporal
            # operand in the question (so a comparison/computation retrieves BOTH
            # sides), and one option-boosted query per option (for fact-matching
            # mcq, where the answer sentence echoes an option's wording).
            subqueries = [question]
            subqueries += [f"{a} {question}" for a in _detect_anchors(question)]
            subqueries += [f"{question} {o}" for o in (options or [])]
            best: Dict[int, Tuple[float, str]] = {}
            for sq in subqueries:
                for rank, (rid, content) in enumerate(
                    self._hybrid_hits(sq, allowed, self.config.top_k)
                ):
                    score = 1.0 / (1 + rank)  # best rank of this chunk across sub-queries
                    if rid not in best or score > best[rid][0]:
                        best[rid] = (score, content)
            ranked = sorted(best.values(), key=lambda sc: sc[0], reverse=True)
            cap = max(self.config.top_k, self.config.multi_query_cap)
            return [content for _score, content in ranked[:cap]]

        enriched = self._enrich(question, options)
        return [c for _rid, c in self._hybrid_hits(enriched, allowed, self.config.top_k)]

    # -- answer -----------------------------------------------------------
    def answer_question(
        self,
        qid: str,
        question: str,
        options: Sequence[str],
        qtype: str = "mcq",
        doc_ids: Optional[Sequence[str]] = None,
    ) -> AnswerResult:
        """Answer one question and report the tokens it cost."""
        passages = self.retrieve(question, doc_ids=doc_ids, options=options)
        # Compress against question + options so an answer sentence matching a
        # distractor's wording is retained, not dropped.
        compressed = self.compressor.compress(self._enrich(question, options), passages)
        user_prompt = self._build_prompt(question, options, qtype, compressed.text)
        result: ChatResult = self.chat_client.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=self.config.max_answer_tokens,
        )
        answer = parse_answer(result.text, qtype, num_options=len(options) or 4)
        return AnswerResult(qid=str(qid), answer=answer, qtype=qtype, usage=result.usage)

    @staticmethod
    def _build_prompt(
        question: str, options: Sequence[str], qtype: str, context: str
    ) -> str:
        labels = [chr(ord("A") + i) for i in range(len(options))]
        opt_lines = "\n".join(f"{lab}) {opt}" for lab, opt in zip(labels, options))
        kind = {
            "mcq": "Single choice: reply with exactly one letter.",
            "tf": "True/false: reply with exactly one letter.",
            "multi": "Multiple choice: reply with all correct letters, e.g. ABD.",
        }.get(qtype, "Reply with the option letter(s).")
        ctx = context or "(no relevant context retrieved)"
        return (
            f"Context:\n{ctx}\n\n"
            f"Question: {question}\n"
            f"Options:\n{opt_lines}\n\n"
            f"{kind}\nAnswer:"
        )
