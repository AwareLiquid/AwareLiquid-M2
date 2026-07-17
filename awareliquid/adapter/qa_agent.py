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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..memory.encoder import SentenceEncoder
from ..memory.knowledge_store import PersistentKnowledgeMemory
from .chunker import chunk_document
from .compressor import ExtractiveCompressor
from .qwen_client import ChatResult, build_chat_client
from .schemas import AnswerResult, parse_answer

_SYSTEM_PROMPT = (
    "You are a precise financial document analyst. Answer the multiple-choice "
    "question using ONLY the provided context. Reason briefly, then reply with "
    "the option letter(s) only. For single-choice or true/false give exactly one "
    "letter; for multiple-choice give every correct letter with no separators "
    "(e.g. ACD). If the context is insufficient, choose the best-supported option."
)


@dataclass
class RetrievalConfig:
    max_chars: int = 800
    overlap_chars: int = 120
    top_k: int = 6
    pool_multiplier: int = 4  # over-retrieve, then filter by allowed doc_ids
    compression_budget: int = 1200
    max_answer_tokens: int = 32


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
    def retrieve(
        self, question: str, doc_ids: Optional[Sequence[str]] = None
    ) -> List[str]:
        """Return the top-k chunk texts for *question*, restricted to *doc_ids*.

        The competition provides a ``doc_ids`` range per question; honouring it
        keeps retrieval on the intended documents and avoids cross-document
        contamination. When ``doc_ids`` is None, all ingested docs are eligible.
        """
        if len(self.store) == 0:
            return []
        allowed = {str(d) for d in doc_ids} if doc_ids else None
        q_key = self.encoder.encode(question, is_query=True)
        # Over-retrieve so post-filtering by doc_id still yields top_k survivors.
        pool = self.config.top_k * (self.config.pool_multiplier if allowed else 1)
        hits = self.store.query(q_key, top_k=min(pool, len(self.store)), center=True)
        texts: List[str] = []
        for content, _score, meta in hits:
            if allowed is not None and str((meta or {}).get("doc_id")) not in allowed:
                continue
            texts.append(str(content))
            if len(texts) >= self.config.top_k:
                break
        return texts

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
        passages = self.retrieve(question, doc_ids=doc_ids)
        compressed = self.compressor.compress(question, passages)
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
