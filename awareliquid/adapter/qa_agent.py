"""End-to-end memory-augmented QA over long documents.

``MemoryQAAgent`` wires the pieces together into the external adapter loop:

    ingest   document text -> chunks -> lexical/vector store
    answer    question -> retrieve chunks -> compress -> compact prompt
              -> Qwen chat API -> parsed answer + token usage

The base model is never modified: it is reached only through the injected chat
client. All retrieval and compression run locally, so they add nothing to the
generation-token bill -- the only tokens spent are the compact prompt and the
short answer.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .compressor import ExtractiveCompressor
from .evidence_index import build_evidence_nodes
from .hybrid import rrf_fuse
from .qwen_client import (
    ChatResult,
    DEFAULT_TOKEN_BUDGET,
    QwenChatClient,
    TokenUsage,
    build_chat_client,
)
from .schemas import AnswerResult, parse_answer

# Temporal anchors that split a comparison/computation question into operands:
# a year ("2023" / "2023 年"), or a quarter ("Q3", "第三季度", "三季度"/"上半年").
_YEAR = re.compile(r"(?:19|20)\d{2}")
_QUARTER = re.compile(r"Q[1-4]|第[一二三四1-4]季度|[上下]半年|[一二三四1-4]季度")
_CLAIM_RELATION = re.compile(
    r"高于|低于|超过|不超过|少于|大于|小于|同比|增长|下降|增幅|复合增速|"
    r"占比|比例|每股|每10股|倍|差额|分别|排序|高到低|低到高|升序|降序"
)
_CALCULATION_MARKER = re.compile(
    r"计算|排序|高到低|低到高|升序|降序|合计|总计|平均|金额|现金价值|退保所得|"
    r"每股|每10股"
)
_OPTION_STATE = re.compile(
    r"^\s*([A-Z])\s*=\s*(SUPPORTED|REFUTED|INSUFFICIENT)\b",
    re.IGNORECASE,
)
_OPTION_RELATION_STATE = re.compile(
    r"^\s*([A-Z])\s*:\s*.*?\bstate\s*=\s*"
    r"(SUPPORTED|REFUTED|INSUFFICIENT)\b",
    re.IGNORECASE,
)
_ANSWER_LINE = re.compile(r"(?:^|\n)\s*ANSWER\s*:\s*([A-Z]+)\b", re.IGNORECASE)
_EVIDENCE_SLOT_GROUPS = (
    ("营业收入", "营业总收入", "主营业务收入"),
    ("归母净利润", "归属于上市公司股东的净利润"),
    ("经营现金流", "经营活动产生的现金流量净额", "经营活动现金流量净额"),
    ("研发投入", "研发费用", "研发投入占营业收入比例"),
    ("现金分红", "每10股现金分红", "利润分配方案"),
    ("一般医疗", "一般医疗保险金", "住院医疗费用"),
    ("免赔额", "共享免赔额", "家庭共享免赔额"),
    ("施救费用", "施救费", "必要合理施救费用"),
    ("除外责任", "免责", "不承担责任"),
)
_EVIDENCE_ANCHOR = re.compile(
    r"(?:19|20)\d{2}\s*年?"
    r"|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:个百分点|亿元|万元|美元|港元|bp|BP|%|％|亿|万|元)"
    r"|第\s*[一二三四五六七八九十百\d]+\s*(?:条|章|节|款|项|个保单年度|保单年度)"
    r"|AAA|AA\+?"
    r"|除外|例外|但|不适用|特殊情形|免责|等待期"
    ,
)
def _normalise_evidence_anchor(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def _missing_evidence_anchors(question: str, option: str, evidence: str) -> List[str]:
    """Find exact, high-signal anchors from a claim missing in its evidence."""
    anchors: List[str] = []
    seen = set()
    # Option anchors are the primary coverage contract.  Question anchors are
    # added only when they are structural facts (dates, amounts, clauses, ...),
    # not ordinary connective words such as ``但``.  This keeps a supplement
    # query targeted instead of turning every multi-choice question into a
    # broad second retrieval pass.
    for match in _EVIDENCE_ANCHOR.finditer(f"{option} {question}"):
        anchor = match.group(0).strip()
        key = _normalise_evidence_anchor(anchor)
        if key and key not in seen and len(key) > 1:
            seen.add(key)
            anchors.append(anchor)
    evidence_text = _normalise_evidence_anchor(evidence)
    return [anchor for anchor in anchors if _normalise_evidence_anchor(anchor) not in evidence_text]


def _missing_evidence_slots(question: str, option: str, evidence: str) -> List[str]:
    source = f"{question} {option}"
    normalized = _normalise_evidence_anchor(evidence)
    missing = []
    for group in _EVIDENCE_SLOT_GROUPS:
        if not any(term in source for term in group):
            continue
        if any(_normalise_evidence_anchor(term) in normalized for term in group):
            continue
        missing.append(" ".join(group))
    return missing


def _verify_calculation_draft(raw: str) -> str:
    """Validate a Qwen calculation ledger with local Decimal arithmetic.

    The model may extract facts and propose derived values, but it is not
    trusted to perform the arithmetic.  Invalid or incomplete ledgers are
    discarded instead of being repeated as authoritative evidence.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""

    def parse_decimal(value):
        if isinstance(value, bool) or value is None:
            raise ValueError("invalid numeric value")
        text = str(value).strip().replace(",", "")
        if text.endswith("%"):
            return Decimal(text[:-1]) / Decimal("100")
        return Decimal(text)

    values = {}
    rendered = []
    for item in payload.get("facts", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        value = item.get("value")
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            parsed = parse_decimal(value)
        except (InvalidOperation, ValueError, TypeError):
            continue
        if not parsed.is_finite():
            continue
        key = name.strip()
        values[key] = parsed
        unit = item.get("unit") or ""
        rendered.append(f"- fact {key} = {parsed}{unit}")

    operations = {"add", "subtract", "multiply", "divide"}
    pending = [item for item in payload.get("derived", []) if isinstance(item, dict)]
    while pending:
        progressed = False
        remaining = []
        for item in pending:
            name = item.get("name")
            operation = item.get("operation")
            operands = item.get("operands")
            reported = item.get("value")
            if (
                not isinstance(name, str)
                or not name.strip()
                or operation not in operations
                or not isinstance(operands, list)
                or not operands
            ):
                continue
            if operation in {"subtract", "divide"} and len(operands) != 2:
                continue
            try:
                operand_values = [
                    values[ref] if isinstance(ref, str) and ref in values else parse_decimal(ref)
                    for ref in operands
                ]
                if operation == "add":
                    computed = sum(operand_values, Decimal("0"))
                elif operation == "subtract":
                    computed = operand_values[0] - operand_values[1]
                elif operation == "multiply":
                    computed = Decimal("1")
                    for value in operand_values:
                        computed *= value
                else:
                    if operand_values[1] == 0:
                        continue
                    computed = operand_values[0] / operand_values[1]
                reported_decimal = parse_decimal(reported)
            except (InvalidOperation, ValueError, TypeError, ZeroDivisionError):
                remaining.append(item)
                continue
            if not reported_decimal.is_finite() or reported_decimal != computed:
                continue
            key = name.strip()
            values[key] = computed
            unit = item.get("unit") or ""
            rendered.append(
                f"- derived {key} = {computed}{unit} "
                f"({operation}: {', '.join(str(ref) for ref in operands)})"
            )
            progressed = True
        if not progressed:
            break
        pending = remaining

    return "\n".join(rendered)


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


def _needs_claim_check(question: str, options: Sequence[str]) -> bool:
    """Enable relation checking only for claims that actually express one."""
    text = " ".join([question, *(str(option) for option in options)])
    return bool(_CLAIM_RELATION.search(text))


def _majority_vote(answers: Sequence[str], qtype: str, num_options: int) -> str:
    """Aggregate independent answer samples into one answer.

    Voting happens per LETTER, not on the whole answer string. That distinction
    is what makes this useful for multi-select: a spurious letter the model only
    emits in a minority of samples is dropped, while a letter it consistently
    supports survives. Whole-string voting could not separate the two, because
    "AB" and "ABD" are simply different strings.

    Single-answer formats take the most frequent letter. Ill-formed samples are
    ignored; if every sample is ill-formed the first one is returned unchanged so
    the caller's existing canonicality retry still applies.
    """
    valid = [a for a in answers if _is_canonical_answer(a, qtype, num_options)]
    if not valid:
        return answers[0] if answers else ""

    if qtype in ("mcq", "tf"):
        return Counter(valid).most_common(1)[0][0]

    # multi: keep a letter only when a strict majority of samples carries it.
    threshold = len(valid) / 2
    letter_votes = Counter(letter for answer in valid for letter in set(answer))
    kept = sorted(letter for letter, votes in letter_votes.items() if votes > threshold)
    if kept:
        return "".join(kept)
    # Nothing reached a majority (e.g. two samples disagreeing completely) --
    # fall back to the most frequent complete answer rather than inventing one.
    return Counter(valid).most_common(1)[0][0]


def _needs_calculation_judgement(question: str, options: Sequence[str]) -> bool:
    """Use a second Qwen pass only when the question explicitly requires math."""
    # Do not trigger merely because a distractor contains a percentage, amount,
    # or per-share figure.  The stem must itself ask for a computation/order.
    return bool(_CALCULATION_MARKER.search(str(question)))


_SYSTEM_PROMPT = (
    "You are a precise financial document analyst. Answer the multiple-choice "
    "question using ONLY the provided context. Reply with the option letter(s) "
    "and NOTHING else -- no words, no reasoning, no punctuation. For single-choice "
    "or true/false output exactly one letter (e.g. B). For multiple-choice output "
    "every correct letter with no separators (e.g. ACD). If the context is "
    "insufficient, output the single best-supported letter."
)
_STRUCTURED_SYSTEM_PROMPT = (
    "You are a precise financial document analyst. Use only the supplied context. "
    "Follow the requested compact option-state format exactly and do not add explanations."
)


def _parse_structured_answer(raw: str, qtype: str, num_options: int) -> str:
    """Parse one-call option states, with a safe fallback for nonconforming output."""
    text = raw or ""
    state_pairs = []
    for line in text.splitlines():
        match = _OPTION_RELATION_STATE.search(line) or _OPTION_STATE.search(line)
        if match:
            state_pairs.append((match.group(1), match.group(2)))
    states = {
        label.upper(): state.upper()
        for label, state in state_pairs
        if ord("A") <= ord(label.upper()) < ord("A") + max(1, num_options)
    }
    if states:
        supported = "".join(
            label for label in sorted(states) if states[label] == "SUPPORTED"
        )
        if qtype == "multi":
            return supported
        if supported:
            return supported[0]
    answer_line = _ANSWER_LINE.search(text)
    if answer_line:
        return parse_answer(answer_line.group(1), qtype, num_options=num_options)
    return parse_answer(text, qtype, num_options=num_options)


def _is_canonical_answer(answer: str, qtype: str, num_options: int) -> bool:
    allowed = {chr(ord("A") + i) for i in range(max(1, num_options))}
    if qtype in {"mcq", "tf"}:
        return answer in allowed and (qtype != "tf" or answer in {"A", "B"})
    return bool(answer) and set(answer) <= allowed and answer == "".join(sorted(set(answer)))


def _merge_evidence_blocks(contexts: Sequence[str], max_chars: int) -> str:
    """Deduplicate source blocks and enforce one final context budget."""
    if max_chars <= 0:
        return ""
    blocks: List[str] = []
    seen = set()
    used = 0
    for context in contexts:
        for block in (part.strip() for part in (context or "").split("\n\n")):
            if not block:
                continue
            key = re.sub(r"\s+", " ", block).strip().lower()
            if key in seen:
                continue
            separator = 2 if blocks else 0
            remaining = max_chars - used - separator
            if remaining <= 0:
                return "\n\n".join(blocks)
            seen.add(key)
            selected = block[:remaining]
            blocks.append(selected)
            used += separator + len(selected)
            if len(selected) < len(block):
                return "\n\n".join(blocks)
    return "\n\n".join(blocks)


@dataclass
class RetrievalConfig:
    # ``hybrid`` preserves the original research path. ``lexical`` is the
    # dependency-light path used for local/offline testing; it needs no encoder
    # download and no GPU.
    retrieval_backend: str = "hybrid"  # "hybrid" | "lexical"
    competition_mode: bool = False      # opt-in future API validation only
    token_budget: int = 5_000_000
    usage_ledger_path: Optional[str] = None
    formal_run_id: Optional[str] = None
    formal_ledger_path: Optional[str] = None
    # max_chars stays at or below the encoder's token window (e5: 512) so a whole
    # chunk is embedded rather than silently truncated.
    max_chars: int = 450
    overlap_chars: int = 80
    top_k: int = 6
    compression_budget: int = 3000
    max_answer_tokens: int = 16
    # -- hybrid retrieval (dense cosine + lexical BM25, fused with RRF) --
    hybrid: bool = True          # fuse BM25 with cosine when FTS5 is available
    rrf_pool: int = 20           # candidates per channel before fusion
    rrf_k: int = 10              # RRF constant (small = sharper head)
    w_dense: float = 0.7         # dense (semantic) fusion weight
    w_sparse: float = 0.3        # lexical (BM25) fusion weight
    center: bool = True          # anisotropy-robust centered cosine for the dense channel
    # -- multi-query retrieval (per-option sub-queries, union) --
    multi_query: bool = True     # retrieve the question + each option separately, then union
    multi_query_cap: int = 12    # max chunks kept after the union
    option_audit: bool = False   # opt-in; audit can add cost without improving every task
    option_audit_max_tokens: int = 256
    multi_option_audit: bool = True  # independently check every option on multi-choice
    structured_judgement: bool = True
    structured_judgement_max_tokens: int = 256
    # Independent answer samples aggregated by per-letter majority vote. 1 = off.
    # The provider is non-deterministic even at temperature 0, so a single sample
    # carries real variance; >1 trades tokens for a more stable answer.
    self_consistency: int = 1
    self_consistency_temperature: float = 0.0
    calculation_judgement: bool = True
    calculation_judgement_max_tokens: int = 512
    option_evidence_budget: int = 900
    option_evidence_per_document: bool = True
    # Deterministic, option-local gap filling.  It is deliberately enabled by
    # default: it changes only local retrieval, never the Qwen answer policy.
    evidence_coverage_supplement: bool = True
    evidence_coverage_max_anchors: int = 4
    evidence_coverage_supplement_budget: int = 900
    structure_index: bool = True
    structure_neighbor_expansion: bool = True
    structure_neighbor_cap: int = 2
    answer_context_budget: int = 12_000
    adaptive_multi_review: bool = False
    adaptive_multi_review_max_tokens: int = 128


class MemoryQAAgent:
    """Retrieve-compress-answer agent backed by a local vector store."""

    def __init__(
        self,
        encoder: Optional[Any] = None,
        store: Optional[Any] = None,
        chat_client=None,
        compressor: Optional[ExtractiveCompressor] = None,
        config: Optional[RetrievalConfig] = None,
        db_path: str = ":memory:",
    ):
        self.config = config or RetrievalConfig()
        if self.config.retrieval_backend not in {"hybrid", "lexical"}:
            raise ValueError(
                "retrieval_backend must be 'hybrid' or 'lexical', got "
                f"{self.config.retrieval_backend!r}"
            )

        if self.config.retrieval_backend == "lexical":
            if encoder is not None:
                raise ValueError("lexical retrieval must not receive an encoder")
            from ..memory.lexical_store import LexicalKnowledgeMemory

            self.encoder = None
            self.store = store or LexicalKnowledgeMemory(db_path=db_path)
            if not hasattr(self.store, "search_bm25"):
                raise TypeError("lexical retrieval requires a lexical store")
        else:
            # Keep the vector/embedding path lazy so importing or using the
            # lexical path never initializes a model or downloads weights.
            from ..memory.encoder import SentenceEncoder
            from ..memory.knowledge_store import PersistentKnowledgeMemory

            self.encoder = encoder or SentenceEncoder()
            # The store's key_dim must equal the encoder's output dim; probe once.
            self.store = store or PersistentKnowledgeMemory(
                key_dim=self.encoder.dim, db_path=db_path
            )

        self.chat_client = chat_client or build_chat_client(
            competition_mode=self.config.competition_mode,
            token_budget=self.config.token_budget,
            usage_ledger_path=self.config.usage_ledger_path,
            formal_run_id=self.config.formal_run_id,
            formal_ledger_path=self.config.formal_ledger_path,
        )
        if self.config.competition_mode:
            # Formal execution has exactly one accepted chat dependency. Exact
            # type checking prevents an injected look-alike/subclass from
            # bypassing Qwen's provider, ledger, or transport-denial gates.
            if type(self.chat_client) is not QwenChatClient:
                raise RuntimeError(
                    "formal mode requires a validated formal QwenChatClient; "
                    "mock or arbitrary injected clients are forbidden"
                )
            if self.config.token_budget != DEFAULT_TOKEN_BUDGET:
                raise ValueError(
                    f"formal mode requires token_budget={DEFAULT_TOKEN_BUDGET}"
                )
            self.chat_client.assert_formal_configuration()
            if not isinstance(self.config.formal_run_id, str) or not self.config.formal_run_id.strip():
                raise ValueError("formal mode requires a non-empty formal_run_id")
            if not isinstance(self.config.formal_ledger_path, str) or not self.config.formal_ledger_path.strip():
                raise ValueError("formal mode requires an explicit formal_ledger_path")
            if self.chat_client.formal_run_id != self.config.formal_run_id.strip():
                raise ValueError("formal Qwen client run id does not match agent configuration")
            if self.chat_client.formal_ledger_path != str(
                Path(self.config.formal_ledger_path).expanduser().resolve(strict=False)
            ):
                raise ValueError("formal Qwen client ledger path does not match agent configuration")
        self.compressor = compressor or ExtractiveCompressor(
            budget_chars=self.config.compression_budget
        )
        self._ingested_docs: set = set()

    # -- ingest -----------------------------------------------------------
    def ingest_document(self, doc_id: str, text: str) -> int:
        """Ingest exact text with deterministic structure metadata."""
        nodes = build_evidence_nodes(
            doc_id,
            text,
            max_chars=self.config.max_chars,
            overlap_chars=self.config.overlap_chars,
        )
        for node in nodes:
            meta = node.as_meta() if self.config.structure_index else {
                "doc_id": node.doc_id,
                "chunk_idx": node.chunk_idx,
            }
            if self.config.retrieval_backend == "lexical":
                self.store.write(node.content, meta=meta)
            else:
                key = self.encoder.encode(node.content, is_query=False)
                self.store.write(key, node.content, meta=meta)
        self._ingested_docs.add(str(doc_id))
        return len(nodes)

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
    ) -> List[Tuple[int, str, Any]]:
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
            return dense_hits[:limit]
        sparse_hits = self.store.search_bm25(query_text, top_k=pool, doc_ids=allowed)
        fused = rrf_fuse(
            dense_hits, sparse_hits, k=self.config.rrf_k,
            w_dense=self.config.w_dense, w_sparse=self.config.w_sparse, top_k=limit,
        )
        return fused

    def _lexical_hits(
        self, query_text: str, allowed: Optional[List[str]], limit: int
    ) -> List[Tuple[int, str, Any]]:
        """Retrieve only through local FTS5/BM25, with no dense fallback."""
        if not getattr(self.store, "fts_enabled", False):
            if self.config.competition_mode:
                raise RuntimeError("lexical retrieval requires SQLite FTS5")
            return []
        hits = self.store.search_bm25(query_text, top_k=limit, doc_ids=allowed)
        return hits

    @staticmethod
    def _format_passage(content: str, meta: Any) -> str:
        """Keep source provenance attached to every retrieved passage."""
        if isinstance(meta, dict):
            source = meta.get("doc_id", "unknown")
            chunk = meta.get("chunk_idx")
            suffix = f" chunk={chunk}" if chunk is not None else ""
            page = meta.get("page")
            if page is not None:
                suffix += f" page={page}"
            section = meta.get("section")
            if section:
                suffix += f" section={section}"
            clause = meta.get("clause_id")
            if clause:
                suffix += f" clause={clause}"
            # Polarity is answer-bearing in clause text and easy to skim past:
            # "X 属于责任免除" and "X 不属于责任免除" differ by one character.
            # Restating the negated sentences after the passage makes the
            # polarity explicit instead of leaving it buried mid-paragraph.
            notes = []
            negations = meta.get("negations") or []
            if negations:
                joined = " / ".join(str(item) for item in negations)
                notes.append(f"[NEGATED — read literally, this DENIES the claim: {joined}]")
            exceptions = meta.get("exceptions") or []
            if exceptions:
                joined = " / ".join(str(item) for item in exceptions)
                notes.append(
                    "[EXCEPTION — this carves a case OUT of the surrounding rule, so "
                    f"the carved-out case is treated the OPPOSITE way: {joined}]"
                )
            if notes:
                return f"[source doc_id={source}{suffix}]\n{content}\n" + "\n".join(notes)
        else:
            source = "unknown"
            suffix = ""
        return f"[source doc_id={source}{suffix}]\n{content}"

    def _expand_structural_hits(self, hits):
        if (
            self.config.retrieval_backend != "lexical"
            or not self.config.structure_index
            or not self.config.structure_neighbor_expansion
        ):
            return hits
        return self.store.expand_neighbors(
            hits, max_extra=self.config.structure_neighbor_cap
        )

    def _locate_documents(
        self, query_text: str, options: Optional[Sequence[str]]
    ) -> List[str]:
        """Find B榜 candidate documents before searching their passages.

        This is deliberately a small lexical locator. It prevents an omitted
        ``doc_ids`` value from silently turning a formal run into an unrestricted
        dense search while keeping the local test path simple.
        """
        if not getattr(self.store, "fts_enabled", False):
            raise RuntimeError("document locator requires SQLite FTS5")
        query = self._enrich(query_text, options)
        hits = self.store.search_bm25(query, top_k=max(32, self.config.top_k * 4))
        candidates: List[str] = []
        seen = set()
        for _rid, _content, meta in hits:
            if not isinstance(meta, dict) or meta.get("doc_id") is None:
                continue
            doc_id = str(meta["doc_id"])
            if doc_id not in seen:
                seen.add(doc_id)
                candidates.append(doc_id)
        return candidates

    def retrieve(
        self,
        question: str,
        doc_ids: Optional[Sequence[str]] = None,
        options: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
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
        normalized_split = split.upper() if isinstance(split, str) else None
        if normalized_split not in {None, "A", "B"}:
            raise ValueError("split must be A or B")
        if normalized_split == "A" and not doc_ids:
            raise ValueError("split A requires non-empty doc_ids")
        if normalized_split == "B" and doc_ids:
            raise ValueError("split B must not receive doc_ids")

        allowed: Optional[List[str]]
        if normalized_split == "B":
            allowed = self._locate_documents(question, options)
            if not allowed:
                raise RuntimeError("B candidate locator found no documents")
        elif doc_ids is None:
            allowed = None
        else:
            allowed = [str(d) for d in doc_ids]
            if not allowed:
                if self.config.competition_mode:
                    raise ValueError("an explicit document filter cannot be empty")
                return []

        if self.config.competition_mode:
            if self.config.retrieval_backend != "lexical":
                raise RuntimeError(
                    "competition mode is opt-in and requires retrieval_backend='lexical'"
                )
            if allowed is None:
                allowed = self._locate_documents(question, options)
                if not allowed:
                    raise RuntimeError("document locator found no candidate documents")
            elif not self.store.has_doc_ids(allowed):
                raise ValueError(f"unknown document id in filter: {allowed}")

        if len(self.store) == 0:
            return []

        if self.config.multi_query and (options or _detect_anchors(question)):
            # Sub-queries: the bare question, one anchor-boosted query per temporal
            # operand in the question (so a comparison/computation retrieves BOTH
            # sides), and one option-boosted query per option (for fact-matching
            # mcq, where the answer sentence echoes an option's wording).
            subqueries = [question]
            # Keep the temporal channel narrowly anchored.  Adding the full
            # question here turns a precise year/quarter lookup into a large
            # OR query whose tied BM25 results can crowd out the very operand
            # this channel exists to protect.  The question and option
            # channels below still provide semantic/lexical context.
            anchor_queries = list(_detect_anchors(question))
            subqueries += anchor_queries
            subqueries += [f"{question} {o}" for o in (options or [])]
            best: Dict[int, Tuple[float, str, Any]] = {}
            required_anchor_ids: List[int] = []
            for sq in subqueries:
                hits = (
                    self._lexical_hits(sq, allowed, self.config.top_k)
                    if self.config.retrieval_backend == "lexical"
                    else self._hybrid_hits(sq, allowed, self.config.top_k)
                )
                if sq in anchor_queries and hits:
                    # Keep one best passage for every temporal operand before
                    # global fusion fills the remaining budget.
                    required_anchor_ids.append(hits[0][0])
                for rank, (rid, content, meta) in enumerate(hits):
                    score = 1.0 / (1 + rank)  # best rank of this chunk across sub-queries
                    if rid not in best or score > best[rid][0]:
                        best[rid] = (score, content, meta)
            ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
            cap = max(self.config.top_k, self.config.multi_query_cap)
            required_ids = list(dict.fromkeys(required_anchor_ids))
            # A question may explicitly reference several documents.  Global
            # BM25 ranking can otherwise spend the entire cap on the document
            # whose wording happens to overlap most, hiding evidence from the
            # other referenced documents.  Reserve the best retrieved chunk
            # from every allowed document before filling the remaining budget.
            if allowed and len(allowed) > 1:
                best_by_doc: Dict[str, int] = {}
                for rid, (_score, _content, meta) in ranked:
                    if not isinstance(meta, dict):
                        continue
                    doc_id = str(meta.get("doc_id", ""))
                    if doc_id and doc_id in allowed and doc_id not in best_by_doc:
                        best_by_doc[doc_id] = rid
                required_ids.extend(best_by_doc.values())
            selected_ids = required_ids + [
                rid for rid, _value in ranked if rid not in set(required_ids)
            ]
            selected = [best[rid] for rid in selected_ids[:cap]]
            selected_hits = [
                (rid, content, meta)
                for rid, (_score, content, meta) in zip(selected_ids, selected)
            ]
            selected_hits = self._expand_structural_hits(selected_hits)
            return [
                self._format_passage(content, meta)
                for _rid, content, meta in selected_hits
            ]

        enriched = self._enrich(question, options)
        hits = (
            self._lexical_hits(enriched, allowed, self.config.top_k)
            if self.config.retrieval_backend == "lexical"
            else self._hybrid_hits(enriched, allowed, self.config.top_k)
        )
        hits = self._expand_structural_hits(hits)
        return [self._format_passage(content, meta) for _rid, content, meta in hits]

    # -- answer -----------------------------------------------------------
    def answer_question(
        self,
        qid: str,
        question: str,
        options: Sequence[str],
        qtype: str = "mcq",
        doc_ids: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
    ) -> AnswerResult:
        """Answer one question and report the tokens it cost."""
        passages = self.retrieve(
            question, doc_ids=doc_ids, options=options, split=split
        )
        # Compress against question + options so an answer sentence matching a
        # distractor's wording is retained, not dropped.
        compressed = self.compressor.compress(
            self._enrich(question, options),
            passages,
            coverage_queries=[question, *options],
        )
        context = compressed.text
        option_evidence = []
        evidence_scope = None
        if self.config.option_evidence_per_document:
            if isinstance(split, str) and split.upper() == "B":
                evidence_scope = self._locate_documents(question, options)
            elif doc_ids:
                evidence_scope = [str(doc_id) for doc_id in doc_ids]
        for label, option in zip(
            (chr(ord("A") + i) for i in range(len(options))), options
        ):
            if evidence_scope:
                option_passages = []
                for evidence_doc_id in evidence_scope:
                    option_passages.extend(
                        self.retrieve(
                            question,
                            doc_ids=[evidence_doc_id],
                            options=[option],
                            split="A",
                        )
                    )
            else:
                option_passages = self.retrieve(
                    question,
                    doc_ids=doc_ids,
                    options=[option],
                    split=split,
                )
            option_compressed = ExtractiveCompressor(
                budget_chars=self.config.option_evidence_budget
            ).compress(
                option,
                option_passages,
                coverage_queries=[question, option],
            )
            option_text = option_compressed.text
            if self.config.evidence_coverage_supplement:
                missing_targets = [
                    *_missing_evidence_anchors(question, option, option_text),
                    *_missing_evidence_slots(question, option, option_text),
                ][: self.config.evidence_coverage_max_anchors]
                supplement_texts = []
                for target in missing_targets:
                    # Each supplement is a narrow lexical query.  It does not
                    # alter the primary ranking and never invokes Qwen.
                    supplement_passages = self.retrieve(
                        target,
                        doc_ids=evidence_scope if evidence_scope else doc_ids,
                        split="A" if evidence_scope else split,
                    )
                    supplement = ExtractiveCompressor(
                        budget_chars=self.config.evidence_coverage_supplement_budget
                    ).compress(
                        target,
                        supplement_passages,
                        coverage_queries=[target],
                    )
                    if supplement.text:
                        supplement_texts.append(supplement.text)
                if supplement_texts:
                    option_text = "\n".join([option_text, *supplement_texts])
            option_evidence.append(
                f"Option {label} evidence:\n"
                + (option_text or "(no option-specific evidence)")
            )
        context = _merge_evidence_blocks(
            [context, *option_evidence], self.config.answer_context_budget
        )
        audit_text = ""
        audit_usage = TokenUsage()
        if self.config.option_audit or (
            qtype == "multi" and self.config.multi_option_audit
        ):
            audit_result: ChatResult = self.chat_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an evidence auditor. Use ONLY the provided context. "
                            "Evaluate every option independently. For each option output "
                            "one compact line: LETTER=SUPPORTED, REFUTED, or INSUFFICIENT; "
                            "then cite the shortest supporting or contradicting phrase. "
                            "Do not give a final answer set."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_audit_prompt(
                            question, options, qtype, context
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=self.config.option_audit_max_tokens,
            )
            audit_text = audit_result.text.strip()
            audit_usage = audit_result.usage
        calculation_draft = ""
        calculation_usage = TokenUsage()
        if (
            self.config.structured_judgement
            and not audit_text
            and self.config.calculation_judgement
            and _needs_calculation_judgement(question, options)
        ):
            calculation_result: ChatResult = self.chat_client.chat(
                messages=[
                    {"role": "system", "content": _STRUCTURED_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self._build_calculation_prompt(
                            question, options, context
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=self.config.calculation_judgement_max_tokens,
            )
            calculation_draft = _verify_calculation_draft(calculation_result.text)
            calculation_usage = calculation_result.usage
        user_prompt = (
            self._build_structured_prompt(
                question, options, qtype, context, calculation_draft=calculation_draft
            )
            if self.config.structured_judgement and not audit_text
            else self._build_prompt(
                question, options, qtype, context, audit_text=audit_text
            )
        )
        messages = [
            {
                "role": "system",
                "content": (
                    _STRUCTURED_SYSTEM_PROMPT
                    if self.config.structured_judgement
                    else _SYSTEM_PROMPT
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        answer_max_tokens = (
            self.config.structured_judgement_max_tokens
            if self.config.structured_judgement
            else self.config.max_answer_tokens
        )
        # Self-consistency: draw N independent samples and aggregate them. The
        # provider is not deterministic even at temperature 0 (MoE routing and
        # server-side batching), so a single sample carries real variance; voting
        # over several samples reports the answer the model actually favours
        # rather than whichever one this particular call happened to produce.
        samples = max(1, int(self.config.self_consistency))
        sampled_answers: List[str] = []
        first_text = ""
        total_usage = TokenUsage()
        for index in range(samples):
            sample: ChatResult = self.chat_client.chat(
                messages=messages,
                temperature=self.config.self_consistency_temperature if samples > 1 else 0.0,
                max_tokens=answer_max_tokens,
            )
            total_usage = total_usage + sample.usage
            if index == 0:
                first_text = sample.text
            sampled_answers.append(
                _parse_structured_answer(sample.text, qtype, len(options) or 4)
                if self.config.structured_judgement
                else parse_answer(sample.text, qtype, num_options=len(options) or 4)
            )
        result = ChatResult(text=first_text, usage=total_usage)
        answer = (
            sampled_answers[0]
            if samples == 1
            else _majority_vote(sampled_answers, qtype, len(options) or 4)
        )
        # A provider can occasionally return an empty or non-canonical boolean
        # despite the structured prompt.  Retry the format conversion once
        # without changing the retrieved evidence or the underlying question.
        # Never guess a side of a true/false question locally.
        if not _is_canonical_answer(answer, qtype, len(options) or 4):
            retry_usage = TokenUsage()
            retry_options = "\n".join(
                f"{chr(ord('A') + i)}: {option}" for i, option in enumerate(options)
            )
            retry_prompt = (
                f"Question: {question}\nOptions:\n{retry_options}\n"
                f"Evidence:\n{context[:12000]}\n"
            )
            for _ in range(3):
                if qtype == "tf":
                    instruction = "return A if the statement is true/correct, otherwise return B"
                elif qtype == "multi":
                    instruction = "return every correct option letter, sorted with no separators"
                else:
                    instruction = "return exactly one valid option letter"
                retry = self.chat_client.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return only the canonical answer. No other text. "
                                + instruction + "."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Using the same question and evidence above, "
                                + instruction + ".\n\n"
                                + retry_prompt
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=8,
                )
                retry_usage = retry_usage + retry.usage
                retry_answer = parse_answer(retry.text, qtype, num_options=len(options) or 4)
                if _is_canonical_answer(retry_answer, qtype, len(options) or 4):
                    answer = retry_answer
                    break
            if _is_canonical_answer(answer, qtype, len(options) or 4):
                result = ChatResult(
                    text=result.text,
                    usage=result.usage + retry_usage,
                )
        review_usage = TokenUsage()
        if (
            self.config.adaptive_multi_review
            and qtype == "multi"
            and len(answer) <= 1
        ):
            review_result = self.chat_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": _STRUCTURED_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": self._build_multi_review_prompt(
                            question, options, context, answer
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=self.config.adaptive_multi_review_max_tokens,
            )
            review_usage = review_result.usage
            reviewed = _parse_structured_answer(
                review_result.text, qtype, len(options) or 4
            )
            if _is_canonical_answer(reviewed, qtype, len(options) or 4):
                answer = reviewed
        return AnswerResult(
            qid=str(qid), answer=answer, qtype=qtype,
            usage=audit_usage + calculation_usage + result.usage + review_usage,
        )

    @staticmethod
    def _build_prompt(
        question: str,
        options: Sequence[str],
        qtype: str,
        context: str,
        audit_text: str = "",
    ) -> str:
        labels = [chr(ord("A") + i) for i in range(len(options))]
        opt_lines = "\n".join(f"{lab}) {opt}" for lab, opt in zip(labels, options))
        kind = {
            "mcq": "Single choice: reply with exactly one letter.",
            "tf": "True/false: reply with exactly one letter.",
            "multi": "Multiple choice: reply with all correct letters, e.g. ABD.",
        }.get(qtype, "Reply with the option letter(s).")
        ctx = context or "(no relevant context retrieved)"
        claim_check = (
            "Before choosing, silently check each option as a claim: identify its subject, "
            "metric, time period, qualifiers, and stated relation. For greater/less than, "
            "compare the explicitly stated quantities only when their units and periods "
            "match; do not reverse the relation or substitute a nearby but different "
            "quantity. Treat absent evidence as insufficient rather than as a contradiction.\n"
            if _needs_claim_check(question, options)
            else ""
        )
        return (
            f"Context:\n{ctx}\n\n"
            + (
                "Option evidence audit (a draft; verify it against Context):\n"
                + audit_text
                + "\n\n"
                if audit_text
                else ""
            )
            + f"Question: {question}\n"
            + f"Options:\n{opt_lines}\n\n"
            + f"{kind}\n"
            + claim_check
            + ("Evaluate every option independently against the context; include a letter only when that option is supported.\n"
               if qtype == "multi" else "")
            + "Answer:"
        )

    @staticmethod
    def _build_structured_prompt(
        question: str,
        options: Sequence[str],
        qtype: str,
        context: str,
        calculation_draft: str = "",
    ) -> str:
        labels = [chr(ord("A") + i) for i in range(len(options))]
        opt_lines = "\n".join(
            f"{label}) {option}" for label, option in zip(labels, options)
        )
        relation = (
            "For every comparison claim, first identify the two compared quantities and "
            "write each option exactly as LETTER: left=value; relation=<direction>; "
            "right=value; state=SUPPORTED|REFUTED|INSUFFICIENT before assigning "
            "the state. Preserve the question's direction and compare only matching units "
            "and periods; do not substitute a nearby unrelated number. A claim is supported "
            "when it is directly entailed by the context, even if the context contains "
            "additional detail beyond the wording of the option. For an ordering/ranking "
            "claim, list every object and its computed value, and support the option only "
            "when the complete ordering and all stated values match.\n"
            if _needs_claim_check(question, options)
            else (
                "A multi-choice option is supported when its complete claim is directly "
                "entailed by the context, even if the context contains additional detail. "
                "For multi-choice questions, evaluate each option independently and "
                "combine evidence across passages and referenced documents; do not require "
                "one passage to support every option, and do not infer support from silence "
                "or a related product.\n"
                if qtype == "multi"
                else ""
            )
        )
        return (
            f"Context:\n{context or '(no relevant context retrieved)'}\n\n"
            + (
                "A Qwen calculation draft is provided below. Verify it against Context "
                "and Question; correct it if unsupported. Do not use unsupported values.\n"
                "CALCULATION DRAFT:\n"
                + calculation_draft
                + "\n\n"
                if calculation_draft
                else ""
            )
            + f"Question: {question}\nOptions:\n{opt_lines}\n\n"
            "Evaluate every option independently using only direct evidence in Context. "
            "Do not infer a positive claim from silence or from a related product. "
            "An explicit negation or exclusion in Context (e.g. 不属于, 不免责, 不承担, 除外, "
            "不适用, 免收, does not, is not, excluded) REFUTES the matching option. "
            "For a sameness or equality claim (相同, 相等, 一致, the same, equal), mark SUPPORTED "
            "only when the compared values are explicitly equal; if they differ, REFUTE. "
            "Mark an option SUPPORTED only when Context contains an explicit statement that "
            "entails its complete claim; when the supporting evidence is weak, partial, or "
            "absent, mark INSUFFICIENT and exclude it rather than guessing. "
            + relation
            + "Output exactly one line per option in the form LETTER=STATE, where STATE is "
            "SUPPORTED, REFUTED, or INSUFFICIENT, followed by one final line ANSWER:LETTERS. "
            "Include only SUPPORTED letters in ANSWER, sorted; exclude INSUFFICIENT and REFUTED. "
            "Do not explain."
        )

    @staticmethod
    def _build_calculation_prompt(
        question: str, options: Sequence[str], context: str
    ) -> str:
        labels = [chr(ord("A") + i) for i in range(len(options))]
        opt_lines = "\n".join(
            f"{label}) {option}" for label, option in zip(labels, options)
        )
        return (
            f"Context:\n{context or '(no relevant context retrieved)'}\n\n"
            f"Question: {question}\nOptions:\n{opt_lines}\n\n"
            "Do not choose an option. Return JSON only with this shape: "
            '{"facts":[{"name":"fact name","value":"12.5","unit":"万元"}],'
            '"derived":[{"name":"result name","operation":"add",'
            '"operands":["fact name"],"value":"12.5","unit":"万元"}]}.'
            "Extract only numerical inputs explicitly stated in the question and "
            "formulas or rates explicitly stated in Context. Before deriving the result, "
            "include EVERY rate, fee, deduction, adjustment, or exception stated in Context "
            "that applies to this specific scenario — for example a surrender or handling fee, "
            "a year-tiered rate, or a waiting-period rule keyed to the stated policy year. A "
            "derived value that omits an applicable fee or adjustment is wrong. Use only add, "
            "subtract, multiply, or divide. Do not use numbers from answer options as inputs, "
            "do not invent fees or rates that Context does not state, and do not include a "
            "derived value unless its operands are present in facts."
        )

    @staticmethod
    def _build_audit_prompt(
        question: str, options: Sequence[str], qtype: str, context: str
    ) -> str:
        labels = [chr(ord("A") + i) for i in range(len(options))]
        opt_lines = "\n".join(f"{lab}) {opt}" for lab, opt in zip(labels, options))
        return (
            f"Context:\n{context or '(no relevant context retrieved)'}\n\n"
            f"Question: {question}\n"
            f"Options:\n{opt_lines}\n\n"
            f"Answer format: {qtype}.\n"
            "Audit each option independently."
        )

    @staticmethod
    def _build_multi_review_prompt(
        question: str,
        options: Sequence[str],
        context: str,
        initial_answer: str,
    ) -> str:
        labels = [chr(ord("A") + i) for i in range(len(options))]
        opt_lines = "\n".join(
            f"{label}) {option}" for label, option in zip(labels, options)
        )
        return (
            f"Initial answer: {initial_answer or '(empty)'}\n"
            f"Question: {question}\nOptions:\n{opt_lines}\n\n"
            f"Context:\n{context or '(no relevant context retrieved)'}\n\n"
            "The initial answer may have omitted correct options. Re-evaluate every "
            "option independently against the context, especially options omitted from "
            "the initial answer. A multi-choice option is SUPPORTED only when its complete "
            "claim is directly entailed; do not treat silence as REFUTED. Output exactly "
            "one line per option as LETTER=SUPPORTED, LETTER=REFUTED, or "
            "LETTER=INSUFFICIENT, followed by ANSWER:LETTERS containing all and only "
            "SUPPORTED options. Do not explain."
        )
