# AwareLiquid

**An external memory-compression adapter for long-document multiple-choice QA.**

AwareLiquid answers questions over documents far longer than a model's context
window, while spending as few generation tokens as possible. It is designed for
settings where the base model is **frozen and reachable only through an API** —
no fine-tuning, no LoRA, no embedding or reranking models in the answer path.

Everything that makes it work — structural document indexing, lexical retrieval,
evidence coverage, and context compression — runs **locally**, around the model
rather than inside it. The only tokens billed are the compact prompt and the
short answer.

```
          ┌────────────────── AwareLiquid adapter (local, no model weights) ─────────────────┐
document → │ evidence index → lexical (BM25/FTS5) retrieval → coverage → compression        │ → compact prompt ─┐
          └────────────────────────────────────────────────────────────────────────────────┘                   │
question ────────────────────────────────────────────────────────────────────────────────────────────────────► │
                                                                                                                ▼
                                                                                                   frozen LLM (HTTP API)
                                                                                                                │
                                                                                          answer + exact token usage ◄┘
```

## Why

A long filing rarely fits in a context window, and even when it does, feeding the
whole thing for every question is wasteful — most of the document is irrelevant to
any one question, and tokens cost money. AwareLiquid keeps documents in a local
index, pulls in only the passages a question needs, compresses those to their
load-bearing sentences, and sends the model a prompt that is a fraction of the
source.

It was built under three hard constraints that shape every design decision:

1. **The base model is frozen.** Generation happens only over HTTP. Nothing in
   this repository trains, fine-tunes, or modifies model weights.
2. **No embedding/rerank models in the answer path.** Retrieval is purely
   lexical (BM25 over SQLite FTS5) plus document structure. A dense/hybrid path
   exists but is research-only (see below).
3. **A hard total token budget.** Spend is metered by a persistent, crash-safe
   ledger that fails closed when the budget is exhausted.

## How it works

| Stage | Module | What it does |
|-------|--------|--------------|
| **Index** | `adapter/evidence_index.py` | Splits documents into evidence nodes carrying page, section, clause id, table row/column and numeric *anchors* (years, amounts, %, clause numbers, ratings). |
| **Store** | `memory/lexical_store.py` | SQLite FTS5 with an 11-column **field-weighted** BM25 index (clause id and anchors weighted 6× content), explicit CJK n-gram expansion, structural re-ranking, and neighbour expansion. |
| **Retrieve** | `adapter/qa_agent.py` | Multi-query retrieval: the bare question, one anchor-boosted sub-query per temporal operand, and one per option — unioned by best rank. Document filtering happens *inside* the scan, so ranking occurs within the allowed set. |
| **Compress** | `adapter/compressor.py` | Extractive, **LLM-free** sentence selection under a budget, reserving coverage for each passage *and each option* so one option cannot starve the others of evidence. |
| **Answer** | `adapter/qa_agent.py` | Structured judgement (per-option SUPPORTED/REFUTED/INSUFFICIENT), optional local `Decimal` re-verification of arithmetic, canonicality retries. |
| **Account** | `adapter/qwen_client.py` | OpenAI-compatible client with exact per-call token accounting and a persistent budget ledger. |
| **Submit** | `submit.py` | Batch runner: contract-validated CSV output, atomic per-question checkpoints, resumable. |

The dense/hybrid path (e5 embeddings + RRF fusion in `memory/knowledge_store.py`
and `adapter/hybrid.py`) is kept for research comparison but is **not** part of
the default answer path.

## Honest status — what is and isn't validated

This matters more than a feature list:

**Validated**
- 179 tests pass on Windows and POSIX (contracts, retrieval, compression, answer
  parsing, ledger concurrency, submission checkpointing).
- Token cost measured end-to-end: **~1.3 model calls and ~2.8k tokens per
  question** (`benchmarks/bench_token_cost.py`).
- Lexical retrieval verified to recover exact tokens (ratings, rates, figures)
  and to honour document filtering.
- Ledger verified crash-safe and cross-process locked.

**NOT validated**
- **No real-world accuracy number exists.** The pipeline has never been scored
  against a labelled dataset. Any accuracy claim would be fabrication.
- The retrieval-vs-truncation trade-off is **unmeasured**. Published results on
  similar tasks report naive truncation *beating* chunk retrieval; this adapter
  bets the other way and that bet is untested here.
- No code-execution path for arithmetic questions; local verification only
  re-checks a model-supplied ledger.

`benchmarks/bench_adapter.py` measures retrieval recall, answer-sentence
retention and compression ratio on synthetic corpora — useful for regression,
not a substitute for a labelled evaluation.

## Install

```bash
pip install -e .        # or: pip install -r requirements.txt
```

Python 3.9+. The default lexical path needs no GPU and no model download; only
the optional research path pulls in `torch`/`transformers`.

## Quick start

```python
from awareliquid import MemoryQAAgent, RetrievalConfig

agent = MemoryQAAgent(config=RetrievalConfig(retrieval_backend="lexical"))
agent.ingest_document("report-1", long_text)

res = agent.answer_question(
    qid="q1",
    question="公司 2023 年营业收入是多少？",
    options=["10.2 亿元", "12.4 亿元", "15.1 亿元", "20.0 亿元"],
    qtype="mcq",                 # "mcq" | "tf" | "multi"
    doc_ids=["report-1"],
)
print(res.answer, res.usage.total_tokens)
```

## Batch runs

```bash
python submit.py --questions questions.jsonl --docs docs.json --out answers.csv
```

Questions are JSONL/JSON (`qid`, `question`, `options` as a `{letter: text}` dict
or list, `answer_format`, `split`, `doc_ids`); documents are a `{doc_id: text}`
JSON or a directory of `<doc_id>.txt`. Output is a five-column CSV
(`qid,answer,prompt_tokens,completion_tokens,total_tokens`) with a leading
`summary` totals row. Runs checkpoint after every question and resume safely;
`--reset-ledger` zeroes recorded spend.

## Configuration

Nothing is auto-loaded — export these yourself. The client **fails closed**: a
missing key raises rather than silently falling back to a mock.

| Variable | Default | Meaning |
|----------|---------|---------|
| `AWARELIQUID_LLM_API_KEY` | — | API key (falls back to `DASHSCOPE_API_KEY`). Required. |
| `AWARELIQUID_LLM_BASE_URL` | DashScope compatible-mode | OpenAI-compatible base URL. |
| `AWARELIQUID_LLM_MODEL` | `qwen-plus` | Model id (allowlisted in formal mode). |
| `AWARELIQUID_TEST_MODE` | — | `1` (with backend `mock`) to allow the offline mock. |
| `AWARELIQUID_ALLOW_FORMAL_NETWORK` | `0` | Formal runs deny egress unless exactly `1`. |

See [`.env.example`](.env.example) for the full set. Retrieval and compression
are tunable via `RetrievalConfig`.

## Tests

```bash
pytest
```

Runs fully offline.

## License

MIT — see [LICENSE](LICENSE).
