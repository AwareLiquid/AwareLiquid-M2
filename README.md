# AwareLiquid

**A memory-compression adapter for Qwen.** AwareLiquid wraps a frozen Qwen model
with an external long-term memory layer so it can answer questions over documents
far longer than its context window — while spending as few generation tokens as
possible.

The adapter is **non-invasive**: it never modifies the base model's weights and
reaches generation only through the official Qwen API. Everything that makes it
work — chunking, embedding-based retrieval, and dynamic context compression —
runs locally, *around* the model rather than inside it.

```
          ┌─────────────────────── AwareLiquid adapter (local) ──────────────────────┐
document → │  chunker → sentence encoder → vector store → retrieve → compressor      │ → compact prompt ─┐
          └──────────────────────────────────────────────────────────────────────────┘                  │
question ─────────────────────────────────────────────────────────────────────────────────────────────► │
                                                                                                          ▼
                                                                                            frozen Qwen model (API)
                                                                                                          │
                                                                                       answer + token usage ◄┘
```

## Why

A long report rarely fits in a context window, and even when it does, feeding the
whole thing to the model for every question is wasteful — most of the document is
irrelevant to any one question, and tokens cost money and latency. AwareLiquid
keeps the document in a local vector store, pulls in only the passages a question
needs, compresses those to their load-bearing sentences, and sends the model a
prompt that is a small fraction of the source. The base model does what it is good
at — reading a short, relevant context and answering — and nothing else.

## How it works

| Stage | Module | What it does |
|-------|--------|--------------|
| **Chunk** | `adapter/chunker.py` | Split documents on sentence/paragraph boundaries into overlapping passages (language-agnostic, handles Chinese). |
| **Embed** | `memory/encoder.py` | Lazy, L2-normalised multilingual sentence embedder (`intfloat/multilingual-e5-small` by default; `bge` selectable). |
| **Store** | `memory/knowledge_store.py` | SQLite-backed, content-addressable vector store with cosine retrieval, an anisotropy-robust centering option, and an optional LRU cap. |
| **Retrieve** | `adapter/qa_agent.py` | Top-k passage retrieval, restricted to a caller-supplied document set. |
| **Compress** | `adapter/compressor.py` | Extractive, **LLM-free** sentence selection under a character budget — keeps sentences by question overlap plus a salience bonus for numbers, %, currency and dates. Compression itself costs **zero** generation tokens. |
| **Answer** | `adapter/qwen_client.py` | OpenAI-compatible Qwen chat call with exact per-call token accounting. |

Retrieval and compression run entirely locally, so the only generation tokens
spent are the compact prompt and the short answer.

## Install

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Quick start

```python
from awareliquid import MemoryQAAgent

agent = MemoryQAAgent()                     # local store + local embedder
agent.ingest_document("report-1", long_text)

res = agent.answer_question(
    qid="q1",
    question="公司 2023 年营业收入是多少？",
    options=["10.2 亿元", "12.4 亿元", "15.1 亿元", "20.0 亿元"],
    qtype="mcq",                            # "mcq" | "tf" | "multi"
    doc_ids=["report-1"],                   # restrict retrieval to these docs
)
print(res.answer, res.usage.total_tokens)   # e.g. "B" 812
```

Run the bundled demo (works offline with the mock backend):

```bash
python examples/run_qa.py
```

## Configuration

Generation is configured entirely through environment variables — nothing is
hard-coded, and the client falls back to a deterministic offline mock when no key
is present:

| Variable | Default | Meaning |
|----------|---------|---------|
| `AWARELIQUID_LLM_BACKEND` | `qwen` | `qwen` or `mock` (offline). |
| `AWARELIQUID_LLM_API_KEY` | — | API key (falls back to `DASHSCOPE_API_KEY`). |
| `AWARELIQUID_LLM_BASE_URL` | DashScope compatible-mode | OpenAI-compatible base URL. |
| `AWARELIQUID_LLM_MODEL` | `qwen-plus` | Model id. |

```bash
export AWARELIQUID_LLM_API_KEY="sk-..."
export AWARELIQUID_LLM_MODEL="qwen-plus"
python examples/run_qa.py
```

Retrieval and compression are tunable via `RetrievalConfig` (chunk size, overlap,
`top_k`, compression budget, answer-token cap).

## Token accounting

Every chat call reports exact `prompt_tokens` / `completion_tokens` /
`total_tokens`. `AnswerResult.usage` carries per-question usage and
`summarize_usage(results)` aggregates a batch, so total generation cost across a
run is always known.

## Tests

```bash
pytest
```

The suite runs offline (mock backend, in-memory store) and covers chunking,
compression, answer parsing and the end-to-end agent loop.

## License

MIT — see [LICENSE](LICENSE).
