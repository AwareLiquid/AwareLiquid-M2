# Architecture

AwareLiquid is an **external memory-compression adapter** for a frozen Qwen
model. It answers questions over documents that are far longer than the model's
context window, and it is built for **token-metered** settings where the cost of
a run is dominated by how many generation tokens are spent.

## Design principles

1. **Non-invasive.** The base model's weights are never modified. Generation is
   reached only through the official Qwen API (`adapter/qwen_client.py`). The
   adapter is entirely *outside* the model.
2. **Local work is free (in generation tokens).** Chunking, embedding,
   retrieval, and compression all run on the host. They cost no API tokens, so
   the only tokens billed are the compact prompt and the short answer.
3. **Send the model only what it needs.** A long report is stored once and
   queried per question; the model sees a handful of compressed, relevant
   sentences instead of the whole document.
4. **Everything is measured.** Every API call returns exact token usage, so the
   total generation cost of a run is always known and can be optimised against.

## Data flow

```
ingest(doc_id, text)
        │
        ├─ chunk_document ........ sentence/paragraph-aware, overlapping passages
        ├─ SentenceEncoder ....... L2-normalised multilingual embedding per chunk
        └─ PersistentKnowledgeMemory.write ... {vector → chunk text, doc_id, idx}

answer_question(qid, question, options, qtype, doc_ids)
        │
        ├─ SentenceEncoder(query) ............ embed the question
        ├─ store.query (centered cosine) ..... top-k passages, filtered to doc_ids
        ├─ ExtractiveCompressor .............. keep only load-bearing sentences
        │                                      (question overlap + numeric salience),
        │                                      LLM-free → 0 generation tokens
        ├─ build compact prompt .............. system + question + options + context
        ├─ QwenChatClient.chat ............... Qwen API call, returns text + usage
        └─ parse_answer ...................... normalise to mcq / tf / multi form
                                               → AnswerResult(answer, usage)
```

## Components

| Module | Responsibility | Key properties |
|--------|----------------|----------------|
| `memory/knowledge_store.py` | SQLite vector store | cosine retrieval, anisotropy-robust centering, thread-safe, optional LRU cap, zero model coupling |
| `memory/encoder.py` | Sentence embedding | lazy load, multilingual `e5` default (Chinese + English), `bge` selectable, asymmetric query/passage prefixes |
| `adapter/chunker.py` | Document → passages | boundary-aware, decimal-safe splitting, configurable size + overlap |
| `adapter/compressor.py` | Context compression | extractive, LLM-free, char-budgeted, numeric/currency/date salience bonus |
| `adapter/qwen_client.py` | Generation + accounting | OpenAI-compatible, stdlib HTTP, exact token usage, offline mock fallback |
| `adapter/schemas.py` | Answer normalisation | robust mcq / tf / multi parsing, usage aggregation |
| `adapter/qa_agent.py` | Orchestration | wires the stages, restricts retrieval to a document set |

## Why compression matters

Retrieval alone still hands the model whole passages, most of whose sentences do
not bear on the question. The compressor is the token-saving stage: it ranks the
sentences within retrieved passages by lexical overlap with the question plus a
salience bonus for numbers, percentages, currency and dates — the tokens
financial answers hinge on — and packs the top ones into a character budget,
re-emitting them in document order. Because it is purely local and extractive, it
shrinks the prompt without spending a single generation token.

## Extending

* **Different embedder.** Pass a `SentenceEncoder(model_id=...)` (any `e5`/`bge`
  HF id) or inject a custom object exposing `.dim` and
  `.encode(text, is_query=...)`.
* **Different generation backend.** Any object with a
  `chat(messages, temperature, max_tokens) -> ChatResult` method can replace the
  Qwen client; token accounting flows through unchanged.
* **Persistent store.** Construct `PersistentKnowledgeMemory(db_path="…")` to
  keep the index on disk and reuse it across runs.
* **LLM-based compression.** The extractive compressor is the default; a model-
  based summariser can be swapped in where token budget allows.
