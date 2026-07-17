"""AwareLiquid — a memory-compression adapter for Qwen.

AwareLiquid wraps a frozen Qwen model with an external, long-term memory layer
so it can answer questions over documents far longer than its context window,
while spending as few generation tokens as possible.

The design is deliberately non-invasive: the base model's weights are never
modified and generation is reached only through the official Qwen API. The
adapter's work -- chunking, embedding-based retrieval and dynamic context
compression -- all happens locally, around the model rather than inside it.

Quick start
-----------
>>> from awareliquid import MemoryQAAgent
>>> agent = MemoryQAAgent()
>>> agent.ingest_document("report-1", long_text)
>>> res = agent.answer_question(
...     qid="q1",
...     question="What was 2023 revenue?",
...     options=["1.0bn", "1.2bn", "1.5bn", "2.0bn"],
...     qtype="mcq",
...     doc_ids=["report-1"],
... )
>>> res.answer, res.usage.total_tokens
"""

from .adapter import (
    AnswerResult,
    MemoryQAAgent,
    RetrievalConfig,
    TokenUsage,
    build_chat_client,
    summarize_usage,
)
from .memory import (
    PersistentKnowledgeMemory,
    SentenceEncoder,
)

__version__ = "0.1.0"

__all__ = [
    "MemoryQAAgent",
    "RetrievalConfig",
    "AnswerResult",
    "TokenUsage",
    "summarize_usage",
    "build_chat_client",
    "PersistentKnowledgeMemory",
    "SentenceEncoder",
    "__version__",
]
