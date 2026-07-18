"""The Qwen adapter layer: turn long documents into token-efficient answers.

This subpackage sits *outside* the base model. It reaches generation only
through :class:`~awareliquid.adapter.qwen_client.QwenChatClient`, so the model's
weights are never touched; everything else -- chunking, retrieval, compression,
answer parsing and token accounting -- runs locally.
"""

from .qa_agent import MemoryQAAgent, RetrievalConfig
from .qwen_client import (
    ChatResult,
    MockChatClient,
    QwenChatClient,
    TokenUsage,
    build_chat_client,
)
from .chunker import Chunk, chunk_document
from .compressor import CompressedContext, ExtractiveCompressor
from .hybrid import rrf_fuse
from .schemas import AnswerResult, parse_answer, summarize_usage

__all__ = [
    "MemoryQAAgent",
    "RetrievalConfig",
    "rrf_fuse",
    "QwenChatClient",
    "MockChatClient",
    "ChatResult",
    "TokenUsage",
    "build_chat_client",
    "Chunk",
    "chunk_document",
    "ExtractiveCompressor",
    "CompressedContext",
    "AnswerResult",
    "parse_answer",
    "summarize_usage",
]
