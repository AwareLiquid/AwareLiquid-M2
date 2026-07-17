"""Long-term, content-addressable memory for the AwareLiquid adapter.

This subpackage is a self-contained, dependency-light memory tier:

* :class:`~awareliquid.memory.knowledge_store.PersistentKnowledgeMemory`
    an SQLite-backed vector store (cosine retrieval, LRU cap, thread-safe).
* :class:`~awareliquid.memory.encoder.SentenceEncoder`
    a lazy, L2-normalised multilingual sentence embedder (``e5``/``bge``).

Neither module imports a base language model, so they attach to any generation
backend through the small interfaces they expose.
"""

from .knowledge_store import PersistentKnowledgeMemory
from .encoder import SentenceEncoder

__all__ = [
    "PersistentKnowledgeMemory",
    "SentenceEncoder",
]
