"""
knowledge_store.py — persistent, content-addressable long-term memory.

A large, slow store that is queried *by similarity* and pulled in on demand —
the complement to a model's fixed weights. The adapter keeps document passages
here and retrieves only the few a question needs, so the base model never has to
hold the whole document in context.

Design goals
------------
* **Zero coupling.** Imports only ``torch`` + the stdlib ``sqlite3``. It never
  imports a language model and no model imports it; it attaches to a pipeline
  through a tiny, explicit interface.
* **Local-first.** Everything lives in a single SQLite file on the device. The
  same file can be copied/synced across machines, so a big knowledge base is
  shared without growing any model's parameter count.
* **Bounded footprint.** An optional ``max_entries`` cap evicts the
  least-recently-used record so the store stays small and predictable.

Interface
---------
>>> from awareliquid.memory import PersistentKnowledgeMemory
>>> kb = PersistentKnowledgeMemory(key_dim=64, db_path=":memory:")
>>> kb.write(key_vec, "Paris is the capital of France")
>>> hits = kb.query(query_vec, top_k=3)      # [(content, score, meta), ...]
>>> best = kb.recall(query_vec)              # content of the top hit, or None

Keys are L2-normalised on write and query, so ``query`` ranks by cosine
similarity. Contents are arbitrary picklable Python objects (strings, dicts,
small tensors) stored via ``torch.save``.
"""

from __future__ import annotations

import io
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_vec     BLOB    NOT NULL,    -- float32 L2-normalised key, length = key_dim
    content     BLOB    NOT NULL,    -- torch.save(content) bytes
    meta        BLOB,                -- torch.save(meta) bytes or NULL
    doc_id      TEXT,                -- meta["doc_id"] hoisted out for fast filtering
    created_at  TEXT    NOT NULL,
    accessed_at TEXT    NOT NULL,    -- ISO time of last access (informational)
    access_seq  INTEGER NOT NULL     -- monotonic recency counter; the LRU sort key
);
CREATE INDEX IF NOT EXISTS idx_knowledge_doc_id ON knowledge(doc_id);
"""

_PRAGMA = "PRAGMA journal_mode=WAL;"


# ---------------------------------------------------------------------------
# (de)serialisation helpers
# ---------------------------------------------------------------------------

def _obj_to_bytes(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _bytes_to_obj(data: bytes) -> Any:
    buf = io.BytesIO(data)
    # weights_only=False: contents may be arbitrary python objects (str/dict),
    # not just tensors. The DB is local and written only by this process.
    return torch.load(buf, weights_only=False, map_location="cpu")


def _key_to_bytes(vec: torch.Tensor) -> bytes:
    return vec.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()


def _bytes_to_key(data: bytes, key_dim: int) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.float32).view(key_dim)


def _normalise(vec: torch.Tensor) -> torch.Tensor:
    """L2-normalise a 1-D key so dot products are cosine similarities."""
    return F.normalize(vec.detach().to(torch.float32).flatten(), dim=0, eps=1e-8)


# ---------------------------------------------------------------------------
# PersistentKnowledgeMemory
# ---------------------------------------------------------------------------

class PersistentKnowledgeMemory:
    """SQLite-backed, content-addressable long-term knowledge store.

    Parameters
    ----------
    key_dim:
        Dimensionality of the key (embedding) vectors. All writes/queries must
        match this — a mismatch raises ``ValueError`` rather than silently
        corrupting the index.
    db_path:
        Path to the SQLite file. ``":memory:"`` gives an ephemeral in-process
        store (handy for tests). A real path persists across processes/restarts
        and can be synced across devices.
    max_entries:
        Optional cap on the number of stored records. When exceeded, the
        least-recently-accessed record is evicted (LRU). ``None`` = unbounded.
    """

    def __init__(
        self,
        key_dim: int,
        db_path: Union[str, Path] = ".mt_lnn_knowledge.db",
        max_entries: Optional[int] = None,
    ):
        if key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {key_dim}")
        if max_entries is not None and max_entries <= 0:
            raise ValueError(f"max_entries must be positive or None, got {max_entries}")

        self.key_dim = int(key_dim)
        self.db_path = str(db_path)
        self.max_entries = max_entries

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(_PRAGMA)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # Optional lexical channel: an FTS5 full-text table mirroring the text
        # content, for BM25 keyword retrieval alongside the dense (cosine) channel
        # (hybrid retrieval). FTS5 is compiled into most SQLite builds but not all,
        # so degrade gracefully to dense-only when it is unavailable. The trigram
        # tokenizer gives substring matching that works for space-less CJK too.
        self._fts_enabled = False
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts "
                "USING fts5(content, doc_id UNINDEXED, tokenize='trigram')"
            )
            self._conn.commit()
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False

        # Monotonic recency counter used as the LRU sort key. Resolution-
        # independent (a wall-clock timestamp can collide when many ops land in
        # the same microsecond, which silently breaks LRU ordering). Seed it from
        # the persisted max so recency survives reopen.
        row = self._conn.execute("SELECT MAX(access_seq) FROM knowledge").fetchone()
        self._seq = int(row[0]) if row and row[0] is not None else 0
        # One shared sqlite3.Connection (check_same_thread=False) + the _seq
        # counter are touched by every request; FastAPI runs the sync serve
        # endpoints on an anyio threadpool (concurrent). Serialize all access
        # behind an RLock (reentrant: write()->_enforce_capacity(),
        # query(touch=True) both re-enter). Prevents interleaved sqlite ops and
        # the non-atomic `self._seq += 1` read-modify-write.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        key: torch.Tensor,
        content: Any,
        meta: Optional[Any] = None,
    ) -> int:
        """Store *content* indexed by the embedding *key*.

        Returns the row id of the new record. After insertion the LRU cap (if
        any) is enforced, so a write may evict the oldest-accessed record.
        """
        key = self._validate_key(key)
        now = datetime.now(timezone.utc).isoformat()
        # Hoist doc_id out of meta into its own indexed column so retrieval can
        # rank WITHIN an allowed document set instead of globally-then-filtering.
        doc_id = None
        if isinstance(meta, dict) and meta.get("doc_id") is not None:
            doc_id = str(meta["doc_id"])
        with self._lock:
            self._seq += 1
            cur = self._conn.execute(
                """
                INSERT INTO knowledge (key_vec, content, meta, doc_id, created_at, accessed_at, access_seq)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _key_to_bytes(key),
                    _obj_to_bytes(content),
                    _obj_to_bytes(meta) if meta is not None else None,
                    doc_id,
                    now,
                    now,
                    self._seq,
                ),
            )
            row_id = int(cur.lastrowid)
            # Mirror plain-text content into the lexical (FTS5) index so the same
            # record is retrievable by BM25. Only str content is indexable.
            if self._fts_enabled and isinstance(content, str):
                self._conn.execute(
                    "INSERT INTO knowledge_fts (rowid, content, doc_id) VALUES (?, ?, ?)",
                    (row_id, content, doc_id),
                )
            self._conn.commit()
            self._enforce_capacity()
            return row_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        key: torch.Tensor,
        top_k: int = 5,
        touch: bool = True,
        center: bool = False,
        return_ids: bool = False,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple]:
        """Return the *top_k* records most similar to *key* by cosine similarity.

        Each hit is ``(content, score, meta)`` with ``score`` in [-1, 1], sorted
        descending (or ``(id, content, score, meta)`` when ``return_ids=True``).
        Returns an empty list when the store is empty.

        Parameters
        ----------
        return_ids:
            When True, each hit is prefixed with the record's integer row id
            ``(id, content, score, meta)`` so a caller (e.g. a graph layer) can
            map a recalled record back to a stable node id. Default False
            preserves the 3-tuple ``(content, score, meta)`` contract.
        touch:
            When True (default), the matched records' ``accessed_at`` is bumped so
            the LRU eviction policy treats recalled knowledge as "fresh". Pass
            False for a read-only peek that does not influence eviction.
        center:
            Anisotropy-robust scoring. Keys produced by mean-pooling a language
            model's hidden states share a dominant direction, which inflates and
            compresses every cosine score into a narrow high band -- so plain
            cosine makes one record a near-universal nearest neighbour and barely
            discriminates queries. When True, subtract the corpus mean of the
            stored keys from both the stored keys and the query before scoring
            (the "all-but-the-top" post-processing of Mu & Viswanath 2018), then
            renormalise. This cancels the shared direction so scores spread out
            and rank by genuine semantic content. Default False preserves the
            plain-cosine behaviour. Recommended True for LM-encoded keys.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        q = self._validate_key(key)

        # Restrict the candidate set to the given documents BEFORE scoring, so
        # ranking (and centering) happen within the allowed set rather than
        # globally-then-filtered -- which would drop allowed chunks that rank
        # below the global top-k. An empty/None list means "all documents".
        with self._lock:
            if doc_ids:
                placeholders = ",".join("?" * len(doc_ids))
                rows = self._conn.execute(
                    f"SELECT id, key_vec, content, meta FROM knowledge "
                    f"WHERE doc_id IN ({placeholders})",
                    [str(d) for d in doc_ids],
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, key_vec, content, meta FROM knowledge"
                ).fetchall()
        if not rows:
            return []

        # Stack all keys into one matrix and score in a single mat-vec. The store
        # is meant for edge-scale knowledge bases (thousands, not billions), so a
        # brute-force cosine scan is the right, dependency-free choice.
        keys = torch.stack(
            [_bytes_to_key(r[1], self.key_dim) for r in rows], dim=0
        )                                                   # (N, key_dim), already unit-norm
        # Centering needs enough vectors for the mean to be a meaningful estimate
        # of the shared direction. At tiny N it is degenerate (at N=2 the two
        # residuals are antipodal, forcing scores to +c/-c regardless of content),
        # so fall back to plain cosine below the floor.
        if center and keys.shape[0] >= 8:
            # Remove the shared dominant direction (corpus mean) from keys AND
            # query, then renormalise so the scores are cosines on the residual
            # (semantic) component. Computed from the stored keys only, so the
            # query is projected against the same offset.
            mu = keys.mean(dim=0, keepdim=True)             # (1, key_dim)
            keys = F.normalize(keys - mu, dim=-1)
            q = F.normalize(q - mu.squeeze(0), dim=0)
        scores = keys @ q                                   # (N,) cosine sims

        k = min(top_k, scores.shape[0])
        top_scores, top_idx = torch.topk(scores, k)

        hits: List[Tuple[Any, float, Optional[Any]]] = []
        touched_ids: List[int] = []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            row = rows[idx]
            content = _bytes_to_obj(row[2])
            meta = _bytes_to_obj(row[3]) if row[3] is not None else None
            rid = int(row[0])
            if return_ids:
                hits.append((rid, content, float(score), meta))
            else:
                hits.append((content, float(score), meta))
            touched_ids.append(rid)

        if touch and touched_ids:
            now = datetime.now(timezone.utc).isoformat()
            # Bump the recency counter for each hit so they become the freshest
            # records for LRU purposes (counter, not wall-clock → collision-free).
            with self._lock:
                updates = []
                for rid in touched_ids:
                    self._seq += 1
                    updates.append((now, self._seq, rid))
                self._conn.executemany(
                    "UPDATE knowledge SET accessed_at = ?, access_seq = ? WHERE id = ?",
                    updates,
                )
                self._conn.commit()

        return hits

    def recall(
        self, key: torch.Tensor, touch: bool = True, center: bool = False
    ) -> Optional[Any]:
        """Convenience: the content of the single best match, or ``None`` if the
        store is empty. See :meth:`query` for the ``center`` flag."""
        hits = self.query(key, top_k=1, touch=touch, center=center)
        return hits[0][0] if hits else None

    # ------------------------------------------------------------------
    # Lexical (BM25) channel
    # ------------------------------------------------------------------

    def search_bm25(
        self,
        query_text: str,
        top_k: int = 10,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Tuple[int, str, Optional[Any]]]:
        """Return up to *top_k* records matching *query_text* by BM25, best first.

        Each hit is ``(id, content, meta)``. Returns an empty list when FTS5 is
        unavailable, the query has no usable terms, or nothing matches -- so a
        caller can always fall back to the dense channel. This is the lexical half
        of hybrid retrieval; fuse it with :meth:`query` (dense) via RRF.
        """
        if not self._fts_enabled or not query_text or not query_text.strip():
            return []
        match = self._fts_query(query_text)
        if not match:
            return []
        sql = "SELECT rowid, content FROM knowledge_fts WHERE knowledge_fts MATCH ?"
        params: List[Any] = [match]
        if doc_ids:
            placeholders = ",".join("?" * len(doc_ids))
            sql += f" AND doc_id IN ({placeholders})"
            params += [str(d) for d in doc_ids]
        # bm25() is ascending-better; ORDER BY it directly for best-first.
        sql += " ORDER BY bm25(knowledge_fts) LIMIT ?"
        params.append(int(top_k))
        with self._lock:
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []
            if not rows:
                return []
            ids = [int(r[0]) for r in rows]
            ph = ",".join("?" * len(ids))
            meta_rows = self._conn.execute(
                f"SELECT id, meta FROM knowledge WHERE id IN ({ph})", ids
            ).fetchall()
        meta_by_id = {
            int(r[0]): (_bytes_to_obj(r[1]) if r[1] is not None else None)
            for r in meta_rows
        }
        return [(int(r[0]), str(r[1]), meta_by_id.get(int(r[0]))) for r in rows]

    @property
    def fts_enabled(self) -> bool:
        """Whether the lexical (FTS5/BM25) channel is available in this build."""
        return self._fts_enabled

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0])

    def clear(self) -> None:
        """Delete all stored knowledge."""
        with self._lock:
            self._conn.execute("DELETE FROM knowledge")
            if self._fts_enabled:
                self._conn.execute("DELETE FROM knowledge_fts")
            self._conn.commit()

    def delete(self, ids) -> int:
        """Delete records by row id. Returns the count requested. Enables true
        per-record consolidation/UPSERT (previously a documented follow-on:
        only LRU eviction could remove records)."""
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        with self._lock:
            self._conn.executemany("DELETE FROM knowledge WHERE id = ?",
                                   [(i,) for i in ids])
            if self._fts_enabled:
                self._conn.executemany("DELETE FROM knowledge_fts WHERE rowid = ?",
                                       [(i,) for i in ids])
            self._conn.commit()
        return len(ids)

    def all_meta(self):
        """Return ``[(id, meta_dict_or_None), ...]`` for every record — for
        consolidation policies that RANK by a meta field (e.g. surprise)
        without needing a query key."""
        with self._lock:
            rows = self._conn.execute("SELECT id, meta FROM knowledge").fetchall()
        return [(int(r[0]), _bytes_to_obj(r[1]) if r[1] is not None else None)
                for r in rows]

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_key(self, key: torch.Tensor) -> torch.Tensor:
        vec = key.detach().to(torch.float32).flatten()
        if vec.numel() != self.key_dim:
            raise ValueError(
                f"key has {vec.numel()} elements, expected key_dim={self.key_dim}"
            )
        return _normalise(vec)

    @staticmethod
    def _fts_query(text: str) -> Optional[str]:
        """Build a safe FTS5 MATCH string from free text.

        Emits an OR of QUOTED tokens so punctuation in the raw text can never be
        mis-parsed as FTS5 query syntax. Under the ``trigram`` tokenizer a token
        must be >= 3 characters to form an index token, so:

        * Latin / numeric runs (incl. ``.``/``%``, e.g. ``3.45%``, ``124.5``,
          ``AAA``) are kept whole when >= 3 chars -- exact financial tokens.
        * CJK runs are expanded into overlapping 3-char slices, which the trigram
          tokenizer matches as substrings of the stored text.
        """
        tokens: List[str] = []
        for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%\-/]*", text):
            if len(w) >= 3:
                tokens.append(w.lower())
        for run in re.findall(r"[一-鿿]+", text):
            if len(run) >= 3:
                tokens.extend(run[i : i + 3] for i in range(len(run) - 2))
        if not tokens:
            return None
        seen: set = set()
        quoted: List[str] = []
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            quoted.append('"' + t.replace('"', '""') + '"')
            if len(quoted) >= 80:  # keep the MATCH string bounded
                break
        return " OR ".join(quoted)

    def _enforce_capacity(self) -> None:
        """Evict least-recently-accessed records until within ``max_entries``."""
        if self.max_entries is None:
            return
        n = len(self)
        if n <= self.max_entries:
            return
        overflow = n - self.max_entries
        # Lowest recency counter first → least-recently-used eviction.
        evicted = self._conn.execute(
            "SELECT id FROM knowledge ORDER BY access_seq ASC, id ASC LIMIT ?",
            (overflow,),
        ).fetchall()
        evicted_ids = [(int(r[0]),) for r in evicted]
        self._conn.executemany("DELETE FROM knowledge WHERE id = ?", evicted_ids)
        if self._fts_enabled and evicted_ids:
            self._conn.executemany("DELETE FROM knowledge_fts WHERE rowid = ?", evicted_ids)
        self._conn.commit()

    # Context-manager support
    def __enter__(self) -> "PersistentKnowledgeMemory":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"PersistentKnowledgeMemory(key_dim={self.key_dim}, "
            f"db={self.db_path!r}, entries={len(self)}, max_entries={self.max_entries})"
        )
