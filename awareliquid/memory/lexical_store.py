"""SQLite/FTS5 lexical store for the competition-safe retrieval path.

This module deliberately has no torch, transformers, or embedding dependency.
It is the store used by ``MemoryQAAgent`` when ``retrieval_backend="lexical"``
is selected. The older vector store remains available for research and
backward-compatible tests, but the formal competition path must use this one.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    meta        TEXT,
    doc_id      TEXT,
    search_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_doc_id ON records(doc_id);
"""


def _json_dump(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _json_load(value: Optional[str]) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _cjk_ngrams(run: str) -> List[str]:
    terms: List[str] = []
    if len(run) >= 2:
        terms.extend(run[i : i + 2] for i in range(len(run) - 1))
    if len(run) >= 3:
        terms.extend(run[i : i + 3] for i in range(len(run) - 2))
    return terms


def _search_text(content: str, meta: Optional[Any]) -> str:
    """Expand CJK bigrams and metadata into FTS5-searchable tokens."""
    parts = [content or ""]
    if isinstance(meta, dict):
        for key in (
            "title",
            "section",
            "page",
            "table_id",
            "row_id",
            "column_ids",
            "keywords",
            "unit",
        ):
            value = meta.get(key)
            if value is not None:
                parts.append(str(value))
    raw = " ".join(parts)
    expanded: List[str] = [raw]
    for run in re.findall(r"[一-鿿]+", raw):
        expanded.extend(_cjk_ngrams(run))
    return " ".join(expanded)


def _query_terms(text: str) -> List[str]:
    terms: List[str] = []
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%_/-]*", text or ""):
        if len(word) >= 2:
            terms.append(word.lower())
    for run in re.findall(r"[一-鿿]+", text or ""):
        terms.extend(_cjk_ngrams(run))
    seen = set()
    unique: List[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)
    return unique[:96]


class LexicalKnowledgeMemory:
    """Small local store with BM25-style FTS5 retrieval and strict doc filters."""

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.RLock()
        self._fts_enabled = False
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS records_fts "
                "USING fts5(search_text, doc_id UNINDEXED, tokenize='unicode61')"
            )
            self._conn.commit()
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False

        # Rebuild the FTS mirror if a database was copied without the virtual
        # table rows. This is intentionally deterministic and local-only.
        if self._fts_enabled:
            fts_count = int(self._conn.execute("SELECT COUNT(*) FROM records_fts").fetchone()[0])
            record_count = len(self)
            if fts_count != record_count:
                self._conn.execute("DELETE FROM records_fts")
                rows = self._conn.execute("SELECT id, search_text, doc_id FROM records").fetchall()
                self._conn.executemany(
                    "INSERT INTO records_fts(rowid, search_text, doc_id) VALUES (?, ?, ?)",
                    rows,
                )
                self._conn.commit()

    def write(self, content: str, meta: Optional[Dict[str, Any]] = None) -> int:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        doc_id = None
        if isinstance(meta, dict) and meta.get("doc_id") is not None:
            doc_id = str(meta["doc_id"])
        metadata = _json_dump(meta)
        searchable = _search_text(content, meta)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO records(content, meta, doc_id, search_text) VALUES (?, ?, ?, ?)",
                (content, metadata, doc_id, searchable),
            )
            row_id = int(cur.lastrowid)
            if self._fts_enabled:
                self._conn.execute(
                    "INSERT INTO records_fts(rowid, search_text, doc_id) VALUES (?, ?, ?)",
                    (row_id, searchable, doc_id),
                )
            self._conn.commit()
        return row_id

    def search_bm25(
        self,
        query_text: str,
        top_k: int = 10,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Tuple[int, str, Optional[Dict[str, Any]]]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if doc_ids is not None and not doc_ids:
            return []
        if not self._fts_enabled:
            return []
        terms = _query_terms(query_text)
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        sql = "SELECT rowid FROM records_fts WHERE records_fts MATCH ?"
        params: List[Any] = [match]
        if doc_ids is not None:
            placeholders = ",".join("?" * len(doc_ids))
            sql += f" AND doc_id IN ({placeholders})"
            params.extend(str(doc_id) for doc_id in doc_ids)
        sql += " ORDER BY bm25(records_fts) LIMIT ?"
        params.append(int(top_k))
        with self._lock:
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []
            results: List[Tuple[int, str, Optional[Dict[str, Any]]]] = []
            for (row_id,) in rows:
                row = self._conn.execute(
                    "SELECT content, meta FROM records WHERE id = ?", (int(row_id),)
                ).fetchone()
                if row is None:
                    continue
                results.append((int(row_id), str(row[0]), _json_load(row[1])))
            return results

    def available_doc_ids(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT doc_id FROM records WHERE doc_id IS NOT NULL ORDER BY doc_id"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def has_doc_ids(self, doc_ids: Sequence[str]) -> bool:
        wanted = {str(doc_id) for doc_id in doc_ids}
        return wanted.issubset(set(self.available_doc_ids()))

    @property
    def fts_enabled(self) -> bool:
        return self._fts_enabled

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM records")
            if self._fts_enabled:
                self._conn.execute("DELETE FROM records_fts")
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "LexicalKnowledgeMemory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
