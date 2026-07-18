"""Hybrid retrieval: BM25 (FTS5) channel + RRF fusion."""

import torch

from awareliquid.adapter.hybrid import rrf_fuse
from awareliquid.adapter.qa_agent import MemoryQAAgent, RetrievalConfig
from awareliquid.adapter.qwen_client import MockChatClient
from awareliquid.memory.knowledge_store import PersistentKnowledgeMemory

from conftest import FakeEncoder


def _store():
    enc = FakeEncoder(dim=32)
    return enc, PersistentKnowledgeMemory(key_dim=enc.dim, db_path=":memory:")


# ---- BM25 / FTS5 channel ------------------------------------------------

def test_fts_enabled_in_this_build():
    _, s = _store()
    assert s.fts_enabled, "FTS5 expected in the test SQLite build"


def test_bm25_retrieves_exact_financial_tokens():
    enc, s = _store()
    for txt in ["本期债券的信用评级为 AAA，评级机构为中诚信国际。",
                "本期债券发行规模 20 亿元。",
                "公司2023年营业收入为 124.5 亿元。"]:
        s.write(enc.encode(txt), txt, meta={"doc_id": "d"})
    # Exact tokens a dense embedder blurs but BM25 nails.
    assert "AAA" in s.search_bm25("信用评级 AAA")[0][1]
    assert "124.5" in s.search_bm25("124.5")[0][1]
    assert "发行规模" in s.search_bm25("发行规模")[0][1]


def test_bm25_respects_doc_id_filter():
    enc, s = _store()
    s.write(enc.encode("信用评级为 AAA"), "信用评级为 AAA", meta={"doc_id": "bond"})
    s.write(enc.encode("研发投入 9.7 亿元"), "研发投入 9.7 亿元", meta={"doc_id": "ann"})
    assert s.search_bm25("信用评级", doc_ids=["bond"])
    assert s.search_bm25("信用评级", doc_ids=["ann"]) == []


# ---- RRF fusion ---------------------------------------------------------

def test_rrf_fuse_combines_and_dedupes():
    dense = [(1, "a", None), (2, "b", None), (3, "c", None)]
    sparse = [(3, "c", None), (4, "d", None)]
    fused = rrf_fuse(dense, sparse, k=10, w_dense=0.7, w_sparse=0.3, top_k=4)
    ids = [rid for rid, _c, _m in fused]
    # id 3 appears in both channels -> boosted; no duplicate ids.
    assert len(ids) == len(set(ids))
    assert 3 in ids
    assert set(ids) <= {1, 2, 3, 4}


def test_rrf_agreement_beats_single_channel():
    # A doc both channels rank highly should outrank a doc only one channel saw.
    dense = [(1, "shared", None), (2, "dense-only", None)]
    sparse = [(1, "shared", None), (3, "sparse-only", None)]
    fused = rrf_fuse(dense, sparse, k=10, top_k=3)
    assert fused[0][0] == 1  # agreed-upon doc ranks first


# ---- end-to-end agent ---------------------------------------------------

def _agent(hybrid: bool) -> MemoryQAAgent:
    enc = FakeEncoder(dim=64)
    store = PersistentKnowledgeMemory(key_dim=enc.dim, db_path=":memory:")
    cfg = RetrievalConfig(max_chars=60, overlap_chars=10, top_k=3, hybrid=hybrid)
    return MemoryQAAgent(encoder=enc, store=store, chat_client=MockChatClient(), config=cfg)


def test_hybrid_and_dense_both_retrieve_target():
    for hybrid in (False, True):
        agent = _agent(hybrid)
        agent.ingest_document("bond", "本期债券的信用评级为 AAA。发行规模 20 亿元。票面利率 3.45%。")
        hits = agent.retrieve("信用评级是什么", doc_ids=["bond"], options=["AAA", "AA"])
        assert hits, f"hybrid={hybrid} returned nothing"
        assert any("AAA" in h for h in hits), f"hybrid={hybrid} missed the answer chunk"


def test_hybrid_falls_back_when_disabled():
    agent = _agent(hybrid=False)
    agent.ingest_document("d", "公司2023年营业收入为 124.5 亿元。")
    # Dense-only path must still work end-to-end.
    res = agent.answer_question(qid="q", question="营业收入是多少",
                                options=["124.5 亿元", "100 亿元"], qtype="mcq", doc_ids=["d"])
    assert res.answer in {"A", "B"}
