"""Multi-query retrieval: anchor detection + per-operand/option union."""

from awareliquid.adapter.qa_agent import MemoryQAAgent, RetrievalConfig, _detect_anchors
from awareliquid.adapter.qwen_client import MockChatClient
from awareliquid.memory.knowledge_store import PersistentKnowledgeMemory

from conftest import FakeEncoder


# ---- anchor detection ---------------------------------------------------

def test_detect_anchors_needs_at_least_two():
    assert _detect_anchors("2023 年营业收入是多少？") == []          # one year -> no fan-out
    assert _detect_anchors("2021 年和 2023 年的营业收入对比") == ["2021", "2023"]


def test_detect_anchors_quarters():
    got = _detect_anchors("第一季度和第三季度的净利润哪个高？")
    assert len(got) >= 2


def test_detect_anchors_dedupes():
    assert _detect_anchors("2023 年，2023 年，还是 2020 年？") == ["2023", "2020"]


# ---- multi-query retrieval ----------------------------------------------

def _agent(multi_query: bool, top_k: int = 2) -> MemoryQAAgent:
    enc = FakeEncoder(dim=64)
    store = PersistentKnowledgeMemory(key_dim=enc.dim, db_path=":memory:")
    cfg = RetrievalConfig(max_chars=26, overlap_chars=0, top_k=top_k,
                          multi_query=multi_query, multi_query_cap=6)
    return MemoryQAAgent(encoder=enc, store=store, chat_client=MockChatClient(), config=cfg)


def _ingest_years(agent: MemoryQAAgent) -> None:
    rows = [f"{y} 年营业收入为 {y-1900}.5 亿元。" for y in range(2000, 2024)]
    agent.ingest_document("m", "\n".join(rows))


def test_multi_query_gathers_both_operands():
    agent = _agent(multi_query=True, top_k=2)
    _ingest_years(agent)
    hits = agent.retrieve("2003 年和 2019 年的营业收入分别是多少？",
                          doc_ids=["m"], options=["2003 更高", "2019 更高"])
    joined = " ".join(hits)
    # The anchor-boosted sub-queries should surface BOTH years' rows.
    assert "103.5" in joined and "119.5" in joined
    # Union may exceed the single-pass top_k (bounded by multi_query_cap).
    assert len(hits) >= 2


def test_multi_query_default_off_preserves_single_pass():
    agent = _agent(multi_query=False, top_k=2)
    _ingest_years(agent)
    hits = agent.retrieve("2019 年的营业收入是多少？", doc_ids=["m"], options=["A", "B"])
    assert len(hits) <= 2  # single pass respects top_k


def test_multi_query_falls_back_without_anchors_or_options():
    # No 2+ anchors and no options -> behaves like a single pass.
    agent = _agent(multi_query=True, top_k=2)
    _ingest_years(agent)
    hits = agent.retrieve("公司的营业收入怎么样？", doc_ids=["m"])
    assert isinstance(hits, list)
