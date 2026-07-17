"""End-to-end agent loop with a fake encoder + mock chat client (offline)."""

from awareliquid.adapter.qa_agent import MemoryQAAgent, RetrievalConfig
from awareliquid.adapter.qwen_client import MockChatClient
from awareliquid.memory.knowledge_store import PersistentKnowledgeMemory

from conftest import FakeEncoder

REPORT = (
    "本公司 2023 年度实现营业收入 124.5 亿元，同比增长 8.3%。"
    "归属于母公司股东的净利润为 18.2 亿元。"
    "研发投入 9.7 亿元，占营业收入的 7.8%。"
    "2022 年营业收入为 114.9 亿元。"
)


def _agent() -> MemoryQAAgent:
    enc = FakeEncoder(dim=128)
    store = PersistentKnowledgeMemory(key_dim=enc.dim, db_path=":memory:")
    return MemoryQAAgent(
        encoder=enc,
        store=store,
        chat_client=MockChatClient(),
        config=RetrievalConfig(max_chars=60, overlap_chars=10, top_k=3),
    )


def test_ingest_returns_chunk_count():
    agent = _agent()
    n = agent.ingest_document("annual-2023", REPORT)
    assert n >= 1
    assert len(agent.store) == n


def test_retrieve_is_restricted_to_doc_ids():
    agent = _agent()
    agent.ingest_document("annual-2023", REPORT)
    agent.ingest_document("other", "完全无关的内容，讲的是天气和旅行。")
    hits = agent.retrieve("营业收入是多少", doc_ids=["annual-2023"])
    assert hits, "expected at least one retrieved passage"
    # Nothing from the excluded document should surface.
    assert all("天气" not in h for h in hits)


def test_doc_id_filter_ranks_within_allowed_set():
    # Fill the store with many chunks from a distractor doc, so the target doc's
    # chunk would rank below a global top-k. In-store filtering must still find it.
    agent = _agent()
    for i in range(30):
        agent.ingest_document(f"noise-{i}", "无关内容，讨论天气、旅行与美食的段落。")
    agent.ingest_document("target", "本公司 2023 年营业收入为 124.5 亿元。")
    hits = agent.retrieve("营业收入是多少", doc_ids=["target"])
    assert hits, "target doc chunk must survive filtering even amid many distractors"
    assert any("124.5" in h for h in hits)


def test_answer_question_shape_and_usage():
    agent = _agent()
    agent.ingest_document("annual-2023", REPORT)
    res = agent.answer_question(
        qid="q1",
        question="公司 2023 年营业收入是多少？",
        options=["114.9 亿元", "124.5 亿元", "18.2 亿元", "9.7 亿元"],
        qtype="mcq",
        doc_ids=["annual-2023"],
    )
    assert res.qid == "q1"
    assert res.answer in {"A", "B", "C", "D"}
    assert res.usage.total_tokens > 0


def test_answer_without_ingest_still_returns_valid_row():
    agent = _agent()
    res = agent.answer_question(
        qid="q9",
        question="无上下文的问题",
        options=["对", "错"],
        qtype="tf",
    )
    # No context retrieved, but the pipeline must still produce a valid answer.
    assert res.answer in {"A", "B"}
    assert res.usage.total_tokens > 0
