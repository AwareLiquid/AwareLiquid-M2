from awareliquid.adapter.qwen_client import MockChatClient
from awareliquid.adapter.qa_agent import MemoryQAAgent, RetrievalConfig
from awareliquid.memory.lexical_store import LexicalKnowledgeMemory


def _lexical_agent():
    return MemoryQAAgent(
        chat_client=MockChatClient(),
        config=RetrievalConfig(
            retrieval_backend="lexical",
            max_chars=120,
            overlap_chars=10,
            top_k=3,
        ),
        db_path=":memory:",
    )


def test_lexical_store_matches_two_character_chinese_terms_and_metadata():
    store = LexicalKnowledgeMemory(":memory:")
    store.write(
        "本产品利率为 3.25%，按季度调整。",
        meta={"doc_id": "rates", "section": "贷款利率", "page": 4},
    )
    store.write(
        "本年度保费收入增长 8.1%。",
        meta={"doc_id": "insurance", "section": "保费收入", "page": 8},
    )

    rate_hits = store.search_bm25("利率")
    assert rate_hits and rate_hits[0][1].startswith("本产品利率")
    assert store.search_bm25("利率", doc_ids=["insurance"]) == []
    assert store.search_bm25("保费", doc_ids=["insurance"])
    assert store.available_doc_ids() == ["insurance", "rates"]


def test_explicit_empty_doc_filter_never_means_all_documents():
    store = LexicalKnowledgeMemory(":memory:")
    store.write("营业收入为 124.5 亿元。", meta={"doc_id": "annual"})
    assert store.search_bm25("营业收入")
    assert store.search_bm25("营业收入", doc_ids=[]) == []


def test_agent_lexical_backend_does_not_construct_an_encoder():
    agent = _lexical_agent()
    assert agent.encoder is None
    agent.ingest_document("annual", "2023 年营业收入为 124.5 亿元，利率为 3.25%。")
    agent.ingest_document("other", "本年度保费收入增长 8.1%。")

    hits = agent.retrieve("利率是多少？", doc_ids=["annual"])
    assert hits and "3.25%" in hits[0]
    assert agent.retrieve("保费是多少？", doc_ids=["annual"]) == []
    assert agent.retrieve("营业收入是多少？", doc_ids=[]) == []


def test_lexical_agent_can_answer_offline_without_qwen_key():
    agent = _lexical_agent()
    agent.ingest_document("annual", "2023 年营业收入为 124.5 亿元。")
    result = agent.answer_question(
        qid="q1",
        question="2023 年营业收入是多少？",
        options=["114.9 亿元", "124.5 亿元"],
        doc_ids=["annual"],
    )
    assert result.answer in {"A", "B"}
    assert result.usage.total_tokens > 0


def test_lexical_agent_requires_explicit_mock_without_api_key(monkeypatch):
    monkeypatch.delenv("AWARELIQUID_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("AWARELIQUID_LLM_BACKEND", "mock")
    monkeypatch.setenv("AWARELIQUID_TEST_MODE", "1")
    agent = MemoryQAAgent(
        config=RetrievalConfig(retrieval_backend="lexical"),
    )
    assert isinstance(agent.chat_client, MockChatClient)
