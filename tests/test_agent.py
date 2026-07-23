"""End-to-end agent loop with a fake encoder + mock chat client (offline)."""

import pytest

from awareliquid.adapter.qa_agent import (
    MemoryQAAgent,
    RetrievalConfig,
    _missing_evidence_anchors,
    _missing_evidence_slots,
    _verify_calculation_draft,
    _parse_structured_answer,
)
from awareliquid.adapter.qwen_client import (
    ChatResult,
    QwenChatClient,
    TokenUsage,
    UsageLedger,
    MockChatClient,
)
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
        config=RetrievalConfig(retrieval_backend="hybrid", max_chars=60, overlap_chars=10, top_k=3),
    )


@pytest.mark.parametrize("chat_client", [MockChatClient(), object()])
def test_formal_agent_rejects_mock_or_arbitrary_injected_client(chat_client):
    # Formal mode runs on the lexical backend, so no encoder is involved here;
    # this test is purely about which chat client is accepted.
    with pytest.raises(RuntimeError, match="requires a validated formal QwenChatClient"):
        MemoryQAAgent(
            chat_client=chat_client,
            config=RetrievalConfig(competition_mode=True),
        )


def test_formal_agent_requires_exact_configured_budget(tmp_path):
    ledger_path = str(tmp_path / "usage.json")
    client = QwenChatClient(
        api_key="test-key",
        competition_mode=True,
        usage_ledger=UsageLedger(max_tokens=5_000_000, path=ledger_path),
        formal_run_id=f"test-run-{tmp_path.name}",
        formal_ledger_path=ledger_path,
    )
    with pytest.raises(ValueError, match="token_budget=5000000"):
        MemoryQAAgent(
            chat_client=client,
            config=RetrievalConfig(
                competition_mode=True,
                token_budget=4_999_999,
                formal_run_id=f"test-run-{tmp_path.name}",
                formal_ledger_path=ledger_path,
            ),
        )


def test_formal_agent_accepts_only_validated_formal_qwen_client(tmp_path):
    ledger_path = str(tmp_path / "usage.json")
    client = QwenChatClient(
        api_key="test-key",
        competition_mode=True,
        usage_ledger=UsageLedger(max_tokens=5_000_000, path=ledger_path),
        formal_run_id=f"test-run-{tmp_path.name}",
        formal_ledger_path=ledger_path,
    )
    agent = MemoryQAAgent(
        chat_client=client,
        config=RetrievalConfig(
            competition_mode=True,
            retrieval_backend="lexical",
            formal_run_id=f"test-run-{tmp_path.name}",
            formal_ledger_path=ledger_path,
        ),
    )
    assert agent.chat_client is client


def test_formal_agent_rejects_nonformal_qwen_client():
    client = QwenChatClient(api_key="test-key")
    with pytest.raises(ValueError, match="formal mode requires competition_mode=True"):
        MemoryQAAgent(
            chat_client=client,
            config=RetrievalConfig(competition_mode=True, retrieval_backend="lexical"),
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


def test_empty_doc_id_filter_does_not_fall_back_to_all_documents():
    agent = _agent()
    agent.ingest_document("annual-2023", REPORT)
    assert agent.retrieve("营业收入是多少", doc_ids=[]) == []


def test_split_a_requires_doc_ids_and_split_b_resolves_candidates():
    agent = MemoryQAAgent(
        chat_client=MockChatClient(),
        config=RetrievalConfig(retrieval_backend="lexical"),
    )
    agent.ingest_document("annual", "2025 年营业收入为 200 亿元。")
    agent.ingest_document("noise", "本段只讨论天气和旅行。")

    with pytest.raises(ValueError, match="split A requires"):
        agent.retrieve("营业收入是多少", split="A")
    with pytest.raises(ValueError, match="split B must not"):
        agent.retrieve("营业收入是多少", doc_ids=["annual"], split="B")

    hits = agent.retrieve("营业收入是多少", split="B")
    assert hits and any("200 亿元" in hit for hit in hits)


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


def test_structured_judgement_parses_option_states_and_falls_back():
    assert _parse_structured_answer(
        "A=REFUTED\nB=SUPPORTED\nC=INSUFFICIENT\nANSWER:B",
        "multi",
        3,
    ) == "B"
    assert _parse_structured_answer(
        "A: left=12%; relation=<; right=50%; state=SUPPORTED\nANSWER:A",
        "mcq",
        2,
    ) == "A"
    assert _parse_structured_answer("the answer is C", "mcq", 4) == "C"


def test_structured_prompt_uses_relation_check_only_when_needed():
    relation = MemoryQAAgent._build_structured_prompt(
        "哪一项增速高于另一项？", ["12%", "8%"], "mcq", "context"
    )
    ordinary = MemoryQAAgent._build_structured_prompt(
        "关于保单贷款，哪些说法正确？", ["允许", "不允许"], "multi", "context"
    )
    assert "matching units and periods" in relation
    assert "matching units and periods" not in ordinary


def test_option_evidence_coverage_finds_missing_high_signal_anchors():
    missing = _missing_evidence_anchors(
        "2023 年营业收入是多少？",
        "营业收入为 124.5 亿元，适用第七条但有特殊情形",
        "[source doc=annual] 2023 年营业收入已披露。",
    )
    assert "124.5 亿元" in missing
    assert "第七条" in missing
    assert "特殊情形" in missing
    # A connective word is not useful enough to trigger a broad supplement.
    assert "但" not in missing


def test_option_evidence_coverage_finds_missing_domain_slots():
    missing = _missing_evidence_slots(
        "比较两家公司的营业收入和经营现金流。",
        "经营现金流更高。",
        "2023 年营业收入已披露，但没有现金流数据。",
    )
    assert any("经营现金流" in item for item in missing)


def test_evidence_coverage_supplement_is_enabled_by_default():
    assert RetrievalConfig().evidence_coverage_supplement is True


def test_calculation_draft_is_recomputed_with_decimal_and_invalid_values_are_dropped():
    valid = _verify_calculation_draft(
        '{"facts":[{"name":"premium","value":"10","unit":"万元"},'
        '{"name":"gain","value":"2","unit":"万元"}],'
        '"derived":[{"name":"surrender","operation":"add",'
        '"operands":["premium","gain"],"value":"12","unit":"万元"}]}'
    )
    assert "derived surrender = 12万元" in valid

    invalid = _verify_calculation_draft(
        '{"facts":[{"name":"premium","value":"10"},'
        '{"name":"gain","value":"2"}],'
        '"derived":[{"name":"surrender","operation":"add",'
        '"operands":["premium","gain"],"value":"11"}]}'
    )
    assert "derived surrender" not in invalid


def test_calculation_draft_accepts_percentages_literals_and_forward_references():
    verified = _verify_calculation_draft(
        '{"facts":[{"name":"gain","value":"2"},{"name":"rate","value":"75%"}],'
        '"derived":[{"name":"surrender","operation":"add",'
        '"operands":["premium_after_rate","1"],"value":"2.5"},'
        '{"name":"premium_after_rate","operation":"multiply",'
        '"operands":["gain","rate"],"value":"1.5"}]}'
    )
    assert "derived premium_after_rate = 1.5" in verified
    assert "derived surrender = 2.5" in verified


class _RecordingStructuredClient:
    def __init__(self):
        self.prompts = []

    def chat(self, messages, temperature, max_tokens):
        self.prompts.append(messages[-1]["content"])
        return ChatResult(
            "A=SUPPORTED\nB=REFUTED\nANSWER:A",
            TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )


def test_calculation_judgement_is_targeted_and_usage_is_counted():
    client = _RecordingStructuredClient()
    agent = MemoryQAAgent(
        encoder=FakeEncoder(dim=128),
        store=PersistentKnowledgeMemory(key_dim=128, db_path=":memory:"),
        chat_client=client,
        config=RetrievalConfig(retrieval_backend="hybrid", max_chars=60, overlap_chars=10, top_k=3),
    )
    agent.ingest_document("annual", "现金价值为 12 万元，手续费比例为 0%。")
    calculated = agent.answer_question(
        qid="calc", question="退保所得金额是多少？", options=["12万元", "10万元"],
        qtype="mcq", doc_ids=["annual"],
    )
    factual = agent.answer_question(
        qid="fact", question="产品名称是什么？", options=["甲", "乙"],
        qtype="mcq", doc_ids=["annual"],
    )
    assert len(client.prompts) == 3
    assert calculated.usage.total_tokens == 6
    assert factual.usage.total_tokens == 3


def test_b_answer_uses_same_answer_path_without_doc_ids():
    client = _RecordingStructuredClient()
    agent = MemoryQAAgent(
        chat_client=client,
        config=RetrievalConfig(retrieval_backend="lexical"),
    )
    agent.ingest_document("candidate", "2025 年营业收入为 200 亿元。")
    result = agent.answer_question(
        qid="b1",
        question="营业收入是多少？",
        options=["200 亿元", "100 亿元"],
        qtype="mcq",
        split="B",
    )
    assert result.answer == "A"
    assert len(client.prompts) == 1


def test_default_backend_is_lexical_and_needs_no_embedding_model():
    """The advertised default must match what every entry point actually uses.

    A "hybrid" default silently pulled in the dense/e5 path — and therefore
    torch, a model download, and a dependency that is unusable wherever
    embedding models are disallowed — for anyone who just called
    MemoryQAAgent().
    """
    assert RetrievalConfig().retrieval_backend == "lexical"
    agent = MemoryQAAgent(chat_client=MockChatClient())
    assert agent.encoder is None, "the default path must not construct an encoder"
    assert hasattr(agent.store, "search_bm25")
