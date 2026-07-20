from awareliquid.adapter.evidence_index import build_evidence_nodes
from awareliquid.adapter.qa_agent import _merge_evidence_blocks
from awareliquid.memory.lexical_store import LexicalKnowledgeMemory


def test_evidence_nodes_keep_structure_and_source_anchors():
    text = "第七条 现金价值\n2023 年金额为 124.5 亿元。\f第三章\n免责等待期。"
    nodes = build_evidence_nodes("policy", text, max_chars=80, overlap_chars=0)

    assert len(nodes) == 2
    assert nodes[0].page == 1
    assert nodes[0].clause_id == "第七条"
    assert "2023 年" in nodes[0].anchors
    assert "124.5 亿元" in nodes[0].anchors
    assert nodes[0].neighbor_chunk_idxs == (1,)
    assert nodes[1].page == 2


def test_decimal_values_are_not_mistaken_for_clause_ids():
    nodes = build_evidence_nodes(
        "report",
        "研发投入占比为 8.44%。\n2.2.2 免赔额\f第七条 保险责任。",
    )
    clause_ids = {node.clause_id for node in nodes}
    assert "8.44" not in clause_ids
    assert "2.2.2" in clause_ids
    assert "第七条" in clause_ids


def test_table_nodes_keep_row_and_column_fields():
    nodes = build_evidence_nodes(
        "report",
        "| 指标 | 2023 | 2024 |\n| 营业收入 | 100 | 120 |",
    )

    assert nodes[0].table_id == "report:page-1"
    assert nodes[0].row_id == "指标 营业收入"
    assert nodes[0].column_ids == ("2023", "2024")


def test_lexical_store_expands_only_source_neighbors():
    store = LexicalKnowledgeMemory(":memory:")
    first = {
        "doc_id": "policy",
        "chunk_idx": 0,
        "neighbor_chunk_idxs": [1],
    }
    second = {"doc_id": "policy", "chunk_idx": 1, "neighbor_chunk_idxs": [0]}
    other = {"doc_id": "other", "chunk_idx": 1, "neighbor_chunk_idxs": []}
    first_id = store.write("核心条款", meta=first)
    store.write("相邻例外条件", meta=second)
    store.write("无关文档", meta=other)

    hits = [(first_id, "核心条款", first)]
    expanded = store.expand_neighbors(hits, max_extra=2)
    contents = [content for _row_id, content, _meta in expanded]

    assert contents == ["核心条款", "相邻例外条件"]


def test_evidence_merge_deduplicates_and_respects_budget():
    merged = _merge_evidence_blocks(
        ["[source doc_id=d]\n同一证据", "[source doc_id=d]\n同一证据", "新增证据"],
        max_chars=30,
    )

    assert merged.count("同一证据") == 1
    assert len(merged) <= 30
