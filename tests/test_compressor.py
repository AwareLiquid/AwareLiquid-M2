from awareliquid.adapter.compressor import ExtractiveCompressor


def test_keeps_salient_number_sentence():
    comp = ExtractiveCompressor(budget_chars=200)
    passages = [
        "公司是一家科技企业。2023 年营业收入为 124.5 亿元。团队很努力。"
    ]
    out = comp.compress("营业收入是多少", passages)
    assert "124.5" in out.text
    assert out.kept_sentences >= 1
    assert out.source_sentences == 3


def test_respects_budget():
    comp = ExtractiveCompressor(budget_chars=40)
    passages = ["。".join(f"这是一个较长的句子编号{i}且包含内容" for i in range(20))]
    out = comp.compress("句子", passages)
    # Should not blow far past the budget (allow one over-budget sentence).
    assert len(out.text) <= 40 + 60


def test_empty_passages():
    comp = ExtractiveCompressor()
    out = comp.compress("anything", [])
    assert out.text == ""
    assert out.kept_sentences == 0
    assert out.source_sentences == 0


def test_relevance_ordering_preserves_document_order():
    comp = ExtractiveCompressor(budget_chars=500)
    passages = ["净利润 18.2 亿元。研发投入 9.7 亿元。营业收入 124.5 亿元。"]
    out = comp.compress("净利润 研发 营业收入", passages)
    # Kept sentences are re-emitted in original order.
    pos_profit = out.text.find("18.2")
    pos_rev = out.text.find("124.5")
    assert pos_profit != -1 and pos_rev != -1
    assert pos_profit < pos_rev


def test_coverage_keeps_one_relevant_sentence_from_each_passage():
    comp = ExtractiveCompressor(budget_chars=200, ensure_passage_coverage=True)
    passages = [
        "甲公司营业收入为 100 亿元。甲公司的其他说明。",
        "乙公司营业收入为 200 亿元。乙公司的其他说明。",
    ]

    out = comp.compress("比较两家公司营业收入", passages)

    assert "100 亿元" in out.text
    assert "200 亿元" in out.text


def test_coverage_can_be_disabled_for_legacy_global_ranking():
    comp = ExtractiveCompressor(budget_chars=100, ensure_passage_coverage=False)
    out = comp.compress(
        "营业收入",
        ["甲公司营业收入为 100 亿元。", "乙公司营业收入为 200 亿元。"],
    )

    assert out.kept_sentences >= 1


def test_coverage_queries_preserve_independent_option_evidence():
    comp = ExtractiveCompressor(budget_chars=220)
    passages = [
        "收入同比增长 10%。研发费用占收入 5%。现金流同比下降。",
    ]

    out = comp.compress(
        "哪些说法正确",
        passages,
        coverage_queries=["收入增长", "研发费用占收入", "现金流下降"],
    )

    assert "增长 10%" in out.text
    assert "研发费用" in out.text
    assert "现金流" in out.text
