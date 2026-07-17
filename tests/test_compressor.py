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
