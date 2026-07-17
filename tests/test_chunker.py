from awareliquid.adapter.chunker import chunk_document


def test_empty_document_yields_no_chunks():
    assert chunk_document("d", "") == []
    assert chunk_document("d", "   \n  ") == []


def test_chunks_respect_budget_and_index():
    text = "。".join(f"这是第{i}个句子，包含数字{i}" for i in range(40))
    chunks = chunk_document("doc-1", text, max_chars=120, overlap_chars=20)
    assert len(chunks) > 1
    for i, ch in enumerate(chunks):
        assert ch.doc_id == "doc-1"
        assert ch.chunk_idx == i
        assert len(ch.text) <= 120 + 20  # budget plus carried overlap


def test_long_single_sentence_is_hard_split():
    text = "x" * 500  # no sentence boundary at all
    chunks = chunk_document("d", text, max_chars=100, overlap_chars=0)
    assert len(chunks) == 5
    assert "".join(c.text for c in chunks) == text


def test_overlap_carries_context_forward():
    text = "。".join(f"句子{i}" for i in range(30))
    chunks = chunk_document("d", text, max_chars=60, overlap_chars=15)
    # Every chunk except the first should start from an overlap tail.
    assert len(chunks) >= 2


def test_invalid_params_raise():
    import pytest

    with pytest.raises(ValueError):
        chunk_document("d", "abc", max_chars=0)
    with pytest.raises(ValueError):
        chunk_document("d", "abc", max_chars=100, overlap_chars=100)
