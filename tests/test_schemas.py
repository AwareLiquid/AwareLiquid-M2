import pytest

from awareliquid.adapter.schemas import AnswerResult, parse_answer, summarize_usage
from awareliquid.adapter.qwen_client import TokenUsage


def test_mcq_takes_first_valid_letter():
    assert parse_answer("The answer is B.", "mcq", num_options=4) == "B"
    assert parse_answer("我认为应选 C", "mcq", num_options=4) == "C"


def test_mcq_ignores_out_of_range_letters():
    # Only A/B are valid options; "GDP" must not leak a G.
    assert parse_answer("Because of GDP growth, choose B", "mcq", num_options=2) == "B"


def test_tf_single_letter():
    assert parse_answer("A) 正确", "tf", num_options=2) == "A"


def test_multi_sorts_and_dedupes():
    assert parse_answer("正确选项是 C、A、A、D", "multi", num_options=4) == "ACD"


def test_reasoning_with_acronyms_does_not_corrupt_answer():
    # Financial acronyms in reasoning must not leak option letters.
    assert parse_answer("Comparing EBITDA and ROE, the answer is C", "mcq", 4) == "C"
    assert parse_answer("Given the USD exposure, Answer: A", "mcq", 4) == "A"
    # multi: only letters after the marker count, not the acronym letters.
    assert parse_answer("EBITDA rose while ROA fell. 答案 AC", "multi", 4) == "AC"


def test_last_line_is_used_when_no_marker():
    assert parse_answer("Some reasoning about DCF models.\nB", "mcq", 4) == "B"


def test_empty_reply_returns_empty_string():
    assert parse_answer("no letters here", "multi", num_options=4) == ""
    assert parse_answer("", "mcq", num_options=4) == ""


def test_unknown_qtype_raises():
    with pytest.raises(ValueError):
        parse_answer("A", "essay")


def test_summarize_usage_aggregates():
    rs = [
        AnswerResult("q1", "A", "mcq", TokenUsage(10, 2, 12)),
        AnswerResult("q2", "B", "mcq", TokenUsage(20, 3, 23)),
    ]
    total = summarize_usage(rs)
    assert total.prompt_tokens == 30
    assert total.completion_tokens == 5
    assert total.total_tokens == 35
