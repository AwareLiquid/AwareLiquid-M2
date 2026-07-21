import csv
import io

import pytest

from awareliquid.adapter.submission_contract import (
    CSV_HEADER,
    TOKEN_BUDGET,
    ContractError,
    SubmissionAnswer,
    SubmissionSummary,
    parse_question,
    parse_questions,
    render_submission_csv,
    validate_submission,
)


def test_official_csv_is_utf8_five_columns_with_summary_and_no_unused_tokens():
    questions = _questions()
    result = render_submission_csv(
        questions,
        [
            SubmissionAnswer("synthetic_mcq", "D", 10, 2, 12),
            SubmissionAnswer("synthetic_multi", "AC", 20, 3, 23),
            SubmissionAnswer("synthetic_tf", "B", 30, 4, 34),
        ],
    )

    assert result.encode("utf-8").decode("utf-8") == result
    rows = list(csv.reader(io.StringIO(result)))
    assert rows[0] == list(CSV_HEADER)
    assert rows[1] == ["summary", "", "60", "9", "69"]
    assert all(len(row) == 5 for row in rows)
    assert "unused_tokens" not in result


@pytest.mark.parametrize(
    ("qid", "answer", "code"),
    [
        ("synthetic_mcq", "a", "INVALID_FORMAL_ANSWER"),
        ("synthetic_mcq", "E", "INVALID_FORMAL_ANSWER"),
        ("synthetic_multi", "CA", "INVALID_FORMAL_ANSWER"),
        ("synthetic_multi", "AAC", "INVALID_FORMAL_ANSWER"),
        ("synthetic_multi", "A,C", "INVALID_FORMAL_ANSWER"),
        ("synthetic_tf", "T", "INVALID_FORMAL_ANSWER"),
        ("synthetic_tf", "C", "INVALID_FORMAL_ANSWER"),
        ("synthetic_tf", "F", "INVALID_FORMAL_ANSWER"),
    ],
)
def test_answers_fail_closed_against_their_provided_question_options(qid, answer, code):
    answers = _answers()
    index = [item.qid for item in answers].index(qid)
    item = answers[index]
    answers[index] = SubmissionAnswer(qid, answer, item.prompt_tokens, item.completion_tokens, item.total_tokens)

    with pytest.raises(ContractError) as caught:
        render_submission_csv(_questions(), answers)

    assert caught.value.code == code


def test_tf_output_uses_a_or_b_from_the_provided_tf_options():
    question = parse_question(
        _question(
            "synthetic_tf",
            "tf",
            {"A": "false", "B": "true"},
        )
    )

    output = render_submission_csv([question], [SubmissionAnswer(question.qid, "B", 1, 1, 2)])

    assert list(csv.reader(io.StringIO(output)))[2][1] == "B"


@pytest.mark.parametrize(
    "answers_factory",
    [
        lambda: [SubmissionAnswer("synthetic_mcq", "A", 1, 1, 2)] * 2,
        lambda: [SubmissionAnswer("synthetic_mcq", "A", 1, 1, 2)],
        lambda: _answers() + [SubmissionAnswer("unknown", "A", 1, 1, 2)],
    ],
)
def test_duplicate_incomplete_and_unknown_output_qids_fail_closed(answers_factory):
    with pytest.raises(ContractError) as caught:
        render_submission_csv(_questions(), answers_factory())

    assert caught.value.code in {"DUPLICATE_QID", "INCOMPLETE_OR_UNKNOWN_QIDS"}


def test_input_duplicate_qids_and_malformed_question_fail_closed():
    payload = _question("same", "mcq", {"A": "a", "B": "b", "C": "c", "D": "d"})

    with pytest.raises(ContractError, match="unique"):
        parse_questions([payload, payload])
    payload["options"] = {"A": "a", "B": "b"}
    with pytest.raises(ContractError) as caught:
        parse_question(payload)
    assert caught.value.code == "INVALID_OPTIONS"


def test_split_a_and_b_scope_rules_are_independent_of_answer_format():
    a = _question("a_scope", "tf", {"A": "false", "B": "true"})
    b = dict(a)
    b.update({"qid": "b_scope", "split": "B", "doc_ids": None})

    assert parse_question(a).split == "A"
    assert parse_question(b).split == "B"
    b_with_scope = dict(b, doc_ids=["doc-1"])
    with pytest.raises(ContractError, match="must not include doc_ids"):
        parse_question(b_with_scope)


@pytest.mark.parametrize(
    "usage",
    [
        (1, 1, None),
        (True, 0, 1),
        (1, 1, 1),
    ],
)
def test_missing_unknown_or_inconsistent_per_question_usage_fails_closed(usage):
    with pytest.raises(ContractError) as caught:
        SubmissionAnswer("synthetic_mcq", "A", *usage)

    assert caught.value.code in {"UNKNOWN_OR_INVALID_TOKEN_USAGE", "INCONSISTENT_TOKEN_TOTAL"}


def test_aggregate_token_equality_and_budget_are_fail_closed():
    questions = _questions()
    oversized = [
        SubmissionAnswer("synthetic_mcq", "A", TOKEN_BUDGET, 1, TOKEN_BUDGET + 1),
        SubmissionAnswer("synthetic_multi", "AC", 0, 0, 0),
        SubmissionAnswer("synthetic_tf", "B", 0, 0, 0),
    ]

    with pytest.raises(ContractError) as caught:
        render_submission_csv(questions, oversized)
    assert caught.value.code == "TOKEN_BUDGET_EXCEEDED"

    with pytest.raises(ContractError) as caught:
        SubmissionSummary(1, 1, 1)
    assert caught.value.code == "INCONSISTENT_TOKEN_TOTAL"


def test_validate_submission_returns_exact_aggregate_totals():
    summary = validate_submission(_questions(), _answers())

    assert summary == SubmissionSummary(prompt_tokens=60, completion_tokens=9, total_tokens=69)


def _questions():
    return parse_questions(
        [
            _question("synthetic_mcq", "mcq", {"A": "a", "B": "b", "C": "c", "D": "d"}),
            _question("synthetic_multi", "multi", {"A": "a", "B": "b", "C": "c", "D": "d"}),
            _question("synthetic_tf", "tf", {"A": "false", "B": "true"}),
        ]
    )


def _answers():
    return [
        SubmissionAnswer("synthetic_mcq", "A", 10, 2, 12),
        SubmissionAnswer("synthetic_multi", "AC", 20, 3, 23),
        SubmissionAnswer("synthetic_tf", "B", 30, 4, 34),
    ]


def _question(qid, answer_format, options):
    return {
        "qid": qid,
        "domain": "synthetic",
        "split": "A",
        "question": "synthetic question",
        "options": options,
        "answer_format": answer_format,
        "type": "synthetic",
        "doc_ids": ["synthetic-doc"],
    }
