import json
import glob
from pathlib import Path

import pytest

from awareliquid.adapter.contracts import (
    AFAC_TOKEN_BUDGET,
    ContractError,
    SubmissionTokenUsage,
    build_question_row,
    build_summary_row,
    validate_formal_answer,
    validate_question,
)
from awareliquid.adapter.afac_contract import parse_question


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "afac_synthetic_golden" / "questions.json"
)


def _question(**overrides):
    question = {
        "qid": "synthetic_fin_a_001",
        "domain": "financial_reports",
        "split": "A",
        "question": "Synthetic question?",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "answer_format": "mcq",
        "type": "事实查找",
        "doc_ids": ["synthetic_doc_01"],
    }
    question.update(overrides)
    return question


@pytest.mark.parametrize("doc_ids", [None, []])
def test_split_a_fails_closed_without_nonempty_doc_ids(doc_ids):
    with pytest.raises(ContractError) as exc_info:
        validate_question(_question(doc_ids=doc_ids))

    assert exc_info.value.as_dict()["code"] == "INVALID_DOC_IDS_FOR_SPLIT_A"


def test_split_a_fails_closed_when_doc_ids_is_missing():
    question = _question()
    question.pop("doc_ids")

    with pytest.raises(ContractError) as exc_info:
        validate_question(question)

    assert exc_info.value.as_dict()["code"] == "INVALID_DOC_IDS_FOR_SPLIT_A"


@pytest.mark.parametrize("include_null", [False, True])
def test_split_b_normalizes_missing_or_null_doc_ids(include_null):
    question = _question(qid="synthetic_fin_b_001", split="B")
    if include_null:
        question["doc_ids"] = None
    else:
        question.pop("doc_ids")

    normalized = validate_question(question)

    assert normalized.split == "B"
    assert normalized.doc_ids is None


@pytest.mark.parametrize("split", ["a", "b"])
def test_explicit_lowercase_split_is_normalized_consistently_by_both_entrypoints(split):
    payload = _question(split=split)
    if split == "b":
        payload.pop("doc_ids")

    canonical = parse_question(payload)
    internal = validate_question(payload)

    assert canonical.split == internal.split == split.upper()
    assert canonical.doc_ids == internal.doc_ids


@pytest.mark.parametrize("doc_ids", [[], ["synthetic_doc_01"]])
def test_split_b_rejects_any_doc_ids_list(doc_ids):
    with pytest.raises(ContractError) as exc_info:
        validate_question(_question(split="B", doc_ids=doc_ids))

    assert exc_info.value.as_dict()["code"] == "DOC_IDS_SPLIT_CONFLICT"


def test_split_is_required_and_never_inferred_from_doc_ids():
    question = _question()
    question.pop("split")

    with pytest.raises(ContractError) as exc_info:
        validate_question(question)

    assert exc_info.value.as_dict()["code"] == "MISSING_SPLIT"


def test_type_hint_cannot_override_explicit_contract_fields():
    normalized = validate_question(_question(type="tf"))

    assert normalized.answer_format == "mcq"
    assert normalized.type_hint == "tf"


def test_options_are_exactly_the_four_official_keys():
    with pytest.raises(ContractError) as exc_info:
        validate_question(_question(options={"A": "yes", "B": "no"}))

    assert exc_info.value.as_dict()["code"] == "INVALID_OPTIONS"


def test_real_group_a_questions_pass_both_contract_entrypoints():
    """The internal and canonical validators must agree on audited A data."""

    paths = sorted(glob.glob("data/questions/group_a/*.json"))
    questions = [item for path in paths for item in json.loads(Path(path).read_text(encoding="utf-8"))]

    assert len(paths) == 5
    assert len(questions) == 100
    for payload in questions:
        canonical = parse_question(payload)
        internal = validate_question(payload)
        assert internal.answer_format == canonical.answer_format
        assert tuple(internal.options) == tuple(canonical.options)


@pytest.mark.parametrize(
    ("answer_format", "answer"),
    [("mcq", "A"), ("multi", "BD"), ("tf", "T"), ("tf", "F")],
)
def test_formal_answers_follow_official_output_contract(answer_format, answer):
    assert validate_formal_answer(answer_format, answer) == answer


@pytest.mark.parametrize(
    ("answer_format", "answer"),
    [("mcq", "AB"), ("multi", "DB"), ("multi", "BB"), ("tf", "A")],
)
def test_invalid_formal_answers_return_structured_errors(answer_format, answer):
    with pytest.raises(ContractError) as exc_info:
        validate_formal_answer(answer_format, answer)

    assert exc_info.value.as_dict()["code"] == "INVALID_FORMAL_ANSWER"


def test_csv_contract_has_no_header_and_summary_is_first_row():
    usage = SubmissionTokenUsage(prompt_tokens=900, completion_tokens=2, total_tokens=902)

    rows = [
        build_summary_row(
            budget_tokens=AFAC_TOKEN_BUDGET,
            used_tokens=902,
            unused_tokens=0,
        ),
        build_question_row("synthetic_fin_a_001", "A", usage),
    ]

    assert AFAC_TOKEN_BUDGET == 5_000_000
    assert rows[0] == ["summary", 5_000_000, 902, 0]
    assert rows[1] == ["synthetic_fin_a_001", "A", 900, 2, 902]
    assert rows[0][0] != "qid"


def test_summary_row_rejects_missing_unused_tokens():
    with pytest.raises(ContractError) as exc_info:
        build_summary_row(budget_tokens=AFAC_TOKEN_BUDGET, used_tokens=902)

    assert exc_info.value.as_dict()["code"] == "MISSING_UNUSED_TOKENS"


def test_question_token_columns_must_be_consistent():
    with pytest.raises(ContractError) as exc_info:
        SubmissionTokenUsage(prompt_tokens=900, completion_tokens=2, total_tokens=901)

    assert exc_info.value.as_dict()["code"] == "INVALID_TOKEN_USAGE"


def test_golden_fixture_is_explicitly_synthetic_and_contract_valid():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert fixture["fixture_metadata"]["synthetic"] is True
    assert fixture["fixture_metadata"]["real_dataset"] is False
    normalized = [validate_question(item) for item in fixture["questions"]]
    assert {question.split for question in normalized} == {"A", "B"}
