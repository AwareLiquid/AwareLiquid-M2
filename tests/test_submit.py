"""Focused local tests for the pure official submission renderer."""

import csv
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import submit  # noqa: E402
from awareliquid.adapter.afac_contract import ContractError, SubmissionAnswer, parse_questions


def test_render_csv_delegates_to_the_official_five_column_contract():
    questions = _questions()

    output = submit.render_csv(questions, _answers())

    assert list(csv.reader(io.StringIO(output))) == [
        ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"],
        ["summary", "", "3", "2", "5"],
        ["q1", "A", "3", "2", "5"],
    ]


def test_render_csv_does_not_accept_partial_outputs_or_unused_token_fields():
    with pytest.raises(ContractError) as caught:
        submit.render_csv(_questions(), [])

    assert caught.value.code == "INCOMPLETE_OR_UNKNOWN_QIDS"
    assert "unused_tokens" not in submit.render_csv(_questions(), _answers())


def test_options_to_list_preserves_official_option_order():
    assert submit.options_to_list({"C": "third", "A": "first", "B": "second", "D": "fourth"}) == [
        "first", "second", "third", "fourth",
    ]


def test_checkpoint_is_atomic_ordered_and_input_bound(tmp_path):
    questions = _questions()
    payloads = [{
        "qid": "q1", "domain": "synthetic", "split": "A",
        "question": "synthetic", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "answer_format": "mcq", "type": "synthetic", "doc_ids": ["doc"],
    }]
    docs = {"doc": "source"}
    checkpoint = tmp_path / "run.checkpoint.json"
    answers = _answers()
    submit._atomic_write_text(
        checkpoint,
        json.dumps(submit._checkpoint_payload(payloads, docs, "lexical", answers)),
    )

    restored = submit._load_checkpoint(checkpoint, payloads, questions, docs, "lexical")
    assert restored == answers
    assert not list(tmp_path.glob("*.tmp"))

    with pytest.raises(ValueError, match="question set"):
        submit._load_checkpoint(checkpoint, [dict(payloads[0], qid="other")], questions, docs, "lexical")


def _questions():
    return parse_questions(
        [{
            "qid": "q1", "domain": "synthetic", "split": "A",
            "question": "synthetic", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer_format": "mcq", "type": "synthetic", "doc_ids": ["doc"],
        }]
    )


def _answers():
    return [SubmissionAnswer("q1", "A", 3, 2, 5)]
