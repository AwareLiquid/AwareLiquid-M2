"""Integrity checks for the labelled evaluation set.

A labelled set is only as good as its labels. These tests cannot verify that a
gold answer is *semantically* right, but they catch the mechanical failures that
would silently invalidate a whole evaluation run: a question citing a document
that does not exist, an answer letter outside its own options, a judgment
question with four options, or a multi answer that is not sorted/deduped.
"""

import json
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent / "evals"
QUESTIONS = EVAL_DIR / "questions.jsonl"
CORPUS = EVAL_DIR / "corpus"


def _questions():
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_eval_set_is_present_and_non_trivial():
    questions = _questions()
    assert len(questions) >= 20
    assert len(list(CORPUS.glob("*.md"))) >= 4


def test_every_cited_document_exists():
    available = {p.stem for p in CORPUS.glob("*.md")}
    for q in _questions():
        missing = set(q["doc_ids"]) - available
        assert not missing, f"{q['qid']} cites missing document(s): {missing}"


def test_qids_are_unique():
    qids = [q["qid"] for q in _questions()]
    assert len(qids) == len(set(qids))


@pytest.mark.parametrize("question", _questions(), ids=lambda q: q["qid"])
def test_option_shape_matches_answer_format(question):
    keys = set(question["options"])
    if question["answer_format"] == "tf":
        assert keys == {"A", "B"}, "judgment questions carry exactly A/B"
    else:
        assert keys == {"A", "B", "C", "D"}


@pytest.mark.parametrize("question", _questions(), ids=lambda q: q["qid"])
def test_gold_answer_is_well_formed(question):
    gold, fmt, keys = question["answer"], question["answer_format"], set(question["options"])

    assert gold, "gold answer must not be empty"
    assert set(gold) <= keys, f"gold {gold} uses letters outside the options"

    if fmt in ("mcq", "tf"):
        assert len(gold) == 1, f"{fmt} gold must be a single letter, got {gold!r}"
    else:  # multi
        assert gold == "".join(sorted(set(gold))), "multi gold must be sorted and deduped"
        assert len(gold) >= 2, "a multi answer with one letter is probably a mistake"


def test_required_contract_fields_are_present():
    for q in _questions():
        for field in ("qid", "domain", "split", "type", "answer_format", "doc_ids", "question"):
            assert q.get(field), f"{q.get('qid')} is missing {field}"
        assert q["split"] == "A" and q["doc_ids"], "split A requires non-empty doc_ids"


def test_the_set_exercises_every_answer_format_and_several_question_types():
    questions = _questions()
    assert {q["answer_format"] for q in questions} == {"mcq", "tf", "multi"}
    # Guessing "A" everywhere must not score well, or the set measures nothing.
    all_a = sum(1 for q in questions if q["answer"] == "A")
    assert all_a / len(questions) < 0.5, "gold answers are too concentrated on A"
    assert len({q["type"] for q in questions}) >= 4
