"""Unit tests for the submission builder's input parsing (pure helpers)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `submit`

import submit  # noqa: E402


def test_options_dict_ordered_by_letter():
    # A {letter: text} dict must become a list in A,B,C,D order so the agent's
    # positional labels line up with the official option letters.
    opts = {"C": "third", "A": "first", "B": "second", "D": "fourth"}
    assert submit.options_to_list(opts) == ["first", "second", "third", "fourth"]


def test_options_list_passthrough():
    assert submit.options_to_list(["x", "y"]) == ["x", "y"]
    assert submit.options_to_list(None) == []


def test_answer_format_maps_to_qtype():
    assert submit._QTYPE["mcq"] == "mcq"
    assert submit._QTYPE["multiple_choice"] == "multi"
    assert submit._QTYPE["判断"] == "tf"
    assert submit._QTYPE["true_false"] == "tf"


def test_load_questions_jsonl_and_array(tmp_path):
    items = [{"qid": "1", "question": "a"}, {"qid": "2", "question": "b"}]
    jsonl = tmp_path / "q.jsonl"
    jsonl.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items), encoding="utf-8")
    assert submit.load_questions(str(jsonl)) == items

    arr = tmp_path / "q.json"
    arr.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    assert submit.load_questions(str(arr)) == items


def test_load_docs_dict_and_array_and_dir(tmp_path):
    d1 = tmp_path / "docs.json"
    d1.write_text(json.dumps({"a": "text-a"}, ensure_ascii=False), encoding="utf-8")
    assert submit.load_docs(str(d1)) == {"a": "text-a"}

    d2 = tmp_path / "docs2.json"
    d2.write_text(json.dumps([{"doc_id": "b", "content": "text-b"}], ensure_ascii=False), encoding="utf-8")
    assert submit.load_docs(str(d2)) == {"b": "text-b"}

    ddir = tmp_path / "docs_dir"
    ddir.mkdir()
    (ddir / "c.txt").write_text("text-c", encoding="utf-8")
    assert submit.load_docs(str(ddir)) == {"c": "text-c"}
