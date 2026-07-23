"""Clause polarity is extracted as a structural field and surfaced to the model.

Motivation, from a real measured failure: on a question asking which situations
fall under a policy's exclusions, the model answered the same wrong letter in
3/3 samples because it read "酒后驾驶不属于责任免除" as if it said "属于". The
sentence was retrieved and survived compression — it was simply skimmed past.
Polarity is therefore extracted like any other structural field (clause id,
page, table row) rather than left to the model to notice.
"""

from awareliquid.adapter.evidence_index import (
    _negated_statements,
    _polarity_statements,
    build_evidence_nodes,
)
from awareliquid.adapter.qa_agent import MemoryQAAgent


def test_negated_sentences_are_extracted():
    found = _negated_statements(
        "本公司按基本保险金额给付。酒后驾驶不属于本条款的责任免除情形。等待期为 90 日。"
    )
    assert len(found) == 1
    assert "不属于" in found[0]


def test_affirmative_text_yields_no_negation():
    assert _negated_statements("本公司按已交保费给付身故保险金。等待期为 90 日。") == ()


def test_common_clause_negations_are_recognised():
    for sentence in (
        "本公司不承担给付保险金的责任。",
        "该情形不适用本条款。",
        "第 4 个保单年度起免收手续费。",
    ):
        assert _negated_statements(sentence), sentence
    # "除外" is an exception, deliberately NOT a negation -- see
    # test_exceptions_are_separated_from_plain_negations.
    assert _negated_statements("因输血导致的情形除外。") == ()


def test_a_decimal_does_not_split_a_sentence():
    # "3.45" must not be treated as a sentence boundary.
    found = _negated_statements("票面利率为 3.45%，该情形不适用本条款。")
    assert found and "3.45" in found[0]


def test_nodes_carry_polarity_into_metadata():
    nodes = build_evidence_nodes(
        "policy",
        "第六条 酒后驾驶不属于本条款的责任免除情形，本公司照常承担给付责任。",
    )
    assert nodes
    meta = nodes[0].as_meta()
    assert meta["negations"], "negation should reach the stored metadata"
    assert any("不属于" in item for item in meta["negations"])


def test_formatted_passage_states_the_negation_explicitly():
    rendered = MemoryQAAgent._format_passage(
        "第六条 酒后驾驶不属于本条款的责任免除情形。",
        {"doc_id": "policy", "chunk_idx": 0, "negations": ["酒后驾驶不属于本条款的责任免除情形"]},
    )
    assert "NEGATED" in rendered and "DENIES" in rendered
    assert "不属于" in rendered


def test_formatted_passage_is_unchanged_without_negation():
    rendered = MemoryQAAgent._format_passage(
        "第一条 本公司按基本保险金额给付。",
        {"doc_id": "policy", "chunk_idx": 0, "negations": []},
    )
    assert "NEGATED" not in rendered
    assert rendered.endswith("本公司按基本保险金额给付。")


def test_exceptions_are_separated_from_plain_negations():
    """"但……除外" carves a case OUT of a rule; it must not read as a negation."""
    negated, excepted = _polarity_statements(
        "被保险人感染艾滋病病毒导致重大疾病，但因输血、职业暴露导致的除外。"
    )
    assert excepted and "除外" in excepted[0]
    assert negated == (), "an exception must not be reported as a negation"


def test_plain_negation_is_not_reported_as_an_exception():
    negated, excepted = _polarity_statements("酒后驾驶不属于本条款的责任免除情形。")
    assert negated and excepted == ()


def test_passage_labels_the_two_polarity_kinds_differently():
    rendered = MemoryQAAgent._format_passage(
        "感染艾滋病……但因输血导致的除外。",
        {"doc_id": "policy", "chunk_idx": 0, "negations": [], "exceptions": ["但因输血导致的除外"]},
    )
    assert "EXCEPTION" in rendered and "OPPOSITE" in rendered
    assert "NEGATED" not in rendered
