"""Synthetic-only tests for the isolated A4 structural reasoning module."""

import ast
from decimal import Decimal
from pathlib import Path

import pytest

from awareliquid.evidence import Evidence
from awareliquid.formal_reasoning import (
    EvidenceContext,
    EvidenceDecimal,
    ExactLocator,
    ExactQuery,
    OptionState,
    evaluate_option,
    structural_arithmetic,
)


def _evidence(doc_id, page, start, content, **boundaries):
    return Evidence(
        domain="synthetic",
        doc_id=doc_id,
        page=page,
        source_path=f"/synthetic/{doc_id}.txt",
        char_start=start,
        char_end=start + len(content),
        content=content,
        section=boundaries.get("section"),
        title=boundaries.get("title"),
        table_id=boundaries.get("table_id"),
        row_id=boundaries.get("row_id"),
        column_ids=boundaries.get("column_ids", ()),
        unit=boundaries.get("unit"),
        footnote=boundaries.get("footnote"),
        parent_evidence_id=boundaries.get("parent_evidence_id"),
        neighbor_evidence_ids=boundaries.get("neighbor_evidence_ids", ()),
        parse_warning=boundaries.get("parse_warning", ()),
    )


def _context():
    later = _evidence(
        "doc-z", 2, 30, "Revenue: 12.50 CNY million; exception none.",
        section="Clause 2", title="Synthetic report", table_id="table-1",
        row_id="revenue", column_ids=("metric", "amount"), unit="CNY million",
        footnote="exact synthetic footnote", parent_evidence_id="parent-1",
        neighbor_evidence_ids=("before-1",), parse_warning=("synthetic-warning",),
    )
    first = _evidence(
        "doc-a", 1, 5, "Rating: AA; date 2026-01-01; rate 5%.",
        section="Clause 1", title="Synthetic schedule", table_id="table-2",
        row_id="rating", column_ids=("rating",), unit="percent",
    )
    return EvidenceContext((later, first))


def test_context_keeps_exact_text_and_every_boundary_without_transformation():
    context = _context()
    original = context.records[0]

    assert context.records[0] is original
    assert context.to_dicts()[0]["content"] == "Revenue: 12.50 CNY million; exception none."
    assert context.to_dicts()[0]["section"] == "Clause 2"
    assert context.to_dicts()[0]["table_id"] == "table-1"
    assert context.to_dicts()[0]["footnote"] == "exact synthetic footnote"
    assert context.select((original.evidence_id,)).records == (original,)


def test_decimal_arithmetic_is_decimal_only_bound_and_whitelisted():
    context = _context()
    left = EvidenceDecimal("12.50", (context.records[0].evidence_id,), context, "CNY")
    right = EvidenceDecimal(Decimal("2.50"), (context.records[1].evidence_id,), context, "CNY")

    result = structural_arithmetic("subtract", (left, right))

    assert result.value == Decimal("10.00")
    assert result.evidence_ids == (left.evidence_ids[0], right.evidence_ids[0])
    with pytest.raises(TypeError, match="decimal values"):
        EvidenceDecimal(1.5, left.evidence_ids, context)
    with pytest.raises(ValueError, match="unknown structural arithmetic"):
        structural_arithmetic("unknown", (left,))
    with pytest.raises(ZeroDivisionError):
        structural_arithmetic("divide", (left, EvidenceDecimal("0", right.evidence_ids, context)))


def test_option_states_are_deterministic_and_evidence_backed():
    context = _context()
    actual = EvidenceDecimal("12.50", (context.records[0].evidence_id,), context, "CNY")
    expected = EvidenceDecimal("12.50", (context.records[1].evidence_id,), context, "CNY")

    supported = evaluate_option(
        "A", actual, expected, observed_text="exception none", required_text="exception none"
    )
    refuted = evaluate_option("B", actual, EvidenceDecimal("10", expected.evidence_ids, context, "CNY"))
    insufficient = evaluate_option("C", actual, None)

    assert supported.state is OptionState.SUPPORTED
    assert supported.evidence_ids == (actual.evidence_ids[0], expected.evidence_ids[0])
    assert refuted.state is OptionState.REFUTED
    assert insufficient.state is OptionState.INSUFFICIENT
    assert insufficient.evidence_ids == actual.evidence_ids


def test_exact_locator_is_a_scoped_and_doc_page_char_stable():
    context = _context()
    locator = ExactLocator(context)

    result = locator.locate(ExactQuery(split="A", doc_ids=("doc-z", "doc-a")))
    assert result == (context.records[1].evidence_id, context.records[0].evidence_id)

    record = context.records[0]
    exact = locator.locate(ExactQuery(
        split="A", doc_ids=("doc-z",), title="Synthetic report",
        clause="Clause 2", phrase="Revenue", numbers=("12.50",),
        currencies=("CNY",), table_fields=(("row_id", "revenue"), ("column_ids", "amount")),
        char_start=record.char_start, char_end=record.char_end,
    ))
    assert exact == (record.evidence_id,)
    assert locator.locate(ExactQuery(split="A", doc_ids=("doc-z",), phrase="Income")) == ()
    assert locator.locate(ExactQuery(split="A", doc_ids=("doc-a",), phrase="Revenue")) == ()


def test_locator_fails_closed_for_missing_a_scope_and_unofficial_b_data():
    with pytest.raises(ValueError, match="A queries require"):
        ExactQuery(split="A")
    with pytest.raises(ValueError, match="official B data"):
        ExactQuery(split="B")
    with pytest.raises(ValueError, match="must not supply doc_ids"):
        ExactQuery(split="B", official_b_data=True, doc_ids=("doc-a",))
    with pytest.raises(ValueError, match="phrase must be a non-empty"):
        ExactQuery(split="A", doc_ids=("doc-a",), phrase="")


def test_module_has_no_prohibited_imports_or_dynamic_execution_nodes():
    source = Path(__file__).parents[1] / "awareliquid" / "formal_reasoning.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0].lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    banned_roots = {"torch", "transformers", "sentence_transformers"}
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert not imported & banned_roots
    assert not {"eval", "exec"} & calls
