import json
from pathlib import Path

import pytest

from awareliquid.evidence import (
    REQUIRED_PROVENANCE_FIELDS,
    Evidence,
    normalize_evidence,
)


FIXTURE = Path(__file__).parent / "fixtures" / "evidence_synthetic.json"


def _records():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [
        {**record, "parser_version": fixture["parser_version"]}
        for record in fixture["records"]
    ]


def test_synthetic_provenance_preserves_all_required_boundary_fields():
    evidence = normalize_evidence(_records()[0])

    assert evidence.page == 2
    assert (evidence.char_start, evidence.char_end) == (41, 66)
    assert evidence.table_id == "table-2"
    assert evidence.row_id == "row-revenue"
    assert evidence.column_ids == ("metric", "amount", "unit")
    assert evidence.unit == "CNY million"
    assert evidence.footnote == "Amounts exclude tax."
    assert evidence.parent_evidence_id == "evidence-parent-fixture"
    assert evidence.neighbor_evidence_ids == (
        "evidence-before-fixture",
        "evidence-after-fixture",
    )
    assert evidence.parse_warning == ("table-grid-reconstructed",)

    serialized = evidence.to_dict()
    required = {
        "domain", "doc_id", "page", "source_path", "char_start", "char_end",
        "section", "title", "table_id", "row_id", "column_ids", "unit",
        "footnote", "parent_evidence_id", "neighbor_evidence_ids", "parse_warning",
    }
    assert required <= serialized.keys()


def test_evidence_id_is_stable_and_independent_of_boundary_annotations():
    record = _records()[0]
    first = Evidence.from_dict(record)
    second = Evidence.from_dict(dict(record))
    annotated = Evidence.from_dict({**record, "parse_warning": ["different-warning"]})

    assert first.evidence_id == second.evidence_id == annotated.evidence_id
    assert first.evidence_id == first.stable_id()

    changed_content = Evidence.from_dict({**record, "content": record["content"] + "!"})
    changed_offset = Evidence.from_dict({**record, "char_end": 67})
    assert changed_content.evidence_id != first.evidence_id
    assert changed_offset.evidence_id != first.evidence_id


def test_empty_pages_and_parse_warnings_are_retained_without_inference():
    empty_page = Evidence.from_dict(_records()[1])

    assert empty_page.content == ""
    assert empty_page.char_start == empty_page.char_end == 67
    assert empty_page.parse_warning == ("empty-page", "ocr-unavailable")


@pytest.mark.parametrize("field", REQUIRED_PROVENANCE_FIELDS)
def test_missing_required_provenance_or_boundary_field_fails_closed(field):
    record = _records()[0]
    del record[field]

    with pytest.raises(ValueError, match=rf"missing required evidence fields: .*{field}"):
        Evidence.from_dict(record)


def test_explicit_null_optional_boundaries_are_preserved():
    empty_page = Evidence.from_dict(_records()[1])

    serialized = empty_page.to_dict()
    for field in (
        "section",
        "title",
        "table_id",
        "row_id",
        "unit",
        "footnote",
        "parent_evidence_id",
    ):
        assert serialized[field] is None
    assert serialized["column_ids"] == []
    assert serialized["neighbor_evidence_ids"] == []


@pytest.mark.parametrize("field", ("column_ids", "neighbor_evidence_ids", "parse_warning"))
def test_explicit_null_for_required_collection_boundary_fails_closed(field):
    with pytest.raises(TypeError, match=rf"{field} must be a sequence of strings"):
        Evidence.from_dict({**_records()[0], field: None})


@pytest.mark.parametrize(
    "record, error",
    [
        ({"domain": "x"}, "missing required evidence fields"),
        ({**_records()[0], "page": 0}, "page must be a positive integer"),
        ({**_records()[0], "char_start": 8, "char_end": 7}, "offsets must satisfy"),
        ({**_records()[0], "column_ids": "metric"}, "column_ids must be a sequence"),
    ],
)
def test_malformed_provenance_fails_closed(record, error):
    with pytest.raises((TypeError, ValueError), match=error):
        Evidence.from_dict(record)
