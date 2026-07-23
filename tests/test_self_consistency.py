"""Per-letter majority voting over independent answer samples."""

import pytest

from awareliquid.adapter.qa_agent import RetrievalConfig, _majority_vote


def test_disabled_by_default():
    assert RetrievalConfig().self_consistency == 1


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        (["B", "B", "A"], "B"),          # noise flip out-voted
        (["A", "B", "A"], "A"),
        (["C", "C", "C"], "C"),
    ],
)
def test_single_answer_formats_take_the_most_frequent_letter(samples, expected):
    assert _majority_vote(samples, "mcq", 4) == expected


def test_multi_drops_a_letter_that_lacks_a_majority():
    # The spurious "D" appears once in three samples -> excluded. This is the
    # over-selection failure mode that costs a multi-select question everything.
    assert _majority_vote(["AB", "ABD", "AB"], "multi", 4) == "AB"


def test_multi_keeps_every_letter_the_model_consistently_supports():
    assert _majority_vote(["AD", "AD", "ABD"], "multi", 4) == "AD"


def test_multi_requires_a_strict_majority():
    # Two of four samples is not a majority, so "D" is dropped.
    assert _majority_vote(["AB", "ABD", "AB", "ABD"], "multi", 4) == "AB"


def test_multi_falls_back_to_the_most_common_answer_when_nothing_wins():
    # Total disagreement: no letter clears half, so return the modal answer
    # rather than inventing an empty or merged one.
    assert _majority_vote(["AB", "CD"], "multi", 4) in {"AB", "CD"}


def test_ill_formed_samples_are_ignored():
    assert _majority_vote(["", "B", "B"], "mcq", 4) == "B"


def test_all_ill_formed_returns_first_sample_for_the_existing_retry_path():
    assert _majority_vote(["", ""], "mcq", 4) == ""


def test_tf_voting():
    assert _majority_vote(["A", "B", "A"], "tf", 2) == "A"
