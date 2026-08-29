from __future__ import annotations

import pytest

from miq_grpo.parsing import ParseStatus, completion_text, extract_smiles, parse_completion


def test_conversational_completion_and_strict_format() -> None:
    completion = [{"role": "assistant", "content": '<answer>{"ring_count": 2}</answer>'}]
    parsed = parse_completion(completion, "count")

    assert completion_text(completion).startswith("<answer>")
    assert parsed.status == ParseStatus.OK
    assert parsed.answer == {"ring_count": 2}
    assert parsed.format_ok


@pytest.mark.parametrize(
    "text",
    [
        '<answer>{"ring_count": 1, "ring_count": 2}</answer>',
        '<answer>{"ring_count": NaN}</answer>',
        '<answer>{"ring_count": 1e999}</answer>',
    ],
)
def test_non_strict_or_ambiguous_json_cannot_be_rewarded_as_valid_json(text: str) -> None:
    assert parse_completion(text, "count").status != ParseStatus.OK


def test_conflicting_answer_blocks_fail_closed() -> None:
    parsed = parse_completion(
        '<answer>{"ring_count": 1}</answer> then <answer>{"ring_count": 2}</answer>',
        "count",
    )
    assert parsed.status == ParseStatus.MULTIPLE_CONFLICTING_ANSWERS


def test_identical_repeated_blocks_can_be_scored_but_get_no_format_bonus() -> None:
    parsed = parse_completion(
        '<answer>{"ring_count": 1}</answer><answer>{"ring_count": 1}</answer>',
        "count",
    )
    assert parsed.status == ParseStatus.OK
    assert not parsed.format_ok


def test_unbalanced_answer_tags_are_malformed() -> None:
    assert parse_completion('<answer>{"ring_count": 1}', "count").status == ParseStatus.MALFORMED


def test_code_fenced_json_is_scored_without_format_bonus() -> None:
    parsed = parse_completion('```json\n{"ring_count": 1}\n```', "count")
    assert parsed.status == ParseStatus.OK
    assert parsed.answer == {"ring_count": 1}
    assert not parsed.format_ok


def test_index_list_and_count_scalar_are_supported_by_official_semantics() -> None:
    assert parse_completion("[0, 2]", "index").answer == [0, 2]
    assert parse_completion("2", "count").answer == 2


def test_constraint_requires_one_whitespace_free_smiles_value() -> None:
    valid = parse_completion('<answer>{"SMILES":"CCO"}</answer>', "constraint")
    invalid = parse_completion('<answer>{"smiles":"C C"}</answer>', "constraint")

    assert valid.status == ParseStatus.OK
    assert extract_smiles(valid.answer) == "CCO"
    assert invalid.status == ParseStatus.INVALID_JSON


def test_empty_and_unsupported_inputs_are_explicit() -> None:
    assert parse_completion("  ", "count").status == ParseStatus.EMPTY
    assert parse_completion("{}", "other").status == ParseStatus.UNSUPPORTED_TASK
