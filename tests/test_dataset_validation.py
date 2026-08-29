from __future__ import annotations

import hashlib
import json

import pytest

from miq_grpo.constants import SYSTEM_PROMPT
from miq_grpo.dataset_validation import DatasetIntegrityError, validate_records


def row_for(task_family: str = "count") -> dict:
    target = {
        "count": {"ring_count": 1},
        "index": {"ring_indices": [0, 1, 2, 3, 4, 5]},
        "constraint": {},
    }[task_family]
    constraints = [{"type": "ring_count", "operator": "=", "value": 1}] if task_family == "constraint" else []
    task_type = {
        "count": "single_count",
        "index": "single_index_identification",
        "constraint": "constraint_generation",
    }[task_family]
    source_smiles = "c1ccccc1"
    metadata = (
        {"task_type": task_type, "constraints": constraints}
        if task_family == "constraint"
        else {
            "task_type": task_type,
            "smiles": source_smiles,
            "properties": ["ring_count" if task_family == "count" else "ring_index"],
            "key_names": list(target),
        }
    )
    return {
        "example_id": f"example-{task_family}",
        "molecule_id": hashlib.sha256(source_smiles.encode()).hexdigest()[:24],
        "task_family": task_family,
        "task_type": task_type,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How many rings? key=ring_count"},
        ],
        "target_json": json.dumps(target, separators=(",", ":")),
        "constraints_json": json.dumps(constraints, separators=(",", ":")),
        "reference_smiles": "c1ccccc1" if task_family == "constraint" else "",
        "metadata_json": json.dumps(metadata, separators=(",", ":")),
        "molecular_complexity": 6.0,
        "multitask_load": 1,
        "question_seed": 123,
        "source_dataset_revision": "train-pool-sha",
        "generator_revision": "moleculariq-core@commit",
        "preprocessing_config_hash": "config-sha",
    }


def test_valid_records_return_stable_count_and_hash() -> None:
    first = row_for()
    second = {**row_for(), "example_id": "example-count-2", "question_seed": 124}

    count, digest = validate_records([first, second], "count")

    assert count == 2
    assert len(digest) == 64
    assert validate_records([first, second], "count")[1] == digest


def test_prompt_messages_require_both_role_and_content() -> None:
    row = row_for()
    row["prompt"] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user"}]
    with pytest.raises(DatasetIntegrityError, match="role and content"):
        validate_records([row], "count")

    wrong_prompt = row_for()
    wrong_prompt["prompt"][0]["content"] = "A different system prompt"
    with pytest.raises(DatasetIntegrityError, match="canonical system"):
        validate_records([wrong_prompt], "count")


def test_benchmark_source_and_cross_family_rows_are_rejected() -> None:
    benchmark = row_for()
    benchmark["source_dataset_revision"] = "ml-jku/moleculariq-v0.0/test"
    with pytest.raises(DatasetIntegrityError, match="forbidden benchmark"):
        validate_records([benchmark], "count")

    with pytest.raises(DatasetIntegrityError, match="task-family"):
        validate_records([row_for("index")], "count")


def test_constraint_witness_is_reward_side_only() -> None:
    valid = row_for("constraint")
    assert validate_records([valid], "constraint")[0] == 1

    leaked = row_for("constraint")
    leaked["prompt"][1]["content"] += " c1ccccc1"
    with pytest.raises(DatasetIntegrityError, match="witness"):
        validate_records([leaked], "constraint")


def test_serialized_target_cannot_be_model_visible() -> None:
    row = row_for()
    row["prompt"][1]["content"] += " " + row["target_json"]
    with pytest.raises(DatasetIntegrityError, match="serialized target"):
        validate_records([row], "count")


@pytest.mark.parametrize("indices", ([0, 0], [1, 0], [0, -1], [0, True], [0, 1.0]))
def test_index_targets_are_strict_nonnegative_sorted_unique_integers(indices: list) -> None:
    row = row_for("index")
    row["target_json"] = json.dumps({"ring_indices": indices}, separators=(",", ":"))
    with pytest.raises(DatasetIntegrityError, match="atom-index"):
        validate_records([row], "index")


def test_reward_metadata_task_type_must_match_row() -> None:
    row = row_for()
    metadata = json.loads(row["metadata_json"])
    metadata["task_type"] = "multi_count"
    row["metadata_json"] = json.dumps(metadata, separators=(",", ":"))
    with pytest.raises(DatasetIntegrityError, match="metadata/task_type"):
        validate_records([row], "count")
