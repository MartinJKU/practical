"""Fail-closed validation for frozen MolecularIQ training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .constants import (
    BENCHMARK_DENYLIST,
    MOLECULARIQ_CORE_COMMIT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    TASK_FAMILIES,
    TRAIN_POOL_ID,
)
from .io_utils import read_json, records_sha256


class DatasetIntegrityError(RuntimeError):
    """Raised when a frozen dataset violates the training contract."""


REQUIRED_COLUMNS = {
    "example_id",
    "molecule_id",
    "task_family",
    "task_type",
    "prompt",
    "target_json",
    "constraints_json",
    "reference_smiles",
    "metadata_json",
    "molecular_complexity",
    "multitask_load",
    "question_seed",
    "source_dataset_revision",
    "generator_revision",
    "preprocessing_config_hash",
}

TASK_TYPES = {
    "count": {"single_count", "multi_count"},
    "index": {"single_index_identification", "multi_index_identification"},
    "constraint": {"constraint_generation"},
}


def _prompt_text(prompt: Any) -> str:
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise DatasetIntegrityError("prompt must contain exactly one system and one user message")
    parts: list[str] = []
    for message in prompt:
        if not isinstance(message, dict) or not {"role", "content"} <= set(message):
            raise DatasetIntegrityError("each prompt message must contain role and content")
        if message["role"] not in {"system", "user"} or not isinstance(message["content"], str):
            raise DatasetIntegrityError("prompt contains an invalid role or non-string content")
        parts.append(message["content"])
    if [message["role"] for message in prompt] != ["system", "user"]:
        raise DatasetIntegrityError("prompt roles must be ordered exactly as system, user")
    if prompt[0]["content"] != SYSTEM_PROMPT:
        raise DatasetIntegrityError("prompt does not use the pinned canonical system instruction")
    return "\n".join(parts)


def validate_records(records: Iterable[dict[str, Any]], task_family: str) -> tuple[int, str]:
    if task_family not in TASK_FAMILIES:
        raise DatasetIntegrityError(f"Unknown task family: {task_family}")
    materialized = list(records)
    if not materialized:
        raise DatasetIntegrityError("dataset has no rows")
    ids: set[str] = set()
    for row_number, row in enumerate(materialized):
        missing = REQUIRED_COLUMNS - set(row)
        if missing:
            raise DatasetIntegrityError(f"row {row_number} missing columns: {sorted(missing)}")
        example_id = row["example_id"]
        if not isinstance(example_id, str) or not example_id or example_id in ids:
            raise DatasetIntegrityError(f"row {row_number} has an empty or duplicate example_id")
        ids.add(example_id)
        if row["task_family"] != task_family:
            raise DatasetIntegrityError(f"row {row_number} crosses task-family boundary")
        if row["task_type"] not in TASK_TYPES[task_family]:
            raise DatasetIntegrityError(f"row {row_number} has a task type outside its family")

        prompt_text = _prompt_text(row["prompt"])
        target = json.loads(row["target_json"])
        constraints = json.loads(row["constraints_json"])
        metadata = json.loads(row["metadata_json"])
        if (
            not isinstance(target, dict)
            or not isinstance(constraints, list)
            or not isinstance(metadata, dict)
        ):
            raise DatasetIntegrityError(f"row {row_number} has invalid serialized reward metadata")
        if metadata.get("task_type") != row["task_type"]:
            raise DatasetIntegrityError(f"row {row_number} metadata/task_type mismatch")

        source_text = " ".join(
            str(row.get(key, "")) for key in ("source_dataset_revision", "generator_revision")
        ).lower()
        if any(token in source_text for token in BENCHMARK_DENYLIST):
            raise DatasetIntegrityError(f"row {row_number} references a forbidden benchmark/test source")

        if task_family in {"count", "index"}:
            if not target or constraints:
                raise DatasetIntegrityError(f"row {row_number} has wrong target/constraint fields")
            key_names = metadata.get("key_names")
            properties = metadata.get("properties")
            if (
                not isinstance(key_names, list)
                or set(key_names) != set(target)
                or len(key_names) != len(target)
                or not isinstance(properties, list)
                or len(properties) != len(key_names)
            ):
                raise DatasetIntegrityError(f"row {row_number} metadata does not agree with stored targets")
            source_smiles = metadata.get("smiles")
            expected_molecule_id = (
                hashlib.sha256(source_smiles.encode("utf-8")).hexdigest()[:24]
                if isinstance(source_smiles, str)
                else None
            )
            if expected_molecule_id != row["molecule_id"]:
                raise DatasetIntegrityError(f"row {row_number} molecule identity differs from metadata")
            if task_family == "count" and any(
                type(value) is not int or value < 0 for value in target.values()
            ):
                raise DatasetIntegrityError(f"row {row_number} has a non-integer count target")
            if task_family == "index":
                for values in target.values():
                    if (
                        not isinstance(values, list)
                        or any(type(value) is not int or value < 0 for value in values)
                        or values != sorted(set(values))
                    ):
                        raise DatasetIntegrityError(f"row {row_number} has invalid atom-index targets")
        else:
            if target or not constraints:
                raise DatasetIntegrityError(f"row {row_number} has wrong target/constraint fields")
            reference = row["reference_smiles"]
            if not isinstance(reference, str) or not reference:
                raise DatasetIntegrityError(f"row {row_number} lacks its reward-side reference witness")
            if reference in prompt_text:
                raise DatasetIntegrityError(f"row {row_number} leaks the constraint witness into the prompt")
            if metadata.get("constraints") != constraints:
                raise DatasetIntegrityError(
                    f"row {row_number} metadata does not agree with stored constraints"
                )
            for constraint in constraints:
                if (
                    not isinstance(constraint, dict)
                    or set(constraint) != {"type", "operator", "value"}
                    or not isinstance(constraint["type"], str)
                    or constraint["operator"] != "="
                    or isinstance(constraint["value"], bool)
                    or not isinstance(constraint["value"], (int, float))
                    or not math.isfinite(float(constraint["value"]))
                ):
                    raise DatasetIntegrityError(f"row {row_number} has an invalid stored constraint")

        # The exact serialized reward object must never be model-visible. Key hints
        # are intentionally visible, so checking individual keys/values is invalid.
        if row["target_json"] != "{}" and row["target_json"] in prompt_text:
            raise DatasetIntegrityError(f"row {row_number} leaks its serialized target")
        if row["constraints_json"] != "[]" and row["constraints_json"] in prompt_text:
            raise DatasetIntegrityError(f"row {row_number} leaks its serialized constraints")
        if not isinstance(row["multitask_load"], int) or row["multitask_load"] < 1:
            raise DatasetIntegrityError(f"row {row_number} has invalid multitask_load")
        expected_load = len(target) if task_family != "constraint" else len(constraints)
        if row["multitask_load"] != expected_load:
            raise DatasetIntegrityError(f"row {row_number} multitask_load differs from its reward object")
        complexity = row["molecular_complexity"]
        if not isinstance(complexity, (int, float)) or not math.isfinite(float(complexity)):
            raise DatasetIntegrityError(f"row {row_number} has non-finite molecular_complexity")

    return len(materialized), records_sha256(materialized)


def validate_artifact(artifact_path: str | Path, *, expected_task: str | None = None) -> dict[str, Any]:
    root = Path(artifact_path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    ready_path = root / "_READY"
    if not manifest_path.is_file() or not ready_path.is_file():
        raise DatasetIntegrityError(f"artifact is incomplete (manifest/_READY missing): {root}")
    manifest = read_json(manifest_path)
    task_family = manifest.get("task_family")
    if expected_task is not None and task_family != expected_task:
        raise DatasetIntegrityError(f"expected {expected_task}, artifact contains {task_family}")
    from datasets import load_from_disk

    dataset = load_from_disk(str(root / "dataset"))
    rows = [dict(row) for row in dataset]
    count, content_hash = validate_records(rows, task_family)
    build_provenance = manifest.get("build_runtime_provenance")
    if (
        manifest.get("source_dataset") != TRAIN_POOL_ID
        or manifest.get("moleculariq_core_commit") != MOLECULARIQ_CORE_COMMIT
        or manifest.get("system_prompt_sha256") != SYSTEM_PROMPT_SHA256
        or manifest.get("official_benchmark_used") is not False
        or not isinstance(build_provenance, dict)
        or not build_provenance.get("source_tree_sha256")
        or not isinstance(build_provenance.get("slurm"), dict)
    ):
        raise DatasetIntegrityError("artifact manifest violates the pinned training-data contract")
    for row in rows:
        if (
            row["source_dataset_revision"] != manifest.get("source_dataset_revision")
            or row["generator_revision"] != f"moleculariq-core@{MOLECULARIQ_CORE_COMMIT}"
            or row["preprocessing_config_hash"] != manifest.get("preprocessing_config_hash")
        ):
            raise DatasetIntegrityError("artifact row provenance differs from its manifest")
    if count != manifest.get("num_examples"):
        raise DatasetIntegrityError("row count differs from manifest")
    if content_hash != manifest.get("records_sha256"):
        raise DatasetIntegrityError("content hash differs from manifest")
    if manifest.get("dataset_artifact_id") != root.name:
        raise DatasetIntegrityError("artifact directory name does not match dataset_artifact_id")
    return manifest


def resolve_bundle_artifact(bundle_path: str | Path, task_family: str) -> Path:
    bundle = Path(bundle_path).expanduser().resolve()
    if not (bundle / "_READY").is_file():
        raise DatasetIntegrityError(f"dataset bundle is not ready: {bundle}")
    manifest = read_json(bundle / "bundle_manifest.json")
    if manifest.get("bundle_id") != bundle.name:
        raise DatasetIntegrityError("bundle ID does not match directory name")
    entries = manifest.get("artifacts", {})
    if (
        manifest.get("source_dataset") != TRAIN_POOL_ID
        or manifest.get("official_benchmark_used") is not False
    ):
        raise DatasetIntegrityError("bundle manifest violates the training-only source contract")
    if task_family not in entries:
        raise DatasetIntegrityError(f"bundle lacks task family {task_family}")
    artifact = (bundle / entries[task_family]["relative_path"]).resolve()
    if bundle not in artifact.parents:
        raise DatasetIntegrityError("bundle manifest contains a path traversal")
    artifact_manifest = validate_artifact(artifact, expected_task=task_family)
    entry = entries[task_family]
    if (
        entry.get("artifact_id") != artifact_manifest.get("dataset_artifact_id")
        or entry.get("records_sha256") != artifact_manifest.get("records_sha256")
        or entry.get("num_examples") != artifact_manifest.get("num_examples")
        or manifest.get("offline_bundle_files") != artifact_manifest.get("offline_bundle_files")
        or manifest.get("build_runtime_provenance") != artifact_manifest.get("build_runtime_provenance")
    ):
        raise DatasetIntegrityError("bundle entry differs from its referenced artifact")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--task", choices=TASK_FAMILIES)
    args = parser.parse_args()
    if bool(args.artifact) == bool(args.bundle):
        parser.error("provide exactly one of --artifact or --bundle")
    if args.bundle and not args.task:
        parser.error("--bundle requires --task")
    manifest = (
        validate_artifact(args.artifact, expected_task=args.task)
        if args.artifact
        else validate_artifact(resolve_bundle_artifact(args.bundle, args.task), expected_task=args.task)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
