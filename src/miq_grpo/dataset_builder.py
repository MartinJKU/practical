"""Offline question generation from the staged MolecularIQ training pool."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import ConfigError, load_yaml, require_keys
from .constants import (
    MOLECULARIQ_CORE_COMMIT,
    PROJECT_SCHEMA_VERSION,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    TASK_FAMILIES,
    TRAIN_POOL_ID,
)
from .dataset_validation import validate_artifact
from .io_utils import (
    canonical_json,
    offline_bundle_file_identities,
    read_json,
    records_sha256,
    runtime_provenance,
    sha256_bytes,
    utc_now,
    write_json_exclusive,
)

_WORKER_MQD = None


def assert_moleculariq_api() -> None:
    """Fail before heavy work if the pinned core API is not available."""
    from moleculariq_core import MolecularIQD, evaluate_answer, load_molecule_pool

    required_methods = {
        "generate_count_question": {"smiles", "count_properties"},
        "generate_index_question": {"smiles", "index_properties"},
        "generate_constraint_question": {"constraints"},
        "compute_property": {"smiles", "property_name"},
        "validate_constraint_answer": {"predicted_smiles", "constraints"},
    }
    for name, parameters in required_methods.items():
        method = getattr(MolecularIQD, name, None)
        if method is None or not parameters <= set(inspect.signature(method).parameters):
            raise RuntimeError(f"Pinned MolecularIQ API mismatch for MolecularIQD.{name}")
    if not callable(evaluate_answer) or not callable(load_molecule_pool):
        raise RuntimeError("Pinned MolecularIQ evaluator or training-pool loader is unavailable")


def _init_worker(enable_random_phrasing: bool) -> None:
    global _WORKER_MQD
    from moleculariq_core import MolecularIQD

    _WORKER_MQD = MolecularIQD(seed=0, enable_random_phrasing=enable_random_phrasing)


def _json(value: Any) -> str:
    return canonical_json(value)


def _make_example_id(task: str, molecule_id: str, question_seed: int, payload: Any) -> str:
    raw = f"{task}\0{molecule_id}\0{question_seed}\0{canonical_json(payload)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _base_record(spec: dict[str, Any], question: str, metadata_value: dict[str, Any]) -> dict[str, Any]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(spec["smiles"])
    if mol is None:
        raise ValueError("invalid source SMILES")
    return {
        "example_id": "",
        "molecule_id": spec["molecule_id"],
        "task_family": spec["task_family"],
        "task_type": metadata_value["task_type"],
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "target_json": "{}",
        "constraints_json": "[]",
        "reference_smiles": "",
        "metadata_json": _json(metadata_value),
        "molecular_complexity": float(mol.GetNumHeavyAtoms()),
        "multitask_load": len(spec["properties"]),
        "question_seed": spec["question_seed"],
        "source_dataset_revision": spec["source_dataset_revision"],
        "generator_revision": f"moleculariq-core@{MOLECULARIQ_CORE_COMMIT}",
        "preprocessing_config_hash": spec["preprocessing_config_hash"],
    }


def _normalized_numeric(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"count/constraint property is non-numeric: {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError("non-finite property")
    return int(value) if float(value).is_integer() else float(value)


def _strict_count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"count property is not a nonnegative integer: {value!r}")
    return value


def _generate_record(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if _WORKER_MQD is None:
            _init_worker(spec["enable_random_phrasing"])
        mqd = _WORKER_MQD
        mqd.rng.seed(spec["question_seed"])
        task = spec["task_family"]
        properties = spec["properties"]
        if task == "count":
            question, target, meta = mqd.generate_count_question(spec["smiles"], properties)
            target = {key: _strict_count(value) for key, value in target.items()}
            row = _base_record(spec, question, meta)
            row["target_json"] = _json(target)
            row["example_id"] = _make_example_id(task, spec["molecule_id"], spec["question_seed"], target)
        elif task == "index":
            question, target, meta = mqd.generate_index_question(spec["smiles"], properties)
            normalized_target: dict[str, list[int]] = {}
            for key, values in target.items():
                if not isinstance(values, list):
                    raise ValueError("index property did not produce a list")
                if any(type(value) is not int or value < 0 for value in values):
                    raise ValueError("index property produced a non-integer or negative atom index")
                if len(set(values)) != len(values):
                    raise ValueError("index property produced duplicate atom indices")
                normalized_target[key] = sorted(values)
            row = _base_record(spec, question, meta)
            row["target_json"] = _json(normalized_target)
            row["example_id"] = _make_example_id(
                task, spec["molecule_id"], spec["question_seed"], normalized_target
            )
        elif task == "constraint":
            constraints = [
                {
                    "type": prop,
                    "operator": "=",
                    "value": _normalized_numeric(mqd.compute_property(spec["smiles"], prop)),
                }
                for prop in properties
            ]
            if spec["require_positive_constraint"] and not any(
                float(constraint["value"]) > 0 for constraint in constraints
            ):
                return None, "all_constraints_zero"
            question, meta = mqd.generate_constraint_question(constraints)
            witness_result = mqd.validate_constraint_answer(spec["smiles"], constraints, return_details=True)
            if not isinstance(witness_result, dict):
                return None, "unexpected_constraint_verifier_details"
            if (
                float(witness_result.get("reward", 0.0)) != 1.0
                or witness_result.get("supported") != witness_result.get("total")
                or witness_result.get("total") != len(constraints)
            ):
                return None, "reference_witness_failed_official_verifier"
            row = _base_record(spec, question, meta)
            row["constraints_json"] = _json(constraints)
            row["reference_smiles"] = spec["smiles"]
            row["example_id"] = _make_example_id(
                task, spec["molecule_id"], spec["question_seed"], constraints
            )
        else:
            return None, "unknown_task"
        return row, None
    except Exception as exc:  # expected chemistry failures are accounted, not fatal to the pool
        return None, f"{type(exc).__name__}:{str(exc)[:160]}"


def _specifications(
    rows: list[dict[str, str]],
    task_family: str,
    task_config: dict[str, Any],
    *,
    source_revision: str,
    preprocessing_hash: str,
    seed: int,
    attempts: int,
    enable_random_phrasing: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    molecule_order = list(range(len(rows)))
    rng.shuffle(molecule_order)
    specs: list[dict[str, Any]] = []
    allowed = list(task_config["allowed_properties"])
    minimum = int(task_config["min_properties"])
    maximum = int(task_config["max_properties"])
    if minimum < 1 or maximum < minimum or maximum > len(allowed):
        raise ConfigError(f"invalid property-count range for {task_family}")
    for index in range(attempts):
        if index and index % len(molecule_order) == 0:
            rng.shuffle(molecule_order)
        molecule = rows[molecule_order[index % len(molecule_order)]]
        load = rng.randint(minimum, maximum)
        properties = rng.sample(allowed, load)
        specs.append(
            {
                "task_family": task_family,
                "molecule_id": molecule["molecule_id"],
                "smiles": molecule["smiles"],
                "properties": properties,
                "question_seed": rng.getrandbits(63),
                "source_dataset_revision": source_revision,
                "preprocessing_config_hash": preprocessing_hash,
                "enable_random_phrasing": enable_random_phrasing,
                "require_positive_constraint": bool(task_config.get("require_positive_constraint", False)),
            }
        )
    return specs


def _generate_family(
    specs: list[dict[str, Any]], target_count: int, workers: int, enable_random_phrasing: bool
) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    if workers == 1:
        _init_worker(enable_random_phrasing)
        generated: Iterable[tuple[dict[str, Any] | None, str | None]] = map(_generate_record, specs)
        for record, reason in generated:
            if record is None:
                drops[reason or "unknown"] += 1
            elif len(records) < target_count:
                records.append(record)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(enable_random_phrasing,),
        ) as executor:
            for record, reason in executor.map(_generate_record, specs, chunksize=16):
                if record is None:
                    drops[reason or "unknown"] += 1
                elif len(records) < target_count:
                    records.append(record)
    if len(records) != target_count:
        raise RuntimeError(
            f"only generated {len(records)}/{target_count} valid examples; drops={dict(drops)}"
        )
    return records, drops


def build_bundle(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    require_keys(
        config,
        ("dataset_bundle_name", "source", "output_root", "seed", "task_families"),
        context="preprocessing config",
    )
    if int(config.get("schema_version", -1)) != PROJECT_SCHEMA_VERSION:
        raise ConfigError("unsupported preprocessing schema version")
    source = config["source"]
    if source.get("dataset_id") != TRAIN_POOL_ID or source.get("pool_name") != "train":
        raise ConfigError("preprocessing is locked to the official MolecularIQ training pool")
    if set(config["task_families"]) != set(TASK_FAMILIES):
        raise ConfigError("preprocessing config must materialize count, index, and constraint families")

    assert_moleculariq_api()
    staged_path = Path(source["staged_pool_path"]).expanduser().resolve()
    staged_manifest_path = Path(source["staged_manifest_path"]).expanduser().resolve()
    staged_manifest = read_json(staged_manifest_path)
    if not (staged_path.parent / "_READY").is_file():
        raise RuntimeError("staged training pool is not marked ready")
    if (
        staged_manifest.get("source_dataset") != TRAIN_POOL_ID
        or staged_manifest.get("pool_name") != "train"
        or staged_manifest.get("benchmark_data_present") is not False
    ):
        raise RuntimeError("staged source violates the training-only boundary")

    config_for_hash = {key: value for key, value in config.items() if key != "_config_path"}
    preprocessing_hash = sha256_bytes(canonical_json(config_for_hash).encode("utf-8"))
    source_revision = staged_manifest["source_dataset_revision"]
    bundle_id = f"{config['dataset_bundle_name']}-{preprocessing_hash[:12]}-{source_revision[:12]}"
    output_root = Path(config["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_bundle = output_root / bundle_id
    if final_bundle.exists():
        raise FileExistsError(f"refusing to overwrite dataset bundle: {final_bundle}")

    from datasets import Dataset, load_from_disk

    staged_dataset = load_from_disk(str(staged_path))
    pool_rows = [dict(row) for row in staged_dataset]
    if len(pool_rows) != staged_manifest["num_molecules"]:
        raise RuntimeError("staged training pool count differs from its manifest")
    if records_sha256(pool_rows) != staged_manifest["records_sha256"]:
        raise RuntimeError("staged training pool content hash mismatch")

    workers = int(config.get("workers", 1))
    if workers < 1:
        raise ConfigError("workers must be >= 1")
    attempts_factor = float(config.get("max_attempts_factor", 1.25))
    if attempts_factor < 1.0:
        raise ConfigError("max_attempts_factor must be >= 1")
    random_phrasing = bool(config.get("enable_random_phrasing", True))
    constraint_operators = config["task_families"]["constraint"].get("operators")
    if constraint_operators != ["="]:
        raise ConfigError("this frozen-data generator currently supports exactly operators: ['=']")
    offline_bundle_files = offline_bundle_file_identities()
    project_root_value = os.environ.get("MIQ_PROJECT_ROOT")
    if not project_root_value:
        raise RuntimeError("MIQ_PROJECT_ROOT is required for preprocessing provenance")
    build_runtime_provenance = runtime_provenance(Path(project_root_value).expanduser().resolve())

    temp_bundle = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.building-", dir=output_root))
    bundle_entries: dict[str, Any] = {}
    try:
        for family_index, task_family in enumerate(TASK_FAMILIES):
            task_config = config["task_families"][task_family]
            target_count = int(task_config["num_examples"])
            attempts = math.ceil(target_count * attempts_factor)
            specs = _specifications(
                pool_rows,
                task_family,
                task_config,
                source_revision=source_revision,
                preprocessing_hash=preprocessing_hash,
                seed=int(config["seed"]) + 1009 * family_index,
                attempts=attempts,
                enable_random_phrasing=random_phrasing,
            )
            records, drops = _generate_family(specs, target_count, workers, random_phrasing)
            artifact_id = f"miq-train-{task_family}-{preprocessing_hash[:12]}-{source_revision[:12]}"
            artifact_root = temp_bundle / artifact_id
            artifact_root.mkdir()
            Dataset.from_list(records).save_to_disk(str(artifact_root / "dataset"))
            manifest = {
                "dataset_artifact_id": artifact_id,
                "created_at": utc_now(),
                "task_family": task_family,
                "source_dataset": TRAIN_POOL_ID,
                "source_dataset_revision": source_revision,
                "moleculariq_core_version": metadata.version("moleculariq-core"),
                "moleculariq_core_commit": MOLECULARIQ_CORE_COMMIT,
                "preprocessing_config": config_for_hash,
                "preprocessing_config_hash": preprocessing_hash,
                "random_seed": int(config["seed"]) + 1009 * family_index,
                "num_molecules": len(pool_rows),
                "num_examples": len(records),
                "counts_by_task_type": dict(Counter(row["task_type"] for row in records)),
                "drop_counts_by_reason": dict(drops),
                "records_sha256": records_sha256(records),
                "dataset_fingerprint": None,
                "sampling_strategy": config.get("sampling_strategy"),
                "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
                "offline_bundle_files": offline_bundle_files,
                "build_runtime_provenance": build_runtime_provenance,
                "official_benchmark_used": False,
            }
            write_json_exclusive(artifact_root / "manifest.json", manifest)
            (artifact_root / "_READY").touch(exist_ok=False)
            reloaded = load_from_disk(str(artifact_root / "dataset"))
            manifest["dataset_fingerprint"] = getattr(reloaded, "_fingerprint", None)
            # Manifest is still in the private build directory; replace it before publication.
            (artifact_root / "manifest.json").unlink()
            write_json_exclusive(artifact_root / "manifest.json", manifest)
            validate_artifact(artifact_root, expected_task=task_family)
            bundle_entries[task_family] = {
                "artifact_id": artifact_id,
                "relative_path": artifact_id,
                "records_sha256": manifest["records_sha256"],
                "num_examples": manifest["num_examples"],
            }

        bundle_manifest = {
            "bundle_id": bundle_id,
            "created_at": utc_now(),
            "preprocessing_config_hash": preprocessing_hash,
            "source_dataset": TRAIN_POOL_ID,
            "source_dataset_revision": source_revision,
            "artifacts": bundle_entries,
            "offline_bundle_files": offline_bundle_files,
            "build_runtime_provenance": build_runtime_provenance,
            "official_benchmark_used": False,
        }
        write_json_exclusive(temp_bundle / "bundle_manifest.json", bundle_manifest)
        (temp_bundle / "_READY").touch(exist_ok=False)
        os.replace(temp_bundle, final_bundle)
    except BaseException:
        shutil.rmtree(temp_bundle, ignore_errors=True)
        raise
    return final_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--write-bundle-pointer", type=Path)
    args = parser.parse_args()
    bundle = build_bundle(args.config)
    if args.write_bundle_pointer:
        pointer = args.write_bundle_pointer.expanduser().resolve()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        with pointer.open("x", encoding="utf-8") as handle:
            handle.write(str(bundle) + "\n")
    print(bundle)
