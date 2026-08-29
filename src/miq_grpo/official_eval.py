"""Run one model on the whole pinned official MolecularIQ benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError, load_yaml, require_keys
from .constants import (
    MODEL_ID,
    MOLECULARIQ_EVAL_COMMIT,
    OFFICIAL_BENCHMARK_ID,
    OFFICIAL_TASK,
    OFFICIAL_TASK_YAML_SHA256,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    TASK_FAMILIES,
)
from .io_utils import (
    canonical_json,
    directory_identity,
    offline_bundle_file_identities,
    package_versions,
    read_json,
    records_sha256,
    runtime_provenance,
    sha256_file,
    source_tree_sha256,
    utc_now,
    write_json_exclusive,
)


class _TaggedSafeLoader(yaml.SafeLoader):
    pass


def _unknown_yaml(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node) -> Any:
    del tag_suffix
    return loader.construct_scalar(node)


_TaggedSafeLoader.add_multi_constructor("!", _unknown_yaml)


def _require_offline_eval() -> None:
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    missing = [key for key in required if os.environ.get(key) != "1"]
    if missing:
        raise RuntimeError(f"official evaluation requires enforced offline mode: {', '.join(missing)}")


def _assert_no_limit(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"limit", "samples", "task_subset"} and item not in (None, [], {}):
                raise ConfigError(f"full benchmark config forbids {path}.{key}")
            _assert_no_limit(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_limit(item, f"{path}[{index}]")


def _git_checkout_identity(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("official evaluator checkout is dirty; refuse unreportable benchmark run")
    return {
        "commit": commit,
        "clean": True,
        "source_tree_sha256": source_tree_sha256(repo),
    }


def _inspect_official_task(repo: Path, expected_repeats: int) -> dict[str, Any]:
    task_yaml = repo / "lm_eval" / "tasks" / "moleculariq" / "moleculariq_pass_at_k.yaml"
    task_yaml_hash = sha256_file(task_yaml)
    if task_yaml_hash != OFFICIAL_TASK_YAML_SHA256:
        raise RuntimeError("official task YAML bytes differ from the pinned benchmark protocol")
    with task_yaml.open("r", encoding="utf-8") as handle:
        task = yaml.load(handle, Loader=_TaggedSafeLoader)
    if task.get("task") != OFFICIAL_TASK:
        raise RuntimeError("official task YAML has the wrong task name")
    if task.get("dataset_path") != OFFICIAL_BENCHMARK_ID or task.get("test_split") != "test":
        raise RuntimeError("official task YAML no longer points to the whole held-out benchmark")
    if task.get("repeats") != expected_repeats:
        raise RuntimeError("official repeat count differs from the frozen evaluation protocol")
    if "limit" in task:
        raise RuntimeError("official task YAML unexpectedly contains a limit")
    if task.get("output_type") != "generate_until":
        raise RuntimeError("official task output type changed")
    expected_hooks = {
        "process_docs": "task_processor.process_docs",
        "doc_to_text": "task_processor.doc_to_text",
        "doc_to_target": "target",
        "process_results": "task_processor.process_results_pass_at_k",
    }
    if any(task.get(key) != value for key, value in expected_hooks.items()):
        raise RuntimeError("official task processing or scoring hooks changed")
    expected_generation = {"max_tokens": 32768, "until": [], "do_sample": True}
    if task.get("generation_kwargs") != expected_generation:
        raise RuntimeError("official task generation defaults changed")
    return {
        "path": str(task_yaml),
        "sha256": task_yaml_hash,
        "contents": task,
        "hooks": expected_hooks,
        "generation_kwargs": expected_generation,
    }


def _required_file_identity(environment_key: str) -> dict[str, Any]:
    value = os.environ.get(environment_key)
    if not value:
        raise RuntimeError(f"{environment_key} is required for environment provenance")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{environment_key} does not point to a file: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _project_root() -> Path:
    value = os.environ.get("MIQ_PROJECT_ROOT")
    if not value:
        raise RuntimeError("MIQ_PROJECT_ROOT is required for evaluation provenance")
    root = Path(value).expanduser().resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src" / "miq_grpo").is_dir():
        raise RuntimeError(f"MIQ_PROJECT_ROOT is not this MolecularIQ project: {root}")
    return root


def _verify_base_model_manifest() -> dict[str, Any]:
    value = os.environ.get("MIQ_BASE_MODEL_MANIFEST")
    if not value:
        raise RuntimeError("MIQ_BASE_MODEL_MANIFEST is required")
    path = Path(value).expanduser().resolve()
    manifest = read_json(path)
    if not (path.parent / "_READY").is_file():
        raise RuntimeError("base-model assets are not marked ready")
    if manifest.get("model_id") != MODEL_ID or not manifest.get("model_identity", {}).get("tree_sha256"):
        raise RuntimeError("base-model manifest is incomplete or for the wrong model")
    return manifest


def _verify_sample_documents(sample_files: list[Path], assets_manifest: dict[str, Any]) -> dict[str, Any]:
    documents: dict[int, dict[str, Any]] = {}
    for path in sample_files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                doc_id = row.get("doc_id")
                doc = row.get("doc")
                if type(doc_id) is not int or not isinstance(doc, dict) or doc_id in documents:
                    raise RuntimeError("official sample logs contain malformed or duplicate documents")
                documents[doc_id] = doc
    expected_count = int(assets_manifest["test_rows"])
    if sorted(documents) != list(range(expected_count)):
        raise RuntimeError("official sample logs do not contain every benchmark document exactly once")
    content_hash = records_sha256(documents[index] for index in range(expected_count))
    if content_hash != assets_manifest.get("test_records_sha256"):
        raise RuntimeError("evaluated documents differ from the staged official benchmark snapshot")
    return {"num_documents": expected_count, "records_sha256": content_hash}


def _verify_evaluation_assets() -> dict[str, Any]:
    value = os.environ.get("MIQ_EVAL_ASSETS_MANIFEST")
    if not value:
        raise RuntimeError("MIQ_EVAL_ASSETS_MANIFEST is required")
    path = Path(value).expanduser().resolve()
    manifest = read_json(path)
    if not (path.parent / "_READY").is_file():
        raise RuntimeError("evaluation assets are not ready")
    if (
        manifest.get("dataset_id") != OFFICIAL_BENCHMARK_ID
        or manifest.get("evaluation_only") is not True
        or manifest.get("training_pool_present") is not False
    ):
        raise RuntimeError("evaluation assets violate the held-out benchmark boundary")
    if int(manifest.get("test_rows", -1)) != 5111:
        raise RuntimeError("official benchmark row count changed; inspect upstream before proceeding")
    if not isinstance(manifest.get("test_records_sha256"), str):
        raise RuntimeError("evaluation assets lack the canonical benchmark content hash")
    return manifest


def run_official_evaluation(config_path: Path, model_key: str) -> Path:
    _require_offline_eval()
    config = load_yaml(config_path)
    require_keys(
        config,
        ("evaluation_suite_id", "official_evaluator", "models", "output_root", "benchmark_integrity"),
        context="evaluation config",
    )
    _assert_no_limit(config)
    evaluator = config["official_evaluator"]
    if (
        evaluator.get("expected_commit") != MOLECULARIQ_EVAL_COMMIT
        or evaluator.get("task") != OFFICIAL_TASK
        or evaluator.get("dataset_id") != OFFICIAL_BENCHMARK_ID
        or evaluator.get("full_benchmark") is not True
        or evaluator.get("apply_chat_template") is not True
    ):
        raise ConfigError("evaluation is locked to the full official MolecularIQ benchmark")
    expected_models = {"baseline", "count_grpo", "index_grpo", "constraint_grpo"}
    if set(config["models"]) != expected_models or model_key not in expected_models:
        raise ConfigError(f"unknown model key: {model_key}")
    if any(config["benchmark_integrity"].values()):
        raise ConfigError("evaluation config does not describe a clean held-out comparison")

    repo = Path(evaluator["repo_path"]).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"official evaluator checkout does not exist: {repo}")
    evaluator_identity = _git_checkout_identity(repo)
    if evaluator_identity["commit"] != MOLECULARIQ_EVAL_COMMIT:
        raise RuntimeError(f"official evaluator must be checked out at {MOLECULARIQ_EVAL_COMMIT}")
    task_info = _inspect_official_task(repo, int(evaluator["expected_repeats"]))
    assets_manifest = _verify_evaluation_assets()
    base_model_manifest = _verify_base_model_manifest()
    environment_lock = offline_bundle_file_identities()

    model_entry = config["models"][model_key]
    checkpoint = Path(model_entry["checkpoint_path"]).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    checkpoint_identity = directory_identity(checkpoint)
    if model_key == "baseline":
        if (
            Path(base_model_manifest["model_path"]).resolve() != checkpoint
            or checkpoint_identity["tree_sha256"] != base_model_manifest["model_identity"]["tree_sha256"]
            or model_entry.get("task_family") is not None
        ):
            raise RuntimeError("baseline checkpoint differs from the staged immutable Qwen snapshot")
        training_binding = {
            "kind": "baseline",
            "base_model_id": MODEL_ID,
            "base_model_revision": base_model_manifest["resolved_model_revision"],
            "base_model_tree_sha256": checkpoint_identity["tree_sha256"],
            "base_model_assets_manifest": _required_file_identity("MIQ_BASE_MODEL_MANIFEST"),
        }
    else:
        training_root = checkpoint.parent
        completion_manifest = training_root / "training_complete.json"
        if not completion_manifest.is_file():
            raise RuntimeError(f"trained checkpoint lacks training_complete.json: {checkpoint}")
        training_complete = read_json(completion_manifest)
        if training_complete.get("smoke_run") is not False:
            raise RuntimeError("smoke checkpoints cannot be used for reportable official evaluation")
        if training_complete.get("experiment_id") != model_entry["experiment_id"]:
            raise RuntimeError("checkpoint experiment ID differs from evaluation registry")
        if training_complete.get("task_family") != model_entry.get("task_family"):
            raise RuntimeError("checkpoint task family differs from evaluation registry")
        if model_entry.get("task_family") not in TASK_FAMILIES:
            raise RuntimeError("trained evaluation entry lacks a valid single-task family")
        if Path(training_complete.get("final_model", "")).resolve() != checkpoint:
            raise RuntimeError("training completion manifest points to a different final model")
        if (
            training_complete.get("final_model_identity", {}).get("tree_sha256")
            != checkpoint_identity["tree_sha256"]
        ):
            raise RuntimeError("final model files differ from the recorded training output")
        if (
            training_complete.get("base_model_id") != MODEL_ID
            or training_complete.get("base_model_revision") != base_model_manifest["resolved_model_revision"]
            or training_complete.get("base_model_tree_sha256")
            != base_model_manifest["model_identity"]["tree_sha256"]
        ):
            raise RuntimeError("trained checkpoint is not descended from the staged baseline")
        frozen_path = training_root / "frozen_config.yaml"
        provenance_path = training_root / "provenance.json"
        frozen = load_yaml(frozen_path, expand_environment=False)
        frozen.pop("_config_path", None)
        training_provenance = read_json(provenance_path)
        if (
            frozen.get("experiment_id") != training_complete.get("experiment_id")
            or frozen.get("task_family") != training_complete.get("task_family")
            or frozen.get("model", {}).get("expected_revision")
            != training_complete.get("base_model_revision")
            or frozen.get("dataset", {}).get("artifact_id") != training_complete.get("dataset_artifact_id")
            or frozen.get("dataset", {}).get("records_sha256")
            != training_complete.get("dataset_records_sha256")
        ):
            raise RuntimeError("training completion manifest differs from its frozen launched config")
        if (
            training_provenance.get("experiment_id") != training_complete.get("experiment_id")
            or training_provenance.get("task_family") != training_complete.get("task_family")
            or training_provenance.get("dataset_artifact_id") != training_complete.get("dataset_artifact_id")
            or training_provenance.get("dataset_records_sha256")
            != training_complete.get("dataset_records_sha256")
            or training_provenance.get("model_revision") != training_complete.get("base_model_revision")
            or training_provenance.get("base_model_tree_sha256")
            != training_complete.get("base_model_tree_sha256")
            or training_provenance.get("offline_bundle_files") != environment_lock
        ):
            raise RuntimeError("training provenance differs from its completed model")
        training_binding = {
            "kind": "trained",
            "task_family": training_complete["task_family"],
            "dataset_artifact_id": training_complete["dataset_artifact_id"],
            "dataset_records_sha256": training_complete["dataset_records_sha256"],
            "base_model_revision": training_complete["base_model_revision"],
            "base_model_tree_sha256": training_complete["base_model_tree_sha256"],
            "training_complete": _file_identity(completion_manifest),
            "frozen_config": _file_identity(frozen_path),
            "launch_provenance": _file_identity(provenance_path),
            "resume_provenance": [
                _file_identity(path) for path in sorted(training_root.glob("resume_provenance_*.json"))
            ],
        }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    job_component = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    run_id = f"{config['evaluation_suite_id']}-{model_key}-{job_component}-{timestamp}"
    run_root = Path(config["output_root"]).expanduser().resolve() / model_key / run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite evaluation run: {run_root}")
    raw_root = run_root / "raw"
    raw_root.mkdir(parents=True)

    generation_overrides = evaluator["generation"]
    effective_generation = dict(task_info["generation_kwargs"])
    effective_generation.update(
        {key: value for key, value in generation_overrides.items() if value is not None}
    )
    gen_kwargs = ",".join(
        f"{key}={value}" for key, value in generation_overrides.items() if value is not None
    )
    model_args = ",".join(
        (
            f"pretrained={checkpoint}",
            f"dtype={evaluator['dtype']}",
            "trust_remote_code=False",
        )
    )
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        evaluator["model_backend"],
        "--model_args",
        model_args,
        "--tasks",
        OFFICIAL_TASK,
        "--apply_chat_template",
        "--system_instruction",
        SYSTEM_PROMPT,
        "--batch_size",
        str(evaluator["batch_size"]),
        "--gen_kwargs",
        gen_kwargs,
        "--log_samples",
        "--output_path",
        str(raw_root),
        "--seed",
        "0,1234,1234,1234",
    ]
    resolved_config = {key: value for key, value in config.items() if key != "_config_path"}
    manifest = {
        "evaluation_run_id": run_id,
        "evaluation_suite_id": config["evaluation_suite_id"],
        "model_key": model_key,
        "experiment_id": model_entry["experiment_id"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_identity": checkpoint_identity,
        "training_binding": training_binding,
        "official_evaluator_commit": MOLECULARIQ_EVAL_COMMIT,
        "official_evaluator_identity": evaluator_identity,
        "official_task": OFFICIAL_TASK,
        "official_task_yaml_sha256": task_info["sha256"],
        "official_task_hooks": task_info["hooks"],
        "official_benchmark_id": OFFICIAL_BENCHMARK_ID,
        "official_benchmark_revision": assets_manifest["dataset_revision"],
        "official_benchmark_records_sha256": assets_manifest["test_records_sha256"],
        "full_benchmark": True,
        "limit": None,
        "repeat_count": evaluator["expected_repeats"],
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "system_prompt": SYSTEM_PROMPT,
        "backend": evaluator["model_backend"],
        "dtype": evaluator["dtype"],
        "official_task_generation_kwargs": task_info["generation_kwargs"],
        "generation_overrides": generation_overrides,
        "effective_generation_kwargs": effective_generation,
        "generation": effective_generation,
        "seeds": [0, 1234, 1234, 1234],
        "command": command,
        "resolved_config_sha256": hashlib.sha256(canonical_json(resolved_config).encode("utf-8")).hexdigest(),
        "package_versions": package_versions(
            ("torch", "transformers", "datasets", "moleculariq-core", "lm_eval")
        ),
        "installed_miq_grpo_tree_sha256": source_tree_sha256(Path(__file__).resolve().parent),
        "environment_lock": environment_lock,
        "runtime_provenance": runtime_provenance(_project_root()),
        "benchmark_used_for_training": False,
        "benchmark_used_for_hparam_selection": False,
        "benchmark_used_for_prompt_tuning": False,
        "benchmark_used_for_checkpoint_selection": False,
        "started_at": utc_now(),
    }
    write_json_exclusive(run_root / "eval_manifest.json", manifest)
    with (
        (run_root / "stdout.log").open("x", encoding="utf-8") as stdout_handle,
        (run_root / "stderr.log").open("x", encoding="utf-8") as stderr_handle,
    ):
        completed = subprocess.run(
            command,
            cwd=repo,
            env=dict(os.environ),
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            text=True,
        )
    evaluator_identity_after = _git_checkout_identity(repo)
    if evaluator_identity_after != evaluator_identity:
        write_json_exclusive(
            run_root / "evaluation_failed.json",
            {
                "failed_at": utc_now(),
                "reason": "official evaluator source identity changed during the run",
            },
        )
        raise RuntimeError("official evaluator source identity changed during the run")
    if completed.returncode != 0:
        write_json_exclusive(
            run_root / "evaluation_failed.json",
            {"failed_at": utc_now(), "returncode": completed.returncode},
        )
        raise RuntimeError(f"official evaluator failed with exit code {completed.returncode}")

    result_files = sorted(raw_root.rglob("results*.json"))
    sample_files = sorted(raw_root.rglob("samples*.jsonl"))
    if not result_files or not sample_files:
        raise RuntimeError("official evaluator completed without both result and sample files")
    evaluated_documents = _verify_sample_documents(sample_files, assets_manifest)
    completion = {
        "completed_at": utc_now(),
        "result_files": [
            {"path": str(path.relative_to(run_root)), "sha256": sha256_file(path)} for path in result_files
        ],
        "sample_files": [
            {"path": str(path.relative_to(run_root)), "sha256": sha256_file(path)} for path in sample_files
        ],
        "log_files": [
            {"path": name, "sha256": sha256_file(run_root / name)} for name in ("stdout.log", "stderr.log")
        ],
        "evaluated_documents": evaluated_documents,
        "official_evaluator_identity_after": evaluator_identity_after,
    }
    write_json_exclusive(run_root / "evaluation_complete.json", completion)
    (run_root / "_COMPLETE").touch(exist_ok=False)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    args = parser.parse_args()
    print(run_official_evaluation(args.config.expanduser().resolve(), args.model_key))


if __name__ == "__main__":
    main()
