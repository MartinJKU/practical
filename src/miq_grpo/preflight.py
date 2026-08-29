"""Cheap gates to run before spending Leonardo GPU hours."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .constants import (
    MODEL_ID,
    MOLECULARIQ_CORE_COMMIT,
    MOLECULARIQ_EVAL_COMMIT,
    OFFICIAL_BENCHMARK_ID,
    OFFICIAL_TASK,
    OFFICIAL_TASK_YAML_SHA256,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    TASK_FAMILIES,
    TRL_VERSION,
)
from .dataset_builder import assert_moleculariq_api
from .dataset_validation import resolve_bundle_artifact, validate_artifact
from .io_utils import directory_identity, read_json, records_sha256, sha256_file, source_tree_sha256
from .rewards import score_completion

EXPECTED_VERSIONS = {
    "trl": "1.12.0",
    "transformers": "5.16.1",
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
    "rdkit": "2026.3.5",
    "torch": "2.5.1+cu121",
}


def environment_preflight(*, require_cuda: bool) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package, expected in EXPECTED_VERSIONS.items():
        actual = metadata.version(package)
        versions[package] = actual
        if actual != expected:
            raise RuntimeError(f"expected {package}=={expected}, got {actual}")
    core_version = metadata.version("moleculariq-core")
    assert_moleculariq_api()

    from trl import GRPOConfig, GRPOTrainer

    reward_required = {"reward_funcs", "train_dataset", "processing_class"}
    if not reward_required <= set(inspect.signature(GRPOTrainer.__init__).parameters):
        raise RuntimeError("installed GRPOTrainer signature is incompatible")
    config_fields = set(GRPOConfig.__dataclass_fields__)
    required_config = {
        "num_generations",
        "max_completion_length",
        "remove_unused_columns",
        "loss_type",
        "mask_truncated_completions",
    }
    if not required_config <= config_fields:
        raise RuntimeError("installed GRPOConfig lacks required fields")

    import torch

    cuda = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "torch_cuda": torch.version.cuda,
    }
    if require_cuda and (not cuda["available"] or cuda["device_count"] < 1):
        raise RuntimeError("CUDA preflight requested but no GPU is visible")
    if require_cuda and torch.version.cuda != "12.1":
        raise RuntimeError(f"expected the Leonardo-compatible CUDA 12.1 wheel, got {torch.version.cuda}")
    return {
        "versions": versions,
        "moleculariq_core_version": core_version,
        "moleculariq_core_expected_commit": MOLECULARIQ_CORE_COMMIT,
        "trl_expected_version": TRL_VERSION,
        "cuda": cuda,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def reward_oracle_preflight() -> dict[str, Any]:
    cases = {
        "count_correct": score_completion(
            '<answer>{"ring_count": 1}</answer>',
            task_family="count",
            task_type="single_count",
            target_json='{"ring_count":1}',
            constraints_json="[]",
            validity_weight=0.0,
        ),
        "count_wrong": score_completion(
            '<answer>{"ring_count": 2}</answer>',
            task_family="count",
            task_type="single_count",
            target_json='{"ring_count":1}',
            constraints_json="[]",
            validity_weight=0.0,
        ),
        "index_correct": score_completion(
            '<answer>{"carbon_atom_index": [0, 1]}</answer>',
            task_family="index",
            task_type="single_index_identification",
            target_json='{"carbon_atom_index":[0,1]}',
            constraints_json="[]",
            validity_weight=0.0,
        ),
        "index_off_by_one": score_completion(
            '<answer>{"carbon_atom_index": [1, 2]}</answer>',
            task_family="index",
            task_type="single_index_identification",
            target_json='{"carbon_atom_index":[0,1]}',
            constraints_json="[]",
            validity_weight=0.0,
        ),
        "constraint_satisfying": score_completion(
            '<answer>{"smiles":"c1ccccc1"}</answer>',
            task_family="constraint",
            task_type="constraint_generation",
            target_json="{}",
            constraints_json='[{"type":"ring_count","operator":"=","value":1}]',
        ),
        "constraint_violating": score_completion(
            '<answer>{"smiles":"CCO"}</answer>',
            task_family="constraint",
            task_type="constraint_generation",
            target_json="{}",
            constraints_json='[{"type":"ring_count","operator":"=","value":1}]',
        ),
        "constraint_invalid": score_completion(
            '<answer>{"smiles":"not_a_smiles"}</answer>',
            task_family="constraint",
            task_type="constraint_generation",
            target_json="{}",
            constraints_json='[{"type":"ring_count","operator":"=","value":1}]',
        ),
    }
    expectations = {
        "count_correct": 1.0,
        "count_wrong": 0.0,
        "index_correct": 1.0,
        "index_off_by_one": 0.0,
        "constraint_satisfying": 1.0,
        "constraint_violating": 0.0,
        "constraint_invalid": 0.0,
    }
    for name, expected in expectations.items():
        if cases[name].correctness != expected:
            raise RuntimeError(f"reward oracle failed {name}: {cases[name]}")
    return {
        name: {
            "total": result.total,
            "correctness": result.correctness,
            "format": result.format_reward,
            "validity": result.validity,
            "status": result.parsed.status.value,
        }
        for name, result in cases.items()
    }


def checkpoint_preflight(path: Path, prompt: str, reference_path: Path | None = None) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    completion = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    result: dict[str, Any] = {
        "checkpoint": str(path.resolve()),
        "completion": completion,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "padding_side": tokenizer.padding_side,
    }
    if reference_path is not None:
        reference = reference_path.expanduser().resolve()
        if not reference.is_dir() or reference == path.resolve():
            raise RuntimeError("parameter-change preflight requires a distinct reference checkpoint")
        reference_model = AutoModelForCausalLM.from_pretrained(
            reference,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        reference_parameters = dict(reference_model.named_parameters())
        changed_name: str | None = None
        maximum_delta = 0.0
        with torch.inference_mode():
            for name, trained_parameter in model.named_parameters():
                reference_parameter = reference_parameters.get(name)
                if reference_parameter is None or reference_parameter.shape != trained_parameter.shape:
                    raise RuntimeError(f"reference checkpoint parameter mismatch: {name}")
                delta = float((trained_parameter.detach() - reference_parameter.detach()).abs().max().item())
                if not math.isfinite(delta):
                    raise RuntimeError(f"non-finite trained/reference parameter delta: {name}")
                if delta > 0.0:
                    changed_name = name
                    maximum_delta = delta
                    break
        if changed_name is None:
            raise RuntimeError("trained checkpoint has no changed parameters relative to the baseline")
        result["parameter_change"] = {
            "reference_checkpoint": str(reference),
            "changed_parameter": changed_name,
            "max_abs_delta": maximum_delta,
        }
    return result


def smoke_metrics_preflight(run_dir: Path, task_family: str) -> dict[str, Any]:
    """Require a completed, finite smoke run with usable GRPO signal."""
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"unknown task family: {task_family}")
    root = run_dir.expanduser().resolve()
    completion_path = root / "training_complete.json"
    metrics_path = root / "logs" / "metrics.jsonl"
    if not completion_path.is_file() or not metrics_path.is_file():
        raise RuntimeError(f"smoke run lacks training_complete.json or metrics.jsonl: {root}")

    completion = read_json(completion_path)
    if completion.get("smoke_run") is not True or completion.get("task_family") != task_family:
        raise RuntimeError("training completion manifest is not for the requested smoke task")
    if int(completion.get("global_step", 0)) < 1:
        raise RuntimeError("smoke training did not complete an optimizer step")
    training_loss = completion.get("training_loss")
    if isinstance(training_loss, bool) or not isinstance(training_loss, (int, float)):
        raise RuntimeError("smoke training did not record a numeric training loss")
    if not math.isfinite(float(training_loss)):
        raise RuntimeError("smoke training loss is not finite")

    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"metrics line {line_number} is not an object")
            for key, value in row.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"non-finite smoke metric {key!r} on line {line_number}")
            rows.append(row)
    if not rows:
        raise RuntimeError("smoke metrics log is empty")

    zero_variance_values = [
        float(value)
        for row in rows
        for key, value in row.items()
        if key.endswith("grpo/zero_variance_group_fraction")
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
    ]
    if not zero_variance_values:
        raise RuntimeError("smoke metrics lack grpo/zero_variance_group_fraction")
    if not any(0.0 <= value < 1.0 for value in zero_variance_values):
        raise RuntimeError(
            f"{task_family} smoke rollout has no within-group reward variance; refusing to scale training"
        )
    return {
        "run_dir": str(root),
        "task_family": task_family,
        "global_step": int(completion["global_step"]),
        "training_loss": float(training_loss),
        "metric_rows": len(rows),
        "minimum_zero_variance_group_fraction": min(zero_variance_values),
        "usable_reward_variance_observed": True,
    }


def _inspect_evaluation_task_yaml(task_yaml: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    import yaml

    class TaggedSafeLoader(yaml.SafeLoader):
        pass

    def unknown_tag(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node) -> Any:
        del tag_suffix
        return loader.construct_scalar(node)

    TaggedSafeLoader.add_multi_constructor("!", unknown_tag)
    task_yaml_hash = sha256_file(task_yaml)
    if expected_sha256 is not None and task_yaml_hash != expected_sha256:
        raise RuntimeError("official MolecularIQ task YAML differs from its pinned checksum")
    with task_yaml.open("r", encoding="utf-8") as handle:
        task = yaml.load(handle, Loader=TaggedSafeLoader)
    if not isinstance(task, dict):
        raise RuntimeError("official MolecularIQ task YAML is not a mapping")
    expected = {
        "task": OFFICIAL_TASK,
        "dataset_path": OFFICIAL_BENCHMARK_ID,
        "dataset_name": "default",
        "test_split": "test",
        "output_type": "generate_until",
        "process_docs": "task_processor.process_docs",
        "doc_to_text": "task_processor.doc_to_text",
        "doc_to_target": "target",
        "process_results": "task_processor.process_results_pass_at_k",
        "repeats": 3,
    }
    mismatches = {key: task.get(key) for key, value in expected.items() if task.get(key) != value}
    if mismatches or "limit" in task:
        raise RuntimeError(f"official MolecularIQ task YAML contract changed: {mismatches}")
    metric_names = {entry.get("metric") for entry in task.get("metric_list", []) if isinstance(entry, dict)}
    if not {"pass_at_1", "pass_at_3", "avg_accuracy"} <= metric_names:
        raise RuntimeError("official MolecularIQ task YAML lacks required metrics")
    generation = task.get("generation_kwargs", {})
    if generation != {"max_tokens": 32768, "until": [], "do_sample": True}:
        raise RuntimeError("official MolecularIQ task generation contract changed")
    return {"path": str(task_yaml), "sha256": task_yaml_hash, "repeats": task["repeats"]}


def _validate_cached_benchmark_rows(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    expected_count = int(manifest.get("test_rows", -1))
    if len(rows) != expected_count or expected_count != 5111:
        raise RuntimeError(f"cached benchmark length mismatch: expected {expected_count}, loaded {len(rows)}")
    actual_hash = records_sha256(rows)
    expected_hash = manifest.get("test_records_sha256")
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise RuntimeError("cached benchmark content hash differs from its staged manifest")
    return actual_hash


def evaluation_infrastructure_preflight(
    evaluator_repo: Path, assets_manifest_path: Path, baseline_path: Path
) -> dict[str, Any]:
    """Load official evaluation infrastructure without generating or scoring answers."""
    offline_flags = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    missing = [key for key in offline_flags if os.environ.get(key) != "1"]
    if missing:
        raise RuntimeError(f"evaluation infrastructure preflight requires offline mode: {missing}")

    repo = evaluator_repo.expanduser().resolve()
    manifest_path = assets_manifest_path.expanduser().resolve()
    baseline = baseline_path.expanduser().resolve()
    if not repo.is_dir() or not manifest_path.is_file() or not baseline.is_dir():
        raise FileNotFoundError("evaluator repo, evaluation manifest, or baseline checkpoint is missing")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != MOLECULARIQ_EVAL_COMMIT:
        raise RuntimeError(f"official evaluator checkout must be {MOLECULARIQ_EVAL_COMMIT}, got {commit}")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("official evaluator checkout is dirty")
    evaluator_source_hash = source_tree_sha256(repo)

    manifest = read_json(manifest_path)
    if not (manifest_path.parent / "_READY").is_file():
        raise RuntimeError("official evaluation assets are not marked ready")
    if (
        manifest.get("dataset_id") != OFFICIAL_BENCHMARK_ID
        or manifest.get("evaluation_only") is not True
        or manifest.get("training_pool_present") is not False
    ):
        raise RuntimeError("staged evaluation manifest violates the held-out boundary")
    expected_hf_home = Path(str(manifest.get("hf_home", ""))).resolve()
    active_hf_home = Path(os.environ.get("HF_HOME", "")).resolve()
    if expected_hf_home != active_hf_home:
        raise RuntimeError("active evaluation HF cache differs from the staged manifest")

    bundle_value = os.environ.get("MIQ_OFFLINE_BUNDLE_ROOT")
    if not bundle_value:
        raise RuntimeError("MIQ_OFFLINE_BUNDLE_ROOT is required to identify the staged baseline")
    training_assets_root = Path(bundle_value).expanduser().resolve() / "training_assets"
    training_manifest_path = training_assets_root / "assets_manifest.json"
    if not training_manifest_path.is_file() or not (training_assets_root / "_READY").is_file():
        raise RuntimeError("staged training assets are unavailable for baseline identity verification")
    training_manifest = read_json(training_manifest_path)
    if (
        training_manifest.get("model_id") != MODEL_ID
        or training_manifest.get("official_benchmark_staged") is not False
        or Path(str(training_manifest.get("model_path", ""))).resolve() != baseline
        or training_manifest.get("model_identity") != directory_identity(baseline)
    ):
        raise RuntimeError("baseline path is not the staged Qwen training baseline")

    task_yaml = repo / "lm_eval" / "tasks" / "moleculariq" / "moleculariq_pass_at_k.yaml"
    task_info = _inspect_evaluation_task_yaml(task_yaml, OFFICIAL_TASK_YAML_SHA256)
    sys.path.insert(0, str(repo))
    try:
        lm_eval = importlib.import_module("lm_eval")
        task_processor = importlib.import_module("lm_eval.tasks.moleculariq.task_processor")
        from lm_eval.tasks import TaskManager

        discovered = TaskManager().match_tasks([OFFICIAL_TASK])
    finally:
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)
    lm_eval_path = Path(lm_eval.__file__).resolve()
    if repo not in lm_eval_path.parents:
        raise RuntimeError(f"lm_eval imported outside the pinned evaluator checkout: {lm_eval_path}")
    if OFFICIAL_TASK not in discovered:
        raise RuntimeError("pinned lm-eval cannot discover the official MolecularIQ task")
    imported_prompt = getattr(task_processor, "SYSTEM_PROMPT", None)
    if imported_prompt != SYSTEM_PROMPT:
        raise RuntimeError("official evaluator system prompt differs from the training/evaluation prompt")
    if not callable(getattr(task_processor, "process_results_pass_at_k", None)):
        raise RuntimeError("official evaluator scoring hook is unavailable")

    from datasets import load_dataset

    benchmark = load_dataset(
        OFFICIAL_BENCHMARK_ID,
        split="test",
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    benchmark_rows = [dict(row) for row in benchmark]
    benchmark_hash = _validate_cached_benchmark_rows(benchmark_rows, manifest)
    del benchmark, benchmark_rows

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(baseline, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        baseline,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    if not tokenizer.chat_template or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("staged baseline tokenizer lacks its chat template, EOS, or pad token")
    fixed_question = "Return the requested ring_count as JSON."
    processed_question = task_processor.doc_to_text({"question": fixed_question})
    if processed_question != fixed_question:
        raise RuntimeError("official evaluator changed the model-visible MolecularIQ question")
    fixed_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": processed_question},
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        fixed_messages, tokenize=False, add_generation_prompt=True
    )
    if fixed_question not in rendered_prompt or SYSTEM_PROMPT not in rendered_prompt:
        raise RuntimeError("Qwen chat template omitted the fixed system prompt or question")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count < 1:
        raise RuntimeError("staged baseline model has no parameters")
    model_type = str(getattr(model.config, "model_type", ""))
    tokenizer_identity = {
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
    }
    del model, tokenizer
    return {
        "offline_mode": True,
        "official_evaluator_commit": commit,
        "official_evaluator_source_tree_sha256": evaluator_source_hash,
        "lm_eval_import_path": str(lm_eval_path),
        "official_task_discovered": True,
        "official_task_yaml": task_info,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "qwen_prompt_regression_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
        "tokenizer": tokenizer_identity,
        "benchmark_rows": int(manifest["test_rows"]),
        "benchmark_records_sha256": benchmark_hash,
        "benchmark_outcomes_read_or_scored": False,
        "baseline_checkpoint": str(baseline),
        "baseline_revision": training_manifest.get("resolved_model_revision"),
        "baseline_model_type": model_type,
        "baseline_parameter_count": parameter_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    environment_parser = subparsers.add_parser("environment")
    environment_parser.add_argument("--require-cuda", action="store_true")
    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument("--bundle", type=Path, required=True)
    dataset_parser.add_argument("--task", choices=TASK_FAMILIES, required=True)
    subparsers.add_parser("rewards")
    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--path", type=Path, required=True)
    checkpoint_parser.add_argument("--reference-path", type=Path)
    checkpoint_parser.add_argument("--prompt", default='Return <answer>{"ok": 1}</answer>.')
    smoke_metrics_parser = subparsers.add_parser("smoke-metrics")
    smoke_metrics_parser.add_argument("--run-dir", type=Path, required=True)
    smoke_metrics_parser.add_argument("--task", choices=TASK_FAMILIES, required=True)
    evaluation_parser = subparsers.add_parser("evaluation-infrastructure")
    evaluation_parser.add_argument("--evaluator-repo", type=Path, required=True)
    evaluation_parser.add_argument("--assets-manifest", type=Path, required=True)
    evaluation_parser.add_argument("--baseline-path", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "environment":
        result = environment_preflight(require_cuda=args.require_cuda)
    elif args.command == "dataset":
        artifact = resolve_bundle_artifact(args.bundle, args.task)
        result = validate_artifact(artifact, expected_task=args.task)
    elif args.command == "rewards":
        result = reward_oracle_preflight()
    elif args.command == "checkpoint":
        result = checkpoint_preflight(
            args.path.expanduser().resolve(),
            args.prompt,
            args.reference_path.expanduser().resolve() if args.reference_path else None,
        )
    elif args.command == "smoke-metrics":
        result = smoke_metrics_preflight(args.run_dir, args.task)
    elif args.command == "evaluation-infrastructure":
        result = evaluation_infrastructure_preflight(
            args.evaluator_repo, args.assets_manifest, args.baseline_path
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
