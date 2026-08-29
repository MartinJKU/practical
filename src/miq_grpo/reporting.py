"""Aggregate four official runs and create report-ready tables and plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import load_yaml
from .constants import (
    MOLECULARIQ_EVAL_COMMIT,
    OFFICIAL_TASK,
    OFFICIAL_TASK_YAML_SHA256,
    SYSTEM_PROMPT_SHA256,
)
from .io_utils import (
    canonical_json,
    directory_identity,
    offline_bundle_file_identities,
    read_json,
    runtime_provenance,
    sha256_file,
    utc_now,
    write_json_exclusive,
)

MODEL_ORDER = ("baseline", "count_grpo", "index_grpo", "constraint_grpo")
MODEL_LABELS = {
    "baseline": "Baseline",
    "count_grpo": "Count GRPO",
    "index_grpo": "Index GRPO",
    "constraint_grpo": "Constraint GRPO",
}
COLORS = {
    "baseline": "#6B7280",
    "count_grpo": "#2563EB",
    "index_grpo": "#D97706",
    "constraint_grpo": "#059669",
}


def _completed_run(results_root: Path, model_key: str) -> Path:
    root = results_root / model_key
    candidates = sorted(path.parent for path in root.glob("*/_COMPLETE")) if root.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one completed official run for {model_key}, found {len(candidates)}; "
            "move reruns aside instead of selecting by benchmark outcome"
        )
    return candidates[0]


def _result_metrics(run_root: Path) -> dict[str, float]:
    files = sorted((run_root / "raw").rglob("results*.json"))
    for path in files:
        payload = read_json(path)
        results = payload.get("results", {})
        task = results.get(OFFICIAL_TASK)
        if isinstance(task, dict):
            required = ("pass_at_1", "pass_at_3", "avg_accuracy")
            metrics: dict[str, float] = {}
            for metric in required:
                value = task.get(f"{metric},all")
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    break
                metrics[metric] = float(value)
            if len(metrics) == len(required):
                return metrics
    raise RuntimeError(f"no official MolecularIQ metric payload found in {run_root}")


def _task_family(task_type: str) -> str:
    lowered = task_type.lower()
    if "constraint" in lowered or "generation" in lowered:
        return "constraint"
    if "index" in lowered:
        return "index"
    if "count" in lowered:
        return "count"
    return "unknown"


def _sample_metrics(run_root: Path) -> tuple[dict[str, float], dict[str, float]]:
    family_values: dict[str, list[float]] = defaultdict(list)
    supercategory_values: dict[str, list[float]] = defaultdict(list)
    for path in sorted((run_root / "raw").rglob("samples*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                doc = row.get("doc", {})
                # lm-eval SampleResult serializes the metric names under
                # ``metrics`` and each computed value as a top-level field.
                accuracy = row.get("avg_accuracy")
                if (
                    not isinstance(accuracy, (int, float))
                    or not math.isfinite(float(accuracy))
                    or not 0.0 <= float(accuracy) <= 1.0
                ):
                    raise RuntimeError(f"sample logs in {run_root} contain invalid accuracy")
                family = _task_family(str(doc.get("task_type", "")))
                if family == "unknown":
                    raise RuntimeError(f"sample logs in {run_root} contain an unknown task family")
                family_values[family].append(float(accuracy))
                supercategory = str(doc.get("supercategory", "unknown"))
                if not supercategory or supercategory == "unknown":
                    raise RuntimeError(f"sample logs in {run_root} contain an unknown supercategory")
                supercategory_values[supercategory].append(float(accuracy))
    if not all(family_values.get(family) for family in ("count", "index", "constraint")):
        raise RuntimeError(f"sample logs in {run_root} lack one or more MolecularIQ task families")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return (
        {key: mean(values) for key, values in family_values.items() if key != "unknown"},
        {key: mean(values) for key, values in supercategory_values.items() if key != "unknown"},
    )


def _sample_observations(run_root: Path) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for path in sorted((run_root / "raw").rglob("samples*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                doc = row.get("doc", {})
                doc_hash = row.get("doc_hash")
                accuracy = row.get("avg_accuracy")
                if not isinstance(doc_hash, str) or not doc_hash or doc_hash in observations:
                    raise RuntimeError(f"sample logs in {run_root} lack unique stable document hashes")
                if (
                    not isinstance(accuracy, (int, float))
                    or not math.isfinite(float(accuracy))
                    or not 0.0 <= float(accuracy) <= 1.0
                ):
                    raise RuntimeError(f"sample logs in {run_root} contain a non-finite accuracy")
                observations[doc_hash] = {
                    "accuracy": float(accuracy),
                    "task_family": _task_family(str(doc.get("task_type", ""))),
                    "supercategory": str(doc.get("supercategory", "unknown")),
                }
                if observations[doc_hash]["task_family"] == "unknown":
                    raise RuntimeError(f"sample logs in {run_root} contain an unknown task family")
                if observations[doc_hash]["supercategory"] in {"", "unknown"}:
                    raise RuntimeError(f"sample logs in {run_root} contain an unknown supercategory")
    if len(observations) != 5111:
        raise RuntimeError(f"expected 5,111 paired sample observations in {run_root}")
    return observations


def _paired_bootstrap_rows(
    observations_by_model: dict[str, dict[str, dict[str, Any]]],
    *,
    replicates: int = 2000,
) -> list[dict[str, Any]]:
    import numpy as np

    baseline = observations_by_model["baseline"]
    baseline_keys = set(baseline)
    rows: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER[1:]:
        trained = observations_by_model[model_key]
        if set(trained) != baseline_keys:
            raise RuntimeError(f"{model_key} sample documents differ from the baseline")
        for doc_hash in baseline_keys:
            if (
                trained[doc_hash]["task_family"] != baseline[doc_hash]["task_family"]
                or trained[doc_hash]["supercategory"] != baseline[doc_hash]["supercategory"]
            ):
                raise RuntimeError(f"{model_key} sample metadata differs from the baseline")
        for scope in ("overall", "count", "index", "constraint"):
            keys = sorted(
                doc_hash
                for doc_hash in baseline_keys
                if scope == "overall" or baseline[doc_hash]["task_family"] == scope
            )
            if not keys:
                raise RuntimeError(f"official samples have no observations for scope {scope}")
            differences = np.asarray(
                [trained[key]["accuracy"] - baseline[key]["accuracy"] for key in keys],
                dtype=np.float64,
            )
            seed_material = f"miq-paired-bootstrap-v1:{model_key}:{scope}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            bootstrap_means: list[Any] = []
            for start in range(0, replicates, 200):
                chunk = min(200, replicates - start)
                indices = rng.integers(0, len(differences), size=(chunk, len(differences)))
                bootstrap_means.append(differences[indices].mean(axis=1))
            samples = np.concatenate(bootstrap_means)
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "scope": scope,
                    "n_documents": len(differences),
                    "avg_accuracy_delta": float(differences.mean()),
                    "ci95_lower": float(np.quantile(samples, 0.025)),
                    "ci95_upper": float(np.quantile(samples, 0.975)),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                }
            )
    return rows


def _protocol_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    clean_flags = {
        key: manifest.get(key)
        for key in (
            "benchmark_used_for_training",
            "benchmark_used_for_hparam_selection",
            "benchmark_used_for_prompt_tuning",
            "benchmark_used_for_checkpoint_selection",
        )
    }
    if any(value is not False for value in clean_flags.values()):
        raise RuntimeError("evaluation manifest does not preserve the held-out benchmark boundary")
    evaluator_identity = manifest.get("official_evaluator_identity", {})
    if (
        manifest.get("official_evaluator_commit") != MOLECULARIQ_EVAL_COMMIT
        or manifest.get("official_task") != OFFICIAL_TASK
        or manifest.get("official_task_yaml_sha256") != OFFICIAL_TASK_YAML_SHA256
        or evaluator_identity.get("commit") != MOLECULARIQ_EVAL_COMMIT
        or evaluator_identity.get("clean") is not True
        or not evaluator_identity.get("source_tree_sha256")
    ):
        raise RuntimeError("evaluation manifest is not bound to the clean pinned evaluator")
    return {
        "evaluation_suite_id": manifest.get("evaluation_suite_id"),
        "official_evaluator_commit": manifest.get("official_evaluator_commit"),
        "official_evaluator_identity": evaluator_identity,
        "official_task": manifest.get("official_task"),
        "official_task_yaml_sha256": manifest.get("official_task_yaml_sha256"),
        "official_task_hooks": manifest.get("official_task_hooks"),
        "official_benchmark_id": manifest.get("official_benchmark_id"),
        "official_benchmark_revision": manifest.get("official_benchmark_revision"),
        "official_benchmark_records_sha256": manifest.get("official_benchmark_records_sha256"),
        "full_benchmark": manifest.get("full_benchmark"),
        "limit": manifest.get("limit"),
        "repeat_count": manifest.get("repeat_count"),
        "system_prompt_sha256": manifest.get("system_prompt_sha256"),
        "backend": manifest.get("backend"),
        "dtype": manifest.get("dtype"),
        "official_task_generation_kwargs": manifest.get("official_task_generation_kwargs"),
        "generation_overrides": manifest.get("generation_overrides"),
        "effective_generation_kwargs": manifest.get("effective_generation_kwargs"),
        "seeds": manifest.get("seeds"),
        "resolved_config_sha256": manifest.get("resolved_config_sha256"),
        "package_versions": manifest.get("package_versions"),
        "installed_miq_grpo_tree_sha256": manifest.get("installed_miq_grpo_tree_sha256"),
        "environment_lock": manifest.get("environment_lock"),
        "clean_benchmark_flags": clean_flags,
    }


def _verify_bound_file(entry: Any, description: str) -> Path:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise RuntimeError(f"evaluation manifest lacks {description} file binding")
    path = Path(entry["path"]).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise RuntimeError(f"{description} file differs from the evaluated run: {path}")
    return path


def _verify_model_binding(config: dict[str, Any], manifest: dict[str, Any], model_key: str) -> None:
    model_entry = config["models"][model_key]
    checkpoint = Path(model_entry["checkpoint_path"]).expanduser().resolve()
    if (
        manifest.get("experiment_id") != model_entry.get("experiment_id")
        or Path(manifest.get("checkpoint_path", "")).expanduser().resolve() != checkpoint
    ):
        raise RuntimeError(f"{model_key} evaluation is bound to a different registered model")
    checkpoint_identity = directory_identity(checkpoint)
    if checkpoint_identity.get("tree_sha256") != manifest.get("checkpoint_identity", {}).get("tree_sha256"):
        raise RuntimeError(f"{model_key} checkpoint changed after official evaluation")

    binding = manifest.get("training_binding", {})
    if model_key == "baseline":
        if binding.get("kind") != "baseline" or model_entry.get("task_family") is not None:
            raise RuntimeError("baseline evaluation has an invalid training binding")
        assets_path = _verify_bound_file(
            binding.get("base_model_assets_manifest"), "base-model assets manifest"
        )
        assets = read_json(assets_path)
        if (
            Path(assets.get("model_path", "")).expanduser().resolve() != checkpoint
            or assets.get("model_identity", {}).get("tree_sha256") != checkpoint_identity["tree_sha256"]
            or assets.get("resolved_model_revision") != binding.get("base_model_revision")
        ):
            raise RuntimeError("baseline assets manifest differs from the evaluated checkpoint")
        return

    if (
        binding.get("kind") != "trained"
        or binding.get("task_family") != model_entry.get("task_family")
        or not binding.get("dataset_artifact_id")
        or not binding.get("dataset_records_sha256")
    ):
        raise RuntimeError(f"{model_key} lacks a valid single-task training binding")
    completion_path = _verify_bound_file(binding.get("training_complete"), "training completion")
    _verify_bound_file(binding.get("frozen_config"), "frozen training config")
    _verify_bound_file(binding.get("launch_provenance"), "launch provenance")
    resumes = binding.get("resume_provenance")
    if not isinstance(resumes, list):
        raise RuntimeError(f"{model_key} resume provenance binding is malformed")
    for entry in resumes:
        _verify_bound_file(entry, "resume provenance")
    completion = read_json(completion_path)
    if (
        completion.get("smoke_run") is not False
        or completion.get("experiment_id") != model_entry.get("experiment_id")
        or completion.get("task_family") != model_entry.get("task_family")
        or completion.get("dataset_artifact_id") != binding.get("dataset_artifact_id")
        or completion.get("dataset_records_sha256") != binding.get("dataset_records_sha256")
        or completion.get("final_model_identity", {}).get("tree_sha256") != checkpoint_identity["tree_sha256"]
    ):
        raise RuntimeError(f"{model_key} training completion differs from the evaluated checkpoint")


def _verify_completed_files(run_root: Path, manifest: dict[str, Any]) -> None:
    completion_path = run_root / "evaluation_complete.json"
    if not completion_path.is_file():
        raise RuntimeError(f"completed marker lacks evaluation_complete.json: {run_root}")
    completion = read_json(completion_path)
    for category in ("result_files", "sample_files", "log_files"):
        entries = completion.get(category)
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"evaluation completion manifest lacks {category}: {run_root}")
        for entry in entries:
            path = (run_root / entry["path"]).resolve()
            if run_root.resolve() not in path.parents or sha256_file(path) != entry.get("sha256"):
                raise RuntimeError(f"evaluation output hash mismatch: {path}")
    documents = completion.get("evaluated_documents", {})
    if documents.get("num_documents") != 5111 or documents.get("records_sha256") != manifest.get(
        "official_benchmark_records_sha256"
    ):
        raise RuntimeError("evaluation did not prove all staged benchmark documents were consumed")
    if completion.get("official_evaluator_identity_after") != manifest.get("official_evaluator_identity"):
        raise RuntimeError("official evaluator source identity changed during evaluation")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _training_metrics(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER[1:]:
        checkpoint = Path(config["models"][model_key]["checkpoint_path"]).resolve()
        log_path = checkpoint.parent / "logs" / "metrics.jsonl"
        if not log_path.is_file():
            raise RuntimeError(f"reportable training run lacks metrics log: {log_path}")
        model_rows = 0
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if not isinstance(payload, dict) or type(payload.get("step")) is not int:
                    raise RuntimeError(f"training metrics contain a malformed row: {log_path}")
                payload["model_key"] = model_key
                rows.append(payload)
                model_rows += 1
        if model_rows == 0:
            raise RuntimeError(f"reportable training metrics log is empty: {log_path}")
    return rows


def _save_figure(fig: Any, figure_root: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(figure_root / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_root / f"{name}.pdf", bbox_inches="tight")


def _create_plots(
    overall_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    paired_delta_rows: list[dict[str, Any]],
    figure_root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_root.mkdir(parents=True, exist_ok=False)
    metrics = ("pass_at_1", "pass_at_3", "avg_accuracy")
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9, 5))
    by_model = {row["model_key"]: row for row in overall_rows}
    for index, model_key in enumerate(MODEL_ORDER):
        values = [by_model[model_key][metric] for metric in metrics]
        ax.bar(
            x + (index - 1.5) * width, values, width, label=MODEL_LABELS[model_key], color=COLORS[model_key]
        )
    ax.set_xticks(x, ["Pass@1", "Pass@3", "Average accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Official MolecularIQ score")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_root, "overall_official_metrics")
    plt.close(fig)

    families = ("count", "index", "constraint")
    family_lookup = {(row["model_key"], row["task_family"]): row["avg_accuracy"] for row in family_rows}
    x = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, model_key in enumerate(MODEL_ORDER):
        values = [family_lookup[(model_key, family)] for family in families]
        ax.bar(
            x + (index - 1.5) * width, values, width, label=MODEL_LABELS[model_key], color=COLORS[model_key]
        )
    ax.set_xticks(x, ["Count", "Index", "Constrained generation"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Official average accuracy")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_root, "accuracy_by_task_family")
    plt.close(fig)

    overall_deltas = {row["model_key"]: row for row in paired_delta_rows if row["scope"] == "overall"}
    keys = list(MODEL_ORDER[1:])
    deltas = [overall_deltas[key]["avg_accuracy_delta"] for key in keys]
    errors = [
        [overall_deltas[key]["avg_accuracy_delta"] - overall_deltas[key]["ci95_lower"] for key in keys],
        [overall_deltas[key]["ci95_upper"] - overall_deltas[key]["avg_accuracy_delta"] for key in keys],
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        [MODEL_LABELS[key] for key in keys],
        deltas,
        yerr=errors,
        capsize=4,
        color=[COLORS[key] for key in keys],
    )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_ylabel("Average-accuracy change vs baseline")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_root, "paired_overall_delta_vs_baseline")
    plt.close(fig)

    family_deltas = {
        (row["model_key"], row["scope"]): row for row in paired_delta_rows if row["scope"] in families
    }
    x = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(9, 5))
    delta_width = 0.25
    for index, model_key in enumerate(keys):
        selected = [family_deltas[(model_key, family)] for family in families]
        values = [row["avg_accuracy_delta"] for row in selected]
        yerr = [
            [row["avg_accuracy_delta"] - row["ci95_lower"] for row in selected],
            [row["ci95_upper"] - row["avg_accuracy_delta"] for row in selected],
        ]
        ax.bar(
            x + (index - 1) * delta_width,
            values,
            delta_width,
            yerr=yerr,
            capsize=3,
            label=MODEL_LABELS[model_key],
            color=COLORS[model_key],
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x, ["Count", "Index", "Constrained generation"])
    ax.set_ylabel("Paired average-accuracy change vs baseline (95% bootstrap CI)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_root, "paired_task_family_deltas_vs_baseline")
    plt.close(fig)

    if training_rows:
        metric_candidates = ("reward", "reward/correctness_mean", "rewards/moleculariq_reward/mean")
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for model_key in MODEL_ORDER[1:]:
            rows = sorted(
                (row for row in training_rows if row["model_key"] == model_key), key=lambda row: row["step"]
            )
            model_plotted = False
            for metric in metric_candidates:
                points = [
                    (row["step"], row[metric]) for row in rows if isinstance(row.get(metric), (int, float))
                ]
                if points:
                    ax.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        label=MODEL_LABELS[model_key],
                        color=COLORS[model_key],
                    )
                    plotted = True
                    model_plotted = True
                    break
            if not model_plotted:
                raise RuntimeError(f"training logs lack a reward curve for {model_key}")
        if plotted:
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Training reward / correctness")
            ax.legend()
            ax.grid(alpha=0.25)
            _save_figure(fig, figure_root, "training_reward_curves")
        plt.close(fig)


def _aggregate_into(config_path: Path, config: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=False, exist_ok=False)
    if set(config.get("models", {})) != set(MODEL_ORDER):
        raise RuntimeError("report config must contain exactly the baseline and three single-task models")
    project_root_value = os.environ.get("MIQ_PROJECT_ROOT")
    if not project_root_value:
        raise RuntimeError("MIQ_PROJECT_ROOT is required for report provenance")
    project_root = Path(project_root_value).expanduser().resolve()
    offline_bundle_files = offline_bundle_file_identities()
    report_runtime_provenance = runtime_provenance(project_root)
    results_root = Path(config["output_root"]).expanduser().resolve()
    manifests: dict[str, dict[str, Any]] = {}
    overall_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    supercategory_rows: list[dict[str, Any]] = []
    run_roots: dict[str, Path] = {}
    observations_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    input_artifacts: list[dict[str, str]] = [
        {"path": str(config_path), "sha256": sha256_file(config_path)},
        *[dict(identity) for identity in offline_bundle_files.values()],
    ]
    resolved_config = {key: value for key, value in config.items() if key != "_config_path"}
    expected_config_hash = hashlib.sha256(canonical_json(resolved_config).encode("utf-8")).hexdigest()
    reference_protocol: dict[str, Any] | None = None
    for model_key in MODEL_ORDER:
        run_root = _completed_run(results_root, model_key)
        run_roots[model_key] = run_root
        manifest = read_json(run_root / "eval_manifest.json")
        manifests[model_key] = manifest
        input_artifacts.extend(
            [
                {
                    "path": str((run_root / "eval_manifest.json").resolve()),
                    "sha256": sha256_file(run_root / "eval_manifest.json"),
                },
                {
                    "path": str((run_root / "evaluation_complete.json").resolve()),
                    "sha256": sha256_file(run_root / "evaluation_complete.json"),
                },
            ]
        )
        if manifest.get("model_key") != model_key:
            raise RuntimeError(f"{model_key} directory contains a manifest for another model")
        if (
            manifest.get("full_benchmark") is not True
            or manifest.get("limit") is not None
            or manifest.get("official_evaluator_commit") != MOLECULARIQ_EVAL_COMMIT
            or manifest.get("official_task_yaml_sha256") != OFFICIAL_TASK_YAML_SHA256
            or manifest.get("system_prompt_sha256") != SYSTEM_PROMPT_SHA256
            or manifest.get("evaluation_suite_id") != config.get("evaluation_suite_id")
            or manifest.get("resolved_config_sha256") != expected_config_hash
        ):
            raise RuntimeError(f"{model_key} is not a comparable whole-benchmark official run")
        _verify_model_binding(config, manifest, model_key)
        protocol = _protocol_identity(manifest)
        if reference_protocol is None:
            reference_protocol = protocol
        elif canonical_json(protocol) != canonical_json(reference_protocol):
            raise RuntimeError(f"{model_key} used a different official evaluation protocol")
        _verify_completed_files(run_root, manifest)
        completion = read_json(run_root / "evaluation_complete.json")
        for category in ("result_files", "sample_files", "log_files"):
            for entry in completion[category]:
                path = (run_root / entry["path"]).resolve()
                input_artifacts.append({"path": str(path), "sha256": entry["sha256"]})
        metrics = _result_metrics(run_root)
        overall_rows.append({"model_key": model_key, "model_label": MODEL_LABELS[model_key], **metrics})
        family_metrics, supercategory_metrics = _sample_metrics(run_root)
        observations_by_model[model_key] = _sample_observations(run_root)
        family_rows.extend(
            {"model_key": model_key, "task_family": family, "avg_accuracy": accuracy}
            for family, accuracy in sorted(family_metrics.items())
        )
        supercategory_rows.extend(
            {"model_key": model_key, "supercategory": category, "avg_accuracy": accuracy}
            for category, accuracy in sorted(supercategory_metrics.items())
        )

    comparison_fields = ["model_key", "model_label", "pass_at_1", "pass_at_3", "avg_accuracy"]
    _write_csv(output_root / "official_overall_metrics.csv", comparison_fields, overall_rows)
    _write_csv(
        output_root / "official_task_family_metrics.csv",
        ["model_key", "task_family", "avg_accuracy"],
        family_rows,
    )
    _write_csv(
        output_root / "official_supercategory_metrics.csv",
        ["model_key", "supercategory", "avg_accuracy"],
        supercategory_rows,
    )
    training_rows = _training_metrics(config)
    if training_rows:
        all_fields = sorted({key for row in training_rows for key in row})
        _write_csv(output_root / "training_metrics.csv", all_fields, training_rows)
        for model_key in MODEL_ORDER[1:]:
            checkpoint = Path(config["models"][model_key]["checkpoint_path"]).resolve()
            log_path = checkpoint.parent / "logs" / "metrics.jsonl"
            if log_path.is_file():
                input_artifacts.append({"path": str(log_path.resolve()), "sha256": sha256_file(log_path)})
    paired_delta_rows = _paired_bootstrap_rows(observations_by_model)
    _write_csv(
        output_root / "official_paired_deltas_with_ci.csv",
        [
            "model_key",
            "model_label",
            "scope",
            "n_documents",
            "avg_accuracy_delta",
            "ci95_lower",
            "ci95_upper",
            "bootstrap_replicates",
            "bootstrap_seed",
        ],
        paired_delta_rows,
    )
    _create_plots(
        overall_rows,
        family_rows,
        training_rows,
        paired_delta_rows,
        output_root / "figures",
    )
    output_artifacts = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    ]
    write_json_exclusive(
        output_root / "report_manifest.json",
        {
            "created_at": utc_now(),
            "evaluation_suite_id": config["evaluation_suite_id"],
            "model_runs": {key: str(path) for key, path in run_roots.items()},
            "evaluation_protocol": reference_protocol,
            "input_artifacts": input_artifacts,
            "output_artifacts": output_artifacts,
            "offline_bundle_files": offline_bundle_files,
            "report_runtime_provenance": report_runtime_provenance,
            "full_official_benchmark_only": True,
            "benchmark_used_for_model_or_checkpoint_selection": False,
            "paired_bootstrap_replicates": 2000,
            "figures": sorted(path.name for path in (output_root / "figures").iterdir()),
        },
    )
    (output_root / "_COMPLETE").touch(exist_ok=False)


def aggregate(config_path: Path, output_root: Path) -> Path:
    config_path = config_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    config = load_yaml(config_path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite report artifact: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    staging.rmdir()
    try:
        _aggregate_into(config_path, config, staging)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(aggregate(args.config.expanduser().resolve(), args.output_root.expanduser().resolve()))


if __name__ == "__main__":
    main()
