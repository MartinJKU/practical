from __future__ import annotations

import json
from pathlib import Path

import pytest

import miq_grpo.reporting as reporting
from miq_grpo.constants import (
    MOLECULARIQ_EVAL_COMMIT,
    OFFICIAL_TASK,
    OFFICIAL_TASK_YAML_SHA256,
    SYSTEM_PROMPT_SHA256,
)
from miq_grpo.reporting import (
    _completed_run,
    _paired_bootstrap_rows,
    _protocol_identity,
    _result_metrics,
    _sample_metrics,
)


def test_official_result_and_sample_schema_are_parsed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    raw = run / "raw"
    raw.mkdir(parents=True)
    (raw / "results.json").write_text(
        json.dumps(
            {
                "results": {
                    OFFICIAL_TASK: {
                        "pass_at_1,all": 0.1,
                        "pass_at_1_stderr,all": 0.01,
                        "pass_at_3,all": 0.2,
                        "pass_at_3_stderr,all": 0.02,
                        "avg_accuracy,all": 0.15,
                        "avg_accuracy_stderr,all": 0.03,
                        "single_count_avg_accuracy,all": 0.9,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "doc": {"task_type": "single_count", "supercategory": "composition"},
            "metrics": ["avg_accuracy"],
            "avg_accuracy": 1.0,
        },
        {
            "doc": {"task_type": "single_index_identification", "supercategory": "composition"},
            "metrics": ["avg_accuracy"],
            "avg_accuracy": 0.5,
        },
        {
            "doc": {"task_type": "constraint_generation", "supercategory": "generation"},
            "metrics": ["avg_accuracy"],
            "avg_accuracy": 0.0,
        },
    ]
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    assert _result_metrics(run) == {
        "pass_at_1": 0.1,
        "pass_at_3": 0.2,
        "avg_accuracy": 0.15,
    }
    family, supercategory = _sample_metrics(run)
    assert family == {"constraint": 0.0, "count": 1.0, "index": 0.5}
    assert supercategory["composition"] == 0.75


def test_completed_run_refuses_score_based_selection(tmp_path: Path) -> None:
    model_root = tmp_path / "baseline"
    first = model_root / "run-1"
    first.mkdir(parents=True)
    (first / "_COMPLETE").touch()
    assert _completed_run(tmp_path, "baseline") == first

    second = model_root / "run-2"
    second.mkdir()
    (second / "_COMPLETE").touch()
    with pytest.raises(RuntimeError, match="exactly one"):
        _completed_run(tmp_path, "baseline")


def _protocol_manifest() -> dict:
    return {
        "evaluation_suite_id": "suite-v1",
        "official_evaluator_commit": MOLECULARIQ_EVAL_COMMIT,
        "official_evaluator_identity": {
            "commit": MOLECULARIQ_EVAL_COMMIT,
            "clean": True,
            "source_tree_sha256": "1" * 64,
        },
        "official_task": OFFICIAL_TASK,
        "official_task_yaml_sha256": OFFICIAL_TASK_YAML_SHA256,
        "official_task_hooks": {"process_results": "task_processor.process_results_pass_at_k"},
        "official_benchmark_id": "ml-jku/moleculariq-v0.0",
        "official_benchmark_revision": "2" * 40,
        "official_benchmark_records_sha256": "3" * 64,
        "full_benchmark": True,
        "limit": None,
        "repeat_count": 3,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "backend": "hf",
        "dtype": "bfloat16",
        "official_task_generation_kwargs": {
            "max_tokens": 32768,
            "until": [],
            "do_sample": True,
        },
        "generation_overrides": {"temperature": 0.7, "top_p": 0.95, "top_k": 50},
        "effective_generation_kwargs": {
            "max_tokens": 32768,
            "until": [],
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
        },
        "seeds": [0, 1234, 1234, 1234],
        "resolved_config_sha256": "4" * 64,
        "package_versions": {"lm_eval": "0.4.9"},
        "installed_miq_grpo_tree_sha256": "5" * 64,
        "environment_lock": {"requirements_lock": {"sha256": "6" * 64}},
        "benchmark_used_for_training": False,
        "benchmark_used_for_hparam_selection": False,
        "benchmark_used_for_prompt_tuning": False,
        "benchmark_used_for_checkpoint_selection": False,
    }


def test_protocol_identity_requires_clean_pinned_evaluator_and_heldout_flags() -> None:
    protocol = _protocol_identity(_protocol_manifest())
    assert protocol["official_task_yaml_sha256"] == OFFICIAL_TASK_YAML_SHA256
    assert protocol["effective_generation_kwargs"]["do_sample"] is True

    dirty = _protocol_manifest()
    dirty["official_evaluator_identity"]["clean"] = False
    with pytest.raises(RuntimeError, match="clean pinned evaluator"):
        _protocol_identity(dirty)

    tuned = _protocol_manifest()
    tuned["benchmark_used_for_hparam_selection"] = True
    with pytest.raises(RuntimeError, match="held-out benchmark"):
        _protocol_identity(tuned)


def test_report_publication_cleans_failed_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "evaluation.yaml"
    config.write_text("evaluation_suite_id: test\n", encoding="utf-8")
    output = tmp_path / "report-v1"

    def fail_after_partial_write(config_path: Path, loaded: dict, staging: Path) -> None:
        del config_path, loaded
        staging.mkdir()
        (staging / "partial.csv").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated plotting failure")

    monkeypatch.setattr(reporting, "_aggregate_into", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated plotting failure"):
        reporting.aggregate(config, output)

    assert not output.exists()
    assert list(tmp_path.glob(".report-v1.building-*")) == []


def test_paired_bootstrap_uses_matched_documents_and_is_deterministic() -> None:
    baseline: dict[str, dict] = {}
    for family_index, family in enumerate(("count", "index", "constraint")):
        for index in range(4):
            baseline[f"{family}-{index}"] = {
                "accuracy": float((family_index + index) % 2),
                "task_family": family,
                "supercategory": family,
            }
    observations = {"baseline": baseline}
    for model_key in ("count_grpo", "index_grpo", "constraint_grpo"):
        observations[model_key] = {
            key: {**value, "accuracy": min(1.0, value["accuracy"] + 0.25)} for key, value in baseline.items()
        }

    first = _paired_bootstrap_rows(observations, replicates=100)
    second = _paired_bootstrap_rows(observations, replicates=100)

    assert first == second
    assert len(first) == 12
    assert all(row["ci95_lower"] <= row["avg_accuracy_delta"] <= row["ci95_upper"] for row in first)
