from __future__ import annotations

import json
from pathlib import Path

import pytest

from miq_grpo.constants import OFFICIAL_BENCHMARK_ID, OFFICIAL_TASK
from miq_grpo.io_utils import records_sha256
from miq_grpo.preflight import (
    _inspect_evaluation_task_yaml,
    _validate_cached_benchmark_rows,
    smoke_metrics_preflight,
)


def _write_smoke_run(
    root: Path,
    *,
    task: str = "count",
    training_loss: float = 0.25,
    zero_variance_values: tuple[float, ...] = (1.0, 0.5),
) -> None:
    (root / "logs").mkdir(parents=True)
    (root / "training_complete.json").write_text(
        json.dumps(
            {
                "smoke_run": True,
                "task_family": task,
                "global_step": 1,
                "training_loss": training_loss,
            }
        ),
        encoding="utf-8",
    )
    with (root / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for value in zero_variance_values:
            handle.write(
                json.dumps(
                    {
                        "step": 1,
                        "loss": 0.0,
                        "rewards/moleculariq_reward/grpo/zero_variance_group_fraction": value,
                    }
                )
                + "\n"
            )


def test_smoke_metrics_requires_finite_run_and_nonzero_group_variance(tmp_path: Path) -> None:
    _write_smoke_run(tmp_path)

    result = smoke_metrics_preflight(tmp_path, "count")

    assert result["global_step"] == 1
    assert result["minimum_zero_variance_group_fraction"] == 0.5
    assert result["usable_reward_variance_observed"] is True


def test_smoke_metrics_rejects_fully_zero_variance_rollout(tmp_path: Path) -> None:
    _write_smoke_run(tmp_path, zero_variance_values=(1.0, 1.0))

    with pytest.raises(RuntimeError, match="no within-group reward variance"):
        smoke_metrics_preflight(tmp_path, "count")


def test_smoke_metrics_rejects_nonfinite_values(tmp_path: Path) -> None:
    _write_smoke_run(tmp_path, training_loss=float("nan"), zero_variance_values=(0.5,))

    with pytest.raises(RuntimeError, match="training loss is not finite"):
        smoke_metrics_preflight(tmp_path, "count")


def test_official_task_yaml_contract_is_checked_without_loading_dataset(tmp_path: Path) -> None:
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        "\n".join(
            (
                f"task: {OFFICIAL_TASK}",
                f"dataset_path: {OFFICIAL_BENCHMARK_ID}",
                "dataset_name: default",
                "test_split: test",
                "output_type: generate_until",
                "repeats: 3",
                "process_docs: !function task_processor.process_docs",
                "doc_to_text: !function task_processor.doc_to_text",
                "doc_to_target: target",
                "process_results: !function task_processor.process_results_pass_at_k",
                "generation_kwargs:",
                "  max_tokens: 32768",
                "  until: []",
                "  do_sample: true",
                "metric_list:",
                "  - metric: pass_at_1",
                "  - metric: pass_at_3",
                "  - metric: avg_accuracy",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = _inspect_evaluation_task_yaml(task_yaml)

    assert result["repeats"] == 3
    assert len(result["sha256"]) == 64


def test_cached_benchmark_must_match_staged_count_and_content_hash() -> None:
    rows = [{"row_id": index, "question": f"q-{index}"} for index in range(5111)]
    manifest = {"test_rows": 5111, "test_records_sha256": records_sha256(rows)}
    assert _validate_cached_benchmark_rows(rows, manifest) == manifest["test_records_sha256"]

    changed = [*rows]
    changed[-1] = {"row_id": 5110, "question": "changed"}
    with pytest.raises(RuntimeError, match="content hash"):
        _validate_cached_benchmark_rows(changed, manifest)
