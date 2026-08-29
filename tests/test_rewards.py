from __future__ import annotations

from typing import Any

import pytest

import miq_grpo.rewards as rewards
from miq_grpo.parsing import ParseStatus
from miq_grpo.rewards import make_trl_reward, score_completion


def fake_evaluator(
    *,
    task_type: str,
    predicted: Any,
    target: dict[str, Any] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    return_details: bool,
) -> dict[str, Any]:
    assert return_details
    if task_type == "constraint_generation":
        assert constraints is not None
        smiles = predicted.get("smiles", "") if isinstance(predicted, dict) else str(predicted)
        reward = float(smiles in {"c1ccccc1", "C1=CC=CC=C1"})
        return {"reward": reward, "supported": len(constraints), "total": len(constraints)}
    return {"reward": float(predicted == target), "matched": int(predicted == target), "total": 1}


def test_count_correct_and_wrong_use_stored_target() -> None:
    correct = score_completion(
        '<answer>{"ring_count":1}</answer>',
        task_family="count",
        task_type="single_count",
        target_json='{"ring_count":1}',
        constraints_json="[]",
        evaluator=fake_evaluator,
        validity_weight=0.0,
    )
    wrong = score_completion(
        '<answer>{"ring_count":2}</answer>',
        task_family="count",
        task_type="single_count",
        target_json='{"ring_count":1}',
        constraints_json="[]",
        evaluator=fake_evaluator,
        validity_weight=0.0,
    )

    assert (correct.correctness, correct.total) == (1.0, 1.05)
    assert (wrong.correctness, wrong.total) == (0.0, 0.05)


def test_constraint_scores_any_satisfying_molecule_not_reference_equality() -> None:
    common = {
        "task_family": "constraint",
        "task_type": "constraint_generation",
        "target_json": "{}",
        "constraints_json": '[{"type":"ring_count","operator":"=","value":1}]',
        "evaluator": fake_evaluator,
        "smiles_validator": lambda value: value in {"c1ccccc1", "C1=CC=CC=C1", "CCO"},
    }
    aromatic = score_completion('<answer>{"smiles":"c1ccccc1"}</answer>', **common)
    kekule = score_completion('<answer>{"smiles":"C1=CC=CC=C1"}</answer>', **common)
    violating = score_completion('<answer>{"smiles":"CCO"}</answer>', **common)

    assert aromatic.correctness == kekule.correctness == 1.0
    assert violating.correctness == 0.0
    assert violating.total == pytest.approx(0.1)
    assert aromatic.total > violating.total


def test_invalid_smiles_and_verifier_errors_fail_closed() -> None:
    invalid = score_completion(
        '<answer>{"smiles":"invalid"}</answer>',
        task_family="constraint",
        task_type="constraint_generation",
        target_json="{}",
        constraints_json='[{"type":"ring_count","operator":"=","value":1}]',
        evaluator=fake_evaluator,
        smiles_validator=lambda value: False,
    )

    def broken_evaluator(**kwargs: Any) -> Any:
        raise RuntimeError("oracle unavailable")

    broken = score_completion(
        '<answer>{"ring_count":1}</answer>',
        task_family="count",
        task_type="single_count",
        target_json='{"ring_count":1}',
        constraints_json="[]",
        evaluator=broken_evaluator,
        validity_weight=0.0,
    )

    assert invalid.total == 0.0
    assert invalid.parsed.status == ParseStatus.INVALID_SMILES
    assert broken.total == 0.0
    assert broken.parsed.status == ParseStatus.VERIFIER_ERROR


def test_unsupported_constraint_details_fail_closed() -> None:
    result = score_completion(
        '<answer>{"smiles":"c1ccccc1"}</answer>',
        task_family="constraint",
        task_type="constraint_generation",
        target_json="{}",
        constraints_json='[{"type":"ring_count","operator":"=","value":1}]',
        evaluator=lambda **kwargs: {"reward": 1.0, "supported": 0, "total": 1},
        smiles_validator=lambda value: True,
    )
    assert result.correctness == 0.0


def test_out_of_range_official_reward_is_zeroed() -> None:
    result = score_completion(
        '<answer>{"ring_count":1}</answer>',
        task_family="count",
        task_type="single_count",
        target_json='{"ring_count":1}',
        constraints_json="[]",
        evaluator=lambda **kwargs: {"reward": 2.0},
        validity_weight=0.0,
    )
    assert result.total == 0.0
    assert result.parsed.status == ParseStatus.VERIFIER_ERROR


def test_trl_adapter_alignment_and_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(rewards, "_default_evaluator", lambda: fake_evaluator)
    metric_log: dict[str, float] = {}
    extra_log: dict[str, list[Any]] = {}
    reward_fn = make_trl_reward(
        "count",
        {
            "correctness_weight": 1.0,
            "format_weight": 0.05,
            "validity_weight": 0.0,
            "correct_answer_must_dominate": True,
        },
    )
    values = reward_fn(
        prompts=[[{"role": "user", "content": "q"}]] * 2,
        completions=['<answer>{"ring_count":1}</answer>', '<answer>{"ring_count":2}</answer>'],
        completion_ids=[[1], [2]],
        task_family=["count", "count"],
        task_type=["single_count", "single_count"],
        target_json=['{"ring_count":1}', '{"ring_count":1}'],
        constraints_json=["[]", "[]"],
        example_id=["same", "same"],
        log_extra=extra_log.__setitem__,
        log_metric=metric_log.__setitem__,
    )

    assert values == [1.05, 0.05]
    assert metric_log["reward/correctness_mean"] == 0.5
    assert metric_log["grpo/duplicate_completion_fraction"] == 0.0
    assert metric_log["grpo/mean_group_reward_variance"] > 0.0
    assert metric_log["grpo/zero_variance_group_fraction"] == 0.0
    assert extra_log["parse_status"] == ["OK", "OK"]

    with pytest.raises(ValueError, match="misaligned"):
        reward_fn(
            prompts=[],
            completions=["1"],
            completion_ids=[[1]],
            task_family=["count"],
            task_type=[],
            target_json=['{"ring_count":1}'],
            constraints_json=["[]"],
            example_id=["x"],
        )


def test_auxiliary_rewards_cannot_outweigh_correctness() -> None:
    with pytest.raises(ValueError, match="correctness weight"):
        make_trl_reward(
            "count",
            {
                "correctness_weight": 0.1,
                "format_weight": 0.05,
                "validity_weight": 0.05,
                "correct_answer_must_dominate": True,
            },
        )
