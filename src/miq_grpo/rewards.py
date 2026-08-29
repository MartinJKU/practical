"""Stored-target MolecularIQ scoring and the TRL reward adapter."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .parsing import FAILURE_STATUSES, ParsedCompletion, ParseStatus, extract_smiles, parse_completion


@dataclass(frozen=True)
class ScoreResult:
    total: float
    correctness: float
    format_reward: float
    validity: float
    parsed: ParsedCompletion
    verifier_details: dict[str, Any]


def _default_evaluator() -> Callable[..., Any]:
    from moleculariq_core import evaluate_answer

    return evaluate_answer


def _default_smiles_validator() -> Callable[[str], bool]:
    from moleculariq_core import is_reasonable_molecule, valid_smiles

    return lambda smiles: valid_smiles(smiles) and is_reasonable_molecule(smiles)


def score_completion(
    completion: Any,
    *,
    task_family: str,
    task_type: str,
    target_json: str,
    constraints_json: str,
    correctness_weight: float = 1.0,
    format_weight: float = 0.05,
    validity_weight: float = 0.05,
    evaluator: Callable[..., Any] | None = None,
    smiles_validator: Callable[[str], bool] | None = None,
) -> ScoreResult:
    """Score one new completion without generating or changing its question."""
    parsed = parse_completion(completion, task_family)
    correctness = 0.0
    validity = 0.0
    details: dict[str, Any] = {}
    if parsed.status == ParseStatus.OK:
        try:
            evaluator_fn = evaluator or _default_evaluator()
            if task_family == "constraint":
                constraints = json.loads(constraints_json)
                smiles = extract_smiles(parsed.answer)
                validator = smiles_validator or _default_smiles_validator()
                try:
                    validity = 1.0 if validator(smiles) else 0.0
                except Exception:
                    parsed = replace(parsed, status=ParseStatus.SANITIZATION_FAILURE)
                    validity = 0.0
                if parsed.status == ParseStatus.OK and not validity:
                    parsed = replace(parsed, status=ParseStatus.INVALID_SMILES)
                if parsed.status == ParseStatus.OK:
                    raw_result = evaluator_fn(
                        task_type=task_type,
                        predicted=parsed.answer,
                        constraints=constraints,
                        return_details=True,
                    )
                    details = raw_result if isinstance(raw_result, dict) else {"reward": raw_result}
                    # Fail closed when the official implementation reports an unsupported constraint.
                    supported = details.get("supported", len(constraints))
                    total = details.get("total", len(constraints))
                    if supported == total == len(constraints):
                        correctness = float(details.get("reward", 0.0))
            elif task_family in {"count", "index"}:
                target = json.loads(target_json)
                raw_result = evaluator_fn(
                    task_type=task_type,
                    predicted=parsed.answer,
                    target=target,
                    return_details=True,
                )
                details = raw_result if isinstance(raw_result, dict) else {"reward": raw_result}
                correctness = float(details.get("reward", 0.0))
            else:
                parsed = replace(parsed, status=ParseStatus.UNSUPPORTED_TASK)
        except Exception as exc:
            parsed = replace(parsed, status=ParseStatus.VERIFIER_ERROR)
            details = {"error_type": type(exc).__name__, "error": str(exc)[:240]}
            correctness = 0.0

    format_reward = 1.0 if parsed.format_ok else 0.0
    components = (correctness, format_reward, validity)
    if any(not math.isfinite(component) or component < 0.0 or component > 1.0 for component in components):
        parsed = replace(parsed, status=ParseStatus.VERIFIER_ERROR)
        correctness = format_reward = validity = 0.0
        details = {"error": "non-finite or out-of-range reward component"}
    total_reward = (
        correctness_weight * correctness + format_weight * format_reward + validity_weight * validity
    )
    if not math.isfinite(total_reward):
        total_reward = 0.0
        parsed = replace(parsed, status=ParseStatus.VERIFIER_ERROR)
    return ScoreResult(total_reward, correctness, format_reward, validity, parsed, details)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _duplicate_fraction(example_ids: list[str], completion_texts: list[str]) -> float:
    groups: dict[str, list[str]] = defaultdict(list)
    for example_id, text in zip(example_ids, completion_texts, strict=True):
        groups[example_id].append(text.strip())
    duplicate_count = 0
    total = 0
    for values in groups.values():
        counts = Counter(values)
        duplicate_count += sum(count - 1 for count in counts.values() if count > 1)
        total += len(values)
    return duplicate_count / total if total else 0.0


def _group_variance_metrics(example_ids: list[str], rewards: list[float]) -> tuple[float, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for example_id, reward in zip(example_ids, rewards, strict=True):
        groups[example_id].append(reward)
    variances: list[float] = []
    for values in groups.values():
        group_mean = _mean(values)
        variances.append(_mean([(value - group_mean) ** 2 for value in values]))
    mean_variance = _mean(variances)
    zero_fraction = _mean([float(variance <= 1e-12) for variance in variances])
    return mean_variance, zero_fraction


def make_trl_reward(expected_task_family: str, reward_config: dict[str, Any]) -> Callable[..., list[float]]:
    correctness_weight = float(reward_config["correctness_weight"])
    format_weight = float(reward_config["format_weight"])
    validity_weight = float(reward_config["validity_weight"])
    if reward_config.get("correct_answer_must_dominate", True) and correctness_weight <= (
        format_weight + validity_weight
    ):
        raise ValueError("correctness weight must exceed the maximum auxiliary reward")

    def moleculariq_reward(
        prompts: list[Any],
        completions: list[Any],
        completion_ids: list[list[int]],
        task_family: list[str],
        task_type: list[str],
        target_json: list[str],
        constraints_json: list[str],
        example_id: list[str],
        log_extra: Callable[[str, list], None] | None = None,
        log_metric: Callable[[str, float], None] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """TRL-compatible reward using only frozen reward-side columns."""
        del prompts, completion_ids, kwargs
        lengths = {
            len(completions),
            len(task_family),
            len(task_type),
            len(target_json),
            len(constraints_json),
            len(example_id),
        }
        if len(lengths) != 1:
            raise ValueError(f"reward inputs are misaligned: lengths={sorted(lengths)}")
        if any(family != expected_task_family for family in task_family):
            raise ValueError("single-task run received a row from another task family")
        results = [
            score_completion(
                completion,
                task_family=family,
                task_type=row_task_type,
                target_json=row_target,
                constraints_json=row_constraints,
                correctness_weight=correctness_weight,
                format_weight=format_weight,
                validity_weight=validity_weight,
            )
            for completion, family, row_task_type, row_target, row_constraints in zip(
                completions,
                task_family,
                task_type,
                target_json,
                constraints_json,
                strict=True,
            )
        ]
        text_values = [result.parsed.raw_text for result in results]
        total_values = [float(result.total) for result in results]
        if log_extra is not None:
            log_extra("example_id", list(example_id))
            log_extra("parse_status", [result.parsed.status.value for result in results])
            log_extra("extracted_answer", [result.parsed.extracted_text for result in results])
            log_extra("correctness", [result.correctness for result in results])
            log_extra("format_reward", [result.format_reward for result in results])
            log_extra("validity", [result.validity for result in results])
        if log_metric is not None:
            log_metric("reward/correctness_mean", _mean([result.correctness for result in results]))
            log_metric("reward/format_mean", _mean([result.format_reward for result in results]))
            log_metric("reward/validity_mean", _mean([result.validity for result in results]))
            log_metric(
                "parse/failure_fraction",
                _mean([float(result.parsed.status in FAILURE_STATUSES) for result in results]),
            )
            log_metric(
                "chem/invalid_smiles_fraction",
                _mean([float(result.parsed.status == ParseStatus.INVALID_SMILES) for result in results]),
            )
            log_metric(
                "verifier/error_fraction",
                _mean([float(result.parsed.status == ParseStatus.VERIFIER_ERROR) for result in results]),
            )
            log_metric("grpo/duplicate_completion_fraction", _duplicate_fraction(example_id, text_values))
            mean_variance, zero_fraction = _group_variance_metrics(example_id, total_values)
            log_metric("grpo/mean_group_reward_variance", mean_variance)
            log_metric("grpo/zero_variance_group_fraction", zero_fraction)
        return total_values

    return moleculariq_reward
