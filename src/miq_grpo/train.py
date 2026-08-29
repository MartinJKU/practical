"""Full-parameter Qwen2.5-0.5B GRPO training on one frozen task family."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ConfigError, dump_yaml, load_yaml, require_keys
from .constants import MODEL_ID, PROJECT_SCHEMA_VERSION, TASK_FAMILIES, TRL_VERSION
from .dataset_validation import resolve_bundle_artifact, validate_artifact
from .io_utils import (
    canonical_json,
    directory_identity,
    offline_bundle_file_identities,
    read_json,
    runtime_provenance,
    sha256_file,
    source_tree_sha256,
    utc_now,
    write_json_exclusive,
)
from .rewards import make_trl_reward

_PREEMPT_REQUESTED = False


def _signal_preemption(signum: int, frame: Any) -> None:
    del signum, frame
    global _PREEMPT_REQUESTED
    _PREEMPT_REQUESTED = True


def _require_offline_mode() -> None:
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    disabled = [key for key in required if os.environ.get(key) != "1"]
    if disabled:
        raise RuntimeError(f"training requires enforced offline mode; set to 1: {', '.join(disabled)}")


def _project_root() -> Path:
    value = os.environ.get("MIQ_PROJECT_ROOT")
    if not value:
        raise RuntimeError("MIQ_PROJECT_ROOT is required to bind runs to the submitted source tree")
    root = Path(value).expanduser().resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src" / "miq_grpo").is_dir():
        raise RuntimeError(f"MIQ_PROJECT_ROOT is not this MolecularIQ project: {root}")
    return root


def _verify_assets_manifest(model_config: dict[str, Any]) -> dict[str, Any]:
    manifest_value = os.environ.get("MIQ_TRAIN_ASSETS_MANIFEST")
    if not manifest_value:
        raise RuntimeError("MIQ_TRAIN_ASSETS_MANIFEST is required for immutable offline training")
    manifest_path = Path(manifest_value).expanduser().resolve()
    manifest = read_json(manifest_path)
    if not (manifest_path.parent / "_READY").is_file():
        raise RuntimeError("training assets are not marked ready")
    if manifest.get("model_id") != MODEL_ID:
        raise RuntimeError("training assets contain the wrong base model")
    if Path(manifest["model_path"]).resolve() != Path(model_config["local_path"]).resolve():
        raise RuntimeError("configured model path differs from the staged-assets manifest")
    if manifest.get("resolved_model_revision") != model_config["expected_revision"]:
        raise RuntimeError("configured model revision differs from the staged-assets manifest")
    if manifest.get("official_benchmark_staged") is not False:
        raise RuntimeError("training assets must not contain the official benchmark")
    actual_identity = directory_identity(model_config["local_path"])
    if actual_identity.get("tree_sha256") != manifest.get("model_identity", {}).get("tree_sha256"):
        raise RuntimeError("local base-model files differ from the immutable staged snapshot")
    return manifest


def _validate_experiment_config(config: dict[str, Any]) -> None:
    require_keys(
        config,
        ("experiment_id", "task_family", "dataset", "model", "trainer", "reward", "benchmark_integrity"),
        context="experiment config",
    )
    if config.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise ConfigError("unsupported experiment schema version")
    if config["task_family"] not in TASK_FAMILIES:
        raise ConfigError("unknown task family")
    if config["dataset"].get("selector") != config["task_family"]:
        raise ConfigError("dataset selector must equal the experiment task family")
    model = config["model"]
    if model.get("id") != MODEL_ID or not model.get("full_finetune"):
        raise ConfigError("this study is locked to full fine-tuning Qwen/Qwen2.5-0.5B-Instruct")
    if model.get("dtype") != "float32" or config["trainer"].get("bf16") is not True:
        raise ConfigError("full fine-tuning requires FP32 master weights with BF16 mixed-precision compute")
    if config["trainer"].get("generation_kwargs") != {"do_sample": True}:
        raise ConfigError("GRPO rollout sampling must explicitly set do_sample: true")
    if not model.get("local_files_only") or model.get("trust_remote_code"):
        raise ConfigError("model must load locally without remote code")
    integrity = config["benchmark_integrity"]
    forbidden_true = [
        key for key, value in integrity.items() if key.startswith("official_benchmark_used") and value
    ]
    if forbidden_true:
        raise ConfigError(f"clean experiment config violates benchmark boundary: {forbidden_true}")


def _batch_arithmetic(trainer: dict[str, Any], world_size: int) -> dict[str, int]:
    per_device = int(trainer["per_device_train_batch_size"])
    accumulation = int(trainer["gradient_accumulation_steps"])
    generations = int(trainer["num_generations"])
    generation_batch = per_device * world_size * accumulation
    if generations < 2 or generation_batch % generations:
        raise ConfigError(
            f"invalid GRPO arithmetic: generation_batch={generation_batch}, num_generations={generations}"
        )
    return {
        "world_size": world_size,
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": accumulation,
        "generation_batch_size": generation_batch,
        "num_generations": generations,
        "unique_prompts_per_generation_batch": generation_batch // generations,
    }


def _run_directory(config: dict[str, Any], smoke: bool) -> Path:
    root = Path(config["trainer"]["output_root"]).expanduser().resolve()
    if smoke:
        stamp = os.environ.get("SLURM_JOB_ID") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return root / "smoke" / f"{config['experiment_id']}-smoke-{stamp}"
    return root / config["experiment_id"]


def _render_prompt_regression(dataset: Any, tokenizer: Any, max_completion_length: int) -> dict[str, Any]:
    if len(dataset) == 0:
        raise RuntimeError("cannot train on an empty dataset")
    max_prompt_tokens = 0
    rendered_first = ""
    for index, row in enumerate(dataset):
        rendered = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        if index == 0:
            rendered_first = rendered
        max_prompt_tokens = max(max_prompt_tokens, len(token_ids))
    model_limit = int(getattr(tokenizer, "model_max_length", 0) or 0)
    # Tokenizers sometimes use a huge sentinel instead of the real model limit.
    if model_limit < 10_000_000 and max_prompt_tokens + max_completion_length > model_limit:
        raise RuntimeError(
            "prompt+completion exceeds tokenizer model_max_length: "
            f"{max_prompt_tokens}+{max_completion_length}"
        )
    import hashlib

    return {
        "first_rendered_prompt_sha256": hashlib.sha256(rendered_first.encode("utf-8")).hexdigest(),
        "first_rendered_prompt": rendered_first,
        "max_prompt_tokens": max_prompt_tokens,
        "max_completion_length": max_completion_length,
        "tokenizer_model_max_length": model_limit,
    }


def _grpo_arguments(
    config: dict[str, Any], run_dir: Path, smoke: bool, smoke_max_steps: int = 1
) -> dict[str, Any]:
    trainer = dict(config["trainer"])
    trainer.pop("output_root")
    if trainer.get("min_p") is None:
        trainer.pop("min_p", None)
    trainer.update(
        {
            "output_dir": str(run_dir),
            "run_name": run_dir.name,
            "overwrite_output_dir": False,
            "eval_strategy": "no",
            "do_eval": False,
            "model_init_kwargs": {
                "dtype": config["model"]["dtype"],
                "attn_implementation": config["model"]["attention_implementation"],
                "local_files_only": True,
                "trust_remote_code": False,
            },
            "trust_remote_code": False,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }
    )
    if smoke:
        if smoke_max_steps < 1:
            raise ConfigError("smoke_max_steps must be positive")
        trainer.update(
            {
                "max_steps": smoke_max_steps,
                "num_train_epochs": 1.0,
                "save_steps": 1,
                "logging_steps": 1,
                "save_total_limit": 1,
                "report_to": [],
            }
        )
    return trainer


def train(
    config_path: Path,
    *,
    resume_from_checkpoint: str | None,
    smoke: bool,
    smoke_max_steps: int = 1,
    smoke_stop_after_step: int | None = None,
) -> Path:
    _require_offline_mode()
    if not smoke and (smoke_max_steps != 1 or smoke_stop_after_step is not None):
        raise ConfigError("smoke controls cannot be used for production training")
    if smoke_stop_after_step is not None and not (0 < smoke_stop_after_step < smoke_max_steps):
        raise ConfigError("smoke_stop_after_step must be positive and below smoke_max_steps")
    config = load_yaml(config_path)
    _validate_experiment_config(config)
    if config["dataset"]["selector"] != config["task_family"]:
        raise ConfigError("single-task dataset selector mismatch")
    model_path = Path(config["model"]["local_path"]).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model snapshot does not exist: {model_path}")
    config["model"]["local_path"] = str(model_path)
    assets_manifest = _verify_assets_manifest(config["model"])

    artifact_path = resolve_bundle_artifact(config["dataset"]["bundle_path"], config["task_family"])
    artifact_manifest = validate_artifact(artifact_path, expected_task=config["task_family"])
    offline_bundle_files = offline_bundle_file_identities()
    if artifact_manifest.get("offline_bundle_files") != offline_bundle_files:
        raise RuntimeError("frozen dataset was built from a different sealed offline bundle")
    config["dataset"]["resolved_artifact_path"] = str(artifact_path)
    config["dataset"]["artifact_id"] = artifact_manifest["dataset_artifact_id"]
    config["dataset"]["records_sha256"] = artifact_manifest["records_sha256"]

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    arithmetic = _batch_arithmetic(config["trainer"], world_size)
    run_dir = _run_directory(config, smoke)
    if resume_from_checkpoint:
        resume_path = Path(resume_from_checkpoint).expanduser().resolve()
        if not resume_path.is_dir():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        if run_dir not in resume_path.parents or not re.fullmatch(r"checkpoint-\d+", resume_path.name):
            raise RuntimeError(
                "resume checkpoint must be a checkpoint-N directory inside this experiment run"
            )
        resume_from_checkpoint = str(resume_path)
    frozen_config = {key: value for key, value in config.items() if key != "_config_path"}
    frozen_path = run_dir / "frozen_config.yaml"
    if run_dir.exists():
        if not resume_from_checkpoint:
            raise FileExistsError(
                f"refusing to reuse run directory without --resume-from-checkpoint: {run_dir}"
            )
        if (run_dir / "training_complete.json").exists():
            raise RuntimeError("the requested experiment is already complete and cannot be resumed")
        existing = load_yaml(frozen_path, expand_environment=False)
        existing.pop("_config_path", None)
        if canonical_json(existing) != canonical_json(frozen_config):
            raise RuntimeError("resume config differs from the frozen launched config")
    else:
        if resume_from_checkpoint:
            raise FileNotFoundError("resume requested but run directory does not exist")
        run_dir.mkdir(parents=True)
        dump_yaml(frozen_config, frozen_path)

    provenance = runtime_provenance(_project_root())
    provenance.update(
        {
            "installed_miq_grpo_tree_sha256": source_tree_sha256(Path(__file__).resolve().parent),
            "experiment_id": config["experiment_id"],
            "task_family": config["task_family"],
            "dataset_artifact_id": artifact_manifest["dataset_artifact_id"],
            "dataset_records_sha256": artifact_manifest["records_sha256"],
            "dataset_manifest_sha256": sha256_file(artifact_path / "manifest.json"),
            "model_id": MODEL_ID,
            "model_revision": config["model"]["expected_revision"],
            "base_model_tree_sha256": assets_manifest["model_identity"]["tree_sha256"],
            "trl_expected_version": TRL_VERSION,
            "grpo_arithmetic": arithmetic,
            "resume_from_checkpoint": resume_from_checkpoint,
            "smoke_run": smoke,
            "smoke_max_steps": smoke_max_steps if smoke else None,
            "smoke_stop_after_step": smoke_stop_after_step,
            "training_assets_manifest": assets_manifest,
            "offline_bundle_files": offline_bundle_files,
        }
    )
    if resume_from_checkpoint:
        resume_component = os.environ.get("SLURM_JOB_ID") or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        provenance_path = run_dir / f"resume_provenance_{resume_component}.json"
    else:
        provenance_path = run_dir / "provenance.json"
    write_json_exclusive(provenance_path, provenance)

    from datasets import load_from_disk
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    if __import__("trl").__version__ != TRL_VERSION:
        raise RuntimeError(f"expected trl=={TRL_VERSION}, got {__import__('trl').__version__}")
    dataset = load_from_disk(str(artifact_path / "dataset"))
    if smoke:
        minimum_rows = max(
            int(config["trainer"]["num_generations"]),
            int(config["trainer"]["per_device_train_batch_size"])
            * int(config["trainer"]["gradient_accumulation_steps"]),
        )
        dataset = dataset.select(range(min(minimum_rows, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    tokenizer_state = {
        "chat_template": tokenizer.chat_template,
        "chat_template_sha256": __import__("hashlib")
        .sha256((tokenizer.chat_template or "").encode("utf-8"))
        .hexdigest(),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
    }
    if (
        not tokenizer.chat_template
        or tokenizer.eos_token_id is None
        or tokenizer.pad_token_id is None
        or tokenizer.padding_side != "left"
    ):
        raise RuntimeError("Qwen tokenizer lacks the required chat, EOS, pad, or left-padding contract")
    prompt_regression = _render_prompt_regression(
        dataset, tokenizer, int(config["trainer"]["max_completion_length"])
    )
    regression_payload = {"tokenizer": tokenizer_state, "prompt_regression": prompt_regression}
    regression_path = run_dir / "prompt_tokenizer_regression.json"
    if resume_from_checkpoint:
        if canonical_json(read_json(regression_path)) != canonical_json(regression_payload):
            raise RuntimeError("tokenizer or rendered-prompt regression changed across resume")
    else:
        write_json_exclusive(regression_path, regression_payload)

    metrics_path = run_dir / "logs" / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    class JsonlMetricsCallback(TrainerCallback):
        def on_log(
            self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
        ):
            del args, control, kwargs
            if state.is_world_process_zero and logs:
                payload = {"logged_at": utc_now(), "step": state.global_step, **logs}
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
            del args, kwargs
            forced_smoke_stop = (
                smoke_stop_after_step is not None and state.global_step >= smoke_stop_after_step
            )
            if _PREEMPT_REQUESTED or forced_smoke_stop:
                control.should_save = True
                control.should_training_stop = True
            return control

    signal.signal(signal.SIGUSR1, _signal_preemption)
    signal.signal(signal.SIGTERM, _signal_preemption)
    grpo_config = GRPOConfig(**_grpo_arguments(config, run_dir, smoke, smoke_max_steps))
    reward_function = make_trl_reward(config["task_family"], config["reward"])
    trainer = GRPOTrainer(
        model=str(model_path),
        reward_funcs=reward_function,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[JsonlMetricsCallback()],
    )
    print(json.dumps({"grpo_arithmetic": arithmetic, "run_dir": str(run_dir)}, indent=2), flush=True)
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    forced_smoke_stop = (
        smoke_stop_after_step is not None and trainer.state.global_step >= smoke_stop_after_step
    )
    if _PREEMPT_REQUESTED or forced_smoke_stop:
        checkpoint = run_dir / f"checkpoint-{trainer.state.global_step}"
        if not checkpoint.is_dir():
            raise RuntimeError(f"interrupted training did not save a checkpoint: {checkpoint}")
        preemption_component = os.environ.get("SLURM_JOB_ID") or datetime.now(UTC).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        preemption_component = re.sub(r"[^A-Za-z0-9_.-]", "_", preemption_component)
        write_json_exclusive(
            run_dir / f"preempted_step_{trainer.state.global_step}_job_{preemption_component}.json",
            {
                "step": trainer.state.global_step,
                "recorded_at": utc_now(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "checkpoint": str(checkpoint.resolve()),
                "forced_smoke_interruption": forced_smoke_stop,
            },
        )
        return run_dir

    final_model = run_dir / "final_model"
    if final_model.exists():
        raise FileExistsError(f"refusing to overwrite final model: {final_model}")
    trainer.save_model(str(final_model))
    tokenizer.save_pretrained(str(final_model))
    final_model_identity = directory_identity(final_model)
    write_json_exclusive(
        run_dir / "training_complete.json",
        {
            "completed_at": utc_now(),
            "experiment_id": config["experiment_id"],
            "task_family": config["task_family"],
            "base_model_id": MODEL_ID,
            "base_model_revision": config["model"]["expected_revision"],
            "base_model_tree_sha256": assets_manifest["model_identity"]["tree_sha256"],
            "dataset_artifact_id": artifact_manifest["dataset_artifact_id"],
            "dataset_records_sha256": artifact_manifest["records_sha256"],
            "global_step": trainer.state.global_step,
            "training_loss": getattr(result, "training_loss", None),
            "final_model": str(final_model.resolve()),
            "final_model_identity": final_model_identity,
            "official_benchmark_evaluated": False,
            "smoke_run": smoke,
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-max-steps", type=int, default=1)
    parser.add_argument("--smoke-stop-after-step", type=int)
    args = parser.parse_args()
    run_dir = train(
        args.config.expanduser().resolve(),
        resume_from_checkpoint=args.resume_from_checkpoint,
        smoke=args.smoke,
        smoke_max_steps=args.smoke_max_steps,
        smoke_stop_after_step=args.smoke_stop_after_step,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
