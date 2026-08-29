from __future__ import annotations

from pathlib import Path

import pytest

from miq_grpo.config import ConfigError, load_yaml
from miq_grpo.train import _batch_arithmetic, _grpo_arguments, _require_offline_mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_grpo_batch_arithmetic() -> None:
    trainer = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_generations": 8,
    }
    arithmetic = _batch_arithmetic(trainer, world_size=1)
    assert arithmetic["generation_batch_size"] == 8
    assert arithmetic["unique_prompts_per_generation_batch"] == 1

    with pytest.raises(ConfigError, match="GRPO arithmetic"):
        _batch_arithmetic({**trainer, "num_generations": 3}, world_size=1)


def test_training_requires_all_offline_switches(monkeypatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="offline mode"):
        _require_offline_mode()

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    _require_offline_mode()


def test_question_generation_symbols_exist_only_in_offline_builder() -> None:
    forbidden = (
        "generate_count_question",
        "generate_index_question",
        "generate_constraint_question",
    )
    offenders: list[str] = []
    for path in sorted((PROJECT_ROOT / "src" / "miq_grpo").glob("*.py")):
        if path.name == "dataset_builder.py":
            continue
        text = path.read_text(encoding="utf-8")
        if any(symbol in text for symbol in forbidden):
            offenders.append(path.name)
    assert offenders == []


def test_training_configs_never_name_the_official_benchmark() -> None:
    paths = [PROJECT_ROOT / "configs" / "preprocessing.yaml"] + sorted(
        (PROJECT_ROOT / "configs" / "experiments").glob("*.yaml")
    )
    assert all("moleculariq-v0.0" not in path.read_text(encoding="utf-8") for path in paths)


def test_full_finetuning_keeps_fp32_master_weights_with_bf16_compute(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MIQ_DATA_BUNDLE", "/frozen/dataset")
    monkeypatch.setenv("MIQ_MODEL_PATH", "/staged/model")
    monkeypatch.setenv("MIQ_MODEL_REVISION", "a" * 40)
    monkeypatch.setenv("MIQ_RUN_ROOT", "/persistent/runs")
    for path in sorted((PROJECT_ROOT / "configs" / "experiments").glob("*.yaml")):
        config = load_yaml(path)
        assert config["model"]["dtype"] == "float32"
        assert config["trainer"]["bf16"] is True
        assert config["trainer"]["generation_kwargs"] == {"do_sample": True}
        smoke = _grpo_arguments(config, tmp_path / path.stem, True, smoke_max_steps=2)
        assert smoke["max_steps"] == 2
        assert smoke["model_init_kwargs"]["dtype"] == "float32"
