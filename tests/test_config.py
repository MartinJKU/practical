from __future__ import annotations

from pathlib import Path

import pytest

from miq_grpo.config import ConfigError, load_yaml


def test_load_yaml_expands_environment_without_shell_semantics(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text('path: "${MIQ_TEST_ROOT}/artifact"\nliteral: "$MIQ_TEST_ROOT"\n', encoding="utf-8")
    monkeypatch.setenv("MIQ_TEST_ROOT", "/persistent/work")

    loaded = load_yaml(config)

    assert loaded["path"] == "/persistent/work/artifact"
    assert loaded["literal"] == "$MIQ_TEST_ROOT"
    assert loaded["_config_path"] == str(config.resolve())


def test_load_yaml_fails_on_missing_environment_variable(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("path: ${MIQ_MISSING_FOR_TEST}\n", encoding="utf-8")
    monkeypatch.delenv("MIQ_MISSING_FOR_TEST", raising=False)

    with pytest.raises(ConfigError, match="MIQ_MISSING_FOR_TEST"):
        load_yaml(config)
