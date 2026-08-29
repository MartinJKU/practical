"""Strict YAML configuration loading and environment resolution."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is invalid."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any, *, environ: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, environ=environ) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, environ=environ) for item in value]
    if not isinstance(value, str):
        return value

    missing = sorted(set(_ENV_PATTERN.findall(value)) - set(environ))
    if missing:
        raise ConfigError(f"Missing environment variables in config: {', '.join(missing)}")
    return _ENV_PATTERN.sub(lambda match: environ[match.group(1)], value)


def load_yaml(path: str | Path, *, expand_environment: bool = True) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    data = copy.deepcopy(raw)
    if expand_environment:
        data = _expand(data, environ=dict(os.environ))
    data["_config_path"] = str(config_path)
    return data


def require_keys(mapping: dict[str, Any], keys: tuple[str, ...], *, context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{context} is missing required keys: {', '.join(missing)}")


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True, default_flow_style=False)
