"""Experiment-config loading, environment expansion, and direct path resolution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_config_path(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    package_relative = Path(__file__).resolve().parent / path
    if package_relative.is_file():
        return package_relative
    raise FileNotFoundError(f"Configuration does not exist: {value}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Config references unset environment variable: {name}")
            return os.environ[name]

        return ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path_value: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_config_path(path_value)
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    resolved = _expand(payload)
    for key in ("name", "manifest", "datasets", "methods", "output_dir", "evaluation"):
        if key not in resolved:
            raise ValueError(f"Config is missing required key '{key}': {path}")
    if not isinstance(resolved["methods"], dict) or not resolved["methods"]:
        raise ValueError("Config must declare at least one method")
    return path, resolved


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def resolve_prediction_path(method: dict[str, Any], split: str, cohort: str, stem: str) -> Path:
    roots = method.get("prediction_roots", {})
    root = roots.get(split)
    if not isinstance(root, str):
        raise ValueError(f"Method is missing prediction root for split '{split}'")
    template = method.get("prediction_template")
    if not isinstance(template, str) or not template:
        raise ValueError("Method is missing prediction_template")
    return Path(root) / template.format(split=split, cohort=cohort, stem=stem, method=method.get("id", ""))


def resolve_dataset_path(dataset: dict[str, Any], field: str, stem: str, suffix: str) -> Path | None:
    root = dataset.get(field)
    if root is None:
        return None
    if not isinstance(root, str):
        raise ValueError(f"Dataset field '{field}' must be a path string")
    return Path(root) / f"{stem}{suffix}"
