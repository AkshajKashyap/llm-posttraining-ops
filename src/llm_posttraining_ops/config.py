"""Application configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm_posttraining_ops.data.schemas import SPLIT_NAMES


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Configuration for prepared datasets."""

    output_dir: Path
    sft_split_sizes: dict[str, int]
    preference_split_sizes: dict[str, int]


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration."""

    seed: int
    data: DataConfig


def _require_mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a mapping")
    return value


def _parse_split_sizes(value: object, location: str) -> dict[str, int]:
    raw_sizes = _require_mapping(value, location)
    unknown_splits = set(raw_sizes) - set(SPLIT_NAMES)
    missing_splits = set(SPLIT_NAMES) - set(raw_sizes)
    if unknown_splits:
        names = ", ".join(sorted(unknown_splits))
        raise ConfigError(f"{location} contains unsupported split names: {names}")
    if missing_splits:
        names = ", ".join(sorted(missing_splits))
        raise ConfigError(f"{location} is missing split names: {names}")

    sizes: dict[str, int] = {}
    for split in SPLIT_NAMES:
        size = raw_sizes[split]
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ConfigError(f"{location}.{split} must be a positive integer")
        sizes[split] = size
    return sizes


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an application configuration from a YAML file."""

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    root = _require_mapping(raw_config, "config")
    seed = root.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("seed must be a non-negative integer")

    data = _require_mapping(root.get("data"), "data")
    output_dir = data.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("data.output_dir must be a non-empty string")

    return AppConfig(
        seed=seed,
        data=DataConfig(
            output_dir=Path(output_dir),
            sft_split_sizes=_parse_split_sizes(
                data.get("sft_split_sizes"), "data.sft_split_sizes"
            ),
            preference_split_sizes=_parse_split_sizes(
                data.get("preference_split_sizes"), "data.preference_split_sizes"
            ),
        ),
    )
