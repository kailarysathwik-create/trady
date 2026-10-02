"""Configuration loader supporting YAML parsing and environment variable overrides."""

import os
from pathlib import Path
from typing import Any

import yaml

from trady.config.settings import Settings


def find_project_root() -> Path:
    """Detect the project root by searching for pyproject.toml."""
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current


def get_default_config_path() -> Path:
    """Return the default configuration file path."""
    project_root = find_project_root()
    return project_root / "configs" / "default.yaml"


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(config_data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides (prefixed with TRADY_)."""
    data = config_data.copy()

    # System overrides
    if env := os.getenv("TRADY_ENV"):
        data.setdefault("system", {})["environment"] = env
    if seed := os.getenv("TRADY_RANDOM_SEED"):
        data.setdefault("system", {})["random_seed"] = int(seed)

    # Logging overrides
    if log_level := os.getenv("TRADY_LOG_LEVEL"):
        data.setdefault("logging", {})["level"] = log_level.upper()
    if log_format := os.getenv("TRADY_LOG_FORMAT"):
        data.setdefault("logging", {})["format"] = log_format.lower()

    # Hardware overrides
    if device := os.getenv("TRADY_DEVICE"):
        data.setdefault("hardware", {})["device"] = device.lower()
    if max_workers := os.getenv("TRADY_MAX_WORKERS"):
        data.setdefault("hardware", {})["max_workers"] = int(max_workers)
    if batch_size := os.getenv("TRADY_BATCH_SIZE"):
        data.setdefault("hardware", {})["batch_size"] = int(batch_size)

    # Safety overrides (e.g. attempting to enable live trading will
    # trigger validator error)
    if live_trading := os.getenv("TRADY_LIVE_TRADING_ENABLED"):
        data.setdefault("safety", {})["live_trading_enabled"] = (
            live_trading.strip().lower() in ("true", "1", "yes")
        )

    return data


def load_config(config_path: str | Path | None = None) -> Settings:
    """Load, merge, and validate TRADY configuration.

    Args:
        config_path: Optional explicit path to configuration YAML file.
            If None, checks TRADY_CONFIG_PATH env var, then default.yaml.

    Returns:
        Validated, immutable Settings object.
    """
    path_to_load: Path | None = None

    if config_path is not None:
        path_to_load = Path(config_path)
    elif env_path := os.getenv("TRADY_CONFIG_PATH"):
        path_to_load = Path(env_path)
    else:
        default_candidate = get_default_config_path()
        if default_candidate.is_file():
            path_to_load = default_candidate

    raw_data: dict[str, Any] = {}
    if path_to_load and path_to_load.is_file():
        with open(path_to_load, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw_data = loaded

    data_with_env = _apply_env_overrides(raw_data)
    return Settings.model_validate(data_with_env)
