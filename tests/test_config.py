"""Unit tests for configuration loading and validation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from trady.config.loader import load_config
from trady.config.settings import Settings


def test_default_settings_initialization(default_settings: Settings) -> None:
    """Verify default configuration attributes and constraints."""
    assert default_settings.system.app_name == "TRADY"
    assert default_settings.system.random_seed == 42
    assert default_settings.logging.level == "INFO"
    assert default_settings.logging.format == "text"
    assert default_settings.hardware.max_workers == 4
    assert default_settings.hardware.batch_size == 64
    assert default_settings.safety.live_trading_enabled is False
    assert default_settings.safety.paper_trading_only is True


def test_load_config_from_file(temp_config_file: Path) -> None:
    """Test loading configuration from a YAML file."""
    settings = load_config(temp_config_file)
    assert settings.system.app_name == "TRADY_TEST"
    assert settings.system.environment == "testing"
    assert settings.system.random_seed == 123
    assert settings.logging.level == "DEBUG"
    assert settings.hardware.max_workers == 2
    assert settings.hardware.batch_size == 32


def test_environment_variable_override(temp_config_file: Path) -> None:
    """Test that environment variables take precedence over YAML configuration."""
    with patch.dict(
        os.environ,
        {
            "TRADY_ENV": "production_test",
            "TRADY_LOG_LEVEL": "WARNING",
            "TRADY_MAX_WORKERS": "8",
            "TRADY_RANDOM_SEED": "999",
        },
    ):
        settings = load_config(temp_config_file)
        assert settings.system.environment == "production_test"
        assert settings.logging.level == "WARNING"
        assert settings.hardware.max_workers == 8
        assert settings.system.random_seed == 999


def test_invalid_hardware_workers_raises() -> None:
    """Ensure worker count validation enforces limits."""
    with pytest.raises(ValidationError):
        Settings.model_validate({"hardware": {"max_workers": 0}})

    with pytest.raises(ValidationError):
        Settings.model_validate({"hardware": {"max_workers": 999}})
