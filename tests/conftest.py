"""Shared pytest fixtures for TRADY tests."""

from pathlib import Path

import pytest

from trady.config.settings import Settings


@pytest.fixture
def default_settings() -> Settings:
    """Provide a pristine default Settings instance."""
    return Settings()


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary valid YAML config file."""
    yaml_content = """
system:
  app_name: "TRADY_TEST"
  environment: "testing"
  random_seed: 123

logging:
  level: "DEBUG"
  format: "text"

hardware:
  device: "cpu"
  max_workers: 2
  batch_size: 32

safety:
  live_trading_enabled: false
  paper_trading_only: true
  disclaimer_acknowledged: true
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")
    return config_file
