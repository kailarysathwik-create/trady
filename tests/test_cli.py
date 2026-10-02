"""Unit tests for TRADY CLI subcommands and options."""

import pytest

from trady.cli import build_parser, run_health_check


def test_cli_parser_commands() -> None:
    """Ensure parser registers all expected subcommands."""
    parser = build_parser()
    args_health = parser.parse_args(["health"])
    assert args_health.command == "health"

    args_version = parser.parse_args(["version"])
    assert args_version.command == "version"

    args_config = parser.parse_args(["config"])
    assert args_config.command == "config"

    args_system = parser.parse_args(["system"])
    assert args_system.command == "system"


def test_cli_version_flag() -> None:
    """Ensure --version flag is parsed."""
    parser = build_parser()
    args = parser.parse_args(["--version"])
    assert args.version is True


def test_cli_health_check_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """Ensure run_health_check executes cleanly and returns exit code 0."""
    exit_code = run_health_check()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "SYSTEM HEALTH & DIAGNOSTIC CHECK" in captured.out
    expected_firewall_msg = (
        "Live Trading Firewall: [ACTIVE - Real-money trading blocked]"
    )
    assert expected_firewall_msg in captured.out
