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


def test_cli_data_parser_subcommands() -> None:
    """Ensure data subcommands are correctly parsed."""
    parser = build_parser()
    args_val = parser.parse_args(["data", "validate", "dummy.parquet"])
    assert args_val.command == "data"
    assert args_val.data_command == "validate"
    assert args_val.dataset == "dummy.parquet"

    args_norm = parser.parse_args(
        ["data", "normalize", "raw.json", "-o", "proc.parquet", "--deduplicate"]
    )
    assert args_norm.command == "data"
    assert args_norm.data_command == "normalize"
    assert args_norm.input == "raw.json"
    assert args_norm.output == "proc.parquet"
    assert args_norm.deduplicate is True

    args_query = parser.parse_args(["data", "query", "SELECT 1"])
    assert args_query.command == "data"
    assert args_query.data_command == "query"
    assert args_query.sql == "SELECT 1"


def test_cli_data_workflow(
    tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test full CLI data pipeline execution from generation to query."""
    from trady.data.cli import handle_data_command

    parser = build_parser()

    # 1. Generate synthetic data
    raw_json = str(tmp_path / "raw_spy.json")
    args_gen = parser.parse_args(
        [
            "data",
            "generate-synthetic",
            "--symbol",
            "SPY",
            "--bars",
            "15",
            "-o",
            raw_json,
        ]
    )
    exit_gen = handle_data_command(args_gen)
    assert exit_gen == 0

    # 2. Validate raw data
    args_val = parser.parse_args(["data", "validate", raw_json])
    exit_val = handle_data_command(args_val)
    assert exit_val == 0

    # 3. Normalize into Parquet
    out_parquet = str(tmp_path / "proc_spy.parquet")
    args_norm = parser.parse_args(["data", "normalize", raw_json, "-o", out_parquet])
    exit_norm = handle_data_command(args_norm)
    assert exit_norm == 0

    # 4. Inspect Parquet
    args_insp = parser.parse_args(["data", "inspect", out_parquet])
    exit_insp = handle_data_command(args_insp)
    assert exit_insp == 0

    # 5. Query via DuckDB
    parquet_sql_path = out_parquet.replace("\\", "/")
    args_query = parser.parse_args(
        [
            "data",
            "query",
            f"SELECT symbol, COUNT(*) as n FROM '{parquet_sql_path}' GROUP BY symbol",
        ]
    )
    exit_query = handle_data_command(args_query)
    assert exit_query == 0
