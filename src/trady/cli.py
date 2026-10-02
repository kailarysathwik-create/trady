"""Command-line interface (CLI) entry point for the TRADY platform."""

import argparse
import sys
from typing import NoReturn

from trady import DISCLAIMER, __platform__, __version__
from trady.config.loader import find_project_root, get_default_config_path, load_config
from trady.utils.logging import get_logger, setup_logging
from trady.utils.reproducibility import get_system_info, set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the TRADY CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="trady",
        description=(
            f"{__platform__} v{__version__}: Educational Quantitative Research & "
            "Paper-Trading Platform.\n\n"
            "DISCLAIMER: For educational research only. Does not guarantee profits and "
            "does not execute real-money trades."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Display version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Health check command
    health_parser = subparsers.add_parser(
        "health",
        help="Run comprehensive platform and hardware health checks",
    )
    health_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as critical failures",
    )

    # Version command
    subparsers.add_parser(
        "version",
        help="Print version and educational disclaimers",
    )

    # Config inspect command
    subparsers.add_parser(
        "config",
        help="Inspect current resolved configuration",
    )

    # System info command
    subparsers.add_parser(
        "system",
        help="Inspect target compute environment and hardware specifications",
    )

    return parser


def run_health_check(config_path: str | None = None, strict: bool = False) -> int:
    """Execute comprehensive platform and hardware health checks.

    Returns:
        0 if healthy, 1 if critical failure detected.
    """
    logger = get_logger("cli.health")
    print("=" * 70)
    print(f" {__platform__} SYSTEM HEALTH & DIAGNOSTIC CHECK")
    print("=" * 70)

    critical_failures: list[str] = []
    warnings: list[str] = []

    # 1. Platform & Python runtime
    print("\n[1/5] Runtime & Environment:")
    sys_info = get_system_info()
    py_ver = sys_info.get("python_version", "unknown")
    print(f"  - Python: {py_ver} ({sys_info.get('machine', 'unknown')})")
    print(f"  - OS: {sys_info.get('os', 'unknown')} {sys_info.get('os_release', '')}")
    if sys.version_info < (3, 11):  # noqa: UP036
        critical_failures.append(f"Python 3.11+ required, detected {py_ver}")
        print("  [FAIL] Python version is below 3.11 requirement")
    else:
        print("  [OK] Python runtime meets requirements")

    # 2. Hardware profile (tailored for Lenovo LOQ: i5-12450HX, RTX 3050, 16GB RAM)
    print("\n[2/5] Hardware Profile & Resource Capacity:")
    cpu_cores = sys_info.get("cpu_count", 1)
    ram_mb = sys_info.get("ram_total_mb", "N/A")
    avail_mb = sys_info.get("ram_available_mb", "N/A")
    gpu_name = sys_info.get("gpu_name", "None")
    cuda_avail = sys_info.get("cuda_available", False)

    gpu_desc = (
        f"Enabled ({gpu_name})"
        if cuda_avail
        else "CPU-only mode (No active CUDA torch device)"
    )
    print(f"  - Logical CPU Threads: {cpu_cores}")
    print(f"  - Total RAM: {ram_mb} MB (Available: {avail_mb} MB)")
    print(f"  - GPU / CUDA: {gpu_desc}")
    print("  [OK] Compute specifications audited")

    # 3. Directory structure verification
    print("\n[3/5] Project Directory Integrity:")
    project_root = find_project_root()
    required_dirs = [
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "notebooks",
        "reports",
        "scripts",
        "src/trady",
        "tests",
        "docs",
    ]
    missing_dirs: list[str] = []
    for rel_dir in required_dirs:
        dir_path = project_root / rel_dir
        if dir_path.is_dir():
            print(f"  - {rel_dir:.<25} [OK]")
        else:
            print(f"  - {rel_dir:.<25} [MISSING]")
            missing_dirs.append(rel_dir)

    if missing_dirs:
        critical_failures.append(
            f"Missing required directories: {', '.join(missing_dirs)}"
        )

    # 4. Configuration and Safety Guard verification
    print("\n[4/5] Configuration & Safety Guardrails:")
    try:
        settings = load_config(config_path)
        print(
            f"  - App Name: {settings.system.app_name} "
            f"(Environment: {settings.system.environment})"
        )
        print(f"  - Random Seed: {settings.system.random_seed}")

        # Check safety invariants
        if settings.safety.live_trading_enabled:
            critical_failures.append(
                "SAFETY FIREWALL BREACH: live_trading_enabled is TRUE!"
            )
            print("  - Live Trading Firewall: [BREACHED - DANGER]")
        else:
            print("  - Live Trading Firewall: [ACTIVE - Real-money trading blocked]")

        if not settings.safety.paper_trading_only:
            critical_failures.append(
                "SAFETY FIREWALL BREACH: paper_trading_only is FALSE!"
            )
            print("  - Simulation Mode: [BREACHED]")
        else:
            print("  - Simulation Mode: [ACTIVE - Paper trading strictly enforced]")

        print("  [OK] Configuration schema validated")
    except Exception as exc:
        critical_failures.append(f"Configuration load failed: {exc}")
        print(f"  [FAIL] Failed to load configuration: {exc}")

    # 5. Reproducibility & Logging subsystem
    print("\n[5/5] Reproducibility & Logging Subsystems:")
    try:
        applied_seed = set_seed(42)
        print(f"  - Seed Setter: [OK] (Applied seed: {applied_seed})")
    except Exception as exc:
        critical_failures.append(f"Reproducibility seeding failed: {exc}")
        print(f"  - Seed Setter: [FAIL] ({exc})")

    try:
        setup_logging()
        logger.debug("Health check debug pulse")
        print("  - Structured Logger: [OK]")
    except Exception as exc:
        critical_failures.append(f"Logging setup failed: {exc}")
        print(f"  - Structured Logger: [FAIL] ({exc})")

    # Diagnostic Summary
    print("\n" + "=" * 70)
    print(" HEALTH SUMMARY:")
    print("=" * 70)
    if critical_failures:
        print(f" STATUS: FAILED ({len(critical_failures)} critical issue(s))")
        for fail in critical_failures:
            print(f"  [X] {fail}")
        return 1

    if warnings:
        print(f" STATUS: WARNING ({len(warnings)} non-critical warning(s))")
        for warn in warnings:
            print(f"  [!] {warn}")
        if strict:
            return 1

    print(" STATUS: ALL SYSTEMS HEALTHY")
    print(f" NOTICE: {DISCLAIMER}")
    print("=" * 70)
    return 0


def cmd_version() -> None:
    """Print platform version and regulatory disclaimer."""
    print(f"{__platform__} version {__version__}")
    print(f"\n{DISCLAIMER}")


def cmd_config(config_path: str | None = None) -> None:
    """Print the currently resolved configuration."""
    settings = load_config(config_path)
    import yaml

    src_label = config_path or get_default_config_path()
    print(f"# Resolved {__platform__} Configuration (Source: {src_label})")
    dumped = yaml.dump(
        settings.model_dump(mode="json"), default_flow_style=False, sort_keys=False
    )
    print(dumped)


def cmd_system() -> None:
    """Print detailed system and hardware diagnostics."""
    info = get_system_info()
    print("TRADY Target Compute Diagnostic Info:")
    for k, v in info.items():
        print(f"  {k:.<25}: {v}")


def main(argv: list[str] | None = None) -> NoReturn:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        cmd_version()
        sys.exit(0)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "health":
        exit_code = run_health_check(config_path=args.config, strict=args.strict)
        sys.exit(exit_code)
    elif args.command == "version":
        cmd_version()
        sys.exit(0)
    elif args.command == "config":
        cmd_config(config_path=args.config)
        sys.exit(0)
    elif args.command == "system":
        cmd_system()
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
