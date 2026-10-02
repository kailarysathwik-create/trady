# TRADY: Educational Quantitative Research & Paper-Trading Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing: Pytest](https://img.shields.io/badge/testing-pytest-yellow.svg)](https://docs.pytest.org/)

---

> ### ⚠️ Critical Educational & Regulatory Disclaimers
>
> 1. **No Guaranteed Profits**: TRADY is strictly an educational research and paper-trading platform. Its sole purpose is to research whether statistical and machine-learning models can identify repeatable patterns in historical financial-market data. Financial markets are inherently noisy and stochastic; **TRADY makes no claims, warranties, or implications of profit**.
> 2. **No Real-Money Trading or Live Execution**: TRADY does **NOT** implement live brokerage APIs, order-routing execution systems, or autonomous real-money trading. All simulations, signals, and simulated orders are strictly virtual paper-trading constructs.

---

## 1. System Mission & Architecture Overview

TRADY is designed as a modular, reproducible platform for quantitative researchers. The architecture enforces strict boundaries between data ingestion, causal feature engineering, statistical modeling, simulation backtesting, risk limits, and out-of-sample evaluation.

```
trady/
│
├── README.md               # Overview, specifications, and disclaimers
├── pyproject.toml          # Build configuration, dependencies, and tooling
├── .gitignore              # Source-control ignore rules
├── .env.example            # Environment variable template
│
├── configs/                # Platform configuration profiles (YAML)
│   └── default.yaml        # Base system, hardware, and safety settings
│
├── data/                   # Data tiers (excluded from source control)
│   ├── raw/                # Unaltered source market records
│   ├── interim/            # Cleaned and standardized data slices
│   └── processed/          # Causal feature matrices ready for research
│
├── notebooks/              # Exploratory research notebooks
├── reports/                # Experiment evaluation summaries & audit reports
├── scripts/                # Utility scripts for data maintenance and local runs
│
├── src/
│   └── trady/
│       ├── __init__.py     # Package root with version and disclaimer constants
│       ├── cli.py          # Unified CLI entry point (`trady health`, `trady config`, etc.)
│       ├── config/         # Pydantic configuration schemas and YAML loader
│       ├── data/           # Historical market data contracts and validation
│       ├── features/       # Causal feature engineering protocols
│       ├── models/         # Research model protocols and metadata tracking
│       ├── backtesting/    # Paper-trading simulation engine contracts
│       ├── risk/           # Portfolio and drawdown safety guardrails
│       ├── experiments/    # Experiment registry and provenance tracking
│       ├── evaluation/     # Statistical validation and repeatability metrics
│       └── utils/          # Structured logging and reproducibility helpers
│
├── tests/                  # Automated unit and integration test suite
└── docs/                   # Architecture reference and hardware specs
```

---

## 2. Target Compute & Multi-Machine Workflow

TRADY is architected to support a dual-machine distributed workflow:

* **Main Coding Machine**: Focused on architecture, software development, documentation, and Git source control.
* **Lenovo LOQ Workstation**: Focused on heavy computation, statistical model experiments, backtesting, and paper-trading simulations.
* **GitHub Repository**: Central synchronization bridge (`https://github.com/kailarysathwik-create/trady.git`).

### Target Hardware Profile (Lenovo LOQ)
* **Processor (CPU)**: Intel Core i5-12450HX (8 cores / 12 threads)
* **Graphics (GPU)**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB / 6 GB VRAM)
* **System Memory (RAM)**: 16 GB DDR5
* **Operating System**: Windows

### Hardware Resource Constraints
- **Workers**: `max_workers` defaults to `4` to prevent CPU thermal throttling and leave resources for OS operations.
- **Batch Sizing**: `batch_size` defaults to `64` to prevent Out-Of-Memory (OOM) faults on the RTX 3050 VRAM.
- **Memory Soft Cap**: In-memory datasets should remain below 12 GB (`memory_limit_mb: 12288`) to prevent swapping on 16 GB RAM.

---

## 3. Quickstart & Installation

### Prerequisites
- Python 3.11 or Python 3.12
- Git
- `uv` (recommended) or standard `pip` / `venv`

### Setup Instructions

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/kailarysathwik-create/trady.git
   cd trady
   ```

2. **Create and activate a virtual environment**:
   Using `uv`:
   ```powershell
   uv venv .venv
   .venv\Scripts\activate
   ```
   Or using standard Python:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install the package in editable mode**:
   ```powershell
   uv pip install -e ".[dev]"
   # or: pip install -e ".[dev]"
   ```

4. **Initialize environment configuration**:
   ```powershell
   Copy-Item .env.example .env
   ```

---

## 4. Platform Verification & CLI

TRADY provides a command-line interface with a comprehensive diagnostic health check:

```powershell
# Run the platform health check
trady health

# Inspect version and regulatory disclaimers
trady version

# Inspect active resolved configuration
trady config

# Inspect target compute hardware diagnostics
trady system
```

---

## 5. Development Quality Standards

To maintain engineering excellence and reproducibility, all code must pass formatting, linting, and automated tests.

### Formatting & Linting (Ruff)
```powershell
# Check formatting
uv run ruff format --check .

# Run linter
uv run ruff check .

# Apply auto-fixes
uv run ruff check --fix .
uv run ruff format .
```

### Running Tests (Pytest)
```powershell
# Run all unit tests
uv run pytest
```

---

## 6. Engineering & Safety Rules

- **Minimal Dependencies**: Core platform relies strictly on `pydantic` and `pyyaml`.
- **Zero Live Trading**: `live_trading_enabled` must remain `false`. Any attempt to set it to `true` immediately triggers an unrecoverable exception halting execution.
- **No Secrets in Source Control**: Credentials and `.env` files are strictly gitignored.
- **Immutability & Determinism**: All raw data slices and orders are immutable dataclasses. Random seeds are explicitly tracked via `trady.utils.set_seed()`.
