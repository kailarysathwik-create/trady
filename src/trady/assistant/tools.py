"""Approved research tools for TRADY AI Research Assistant.

These tools provide controlled, read-only inspection of TRADY's quantitative
datasets, feature registry, trained models, and backtest reports.
No shell commands, file modifications, or real-money operations are permitted.
"""

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trady.features.registry import registry
from trady.models.base import TradyModel
from trady.utils.logging import get_logger

logger = get_logger("assistant.tools")


def inspect_datasets(
    project_dir: str | Path | None = None,
    dataset_type: str | None = None,
) -> dict[str, Any]:
    """Inspect available market, feature, and target datasets in TRADY.

    Args:
        project_dir: Root project directory (defaults to current working dir).
        dataset_type: Optional filter ('processed', 'features', 'targets').

    Returns:
        Structured breakdown of available datasets, row counts, and columns.
    """
    root = Path(project_dir) if project_dir else Path.cwd()
    data_dir = root / "data"

    types_to_scan = ["processed", "features", "targets"]
    if dataset_type and dataset_type in types_to_scan:
        types_to_scan = [dataset_type]

    summary: dict[str, list[dict[str, Any]]] = {}

    for dtype in types_to_scan:
        summary[dtype] = []
        target_folder = data_dir / dtype
        if not target_folder.exists():
            continue

        for p_file in target_folder.glob("*.parquet"):
            try:
                table_meta = pq.read_metadata(p_file)
                summary[dtype].append(
                    {
                        "filename": p_file.name,
                        "size_bytes": p_file.stat().st_size,
                        "num_rows": table_meta.num_rows,
                        "num_columns": table_meta.num_columns,
                        "columns": [
                            table_meta.schema.names[i]
                            for i in range(min(15, table_meta.num_columns))
                        ]
                        + (["..."] if table_meta.num_columns > 15 else []),
                    }
                )
            except Exception as e:
                summary[dtype].append(
                    {
                        "filename": p_file.name,
                        "error": f"Could not inspect parquet: {e}",
                    }
                )

    total_files = sum(len(files) for files in summary.values())
    return {
        "status": "success",
        "total_datasets_found": total_files,
        "datasets": summary,
    }


def get_feature_metadata(
    feature_group: str | None = None,
) -> dict[str, Any]:
    """Retrieve metadata and mathematical specifications for registered features.

    Args:
        feature_group: Optional group filter (e.g. 'PRICE', 'MOMENTUM',
            'VOLATILITY', 'VOLUME', 'PRICE_STRUCTURE').

    Returns:
        Dictionary of registered feature names, descriptions, groups,
        and lookback windows.
    """
    if feature_group:
        all_features = registry.list_by_group(feature_group)
    else:
        all_features = registry.list_features()

    filtered = []
    for f in all_features:
        filtered.append(
            {
                "name": f.name,
                "group": f.feature_group,
                "description": f.description,
                "lookback_period": f.lookback_period,
                "is_causal": f.is_point_in_time,
            }
        )

    return {
        "status": "success",
        "total_features": len(filtered),
        "filter_group": feature_group,
        "features": filtered,
    }


def get_experiment_results(
    experiment_id: str | None = None,
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Retrieve performance metrics and training metadata for serialized models.

    Args:
        experiment_id: Optional ID or substring to match specific model.
        project_dir: Root directory of TRADY workspace.

    Returns:
        Dictionary containing model architectures, target definitions, and metrics.
    """
    root = Path(project_dir) if project_dir else Path.cwd()
    models_dir = root / "models"
    if not models_dir.exists():
        return {
            "status": "not_found",
            "message": "No 'models' directory found in workspace.",
            "experiments": [],
        }

    results: list[dict[str, Any]] = []
    for model_path in models_dir.glob("*.joblib"):
        try:
            model = TradyModel.load(model_path)
            meta = model.metadata
            if not meta:
                continue

            # Filter if experiment_id requested
            if (
                experiment_id
                and experiment_id.lower() not in meta.experiment_id.lower()
                and experiment_id.lower() not in meta.model_id.lower()
            ):
                continue

            results.append(
                {
                    "model_file": model_path.name,
                    "model_id": meta.model_id,
                    "experiment_id": meta.experiment_id,
                    "model_type": meta.model_type,
                    "task_type": meta.task_type,
                    "target_name": meta.target_name,
                    "feature_count": meta.feature_count,
                    "train_samples": meta.train_period.get("num_samples", 0),
                    "test_samples": meta.test_period.get("num_samples", 0),
                    "metrics": meta.metrics,
                }
            )
        except Exception as e:
            logger.warning("Could not read model file %s: %s", model_path.name, e)

    return {
        "status": "success",
        "count": len(results),
        "experiments": results,
    }


def compare_experiments(
    exp_id_a: str,
    exp_id_b: str,
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare evaluation metrics of two completed machine learning experiments.

    Args:
        exp_id_a: First experiment/model ID or substring.
        exp_id_b: Second experiment/model ID or substring.
        project_dir: Root directory of TRADY workspace.

    Returns:
        Side-by-side metric comparison and out-of-sample delta analysis.
    """
    all_exps = get_experiment_results(project_dir=project_dir).get("experiments", [])

    def _find_exp(query: str) -> dict[str, Any] | None:
        q = query.lower()
        for e in all_exps:
            if q in e["model_id"].lower() or q in e["experiment_id"].lower():
                return e
        return None

    model_a = _find_exp(exp_id_a)
    model_b = _find_exp(exp_id_b)

    if not model_a or not model_b:
        return {
            "status": "error",
            "message": (
                f"Could not find both experiments for comparison. "
                f"Found A: {bool(model_a)}, Found B: {bool(model_b)}"
            ),
            "available_models": [e["model_id"] for e in all_exps],
        }

    metrics_a = model_a.get("metrics", {}).get("test", {})
    metrics_b = model_b.get("metrics", {}).get("test", {})

    comparison: dict[str, dict[str, Any]] = {}
    all_keys = set(metrics_a.keys()).union(metrics_b.keys())
    for k in sorted(all_keys):
        va = metrics_a.get(k)
        vb = metrics_b.get(k)
        diff = (va - vb) if (va is not None and vb is not None) else None
        comparison[k] = {
            model_a["model_id"]: va,
            model_b["model_id"]: vb,
            "delta (A - B)": round(diff, 5) if diff is not None else None,
        }

    return {
        "status": "success",
        "model_a": {
            "id": model_a["model_id"],
            "type": model_a["model_type"],
            "target": model_a["target_name"],
        },
        "model_b": {
            "id": model_b["model_id"],
            "type": model_b["model_type"],
            "target": model_b["target_name"],
        },
        "test_metrics_comparison": comparison,
    }


def get_backtest_metrics(
    report_name: str | None = None,
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Retrieve simulated performance metrics from completed backtest runs.

    Args:
        report_name: Optional filter for a specific report name.
        project_dir: Root directory of TRADY workspace.

    Returns:
        Structured quantitative performance metrics (Sharpe, Drawdown,
        Win Rate, Returns).
    """
    root = Path(project_dir) if project_dir else Path.cwd()
    reports_dir = root / "reports"
    if not reports_dir.exists():
        return {
            "status": "not_found",
            "message": "No reports directory found in workspace.",
            "reports": [],
        }

    # Check both root reports/ and nested reports/backtests/*/
    all_html = set(reports_dir.glob("*.html")) | set(
        reports_dir.glob("backtests/*/*.html")
    )
    html_reports = sorted(all_html, key=lambda p: p.stat().st_mtime, reverse=True)

    all_trades = set(reports_dir.glob("*_trades.parquet")) | set(
        reports_dir.glob("backtests/*/trades.parquet")
    )
    parquet_trades = list(all_trades)

    reports_info = []
    for r in html_reports:
        if report_name and report_name.lower() not in r.name.lower():
            continue
        rep_entry = {
            "report_file": r.name,
            "path": str(r.relative_to(root)),
            "size_bytes": r.stat().st_size,
            "modified_at": r.stat().st_mtime,
        }
        metrics_file = r.parent / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    m_data = json.load(f)
                rep_entry["metrics"] = {
                    "total_return": m_data.get("total_return"),
                    "annualized_return": m_data.get("annualized_return")
                    or m_data.get("cagr"),
                    "sharpe_ratio": m_data.get("sharpe_ratio"),
                    "sortino_ratio": m_data.get("sortino_ratio"),
                    "max_drawdown": m_data.get("max_drawdown"),
                    "win_rate": m_data.get("win_rate"),
                    "profit_factor": m_data.get("profit_factor"),
                    "total_trades": m_data.get("total_trades"),
                }
            except Exception:
                pass
        reports_info.append(rep_entry)

    trade_logs = []
    for t in parquet_trades:
        try:
            m = pq.read_metadata(t)
            trade_logs.append(
                {
                    "trade_file": t.name,
                    "trade_count": m.num_rows,
                }
            )
        except Exception:
            pass

    return {
        "status": "success",
        "total_html_reports": len(reports_info),
        "reports": reports_info,
        "trade_logs": trade_logs,
        "metrics_guide": [
            "Total Return",
            "CAGR",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Win Rate",
            "Profit Factor",
        ],
    }


def explain_research_report(
    identifier: str | None = None,
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate grounded facts from both latest ML models and backtest
    results for AI explanation.

    Args:
        identifier: Optional keyword matching a specific model or backtest.
        project_dir: Root directory of TRADY workspace.

    Returns:
        Aggregated quantitative evidence for grounded explanation.
    """
    models_info = get_experiment_results(
        experiment_id=identifier, project_dir=project_dir
    )
    backtest_info = get_backtest_metrics(project_dir=project_dir)
    data_info = inspect_datasets(project_dir=project_dir)

    return {
        "status": "success",
        "evidence": {
            "latest_experiments": models_info.get("experiments", [])[:3],
            "backtest_reports": backtest_info.get("reports", [])[:3],
            "dataset_summary": data_info.get("datasets", {}),
            "safety_firewall": "ACTIVE (Paper simulation strictly enforced)",
        },
    }


# -----------------------------------------------------------------------------
# Allowlisted Tool Dispatcher and Schema Definitions
# -----------------------------------------------------------------------------

RESEARCH_TOOLS_DISPATCHER = {
    "inspect_datasets": inspect_datasets,
    "get_feature_metadata": get_feature_metadata,
    "get_experiment_results": get_experiment_results,
    "compare_experiments": compare_experiments,
    "get_backtest_metrics": get_backtest_metrics,
    "explain_research_report": explain_research_report,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_datasets",
            "description": (
                "Inspect available market, feature, and target datasets in "
                "TRADY workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_type": {
                        "type": "string",
                        "description": (
                            "Filter by dataset type: 'processed', 'features', "
                            "or 'targets'"
                        ),
                        "enum": ["processed", "features", "targets"],
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature_metadata",
            "description": (
                "Retrieve metadata, lookback windows, and formulas for "
                "TRADY registered causal features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_group": {
                        "type": "string",
                        "description": (
                            "Filter by group: 'PRICE', 'MOMENTUM', "
                            "'VOLATILITY', 'VOLUME', or 'STRUCTURE'"
                        ),
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_experiment_results",
            "description": (
                "Retrieve performance metrics, target definitions, and "
                "sample counts for trained models in workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {
                        "type": "string",
                        "description": ("Optional model ID or experiment ID filter."),
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_experiments",
            "description": (
                "Compare out-of-sample test metrics side-by-side between "
                "two trained models."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exp_id_a": {
                        "type": "string",
                        "description": ("First model ID or experiment ID."),
                    },
                    "exp_id_b": {
                        "type": "string",
                        "description": ("Second model ID or experiment ID."),
                    },
                },
                "required": ["exp_id_a", "exp_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_backtest_metrics",
            "description": (
                "Retrieve simulated backtest performance reports and trade logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "report_name": {
                        "type": "string",
                        "description": ("Optional report filename filter."),
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_research_report",
            "description": (
                "Aggregate grounded quantitative evidence across models and "
                "backtests to explain findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": (
                            "Optional keyword matching a specific model or "
                            "backtest report."
                        ),
                    }
                },
            },
        },
    },
]
