"""Unit and integration tests for TRADY Phase 2: Local AI Research Assistant."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trady.assistant.client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from trady.assistant.service import AssistantService
from trady.assistant.tools import (
    compare_experiments,
    explain_research_report,
    get_backtest_metrics,
    get_experiment_results,
    get_feature_metadata,
    inspect_datasets,
)
from trady.config.settings import AssistantConfig
from trady.models.linear import TradyLogisticRegression

# -----------------------------------------------------------------------------
# 1. Ollama Client Tests
# -----------------------------------------------------------------------------


def test_ollama_client_check_status_success() -> None:
    """Verify check_status parses installed models when Ollama is running."""
    client = OllamaClient(default_model="qwen3:4b")
    mock_payload = {
        "models": [
            {"name": "qwen3:4b"},
            {"name": "qwen2.5-coder:7b"},
        ]
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        status = client.check_status()
        assert status["is_running"] is True
        assert status["target_model_installed"] is True
        assert "qwen3:4b" in status["installed_models"]


def test_ollama_client_check_status_offline() -> None:
    """Verify check_status handles offline daemon without crashing."""
    import urllib.error

    client = OllamaClient()
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        status = client.check_status()
        assert status["is_running"] is False
        assert status["target_model_installed"] is False
        assert "unreachable" in status["error"].lower()


def test_ollama_client_chat_model_not_found() -> None:
    """Verify missing model raises OllamaModelNotFoundError."""
    import urllib.error

    client = OllamaClient(default_model="non_existent_model:99b")
    mock_err = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error":"model not found"}'),
    )

    with patch("urllib.request.urlopen", side_effect=mock_err):
        with pytest.raises(OllamaModelNotFoundError, match="not found in Ollama"):
            client.chat([{"role": "user", "content": "hi"}])


def test_ollama_client_chat_timeout() -> None:
    """Verify timeout raises OllamaTimeoutError."""

    client = OllamaClient(timeout_seconds=5.0)
    with patch("urllib.request.urlopen", side_effect=TimeoutError()):
        with pytest.raises(OllamaTimeoutError, match="timed out"):
            client.chat([{"role": "user", "content": "hi"}])


# -----------------------------------------------------------------------------
# 2. Research Tools Tests
# -----------------------------------------------------------------------------


def test_tool_inspect_datasets(tmp_path: Path) -> None:
    """Verify inspect_datasets correctly reads Parquet metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Setup dummy data folder
    proc_dir = tmp_path / "data" / "processed"
    proc_dir.mkdir(parents=True)

    table = pa.table(
        {
            "timestamp": [1, 2, 3],
            "open": [10.0, 11.0, 12.0],
            "close": [10.5, 11.5, 12.5],
        }
    )
    pq.write_table(table, proc_dir / "test.parquet")

    result = inspect_datasets(project_dir=tmp_path)
    assert result["status"] == "success"
    assert result["total_datasets_found"] == 1
    assert "processed" in result["datasets"]
    assert result["datasets"]["processed"][0]["num_rows"] == 3
    assert "close" in result["datasets"]["processed"][0]["columns"]


def test_tool_get_feature_metadata() -> None:
    """Verify get_feature_metadata returns registered causal features."""
    all_res = get_feature_metadata()
    assert all_res["status"] == "success"
    assert all_res["total_features"] >= 40
    assert any(f["name"] == "return_1d" for f in all_res["features"])

    mom_res = get_feature_metadata(feature_group="MOMENTUM")
    assert mom_res["status"] == "success"
    assert all(f["group"].upper() == "MOMENTUM" for f in mom_res["features"])
    assert any(f["name"] == "roc_5d" for f in mom_res["features"])


def test_tool_get_and_compare_experiments(tmp_path: Path) -> None:
    """Verify experiment inspection and side-by-side comparison."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    # Create dummy model A
    mA = TradyLogisticRegression(random_seed=42)
    mA.fit(
        np.array([[1.0], [2.0]]),
        np.array([0, 1]),
        feature_names=("f1",),
        target_name="target_binary_up_5d",
    )
    from trady.models.metadata import ModelMetadata

    mA.metadata = ModelMetadata(
        experiment_id="exp_001",
        model_id="logreg_alpha",
        model_type="logistic_regression",
        task_type="classification",
        feature_names=("f1",),
        feature_count=1,
        target_name="target_binary_up_5d",
        target_definition={},
        train_period={"num_samples": 50},
        val_period=None,
        test_period={"num_samples": 20},
        model_parameters={},
        random_seed=42,
        metrics={"test": {"accuracy": 0.65, "roc_auc": 0.70}},
    )
    mA.save(models_dir / "logreg_alpha.joblib")

    # Create dummy model B
    mB = TradyLogisticRegression(random_seed=43)
    mB.fit(
        np.array([[1.0], [2.0]]),
        np.array([0, 1]),
        feature_names=("f1",),
        target_name="target_binary_up_5d",
    )
    mB.metadata = ModelMetadata(
        experiment_id="exp_002",
        model_id="logreg_beta",
        model_type="logistic_regression",
        task_type="classification",
        feature_names=("f1",),
        feature_count=1,
        target_name="target_binary_up_5d",
        target_definition={},
        train_period={"num_samples": 50},
        val_period=None,
        test_period={"num_samples": 20},
        model_parameters={},
        random_seed=43,
        metrics={"test": {"accuracy": 0.75, "roc_auc": 0.80}},
    )
    mB.save(models_dir / "logreg_beta.joblib")

    # 1. Test get_experiment_results
    exp_res = get_experiment_results(project_dir=tmp_path)
    assert exp_res["status"] == "success"
    assert exp_res["count"] == 2

    # 2. Test compare_experiments
    comp_res = compare_experiments("logreg_alpha", "logreg_beta", project_dir=tmp_path)
    assert comp_res["status"] == "success"
    assert "accuracy" in comp_res["test_metrics_comparison"]
    diff_acc = comp_res["test_metrics_comparison"]["accuracy"]["delta (A - B)"]
    assert diff_acc == pytest.approx(-0.10)


def test_tool_get_backtest_metrics_and_explain(tmp_path: Path) -> None:
    """Verify backtest metrics inspection and report explanation aggregation."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "backtest_summary.html").write_text(
        "<html>Report</html>", encoding="utf-8"
    )

    bt_res = get_backtest_metrics(project_dir=tmp_path)
    assert bt_res["status"] == "success"
    assert bt_res["total_html_reports"] == 1

    exp_res = explain_research_report(project_dir=tmp_path)
    assert exp_res["status"] == "success"
    assert "safety_firewall" in exp_res["evidence"]
    assert "ACTIVE" in exp_res["evidence"]["safety_firewall"]


# -----------------------------------------------------------------------------
# 3. Assistant Service & Tool Calling Flow Tests
# -----------------------------------------------------------------------------


def test_assistant_service_tool_execution_loop(tmp_path: Path) -> None:
    """Verify tool execution loop with mocked Ollama function calls."""
    cfg = AssistantConfig(timeout_seconds=10.0)
    service = AssistantService(config=cfg, project_dir=tmp_path)

    # 1. Mock first response: LLM decides to call inspect_datasets
    first_resp = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "inspect_datasets",
                        "arguments": {"dataset_type": "processed"},
                    }
                }
            ],
        }
    }

    # 2. Mock second response: LLM synthesizes findings
    final_resp = {
        "message": {
            "role": "assistant",
            "content": "In your workspace, 0 processed datasets were discovered.",
        }
    }

    with patch.object(
        service.client, "chat", side_effect=[first_resp, final_resp]
    ) as mock_chat:
        resp = service.chat("What datasets do we have?")
        assert resp.status == "ok"
        assert len(resp.tool_calls_made) == 1
        assert resp.tool_calls_made[0]["tool"] == "inspect_datasets"
        assert "0 processed datasets" in resp.content
        assert mock_chat.call_count == 2


def test_assistant_service_handles_offline_gracefully(tmp_path: Path) -> None:
    """Verify service returns clear diagnostic message when Ollama is offline."""
    cfg = AssistantConfig()
    service = AssistantService(config=cfg, project_dir=tmp_path)

    with patch.object(
        service.client,
        "chat",
        side_effect=OllamaConnectionError("Daemon offline"),
    ):
        resp = service.chat("Hello?")
        assert resp.status == "error"
        assert "TRADY AI Unavailable" in resp.content


def test_assistant_safety_blocks_unauthorized_tools(tmp_path: Path) -> None:
    """Verify assistant service strictly rejects arbitrary tool names."""
    service = AssistantService(project_dir=tmp_path)
    res = service.execute_tool("execute_bash_command", {"cmd": "rm -rf /"})
    assert res["status"] == "error"
    assert "not in allowlisted" in res["error"]
