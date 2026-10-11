"""Unit tests for TRADE9 Claymorphic Web Studio and Interactive Terminal."""

import json
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from trady.studio.server import StudioHandler, ThreadingHTTPServer


@pytest.fixture(scope="module")
def studio_server():
    """Spin up local Studio server on ephemeral port for tests."""
    port = 8999
    server = ThreadingHTTPServer(("127.0.0.1", port), StudioHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def test_studio_serves_html(studio_server: str) -> None:
    """Verify that GET / returns the Claymorphism HTML page."""
    with urlopen(f"{studio_server}/") as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "<title>TRADE9 Studio" in content
        assert "clay-card" in content
        assert "terminal-box" in content


def test_studio_api_config_roundtrip(studio_server: str, tmp_path: Path) -> None:
    """Verify GET and POST /api/config endpoints."""
    # 1. GET /api/config
    with urlopen(f"{studio_server}/api/config") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["platform"] == "TRADE9"
        assert "project_dir" in data

    # 2. POST /api/config with valid dir
    req = Request(
        f"{studio_server}/api/config",
        data=json.dumps({"project_dir": str(tmp_path)}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        res_data = json.loads(resp.read().decode("utf-8"))
        assert res_data["success"] is True
        assert res_data["project_dir"] == str(tmp_path)


def test_studio_api_health(studio_server: str) -> None:
    """Verify GET /api/health returns diagnostic data."""
    with urlopen(f"{studio_server}/api/health") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "ALL SYSTEMS HEALTHY"
        assert "system" in data
        assert "cpu_count" in data["system"]


def test_studio_terminal_run_and_poll(studio_server: str) -> None:
    """Verify executing command via terminal API and polling output."""
    echo_cmd = "echo TRADE9_TEST_OK"
    req = Request(
        f"{studio_server}/api/terminal/run",
        data=json.dumps({"command": echo_cmd}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["status"] == "running"
        job_id = res["job_id"]

    # Poll output
    output_lines = []
    for _ in range(20):
        time.sleep(0.15)
        with urlopen(
            f"{studio_server}/api/terminal/poll?job_id={job_id}&offset=0"
        ) as poll_resp:
            poll_data = json.loads(poll_resp.read().decode("utf-8"))
            output_lines = poll_data.get("lines", [])
            if not poll_data.get("is_running", True):
                break

    full_text = "".join(output_lines)
    assert "TRADE9_TEST_OK" in full_text
