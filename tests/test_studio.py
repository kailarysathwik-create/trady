"""Unit and security regression tests for TRADE9 Claymorphic Web Studio."""

import json
import sys
import threading
import time
import urllib.error
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from trady.studio.server import (
    GLOBAL_CONTEXT,
    StudioHandler,
    ThreadingHTTPServer,
    start_studio_server,
)


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
    """Verify that GET / returns the Claymorphism HTML page with injected auth token."""
    with urlopen(f"{studio_server}/") as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "<title>TRADE9 Studio" in content
        assert "clay-card" in content
        assert "terminal-box" in content
        assert "window.__STUDIO_AUTH_TOKEN__" in content


def test_studio_api_config_roundtrip(studio_server: str, tmp_path: Path) -> None:
    """Verify GET and POST /api/config endpoints with authorized token."""
    orig_dir = Path.cwd()
    try:
        # 1. GET /api/config
        with urlopen(f"{studio_server}/api/config") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["platform"] == "TRADE9"
            assert "project_dir" in data

        # 2. POST /api/config with valid dir and auth token
        req = Request(
            f"{studio_server}/api/config",
            data=json.dumps({"project_dir": str(tmp_path)}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Studio-Token": GLOBAL_CONTEXT.auth_token,
            },
        )
        with urlopen(req) as resp:
            assert resp.status == 200
            res_data = json.loads(resp.read().decode("utf-8"))
            assert res_data["success"] is True
            assert res_data["project_dir"] == str(tmp_path)
    finally:
        restore_req = Request(
            f"{studio_server}/api/config",
            data=json.dumps({"project_dir": str(orig_dir)}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Studio-Token": GLOBAL_CONTEXT.auth_token,
            },
        )
        with urlopen(restore_req) as resp:
            assert resp.status == 200


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
    echo_cmd = f'"{sys.executable}" -c "print(\'TRADE9_TEST_OK\')"'
    req = Request(
        f"{studio_server}/api/terminal/run",
        data=json.dumps({"command": echo_cmd}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Studio-Token": GLOBAL_CONTEXT.auth_token,
        },
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["status"] == "running"
        job_id = res["job_id"]

    # Poll output
    output_lines = []
    for _ in range(25):
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


def test_studio_api_assistant_status(studio_server: str) -> None:
    """Verify GET /api/assistant/status returns assistant diagnostics."""
    with urlopen(f"{studio_server}/api/assistant/status") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "is_running" in data
        assert "target_model" in data
        assert "available_tools" in data
        assert len(data["available_tools"]) == 6


def test_studio_terminal_utf8_output(studio_server: str) -> None:
    """Verify terminal runner safely handles UTF-8 bytes without charmap errors."""
    echo_cmd = (
        f'"{sys.executable}" -c '
        f"\"import sys; sys.stdout.buffer.write(b'\\xe2\\x96\\x88\\x8f\\n'); "
        f'sys.stdout.flush()"'
    )
    req = Request(
        f"{studio_server}/api/terminal/run",
        data=json.dumps({"command": echo_cmd}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Studio-Token": GLOBAL_CONTEXT.auth_token,
        },
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        job_id = res["job_id"]

    lines = []
    for _ in range(25):
        time.sleep(0.15)
        with urlopen(
            f"{studio_server}/api/terminal/poll?job_id={job_id}&offset=0"
        ) as poll_resp:
            poll_data = json.loads(poll_resp.read().decode("utf-8"))
            lines = poll_data.get("lines", [])
            if not poll_data.get("is_running", True):
                break

    full_output = "".join(lines)
    assert "[Execution Error]" not in full_output
    assert poll_data.get("exit_code") == 0


# ==============================================================================
# SECURITY REGRESSION TESTS
# ==============================================================================


def test_terminal_run_unauthorized_fails(studio_server: str) -> None:
    """Security check: Requests without X-Studio-Token return 401 Unauthorized."""
    echo_cmd = f'"{sys.executable}" -c "print(\'FAIL\')"'
    req = Request(
        f"{studio_server}/api/terminal/run",
        data=json.dumps({"command": echo_cmd}).encode("utf-8"),
        headers={"Content-Type": "application/json"},  # Missing X-Studio-Token!
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urlopen(req)
    assert exc_info.value.code == 401


def test_terminal_run_rejected_command(studio_server: str) -> None:
    """Security check: Non-allowlisted binaries return 403 Forbidden."""
    bad_commands = [
        "whoami",
        "calc.exe",
        "powershell -Command Get-Process",
        "cmd.exe /c dir",
    ]
    for bad_cmd in bad_commands:
        req = Request(
            f"{studio_server}/api/terminal/run",
            data=json.dumps({"command": bad_cmd}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Studio-Token": GLOBAL_CONTEXT.auth_token,
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 403


def test_terminal_run_metacharacters_rejected(studio_server: str) -> None:
    """Security check: Shell operators and chaining are rejected with 403."""
    chained_commands = [
        "trade9 health; whoami",
        "trade9 health && calc.exe",
        "trade9 health | grep test",
        "trade9 health > output.txt",
        "trade9 `whoami`",
    ]
    for chained in chained_commands:
        req = Request(
            f"{studio_server}/api/terminal/run",
            data=json.dumps({"command": chained}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Studio-Token": GLOBAL_CONTEXT.auth_token,
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 403


def test_hostile_origin_rejected(studio_server: str) -> None:
    """Security check: Hostile Origin headers are rejected with 403."""
    req = Request(
        f"{studio_server}/api/health",
        headers={"Origin": "http://malicious-attacker.com"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urlopen(req)
    assert exc_info.value.code == 403


def test_invalid_workspace_path_rejected(studio_server: str) -> None:
    """Security check: Attempting to point workspace to system dirs returns 403."""
    forbidden_paths = ["C:\\Windows", "/etc", "/bin", "C:\\"]
    for f_path in forbidden_paths:
        req = Request(
            f"{studio_server}/api/config",
            data=json.dumps({"project_dir": f_path}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Studio-Token": GLOBAL_CONTEXT.auth_token,
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code in (400, 403)


def test_studio_server_remote_bind_prevention() -> None:
    """Security check: Remote binding without allow_remote raises ValueError."""
    with pytest.raises(ValueError, match="SAFETY VIOLATION"):
        start_studio_server(host="0.0.0.0", port=8998, allow_remote=False)
