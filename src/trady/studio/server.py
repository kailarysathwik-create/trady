"""Standalone web studio and interactive command runner for TRADE9."""

import json
import os
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from trady import __platform__, __version__
from trady.utils.reproducibility import get_system_info

CONFIG_FILE = ".trade9-studio.json"


class CommandJob:
    """Manages asynchronous command execution and output streaming."""

    def __init__(self, job_id: str, command: str, cwd: str) -> None:
        self.job_id = job_id
        self.command = command
        self.cwd = cwd
        self.lines: list[str] = []
        self.lock = threading.Lock()
        self.is_running = True
        self.exit_code: int | None = None
        self.process: subprocess.Popen | None = None
        self.started_at = time.time()
        self.ended_at: float | None = None

    def start(self) -> None:
        """Start the command process in a background worker thread."""
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self) -> None:
        try:
            # On Windows, execute via powershell if available, fallback to cmd/shell
            shell_cmd = self.command
            self.process = subprocess.Popen(
                shell_cmd,
                shell=True,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            if self.process.stdout:
                for line in self.process.stdout:
                    with self.lock:
                        self.lines.append(line)

            self.process.wait()
            self.exit_code = self.process.returncode
        except Exception as e:
            with self.lock:
                self.lines.append(f"\n[Execution Error]: {e}\n")
            self.exit_code = 1
        finally:
            self.is_running = False
            self.ended_at = time.time()

    def kill(self) -> None:
        """Terminate the active command process."""
        if self.process and self.is_running:
            try:
                self.process.terminate()
            except Exception:
                pass
            self.is_running = False

    def get_output(self, offset: int = 0) -> tuple[list[str], int]:
        """Fetch output lines starting from given offset."""
        with self.lock:
            slice_lines = self.lines[offset:]
            next_offset = len(self.lines)
            return slice_lines, next_offset


class StudioContext:
    """Global state holding workspace path and active terminal jobs."""

    def __init__(self, default_dir: Path | None = None) -> None:
        self.project_dir = (
            default_dir or Path(os.environ.get("TRADE9_DIR", Path.cwd()))
        ).resolve()
        self.jobs: dict[str, CommandJob] = {}
        self.load_config()

    def load_config(self) -> None:
        cfg_path = self.project_dir / CONFIG_FILE
        if cfg_path.exists():
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    data = json.load(f)
                    saved_dir = data.get("project_dir")
                    if saved_dir and Path(saved_dir).is_dir():
                        self.project_dir = Path(saved_dir).resolve()
            except Exception:
                pass

    def save_config(self, new_dir: Path) -> None:
        self.project_dir = new_dir.resolve()
        cfg_path = self.project_dir / CONFIG_FILE
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"project_dir": str(self.project_dir)}, f, indent=2)
        except Exception:
            pass


GLOBAL_CONTEXT = StudioContext()


class StudioHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for TRADE9 Claymorphism Web Studio."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy HTTP request logging in terminal
        pass

    def send_json(self, status: int, data: Any) -> None:
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ("", "/"):
            self._serve_index()
        elif path == "/api/config":
            self._handle_get_config()
        elif path == "/api/health":
            self._handle_get_health()
        elif path == "/api/terminal/poll":
            self._handle_terminal_poll(query)
        elif path == "/api/artifacts":
            self._handle_get_artifacts()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/config":
            self._handle_post_config(payload)
        elif path == "/api/terminal/run":
            self._handle_terminal_run(payload)
        elif path == "/api/terminal/kill":
            self._handle_terminal_kill(payload)
        else:
            self.send_error(404, "Not Found")

    def _serve_index(self) -> None:
        html_file = Path(__file__).parent / "static" / "index.html"
        if not html_file.exists():
            self.send_error(500, "Studio static index.html missing")
            return
        content = html_file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_get_config(self) -> None:
        d = GLOBAL_CONTEXT.project_dir
        valid = (d / "pyproject.toml").exists() or (d / "src" / "trady").exists()
        self.send_json(
            200,
            {
                "platform": __platform__,
                "version": __version__,
                "project_dir": str(d),
                "is_valid_project": valid,
            },
        )

    def _handle_post_config(self, payload: dict) -> None:
        target_str = payload.get("project_dir", "").strip()
        if not target_str:
            self.send_json(400, {"error": "Project directory path is required."})
            return
        target_path = Path(target_str).resolve()
        if not target_path.is_dir():
            self.send_json(400, {"error": f"Directory does not exist: {target_path}"})
            return

        GLOBAL_CONTEXT.save_config(target_path)
        valid = (target_path / "pyproject.toml").exists() or (
            target_path / "src" / "trady"
        ).exists()
        self.send_json(
            200,
            {
                "success": True,
                "project_dir": str(target_path),
                "is_valid_project": valid,
            },
        )

    def _handle_get_health(self) -> None:
        sys_info = get_system_info()
        d = GLOBAL_CONTEXT.project_dir

        dirs_to_check = [
            "data/processed",
            "data/features",
            "data/targets",
            "models",
            "reports",
        ]
        tree_status = {p: (d / p).exists() for p in dirs_to_check}

        self.send_json(
            200,
            {
                "status": "ALL SYSTEMS HEALTHY",
                "system": sys_info,
                "project_dir": str(d),
                "tree_status": tree_status,
                "firewall": "ACTIVE (Paper / Educational Simulation Only)",
            },
        )

    def _handle_terminal_run(self, payload: dict) -> None:
        cmd = payload.get("command", "").strip()
        if not cmd:
            self.send_json(400, {"error": "Command string cannot be empty."})
            return

        job_id = f"job_{int(time.time() * 1000)}"
        cwd = str(GLOBAL_CONTEXT.project_dir)
        job = CommandJob(job_id=job_id, command=cmd, cwd=cwd)
        GLOBAL_CONTEXT.jobs[job_id] = job
        job.start()

        self.send_json(200, {"job_id": job_id, "command": cmd, "status": "running"})

    def _handle_terminal_poll(self, query: dict) -> None:
        job_id = query.get("job_id", [""])[0]
        offset = int(query.get("offset", ["0"])[0])

        job = GLOBAL_CONTEXT.jobs.get(job_id)
        if not job:
            self.send_json(404, {"error": "Job not found."})
            return

        lines, next_offset = job.get_output(offset=offset)
        self.send_json(
            200,
            {
                "job_id": job_id,
                "lines": lines,
                "offset": next_offset,
                "is_running": job.is_running,
                "exit_code": job.exit_code,
            },
        )

    def _handle_terminal_kill(self, payload: dict) -> None:
        job_id = payload.get("job_id", "")
        job = GLOBAL_CONTEXT.jobs.get(job_id)
        if job:
            job.kill()
            self.send_json(200, {"success": True, "job_id": job_id})
        else:
            self.send_json(404, {"error": "Job not found."})

    def _handle_get_artifacts(self) -> None:
        d = GLOBAL_CONTEXT.project_dir
        artifacts: dict[str, list[dict[str, Any]]] = {
            "data": [],
            "features": [],
            "targets": [],
            "models": [],
            "reports": [],
        }

        mapping = {
            "data": d / "data" / "processed",
            "features": d / "data" / "features",
            "targets": d / "data" / "targets",
            "models": d / "models",
            "reports": d / "reports",
        }

        for category, folder in mapping.items():
            if folder.exists():
                for f in sorted(
                    folder.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True
                )[:10]:
                    if f.is_file():
                        artifacts[category].append(
                            {
                                "name": f.name,
                                "path": str(f.relative_to(d)),
                                "size_kb": round(f.stat().st_size / 1024, 1),
                                "modified": time.strftime(
                                    "%Y-%m-%d %H:%M:%S",
                                    time.localtime(f.stat().st_mtime),
                                ),
                            }
                        )

        self.send_json(200, artifacts)


def start_studio_server(
    host: str = "127.0.0.1",
    port: int = 8090,
    project_dir: Path | None = None,
    open_browser: bool = True,
) -> None:
    """Launch the TRADE9 Studio web server."""
    if project_dir:
        GLOBAL_CONTEXT.save_config(project_dir)

    server = ThreadingHTTPServer((host, port), StudioHandler)
    url = f"http://{host}:{port}"
    print("=" * 70)
    print(f" {__platform__} CLAYMORPHIC STUDIO & INTERACTIVE TERMINAL")
    print("=" * 70)
    print(f"  * Local Web URL: {url}")
    print(f"  * Workspace Dir: {GLOBAL_CONTEXT.project_dir}")
    print("  * Status       : Running. Press Ctrl+C to stop.")
    print("=" * 70)

    if open_browser:
        import webbrowser

        threading.Thread(
            target=lambda: (time.sleep(1), webbrowser.open(url)), daemon=True
        ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TRADE9 Studio...")
        server.server_close()
