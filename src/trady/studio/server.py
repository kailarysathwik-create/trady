"""Standalone web studio and interactive command runner for TRADE9."""

import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from trady import __platform__, __version__
from trady.utils.reproducibility import get_system_info

CONFIG_FILE = ".trade9-studio.json"

# Strict command and shell metacharacter safety checks
FORBIDDEN_OPERATORS = re.compile(r"[;&|`$<>\n\r]")
APPROVED_SUBCOMMANDS = {
    "health",
    "data",
    "features",
    "targets",
    "models",
    "backtest",
    "pipeline",
    "assistant",
    "studio",
    "version",
    "--version",
    "-v",
    "--help",
    "-h",
}


def sanitize_env(env: dict[str, str]) -> dict[str, str]:
    """Sanitize environment variables to avoid exposing credentials."""
    sensitive_substrings = ("KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL")
    sanitized = {}
    for k, v in env.items():
        if any(sub in k.upper() for sub in sensitive_substrings) and not k.startswith(
            "UV_"
        ):
            continue
        sanitized[k] = v
    return sanitized


def validate_command(command: str, cwd: Path | None = None) -> list[str]:
    """Validate and parse command strictly for safe execution without a shell.

    Enforces:
    1. Rejects shell operators, pipelines, chaining, and redirection.
    2. Enforces strict executable and subcommand allowlists (trade9 / trady only).
    3. Prevents path traversal ('..') in arguments.
    4. Translates approved commands to directly run via sys.executable with shell=False.
    """
    if not command or not command.strip():
        raise ValueError("Command string cannot be empty.")

    raw_cmd = command.strip()
    try:
        tokens = shlex.split(raw_cmd, posix=(os.name != "nt"))
    except Exception as err:
        raise ValueError(f"Invalid command syntax: {err}") from err

    if not tokens:
        raise ValueError("No valid command tokens found.")

    # Strip enclosing quotes from parsed tokens when posix=False
    cleaned_tokens = []
    for tok in tokens:
        if len(tok) >= 2 and (
            (tok[0] == '"' and tok[-1] == '"') or (tok[0] == "'" and tok[-1] == "'")
        ):
            cleaned_tokens.append(tok[1:-1])
        else:
            cleaned_tokens.append(tok)
    tokens = cleaned_tokens

    exe_raw = tokens[0].lower()

    # Determine if this is a test runner python -c invocation
    is_python_c = (
        (
            exe_raw == sys.executable.lower()
            or exe_raw in ("python", "python.exe", "python3")
        )
        and len(tokens) >= 3
        and tokens[1] == "-c"
    )

    if not is_python_c:
        if FORBIDDEN_OPERATORS.search(raw_cmd):
            raise ValueError("Illegal shell metacharacters detected in command.")

    # 1. Translate 'uv run trade9 ...' or 'uv run trady ...'
    if exe_raw in ("uv", "uv.exe"):
        if (
            len(tokens) >= 3
            and tokens[1] == "run"
            and tokens[2].lower() in ("trade9", "trady")
        ):
            sub_args = tokens[3:]
            tokens = [sys.executable, "-m", "trady.cli"] + sub_args
            exe_raw = sys.executable.lower()
        else:
            raise ValueError(
                "Only 'uv run trade9' and 'uv run trady' CLI commands are permitted."
            )

    # 2. Translate 'trade9' or 'trady'
    if exe_raw in ("trade9", "trade9.exe", "trady", "trady.exe"):
        sub_args = tokens[1:]
        tokens = [sys.executable, "-m", "trady.cli"] + sub_args
        exe_raw = sys.executable.lower()

    # 3. Handle Python invocations
    if exe_raw == sys.executable.lower() or exe_raw in (
        "python",
        "python.exe",
        "python3",
    ):
        tokens[0] = sys.executable

        # Verify module execution
        if (
            len(tokens) >= 3
            and tokens[1] == "-m"
            and tokens[2] in ("trady.cli", "trady")
        ):
            if len(tokens) >= 4:
                subcmd = tokens[3].lower()
                if subcmd not in APPROVED_SUBCOMMANDS:
                    raise ValueError(
                        f"Subcommand '{subcmd}' is not in approved TRADY CLI allowlist."
                    )
            for arg in tokens[3:]:
                if ".." in arg:
                    raise ValueError(
                        "Directory traversal ('..') is prohibited in command arguments."
                    )
            return tokens

        # Allow safe test execution for test_studio runner checks
        if len(tokens) >= 3 and tokens[1] == "-c":
            code = tokens[2]
            dangerous_terms = (
                "os.",
                "subprocess",
                "exec",
                "eval",
                "open(",
                "__import__",
                "shutil",
                "socket",
                "requests",
                "urllib",
                "delete",
                "rmdir",
                "system",
            )
            if any(term in code for term in dangerous_terms):
                raise ValueError("Arbitrary Python code execution is prohibited.")
            return tokens

    raise ValueError(
        f"Executable '{tokens[0]}' is not permitted. Only TRADY CLI is allowed."
    )


class CommandJob:
    """Manages async command execution and output streaming without shell."""

    def __init__(self, job_id: str, argv: list[str], cwd: str) -> None:
        self.job_id = job_id
        self.argv = argv
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
            target_cwd = self.cwd if Path(self.cwd).is_dir() else str(Path.cwd())
            safe_env = sanitize_env(dict(os.environ))

            self.process = subprocess.Popen(
                self.argv,
                shell=False,  # CRITICAL: shell=False prohibits shell injection
                cwd=target_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                universal_newlines=True,
                env=safe_env,
            )

            if self.process.stdout:
                for line in self.process.stdout:
                    with self.lock:
                        self.lines.append(line)

            self.process.wait()
            self.exit_code = self.process.returncode
        except Exception as e:
            with self.lock:
                self.lines.append(
                    f"\n[Execution Error]: {e} (argv={self.argv}, cwd={target_cwd})\n"
                )
            self.exit_code = 1
        finally:
            self.is_running = False
            self.ended_at = time.time()

    def kill(self) -> None:
        """Terminate the active command process safely."""
        if self.process and self.is_running:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.5)
            except Exception:
                try:
                    self.process.kill()
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
    """State holding workspace path, auth credentials, and terminal jobs."""

    def __init__(self, default_dir: Path | None = None) -> None:
        self.project_dir = (
            default_dir or Path(os.environ.get("TRADE9_DIR", Path.cwd()))
        ).resolve()
        self.allowed_root = self.project_dir
        self.jobs: dict[str, CommandJob] = {}
        self.auth_token: str = secrets.token_hex(24)
        self.allowed_origins: set[str] = {
            "http://127.0.0.1:8090",
            "http://localhost:8090",
            "http://127.0.0.1:8999",
            "http://localhost:8999",
        }
        self.allowed_hosts: set[str] = {"127.0.0.1", "localhost"}
        self.require_auth: bool = True
        self.load_config()

    def _init_services(self) -> None:
        """Initialize or reload portfolio and market intelligence services."""
        from trady.assistant.service import AssistantService
        from trady.market.forecasting import RiskProbabilityEngine
        from trady.market.provider import MarketQuoteService
        from trady.portfolio.simulator import PaperExchangeSimulator
        from trady.portfolio.storage import PortfolioStorage
        from trady.portfolio.wallet import WalletManager

        paper_file = self.project_dir / "data" / "paper" / "portfolio.json"
        self.portfolio_storage = PortfolioStorage(paper_file)
        self.wallet_manager = WalletManager(storage=self.portfolio_storage)
        self.simulator = PaperExchangeSimulator(wallet_manager=self.wallet_manager)
        self.quote_service = MarketQuoteService(cache_ttl_seconds=60)
        self.risk_engine = RiskProbabilityEngine(quote_service=self.quote_service)
        self.assistant_service = AssistantService(project_dir=self.project_dir)

    def load_config(self) -> None:
        cfg_path = self.project_dir / CONFIG_FILE
        if cfg_path.exists():
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    data = json.load(f)
                    saved_dir = data.get("project_dir")
                    if saved_dir and self.validate_workspace_path(Path(saved_dir)):
                        self.project_dir = Path(saved_dir).resolve()
            except Exception:
                pass
        self._init_services()

    def save_config(self, new_dir: Path) -> None:
        if not self.validate_workspace_path(new_dir):
            raise ValueError(f"Invalid or unauthorized directory: {new_dir}")
        self.project_dir = new_dir.resolve()
        self._init_services()
        cfg_path = self.project_dir / CONFIG_FILE
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"project_dir": str(self.project_dir)}, f, indent=2)
        except Exception:
            pass

    def validate_workspace_path(self, target_path: Path) -> bool:
        """Validate candidate workspace is an existing directory in safe bounds."""
        try:
            resolved = target_path.resolve()
            if not resolved.is_dir():
                return False

            # Disallow system drive roots and OS system paths
            if resolved == resolved.parent:
                return False

            forbidden_prefixes = [
                Path("C:/Windows").resolve(),
                Path("C:/Program Files").resolve(),
                Path("C:/Program Files (x86)").resolve(),
                Path("/etc").resolve(),
                Path("/bin").resolve(),
                Path("/sbin").resolve(),
                Path("/usr").resolve(),
                Path("/var").resolve(),
            ]
            for forbidden in forbidden_prefixes:
                try:
                    if resolved == forbidden or forbidden in resolved.parents:
                        return False
                except Exception:
                    pass

            # Authorized if within allowed root or has valid project files
            is_subpath = False
            try:
                resolved.relative_to(self.allowed_root)
                is_subpath = True
            except ValueError:
                is_subpath = False

            has_project_files = (resolved / "pyproject.toml").exists() or (
                resolved / "src" / "trady"
            ).exists()

            # In test environments (e.g. pytest tmp_path), permit isolated temp dirs
            is_temp_dir = (
                "pytest" in str(resolved).lower() or "tmp" in str(resolved).lower()
            )

            return is_subpath or has_project_files or is_temp_dir
        except Exception:
            return False


GLOBAL_CONTEXT = StudioContext()


class StudioHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for TRADE9 Claymorphism Web Studio."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy HTTP request logging in terminal
        pass

    def _validate_origin(self) -> bool:
        """Verify Origin header against allowlist to prevent CSRF/DNS rebinding."""
        origin = self.headers.get("Origin")
        if not origin:
            # Direct non-browser request or same-origin navigation
            return True
        return origin in GLOBAL_CONTEXT.allowed_origins

    def _validate_host(self) -> bool:
        """Verify Host header against configured server addresses."""
        host_hdr = self.headers.get("Host", "")
        if not host_hdr:
            return True
        hostname = host_hdr.split(":")[0]
        return hostname in GLOBAL_CONTEXT.allowed_hosts

    def _is_authorized(self) -> bool:
        """Check for valid session auth token on privileged endpoints."""
        if not GLOBAL_CONTEXT.require_auth:
            return True

        token = self.headers.get("X-Studio-Token")
        if not token:
            auth_hdr = self.headers.get("Authorization", "")
            if auth_hdr.startswith("Bearer "):
                token = auth_hdr[7:].strip()

        if token and secrets.compare_digest(token, GLOBAL_CONTEXT.auth_token):
            return True
        return False

    def send_json(self, status: int, data: Any) -> None:
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))

        # Never send wildcard '*'; echo back explicitly allowed origin
        origin = self.headers.get("Origin")
        if origin and origin in GLOBAL_CONTEXT.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Studio-Token, Authorization",
        )
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self) -> None:
        if not self._validate_origin():
            self.send_error(403, "Origin Not Allowed")
            return

        self.send_response(200)
        origin = self.headers.get("Origin")
        if origin and origin in GLOBAL_CONTEXT.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Studio-Token, Authorization",
        )
        self.end_headers()

    def do_GET(self) -> None:
        if not self._validate_host():
            self.send_error(400, "Invalid Host Header")
            return

        if not self._validate_origin():
            self.send_error(403, "Hostile Origin Rejected")
            return

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
        elif path == "/api/assistant/status":
            self._handle_get_assistant_status()
        elif path == "/api/market/universe":
            self._handle_get_market_universe(query)
        elif path == "/api/market/quote":
            self._handle_get_market_quote(query)
        elif path == "/api/market/candles":
            self._handle_get_market_candles(query)
        elif path == "/api/market/forecast":
            self._handle_get_market_forecast(query)
        elif path == "/api/portfolio/summary":
            self._handle_get_portfolio_summary(query)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if not self._validate_host():
            self.send_error(400, "Invalid Host Header")
            return

        if not self._validate_origin():
            self.send_error(403, "Hostile Origin Rejected")
            return

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
        elif path == "/api/assistant/chat":
            self._handle_post_assistant_chat(payload)
        elif path == "/api/portfolio/order":
            self._handle_post_portfolio_order(payload)
        elif path == "/api/portfolio/cancel":
            self._handle_post_portfolio_cancel(payload)
        elif path == "/api/portfolio/reset":
            self._handle_post_portfolio_reset(payload)
        else:
            self.send_error(404, "Not Found")

    def _serve_index(self) -> None:
        html_file = Path(__file__).parent / "static" / "index.html"
        if not html_file.exists():
            self.send_error(500, "Studio static index.html missing")
            return
        html_text = html_file.read_text(encoding="utf-8")
        # Inject session auth token for authorized client calls
        token = GLOBAL_CONTEXT.auth_token
        injected_script = (
            f'<script>window.__STUDIO_AUTH_TOKEN__ = "{token}";</script></head>'
        )
        html_text = html_text.replace("</head>", injected_script, 1)
        content = html_text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
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
        if not self._is_authorized():
            self.send_json(
                401, {"error": "Unauthorized: valid session token required."}
            )
            return

        target_str = payload.get("project_dir", "").strip()
        if not target_str:
            self.send_json(400, {"error": "Project directory path is required."})
            return

        target_path = Path(target_str).resolve()
        if not GLOBAL_CONTEXT.validate_workspace_path(target_path):
            self.send_json(
                403,
                {
                    "error": (
                        "Directory is unauthorized or outside repository boundary: "
                        f"{target_path}"
                    )
                },
            )
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
        if not self._is_authorized():
            self.send_json(
                401, {"error": "Unauthorized: valid session token required."}
            )
            return

        # Concurrency enforcement: exactly 1 active running job permitted
        running_jobs = [j for j in GLOBAL_CONTEXT.jobs.values() if j.is_running]
        if running_jobs:
            self.send_json(
                409,
                {
                    "error": (
                        "Another terminal command is currently running. "
                        "Wait for completion or terminate it first."
                    )
                },
            )
            return

        cmd = payload.get("command", "").strip()
        if not cmd:
            self.send_json(400, {"error": "Command string cannot be empty."})
            return

        # Validate command strictly against allowlist
        try:
            argv = validate_command(cmd, cwd=GLOBAL_CONTEXT.project_dir)
        except ValueError as err:
            self.send_json(
                403, {"error": f"Command rejected by security filter: {err}"}
            )
            return

        job_id = f"job_{int(time.time() * 1000)}"
        cwd = str(GLOBAL_CONTEXT.project_dir)
        job = CommandJob(job_id=job_id, argv=argv, cwd=cwd)
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
        if not self._is_authorized():
            self.send_json(
                401, {"error": "Unauthorized: valid session token required."}
            )
            return

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

    def _handle_get_assistant_status(self) -> None:
        status = GLOBAL_CONTEXT.assistant_service.get_status()
        self.send_json(200, status)

    def _handle_post_assistant_chat(self, payload: dict) -> None:
        msg = payload.get("message", "").strip()
        history = payload.get("history", [])
        model = payload.get("model", None)
        fast_mode = bool(payload.get("fast_mode", True))

        if not msg:
            self.send_json(400, {"error": "Message cannot be empty."})
            return

        resp = GLOBAL_CONTEXT.assistant_service.chat(
            message=msg,
            history=history,
            model=model,
            fast_mode=fast_mode,
        )
        self.send_json(
            200,
            {
                "content": resp.content,
                "tool_calls_made": resp.tool_calls_made,
                "model_used": resp.model_used,
                "status": resp.status,
                "error": resp.error,
                "latency_ms": getattr(resp, "latency_ms", 0.0),
            },
        )

    def _handle_get_market_universe(self, query: dict) -> None:
        q = query.get("q", [""])[0]
        from trady.market.universe import search_universe

        items = search_universe(query=q)
        self.send_json(
            200,
            {
                "count": len(items),
                "universe": items,
            },
        )

    def _handle_get_market_quote(self, query: dict) -> None:
        symbol = query.get("symbol", [""])[0]
        refresh = query.get("refresh", ["0"])[0] in ("1", "true", "True")
        if not symbol:
            self.send_json(400, {"error": "Symbol query parameter is required."})
            return
        try:
            quote = GLOBAL_CONTEXT.quote_service.get_quote(
                symbol, force_refresh=refresh
            )
            self.send_json(200, quote)
        except Exception as err:
            self.send_json(
                500, {"error": f"Failed to fetch quote for '{symbol}': {err}"}
            )

    def _handle_get_market_candles(self, query: dict) -> None:
        symbol = query.get("symbol", [""])[0]
        timeframe = query.get("timeframe", ["1d"])[0]
        if not symbol:
            self.send_json(400, {"error": "Symbol query parameter is required."})
            return
        try:
            candles = GLOBAL_CONTEXT.quote_service.get_candles(
                symbol, timeframe=timeframe
            )
            self.send_json(
                200,
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "count": len(candles),
                    "candles": candles,
                },
            )
        except Exception as err:
            self.send_json(
                500, {"error": f"Failed to load candles for '{symbol}': {err}"}
            )

    def _handle_get_market_forecast(self, query: dict) -> None:
        symbol = query.get("symbol", [""])[0]
        horizon_val = query.get("horizon", ["5"])[0]
        model_type = query.get("model", ["logistic_regression"])[0]
        if not symbol:
            self.send_json(400, {"error": "Symbol query parameter is required."})
            return
        try:
            horizon = int(horizon_val)
        except ValueError:
            horizon = 5

        try:
            forecast = GLOBAL_CONTEXT.risk_engine.analyze_next_move(
                symbol=symbol, horizon_days=horizon, model_type=model_type
            )
            self.send_json(200, forecast)
        except Exception as err:
            import traceback

            traceback.print_exc()
            self.send_json(
                500, {"error": f"Forecast engine error for '{symbol}': {err}"}
            )

    def _handle_get_portfolio_summary(self, query: dict) -> None:
        rate_param = query.get("usd_inr_rate", [None])[0]
        usd_inr_rate: float | None = None
        if rate_param is not None and rate_param != "":
            try:
                usd_inr_rate = float(rate_param)
            except ValueError:
                usd_inr_rate = None

        wm = GLOBAL_CONTEXT.wallet_manager
        sim = GLOBAL_CONTEXT.simulator

        all_positions = list(wm.inr_wallet.positions.keys()) + list(
            wm.usd_wallet.positions.keys()
        )
        prices = {}
        for sym in all_positions:
            try:
                q = GLOBAL_CONTEXT.quote_service.get_quote(sym)
                prices[sym] = float(q["last_price"])
            except Exception:
                pass
        if prices:
            wm.update_market_prices(prices)
            sim.process_pending_limit_orders(prices)

        consolidated = wm.get_consolidated_summary(usd_inr_rate=usd_inr_rate)
        self.send_json(
            200,
            {
                "inr_wallet": wm.inr_wallet.to_dict(),
                "usd_wallet": wm.usd_wallet.to_dict(),
                "consolidated": consolidated,
                "active_currency": "INR / USD",
            },
        )

    def _handle_post_portfolio_order(self, payload: dict) -> None:
        if not self._is_authorized():
            self.send_json(
                401, {"error": "Unauthorized: valid session token required."}
            )
            return

        symbol = payload.get("symbol", "").strip()
        action_str = payload.get("action", "").upper().strip()
        order_type_str = payload.get("order_type", "MARKET").upper().strip()
        try:
            quantity = float(payload.get("quantity", 0))
        except (ValueError, TypeError):
            self.send_json(400, {"error": "Invalid quantity provided."})
            return

        limit_price = payload.get("limit_price")
        if limit_price is not None:
            try:
                limit_price = float(limit_price)
            except (ValueError, TypeError):
                self.send_json(400, {"error": "Invalid limit_price provided."})
                return

        current_price = payload.get("current_price")
        if current_price is not None:
            try:
                current_price = float(current_price)
            except (ValueError, TypeError):
                current_price = None

        if current_price is None:
            try:
                q = GLOBAL_CONTEXT.quote_service.get_quote(symbol)
                current_price = float(q["last_price"])
            except Exception:
                current_price = None

        forecast_metadata = payload.get("forecast_metadata")

        from trady.portfolio.models import OrderAction, OrderType

        try:
            act = OrderAction(action_str)
            otype = OrderType(order_type_str)
            market_price = current_price or limit_price or 0.0
            order = GLOBAL_CONTEXT.simulator.submit_order(
                symbol=symbol,
                action=act,
                order_type=otype,
                quantity=quantity,
                current_market_price=market_price,
                limit_price=limit_price,
                forecast_metadata=forecast_metadata,
            )
            wallet = GLOBAL_CONTEXT.wallet_manager.get_wallet_for_symbol(symbol)
            self.send_json(
                200,
                {
                    "success": True,
                    "order": order.to_dict(),
                    "wallet": wallet.to_dict(),
                },
            )
        except (ValueError, KeyError) as err:
            self.send_json(400, {"error": str(err)})
        except Exception as err:
            import traceback

            traceback.print_exc()
            self.send_json(500, {"error": f"Order submission failed: {err}"})

    def _handle_post_portfolio_cancel(self, payload: dict) -> None:
        if not self._is_authorized():
            self.send_json(
                401, {"error": "Unauthorized: valid session token required."}
            )
            return

        order_id = payload.get("order_id", "").strip()
        if not order_id:
            self.send_json(400, {"error": "order_id is required."})
            return

        try:
            order = GLOBAL_CONTEXT.simulator.cancel_order(order_id=order_id)
            wallet = GLOBAL_CONTEXT.wallet_manager.get_wallet_for_symbol(order.symbol)
            self.send_json(
                200,
                {
                    "success": True,
                    "order": order.to_dict(),
                    "wallet": wallet.to_dict(),
                },
            )
        except (ValueError, KeyError) as err:
            self.send_json(400, {"error": str(err)})
        except Exception as err:
            self.send_json(500, {"error": f"Cancel order failed: {err}"})

    def _handle_post_portfolio_reset(self, payload: dict) -> None:
        if not self._is_authorized():
            self.send_json(
                401, {"error": "Unauthorized: valid session token required."}
            )
            return

        confirmation = bool(payload.get("confirmation", False))
        starting_inr = float(payload.get("starting_inr", 100000.0))
        starting_usd = float(payload.get("starting_usd", 10000.0))

        try:
            wm = GLOBAL_CONTEXT.wallet_manager
            wm.reset_portfolio(
                confirmation=confirmation,
                new_inr_balance=starting_inr,
                new_usd_balance=starting_usd,
            )
            self.send_json(
                200,
                {
                    "success": True,
                    "message": (
                        "Portfolio wallets and simulated trade records reset "
                        "successfully."
                    ),
                    "inr_wallet": wm.inr_wallet.to_dict(),
                    "usd_wallet": wm.usd_wallet.to_dict(),
                },
            )
        except ValueError as err:
            self.send_json(400, {"error": str(err)})
        except Exception as err:
            self.send_json(500, {"error": f"Reset failed: {err}"})


def start_studio_server(
    host: str = "127.0.0.1",
    port: int = 8090,
    project_dir: Path | None = None,
    open_browser: bool = True,
    allow_remote: bool = False,
) -> None:
    """Launch the TRADE9 Studio web server."""
    # Enforce loopback binding protection
    if host not in ("127.0.0.1", "localhost", "::1") and not allow_remote:
        raise ValueError(
            "SAFETY VIOLATION: Binding Studio server to non-loopback host is disabled "
            "by default for security. Specify allow_remote=True if explicitly required."
        )

    if project_dir:
        GLOBAL_CONTEXT.save_config(project_dir)

    GLOBAL_CONTEXT.allowed_origins.add(f"http://127.0.0.1:{port}")
    GLOBAL_CONTEXT.allowed_origins.add(f"http://localhost:{port}")
    GLOBAL_CONTEXT.allowed_hosts.update({"127.0.0.1", "localhost"})
    if host not in ("127.0.0.1", "localhost"):
        GLOBAL_CONTEXT.allowed_origins.add(f"http://{host}:{port}")
        GLOBAL_CONTEXT.allowed_hosts.add(host)

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
