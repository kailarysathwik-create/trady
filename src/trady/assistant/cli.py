"""CLI commands for TRADY AI Research Assistant."""

import argparse
import sys

from trady.assistant.service import AssistantService
from trady.config.loader import load_config
from trady.utils.logging import get_logger

logger = get_logger("cli.assistant")


def add_assistant_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register assistant subcommands with main TRADY CLI parser."""
    asst_parser = subparsers.add_parser(
        "assistant",
        help="Local AI quantitative research assistant (Ollama/Qwen3)",
    )
    asst_sub = asst_parser.add_subparsers(
        dest="assistant_command",
        required=True,
        help="Assistant operations",
    )

    # 1. trade9 assistant status
    asst_sub.add_parser(
        "status",
        help="Check local Ollama daemon connection and installed models",
    )

    # 2. trade9 assistant ask "<question>"
    ask_p = asst_sub.add_parser(
        "ask",
        help="Ask TRADY AI a single quantitative question",
    )
    ask_p.add_argument(
        "prompt",
        type=str,
        help="Research question or query",
    )
    ask_p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override (e.g. qwen3:4b, qwen2.5-coder:7b)",
    )

    # 3. trade9 assistant chat
    chat_p = asst_sub.add_parser(
        "chat",
        help="Start interactive conversational research session",
    )
    chat_p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override (e.g. qwen3:4b)",
    )


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))


def handle_assistant_command(args: argparse.Namespace) -> int:
    """Execute assistant CLI commands."""
    cmd = getattr(args, "assistant_command", None)
    settings = load_config()
    service = AssistantService(config=settings.assistant)

    if cmd == "status":
        status = service.get_status()
        print("=" * 65)
        print(" TRADY LOCAL AI ASSISTANT DIAGNOSTICS")
        print("=" * 65)
        online_str = "ONLINE" if status["is_running"] else "OFFLINE"
        installed_str = "YES" if status["target_model_installed"] else "NO"
        models_str = ", ".join(status["installed_models"]) or "None"
        tools_str = ", ".join(status["available_tools"])
        print(f" Ollama Daemon Status:   {online_str}")
        print(f" Endpoint:               {status['endpoint']}")
        print(f" Target Model:           {status['target_model']}")
        print(f" Target Model Installed: {installed_str}")
        print(f" Available Models:       {models_str}")
        print(f" Allowlisted Tools (6):  {tools_str}")
        if status.get("error"):
            print(f" Error details:          {status['error']}")
        print("=" * 65)
        return 0 if status["is_running"] else 1

    elif cmd == "ask":
        prompt = args.prompt
        print(f"\n[You]: {prompt}")
        print("[TRADY AI]: Thinking and inspecting workspace...\n")
        resp = service.chat(prompt, model=args.model)
        if resp.tool_calls_made:
            print("Tools Invoked:")
            for tc in resp.tool_calls_made:
                print(f"  - {tc['tool']} ({tc['status']})")
            print()
        _safe_print(resp.content)
        print()
        return 0 if resp.status == "ok" else 1

    elif cmd == "chat":
        print("=" * 65)
        print(" Welcome to TRADY Interactive AI Research Session")
        print(" Type 'exit' or 'quit' to end session.")
        print("=" * 65)
        history: list[dict] = []
        while True:
            try:
                user_input = input("\nTRADY-AI > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting AI session.")
                    break

                resp = service.chat(user_input, history=history, model=args.model)
                if resp.tool_calls_made:
                    used = ", ".join(t["tool"] for t in resp.tool_calls_made)
                    print(f"  [Used tools: {used}]")
                print(f"\n{resp.content}")

                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": resp.content})
            except (KeyboardInterrupt, EOFError):
                print("\nSession terminated.")
                break
        return 0

    return 0
