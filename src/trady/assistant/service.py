"""Service layer coordinating TRADY AI Research Assistant and allowlisted tools."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trady.assistant.client import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from trady.assistant.tools import (
    RESEARCH_TOOLS_DISPATCHER,
    TOOL_DEFINITIONS,
)
from trady.config.settings import AssistantConfig
from trady.utils.logging import get_logger

logger = get_logger("assistant.service")

SYSTEM_PROMPT = """You are TRADY AI, an institutional quantitative research assistant \
integrated into the TRADY research platform.

CORE PRINCIPLES & SAFETY:
1. TRADY is an educational research and paper-trading platform. Real-money trading \
is strictly prohibited by safety firewalls. Never suggest live order execution or \
promise guaranteed financial profits.
2. Ground all quantitative statements, statistics, metrics, and dataset \
descriptions strictly in the facts returned by TRADY's research tools. Never \
fabricate Sharpe ratios, returns, data sizes, or model scores.
3. You have access to 6 approved research tools:
   - inspect_datasets: Inspects available market, feature, and target datasets in \
the workspace.
   - get_feature_metadata: Retrieves registered causal features, descriptions, \
and lookbacks.
   - get_experiment_results: Retrieves trained ML models and chronological \
evaluation metrics.
   - compare_experiments: Compares two trained models side-by-side.
   - get_backtest_metrics: Retrieves backtest simulation files and performance logs.
   - explain_research_report: Gathers aggregated evidence across models and \
backtests.
4. When the user asks about available data, features, experiments, models, or \
backtests, invoke the appropriate tool to retrieve real facts before answering.
5. You do NOT have arbitrary shell execution access, file modification capabilities, \
or trading authority. Maintain mathematical rigor, objectivity, and conciseness.
"""


@dataclass
class AssistantResponse:
    """Structured response from TRADY AI Assistant."""

    content: str
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    model_used: str = ""
    status: str = "ok"
    error: str | None = None


class AssistantService:
    """Coordinates local Ollama inference, tool dispatching, and safety enforcement."""

    def __init__(
        self,
        config: AssistantConfig | None = None,
        project_dir: str | Path | None = None,
    ) -> None:
        self.config = config or AssistantConfig()
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.client = OllamaClient(
            endpoint=self.config.ollama_endpoint,
            default_model=self.config.model_name,
            timeout_seconds=self.config.timeout_seconds,
            temperature=self.config.temperature,
        )

    def get_status(self) -> dict[str, Any]:
        """Check connection to local Ollama runtime and return status."""
        status = self.client.check_status()
        status["project_dir"] = str(self.project_dir)
        status["available_tools"] = list(RESEARCH_TOOLS_DISPATCHER.keys())
        return status

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Safely execute an allowlisted research tool.

        Args:
            tool_name: Name of tool to execute.
            arguments: Tool arguments dictionary.

        Returns:
            Result dictionary from tool.
        """
        if tool_name not in RESEARCH_TOOLS_DISPATCHER:
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' is not in allowlisted research tools.",
            }

        func = RESEARCH_TOOLS_DISPATCHER[tool_name]
        try:
            # Inject project_dir if accepted by tool and not explicitly passed
            kwargs = dict(arguments)
            if (
                "project_dir" in func.__code__.co_varnames
                and "project_dir" not in kwargs
            ):
                kwargs["project_dir"] = self.project_dir

            return func(**kwargs)
        except Exception as e:
            logger.error("Error executing tool %s: %s", tool_name, e)
            return {
                "status": "error",
                "error": f"Exception executing {tool_name}: {e}",
            }

    def chat(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> AssistantResponse:
        """Process user message, execute requested tools, and return grounded response.

        Args:
            message: User's prompt.
            history: Optional list of previous chat messages.
            model: Optional model override.

        Returns:
            AssistantResponse with final content and list of tools executed.
        """
        clean_msg = message.strip()
        if not clean_msg:
            return AssistantResponse(
                content="Please provide a message or question.",
                status="error",
                error="Empty message",
            )

        target_model = model or self.config.model_name
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Append sanitized history
        if history:
            for h in history:
                role = h.get("role", "user")
                if role in ("user", "assistant"):
                    messages.append(
                        {"role": role, "content": str(h.get("content", ""))}
                    )

        messages.append({"role": "user", "content": clean_msg})

        tools_executed: list[dict[str, Any]] = []

        # 1. Initial LLM Call with tools
        try:
            first_resp = self.client.chat(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                model=target_model,
            )
        except (
            OllamaConnectionError,
            OllamaModelNotFoundError,
            OllamaTimeoutError,
            OllamaClientError,
        ) as err:
            logger.warning("Ollama call failed: %s", err)
            return AssistantResponse(
                content=f"[TRADY AI Unavailable]: {err}",
                model_used=target_model,
                status="error",
                error=str(err),
            )
        except Exception as err:
            logger.error("Unexpected error in AssistantService: %s", err)
            return AssistantResponse(
                content=f"[TRADY AI Error]: {err}",
                model_used=target_model,
                status="error",
                error=str(err),
            )

        resp_msg = first_resp.get("message", {})
        tool_calls = resp_msg.get("tool_calls", [])

        # 2. Handle Tool Calling if requested
        if tool_calls:
            # Append assistant message with tool calls
            messages.append(resp_msg)

            for tc in tool_calls:
                func_data = tc.get("function", {})
                tool_name = func_data.get("name", "")
                tool_args = func_data.get("arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception:
                        tool_args = {}

                result = self.execute_tool(tool_name, tool_args)
                tools_executed.append(
                    {
                        "tool": tool_name,
                        "arguments": tool_args,
                        "status": result.get("status", "completed"),
                    }
                )

                # Append tool result to conversation
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result),
                    }
                )

            # 3. Follow-up call for final synthesis
            try:
                final_resp = self.client.chat(
                    messages=messages,
                    tools=None,
                    model=target_model,
                )
                final_content = final_resp.get("message", {}).get("content", "")
                return AssistantResponse(
                    content=final_content,
                    tool_calls_made=tools_executed,
                    model_used=target_model,
                    status="ok",
                )
            except Exception as follow_err:
                logger.warning("Follow-up synthesis failed: %s", follow_err)
                tools_str = ", ".join(t["tool"] for t in tools_executed)
                return AssistantResponse(
                    content=(
                        f"Tools were executed successfully ({tools_str}), "
                        f"but synthesis failed: {follow_err}"
                    ),
                    tool_calls_made=tools_executed,
                    model_used=target_model,
                    status="partial",
                    error=str(follow_err),
                )

        # No tools needed; return initial response
        content = resp_msg.get("content", "")
        return AssistantResponse(
            content=content,
            tool_calls_made=[],
            model_used=target_model,
            status="ok",
        )
