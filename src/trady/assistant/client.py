"""Ollama local HTTP client for TRADY AI Research Assistant."""

import json
import urllib.error
import urllib.request
from typing import Any

from trady.utils.logging import get_logger

logger = get_logger("assistant.client")


class OllamaClientError(Exception):
    """Base exception for Ollama client interactions."""

    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when the local Ollama daemon cannot be reached."""

    pass


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when the requested model is not downloaded in Ollama."""

    pass


class OllamaTimeoutError(OllamaClientError):
    """Raised when the local Ollama inference times out."""

    pass


class OllamaClient:
    """Client for local Ollama HTTP API with zero cloud credentials."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:11434",
        default_model: str = "qwen3:4b",
        timeout_seconds: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.default_model = default_model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def check_status(self) -> dict[str, Any]:
        """Check if local Ollama daemon is running and retrieve installed models."""
        url = f"{self.endpoint}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "TRADY-Client/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                is_target_ready = any(
                    self.default_model in m or m.startswith(self.default_model)
                    for m in models
                )
                return {
                    "is_running": True,
                    "endpoint": self.endpoint,
                    "target_model": self.default_model,
                    "target_model_installed": is_target_ready,
                    "installed_models": models,
                    "error": None,
                }
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return {
                "is_running": False,
                "endpoint": self.endpoint,
                "target_model": self.default_model,
                "target_model_installed": False,
                "installed_models": [],
                "error": f"Ollama daemon unreachable at {self.endpoint}: {e}",
            }
        except Exception as e:
            return {
                "is_running": False,
                "endpoint": self.endpoint,
                "target_model": self.default_model,
                "target_model_installed": False,
                "installed_models": [],
                "error": f"Unexpected error checking Ollama: {e}",
            }

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Send chat messages and optional tool specifications to local Ollama.

        Args:
            messages: Conversation messages list [{"role": ..., "content": ...}].
            tools: Optional allowlisted JSON schema tool definitions.
            model: Optional model override.

        Returns:
            Ollama chat response dictionary.
        """
        target_model = model or self.default_model
        url = f"{self.endpoint}/api/chat"
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 512,
                "top_k": 40,
                "top_p": 0.9,
            },
        }
        if tools:
            payload["tools"] = tools

        body_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "TRADY-Client/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp_bytes = resp.read()
                try:
                    data = json.loads(resp_bytes.decode("utf-8"))
                    return data
                except Exception as json_err:
                    raise OllamaClientError(
                        f"Malformed response from Ollama: {json_err}"
                    ) from json_err

        except urllib.error.HTTPError as http_err:
            raw_err = ""
            try:
                raw_err = http_err.read().decode("utf-8")
            except Exception:
                pass

            if http_err.code == 404 or "model" in raw_err.lower():
                raise OllamaModelNotFoundError(
                    f"Model '{target_model}' not found in Ollama. "
                    f"Please run 'ollama pull {target_model}' on your system."
                ) from http_err
            raise OllamaClientError(
                f"Ollama HTTP error {http_err.code}: {raw_err or http_err.reason}"
            ) from http_err

        except TimeoutError as to_err:
            raise OllamaTimeoutError(
                f"Ollama request timed out after {self.timeout_seconds} seconds."
            ) from to_err

        except (urllib.error.URLError, ConnectionRefusedError, OSError) as url_err:
            raise OllamaConnectionError(
                f"Could not connect to local Ollama at {self.endpoint}. "
                "Ensure Ollama is started (e.g. 'ollama serve')."
            ) from url_err
