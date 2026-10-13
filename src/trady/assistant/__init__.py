"""TRADY Phase 2: Local AI Research Assistant package."""

from trady.assistant.client import OllamaClient
from trady.assistant.service import AssistantResponse, AssistantService

__all__ = ["OllamaClient", "AssistantService", "AssistantResponse"]
