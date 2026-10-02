"""Target registry and catalog for TRADY quantitative experiments."""

from collections.abc import Callable
from typing import Any

from trady.targets.metadata import TargetMetadata


class TargetRegistry:
    """Central registry for forward-looking target/label generators."""

    def __init__(self) -> None:
        self._targets: dict[str, Callable[..., Any]] = {}
        self._metadata: dict[str, TargetMetadata] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        metadata: TargetMetadata,
    ) -> None:
        """Register a target calculation function and its metadata."""
        if not name or not name.strip():
            raise ValueError("Target name cannot be empty")
        if name in self._targets:
            raise ValueError(f"Target '{name}' is already registered")

        self._targets[name] = func
        self._metadata[name] = metadata

    def get(self, name: str) -> Callable[..., Any]:
        """Retrieve the calculation function for a target."""
        if name not in self._targets:
            avail = list(self._targets.keys())
            raise KeyError(f"Target '{name}' not found. Available targets: {avail}")
        return self._targets[name]

    def get_metadata(self, name: str) -> TargetMetadata:
        """Retrieve metadata for a target."""
        if name not in self._metadata:
            avail = list(self._metadata.keys())
            raise KeyError(f"Target '{name}' metadata not found. Available: {avail}")
        return self._metadata[name]

    def list_targets(self) -> list[TargetMetadata]:
        """List all registered target metadata sorted by name."""
        return sorted(self._metadata.values(), key=lambda m: m.name)

    def contains(self, name: str) -> bool:
        """Check if target name exists in the registry."""
        return name in self._targets

    def clear(self) -> None:
        """Clear registry entries."""
        self._targets.clear()
        self._metadata.clear()


# Default singleton instance
target_registry = TargetRegistry()
