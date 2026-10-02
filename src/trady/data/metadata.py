"""Dataset metadata specification and serialization for TRADY."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata container tracking provenance, temporal bounds, and schema version."""

    provider: str
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    timezone: str = "UTC"
    retrieval_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = "1.0.0"
    record_count: int = 0
    checksum_sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_time > self.end_time:
            raise ValueError(
                f"start_time ({self.start_time}) cannot be after "
                f"end_time ({self.end_time})"
            )
        if self.record_count < 0:
            raise ValueError("record_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a JSON-serializable dictionary."""
        d = asdict(self)
        d["start_time"] = self.start_time.isoformat()
        d["end_time"] = self.end_time.isoformat()
        d["retrieval_time"] = self.retrieval_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMetadata":
        """Instantiate DatasetMetadata from a dictionary."""
        data_copy = dict(data)
        data_copy["start_time"] = datetime.fromisoformat(data_copy["start_time"])
        data_copy["end_time"] = datetime.fromisoformat(data_copy["end_time"])
        if isinstance(data_copy.get("retrieval_time"), str):
            data_copy["retrieval_time"] = datetime.fromisoformat(
                data_copy["retrieval_time"]
            )
        return cls(**data_copy)

    def save(self, file_path: Path | str) -> None:
        """Write metadata to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: Path | str) -> "DatasetMetadata":
        """Load metadata from JSON file."""
        path = Path(file_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
