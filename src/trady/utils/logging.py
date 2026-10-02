"""Structured logging utility for TRADY.

Provides standardized text and JSON formatters with ISO 8601 timestamps.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from trady.config.settings import LoggingConfig


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


class StandardTextFormatter(logging.Formatter):
    """Clean text formatter with ISO 8601 UTC timestamps."""

    def formatTime(  # noqa: N802
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        dt = datetime.fromtimestamp(record.created, tz=UTC)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Initialize structured logging for TRADY."""
    if config is None:
        config = LoggingConfig()

    root_logger = logging.getLogger("trady")
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    if config.format == "json":
        console_handler.setFormatter(StructuredJsonFormatter())
    else:
        fmt = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        if config.show_caller:
            fmt = (
                "[%(asctime)s] [%(levelname)s] [%(name)s] "
                "[%(filename)s:%(lineno)d]: %(message)s"
            )
        console_handler.setFormatter(StandardTextFormatter(fmt=fmt))

    root_logger.addHandler(console_handler)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Retrieve a TRADY-namespaced logger."""
    if not name.startswith("trady"):
        name = f"trady.{name}"
    return logging.getLogger(name)
