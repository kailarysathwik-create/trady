"""Unit tests for structured logging functionality."""

import io
import json
import logging

from trady.config.settings import LoggingConfig
from trady.utils.logging import get_logger, setup_logging


def test_logger_namespacing() -> None:
    """Ensure get_logger prepends 'trady.' if omitted."""
    log1 = get_logger("features")
    assert log1.name == "trady.features"

    log2 = get_logger("trady.data")
    assert log2.name == "trady.data"


def test_structured_json_logging(monkeypatch: object) -> None:
    """Ensure JSON logging formatter produces valid JSON records."""
    cfg = LoggingConfig(level="INFO", format="json")
    setup_logging(cfg)

    logger = get_logger("test.json")

    # Capture stdout
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    from trady.utils.logging import StructuredJsonFormatter

    handler.setFormatter(StructuredJsonFormatter())

    logger.addHandler(handler)
    try:
        logger.info("JSON audit message")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "JSON audit message"
        assert "timestamp" in data
        assert data["logger"] == "trady.test.json"
    finally:
        logger.removeHandler(handler)


def test_standard_text_logging() -> None:
    """Ensure standard text logging formats without errors."""
    cfg = LoggingConfig(level="DEBUG", format="text")
    setup_logging(cfg)
    logger = get_logger("test.text")

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    from trady.utils.logging import StandardTextFormatter

    handler.setFormatter(
        StandardTextFormatter(fmt="[%(asctime)s] [%(levelname)s]: %(message)s")
    )

    logger.addHandler(handler)
    try:
        logger.debug("Debug pulse test")
        output = stream.getvalue()
        assert "DEBUG" in output
        assert "Debug pulse test" in output
    finally:
        logger.removeHandler(handler)
