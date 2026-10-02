"""Unit tests for deterministic reproducibility and system auditing."""

import os
import random

from trady.utils.reproducibility import get_system_info, set_seed


def test_set_seed_deterministic_python_random() -> None:
    """Verify that set_seed ensures reproducible sequences in Python random."""
    set_seed(12345)
    seq1 = [random.random() for _ in range(5)]

    set_seed(12345)
    seq2 = [random.random() for _ in range(5)]

    assert seq1 == seq2
    assert os.environ.get("PYTHONHASHSEED") == "12345"


def test_get_system_info_contains_required_fields() -> None:
    """Verify system diagnostics return expected hardware/environment metadata."""
    info = get_system_info()
    assert "os" in info
    assert "python_version" in info
    assert "cpu_count" in info
    assert info["cpu_count"] >= 1
    assert "cuda_available" in info
