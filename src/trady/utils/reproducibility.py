"""Utilities for deterministic reproducibility and system environment auditing."""

import importlib
import os
import platform
import random
import sys
from typing import Any


def set_seed(seed: int = 42, cuda_deterministic: bool = True) -> int:
    """Set seeds across Python stdlib and numerical frameworks for deterministic runs.

    Args:
        seed: Integer seed value.
        cuda_deterministic: Whether to set PyTorch CUDA deterministic flags
            if PyTorch is installed.

    Returns:
        The seed integer applied.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Conditionally seed numpy if available
    if importlib.util.find_spec("numpy") is not None:
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass

    # Conditionally seed PyTorch if available
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                if cuda_deterministic:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    return seed


def get_system_info() -> dict[str, Any]:
    """Gather diagnostic runtime information about OS, CPU, memory, and hardware."""
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count() or 1,
    }

    # Detect Windows memory if on Windows
    if platform.system() == "Windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total_mb = status.ullTotalPhys / (1024 * 1024)
                avail_mb = status.ullAvailPhys / (1024 * 1024)
                info["ram_total_mb"] = round(total_mb, 1)
                info["ram_available_mb"] = round(avail_mb, 1)
        except Exception:
            pass

    # Detect GPU via PyTorch if installed
    info["cuda_available"] = False
    info["gpu_name"] = "None"
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_count"] = torch.cuda.device_count()
        except Exception:
            pass

    return info
