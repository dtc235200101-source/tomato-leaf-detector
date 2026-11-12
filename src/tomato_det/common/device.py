"""Runtime device detection without importing PyTorch at module import time."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    requested: str
    resolved: str | None
    torch_available: bool
    cuda_available: bool
    torch_version: str | None
    reason: str | None


def detect_device(requested: str = "auto") -> DeviceInfo:
    """Resolve CPU/CUDA only when PyTorch is installed and can initialize safely."""
    if importlib.util.find_spec("torch") is None:
        return DeviceInfo(requested, None, False, False, None, "PyTorch is not installed")
    import torch

    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        resolved = "cuda:0" if cuda_available else "cpu"
    elif requested.startswith("cuda") and not cuda_available:
        return DeviceInfo(
            requested,
            None,
            True,
            False,
            torch.__version__,
            "CUDA was requested but torch.cuda.is_available() is false",
        )
    elif requested in {"cpu", "mps"} or requested.startswith("cuda"):
        resolved = requested
    else:
        return DeviceInfo(
            requested, None, True, cuda_available, torch.__version__, "Unknown device"
        )
    return DeviceInfo(requested, resolved, True, cuda_available, torch.__version__, None)
