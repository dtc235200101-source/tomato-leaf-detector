"""Run metadata needed to reproduce a real training or validation result."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        # Git may be absent, or the project may be shipped as a plain directory.
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_metadata(root: Path) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(root),
        "packages": {
            name: _package_version(name)
            for name in ("torch", "torchvision", "ultralytics", "numpy", "pillow")
        },
    }
