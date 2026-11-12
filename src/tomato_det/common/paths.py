"""Repository-relative paths, with no machine-specific location embedded in source."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_path(filename: str) -> Path:
    return project_root() / "configs" / filename
