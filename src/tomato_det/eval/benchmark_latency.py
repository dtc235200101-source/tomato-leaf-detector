"""Latency measurement utilities; no illustrative latency values are generated."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from tomato_det.common.meta import runtime_metadata
from tomato_det.common.paths import project_root
from tomato_det.models.base import Detector


@dataclass(frozen=True, slots=True)
class LatencySummary:
    samples: int
    p50_ms: float
    p95_ms: float

    def as_dict(self) -> dict[str, float | int]:
        return {"samples": self.samples, "p50_ms": self.p50_ms, "p95_ms": self.p95_ms}


def summarize_latency_ms(samples: list[float]) -> LatencySummary:
    if not samples or any(value < 0 for value in samples):
        raise ValueError("Latency samples must be non-empty and non-negative")
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return LatencySummary(len(samples), median(ordered), ordered[p95_index])


def benchmark_detector(
    detector: Detector,
    image_rgb: Any,
    *,
    warmup_runs: int = 5,
    measured_runs: int = 30,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    max_det: int = 50,
) -> LatencySummary:
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup_runs must be non-negative and measured_runs must be positive")
    arguments = {"conf": conf, "iou": iou, "imgsz": imgsz, "max_det": max_det}
    for _ in range(warmup_runs):
        detector.predict(image_rgb, **arguments)
    timings: list[float] = []
    for _ in range(measured_runs):
        start = perf_counter()
        detector.predict(image_rgb, **arguments)
        timings.append((perf_counter() - start) * 1000)
    return summarize_latency_ms(timings)


def write_benchmark(
    summary: LatencySummary,
    output_path: Path,
    *,
    model_id: str,
    device: str,
    warmup_runs: int,
    measured_runs: int,
    input_description: str,
    image_size: int = 640,
) -> Path:
    """Persist an actual inference-only benchmark with enough context to compare it."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark artifact: {output_path}")
    payload = {
        **runtime_metadata(project_root()),
        "kind": "inference_only_latency",
        "model_id": model_id,
        "device": device,
        "image_size": image_size,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "input": input_description,
        "latency": summary.as_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
