"""Lazy, allowlisted local detector loading with serialized model access."""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tomato_det.models.base import Detector
from tomato_det.models.registry import (
    ModelSpec,
    ModelUnavailableError,
    build_detector,
    load_model_specs,
    model_availability,
)


class ModelNotFoundError(ValueError):
    """The requested model ID is not in the application allowlist."""


@dataclass(frozen=True, slots=True)
class ModelStatus:
    model_id: str
    display_name: str
    available: bool
    reason: str | None


class ModelManager:
    def __init__(
        self,
        *,
        max_loaded_models: int = 1,
        specs: dict[str, ModelSpec] | None = None,
        detector_factory: Callable[[str], Detector] | None = None,
    ) -> None:
        self._specs = specs or load_model_specs()
        self._max_loaded_models = max_loaded_models
        self._detector_factory = detector_factory or build_detector
        self._loaded: dict[str, Detector] = {}
        self._load_lock = threading.Lock()
        self._predict_locks: dict[str, threading.Lock] = {}

    @property
    def loaded_model_ids(self) -> list[str]:
        return sorted(self._loaded)

    def is_allowlisted(self, model_id: str) -> bool:
        return model_id in self._specs

    def list_models(self) -> list[ModelStatus]:
        output: list[ModelStatus] = []
        for spec in self._specs.values():
            available, reason = model_availability(spec)
            if available and importlib.util.find_spec("ultralytics") is None:
                available, reason = False, "Ultralytics is not installed"
            output.append(ModelStatus(spec.model_id, spec.display_name, available, reason))
        return output

    def _load(self, model_id: str) -> Detector:
        if model_id not in self._specs:
            raise ModelNotFoundError(f"Model is not allowlisted: {model_id}")
        with self._load_lock:
            existing = self._loaded.get(model_id)
            if existing is not None:
                return existing
            if len(self._loaded) >= self._max_loaded_models:
                raise ModelUnavailableError("Maximum number of loaded models has been reached")
            detector = self._detector_factory(model_id)
            self._loaded[model_id] = detector
            self._predict_locks[model_id] = threading.Lock()
            return detector

    def predict(self, model_id: str, image_rgb: Any, **options: Any) -> list[Any]:
        detector = self._load(model_id)
        with self._predict_locks[model_id]:
            return detector.predict(image_rgb, **options)
