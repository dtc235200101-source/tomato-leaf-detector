"""Environment-backed serving settings; defaults keep the API local-only."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TDET_", case_sensitive=False)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    device: str = "auto"
    default_model: str = "yolo11s_coco"
    max_frame_bytes: int = Field(default=4_000_000, ge=1)
    max_upload_bytes: int = Field(default=10_000_000, ge=1)
    max_image_pixels: int = Field(default=16_000_000, ge=1)
    max_loaded_models: int = Field(default=1, ge=1)
    max_detections: int = Field(default=100, ge=1, le=1000)
    log_level: str = "INFO"
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")

    @field_validator("host")
    @classmethod
    def require_loopback_host(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("TDET_HOST must be a loopback address")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
