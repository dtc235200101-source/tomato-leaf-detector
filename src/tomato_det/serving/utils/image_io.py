"""Safe in-memory image decoding for request payloads."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageValidationError(ValueError):
    """Expected invalid-image error, safe to expose through the API schema."""


def decode_image_rgb(
    payload: bytes, *, max_bytes: int, max_pixels: int
) -> tuple[np.ndarray, int, int]:
    if not payload:
        raise ImageValidationError("Image payload is empty")
    if len(payload) > max_bytes:
        raise ImageValidationError("Image payload exceeds the configured byte limit")
    try:
        with Image.open(BytesIO(payload)) as opened:
            opened.verify()
        with Image.open(BytesIO(payload)) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            if width * height > max_pixels:
                raise ImageValidationError("Image exceeds the configured pixel limit")
            return np.asarray(image), width, height
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageValidationError("Payload is not a decodable image") from error
