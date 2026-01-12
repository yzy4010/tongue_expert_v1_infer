# src/utils/image_io.py
from __future__ import annotations
from typing import Any
import cv2
import numpy as np


def read_image_any(image: Any) -> np.ndarray:
    """
    Supports:
      - str path
      - bytes
      - np.ndarray (BGR or RGB; we keep as BGR if looks like BGR)
    Returns: uint8 BGR image
    """
    if isinstance(image, str):
        img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image from path: {image}")
        return img

    if isinstance(image, (bytes, bytearray)):
        data = np.frombuffer(image, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        return img

    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        # assume already BGR
        return image

    raise TypeError(f"Unsupported image type: {type(image)}")
