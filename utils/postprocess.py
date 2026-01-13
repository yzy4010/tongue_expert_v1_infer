# src/utils/postprocess.py
from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np


def mask_to_roi(img_bgr: np.ndarray, mask_255: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    mask_255: 0/255 uint8
    Return: roi_bgr, bbox(x1,y1,x2,y2)
    """
    ys, xs = np.where(mask_255 > 0)
    h, w = mask_255.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        # fallback: full image
        return img_bgr.copy(), (0, 0, w, h)

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    # optional padding
    pad = int(0.03 * max(w, h))
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

    roi = img_bgr[y1:y2, x1:x2].copy()
    return roi, (x1, y1, x2, y2)
