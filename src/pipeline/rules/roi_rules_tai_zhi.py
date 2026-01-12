# -*- coding: utf-8 -*-
"""
Rule-based Tai/Zhi masks from tongue mask.

Inputs:
  tongue_mask: uint8 HxW (0/1)

Outputs:
  tai_mask: uint8 HxW (0/1)
  zhi_mask: uint8 HxW (0/1)

Principle:
  Use distance transform to split tongue region into inner-core (zhi) and boundary-ring (tai).
"""

from __future__ import annotations
import numpy as np
import cv2


def infer_tai_zhi_from_tongue_mask(
    tongue_mask: np.ndarray,
    tai_ratio: float = 0.35,
    min_area: int = 200,
    smooth_kernel: int = 3,
):
    """
    Args:
        tongue_mask: HxW uint8 {0,1} or {0,255}
        tai_ratio: proportion of tongue pixels to assign to tai (boundary ring).
                  0.25~0.45 usually works; default 0.35.
        min_area: minimal area to consider valid; otherwise fallback.
        smooth_kernel: morphological smoothing (odd number). 0 disables.

    Returns:
        tai_mask, zhi_mask: HxW uint8 {0,1}
    """
    if tongue_mask is None:
        raise ValueError("tongue_mask is None")

    tm = (tongue_mask > 0).astype(np.uint8)
    area = int(tm.sum())
    if area < min_area:
        # too small: return empty
        h, w = tm.shape[:2]
        return np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)

    # optional smoothing to remove small holes/noise
    if smooth_kernel and smooth_kernel >= 3 and smooth_kernel % 2 == 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel, smooth_kernel))
        tm = cv2.morphologyEx(tm, cv2.MORPH_CLOSE, k, iterations=1)
        tm = cv2.morphologyEx(tm, cv2.MORPH_OPEN,  k, iterations=1)

    # distance transform inside tongue: higher = deeper inside
    dist = cv2.distanceTransform(tm, distanceType=cv2.DIST_L2, maskSize=5)

    # We want tai as boundary ring: select lowest distance pixels within tongue.
    # Choose threshold by quantile to match tai_ratio.
    dist_in = dist[tm > 0]
    if dist_in.size == 0:
        h, w = tm.shape[:2]
        return np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)

    # quantile for boundary ring
    q = np.clip(tai_ratio, 0.05, 0.95)
    thr = float(np.quantile(dist_in, q))

    tai = ((dist > 0) & (dist <= thr)).astype(np.uint8)
    zhi = ((dist > thr)).astype(np.uint8)

    # ensure both non-empty (fallbacks)
    tai_area = int(tai.sum())
    zhi_area = int(zhi.sum())

    # if tai or zhi becomes empty due to degenerate dist, adjust threshold
    if tai_area < min_area or zhi_area < min_area:
        # fallback: use median split
        thr2 = float(np.quantile(dist_in, 0.50))
        tai = ((dist > 0) & (dist <= thr2)).astype(np.uint8)
        zhi = ((dist > thr2)).astype(np.uint8)
        tai_area = int(tai.sum())
        zhi_area = int(zhi.sum())

    # final fallback: if still empty, force split by erosion
    if tai_area < min_area or zhi_area < min_area:
        # erode tongue to get core = zhi; ring = tai
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        core = cv2.erode(tm, k, iterations=1)
        zhi = (core > 0).astype(np.uint8)
        tai = ((tm > 0) & (zhi == 0)).astype(np.uint8)

    return tai.astype(np.uint8), zhi.astype(np.uint8)
