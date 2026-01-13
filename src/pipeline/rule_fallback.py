# src/pipeline/rule_fallback.py
from __future__ import annotations
from typing import Any, Dict
import cv2
import numpy as np


def infer_color_fallback(roi_bgr: np.ndarray) -> Dict[str, Any]:
    """
    最小可用：均值/标准差 + HSV 均值
    """
    roi = roi_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(roi)
    hsv = cv2.cvtColor((roi * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    out = {
        "rule": "mean_std_rgb_hsv",
        "rgb_mean": [float(r.mean()), float(g.mean()), float(b.mean())],
        "rgb_std": [float(r.std()), float(g.std()), float(b.std())],
        "hsv_mean": [float(hsv[..., 0].mean()), float(hsv[..., 1].mean()), float(hsv[..., 2].mean())],
    }
    return out


def infer_texture_fallback(roi_bgr: np.ndarray) -> Dict[str, Any]:
    """
    最小可用：灰度统计 + Laplacian 方差（清晰度/纹理粗糙度的一个 proxy）
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    out = {
        "rule": "gray_stats_laplacian_var",
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "laplacian_var": float(lap.var()),
    }
    return out
