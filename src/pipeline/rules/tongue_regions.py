# src/pipeline/rules/tongue_regions.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def _to01(mask_255: np.ndarray) -> np.ndarray:
    return (mask_255 > 0).astype(np.uint8)


def _to255(mask01: np.ndarray) -> np.ndarray:
    return (mask01 > 0).astype(np.uint8) * 255


def _pca_axes(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0)
    if len(xs) < 80:
        # fallback: assume tongue vertical
        mean = np.array([mask01.shape[1] / 2.0, mask01.shape[0] / 2.0], dtype=np.float32)
        axis1 = np.array([0.0, 1.0], dtype=np.float32)  # length
        axis2 = np.array([1.0, 0.0], dtype=np.float32)  # left-right
        return mean, axis1, axis2

    pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (x,y)
    mean = pts.mean(axis=0)
    X = pts - mean

    cov = (X.T @ X) / max(1, (X.shape[0] - 1))
    eigvals, eigvecs = np.linalg.eigh(cov)  # asc
    axis1 = eigvecs[:, 1]  # major
    axis2 = eigvecs[:, 0]  # minor

    axis1 = axis1 / (np.linalg.norm(axis1) + 1e-6)
    axis2 = axis2 / (np.linalg.norm(axis2) + 1e-6)

    # enforce axis1 points to "tip direction" (usually downward: y+)
    if axis1[1] < 0:
        axis1 = -axis1
        axis2 = -axis2

    return mean, axis1, axis2


def split_tongue_regions(
    tg_mask_255: np.ndarray,
    *,
    ratios: Tuple[float, float, float] = (0.30, 0.40, 0.30),
) -> Dict[str, np.ndarray]:
    """
    Input:
      tg_mask_255 (HxW, 0/255)

    Output (all 0/255, subset of tg):
      tongue_tip / tongue_center / tongue_root
      tongue_left / tongue_right
    """
    assert tg_mask_255.ndim == 2, "tg_mask_255 must be HxW"

    mask01 = _to01(tg_mask_255)
    H, W = mask01.shape[:2]
    if int(mask01.sum()) == 0:
        z = np.zeros((H, W), dtype=np.uint8)
        return {
            "tongue_tip": z.copy(),
            "tongue_center": z.copy(),
            "tongue_root": z.copy(),
            "tongue_left": z.copy(),
            "tongue_right": z.copy(),
        }

    r1, r2, r3 = ratios
    s = r1 + r2 + r3
    r1, r2, r3 = r1 / s, r2 / s, r3 / s

    mean, axis1, axis2 = _pca_axes(mask01)

    ys, xs = np.where(mask01 > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    X = pts - mean[None, :]

    # projection
    t = X @ axis1  # along length
    u = X @ axis2  # left-right signed

    tmin, tmax = float(t.min()), float(t.max())
    trng = max(1e-6, (tmax - tmin))
    tn = (t - tmin) / trng  # [0,1]

    thr1 = r1
    thr2 = r1 + r2

    tip_idx = tn <= thr1
    center_idx = (tn > thr1) & (tn <= thr2)
    root_idx = tn > thr2

    left_idx = u < 0
    right_idx = ~left_idx

    def scatter(sel: np.ndarray) -> np.ndarray:
        m = np.zeros((H, W), dtype=np.uint8)
        m[ys[sel], xs[sel]] = 255
        return m

    out = {
        "tongue_tip": scatter(tip_idx),
        "tongue_center": scatter(center_idx),
        "tongue_root": scatter(root_idx),
        "tongue_left": scatter(left_idx),
        "tongue_right": scatter(right_idx),
    }

    # ensure subset-of-tg
    tg01 = mask01 > 0
    for k in list(out.keys()):
        out[k] = ((out[k] > 0) & tg01).astype(np.uint8) * 255

    return out


def build_tissue_regions(
    tai_mask_255: np.ndarray,
    zhi_mask_255: np.ndarray,
    tongue_regions: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    coating_* = tai ∩ tongue_*
    body_*    = zhi ∩ tongue_*
    """
    out: Dict[str, np.ndarray] = {}
    tai01 = _to01(tai_mask_255) if tai_mask_255 is not None else None
    zhi01 = _to01(zhi_mask_255) if zhi_mask_255 is not None else None

    for k, m in tongue_regions.items():
        reg01 = _to01(m)
        name = k.replace("tongue_", "")  # tip/center/root/left/right

        if tai01 is not None:
            out[f"coating_{name}"] = _to255(tai01 & reg01)
        if zhi01 is not None:
            out[f"body_{name}"] = _to255(zhi01 & reg01)

    return out


def roi_area_stats(
    roi_masks: Dict[str, np.ndarray],
    *,
    ref_key: str = "tongue_mask",
) -> Dict[str, float]:
    """
    输出每个 ROI 的 area & ratio（ratio 相对 ref_key 的 area）
    """
    stats: Dict[str, float] = {}
    ref = roi_masks.get(ref_key)
    ref_area = float((ref > 0).sum()) if isinstance(ref, np.ndarray) else 0.0
    stats[f"{ref_key}_area"] = ref_area

    for k, m in (roi_masks or {}).items():
        if not isinstance(m, np.ndarray):
            continue
        a = float((m > 0).sum())
        stats[f"{k}_area"] = a
        stats[f"{k}_ratio"] = (a / ref_area) if ref_area > 0 else 0.0

    return stats
