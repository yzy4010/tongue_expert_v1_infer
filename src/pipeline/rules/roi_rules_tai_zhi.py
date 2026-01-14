# pipeline/rules/roi_rules_tai_zhi.py
from __future__ import annotations
import cv2
import numpy as np

def _to_u8_mask(mask: np.ndarray) -> np.ndarray:
    """Return uint8 {0,255} mask."""
    if mask is None:
        return None
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # allow {0,1} or {0,255}
    if mask.max() <= 1:
        mask = mask * 255
    mask = (mask > 0).astype(np.uint8) * 255
    return mask

def split_tai_zhi(img_bgr: np.ndarray, tg_mask: np.ndarray, *,
                  hsv_s_max: int = 80,
                  hsv_v_min: int = 160,
                  lab_b_min: int = 145,
                  min_area_ratio: float = 0.01) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Rule-based split within tongue mask:
      tai: coating (bright & low saturation OR yellowish)
      zhi: tg - tai
    Returns:
      tai_mask_u8, zhi_mask_u8, debug_dict
    """
    tg = _to_u8_mask(tg_mask)
    if tg is None or tg.sum() == 0:
        # no tongue -> no tai/zhi
        empty = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        return empty, empty, {"reason": "empty_tg"}

    h, w = tg.shape
    roi = img_bgr.copy()

    # only operate inside tg
    inside = tg > 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # white-coating candidate: low saturation + high value
    cand_white = (S <= hsv_s_max) & (V >= hsv_v_min)
    # yellow-coating candidate: b channel high (more yellow)
    cand_yellow = (B >= lab_b_min) & (L >= 120)  # L constraint avoids dark noise

    cand = (cand_white | cand_yellow) & inside
    tai = cand.astype(np.uint8) * 255

    # morphology cleanup
    k1 = max(3, int(min(h, w) * 0.01) | 1)   # ~1% size, odd
    k2 = max(5, int(min(h, w) * 0.02) | 1)   # ~2% size, odd
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))

    tai = cv2.morphologyEx(tai, cv2.MORPH_OPEN, kernel1, iterations=1)
    tai = cv2.morphologyEx(tai, cv2.MORPH_CLOSE, kernel2, iterations=1)

    # keep within tg strictly
    tai = (tai > 0) & inside
    tai = tai.astype(np.uint8) * 255

    # area sanity: if tai too tiny, still keep but you can force empty
    tg_area = int(inside.sum())
    tai_area = int((tai > 0).sum())
    if tg_area > 0 and (tai_area / tg_area) < min_area_ratio:
        # 可选策略：保留/清空。这里默认“保留”更稳（不丢信息）
        pass

    # zhi = tg - tai
    zhi = (inside & ~(tai > 0)).astype(np.uint8) * 255

    dbg = {
        "tg_area": tg_area,
        "tai_area": tai_area,
        "tai_area_ratio": float(tai_area) / float(tg_area) if tg_area else 0.0,
        "params": {
            "hsv_s_max": hsv_s_max,
            "hsv_v_min": hsv_v_min,
            "lab_b_min": lab_b_min,
            "min_area_ratio": min_area_ratio,
        }
    }
    return tai, zhi, dbg
