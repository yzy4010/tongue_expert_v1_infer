# src/pipeline/roi_exporter_v2.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import cv2


def _mask_area(mask_255: np.ndarray) -> int:
    return int((mask_255 > 0).sum())


def _bbox_from_mask(mask_255: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_255 > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def _crop_with_margin(img: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.10):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return img[y1:y2, x1:x2].copy()


def export_v2_roi_images(
    *,
    img_bgr: np.ndarray,
    roi_masks_v2: Dict[str, np.ndarray],
    sample_id: str,
    roi_root: Path,
    min_area: int = 200,
    margin: float = 0.10,
    save_masked: bool = True,
) -> Dict[str, str]:
    """
    将每个 v2 ROI 导出为 roi_root/<roi_key>/<id>.jpg
    - save_masked=True：mask 外置黑（推荐，更贴近 ROI 含义）
    返回：roi_key -> 保存的 jpg 路径（str）
    """
    roi_root = Path(roi_root)
    roi_root.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, str] = {}

    for roi_key, mask_255 in (roi_masks_v2 or {}).items():
        if not isinstance(mask_255, np.ndarray):
            continue

        if _mask_area(mask_255) < min_area:
            continue

        bbox = _bbox_from_mask(mask_255)
        if bbox is None:
            continue

        # crop img + crop mask
        crop_img = _crop_with_margin(img_bgr, bbox, margin=margin)
        crop_mask = _crop_with_margin(mask_255, bbox, margin=margin)

        if save_masked:
            m = (crop_mask > 0)
            masked = np.zeros_like(crop_img)
            masked[m] = crop_img[m]
            out_img = masked
        else:
            out_img = crop_img

        out_dir = roi_root / roi_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sample_id}.jpg"

        cv2.imencode(".jpg", out_img)[1].tofile(str(out_path))
        saved[roi_key] = str(out_path)

    return saved
