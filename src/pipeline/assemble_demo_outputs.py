# -*- coding: utf-8 -*-
"""
Assemble demo-style outputs from paper-style tables.

Input:
  tables: Dict[str, Dict]  # e.g. P21_Tai_Color, P23_Tai_Texture, P24_Tai_CNN, ...

Output:
  demo_outputs: Dict       # representation / color_features / texture_features / roi_masks / roi_stats / ...
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2


# ============================================================
# Utils
# ============================================================

def _roi_keymap(roi: str) -> Tuple[str, str, str, str]:
    """
    roi: Tai/Zhi/Fissure/Toothmark
    returns: (color_table_key, texture_table_key, cnn_table_key, roi_mask_key)
    """
    if roi == "Tai":
        return "P21_Tai_Color", "P23_Tai_Texture", "P24_Tai_CNN", "tai"
    if roi == "Zhi":
        return "P31_Zhi_Color", "P33_Zhi_Texture", "P34_Zhi_CNN", "zhi"
    if roi == "Fissure":
        return "P41_Fissure_Color", "P43_Fissure_Texture", "P44_Fissure_CNN", "fissure"
    if roi == "Toothmark":
        return "P51_Toothmark_Color", "P53_Toothmark_Texture", "P54_Toothmark_CNN", "tooth_mk"
    raise ValueError(f"Unknown roi={roi}")


def _to_u8_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Normalize mask to uint8 {0,255}. Accept {0,1}, bool, uint8.
    """
    if mask is None:
        return None
    m = mask
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255
    m = (m > 0).astype(np.uint8) * 255
    return m


def _mask_area(mask_u8: Optional[np.ndarray]) -> int:
    if mask_u8 is None:
        return 0
    return int((mask_u8 > 0).sum())


def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_mask_png(mask_u8: np.ndarray, path: Path) -> str:
    """
    Save single-channel u8 mask to png. Returns path string (posix).
    """
    _ensure_dir(path.parent)
    cv2.imwrite(str(path), mask_u8)
    return str(path.as_posix())


def _pick_embedding_preview(cnn_tbl: Dict[str, Any], max_n: int = 10) -> Dict[str, Any]:
    """
    Prefer cnnPC1..10 if present, else take first max_n items.
    """
    if not isinstance(cnn_tbl, dict) or len(cnn_tbl) == 0:
        return {}

    pc_keys = [k for k in cnn_tbl.keys() if "cnnPC" in k]
    if pc_keys:
        def _pc_idx(k: str) -> int:
            digits = "".join([c for c in k if c.isdigit()])
            return int(digits) if digits else 0
        pc_keys = sorted(pc_keys, key=_pc_idx)
        return {k: cnn_tbl.get(k) for k in pc_keys[:max_n]}

    # fallback
    items = list(cnn_tbl.items())[:max_n]
    return {k: v for k, v in items}


# ============================================================
# Fallback simple features (keep your original)
# ============================================================

def _compute_simple_color(img_bgr: np.ndarray, mask01: np.ndarray) -> dict:
    """
    简化版颜色统计（足够 demo，工程稳定）
    输出字段尽量对齐你 demo key_features
    """
    m = (mask01 > 0)
    if m.sum() < 10:
        return {}

    b = img_bgr[..., 0][m].astype(np.float32)
    g = img_bgr[..., 1][m].astype(np.float32)
    r = img_bgr[..., 2][m].astype(np.float32)

    # gray + lab L
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)[m]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0][m]

    eps = 1e-6
    r_mean = float(r.mean()); g_mean = float(g.mean()); b_mean = float(b.mean())
    return {
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "gray_mean": float(gray.mean()),
        "lightness_mean": float(L.mean()),
        "rg_diff": float(r_mean - g_mean),
        "rb_diff": float(r_mean - b_mean),
        "r_over_g": float(r_mean / (g_mean + eps)),
        "r_over_b": float(r_mean / (b_mean + eps)),
    }


def _compute_simple_texture(img_bgr: np.ndarray, mask01: np.ndarray) -> dict:
    """
    简化版纹理统计（demo/工程稳定）
    这里给出一个“可解释且稳定”的替代：灰度梯度强度统计 + 熵
    （完整论文版仍以你现有 P13 模块为准）
    """
    m = (mask01 > 0)
    if m.sum() < 10:
        return {}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)[m]

    # entropy (hist-based)
    vals = gray[m].astype(np.uint8)
    hist = np.bincount(vals, minlength=256).astype(np.float32)
    p = hist / (hist.sum() + 1e-6)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    return {
        "grad_mean": float(mag.mean()),
        "grad_std": float(mag.std()),
        "entropy": ent,
    }


# ============================================================
# NEW: assemble with masks + roi stats + tai/zhi fields
# ============================================================

def assemble_demo_outputs_from_tables(
    tables: dict,
    preferred_roi: str = "Tai",
    img_bgr=None,
    roi_masks=None,
    *,
    # NEW: where to save masks (optional). If None -> do not save, only return in-memory paths as empty.
    artifacts_dir: Optional[str] = None,
    sample_id: Optional[str] = None,
    # whether to include paper tables in outputs (optional)
    include_paper_tables: bool = False,
) -> dict:
    """
    demo 输出优先来自 tables（论文级）
    若 P21/P23 为空，则用 (img_bgr, roi_masks) 做简化补算，保证 demo 不空

    新增能力：
      - 保存 tg/tai/zhi/fissure/tooth_mk mask png（可选）
      - 输出 tai_area_ratio / zhi_area_ratio（以及 tg/zhi/tai 面积）
      - 将 P11/P13/P14 的 tai/zhi 字段保留在 demo_outputs["paper_tables"]（可选）

    参数说明：
      artifacts_dir: 例如 "outputs" 或 "outputs/artifacts"
      sample_id: 用于组织路径 artifacts_dir/<sample_id>/masks/*.png（可选）
    """
    outputs: Dict[str, Any] = {}

    # ---------------------------
    # 0) Normalize masks + stats
    # ---------------------------
    norm_masks: Dict[str, np.ndarray] = {}
    if roi_masks is not None and isinstance(roi_masks, dict):
        for k in ["tg", "tai", "zhi", "fissure", "tooth_mk"]:
            if k in roi_masks and isinstance(roi_masks[k], np.ndarray):
                m = _to_u8_mask(roi_masks[k])
                if m is not None:
                    norm_masks[k] = m

    tg_area = _mask_area(norm_masks.get("tg"))
    tai_area = _mask_area(norm_masks.get("tai"))
    zhi_area = _mask_area(norm_masks.get("zhi"))
    fissure_area = _mask_area(norm_masks.get("fissure"))
    tooth_area = _mask_area(norm_masks.get("tooth_mk"))

    # tai/zhi ratios (relative to tg)
    outputs["roi_stats"] = {
        "tg_area": float(tg_area),
        "tai_area": float(tai_area),
        "zhi_area": float(zhi_area),
        "fissure_area": float(fissure_area),
        "tooth_mk_area": float(tooth_area),
        "tai_area_ratio": _safe_ratio(tai_area, tg_area),
        "zhi_area_ratio": _safe_ratio(zhi_area, tg_area),
        "fissure_area_ratio": _safe_ratio(fissure_area, tg_area),
        "tooth_mk_area_ratio": _safe_ratio(tooth_area, tg_area),
    }

    # also expose at top-level for convenience (your requirement)
    outputs["tai_area_ratio"] = outputs["roi_stats"]["tai_area_ratio"]
    outputs["zhi_area_ratio"] = outputs["roi_stats"]["zhi_area_ratio"]

    # ---------------------------
    # 1) Save masks paths (optional)
    # ---------------------------
    outputs["roi_masks"] = {}
    if artifacts_dir and sample_id and len(norm_masks) > 0:
        base = Path(artifacts_dir) / sample_id / "masks"
        _ensure_dir(base)
        for k, m in norm_masks.items():
            # save only non-empty
            if _mask_area(m) > 0:
                path = base / f"{k}.png"
                outputs["roi_masks"][k] = _save_mask_png(m, path)

    # ---------------------------
    # 2) Select ROI for demo preview (same logic as your original)
    # ---------------------------
    roi_order = [preferred_roi, "Zhi", "Fissure", "Toothmark"]

    picked_roi = None
    color_tbl = tex_tbl = cnn_tbl = None
    roi_mask_key = None

    for roi in roi_order:
        c_key, t_key, e_key, m_key = _roi_keymap(roi)
        c = tables.get(c_key, None)
        t = tables.get(t_key, None)
        e = tables.get(e_key, None)

        # 只要 CNN 非空，就认为这个 ROI 可用（你现在已验证 CNN 有值）
        if isinstance(e, dict) and len(e) > 0:
            picked_roi = roi
            color_tbl, tex_tbl, cnn_tbl = c, t, e
            roi_mask_key = m_key
            break

    if picked_roi is None:
        # 即使没选中 ROI，也把 masks/stats 返回（便于调试）
        if include_paper_tables:
            outputs["paper_tables"] = tables
        return outputs

    # ---------------------------
    # 3) representation (cnnPC1..10)
    # ---------------------------
    preview = _pick_embedding_preview(cnn_tbl, max_n=10)

    outputs["representation"] = {
        "embedding_dim": 128,
        "embedding_preview": preview,
        "intended_usage": ["similarity_search", "clustering", "downstream_classification"],
        "roi_source": picked_roi,
    }

    # ---------------------------
    # 4) color_features
    # ---------------------------
    if isinstance(color_tbl, dict) and len(color_tbl) > 0:
        key_features = dict(list(color_tbl.items())[:8])
        outputs["color_features"] = {"dim": len(color_tbl), "key_features": key_features, "roi_source": picked_roi}
    else:
        # fallback compute
        if img_bgr is not None and roi_masks is not None and roi_mask_key in roi_masks:
            kf = _compute_simple_color(img_bgr, roi_masks[roi_mask_key])
            if kf:
                outputs["color_features"] = {
                    "dim": len(kf),
                    "key_features": kf,
                    "roi_source": picked_roi,
                    "source": "rule_fallback",
                }

    # ---------------------------
    # 5) texture_features
    # ---------------------------
    if isinstance(tex_tbl, dict) and len(tex_tbl) > 0:
        key_features = dict(list(tex_tbl.items())[:6])
        outputs["texture_features"] = {"dim": len(tex_tbl), "key_features": key_features, "roi_source": picked_roi}
    else:
        # fallback compute
        if img_bgr is not None and roi_masks is not None and roi_mask_key in roi_masks:
            kf = _compute_simple_texture(img_bgr, roi_masks[roi_mask_key])
            if kf:
                outputs["texture_features"] = {
                    "dim": len(kf),
                    "key_features": kf,
                    "roi_source": picked_roi,
                    "source": "rule_fallback",
                }

    # ---------------------------
    # 6) OPTIONAL: include paper tables (this is where tai/zhi P11/P13/P14 live)
    # ---------------------------
    if include_paper_tables:
        # 你要的：P11/P13/P14 tai/zhi 字段来自 bridge 输出表
        # 这里直接把 tables 原样挂出去（最稳，不丢字段）
        outputs["paper_tables"] = tables

    return outputs
