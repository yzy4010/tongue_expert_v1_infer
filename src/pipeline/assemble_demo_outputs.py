# -*- coding: utf-8 -*-
"""
Assemble demo-style outputs from paper-style tables.

Input:
  tables: Dict[str, Dict]  # e.g. P11_Tg_Color, P12_Tg_Shape, ...

Output:
  demo_outputs: Dict       # color_features / shape_features / ...
"""


# -*- coding: utf-8 -*-
import numpy as np
import cv2

def _pick_first(tables: dict, keys: list):
    for k in keys:
        if k in tables and isinstance(tables[k], dict) and len(tables[k]) > 0:
            return tables[k]
    return None


def _roi_keymap(roi: str):
    # roi: Tai/Zhi/Fissure/Toothmark
    if roi == "Tai":
        return "P21_Tai_Color", "P23_Tai_Texture", "P24_Tai_CNN", "tai"
    if roi == "Zhi":
        return "P31_Zhi_Color", "P33_Zhi_Texture", "P34_Zhi_CNN", "zhi"
    if roi == "Fissure":
        return "P41_Fissure_Color", "P43_Fissure_Texture", "P44_Fissure_CNN", "fissure"
    if roi == "Toothmark":
        return "P51_Toothmark_Color", "P53_Toothmark_Texture", "P54_Toothmark_CNN", "tooth_mk"
    raise ValueError(f"Unknown roi={roi}")


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


def assemble_demo_outputs_from_tables(
    tables: dict,
    preferred_roi: str = "Tai",
    img_bgr=None,
    roi_masks=None,
) -> dict:
    """
    demo 输出优先来自 tables（论文级）
    若 P21/P23 为空，则用 (img_bgr, roi_masks) 做简化补算，保证 demo 不空
    """
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

    outputs = {}
    if picked_roi is None:
        return outputs

    # ---- representation (cnnPC1..10) ----
    pc_keys = [k for k in cnn_tbl.keys() if "cnnPC" in k]
    pc_keys = sorted(pc_keys, key=lambda x: int("".join([c for c in x if c.isdigit()]) or "0"))
    preview = {k: cnn_tbl[k] for k in pc_keys[:10]} if pc_keys else dict(list(cnn_tbl.items())[:10])

    outputs["representation"] = {
        "embedding_dim": 128,
        "embedding_preview": preview,
        "intended_usage": ["similarity_search", "clustering", "downstream_classification"],
        "roi_source": picked_roi,
    }

    # ---- color_features ----
    if isinstance(color_tbl, dict) and len(color_tbl) > 0:
        key_features = dict(list(color_tbl.items())[:8])
        outputs["color_features"] = {"dim": len(color_tbl), "key_features": key_features, "roi_source": picked_roi}
    else:
        # fallback compute
        if img_bgr is not None and roi_masks is not None and roi_mask_key in roi_masks:
            kf = _compute_simple_color(img_bgr, roi_masks[roi_mask_key])
            if kf:
                outputs["color_features"] = {"dim": len(kf), "key_features": kf, "roi_source": picked_roi, "source": "rule_fallback"}

    # ---- texture_features ----
    if isinstance(tex_tbl, dict) and len(tex_tbl) > 0:
        key_features = dict(list(tex_tbl.items())[:6])
        outputs["texture_features"] = {"dim": len(tex_tbl), "key_features": key_features, "roi_source": picked_roi}
    else:
        # fallback compute
        if img_bgr is not None and roi_masks is not None and roi_mask_key in roi_masks:
            kf = _compute_simple_texture(img_bgr, roi_masks[roi_mask_key])
            if kf:
                outputs["texture_features"] = {"dim": len(kf), "key_features": kf, "roi_source": picked_roi, "source": "rule_fallback"}

    return outputs
