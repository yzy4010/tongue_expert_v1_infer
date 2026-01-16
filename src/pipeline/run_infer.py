# src/pipeline/run_infer.py
# 目标：返回“论文复现 demo JSON”结构，并且每个模块都带 conclusion（固定阈值版 infer v1 + tai/zhi）
# 依赖：opencv-python, numpy, torch
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from src.utils.image_io import read_image_any

from src.pipeline.rules.tongue_regions import split_tongue_regions, build_tissue_regions, roi_area_stats

from src.pipeline.roi_exporter_v2 import export_v2_roi_images
from src.pipeline.roi_feature_bridge import extract_roi_features_v2
from src.pipeline.regression_infer import infer_regression as infer_regression_fn


# =========================
# UNet pad/unpad (fix cat mismatch)
# =========================
def _pad_to_multiple(img_bgr: np.ndarray, m: int = 32):
    h, w = img_bgr.shape[:2]
    new_h = int(np.ceil(h / m) * m)
    new_w = int(np.ceil(w / m) * m)
    pad_bottom = new_h - h
    pad_right = new_w - w
    pad_top = 0
    pad_left = 0
    if pad_bottom == 0 and pad_right == 0:
        return img_bgr, (0, 0, 0, 0)
    padded = cv2.copyMakeBorder(
        img_bgr, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REPLICATE
    )
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def _unpad_mask(mask_255: np.ndarray, pads):
    pad_top, pad_bottom, pad_left, pad_right = pads
    h, w = mask_255.shape[:2]
    y1 = pad_top
    y2 = h - pad_bottom if pad_bottom > 0 else h
    x1 = pad_left
    x2 = w - pad_right if pad_right > 0 else w
    return mask_255[y1:y2, x1:x2]


def _sigmoid_mask_from_logits(logits: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return (prob > thr).astype(np.uint8) * 255


# =========================
# ROI + shape helpers
# =========================
def _draw_roi_overlay(
    img_bgr: np.ndarray,
    tg_mask_255: np.ndarray,
    bbox: tuple,
    *,
    zhi_mask_255: np.ndarray = None,
    tai_mask_255: np.ndarray = None,
    alpha_tg: float = 0.15,   # 更淡
    alpha_zhi: float = 0.10,  # 更淡但可见
    alpha_tai: float = 0.10,  # 更淡
    alpha_roi_boost: float = 0.08,  # ROI 内额外增强一点点
) -> np.ndarray:
    """
    在原图上绘制（颜色更淡，但层次清晰）：
    - tg mask 半透明 overlay（红）
    - zhi mask 半透明 overlay（蓝）
    - tai mask 半透明 overlay（黄）
    - ROI bbox（绿框）
    """
    overlay = img_bgr.copy()
    x1, y1, x2, y2 = bbox

    def _blend(mask_255: np.ndarray, color_bgr: tuple, alpha: float, roi_boost: float = 0.0):
        if mask_255 is None:
            return
        m = (mask_255 > 0)
        if m.sum() == 0:
            return

        color = np.zeros_like(img_bgr)
        color[:, :, 0] = color_bgr[0]
        color[:, :, 1] = color_bgr[1]
        color[:, :, 2] = color_bgr[2]

        # 全图淡淡叠加
        overlay[m] = cv2.addWeighted(overlay[m], 1 - alpha, color[m], alpha, 0)

        # ROI bbox 内再轻微增强（让用户更容易看清）
        if roi_boost > 0:
            m_roi = np.zeros_like(m)
            m_roi[y1:y2, x1:x2] = True
            m2 = m & m_roi
            if m2.sum() > 0:
                overlay[m2] = cv2.addWeighted(overlay[m2], 1 - roi_boost, color[m2], roi_boost, 0)

    # 叠加顺序：tg(底) -> zhi -> tai（这样 tai 会覆盖在上层，更直观）
    _blend(tg_mask_255, (0, 0, 255), alpha_tg, roi_boost=alpha_roi_boost)      # 红：tg
    _blend(zhi_mask_255, (255, 0, 0), alpha_zhi, roi_boost=alpha_roi_boost)    # 蓝：zhi
    _blend(tai_mask_255, (0, 255, 255), alpha_tai, roi_boost=alpha_roi_boost)  # 黄：tai

    # ROI bbox（最上层，确保可见）
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return overlay




def _mask_to_bbox(mask_255: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_255 > 0)
    h, w = mask_255.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, w, h
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def _crop_roi(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return img_bgr[y1:y2, x1:x2].copy()


def _compute_shape_features(mask_255: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
    area_px = float((mask_255 > 0).sum())
    x1, y1, x2, y2 = bbox
    width_px = float(max(0, x2 - x1))
    height_px = float(max(0, y2 - y1))
    ratio = float(width_px / height_px) if height_px > 0 else 0.0
    return {
        "area_px": area_px,
        "width_px": width_px,
        "height_px": height_px,
        "width_height_ratio": ratio,
        "source": "mask_original_size",
    }


# =========================
# Rule split: tai / zhi (NO training)
# =========================
def _split_tai_zhi_rules(
    img_bgr: np.ndarray,
    tg_mask_255: np.ndarray,
    *,
    hsv_s_max: int = 80,
    hsv_v_min: int = 160,
    lab_b_min: int = 145,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Rule-based split within tongue mask:
      tai: coating (bright & low saturation OR yellowish)
      zhi: tg - tai
    """
    tg = (tg_mask_255 > 0)
    H, W = tg_mask_255.shape[:2]
    if tg.sum() == 0:
        empty = np.zeros((H, W), dtype=np.uint8)
        return empty, empty, {"reason": "empty_tg"}

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Hh, Ss, Vv = cv2.split(hsv)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    Ll, Aa, Bb = cv2.split(lab)

    cand_white = (Ss <= hsv_s_max) & (Vv >= hsv_v_min)
    cand_yellow = (Bb >= lab_b_min) & (Ll >= 120)
    cand = (cand_white | cand_yellow) & tg

    tai = cand.astype(np.uint8) * 255

    # morphology cleanup
    k1 = max(3, int(min(H, W) * 0.01) | 1)
    k2 = max(5, int(min(H, W) * 0.02) | 1)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    tai = cv2.morphologyEx(tai, cv2.MORPH_OPEN, kernel1, iterations=1)
    tai = cv2.morphologyEx(tai, cv2.MORPH_CLOSE, kernel2, iterations=1)

    # force subset
    tai = ((tai > 0) & tg).astype(np.uint8) * 255
    zhi = (tg & ~(tai > 0)).astype(np.uint8) * 255

    tg_area = int(tg.sum())
    tai_area = int((tai > 0).sum())
    zhi_area = int((zhi > 0).sum())

    dbg = {
        "tg_area": tg_area,
        "tai_area": tai_area,
        "zhi_area": zhi_area,
        "tai_area_ratio": float(tai_area) / float(tg_area) if tg_area else 0.0,
        "zhi_area_ratio": float(zhi_area) / float(tg_area) if tg_area else 0.0,
        "params": {
            "hsv_s_max": hsv_s_max,
            "hsv_v_min": hsv_v_min,
            "lab_b_min": lab_b_min,
        },
    }
    return tai, zhi, dbg


def _masked_roi(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int], mask_255: np.ndarray) -> np.ndarray:
    """
    Crop bbox, then set outside-mask pixels to 0 (black).
    mask_255 is in full image coords.
    """
    x1, y1, x2, y2 = bbox
    roi = img_bgr[y1:y2, x1:x2].copy()
    m = (mask_255[y1:y2, x1:x2] > 0)
    roi[~m] = 0
    return roi


def _area_ratio(mask_a_255: np.ndarray, mask_b_255: np.ndarray) -> float:
    a = int((mask_a_255 > 0).sum()) if mask_a_255 is not None else 0
    b = int((mask_b_255 > 0).sum()) if mask_b_255 is not None else 0
    return float(a) / float(b) if b > 0 else 0.0


# =========================
# rule_fallback: key features (mask-aware for tai/zhi)
# =========================
def _infer_color_key_features_masked(roi_bgr_masked: np.ndarray) -> Dict[str, float]:
    # roi_bgr_masked: outside ROI pixels are 0
    roi = roi_bgr_masked.astype(np.float32) / 255.0
    b, g, r = cv2.split(roi)
    gray = cv2.cvtColor(roi_bgr_masked, cv2.COLOR_BGR2GRAY).astype(np.float32)

    r_mean = float((r * 255.0).mean())
    g_mean = float((g * 255.0).mean())
    b_mean = float((b * 255.0).mean())
    gray_mean = float(gray.mean())

    lab = cv2.cvtColor(roi_bgr_masked, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness_mean = float(lab[..., 0].mean())

    rg_diff = float(r_mean - g_mean)
    rb_diff = float(r_mean - b_mean)
    r_over_g = float(r_mean / (g_mean + 1e-6))
    r_over_b = float(r_mean / (b_mean + 1e-6))

    return {
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "lightness_mean": lightness_mean,
        "gray_mean": gray_mean,
        "rg_diff": rg_diff,
        "rb_diff": rb_diff,
        "r_over_g": r_over_g,
        "r_over_b": r_over_b,
    }


def _infer_texture_key_features_masked(roi_bgr_masked: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(roi_bgr_masked, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "laplacian_var": float(lap.var()),
    }


def _embedding_preview(vec: np.ndarray, n: int = 10) -> Dict[str, float]:
    out = {}
    for i in range(min(n, vec.shape[0])):
        out[f"e{i:03d}"] = float(vec[i])
    return out


# =========================
# Conclusions (fixed-threshold infer v1 + tai/zhi)
# =========================
def _conclude_color(color_key: Dict[str, float]) -> Dict[str, str]:
    L = float(color_key.get("lightness_mean", 0.0))
    rg = float(color_key.get("rg_diff", 0.0))

    if L >= 165:
        bright_level = "bright_high"
        bright_txt = "整体偏明亮"
    elif L >= 140:
        bright_level = "bright_mid"
        bright_txt = "整体亮度中等"
    else:
        bright_level = "bright_low"
        bright_txt = "整体偏暗"

    if rg >= 12:
        red_txt = "红度偏高"
    elif rg >= 4:
        red_txt = "红度中等"
    else:
        red_txt = "红度偏低"

    return {"level": bright_level, "text_cn": f"{bright_txt}；{red_txt}。"}  # 中文分号


def _conclude_texture(texture_key: Dict[str, float]) -> Dict[str, str]:
    lapv = float(texture_key.get("laplacian_var", 0.0))
    gstd = float(texture_key.get("gray_std", 0.0))

    if lapv < 20:
        level = "smooth"
        txt = "纹理较平滑，变化较少。"
    elif lapv < 45:
        level = "medium"
        if gstd >= 35:
            txt = "纹理中等，局部对比较明显。"
        else:
            txt = "纹理中等，清晰度中等。"
    else:
        level = "rough"
        txt = "纹理较明显，清晰度较高。"

    return {"level": level, "text_cn": txt}


def _conclude_shape(shape_feat: Dict[str, Any]) -> Dict[str, str]:
    ratio = float(shape_feat.get("width_height_ratio", 0.0))

    if ratio >= 1.05:
        level = "broad"
        txt = "外形偏宽。"
    elif ratio >= 0.85:
        level = "normal"
        txt = "宽高比例正常。"
    else:
        level = "slender"
        txt = "外形偏修长。"

    return {"level": level, "text_cn": txt}


def _conclude_representation(emb_vec: np.ndarray) -> Dict[str, str]:
    dim = int(emb_vec.shape[0])
    if dim == 128 and np.isfinite(emb_vec).all():
        return {"level": "ok", "text_cn": "深度表征已生成，可用于相似度检索、聚类与下游建模。"}
    return {"level": "warning", "text_cn": f"深度表征维度/数值异常（dim={dim}），建议检查模型与输入预处理。"}  # noqa


def _conclude_tai_zhi(
    *,
    tai_ratio: float,
    zhi_ratio: float,
    tai_color: Dict[str, float],
    zhi_color: Dict[str, float],
) -> Dict[str, Any]:
    """
    给 tai/zhi 一个“可解释、稳定”的总结性结论：
      - tai_ratio 大小：覆盖程度
      - tai 的亮度/饱和度 proxy：lightness_mean 作为“苔偏白/偏厚”的粗指示（仅工程）
      - zhi 的 rg_diff：红度趋势（仅工程）
    """
    # coverage
    if tai_ratio >= 0.55:
        cov_level = "heavy"
        cov_txt = "舌苔覆盖较多"
    elif tai_ratio >= 0.30:
        cov_level = "medium"
        cov_txt = "舌苔覆盖中等"
    elif tai_ratio >= 0.12:
        cov_level = "light"
        cov_txt = "舌苔覆盖较少"
    else:
        cov_level = "very_light"
        cov_txt = "舌苔覆盖很少"

    # tai brightness proxy
    tai_L = float(tai_color.get("lightness_mean", 0.0))
    if tai_L >= 170:
        tai_tone = "white_like"
        tai_txt = "苔色偏浅"
    elif tai_L >= 145:
        tai_tone = "mid"
        tai_txt = "苔色中等"
    else:
        tai_tone = "dark_like"
        tai_txt = "苔色偏深"

    # zhi redness proxy
    zhi_rg = float(zhi_color.get("rg_diff", 0.0))
    if zhi_rg >= 12:
        zhi_txt = "舌质红度偏高"
    elif zhi_rg >= 4:
        zhi_txt = "舌质红度中等"
    else:
        zhi_txt = "舌质红度偏低"

    return {
        "level": cov_level,
        "text_cn": f"{cov_txt}（tai占比≈{tai_ratio:.2f}）；{tai_txt}；{zhi_txt}。",
        "signals": {
            "tai_area_ratio": float(tai_ratio),
            "zhi_area_ratio": float(zhi_ratio),
            "tai_lightness_mean": tai_L,
            "zhi_rg_diff": zhi_rg,
            "tai_tone": tai_tone,
        },
        "disclaimer": "该结论为工程阈值版描述，仅用于科研/工程验证。",
    }

def _classify_tai_color(
    img_bgr: np.ndarray,
    tai_mask_255: np.ndarray,
) -> Dict[str, Any]:
    """
    在 tai 区域内判断苔色：偏白/偏黄/偏黑(偏暗)/混合
    返回：label + 文案 + 关键统计，方便你调阈值
    """
    m = (tai_mask_255 > 0)
    if m.sum() < 50:
        return {
            "label": "na",
            "text_cn": "tai区域过小，无法稳定判断苔色。",
            "stats": {},
        }

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0][m].astype(np.float32)
    B = lab[..., 2][m].astype(np.float32)   # b*: 越大越黄

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1][m].astype(np.float32)

    L_mean = float(L.mean())
    b_mean = float(B.mean())
    S_mean = float(S.mean())

    # ---- thresholds (可调) ----
    dark_L = 135.0
    yellow_b = 140.0
    white_L = 150.0
    white_S = 70.0

    # ---- decision ----
    if L_mean < dark_L:
        label = "dark"
        txt = "舌苔颜色偏暗（偏黑/偏灰）。"
    elif b_mean >= yellow_b:
        label = "yellow"
        txt = "舌苔颜色偏黄。"
    elif (S_mean < white_S) and (L_mean >= white_L):
        label = "white"
        txt = "舌苔颜色偏白。"
    else:
        label = "mixed"
        txt = "舌苔颜色呈混合/中间态（非明显白/黄/暗）。"

    return {
        "label": label,
        "text_cn": txt,
        "stats": {
            "L_mean": L_mean,
            "b_mean": b_mean,
            "S_mean": S_mean,
            "thresholds": {
                "dark_L": dark_L,
                "yellow_b": yellow_b,
                "white_L": white_L,
                "white_S": white_S,
            },
        },
    }
def _get_first_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


# =========================
# Main infer
# =========================
def infer_one_image(
    image: Any,
    bundle: Any,
    sample_id: str,
    device: str = "cpu",
    seg_thr: float = 0.5,
    save_outputs: bool = True,
    # tai/zhi rules params (exposed)
    tai_hsv_s_max: int = 80,
    tai_hsv_v_min: int = 160,
    tai_lab_b_min: int = 145,
) -> Dict[str, Any]:
    """
    返回结构对齐你示例 demo JSON，并且 outputs 每个模块都带 conclusion（固定阈值版）。
    新增：tai/zhi 规则拆分 + 结论。
    """
    device_t = torch.device(device)

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # === P11 / P13 label csv (feature names) ===
    p11_label_csv = str(PROJECT_ROOT / "data" / "labels" / "p11_tg_color.csv")
    p13_label_csv = str(PROJECT_ROOT / "data" / "labels" / "p13_tg_texture.csv")

    out_root = PROJECT_ROOT / "outputs"
    roi_dir = out_root / "roi" / "test"
    mask_dir = out_root / "pred_masks_original" / "test"
    roi_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 1) read image
    img_bgr = read_image_any(image)
    H, W = img_bgr.shape[:2]

    # 2) tongue seg (pad->infer->unpad)
    img_pad, pads = _pad_to_multiple(img_bgr, m=32)
    x = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device_t)

    bundle.tongue_model.eval()
    logits = bundle.tongue_model(x)
    mask_pad = _sigmoid_mask_from_logits(logits, thr=seg_thr)
    mask_255 = _unpad_mask(mask_pad, pads)  # tg mask

    # 3) roi crop (tg roi)
    bbox = _mask_to_bbox(mask_255)
    roi_bgr = _crop_roi(img_bgr, bbox)

    # ===== Save analysis images (input/mask/roi/overlay + tai/zhi masks) =====
    analysis_dir = out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    input_path = analysis_dir / f"{sample_id}_input.jpg"
    mask_path = analysis_dir / f"{sample_id}_tg_mask.png"
    roi_path = analysis_dir / f"{sample_id}_roi.jpg"
    overlay_path = analysis_dir / f"{sample_id}_roi_overlay.jpg"

    cv2.imencode(".jpg", img_bgr)[1].tofile(str(input_path))
    cv2.imencode(".png", mask_255)[1].tofile(str(mask_path))
    cv2.imencode(".jpg", roi_bgr)[1].tofile(str(roi_path))



    input_url = f"/static/analysis/{sample_id}_input.jpg"
    tg_mask_url = f"/static/analysis/{sample_id}_tg_mask.png"
    roi_url = f"/static/analysis/{sample_id}_roi.jpg"
    roi_overlay_url = f"/static/analysis/{sample_id}_roi_overlay.jpg"

    # 3.5) tai/zhi split (rules)
    tai_mask_255, zhi_mask_255, tai_dbg = _split_tai_zhi_rules(
        img_bgr,
        mask_255,
        hsv_s_max=tai_hsv_s_max,
        hsv_v_min=tai_hsv_v_min,
        lab_b_min=tai_lab_b_min,
    )
    tai_color_cls = _classify_tai_color(img_bgr, tai_mask_255)

    # =========================================================
    # v2 ROI (minimal set): tongue regions + tissue x region
    # 说明：
    # - 不改原项目的 tg/tai/zhi 命名；这里只新增一套 v2 规范命名
    # - tongue_* 由 tg_mask 生成（PCA 主轴分区）
    # - coating_* / body_* 由 tai/zhi 与 tongue_* 求交集得到
    # =========================================================

    # v2 base masks (0/255)
    tongue_mask_255 = mask_255  # tg
    coating_mask_255 = tai_mask_255
    body_mask_255 = zhi_mask_255

    # 1) spatial regions on tongue_mask
    tongue_regions = split_tongue_regions(
        tongue_mask_255,
        ratios=(0.30, 0.40, 0.30),  # tip/center/root
    )

    # 2) tissue x region (做优：coating_* / body_*)
    tissue_regions = build_tissue_regions(
        coating_mask_255,
        body_mask_255,
        tongue_regions,
    )

    # 3) collect v2 roi masks
    roi_masks_v2 = {
        "tongue_mask": tongue_mask_255,
        "coating_mask": coating_mask_255,
        "body_mask": body_mask_255,
        **tongue_regions,  # tongue_tip/center/root/left/right
        **tissue_regions,  # coating_tip.. body_tip..
    }

    # =========================================================
    # Step 2 (v2): export ROI images for P11 / P13
    # 必须在 result 组装之前执行
    # =========================================================
    v2_roi_img_paths = export_v2_roi_images(
        img_bgr=img_bgr,
        roi_masks_v2=roi_masks_v2,
        sample_id=sample_id,
        roi_root=roi_dir,  # outputs/roi/test
        min_area=200,
        margin=0.10,
        save_masked=True,
    )

    # =========================================================
    # Step 4: v2 ROI -> P11 / P13 / P14 tables (paper-style)
    # =========================================================

    # ---- resolve bundle fields (compat) ----
    p11_ckpt = _get_first_attr(bundle, ["p11_ckpt", "P11_CKPT", "ckpt_p11", "p11_checkpoint"])
    p11_norm = _get_first_attr(bundle, ["p11_norm", "P11_NORM", "norm_p11", "p11_normalizer"])
    p11_dim = _get_first_attr(bundle, ["p11_dim", "P11_DIM", "dim_p11"], None)

    p13_ckpt = _get_first_attr(bundle, ["p13_ckpt", "P13_CKPT", "ckpt_p13", "p13_checkpoint"])
    p13_norm = _get_first_attr(bundle, ["p13_norm", "P13_NORM", "norm_p13", "p13_normalizer"])
    p13_dim = _get_first_attr(bundle, ["p13_dim", "P13_DIM", "dim_p13"], None)

    p14_model = _get_first_attr(bundle, ["p14_model", "P14_MODEL", "model_p14"])
    pca = _get_first_attr(bundle, ["pca", "PCA", "p14_pca", "pca_model"])

    tables_v2 = extract_roi_features_v2(
        img_bgr=img_bgr,
        roi_masks_v2=roi_masks_v2,
        sample_id=sample_id,
        roi_root=roi_dir,
        device=str(device_t),

        infer_regression_fn=infer_regression_fn,

        p11_ckpt=p11_ckpt,
        p11_norm=p11_norm,
        p11_dim=p11_dim,

        p13_ckpt=p13_ckpt,
        p13_norm=p13_norm,
        p13_dim=p13_dim,

        p11_label_csv=p11_label_csv,
        p13_label_csv=p13_label_csv,

        p14_model=p14_model,
        pca=pca,

    )

    # 4) stats (area & ratio, ratio ref=tongue_mask)
    roi_stats_v2 = roi_area_stats(roi_masks_v2, ref_key="tongue_mask")

    # 5) optional: save masks for debugging / bridging
    # 保存到 outputs/analysis 和 outputs/pred_masks_original/test
    if save_outputs:
        # analysis_dir / mask_dir 你前面已经创建好了
        for k, m in roi_masks_v2.items():
            if not isinstance(m, np.ndarray):
                continue
            # 保存到 analysis：方便前端/调试查看
            p = analysis_dir / f"{sample_id}_{k}.png"
            cv2.imencode(".png", m)[1].tofile(str(p))

            # 保存到 pred_masks_original：和 tg_mask 同级输出
            p2 = mask_dir / f"{sample_id}_{k}.png"
            cv2.imencode(".png", m)[1].tofile(str(p2))

    # 再生成 overlay（包含 zhi）
    overlay_img = _draw_roi_overlay(
        img_bgr=img_bgr,
        tg_mask_255=mask_255,
        bbox=bbox,
        zhi_mask_255=zhi_mask_255,
        tai_mask_255=tai_mask_255,
    )
    cv2.imencode(".jpg", overlay_img)[1].tofile(str(overlay_path))

    tai_mask_path = analysis_dir / f"{sample_id}_tai_mask.png"
    zhi_mask_path = analysis_dir / f"{sample_id}_zhi_mask.png"
    cv2.imencode(".png", tai_mask_255)[1].tofile(str(tai_mask_path))
    cv2.imencode(".png", zhi_mask_255)[1].tofile(str(zhi_mask_path))
    tai_mask_url = f"/static/analysis/{sample_id}_tai_mask.png"
    zhi_mask_url = f"/static/analysis/{sample_id}_zhi_mask.png"

    # 4) p14 embedding (keep your original: on tg ROI crop)
    bundle.p14_model.eval()
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    roi_rgb = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(roi_rgb).permute(2, 0, 1).unsqueeze(0).to(device_t)

    emb = bundle.p14_model(t)
    if isinstance(emb, (tuple, list)):
        emb = emb[0]
    emb_vec = emb.detach().cpu().numpy().reshape(-1)  # (128,)

    # 5) features (tg ROI as before)
    color_key = _infer_color_key_features_masked(roi_bgr)
    texture_key = _infer_texture_key_features_masked(roi_bgr)
    shape_feat = _compute_shape_features(mask_255, bbox)

    # 5.5) tai/zhi features (mask-aware)
    tai_roi_masked = _masked_roi(img_bgr, bbox, tai_mask_255)
    zhi_roi_masked = _masked_roi(img_bgr, bbox, zhi_mask_255)

    tai_color_key = _infer_color_key_features_masked(tai_roi_masked)
    tai_texture_key = _infer_texture_key_features_masked(tai_roi_masked)
    zhi_color_key = _infer_color_key_features_masked(zhi_roi_masked)
    zhi_texture_key = _infer_texture_key_features_masked(zhi_roi_masked)

    tai_ratio = _area_ratio(tai_mask_255, mask_255)
    zhi_ratio = _area_ratio(zhi_mask_255, mask_255)

    # 6) conclusions
    color_conc = _conclude_color(color_key)
    texture_conc = _conclude_texture(texture_key)
    shape_conc = _conclude_shape(shape_feat)
    repr_conc = _conclude_representation(emb_vec)

    tai_color_conc = _conclude_color(tai_color_key) if tai_color_key else {"level": "na", "text_cn": "tai区域过小，无法稳定计算颜色结论。"}
    tai_texture_conc = _conclude_texture(tai_texture_key) if tai_texture_key else {"level": "na", "text_cn": "tai区域过小，无法稳定计算纹理结论。"}
    zhi_color_conc = _conclude_color(zhi_color_key) if zhi_color_key else {"level": "na", "text_cn": "zhi区域过小，无法稳定计算颜色结论。"}
    zhi_texture_conc = _conclude_texture(zhi_texture_key) if zhi_texture_key else {"level": "na", "text_cn": "zhi区域过小，无法稳定计算纹理结论。"}
    tai_zhi_conc = _conclude_tai_zhi(
        tai_ratio=tai_ratio,
        zhi_ratio=zhi_ratio,
        tai_color=tai_color_key or {},
        zhi_color=zhi_color_key or {},
    )

    # 7) save artifacts (outputs/roi & outputs/pred_masks_original)
    roi_path_rel = None
    tg_mask_path_rel = None
    tai_mask_path_rel = None
    zhi_mask_path_rel = None

    if save_outputs:
        roi_path2 = roi_dir / f"{sample_id}.jpg"
        tg_mask_path2 = mask_dir / f"{sample_id}_tg.png"
        tai_mask_path2 = mask_dir / f"{sample_id}_tai.png"
        zhi_mask_path2 = mask_dir / f"{sample_id}_zhi.png"

        cv2.imencode(".jpg", roi_bgr)[1].tofile(str(roi_path2))
        cv2.imencode(".png", mask_255)[1].tofile(str(tg_mask_path2))
        cv2.imencode(".png", tai_mask_255)[1].tofile(str(tai_mask_path2))
        cv2.imencode(".png", zhi_mask_255)[1].tofile(str(zhi_mask_path2))

        roi_path_rel = str(Path("..") / "outputs" / "roi" / "test" / f"{sample_id}.jpg")
        tg_mask_path_rel = str(Path("..") / "outputs" / "pred_masks_original" / "test" / f"{sample_id}_tg.png")
        tai_mask_path_rel = str(Path("..") / "outputs" / "pred_masks_original" / "test" / f"{sample_id}_tai.png")
        zhi_mask_path_rel = str(Path("..") / "outputs" / "pred_masks_original" / "test" / f"{sample_id}_zhi.png")

    # 8) interpretation summary
    summary_cn = "；".join([
        color_conc["text_cn"].rstrip("。"),
        shape_conc["text_cn"].rstrip("。"),
        texture_conc["text_cn"].rstrip("。"),
        tai_zhi_conc["text_cn"].rstrip("。"),
    ]) + "。"

    signals = [
        {
            "domain": "color",
            "feature": "lightness_mean",
            "value": float(color_key.get("lightness_mean", 0.0)),
            "interpretation": color_conc["level"],
        },
        {
            "domain": "texture",
            "feature": "laplacian_var",
            "value": float(texture_key.get("laplacian_var", 0.0)),
            "interpretation": texture_conc["level"],
        },
        {
            "domain": "shape",
            "feature": "width_height_ratio",
            "value": float(shape_feat.get("width_height_ratio", 0.0)),
            "interpretation": shape_conc["level"],
        },
        {
            "domain": "tai_zhi",
            "feature": "tai_area_ratio",
            "value": float(tai_ratio),
            "interpretation": tai_zhi_conc["level"],
        },
    ]

    # 9) final json
    result = {
        "meta": {
            "id": sample_id,
            "model_version": "tongue_expert_v1_tai_zhi_rules",
            "pipeline": ["tongue_seg", "tai_zhi_rules", "p11_color_demo", "p12_shape_demo", "p13_texture_demo", "p14_embedding_demo"],
            "device": str(device_t),
        },
        "artifacts": {
            "generated": {
                # 前端 URL
                "input_url": input_url,
                "tg_mask_url": tg_mask_url,
                "tai_mask_url": tai_mask_url,
                "zhi_mask_url": zhi_mask_url,
                "roi_url": roi_url,
                "roi_overlay_url": roi_overlay_url,

                # 本地路径（调试）
                "input_path": str(input_path),
                "tg_mask_path": str(mask_path),
                "tai_mask_path": str(tai_mask_path),
                "zhi_mask_path": str(zhi_mask_path),
                "roi_path": str(roi_path),
                "roi_overlay_path": str(overlay_path),

                # outputs 目录相对路径（如你需要）
                "roi_path_rel": roi_path_rel,
                "tg_mask_path_rel": tg_mask_path_rel,
                "tai_mask_path_rel": tai_mask_path_rel,
                "zhi_mask_path_rel": zhi_mask_path_rel,
            },
            "rules_debug": {
                "tai_zhi_rules": tai_dbg,
            },
        },
        "outputs": {
            # ---- tg roi 的 demo 特征（保持兼容）----
            "color_features": {
                "dim": 76,
                "key_features": color_key,
                "conclusion": color_conc,
                "roi_source": "Tg",
            },
            "shape_features": {
                **shape_feat,
                "conclusion": shape_conc,
                "roi_source": "Tg",
            },
            "texture_features": {
                "dim": 16,
                "key_features": texture_key,
                "conclusion": texture_conc,
                "roi_source": "Tg",
            },
            "representation": {
                "embedding_dim": int(emb_vec.shape[0]),
                "embedding_preview": _embedding_preview(emb_vec, n=10),
                "intended_usage": ["similarity_search", "clustering", "downstream_classification"],
                "conclusion": repr_conc,
                "roi_source": "Tg",
            },

            # ---- 新增：tai/zhi ----
            "roi_stats": {
                "tai_area_ratio": float(tai_ratio),
                "zhi_area_ratio": float(zhi_ratio),
                "tg_area": float((mask_255 > 0).sum()),
                "tai_area": float((tai_mask_255 > 0).sum()),
                "zhi_area": float((zhi_mask_255 > 0).sum()),
            },
            "tai": {
                "tai_color_classification": tai_color_cls,
                "color_features": {
                    "key_features": tai_color_key,
                    "conclusion": tai_color_conc,
                },
                "texture_features": {
                    "key_features": tai_texture_key,
                    "conclusion": tai_texture_conc,
                },
            },
            "zhi": {
                "color_features": {
                    "key_features": zhi_color_key,
                    "conclusion": zhi_color_conc,
                },
                "texture_features": {
                    "key_features": zhi_texture_key,
                    "conclusion": zhi_texture_conc,
                },
            },
            "tai_zhi_conclusion": tai_zhi_conc,
            "roi_v2": {
                "roi_keys": sorted(list(roi_masks_v2.keys())),
                "stats": roi_stats_v2,
                "roi_img_paths": v2_roi_img_paths
            },
            "tables_v2": tables_v2,
        },
        "interpretation": {
            "summary_cn": summary_cn,
            "signals": signals,
            "disclaimer": "该结果用于科研/工程验证，不构成医疗诊断或建议。",
        },
    }
    return result
