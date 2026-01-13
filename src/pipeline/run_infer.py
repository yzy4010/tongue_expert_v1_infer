# src/pipeline/run_infer.py
# 目标：返回“论文复现 demo JSON”结构，并且每个模块都带 conclusion（固定阈值版 infer v1）
# 依赖：opencv-python, numpy, torch
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from src.utils.image_io import read_image_any


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
    mask_255: np.ndarray,
    bbox: tuple,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    在原图上绘制：
    - 舌体 mask 半透明 overlay
    - ROI bbox
    """
    overlay = img_bgr.copy()

    # 红色 mask overlay
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255  # BGR -> R
    mask_bool = mask_255 > 0
    overlay[mask_bool] = cv2.addWeighted(
        img_bgr[mask_bool], 1 - alpha, red[mask_bool], alpha, 0
    )

    # ROI bbox
    x1, y1, x2, y2 = bbox
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
# rule_fallback: key features
# =========================
def _infer_color_key_features(roi_bgr: np.ndarray) -> Dict[str, float]:
    roi = roi_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(roi)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    r_mean = float((r * 255.0).mean())
    g_mean = float((g * 255.0).mean())
    b_mean = float((b * 255.0).mean())
    gray_mean = float(gray.mean())

    # LAB L* brightness proxy
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
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


def _infer_texture_key_features(roi_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
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
# Conclusions (fixed-threshold infer v1)
# =========================
def _conclude_color(color_key: Dict[str, float]) -> Dict[str, str]:
    L = float(color_key.get("lightness_mean", 0.0))
    rg = float(color_key.get("rg_diff", 0.0))

    if L >= 165:
        bright_level = "bright_high"
        bright_txt = "舌体整体偏明亮"
    elif L >= 140:
        bright_level = "bright_mid"
        bright_txt = "舌体整体亮度中等"
    else:
        bright_level = "bright_low"
        bright_txt = "舌体整体偏暗"

    if rg >= 12:
        red_txt = "红度偏高"
    elif rg >= 4:
        red_txt = "红度中等"
    else:
        red_txt = "红度偏低"

    return {"level": bright_level, "text_cn": f"{bright_txt}；{red_txt}。"}  # 注意中文分号更自然


def _conclude_texture(texture_key: Dict[str, float]) -> Dict[str, str]:
    """
    固定阈值版（infer v1 推荐）：
    - 主指标：laplacian_var
    - 辅指标：gray_std（用于描述微调）
    """
    lapv = float(texture_key.get("laplacian_var", 0.0))
    gstd = float(texture_key.get("gray_std", 0.0))

    if lapv < 20:
        level = "smooth"
        txt = "表面较平滑，纹理变化较少。"
    elif lapv < 45:
        level = "medium"
        # 可选微调：不改变 level，只让文案更贴合
        if gstd >= 35:
            txt = "表面纹理中等，局部对比较明显。"
        else:
            txt = "表面纹理中等，清晰度中等。"
    else:
        level = "rough"
        txt = "表面纹理较明显，清晰度较高。"

    return {"level": level, "text_cn": txt}


def _conclude_shape(shape_feat: Dict[str, Any]) -> Dict[str, str]:
    ratio = float(shape_feat.get("width_height_ratio", 0.0))

    # 注意：这里 ratio=width/height，通常在 ~0.7~1.2
    if ratio >= 1.05:
        level = "broad"
        txt = "舌体外形偏宽。"
    elif ratio >= 0.85:
        level = "normal"
        txt = "舌体外形宽高比例正常。"
    else:
        level = "slender"
        txt = "舌体外形偏修长。"

    return {"level": level, "text_cn": txt}


def _conclude_representation(emb_vec: np.ndarray) -> Dict[str, str]:
    dim = int(emb_vec.shape[0])
    if dim == 128 and np.isfinite(emb_vec).all():
        return {
            "level": "ok",
            "text_cn": "深度表征已生成，可用于相似度检索、聚类与下游建模。",
        }
    return {
        "level": "warning",
        "text_cn": f"深度表征维度/数值异常（dim={dim}），建议检查模型与输入预处理。",
    }


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
) -> Dict[str, Any]:
    """
    返回结构对齐你示例 demo JSON，并且 outputs 每个模块都带 conclusion（固定阈值版）。
    bundle 至少需要：
      - bundle.tongue_model
      - bundle.p14_model
    """
    device_t = torch.device(device)

    # repo root: .../tongue_expert_v1
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    mask_255 = _unpad_mask(mask_pad, pads)

    # 3) roi crop
    bbox = _mask_to_bbox(mask_255)
    roi_bgr = _crop_roi(img_bgr, bbox)

    # ===== Save analysis images (input/mask/roi/overlay) =====
    analysis_dir = out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    input_path = analysis_dir / f"{sample_id}_input.jpg"
    mask_path = analysis_dir / f"{sample_id}_mask.png"
    roi_path = analysis_dir / f"{sample_id}_roi.jpg"
    overlay_path = analysis_dir / f"{sample_id}_roi_overlay.jpg"

    # 保存原图
    cv2.imencode(".jpg", img_bgr)[1].tofile(str(input_path))

    # 保存 mask
    cv2.imencode(".png", mask_255)[1].tofile(str(mask_path))

    # 保存 ROI
    cv2.imencode(".jpg", roi_bgr)[1].tofile(str(roi_path))

    # 保存 overlay（需要你已有 _draw_roi_overlay）
    overlay_img = _draw_roi_overlay(img_bgr, mask_255, bbox)
    cv2.imencode(".jpg", overlay_img)[1].tofile(str(overlay_path))

    # 生成相对路径（返回给接口）
    # 生成前端友好 URL（配合 app.mount("/static", outputs_dir)）
    input_url = f"/static/analysis/{sample_id}_input.jpg"
    mask_url = f"/static/analysis/{sample_id}_mask.png"
    roi_url = f"/static/analysis/{sample_id}_roi.jpg"
    roi_overlay_url = f"/static/analysis/{sample_id}_roi_overlay.jpg"

    # 4) p14 embedding
    bundle.p14_model.eval()
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    roi_rgb = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(roi_rgb).permute(2, 0, 1).unsqueeze(0).to(device_t)

    emb = bundle.p14_model(t)
    if isinstance(emb, (tuple, list)):
        emb = emb[0]
    emb_vec = emb.detach().cpu().numpy().reshape(-1)  # (128,)

    # 5) features
    color_key = _infer_color_key_features(roi_bgr)
    texture_key = _infer_texture_key_features(roi_bgr)
    shape_feat = _compute_shape_features(mask_255, bbox)

    # 6) conclusions
    color_conc = _conclude_color(color_key)
    texture_conc = _conclude_texture(texture_key)
    shape_conc = _conclude_shape(shape_feat)
    repr_conc = _conclude_representation(emb_vec)

    # 7) save artifacts
    roi_path_rel = None
    mask_path_rel = None
    if save_outputs:
        roi_path = roi_dir / f"{sample_id}.jpg"
        mask_path = mask_dir / f"{sample_id}.png"

        # tofile: Windows 中文路径更稳
        cv2.imencode(".jpg", roi_bgr)[1].tofile(str(roi_path))
        cv2.imencode(".png", mask_255)[1].tofile(str(mask_path))

        roi_path_rel = str(Path("..") / "outputs" / "roi" / "test" / f"{sample_id}.jpg")
        mask_path_rel = str(Path("..") / "outputs" / "pred_masks_original" / "test" / f"{sample_id}.png")

    # 8) interpretation summary
    summary_cn = "；".join([
        color_conc["text_cn"].rstrip("。"),
        shape_conc["text_cn"].rstrip("。"),
        texture_conc["text_cn"].rstrip("。"),
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
    ]

    # 9) final json
    result = {
        "meta": {
            "id": sample_id,
            "model_version": "tongue_expert_v1",
            "pipeline": ["p11_color", "p12_shape", "p13_texture", "p14_embedding"],
            "device": str(device_t),
        },
        "artifacts": {
            "generated": {
                "input_url": input_url,
                "mask_url": mask_url,
                "roi_url": roi_url,
                "roi_overlay_url": roi_overlay_url,
                # 可选：保留本地相对路径用于调试
                "input_path": str(input_path),
                "mask_path": str(mask_path),
                "roi_path": str(roi_path),
                "roi_overlay_path": str(overlay_path),
            }
        },
        "outputs": {
            "color_features": {
                "dim": 76,  # 对齐 demo；当前 key_features 为子集（可逐步补齐）
                "key_features": color_key,
                "conclusion": color_conc,
            },
            "shape_features": {
                **shape_feat,
                "conclusion": shape_conc,
            },
            "texture_features": {
                "dim": 16,  # 对齐 demo；当前 key_features 为子集（可逐步补齐）
                "key_features": texture_key,
                "conclusion": texture_conc,
            },
            "representation": {
                "embedding_dim": int(emb_vec.shape[0]),
                "embedding_preview": _embedding_preview(emb_vec, n=10),
                "intended_usage": ["similarity_search", "clustering", "downstream_classification"],
                "conclusion": repr_conc,
            },
        },
        "interpretation": {
            "summary_cn": summary_cn,
            "signals": signals,
            "disclaimer": "该结果用于科研/工程验证，不构成医疗诊断或建议。",
        },
    }
    return result
