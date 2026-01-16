# src/pipeline/roi_feature_bridge.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import torch


# ============================================================
# Utils
# ============================================================

def replace_prefix(feats: Dict[str, float], old: str, new: str) -> Dict[str, float]:
    """
    将特征字典中的 key 前缀从 old 替换为 new（只替换一次）
    """
    return {k.replace(old, new, 1): float(v) for k, v in feats.items()}


def to_u8_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Normalize mask to uint8 {0,255}. Accept {0,1} or bool or uint8.
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


def mask_area(mask: Optional[np.ndarray]) -> int:
    if mask is None:
        return 0
    return int((mask > 0).sum())


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_by_mask_bbox(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    margin: float = 0.10,
) -> Optional[np.ndarray]:
    """
    用 mask 的 bbox 裁剪 ROI（给 P14 用）
    """
    bb = bbox_from_mask(mask)
    if bb is None:
        return None

    x1, y1, x2, y2 = bb
    H, W = mask.shape[:2]

    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    mx = int(bw * margin)
    my = int(bh * margin)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(W - 1, x2 + mx)
    y2 = min(H - 1, y2 + my)

    return img_bgr[y1:y2 + 1, x1:x2 + 1].copy()


def resize_keep_aspect_pad(
    img_bgr: np.ndarray,
    out_hw: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    oh, ow = out_hw
    h, w = img_bgr.shape[:2]
    scale = min(ow / w, oh / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((oh, ow, 3), dtype=resized.dtype)

    y0 = (oh - nh) // 2
    x0 = (ow - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


# ============================================================
# P11 / P13: regression bridge (shared)
# ============================================================

def run_regression_on_roi(
    infer_regression_fn: Callable,
    roi_dir: Path,
    sample_id: str,
    ckpt_path: str,
    norm_path: str,
    device: str,
    out_dim: Optional[int],
    new_prefix: str,
    label_csv: Optional[str] = None,
) -> Dict[str, float]:
    """
    通用回归入口（P11 / P13 共用）
    注意：这里依赖 roi_dir 下已有该 ROI 的图片（你上游应已输出到 roi_root/roi_name）。
    """
    if not roi_dir.exists():
        return {}

    feats, _, used_ids = infer_regression_fn(
        ckpt_path=ckpt_path,
        norm_path=norm_path,
        roi_dir=str(roi_dir),
        ids=[sample_id],
        device=device,
        out_dim_override=out_dim,
        label_csv=label_csv,
    )

    if len(used_ids) == 0:
        return {}

    # infer_regression 内部产出的是 tg_ 前缀
    return replace_prefix(feats, "tg_", new_prefix)


# ============================================================
# P14: embedding + PCA bridge
# ============================================================

def run_p14_on_roi(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    p14_model,
    pca,
    new_prefix: str,
    device: str,
    embed_hw: Tuple[int, int] = (224, 224),
) -> Dict[str, float]:
    """
    ROI → embedding(128) → PCA → cnnPC1..10

    要求：
      - p14_model 已经 .eval()
      - pca 是 sklearn PCA (has transform)
    """
    crop = crop_by_mask_bbox(img_bgr, mask, margin=0.10)
    if crop is None:
        return {f"{new_prefix}cnnPC{i}": float("nan") for i in range(1, 11)}

    crop_in = resize_keep_aspect_pad(crop, out_hw=embed_hw)

    # === 与训练一致的输入变换（最小版：BGR->RGB, 0-1, CHW）===
    x = crop_in[:, :, ::-1].astype(np.float32) / 255.0  # RGB
    x = np.transpose(x, (2, 0, 1))                      # CHW
    x = torch.from_numpy(x).unsqueeze(0).to(device)     # 1x3xHxW

    with torch.no_grad():
        if hasattr(p14_model, "forward_embedding"):
            emb = p14_model.forward_embedding(x)        # (1, 128)
        else:
            # 备选：不改模型文件，直接取 forward() 的第一个返回值
            emb, _, _ = p14_model(x)

    emb_np = emb.detach().cpu().numpy()                 # (1, 128)
    pc = pca.transform(emb_np)[0]                       # (>=10,)

    return {f"{new_prefix}cnnPC{i}": float(pc[i - 1]) for i in range(1, 11)}


# ============================================================
# Main entry: one sample, all ROI, all modules
# ============================================================

def extract_roi_features_all(
    *,
    img_bgr: np.ndarray,
    roi_masks: Dict[str, np.ndarray],
    sample_id: str,
    roi_root: Path,
    device: str,

    # -------- regression fn (from eval_all_test.py) --------
    infer_regression_fn: Callable,

    # -------- P11 --------
    p11_ckpt: str,
    p11_norm: str,
    p11_dim: Optional[int],

    # -------- P13 --------
    p13_ckpt: str,
    p13_norm: str,
    p13_dim: Optional[int],

    # -------- P14 --------
    p14_model,
    pca,

    min_area: int = 200,
    include_roi_stats: bool = True,
    p11_label_csv: Optional[str] = None,  # ✅ 新增
    p13_label_csv: Optional[str] = None,  # ✅ 新增
) -> Dict[str, Dict[str, float]]:
    """
    对单个样本，输出论文级表结构：

    P21_Tai_Color / P23_Tai_Texture / P24_Tai_CNN
    P31_Zhi_Color / P33_Zhi_Texture / P34_Zhi_CNN
    P41_Fissure_Color / P43_Fissure_Texture / P44_Fissure_CNN
    P51_Toothmark_Color / P53_Toothmark_Texture / P54_Toothmark_CNN

    额外（建议）：
      ROI_Stats: 面积、占比（相对 tg）
    """

    # ---- normalize masks ----
    norm_masks: Dict[str, Optional[np.ndarray]] = {}
    for k, v in (roi_masks or {}).items():
        # 允许把 _debug 等非mask塞进来：直接跳过
        if not isinstance(v, np.ndarray):
            continue
        norm_masks[k] = to_u8_mask(v)

    out: Dict[str, Dict[str, float]] = {}

    # ---- optional ROI stats ----
    if include_roi_stats:
        tg = norm_masks.get("tg")
        tg_a = mask_area(tg)

        tai = norm_masks.get("tai")
        zhi = norm_masks.get("zhi")
        fissure = norm_masks.get("fissure")
        tooth_mk = norm_masks.get("tooth_mk")

        stats: Dict[str, float] = {
            "tg_area": float(tg_a),
            "tai_area": float(mask_area(tai)),
            "zhi_area": float(mask_area(zhi)),
            "fissure_area": float(mask_area(fissure)),
            "tooth_mk_area": float(mask_area(tooth_mk)),
            "tai_area_ratio": safe_ratio(mask_area(tai), tg_a),
            "zhi_area_ratio": safe_ratio(mask_area(zhi), tg_a),
            "fissure_area_ratio": safe_ratio(mask_area(fissure), tg_a),
            "tooth_mk_area_ratio": safe_ratio(mask_area(tooth_mk), tg_a),
        }
        out["ROI_Stats"] = stats

    roi_specs = {
        "tai": (
            "tai_",
            "P21_Tai_Color",
            "P23_Tai_Texture",
            "P24_Tai_CNN",
        ),
        "zhi": (
            "zhi_",
            "P31_Zhi_Color",
            "P33_Zhi_Texture",
            "P34_Zhi_CNN",
        ),
        "fissure": (
            "fissure_",
            "P41_Fissure_Color",
            "P43_Fissure_Texture",
            "P44_Fissure_CNN",
        ),
        "tooth_mk": (
            "tooth_mk_",
            "P51_Toothmark_Color",
            "P53_Toothmark_Texture",
            "P54_Toothmark_CNN",
        ),
    }

    # safety: ensure model eval
    if hasattr(p14_model, "eval"):
        p14_model.eval()

    for roi_name, (prefix, t11, t13, t14) in roi_specs.items():
        mask = norm_masks.get(roi_name)
        roi_dir = roi_root / roi_name

        if mask is None or mask_area(mask) < min_area:
            out[t11] = {}
            out[t13] = {}
            out[t14] = {f"{prefix}cnnPC{i}": float("nan") for i in range(1, 11)}
            continue

        # ---- P11 ----
        out[t11] = run_regression_on_roi(
            infer_regression_fn=infer_regression_fn,
            roi_dir=roi_dir,
            sample_id=sample_id,
            ckpt_path=p11_ckpt,
            norm_path=p11_norm,
            device=device,
            out_dim=p11_dim,
            new_prefix=prefix,
            label_csv=p11_label_csv,
        )

        # ---- P13 ----
        out[t13] = run_regression_on_roi(
            infer_regression_fn=infer_regression_fn,
            roi_dir=roi_dir,
            sample_id=sample_id,
            ckpt_path=p13_ckpt,
            norm_path=p13_norm,
            device=device,
            out_dim=p13_dim,
            new_prefix=prefix,
            label_csv=p13_label_csv,  # ✅ 新增
        )

        # ---- P14 ----
        out[t14] = run_p14_on_roi(
            img_bgr=img_bgr,
            mask=mask,
            p14_model=p14_model,
            pca=pca,
            new_prefix=prefix,
            device=device,
        )

    return out

# ============================================================
# v2 minimal-set ROI tables (coating_* / body_*)
# ============================================================

def _v2_roi_specs_minimal():
    """
    v2 最小集合：仅对 coating_* 与 body_* 五分区输出论文表结构（P60_*）
    ROI 工程命名不带 P60；P60 只在 table key 中出现。
    """
    specs = {}

    parts = [
        ("tip", "Tip"),
        ("center", "Center"),
        ("root", "Root"),
        ("left", "Left"),
        ("right", "Right"),
    ]

    # coating
    for part, cname in parts:
        roi_key = f"coating_{part}"
        prefix = f"coating_{part}_"
        specs[roi_key] = (
            prefix,
            f"P60_Coating{cname}_Color",
            f"P60_Coating{cname}_Texture",
            f"P60_Coating{cname}_CNN",
        )

    # body
    for part, cname in parts:
        roi_key = f"body_{part}"
        prefix = f"body_{part}_"
        specs[roi_key] = (
            prefix,
            f"P60_Body{cname}_Color",
            f"P60_Body{cname}_Texture",
            f"P60_Body{cname}_CNN",
        )

    return specs


def extract_roi_features_v2(
    *,
    img_bgr: np.ndarray,
    roi_masks_v2: Dict[str, np.ndarray],
    sample_id: str,
    roi_root: Path,
    device: str,

    infer_regression_fn: Callable,

    # -------- P11 --------
    p11_ckpt: str,
    p11_norm: str,
    p11_dim: Optional[int],

    # -------- P13 --------
    p13_ckpt: str,
    p13_norm: str,
    p13_dim: Optional[int],

    # -------- P14 --------
    p14_model,
    pca,

    min_area: int = 200,

    p11_label_csv: Optional[str] = None,
    p13_label_csv: Optional[str] = None,

) -> Dict[str, Dict[str, float]]:
    """
    v2 最小集合（coating/body × tip/center/root/left/right）：
    输出论文表结构 tables，用于 JSON：
      P60_CoatingTip_Color / Texture / CNN
      P60_BodyLeft_Color / Texture / CNN
      ...

    注意：
    - P11/P13 从 roi_root/<roi_key>/<id>.jpg 读取（必须先 export）
    - P14 直接用 img + mask 做 embedding（不依赖 roi_dir）
    """
    out: Dict[str, Dict[str, float]] = {}
    roi_specs = _v2_roi_specs_minimal()

    for roi_key, (prefix, t11, t13, t14) in roi_specs.items():
        mask = roi_masks_v2.get(roi_key)
        roi_dir = Path(roi_root) / roi_key

        if mask is None or mask_area(mask) < min_area:
            out[t11] = {}
            out[t13] = {}
            out[t14] = {f"{prefix}cnnPC{i}": np.nan for i in range(1, 11)}
            continue

        # ---- P11 (Color regression) ----
        out[t11] = run_regression_on_roi(
            infer_regression_fn=infer_regression_fn,
            roi_dir=roi_dir,
            sample_id=sample_id,
            ckpt_path=p11_ckpt,
            norm_path=p11_norm,
            device=device,
            out_dim=p11_dim,
            new_prefix=prefix,
            label_csv=p11_label_csv,
        )

        # ---- P13 (Texture regression) ----
        out[t13] = run_regression_on_roi(
            infer_regression_fn=infer_regression_fn,
            roi_dir=roi_dir,
            sample_id=sample_id,
            ckpt_path=p13_ckpt,
            norm_path=p13_norm,
            device=device,
            out_dim=p13_dim,
            new_prefix=prefix,
            label_csv=p13_label_csv,
        )

        # ---- P14 (Embedding -> PCA) ----
        out[t14] = run_p14_on_roi(
            img_bgr=img_bgr,
            mask=mask,
            p14_model=p14_model,
            pca=pca,
            new_prefix=prefix,
            device=device,
        )

    return out

