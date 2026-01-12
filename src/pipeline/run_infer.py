from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from src.utils.image_io import read_image_any
from src.utils.postprocess import mask_to_roi

def _pad_to_multiple(img_bgr: np.ndarray, m: int = 32):
    """Pad H/W to multiple of m using border replicate. Return padded_img, (top,bottom,left,right)."""
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
    """Remove padding from mask. pads=(top,bottom,left,right)"""
    pad_top, pad_bottom, pad_left, pad_right = pads
    h, w = mask_255.shape[:2]
    y1 = pad_top
    y2 = h - pad_bottom if pad_bottom > 0 else h
    x1 = pad_left
    x2 = w - pad_right if pad_right > 0 else w
    return mask_255[y1:y2, x1:x2]



def _to_tensor_rgb_01(img_bgr: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    """BGR uint8 -> RGB float32 [1,3,H,W] in [0,1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    img = img_rgb.astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t


def _sigmoid_mask_from_logits(logits: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    """logits [1,1,H,W] -> mask uint8 0/255"""
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask = (prob > thr).astype(np.uint8) * 255
    return mask


def _infer_color_fallback(roi_bgr: np.ndarray) -> Dict[str, Any]:
    roi = roi_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(roi)
    hsv = cv2.cvtColor((roi * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    return {
        "rule": "mean_std_rgb_hsv",
        "rgb_mean": [float(r.mean()), float(g.mean()), float(b.mean())],
        "rgb_std": [float(r.std()), float(g.std()), float(b.std())],
        "hsv_mean": [float(hsv[..., 0].mean()), float(hsv[..., 1].mean()), float(hsv[..., 2].mean())],
    }


def _infer_texture_fallback(roi_bgr: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return {
        "rule": "gray_stats_laplacian_var",
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "laplacian_var": float(lap.var()),
    }


@torch.inference_mode()
def infer_one_image(
    image: Any,
    bundle: Any,
    sample_id: Optional[str] = None,
    device: Optional[str] = None,
    seg_thr: float = 0.5,
) -> Dict[str, Any]:
    """
    image: np.ndarray(BGR uint8) | bytes | str(path)
    bundle: ModelBundle (由 build_bundle/load_bundle 提供)
    """

    # ---- resolve device ----
    if device is None:
        # bundle 里的模型已经被放到对应 device，这里尽量跟随
        device = "cpu"
    device_t = torch.device(device)

    # ---- read image ----
    img_bgr = read_image_any(image)
    H, W = img_bgr.shape[:2]

    # ---- 1) Tongue segmentation ----
    # 约定：bundle.tongue_model 存在，且 forward 输出 [1,1,H,W] logits
    tongue_model = bundle.tongue_model
    tongue_model.eval()

    # ---- pad to avoid UNet cat size mismatch on odd sizes ----
    img_bgr_pad, pads = _pad_to_multiple(img_bgr, m=32)

    x = cv2.cvtColor(img_bgr_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device_t)

    logits = tongue_model(x)  # [1,1,Hpad,Wpad]
    mask_255_pad = _sigmoid_mask_from_logits(logits, thr=seg_thr)

    # ---- unpad back to original size ----
    mask_255 = _unpad_mask(mask_255_pad, pads)

    # ---- 2) ROI crop from mask ----
    roi_bgr, bbox = mask_to_roi(img_bgr, mask_255)

    # ---- 3) ROI 6-class seg (optional) ----
    roi6_mask = None
    if hasattr(bundle, "roi_stage") and bundle.roi_stage is not None:
        # 这里按你的 stage API 适配：你自己在 bundle 里封装即可
        # 先用 try 保证不影响主流程
        try:
            roi6_mask = bundle.roi_stage.predict(roi_bgr)  # 例如返回 dict 或 mask
        except Exception:
            roi6_mask = None

    # ---- 4) P14 embedding ----
    p14_model = bundle.p14_model
    p14_model.eval()

    # 你训练时的输入尺寸（常见 224）
    t = _to_tensor_rgb_01(roi_bgr, size=(224, 224)).to(device_t)
    emb = p14_model(t)

    # 兼容 emb 为 tuple/list/dict 的情况
    if isinstance(emb, (tuple, list)):
        emb = emb[0]
    elif isinstance(emb, dict) and "embedding" in emb:
        emb = emb["embedding"]

    emb_vec = emb.detach().cpu().numpy().reshape(-1).tolist()

    # ---- 5) rule_fallback for color/texture ----
    color_out = _infer_color_fallback(roi_bgr)
    texture_out = _infer_texture_fallback(roi_bgr)

    # ---- 6) build demo JSON ----
    sid = sample_id or "UNKNOWN"

    result = {
        "meta": {
            "id": sid,
            "model_version": "tongue_expert_v1_infer",
            "pipeline": ["seg", "roi", "p14_embedding", "rule_fallback"],
            "image_hw": [int(H), int(W)],
        },
        "artifacts": {
            "bbox_xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            # 如果你想返回 mask/roi base64，后面再加（先跑通）
        },
        "outputs": {
            "representation": {
                "p14_embedding": emb_vec,
                "dim": int(len(emb_vec)),
            },
            "color": color_out,
            "texture": texture_out,
            "roi_6class": roi6_mask,  # optional
            # 给你 app.py 直接 return 的 demo（兼容你现有写法）
            "demo": {
                "representation": {
                    "p14_embedding": emb_vec,
                    "dim": int(len(emb_vec)),
                },
                "color": color_out,
                "texture": texture_out,
            },
        },
    }
    return result
