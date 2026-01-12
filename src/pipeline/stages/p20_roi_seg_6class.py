from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.mask_ops import fuse_by_priority

NUM_CLASSES = 6


# =========================
# 1) CONFIG
# =========================
@dataclass
class RoiSegConfig:
    ckpt_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # roi model input size (H, W)
    input_size: Tuple[int, int] = (512, 512)

    # normalization
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # if your training used BGR (cv2 default) WITHOUT conversion, set bgr_to_rgb=False
    bgr_to_rgb: bool = True

    use_amp: bool = True
    fill_zhi_from_tongue: bool = True


# =========================
# 2) CHECKPOINT UTILS
# =========================
def _strip_prefix(state: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _unwrap_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Accepts common checkpoint formats:
      - state_dict directly
      - dict with "state_dict"/"model"/"weights"/"net"/"model_state"
    """
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "weights", "net", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # Already a state_dict-like
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Unrecognized checkpoint format; expected a state_dict-like dict.")


def _extract_logits(model_out: Any) -> torch.Tensor:
    """
    Support common forward outputs:
      - Tensor logits
      - Dict with keys: 'out', 'logits', 'pred'
      - Tuple/list where first item is logits
    """
    if isinstance(model_out, torch.Tensor):
        return model_out
    if isinstance(model_out, dict):
        for k in ("out", "logits", "pred"):
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0 and isinstance(model_out[0], torch.Tensor):
        return model_out[0]
    raise TypeError("Model output not supported. Expected Tensor or dict/tuple containing logits Tensor.")


# =========================
# 3) MODEL BUILDER (YOU REPLACE THIS)
# =========================
def build_roi_model(num_classes: int) -> torch.nn.Module:
    """
    Use project UNet implementation at src/models/unet.py
    Supports dynamic num_classes (here we need 6).
    """
    from src.models.unet_roi import UNetROI
    return UNetROI(num_classes=num_classes)



# =========================
# 4) INFER WRAPPER
# =========================
class RoiSeg6ClassInfer:
    def __init__(
        self,
        cfg: RoiSegConfig,
        model_builder: Optional[Callable[[int], torch.nn.Module]] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        if model_builder is None:
            model_builder = build_roi_model

        self.model = model_builder(NUM_CLASSES).to(self.device).eval()
        self._load_weights(cfg.ckpt_path)

    def _load_weights(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ROI seg checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = _unwrap_state_dict(ckpt)

        # Strip common prefixes
        state = _strip_prefix(state, prefixes=("module.", "model.", "net."))

        # Use strict=False to reduce friction; you can switch to strict=True once aligned.
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[ROI-SEG] missing keys (showing up to 12): {missing[:12]}{'...' if len(missing) > 12 else ''}")
        if unexpected:
            print(f"[ROI-SEG] unexpected keys (showing up to 12): {unexpected[:12]}{'...' if len(unexpected) > 12 else ''}")

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("img_bgr must be HxWx3")

        H, W = img_bgr.shape[:2]
        inp_h, inp_w = self.cfg.input_size

        if self.cfg.bgr_to_rgb:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img = img_bgr

        img_resized = cv2.resize(img, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)

        x = img_resized.astype(np.float32) / 255.0
        x = (x - np.array(self.cfg.mean, dtype=np.float32)) / np.array(self.cfg.std, dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))  # CHW
        x = torch.from_numpy(x).unsqueeze(0)  # 1CHW
        return x, (H, W)

    @torch.no_grad()
    def infer_pred_class(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Returns pred_cls: HxW int (0..5) in original image size
        """
        x, orig_hw = self._preprocess(img_bgr)
        x = x.to(self.device)

        use_amp = self.cfg.use_amp and (self.device.type == "cuda")
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            out = self.model(x)
            logits = _extract_logits(out)

        if logits.ndim != 4:
            raise ValueError(f"Unexpected logits ndim: {logits.ndim}, shape={tuple(logits.shape)}")

        if logits.shape[1] != NUM_CLASSES:
            raise ValueError(f"Class mismatch: logits.shape[1]={logits.shape[1]} but expected {NUM_CLASSES}")

        # Resize back to original image size
        logits_up = F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)
        pred = torch.argmax(logits_up, dim=1)[0].detach().cpu().numpy().astype(np.int32)
        return pred

    def infer_roi_masks(self, img_bgr: np.ndarray, tongue_mask_base: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Public API:
          - infer on full image
          - constrain by tongue_mask_base (from unet_tongue_best)
          - fuse by priority + optional zhi fill
        """
        pred_cls = self.infer_pred_class(img_bgr)

        roi_masks = fuse_by_priority(
            pred_cls=pred_cls,
            tongue_mask=tongue_mask_base,
            fill_zhi_from_tongue=self.cfg.fill_zhi_from_tongue,
        )
        return roi_masks
