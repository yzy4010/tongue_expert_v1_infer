# -*- coding: utf-8 -*-
"""
Build and cache model bundle for inference.

This module is responsible for:
- loading models ONCE
- holding paths & shared resources
- providing a unified bundle to infer_one_image
"""

from dataclasses import dataclass
import torch
import joblib

# ===== imports from your existing project =====
from src.models.unet import UNet
from src.pipeline.stages.p20_roi_seg_6class import RoiSeg6ClassInfer, RoiSegConfig
# from scripts.infer_p14_embedding import P14MultiTaskNet
from src.models.p14_multitask import P14MultiTaskNet
from pathlib import Path

# 项目根目录：.../tongue_expert_v1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# build_bundle.py 在 src/pipeline/ 下，parents[0]=pipeline, [1]=src, [2]=repo root


# -------------------------------------------------
# Dataclasses (optional but recommended)
# -------------------------------------------------

@dataclass
class ModelPaths:
    # segmentation
    tongue_ckpt: Path

    # ROI seg
    roi_ckpt: Path

    # P14
    p14_ckpt: Path
    p14_pca: Path

    # P11 / P13 (for legacy bridge)
    p11_ckpt: Path
    p11_norm: Path
    p13_ckpt: Path
    p13_norm: Path



@dataclass
class ModelBundle:
    device: str
    paths: ModelPaths

    tongue_model: torch.nn.Module
    roi_infer: RoiSeg6ClassInfer
    p14_model: torch.nn.Module
    pca: object


# -------------------------------------------------
# Main builder
# -------------------------------------------------


def build_model_bundle(device: str = "cpu") -> ModelBundle:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ===== paths =====
    paths = ModelPaths(
        tongue_ckpt=PROJECT_ROOT / "checkpoints" / "seg" / "unet_tongue_best.pth",
        roi_ckpt=PROJECT_ROOT / "checkpoints" / "roi_seg" / "roi_seg_6class_v1_best.pth",
        p14_ckpt=PROJECT_ROOT / "checkpoints" / "p14" / "p14_multitask_best.pth",
        p14_pca=PROJECT_ROOT / "checkpoints" / "p14" / "p14_pca.pkl",

        # 你说部署禁用 P11/P13 回归，但先把路径也统一；后面可以不加载它们
        p11_ckpt=PROJECT_ROOT / "checkpoints" / "p11" / "p11_color_best.pth",
        p11_norm=PROJECT_ROOT / "checkpoints" / "p11" / "p11_norm.npz",
        p13_ckpt=PROJECT_ROOT / "checkpoints" / "p13" / "p13_texture_best.pth",
        p13_norm=PROJECT_ROOT / "checkpoints" / "p13" / "p13_norm.npz",
    )

    # ===== 1. Tongue segmentation model =====

    ckpt = Path(paths.tongue_ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt} (PROJECT_ROOT={PROJECT_ROOT})")
    tongue_model = UNet().to(device)
    tongue_model.load_state_dict(
        torch.load(paths.tongue_ckpt, map_location=device)
    )
    tongue_model.eval()

    # ===== 2. ROI inference =====
    roi_cfg = RoiSegConfig(
        ckpt_path=str(paths.roi_ckpt),
        device=device,
        input_size=(512, 512),
        bgr_to_rgb=True,
        fill_zhi_from_tongue=True,
    )
    roi_infer = RoiSeg6ClassInfer(cfg=roi_cfg)

    # ===== 3. P14 model =====
    p14_model = P14MultiTaskNet(
        emb_dim=128,
        p11_dim=76,
        p13_dim=16,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(paths.p14_ckpt, map_location=device)
    assert "state_dict" in ckpt, "Invalid P14 checkpoint format"
    p14_model.load_state_dict(ckpt["state_dict"], strict=True)
    p14_model.eval()

    # ===== 4. PCA =====
    pca = joblib.load(paths.p14_pca)

    # ===== bundle =====
    bundle = ModelBundle(
        device=device,
        paths=paths,
        tongue_model=tongue_model,
        roi_infer=roi_infer,
        p14_model=p14_model,
        pca=pca,
    )

    return bundle
