# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import joblib

from src.models.unet import UNet
from scripts.infer_p14_embedding import P14MultiTaskNet  # 你现在用的 import 路径
from src.pipeline.stages.p20_roi_seg_6class import RoiSeg6ClassInfer, RoiSegConfig


def _load_state_dict_compatible(ckpt_obj):
    """
    兼容两种保存格式：
    1) 直接是 state_dict
    2) dict 包含 'state_dict'
    """
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]
    return ckpt_obj


@dataclass
class ModelPaths:
    tongue_ckpt: str = "../checkpoints/seg/unet_tongue_best.pth"
    p14_ckpt: str = "../checkpoints/p14/p14_multitask_best.pth"
    p14_pca: str = "../checkpoints/p14/p14_pca.pkl"
    roi6_ckpt: str = "../checkpoints/roi_seg/roi_seg_6class_v1_best.pth"
    p11_ckpt: str = "../checkpoints/p11/p11_color_best.pth"
    p11_norm: str = "../checkpoints/p11/p11_norm.npz"
    p13_ckpt: str = "../checkpoints/p13/p13_texture_best.pth"
    p13_norm: str = "../checkpoints/p13/p13_norm.npz"


@dataclass
class ModelBundle:
    device: str
    paths: ModelPaths

    tongue_model: Optional[torch.nn.Module] = None
    p14_model: Optional[torch.nn.Module] = None
    pca: Optional[object] = None
    roi6_infer: Optional[RoiSeg6ClassInfer] = None

    def load_all(self):
        self._load_tongue()
        self._load_p14_and_pca()
        self._load_roi6()
        return self

    def _load_tongue(self):
        m = UNet().to(self.device)
        ckpt = torch.load(self.paths.tongue_ckpt, map_location=self.device)
        m.load_state_dict(_load_state_dict_compatible(ckpt), strict=True)
        m.eval()
        self.tongue_model = m

    def _load_p14_and_pca(self):
        emb_dim = 128
        p11_dim = 76
        p13_dim = 16
        dropout = 0.0

        m = P14MultiTaskNet(
            emb_dim=emb_dim,
            p11_dim=p11_dim,
            p13_dim=p13_dim,
            dropout=dropout,
        ).to(self.device)

        ckpt = torch.load(self.paths.p14_ckpt, map_location=self.device)
        m.load_state_dict(_load_state_dict_compatible(ckpt), strict=True)
        m.eval()
        self.p14_model = m

        self.pca = joblib.load(self.paths.p14_pca)

    def _load_roi6(self):
        roi_cfg = RoiSegConfig(
            ckpt_path=self.paths.roi6_ckpt,
            device=self.device,
            input_size=(512, 512),
            bgr_to_rgb=True,
            fill_zhi_from_tongue=True,
        )
        self.roi6_infer = RoiSeg6ClassInfer(cfg=roi_cfg)
