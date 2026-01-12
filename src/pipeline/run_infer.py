# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import cv2
import torch
from src.pipeline.assemble_demo_outputs import assemble_demo_outputs_from_tables

from src.pipeline.model_bundle import ModelBundle, ModelPaths
from src.pipeline.rules.roi_rules_tai_zhi import infer_tai_zhi_from_tongue_mask

# 复用你现有的“推理/特征桥”
from src.pipeline.roi_feature_bridge import extract_roi_features_all
from scripts.eval_all_test import infer_regression


def _to_py(v):
    """把 numpy/torch 标量转成可 JSON 序列化的 Python 标量"""
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.integer, np.int32, np.int64)):
        return int(v)
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
        if v.ndim == 0:
            return float(v)
        return v.tolist()
    return v


def _infer_tongue_mask_from_model(tongue_model, img_bgr: np.ndarray, device: str) -> np.ndarray:
    """
    复用你已经验证过的舌体分割逻辑：resize->sigmoid->threshold->resize回原图
    """
    with torch.no_grad():
        x = cv2.resize(img_bgr, (512, 512))
        x = x.astype("float32") / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
        logits = tongue_model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    tongue_mask = (prob > 0.5).astype(np.uint8)
    tongue_mask = cv2.resize(
        tongue_mask,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return tongue_mask


def build_bundle(device: str = None) -> ModelBundle:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = ModelPaths(
        tongue_ckpt="../checkpoints/seg/unet_tongue_best.pth",
        p14_ckpt="../checkpoints/p14/p14_multitask_best.pth",
        p14_pca="../checkpoints/p14/p14_pca.pkl",
        roi6_ckpt="../checkpoints/roi_seg/roi_seg_6class_v1_best.pth",
        p11_ckpt="../checkpoints/p11/p11_color_best.pth",
        p11_norm="../checkpoints/p11/p11_norm.npz",
        p13_ckpt="../checkpoints/p13/p13_texture_best.pth",
        p13_norm="../checkpoints/p13/p13_norm.npz",
    )
    return ModelBundle(device=device, paths=paths).load_all()


def infer_one_image(
    bundle: ModelBundle,
    img_bgr: np.ndarray,
    sample_id: str,
    save_artifacts: bool = False,
) -> Dict[str, Any]:
    assert hasattr(bundle, "roi_infer"), "ModelBundle missing roi_infer"
    """
    单入口：一张图 → JSON（与你 demo_api_json 一致风格）
    """
    device = bundle.device

    # 1) tongue mask
    tongue_mask = _infer_tongue_mask_from_model(bundle.tongue_model, img_bgr, device)

    # 2) ROI masks（模型 + 规则兜底 Tai/Zhi）
    roi_masks = bundle.roi_infer.infer_roi_masks(img_bgr, tongue_mask)

    tai_rule, zhi_rule = infer_tai_zhi_from_tongue_mask(tongue_mask, tai_ratio=0.35)
    if int((roi_masks.get("tai", 0) > 0).sum()) == 0:
        roi_masks["tai"] = tai_rule
    if int((roi_masks.get("zhi", 0) > 0).sum()) == 0:
        roi_masks["zhi"] = zhi_rule

    # 3) 统一特征提取（复用你现有桥）
    #    注意：这里先保持最少改动，仍复用 infer_regression 的方式
    #    下一步工程化（你之前提到的）是把 P11/P13 改为直接吃内存 roi_crops，去掉磁盘依赖。
    from pathlib import Path
    roi_root = Path("outputs/roi_split_v1")    # 你也可以换成配置项
    tables = extract_roi_features_all(
        img_bgr=img_bgr,
        roi_masks=roi_masks,
        sample_id=sample_id,
        roi_root=roi_root,
        device=device,
        infer_regression_fn=infer_regression,
        p11_ckpt=bundle.paths.p11_ckpt,
        p11_norm=bundle.paths.p11_norm,
        p11_dim=None,
        p13_ckpt=bundle.paths.p13_ckpt,
        p13_norm=bundle.paths.p13_norm,
        p13_dim=None,
        p14_model=bundle.p14_model,
        pca=bundle.pca,
    )

    # 4) 组装成你 demo 的 JSON 结构（先做 tg 主干；ROI 扩展可后续加）
    #    这里假设 extract_roi_features_all 已经返回你期望的结构化表；
    #    如果它返回的是 P21/P23/P24 等表，你可以按需映射进 outputs。
    demo_outputs = assemble_demo_outputs_from_tables(
        tables,
        preferred_roi="Tai",
        img_bgr=img_bgr,
        roi_masks=roi_masks,
    )

    if not demo_outputs:
        print("[WARN] demo_outputs is empty. Available table keys are:")
        for k in sorted(list(tables.keys())):
            print("  -", k)

    result = {
        "meta": {
            "id": sample_id,
            "model_version": "tongue_expert_v1",
            "pipeline": ["p11_color", "p12_shape", "p13_texture", "p14_embedding"],
            "device": device,
        },
        "artifacts": {
            "generated": {}
        },
        "outputs": {
            "demo": demo_outputs,  # 对外展示层（color/texture/representation…）
            "roi_tables": tables,  # 论文级内部表（P21/P23/P24…）
        },
        "interpretation": {
            "disclaimer": "该结果用于科研/工程验证，不构成医疗诊断或建议。"
        }
    }

    # 5) 确保 JSON 可序列化
    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(x) for x in obj]
        return _to_py(obj)

    return _walk(result)
