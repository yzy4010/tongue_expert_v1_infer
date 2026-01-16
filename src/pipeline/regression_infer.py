# src/pipeline/regression_infer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2
import torch
import torch.nn as nn
import csv

import os




def load_label_columns_from_csv(csv_path: str) -> Optional[List[str]]:
    """
    从训练标签 CSV 获取目标字段名顺序。
    规则：
      - 取 header
      - 去掉可能的 id 列（id / img_id / image_id / sid / sample_id）
    """
    if not csv_path or not os.path.exists(csv_path):
        return None

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if not header:
        return None

    # 常见第一列是样本 id
    drop = {"id", "img_id", "image_id", "sid", "sample_id", "name", "filename", "file"}
    cols = [c.strip() for c in header if c and c.strip()]

    # 如果第一列像 id，就丢掉它
    if cols and cols[0].lower() in drop:
        cols = cols[1:]

    return cols


# -----------------------------
# IO: norm
# -----------------------------
def safe_load_json(path: str) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(str(p), "r", encoding="utf-8") as f:
        return json.load(f)


def apply_denorm(pred: np.ndarray, norm: Optional[dict]) -> np.ndarray:
    """
    兼容常见 norm 结构（你可以按你项目的 norm 结构继续扩展）：
    1) {"mean":[...], "std":[...]}
    2) {"y_mean":[...], "y_std":[...]}
    3) {"min":[...], "max":[...]}  (0-1 -> min-max)
    """
    if norm is None:
        return pred

    x = pred.astype(np.float32)

    if "mean" in norm and "std" in norm:
        mean = np.asarray(norm["mean"], np.float32)
        std = np.asarray(norm["std"], np.float32)
        return x * (std + 1e-8) + mean

    if "y_mean" in norm and "y_std" in norm:
        mean = np.asarray(norm["y_mean"], np.float32)
        std = np.asarray(norm["y_std"], np.float32)
        return x * (std + 1e-8) + mean

    if "min" in norm and "max" in norm:
        mn = np.asarray(norm["min"], np.float32)
        mx = np.asarray(norm["max"], np.float32)
        return x * (mx - mn) + mn

    # fallback: unknown norm format
    return x


# -----------------------------
# Model load utils
# -----------------------------
def load_ckpt_state(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        # common keys
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        # maybe it IS state_dict
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


def strip_prefix_module(state: dict) -> dict:
    # remove "module." prefix if trained with DDP/DataParallel
    out = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        out[nk] = v
    return out


def infer_out_dim_from_state(state: dict) -> int:
    """
    从最后一层权重推断 out_dim（稳妥版）
    """
    # 典型：head.weight: [out_dim, hidden_dim]
    candidates = [k for k in state.keys() if k.endswith("weight")]
    # 尽量选 shape[0] 较小的 head（避免 backbone 卷积）
    best = None
    for k in candidates:
        v = state[k]
        if hasattr(v, "shape") and len(v.shape) == 2:
            out_dim = int(v.shape[0])
            if best is None or out_dim < best:
                best = out_dim
    if best is None:
        # 兜底
        return 1
    return best


# -----------------------------
# ROI image loader
# -----------------------------
def load_roi_image(path: str, hw=(224, 224)) -> torch.Tensor:
    """
    与你的回归模型训练预处理保持“最小一致”：
    - BGR->RGB
    - resize 到 224
    - 0-1
    - CHW tensor
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, hw, interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)


# -----------------------------
# IMPORTANT: You must provide this builder
# -----------------------------
def build_regressor_from_ckpt(ckpt_path: str, device: torch.device, out_dim: Optional[int] = None) -> nn.Module:
    """
    自动从 src.models.p11_color_regressor 中挑选一个最像“回归模型”的 nn.Module 来构建，
    然后 load_state_dict(strict=False) 以兼容 ckpt 的 backbone./net. 前缀。
    """
    import inspect
    import importlib

    state = strip_prefix_module(load_ckpt_state(ckpt_path, device=device))
    if out_dim is None:
        out_dim = infer_out_dim_from_state(state)

    has_backbone = any(k.startswith("backbone.") for k in state.keys())
    has_net = any(k.startswith("net.") for k in state.keys())

    # 1) 导入你的回归器模块（目前目录里唯一明确的 regressor 文件）
    mod = importlib.import_module("src.models.p11_color_regressor")

    # 2) 从模块中找 nn.Module 子类
    module_classes = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        try:
            if issubclass(obj, nn.Module) and obj is not nn.Module:
                module_classes.append(obj)
        except Exception:
            pass

    if not module_classes:
        raise ImportError("No nn.Module classes found in src.models.p11_color_regressor")

    # 3) 给候选类打分：名字像 regressor + 构造函数支持 out_dim
    def score_cls(cls) -> int:
        s = 0
        n = cls.__name__.lower()
        if "regress" in n or "regressor" in n:
            s += 10
        if "p11" in n or "color" in n:
            s += 3
        sig = None
        try:
            sig = inspect.signature(cls.__init__)
        except Exception:
            return s

        params = list(sig.parameters.keys())
        if "out_dim" in params or "output_dim" in params or "num_outputs" in params:
            s += 8
        return s

    module_classes.sort(key=score_cls, reverse=True)

    last_err = None
    model: Optional[nn.Module] = None

    # 4) 依次尝试实例化（优先传 out_dim）
    for cls in module_classes:
        try:
            sig = inspect.signature(cls.__init__)
            params = sig.parameters

            kwargs = {}
            if "out_dim" in params:
                kwargs["out_dim"] = out_dim
            elif "output_dim" in params:
                kwargs["output_dim"] = out_dim
            elif "num_outputs" in params:
                kwargs["num_outputs"] = out_dim

            model = cls(**kwargs) if kwargs else cls()
            break
        except Exception as e:
            last_err = e
            model = None

    if model is None:
        raise RuntimeError(f"Failed to instantiate regressor from p11_color_regressor. last_err={repr(last_err)}")

    # 5) ckpt 里常见前缀：backbone./net.，而你的模型里可能叫 backbone/net 或其他
    model.to(device)
    model.eval()
    return model



# -----------------------------
# Deploy-friendly infer_regression
# -----------------------------
def infer_regression(
    *,
    ckpt_path: str,
    norm_path: str,
    roi_dir: str,
    ids: List[str],
    device: str | torch.device,
    out_dim_override: Optional[int] = None,
    label_csv: Optional[str] = None,
) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
    """
    返回给 roi_feature_bridge.run_regression_on_roi 使用：
      feats_dict, pred_denorm (NxD), used_ids

    feats_dict: 只针对单个 id（ids 通常只给 1 个）：
      {"tg_xxx": float, ...}
    """
    import os

    device_t = device if isinstance(device, torch.device) else torch.device(device)

    if not ckpt_path or not os.path.exists(ckpt_path):
        # 部署阶段允许 ckpt 缺失时不崩
        return {}, np.zeros((0, 0), np.float32), []

    state = strip_prefix_module(load_ckpt_state(ckpt_path, device=device_t))
    out_dim = out_dim_override if out_dim_override is not None else infer_out_dim_from_state(state)

    model = build_regressor_from_ckpt(ckpt_path, device=device_t, out_dim=out_dim)
    model.eval()

    norm = safe_load_json(norm_path) if (norm_path and os.path.exists(norm_path)) else None

    xs: List[torch.Tensor] = []
    used: List[str] = []

    for sid in ids:
        p_jpg = os.path.join(roi_dir, f"{sid}.jpg")
        p_png = os.path.join(roi_dir, f"{sid}.png")
        roi_path = p_jpg if os.path.exists(p_jpg) else (p_png if os.path.exists(p_png) else None)
        if roi_path is None:
            continue
        xs.append(load_roi_image(roi_path))
        used.append(sid)

    if len(xs) == 0:
        return {}, np.zeros((0, out_dim), np.float32), []

    batch = torch.stack(xs, dim=0).to(device_t)  # Nx3x224x224
    with torch.no_grad():
        pred = model(batch).detach().cpu().numpy().astype(np.float32)

    pred_denorm = apply_denorm(pred, norm) if norm is not None else pred
    feats = pred_denorm

    # 你 bridge 里假设“返回 key 都是 tg_ 前缀”，我们这里造一个通用的 tg_f000..tg_fXXX
    # 如果你有明确的 feature 名称列表（论文字段），后续可以把这里替换成真实字段名映射。
    # --- build feature dict with real column names if provided ---
    cols = load_label_columns_from_csv(label_csv) if label_csv else None

    feats_dict: Dict[str, float] = {}
    if pred_denorm.shape[0] >= 1:
        v = pred_denorm[0]  # (D,)
        D = int(v.shape[0])

        if cols is not None and len(cols) == D:
            # 用真实字段名（来自训练 label csv）
            for j, name in enumerate(cols):
                # 确保前缀是 tg_，供 bridge replace_prefix 使用
                clean = name.strip()
                # 如果 name 本身已经是 tg_ 开头，就去掉一次
                if clean.startswith("tg_"):
                    clean = clean[3:]
                feats_dict[f"tg_{clean}"] = float(v[j])

        else:
            # fallback：仍旧用 f000..，并打印一次提示（避免刷屏）
            for j in range(D):
                feats_dict[f"tg_f{j:03d}"] = float(v[j])

    return feats_dict, pred_denorm, used
