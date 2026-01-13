# -*- coding: utf-8 -*-
"""
Infer P14 embedding for a split and export to CSV.

Reads:
  - checkpoints/p14/p14_multitask_best.pth
  - outputs/roi/{split}/*.jpg
  - data/splits/{split}.txt

Writes:
  outputs/p14_embedding/p14_emb_{split}.csv
"""

import os
import argparse
import json
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_ids(txt_path: str) -> List[str]:
    ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def build_eval_tf(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class P14MultiTaskNet(nn.Module):
    def __init__(self, emb_dim: int = 128, p11_dim: int = 76, p13_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # [B,512,1,1]
        self.feat_dim = 512

        def mlp_head(out_dim: int):
            layers = [
                nn.Flatten(1),
                nn.Linear(self.feat_dim, 256),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(256, out_dim))
            return nn.Sequential(*layers)

        self.head_emb = mlp_head(emb_dim)
        self.head_p11 = mlp_head(p11_dim)
        self.head_p13 = mlp_head(p13_dim)

    def forward(self, x):
        f = self.backbone(x)
        emb = self.head_emb(f)
        p11 = self.head_p11(f)
        p13 = self.head_p13(f)
        return emb, p11, p13

    def forward_embedding(self, x):
        """
        Return embedding only (P14).
        Safe wrapper for inference / ROI usage.
        """
        emb, _, _ = self.forward(x)
        return emb


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt", type=str, default="../checkpoints/p14/p14_multitask_best.pth")

    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--splits_dir", type=str, default="../data/splits")
    parser.add_argument("--roi_dir", type=str, default="../outputs/roi")
    parser.add_argument("--out_dir", type=str, default="../outputs/p14_embedding")

    args = parser.parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    emb_dim = int(cfg.get("emb_dim", 128))
    p11_dim = int(cfg.get("p11_dim", 76))
    p13_dim = int(cfg.get("p13_dim", 16))

    model = P14MultiTaskNet(emb_dim=emb_dim, p11_dim=p11_dim, p13_dim=p13_dim).to(device).eval()
    model.load_state_dict(ckpt["state_dict"], strict=True)

    ids = read_ids(os.path.join(args.splits_dir, f"{args.split}.txt"))
    roi_split_dir = os.path.join(args.roi_dir, args.split)

    tfm = build_eval_tf(224)

    rows = []
    for sid in ids:
        jp = os.path.join(roi_split_dir, f"{sid}.jpg")
        pp = os.path.join(roi_split_dir, f"{sid}.png")
        if os.path.exists(jp):
            path = jp
        elif os.path.exists(pp):
            path = pp
        else:
            continue

        img = Image.open(path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        emb, _, _ = model(x)
        emb = emb.detach().cpu().numpy().reshape(-1).astype(np.float32)

        row = {"id": sid}
        for i in range(emb_dim):
            row[f"p14_emb_{i:03d}"] = float(emb[i])
        rows.append(row)

    ensure_dir(args.out_dir)
    out_csv = os.path.join(args.out_dir, f"p14_emb_{args.split}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)
    print("Num embeddings:", len(rows))


if __name__ == "__main__":
    main()
