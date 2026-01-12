import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

#
# Dataset：ROI 图像 + P11 数值向量
#

class P11ColorDataset(Dataset):
    def __init__(self, roi_dir: str, split_file: str, label_csv: str, normalize_y: bool = True):
        self.roi_dir = roi_dir

        # ids
        with open(split_file, "r", encoding="utf-8") as f:
            ids = [l.strip() for l in f if l.strip()]

        # labels
        df = pd.read_csv(label_csv)
        df["SID"] = df["SID"].astype(str)
        self.label_cols = [c for c in df.columns if c != "SID"]

        # map: SID -> y vector
        label_map = {}
        for _, row in df.iterrows():
            sid = str(row["SID"])
            y = row[self.label_cols].values.astype(np.float32)
            if np.any(pd.isna(y)):
                continue
            label_map[sid] = y

        # filter valid
        self.ids = []
        for sid in ids:
            img_path = os.path.join(self.roi_dir, sid + ".jpg")
            if os.path.exists(img_path) and sid in label_map:
                self.ids.append(sid)

        self.label_map = label_map
        print(f"[P11ColorDataset] {roi_dir} -> {len(self.ids)} samples, target_dim={len(self.label_cols)}")

        # y normalization stats (optional)
        self.normalize_y = normalize_y
        if self.normalize_y:
            Y = np.stack([self.label_map[sid] for sid in self.ids], axis=0)
            self.y_mean = Y.mean(axis=0).astype(np.float32)
            self.y_std = (Y.std(axis=0) + 1e-8).astype(np.float32)
        else:
            self.y_mean = None
            self.y_std = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img_path = os.path.join(self.roi_dir, sid + ".jpg")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read ROI: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)  # [3,H,W]

        y = self.label_map[sid].copy()
        if self.normalize_y:
            y = (y - self.y_mean) / self.y_std
        y = torch.from_numpy(y)

        return x, y, sid
