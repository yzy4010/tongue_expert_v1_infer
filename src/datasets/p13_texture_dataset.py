import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class P13TextureDataset(Dataset):
    """
    ROI image -> 16-d texture targets:
    [tg_lbp1..5, tg_hog1..5, tg_contrast, tg_dissimilarity, tg_homogeneity, tg_energy, tg_correlation, tg_entropy]
    """
    def __init__(self, roi_dir: str, label_csv: str, split_txt: str, img_size: int = 224):
        self.roi_dir = roi_dir
        self.img_size = img_size

        df = pd.read_csv(label_csv)
        df["SID"] = df["SID"].astype(str)
        self.label_map = {r["SID"]: r for _, r in df.iterrows()}

        with open(split_txt, "r", encoding="utf-8") as f:
            ids = [l.strip() for l in f if l.strip()]

        self.features = [
            "tg_lbp1","tg_lbp2","tg_lbp3","tg_lbp4","tg_lbp5",
            "tg_hog1","tg_hog2","tg_hog3","tg_hog4","tg_hog5",
            "tg_contrast","tg_dissimilarity","tg_homogeneity","tg_energy","tg_correlation","tg_entropy"
        ]

        items = []
        for sid in ids:
            if sid not in self.label_map:
                continue
            img_path = os.path.join(self.roi_dir, sid + ".jpg")
            if not os.path.exists(img_path):
                continue
            items.append(sid)

        self.ids = items
        self.target_dim = len(self.features)

        print(f"[P13TextureDataset] {roi_dir} -> {len(self.ids)} samples, target_dim={self.target_dim}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img_path = os.path.join(self.roi_dir, sid + ".jpg")

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)  # CHW

        row = self.label_map[sid]
        y = np.array([float(row[c]) for c in self.features], dtype=np.float32)
        y = torch.from_numpy(y)

        return x, y, sid
