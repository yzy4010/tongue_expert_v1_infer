import os
import cv2
import torch
from torch.utils.data import Dataset


class TongueSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ids, img_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.img_size = img_size

        self.valid_ids = []
        for sid in self.ids:
            img_path = os.path.join(img_dir, f"{sid}.jpg")
            mask_path = os.path.join(mask_dir, f"{sid}.png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.valid_ids.append(sid)
            else:
                print(f"[Skip] Missing image or mask: {sid}")

        print(f"Loaded {len(self.valid_ids)} valid samples")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        sid = self.valid_ids[idx]

        img_path = os.path.join(self.img_dir, f"{sid}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{sid}.png")

        # ---- load ----
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Failed to load {sid}")

        # ---- BGR → RGB ----
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---- resize (关键修复点) ----
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # ---- normalize ----
        image = image.astype("float32") / 255.0
        mask = (mask > 0).astype("float32")

        # ---- to tensor ----
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)        # [1, H, W]

        return image, mask
