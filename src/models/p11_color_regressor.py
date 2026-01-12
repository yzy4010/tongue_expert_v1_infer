import torch.nn as nn
from torchvision.models import resnet18

# 模型：轻量 CNN 回归（ResNet18）

class P11ColorRegressor(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        m = resnet18(weights=None)  # 先不依赖预训练，保证可控
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # [B,512,1,1]
        self.head = nn.Linear(512, out_dim)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat)
