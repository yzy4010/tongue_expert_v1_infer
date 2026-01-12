# deploy/startup.py
from src.pipeline.build_bundle import build_model_bundle

# 全局 bundle（进程级单例）
BUNDLE = None

def load_bundle(device: str = "cpu"):
    global BUNDLE
    if BUNDLE is None:
        BUNDLE = build_model_bundle(device=device)
    return BUNDLE
