# deploy/app.py
from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from deploy.startup import load_bundle
from src.pipeline.run_infer import infer_one_image

app = FastAPI(title="TongueExpert API", version="v1")

@app.on_event("startup")
def _startup():
    load_bundle(device="cpu")  # 或 "cuda"

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # 1. 读取图片
    data = await file.read()
    img_np = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "Invalid image"}

    # 2. 推理
    bundle = load_bundle()
    result = infer_one_image(
        bundle,
        img_bgr,
        sample_id="API_UPLOAD"
    )

    # 3. 对外只返回 demo（推荐）
    return result["outputs"]["demo"]
