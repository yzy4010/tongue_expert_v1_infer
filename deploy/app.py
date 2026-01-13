# deploy/app.py
from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from deploy.startup import load_bundle
from src.pipeline.run_infer import infer_one_image
from fastapi.staticfiles import StaticFiles
from pathlib import Path


app = FastAPI(title="TongueExpert API", version="v1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(OUTPUTS_DIR)), name="static")



@app.on_event("startup")
def _startup():
    load_bundle(device="cpu")  # 或 "cuda"

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    img_np = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "Invalid image"}

    bundle = load_bundle()
    result = infer_one_image(
        image=img_bgr,
        bundle=bundle,
        sample_id="API_UPLOAD",
        device="cpu",
        save_outputs=True,
    )
    return result
