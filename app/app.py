# app.py
import io, os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2, numpy as np
from infer import PassportOCR

MODEL_PATH = os.getenv("MODEL_PATH", "/app/best.pt")
OCR_MODELS = os.getenv("OCR_MODELS_DIR", "/app/models/easyocr")

app = FastAPI(title="Passport Series/Number OCR")

# 1) Сначала объявляем API-роут
ocr = PassportOCR(MODEL_PATH, OCR_MODELS)

# app.py (только endpoint)
@app.post("/api/ocr")
async def api_ocr(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error":"cannot_read_image"}, status_code=400)
    crops_dir = Path("/app/crops"); crops_dir.mkdir(exist_ok=True)
    crop_path = str(crops_dir / (Path(file.filename).stem + "_crop.jpg"))
    res = ocr.infer_image(img, save_crop_path=crop_path)
    res["filename"] = file.filename
    return res


# 2) Отдаём главную страницу
@app.get("/")
def index():
    return FileResponse("static/index.html")

# 3) Монтируем статику на /static (а НЕ на /)
app.mount("/static", StaticFiles(directory="static"), name="static")
