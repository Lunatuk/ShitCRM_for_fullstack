FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python - <<'PY'
import easyocr, os
os.makedirs("/app/models/easyocr", exist_ok=True)
_ = easyocr.Reader(['en'], gpu=False, download_enabled=True,
                   model_storage_directory="/app/models/easyocr")
print("EasyOCR models ready.")
PY

COPY infer.py app.py ./
COPY static ./static

COPY best.pt /app/best.pt


ENV MODEL_PATH=/app/best.pt
ENV OCR_MODELS_DIR=/app/models/easyocr

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
