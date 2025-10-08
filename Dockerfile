FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchvision==0.18.1
RUN pip install -r requirements.txt

COPY app.py .

ENTRYPOINT ["python", "app.py"]
