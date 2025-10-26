# infer.py
# CPU-friendly OCR пайплайн: YOLO -> (ROI объединение/расширение) -> EasyOCR -> (опционально) PaddleOCR
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# --- опционально: PaddleOCR как надёжный fallback (умный angle-classifier) ---
try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except Exception:
    HAS_PADDLE = False


# -------------------------- парсинг "AA BB CCCCCC" ---------------------------

SERNUM_RE = re.compile(r'(?<!\d)(\d{2})\s+(\d{2})\s+(\d{6})(?!\d)')  # 2-2-6
TEN_RE    = re.compile(r'(?<!\d)(\d{10})(?!\d)')                      # 10 слитных цифр

def clean_text(s: str) -> str:
    """нормализуем OCR-шум: похожие символы -> цифры, чистим всё лишнее"""
    map_like = {'O':'0','o':'0','I':'1','l':'1','S':'5','Z':'2','B':'8','G':'6','q':'9'}
    s = ''.join(map_like.get(ch, ch) for ch in s)
    s = re.sub(r'[^0-9 ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def find_sernum(text: str) -> Optional[str]:
    """ищем серию и номер в очищенном тексте"""
    m = SERNUM_RE.search(text)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"
    t10 = TEN_RE.search(text)
    if t10:
        s = t10.group(1)
        return f"{s[0:2]} {s[2:4]} {s[4:10]}"
    parts = re.findall(r'\d+', text)
    for i in range(len(parts) - 2):
        if len(parts[i]) == 2 and len(parts[i+1]) == 2 and len(parts[i+2]) == 6:
            return f"{parts[i]} {parts[i+1]} {parts[i+2]}"
    return None


# -------------------- утилиты предобработки и геометрии ----------------------

def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """поворот с сохранением размера"""
    (h, w) = img.shape[:2]
    c = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def emphasize_red(bgr: np.ndarray) -> List[np.ndarray]:
    """выделяем красные цифры (HSV), отдаём парочку бинарных вариантов"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 90, 60), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 1)
    red = cv2.bitwise_and(bgr, bgr, mask=mask)
    g = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)]

def preprocess_variants(roi_bgr: np.ndarray) -> List[np.ndarray]:
    """
    минимум вариантов = максимум скорости:
      - оригинал
      - Otsu и инверсия
      - «красный» бинарный (для вертикальных красных цифр)
    """
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # лёгкий sharpening
    g = cv2.filter2D(g, -1, k)

    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    neg = cv2.bitwise_not(otsu)

    variants = [
        roi_bgr,
        cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(neg,  cv2.COLOR_GRAY2BGR),
    ]
    variants += emphasize_red(roi_bgr)  # 1 вариант, сфокусированный на красном
    return variants

def _top_k_boxes(res, k: int = 3) -> List[Tuple[int, int, int, int, float]]:
    """берём топ-k боксов по уверенности"""
    boxes = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            c = b.conf.cpu().numpy()
            conf = float(c.item()) if hasattr(c, "item") else float(c)
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
    boxes.sort(key=lambda t: t[4], reverse=True)
    return boxes[:k]

def _union_and_expand(boxes: List[Tuple[int,int,int,int,float]],
                      W: int, H: int,
                      scale_x: float = 3.0, scale_y: float = 1.7) -> Tuple[int,int,int,int]:
    """
    Общий прямоугольник + расширение.
    Колонка узкая и высокая -> сильно расширяем ширину (scale_x), умеренно высоту (scale_y),
    чтобы захватить обе строки серии/номера.
    """
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    new_w = int(bw * scale_x)
    new_h = int(bh * scale_y)
    nx1 = max(0, cx - new_w // 2)
    ny1 = max(0, cy - new_h // 2)
    nx2 = min(W, cx + new_w // 2)
    ny2 = min(H, cy + new_h // 2)
    return nx1, ny1, nx2, ny2

def _angles_for(roi: np.ndarray) -> List[int]:
    """минимальный набор углов. если ROI «высокий», приоритет горизонтализации"""
    h, w = roi.shape[:2]
    if h / max(1, w) > 1.7:
        return [90, -90, 0, 180]
    return [0, 180, 90, -90]

def _ocr_try(reader: easyocr.Reader, img: np.ndarray) -> str:
    """
    1) строгий allowlist цифр — быстрый;
    2) если пусто — свободный режим (на случай странной бинаризации).
    """
    try1 = reader.readtext(img, detail=0, paragraph=True, allowlist='0123456789 ')
    txt1 = clean_text(' '.join(try1))
    if txt1:
        return txt1
    try2 = reader.readtext(img, detail=0, paragraph=True)
    return clean_text(' '.join(try2))


# ------------------------------- Paddle fallback -----------------------------

class PaddleWrap:
    """обёртка если PaddleOCR доступен (CPU, с angle-classifier)"""
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def read(self, img: np.ndarray) -> str:
        out = self.ocr.ocr(img, cls=True)
        lines = []
        for page in out:
            for (_box, (txt, _conf)) in page:
                lines.append(txt)
        return clean_text(' '.join(lines))


# --------------------------------- основной класс ----------------------------

class PassportOCR:
    """
    Использование:
        ocr = PassportOCR("runs/detect/train/weights/best.pt", "/app/models/easyocr")
        res = ocr.infer_image(cv2.imread("photo.jpg"), save_crop_path="crop.jpg")
    Возвращает dict: bbox, det_conf, series, number, full, ocr_raw, debug_texts
    """
    def __init__(self, yolo_path: str, ocr_models_dir: str = "/app/models/easyocr"):
        self.model = YOLO(yolo_path)
        self.reader = easyocr.Reader(['en'], gpu=False, download_enabled=False,
                                     model_storage_directory=ocr_models_dir)
        self.paddle = PaddleWrap() if HAS_PADDLE else None

    def infer_image(self, img_bgr: np.ndarray, save_crop_path: Optional[str] = None) -> Dict[str, Any]:
        H, W = img_bgr.shape[:2]
        yolo_res = self.model.predict(source=img_bgr, conf=0.25, iou=0.5, imgsz=896, verbose=False)[0]
        boxes = _top_k_boxes(yolo_res, k=3)

        out: Dict[str, Any] = {
            "bbox": None, "det_conf": None,
            "series": None, "number": None, "full": None,
            "ocr_raw": None, "debug_texts": []
        }
        if not boxes:
            return out

        # объединённый и расширенный ROI
        x1, y1, x2, y2 = _union_and_expand(boxes, W, H, scale_x=3.0, scale_y=1.7)
        roi = img_bgr[y1:y2, x1:x2].copy()

        # апскейл (без фанатизма)
        h, w = roi.shape[:2]
        scale = min(1.8, max(1.0, 1100 / float(h)))
        if scale > 1.0:
            roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        if save_crop_path:
            Path(save_crop_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_crop_path, roi)

        # --- быстрый перебор: варианты предобработки × минимальный набор углов ---
        for base in [roi]:
            for v in preprocess_variants(base):
                for ang in _angles_for(v):
                    p = rotate(v, ang) if ang else v
                    txt = _ocr_try(self.reader, p)
                    if txt:
                        out["debug_texts"].append(txt)
                        guess = find_sernum(txt)
                        if guess:
                            a, b, c = guess.split()
                            out.update({
                                "bbox": [x1, y1, x2, y2],
                                "det_conf": boxes[0][4],
                                "series": f"{a} {b}",
                                "number": c,
                                "full": guess,
                                "ocr_raw": txt
                            })
                            return out  # ранний выход — нашли

        # --- fallback на PaddleOCR (если установлен) ---
        if self.paddle is not None:
            txt = self.paddle.read(roi)
            if txt:
                out["debug_texts"].append(f"[paddle] {txt}")
                guess = find_sernum(txt)
                if guess:
                    a, b, c = guess.split()
                    out.update({
                        "bbox": [x1, y1, x2, y2],
                        "det_conf": boxes[0][4],
                        "series": f"{a} {b}",
                        "number": c,
                        "full": guess,
                        "ocr_raw": txt
                    })
                    return out

        # ничего не нашли — вернём debug и bbox
        out.update({"bbox": [x1, y1, x2, y2], "det_conf": boxes[0][4]})
        return out
