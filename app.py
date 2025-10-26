# app.py
import os, sys, json, re
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
import easyocr

ALLOW_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SERNUM_RE = re.compile(r'(?<!\d)(\d{2})\s+(\d{2})\s+(\d{6})(?!\d)')

# ---------- helpers ----------
def clean_text(s: str) -> str:
    s = s.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').replace('S', '5')
    s = re.sub(r'[^0-9 ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def find_sernum(text: str) -> str | None:
    m = SERNUM_RE.search(text)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"
    parts = re.findall(r'\d+', text)
    for i in range(len(parts) - 2):
        if len(parts[i]) == 2 and len(parts[i+1]) == 2 and len(parts[i+2]) == 6:
            return f"{parts[i]} {parts[i+1]} {parts[i+2]}"
    return None

def rotate(img, ang):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess_variants(roi_bgr):
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE даёт контраст на тиснении/мелких цифрах
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    # лёгкий шумодав
    g = cv2.GaussianBlur(g, (3,3), 0)

    # набор бинаризаций
    th_ad = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, th_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    neg_ad = cv2.bitwise_not(th_ad)
    neg_otsu = cv2.bitwise_not(th_otsu)

    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    m1 = cv2.morphologyEx(th_ad, cv2.MORPH_CLOSE, k, 1)
    m2 = cv2.morphologyEx(neg_ad, cv2.MORPH_CLOSE, k, 1)

    return [
        roi_bgr,
        cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(th_otsu, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(neg_otsu, cv2.COLOR_GRAY2BGR),
    ]


def pick_best_box(boxes):
    if not boxes:
        return None
    scored = []
    for x1, y1, x2, y2, conf in boxes:
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        aspect = h / w  # вертикальный бокс ценим выше
        scored.append((conf * (1.0 + min(3.0, aspect / 3.0)), (x1, y1, x2, y2, conf)))
    scored.sort(reverse=True, key=lambda z: z[0])
    return scored[0][1]

def ocr_patch(reader, patch):
    lines = reader.readtext(patch, detail=0, allowlist='0123456789 ')
    return clean_text(' '.join(lines))

def infer_one(model, reader, img_path: Path, pad=10, conf_th=0.25, imgsz=896, save_crop_dir: Path | None=None):
    img = cv2.imread(str(img_path))
    if img is None:
        return {"path": str(img_path), "error": "cannot_read"}
    H, W = img.shape[:2]

    # --- детекция ---
    res = model.predict(source=img, conf=conf_th, iou=0.5, imgsz=imgsz, verbose=False)[0]
    boxes = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf.cpu().numpy().item() if hasattr(b.conf.cpu().numpy(), "item") else b.conf.cpu().numpy())
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))

    best = pick_best_box(boxes)
    out = {"path": str(img_path)}
    if not best:
        out.update({"series": None, "number": None, "note": "no_roi_detected"})
        return out

    x1, y1, x2, y2, det_conf = best
    bw, bh = x2 - x1, y2 - y1

    # --- динамический паддинг и расширение кропа ---
    # вертикальная колонка: расширим по X сильнее
    padx = max(pad, int(0.35 * bw))
    pady = max(pad, int(0.10 * bh))
    x1 = max(0, x1 - padx); y1 = max(0, y1 - pady)
    x2 = min(W, x2 + padx); y2 = min(H, y2 + pady)
    roi = img[y1:y2, x1:x2].copy()

    # --- апскейл для OCR ---
    # целимся в высоту 700–900px, чтобы символы были крупные
    h, w = roi.shape[:2]
    target_h = 800
    scale = max(1.0, target_h / float(h))
    if scale > 1.0:
        roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    if save_crop_dir:
        save_crop_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_crop_dir / (img_path.stem + "_crop.jpg")), roi)

    best_guess, best_score, best_rot, best_raw = None, -1, 0, ""

    # --- предобработка + больше углов ---
    for proc in preprocess_variants(roi):
        for ang in (-95, -90, -85, 0, 85, 90, 95, 180):  # чуточку шире вокруг 90°
            patch = rotate(proc, ang) if ang != 0 else proc
            txt = ocr_patch(reader, patch)
            if not txt:
                # fallback: без allowlist, потом фильтр
                cand = reader.readtext(patch, detail=0)
                txt = clean_text(" ".join(cand))
            guess = find_sernum(txt)
            if guess:
                score = len(guess) + (3 if SERNUM_RE.fullmatch(guess) else 0)
                if score > best_score:
                    best_score, best_guess, best_rot, best_raw = score, guess, ang, txt

    out.update({"bbox":[x1,y1,x2,y2], "det_conf":det_conf})
    if best_guess:
        a,b,c = best_guess.split()
        out.update({"series": f"{a} {b}", "number": c, "full": best_guess, "rot": best_rot, "ocr_raw": best_raw})
    else:
        out.update({"series": None, "number": None, "ocr_raw": ""})
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: python app.py /path/to/best.pt /path/to/image_or_folder [--save-crops DIR] [--ocr-models DIR]")
        sys.exit(0)
    model_path = Path(sys.argv[1])
    src = Path(sys.argv[2])

    save_crops = None
    models_dir = Path("/app/models/easyocr")  # путь по умолчанию внутри контейнера
    if "--save-crops" in sys.argv:
        i = sys.argv.index("--save-crops")
        if i + 1 < len(sys.argv):
            save_crops = Path(sys.argv[i + 1])
    if "--ocr-models" in sys.argv:
        i = sys.argv.index("--ocr-models")
        if i + 1 < len(sys.argv):
            models_dir = Path(sys.argv[i + 1])

    model = YOLO(str(model_path))
    # ВАЖНО: download_enabled=False и указываем локальную директорию с весами
    reader = easyocr.Reader(
        ['en'],
        gpu=False,
        download_enabled=False,
        model_storage_directory=str(models_dir)
    )

    paths = sorted([p for p in src.iterdir() if p.suffix.lower() in ALLOW_EXT]) if src.is_dir() else [src]
    results = [infer_one(model, reader, p, save_crop_dir=save_crops) for p in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
