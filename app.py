import os, sys, re, json
from pathlib import Path
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=False)  # CPU

SERNUM_RE = re.compile(r'(?<!\d)(\d{2})\s+(\d{2})\s+(\d{6})(?!\d)')

# коды подразделений, даты и т.п. исключаем
NOISE_PATTERNS = [
    re.compile(r'\b\d{3}[-–]\d{3}\b'),  # 230-003
    re.compile(r'\b\d{2}\.\d{2}\.\d{4}\b'),  # дата
]

def normalize_orientation(img):
    h,w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def crop_vertical_strips(img):
    """Возвращает узкие полосы из характерных мест:
    very-left, near-left, near-right, very-right и два у корешка."""
    H,W = img.shape[:2]
    strips = []
    def add(x0, x1, name):
        x0 = max(0, int(x0)); x1 = min(W, int(x1))
        if x1 > x0:
            strips.append((name, img[:, x0:x1].copy()))
    # ширина полосы 15% ширины страницы
    s = 0.15*W
    # внешние края
    add(0, s, "very_left")
    add(W-s, W, "very_right")
    # полосы у корешка (1/2 ширины)
    mid = W/2
    add(mid-1.2*s, mid-0.2*s, "gutter_left")
    add(mid+0.2*s, mid+1.2*s, "gutter_right")
    # дополнительные около краёв
    add(0.12*W, 0.24*W, "left2")
    add(W-0.24*W, W-0.12*W, "right2")
    return strips

def preprocess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    bl = cv2.bilateralFilter(eq, 9, 75, 75)
    th = cv2.adaptiveThreshold(bl,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,11,2)
    th_inv = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    close = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 1)
    close_inv = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, kernel, 1)
    # возвращаем BGR, чтобы easyocr не ругался
    return [
        cv2.cvtColor(close, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(close_inv, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    ]

def rotate90(img, direction):
    if direction == 'ccw':
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def clean_text(s):
    s = s.replace('O','0').replace('o','0').replace('I','1').replace('l','1').replace('S','5')
    s = re.sub(r'[^0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # вырежем очевидный «мусор»
    for pat in NOISE_PATTERNS:
        s = pat.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def find_sernum(text):
    # строгий шаблон 45 19 123456
    m = SERNUM_RE.search(text)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"
    # fallback: собрать из отдельных групп подряд
    parts = [p for p in re.findall(r'\d+', text)]
    for i in range(len(parts)-2):
        if len(parts[i])==2 and len(parts[i+1])==2 and len(parts[i+2])==6:
            return f"{parts[i]} {parts[i+1]} {parts[i+2]}"
    return None

def ocr(img):
    # читаем только цифры и пробелы
    lines = reader.readtext(img, detail=0, allowlist='0123456789 ')
    return clean_text(' '.join(lines))

def infer_one(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return {"path": str(path), "series": None, "number": None, "error": "cannot_read"}
    img = normalize_orientation(img)

    best = None
    # Перебираем вертикальные полосы, поворачиваем, пробуем разные препроцессы
    for name, strip in crop_vertical_strips(img):
        for proc in preprocess(strip):
            for rot in ('ccw','cw'):
                patch = rotate90(proc, rot)
                txt = ocr(patch)
                guess = find_sernum(txt)
                if guess:
                    # чем длиннее чистый текст вокруг, тем лучше — простое «score»
                    score = len(guess)
                    # можно усилить баллы, если совпало строго по regex
                    if SERNUM_RE.fullmatch(guess):
                        score += 3
                    if not best or score > best[0]:
                        best = (score, guess, name)

    out = {"path": str(path)}
    if best:
        a,b,c = best[1].split()
        out.update({"series": f"{a} {b}", "number": c, "full": best[1], "where": best[2]})
    else:
        out.update({"series": None, "number": None})
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py /data_or_image_path")
        sys.exit(0)
    p = Path(sys.argv[1])
    imgs = [p] if p.is_file() else [x for x in p.iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]
    res = [infer_one(x) for x in imgs]
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
