"""Read a steel coil tag photo: decode every QR on it and OCR the printed fields.

Why both: on real AM/NS India tags the QR codes hold only the batch ID (and,
on BIS-certified coils, a QCI "Made in India" URL). Manufacturer, size,
weights, grade, heat number and spec exist only as printed text, so OCR is
required. The QR batch ID is treated as ground truth and used to check the OCR.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

# Printed label captions -> canonical field name. Keys are normalised
# (lowercase, alphanumerics only) because OCR drops spaces and punctuation.
LABELS = {
    "batchid": "batch_id", "batch": "batch_id", "batchno": "batch_id", "coilno": "batch_id",
    "heatno": "heat_no", "heat": "heat_no",
    "size": "size", "sizetmmxwmmxlmtr": "size", "sizetmmxwmm": "size",
    "grade": "grade",
    "weight": "net_weight", "netwt": "net_weight", "netweight": "net_weight",
    "grosswt": "gross_weight", "grossweight": "gross_weight",
    "date": "date",
    "packtype": "pack_type",
    "specification": "spec", "spec": "spec", "spec1": "spec", "spect": "spec", "specl": "spec", "speci": "spec",
    "customer": "customer", "customershiptoparty": "customer", "shiptoparty": "customer",
}
MULTILINE = {"customer"}

# Known Indian flat-steel producers: substring in normalised OCR text -> name.
MANUFACTURERS = [
    ("amns", "ArcelorMittal Nippon Steel India (AM/NS India)"),
    ("arcelormittalnippon", "ArcelorMittal Nippon Steel India (AM/NS India)"),
    ("tatasteel", "Tata Steel"),
    ("jswsteel", "JSW Steel"),
    ("sail", "Steel Authority of India (SAIL)"),
    ("jindalsteel", "Jindal Steel & Power"),
    ("jspl", "Jindal Steel & Power"),
    ("essarsteel", "Essar Steel"),
    ("rinl", "RINL / Vizag Steel"),
]
PRODUCTS = ["GALVANIZED", "GALVALUME", "HR COIL", "CR COIL", "HRPO", "PPGI", "COLOR COATED", "TINPLATE"]

QCI_URL = re.compile(r"qcin\.org/product-details/([0-9a-f]+)/([^/]+)/(\d+)", re.I)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


@dataclass
class Box:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass
class TagReading:
    qr_codes: list[str] = field(default_factory=list)
    batch_id: str | None = None
    batch_id_source: str | None = None
    manufacturer: str | None = None
    product: str | None = None
    fields: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ocr_lines: list[str] = field(default_factory=list)
    qci: dict | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------- QR ---------
_wechat = None


def decode_qrs(img: np.ndarray) -> list[str]:
    """Decode all QR codes. WeChat's decoder handles small/curved codes far
    better than cv2.QRCodeDetector; a second pass on upscaled tiles catches the
    small secondary QR that the full-frame pass misses."""
    global _wechat
    if _wechat is None:
        _wechat = cv2.wechat_qrcode_WeChatQRCode()
    found: list[str] = []

    def add(vals):
        for v in vals:
            v = v.strip()
            if v and v not in found:
                found.append(v)

    add(_wechat.detectAndDecode(img)[0])
    h, w = img.shape[:2]
    for ty in range(3):
        for tx in range(3):
            y0, x0 = int(ty * h / 4), int(tx * w / 4)
            tile = img[y0:y0 + h // 2, x0:x0 + w // 2]
            add(_wechat.detectAndDecode(cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))[0])
    return found


# ---------------------------------------------------------------- OCR --------
_ocr = None


def ocr_boxes(img: np.ndarray) -> list[Box]:
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    result, _ = _ocr(img)
    boxes = []
    for pts, text, conf in result or []:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        boxes.append(Box(text.strip(), float(conf), min(xs), min(ys), max(xs), max(ys)))
    boxes.sort(key=lambda b: (b.y0, b.x0))
    return boxes


def label_key(b: Box) -> str | None:
    return LABELS.get(norm(b.text))


def values_below(label: Box, boxes: list[Box], multiline: bool) -> list[Box]:
    """Values are printed directly under their caption, left-aligned with it."""
    out: list[Box] = []
    anchor = label
    while True:
        cands = [
            b for b in boxes
            if b is not anchor and b not in out
            and anchor.y0 + 0.5 * anchor.h < b.y0 < anchor.y1 + 1.8 * label.h
            and abs(b.x0 - label.x0) < 3 * label.h
        ]
        if not cands:
            break
        nxt = min(cands, key=lambda b: b.y0)
        if label_key(nxt):  # ran into the next caption
            break
        out.append(nxt)
        if not multiline or len(out) >= 3:
            break
        anchor = nxt
    return out


# ------------------------------------------------------------ cleanup -------
def clean(key: str, v: str) -> str:
    v = v.strip()
    if key == "size":
        v = re.sub(r"\s*[xX×]\s*", " x ", v).replace("（", "(").replace("）", ")")
    elif key in ("net_weight", "gross_weight"):
        m = re.search(r"\d+(?:\.\d+)?", v.replace(",", "."))
        if m:
            unit = " MT" if "MT" in v.upper() or float(m.group()) < 100 else ""
            v = m.group() + unit
    elif key == "date":
        m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})\s*(\d{1,2}:\d{2}(?::\d{2})?)?\s*([AP]M)?", v, re.I)
        if m:
            v = " ".join(p for p in m.groups() if p)
    elif key == "spec":
        v = re.sub(r"^[1l|]S", "IS", v)  # OCR often reads "IS" as "1S"
    return v


def parse(boxes: list[Box], qrs: list[str]) -> TagReading:
    r = TagReading(qr_codes=qrs, ocr_lines=[b.text for b in boxes])
    full = norm(" ".join(b.text for b in boxes))

    for needle, name in MANUFACTURERS:
        if needle in full:
            r.manufacturer = name
            break
    upper = " ".join(b.text.upper() for b in boxes)
    for p in PRODUCTS:
        if p in upper or p.replace(" ", "") in upper.replace(" ", ""):
            r.product = p.title() if p != "HR COIL" else "HR Coil"
            break

    for b in boxes:
        key = label_key(b)
        if not key:
            continue
        vals = values_below(b, boxes, key in MULTILINE)
        if not vals:
            continue
        value = clean(key, " / ".join(v.text for v in vals))
        k = key
        i = 2
        while k in r.fields:  # e.g. two "Spec." captions on HR tags
            k = f"{key}_{i}"
            i += 1
        r.fields[k] = value

    # QR is ground truth for the batch ID.
    plain = [q for q in qrs if not q.lower().startswith("http")]
    for q in qrs:
        m = QCI_URL.search(q)
        if m:
            r.qci = {"url": q, "product_id": m.group(1), "batch_id": m.group(2), "serial": m.group(3)}
    ocr_batch = r.fields.get("batch_id")
    if plain:
        r.batch_id, r.batch_id_source = plain[0], "qr"
    elif r.qci:
        r.batch_id, r.batch_id_source = r.qci["batch_id"], "qci_url"
    elif ocr_batch:
        r.batch_id, r.batch_id_source = ocr_batch, "ocr"
        r.warnings.append("No QR decoded - batch ID comes from OCR only; verify it.")
    else:
        r.warnings.append("No batch ID found. Retake the photo closer to the tag.")
    if r.batch_id and ocr_batch and norm(ocr_batch) != norm(r.batch_id):
        r.warnings.append(f"Printed batch '{ocr_batch}' does not match QR '{r.batch_id}'.")
    r.fields["batch_id"] = r.batch_id or ""
    if not r.manufacturer:
        r.warnings.append("Manufacturer not recognised - enter it manually.")
    return r


def read_tag(image_bytes: bytes) -> TagReading:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a readable image")
    # Phone photos can be 4000px+; OCR and QR both work best around 1600px.
    h, w = img.shape[:2]
    s = 1600 / max(h, w)
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return parse(ocr_boxes(img), decode_qrs(img))


if __name__ == "__main__":
    import json
    import sys
    for p in sys.argv[1:]:
        with open(p, "rb") as f:
            print(p, json.dumps(read_tag(f.read()).to_dict(), indent=2))
