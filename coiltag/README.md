# CoilTag: steel coil tag scanner with GPS geotagging

You photograph a coil tag on your phone. The server decodes **every QR code** on the tag, OCRs the printed fields and shows them to you to confirm. It then saves the scan with the phone's GPS position. The location is written into the photo's EXIF and shown as a pin on a map.

## What's actually in the QR codes (tested on real AM/NS India tags)

| Tag | QR payload |
|---|---|
| Galvanized (9103KC000H) | `9103KC000H`: batch ID only |
| HR coil (NK9946), middle QR | `NK9946` followed by padding spaces: batch ID only |
| HR coil (NK9946), top-right QR | `https://madeinindia.qcin.org/product-details/<product-id>/NK9946/00000001` (QCI "Made in India" link) |

Manufacturer, size, weight, grade, heat number, spec and customer are **only printed**, so the app OCRs them. The QR batch ID is treated as ground truth. If the OCR'd batch ID disagrees with it, the app shows a warning.

## Install & run

```bash
pip install -r requirements.txt
# rapidocr pulls in plain opencv-python, which lacks the WeChat QR decoder. Swap it out:
pip uninstall -y opencv-python opencv-contrib-python-headless && pip install opencv-contrib-python-headless
python app.py --https
```

* On the phone (same Wi-Fi), open `https://<laptop-ip>:5000` and accept the self-signed certificate warning. Phones only allow camera and GPS on HTTPS.
* Open `/map` for the dashboard and `/api/scans.csv` for an export.
* Data is stored in `data/` (SQLite plus geotagged JPEGs).

## Flow

1. The page starts watching GPS as soon as it opens, so the fix is warm by the time you take the photo. The position is frozen **at the moment the photo is taken**, not when you press save.
2. `POST /api/read` decodes the QRs (OpenCV WeChat decoder, full frame plus upscaled tiles so small secondary QRs are caught) and runs OCR (RapidOCR). Each caption is matched to the value printed below it.
3. The user reviews and corrects the fields. GPS accuracy worse than 50 m triggers a confirmation prompt.
4. `POST /api/scans` writes GPS, date and "Manufacturer | Batch | Product" into the JPEG's EXIF and stores the row.
5. The map shows one pin per coil. Re-scanning the same batch somewhere else draws its movement trail.

## Tests

```bash
python -m pytest -q tests
```
These run against the two real tag photos in `tests/samples/`.
