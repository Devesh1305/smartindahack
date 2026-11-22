"""CoilTag server: read steel coil tags, store geotagged scans, serve the map.

Run:  python app.py --https      (phones need HTTPS for camera + GPS)
Then open https://<this-pc-ip>:5000 on the phone, /map for the dashboard.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from fractions import Fraction

import piexif
from flask import Flask, Response, abort, g, jsonify, request, send_from_directory
from PIL import Image

from label_reader import read_tag

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("COILTAG_DATA", os.path.join(BASE, "data"))
UPLOADS = os.path.join(DATA, "uploads")
IMAGES = os.path.join(DATA, "images")
DB = os.path.join(DATA, "coiltag.db")
for d in (UPLOADS, IMAGES):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    manufacturer TEXT,
    product TEXT,
    fields TEXT NOT NULL,          -- JSON of all label fields (after user review)
    qr_codes TEXT NOT NULL,        -- JSON list of raw QR payloads
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    accuracy_m REAL,
    captured_at TEXT NOT NULL,     -- when the GPS fix / photo was taken (device clock, UTC)
    created_at TEXT NOT NULL,      -- when the server stored it
    image TEXT NOT NULL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_scans_batch ON scans(batch_id);
"""


def db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
        g.db.executescript(SCHEMA)
    return g.db


@app.teardown_appcontext
def _close(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


# ------------------------------------------------------------ geotagging ----
def _dms(value: float):
    value = abs(value)
    d = int(value)
    m = int((value - d) * 60)
    s = Fraction((value - d - m / 60) * 3600).limit_denominator(10000)
    return ((d, 1), (m, 1), (s.numerator, s.denominator))


def geotag_jpeg(src: str, dst: str, lat: float, lon: float, description: str, when: datetime):
    """Write GPS + description into the JPEG's EXIF so the photo itself carries
    the location (shows up in Google Photos, QGIS, exiftool, etc.)."""
    img = Image.open(src)
    try:
        exif = piexif.load(img.info["exif"]) if "exif" in img.info else {}
    except Exception:
        exif = {}
    for k in ("0th", "Exif", "GPS", "1st"):
        exif.setdefault(k, {})
    exif.pop("thumbnail", None)
    exif["GPS"] = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: _dms(lat),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: _dms(lon),
        piexif.GPSIFD.GPSDateStamp: when.strftime("%Y:%m:%d").encode(),
        piexif.GPSIFD.GPSTimeStamp: ((when.hour, 1), (when.minute, 1), (when.second, 1)),
    }
    exif["0th"][piexif.ImageIFD.ImageDescription] = description.encode("utf-8")
    exif["Exif"][piexif.ExifIFD.UserComment] = b"ASCII\0\0\0" + description.encode("ascii", "replace")
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(dst, "JPEG", quality=90, exif=piexif.dump(exif))


# ------------------------------------------------------------------ pages ----
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/map")
def map_page():
    return send_from_directory(app.static_folder, "map.html")


@app.get("/images/<path:name>")
def image(name):
    return send_from_directory(IMAGES, name)


# -------------------------------------------------------------------- API ----
@app.post("/api/read")
def api_read():
    f = request.files.get("image")
    if not f:
        abort(400, "image missing")
    data = f.read()
    try:
        reading = read_tag(data)
    except ValueError as e:
        abort(400, str(e))
    upload_id = uuid.uuid4().hex
    with open(os.path.join(UPLOADS, upload_id), "wb") as out:
        out.write(data)
    return jsonify(upload_id=upload_id, **reading.to_dict())


@app.post("/api/scans")
def api_save():
    body = request.get_json(force=True)
    upload_id = str(body.get("upload_id", ""))
    if not upload_id.isalnum():
        abort(400, "bad upload_id")
    src = os.path.join(UPLOADS, upload_id)
    if not os.path.exists(src):
        abort(400, "upload expired - read the tag again")
    batch = (body.get("batch_id") or "").strip()
    if not batch:
        abort(400, "batch_id is required")
    try:
        lat, lon = float(body["lat"]), float(body["lon"])
    except (KeyError, TypeError, ValueError):
        abort(400, "GPS position is required")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        abort(400, "GPS position out of range")
    acc = body.get("accuracy")
    try:
        captured = datetime.fromisoformat(str(body.get("captured_at")).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        captured = datetime.now(timezone.utc)
    manufacturer = (body.get("manufacturer") or "").strip()
    fields = body.get("fields") or {}
    fields["batch_id"] = batch

    name = f"{batch.replace('/', '_')}_{captured.strftime('%Y%m%dT%H%M%S')}_{upload_id[:6]}.jpg"
    geotag_jpeg(src, os.path.join(IMAGES, name), lat, lon,
                f"{manufacturer} | Batch {batch} | {body.get('product') or ''}".strip(" |"), captured)
    os.remove(src)

    cur = db().execute(
        "INSERT INTO scans (batch_id, manufacturer, product, fields, qr_codes, lat, lon, accuracy_m,"
        " captured_at, created_at, image, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (batch, manufacturer, body.get("product"), json.dumps(fields), json.dumps(body.get("qr_codes") or []),
         lat, lon, float(acc) if acc is not None else None, captured.isoformat(),
         datetime.now(timezone.utc).isoformat(), name, body.get("note")),
    )
    db().commit()
    return jsonify(id=cur.lastrowid, image=f"/images/{name}")


def _rows(batch: str | None):
    q, args = "SELECT * FROM scans", ()
    if batch:
        q, args = q + " WHERE batch_id = ?", (batch,)
    out = []
    for r in db().execute(q + " ORDER BY captured_at DESC", args):
        d = dict(r)
        d["fields"] = json.loads(d["fields"])
        d["qr_codes"] = json.loads(d["qr_codes"])
        d["image_url"] = f"/images/{d['image']}"
        out.append(d)
    return out


@app.get("/api/scans")
def api_list():
    return jsonify(_rows(request.args.get("batch")))


@app.delete("/api/scans/<int:scan_id>")
def api_delete(scan_id):
    db().execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    db().commit()
    return jsonify(ok=True)


@app.get("/api/scans.csv")
def api_csv():
    rows = _rows(request.args.get("batch"))
    keys = sorted({k for r in rows for k in r["fields"]})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "batch_id", "manufacturer", "product", "lat", "lon", "accuracy_m", "captured_at", "image", *keys])
    for r in rows:
        w.writerow([r["id"], r["batch_id"], r["manufacturer"], r["product"], r["lat"], r["lon"], r["accuracy_m"],
                    r["captured_at"], request.host_url.rstrip("/") + r["image_url"], *[r["fields"].get(k, "") for k in keys]])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=coil_scans.csv"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--https", action="store_true", help="self-signed HTTPS so phones allow camera + GPS")
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, ssl_context="adhoc" if a.https else None)
