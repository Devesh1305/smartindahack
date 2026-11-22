"""End-to-end tests against the two real AM/NS India tag photos."""
import io
import os
import sys
import tempfile

import piexif
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["COILTAG_DATA"] = tempfile.mkdtemp()

from label_reader import read_tag  # noqa: E402
import app as server  # noqa: E402

SAMPLES = os.path.join(os.path.dirname(__file__), "samples")


def load(name):
    with open(os.path.join(SAMPLES, name), "rb") as f:
        return f.read()


def test_galvanized_tag():
    r = read_tag(load("galvanized.jpg"))
    assert r.qr_codes == ["9103KC000H"]
    assert r.batch_id == "9103KC000H" and r.batch_id_source == "qr"
    assert "AM/NS" in r.manufacturer
    assert r.product == "Galvanized"
    f = r.fields
    assert f["size"] == "0.410 x 1220 x 5278"
    assert f["net_weight"] == "20.725 MT"
    assert f["gross_weight"] == "20.825 MT"
    assert f["date"] == "11/10/2026 11:36:54 AM"
    assert f["pack_type"] == "DOM_OEM"
    assert "GP250" in f["spec"]
    assert "PUNE" in f["customer"]
    assert not r.warnings


def test_hr_coil_tag_with_two_qrs():
    r = read_tag(load("hr_coil.jpg"))
    assert "NK9946" in r.qr_codes
    assert any("qcin.org" in q for q in r.qr_codes), "secondary QCI QR not decoded"
    assert r.qci["batch_id"] == "NK9946"
    assert r.batch_id == "NK9946" and r.batch_id_source == "qr"
    assert "AM/NS" in r.manufacturer
    assert r.product == "HR Coil"
    f = r.fields
    assert f["heat_no"] == "S40432"
    assert f["grade"] == "ANCRMB2"
    assert f["net_weight"] == "22.760 MT"
    assert f["size"].startswith("2.700 x 1260")
    assert "11513" in f["spec"]
    assert not r.warnings


def test_read_save_list_geotags_image():
    c = server.app.test_client()
    res = c.post("/api/read", data={"image": (io.BytesIO(load("hr_coil.jpg")), "t.jpg")})
    assert res.status_code == 200
    rd = res.get_json()
    lat, lon = 21.1702, 72.8311
    res = c.post("/api/scans", json={"upload_id": rd["upload_id"], "batch_id": rd["batch_id"],
                 "manufacturer": rd["manufacturer"], "product": rd["product"], "fields": rd["fields"],
                 "qr_codes": rd["qr_codes"], "lat": lat, "lon": lon, "accuracy": 8,
                 "captured_at": "2026-11-21T05:30:00Z"})
    assert res.status_code == 200, res.data
    img = c.get(res.get_json()["image"]).data
    gps = piexif.load(img)["GPS"]
    d, m, s = (a / b for a, b in gps[piexif.GPSIFD.GPSLatitude])
    assert abs(d + m / 60 + s / 3600 - lat) < 1e-5
    d, m, s = (a / b for a, b in gps[piexif.GPSIFD.GPSLongitude])
    assert abs(d + m / 60 + s / 3600 - lon) < 1e-5
    assert b"NK9946" in piexif.load(img)["0th"][piexif.ImageIFD.ImageDescription]
    rows = c.get("/api/scans?batch=NK9946").get_json()
    assert rows[0]["lat"] == lat and rows[0]["manufacturer"].startswith("ArcelorMittal")
    assert b"NK9946" in c.get("/api/scans.csv").data


def test_save_rejects_missing_gps():
    c = server.app.test_client()
    rd = c.post("/api/read", data={"image": (io.BytesIO(load("galvanized.jpg")), "t.jpg")}).get_json()
    res = c.post("/api/scans", json={"upload_id": rd["upload_id"], "batch_id": "X"})
    assert res.status_code == 400
