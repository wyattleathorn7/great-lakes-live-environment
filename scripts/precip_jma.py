"""JMA high-resolution precipitation nowcast adapter (Japan Meteorological
Agency, attribution required).

Source: anonymous XYZ tile fetches, no credentials.
  Times:  .../jmatile/data/nowc/targetTimes_N1.json -> [{basetime,
            validtime, ...}] (5-min cadence; analysis = validtime==basetime)
  Tiles:  .../jmatile/data/nowc/{bt}/none/{bt}/surf/hrpns/{z}/{x}/{y}.png
  Grid: Web-Mercator tiles; z5 (256px tiles, ~0.044 deg/px — finer than
    both canvases, so nearest sampling loses nothing).
  Pixels: RGBA; opaque = analyzed precipitation, transparent = none.
  Classes (mm/h lower bounds, JMA published nowcast scale 1/5/10/20/30/
    50/80 — colors below verified live from served tiles):
    (242,242,255)->1, (160,210,255)->5, (33,140,255)->10,
    (0,65,255)->20, (250,245,0)->30, (255,153,0)->50.
    The 80+ color has not been observed in served tiles yet; nearest-
    match undercalls it to the 50 class (documented, never inflated).
  Coverage: fixed radar-plausible box lon 122..148, lat 24..46 (official
    distribution domain is wider: 118..150/20..48; fringes are ceded to
    GFS rather than claimed). Inside the box, JMA-transparent suppresses
    model fill (same dry-is-valid contract as OPERA).

Only intensity is consumed (no precipitation-type product here), so JMA
echoes render undifferentiated on the N0Q key via Z-R estimated dBZ
equivalents (same bridge as ECCC rain) and never as classified snow.
"""

import datetime as dt
import math
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "https://www.jma.go.jp/bosai/jmatile/data/nowc"
UA = {"User-Agent": "great-lakes-live-environment/1.0 (jma-jp-nowcast)"}
PRODUCT = "hrpns"
ZOOM = 5
LON_MIN, LON_MAX = 122.0, 148.0
LAT_MIN, LAT_MAX = 24.0, 46.0
LICENSE = "Japan Meteorological Agency (attribution required)"
FOOTPRINT = "Japan + vicinity (lon 122..148, lat 24..46)"

# (RGB, mm/h lower bound) in JMA published class order.
JMA_CLASSES = [
    ((242, 242, 255), 1.0),
    ((160, 210, 255), 5.0),
    ((33, 140, 255), 10.0),
    ((0, 65, 255), 20.0),
    ((250, 245, 0), 30.0),
    ((255, 153, 0), 50.0),
]
JMA_RATES = [r for _, r in JMA_CLASSES]


def _tile_range():
    """(x0, x1, y0, y1) inclusive z5 tile range covering the box."""
    n = 2 ** ZOOM
    x0 = max(0, int(math.floor((LON_MIN + 180.0) / 360.0 * n)))
    x1 = min(n - 1, int(math.floor((LON_MAX + 180.0) / 360.0 * n - 1e-9)))
    def _y(lat):
        r = math.radians(lat)
        return (1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n
    y0 = max(0, int(math.floor(_y(LAT_MAX))))
    y1 = min(n - 1, int(math.floor(_y(LAT_MIN))))
    return x0, x1, y0, y1


def latest_time(now=None):
    """Newest analysis basetime (validtime==basetime) at or before now."""
    now = now or dt.datetime.now(dt.timezone.utc)
    url = f"{BASE}/targetTimes_N1.json"
    req = urllib.request.Request(url, headers=UA)
    import json
    with urllib.request.urlopen(req, timeout=60) as r:
        items = json.loads(r.read().decode("utf-8", errors="replace"))
    cands = []
    for it in items:
        try:
            if it.get("validtime") != it.get("basetime"):
                continue
            ts = dt.datetime.strptime(it["basetime"], "%Y%m%d%H%M%S").replace(
                tzinfo=dt.timezone.utc)
        except (ValueError, KeyError, TypeError):
            continue
        if ts <= now:
            cands.append(ts)
    if not cands:
        raise RuntimeError("no JMA nowcast analysis found")
    return max(cands).strftime("%Y%m%d%H%M%S")


def fetch(basetime, raw_dir):
    """Download every box tile for one analysis (all-or-nothing: any
    failure raises, keeping the previous raster — a partial tile set
    would punch misleading holes). Returns tile paths."""
    x0, x1, y0, y1 = _tile_range()
    paths = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            dest = os.path.join(raw_dir, f"jma_{basetime}_z{ZOOM}_{x}_{y}.png")
            if not (os.path.exists(dest) and os.path.getsize(dest) > 200):
                url = (f"{BASE}/{basetime}/none/{basetime}/surf/{PRODUCT}/"
                       f"{ZOOM}/{x}/{y}.png")
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"JMA non-PNG tile {ZOOM}/{x}/{y}")
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, dest)
            paths.append(dest)
    return paths


def mosaic_to_canvas(tile_paths, bounds, W, H):
    """Nearest-sample tile mosaic onto a canvas grid. Returns
    (cls, covered, echo): cls = JMA class index (nearest official color;
    -1 where transparent); covered = inside the JMA box; echo = covered
    & opaque."""
    from PIL import Image
    n = 2 ** ZOOM
    tiles = {}
    for p in tile_paths:
        base = os.path.basename(p).replace(".png", "").split("_")
        try:
            tiles[(int(base[-2]), int(base[-1]))] = np.array(
                Image.open(p).convert("RGBA"))
        except Exception:
            continue
    pal = np.array([c for c, _ in JMA_CLASSES], dtype=float)
    lon = bounds["lon_min"] + (np.arange(W) + 0.5) * (
        bounds["lon_max"] - bounds["lon_min"]) / W
    lat = bounds["lat_max"] - (np.arange(H) + 0.5) * (
        bounds["lat_max"] - bounds["lat_min"]) / H
    inbox = ((lon[None, :] >= LON_MIN) & (lon[None, :] <= LON_MAX)
             & (lat[:, None] >= LAT_MIN) & (lat[:, None] <= LAT_MAX))
    gx = np.clip(((lon + 180.0) / 360.0 * n * 256).astype(int), 0, n * 256 - 1)
    r = np.radians(np.clip(lat, -85.0, 85.0))
    gy = np.clip(((1.0 - np.log(np.tan(r) + 1.0 / np.cos(r)) / math.pi)
                  / 2.0 * n * 256).astype(int), 0, n * 256 - 1)
    cls = np.full((H, W), -1, dtype=np.int16)
    opa = np.zeros((H, W), dtype=bool)
    ys, xs = np.where(inbox)
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        tx, ty = gx[xx] // 256, gy[yy] // 256
        t = tiles.get((tx, ty))
        if t is None:
            continue
        px = t[gy[yy] % 256, gx[xx] % 256]
        if px[3] == 0:
            continue
        opa[yy, xx] = True
        d2 = ((pal - px[:3].astype(float)) ** 2).sum(axis=1)
        cls[yy, xx] = int(d2.argmin())
    covered = inbox
    echo = inbox & opa
    return cls, covered, echo
