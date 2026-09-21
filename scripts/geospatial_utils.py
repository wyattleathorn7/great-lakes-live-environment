"""Shared geospatial + rendering utilities for the three live layers.

Every product bins its own source grid (already mapped to WGS84
lat/lon arrays) onto the ONE common canvas defined in
config/great_lakes_bounds.json, so all overlays align exactly.
Rendering is RASTER ONLY: RGBA PNG + KML GroundOverlay. This module
never emits LineString / Polygon / Placemark geometry.
"""

import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
SITE_DIR = os.path.join(REPO_ROOT, "site")
KML_DIR = os.path.join(REPO_ROOT, "kml")

USER_AGENT = {"User-Agent": "great-lakes-live-environment/1.0 (NOAA data automation; contact: repo owner)"}


def load_bounds():
    with open(os.path.join(CONFIG_DIR, "great_lakes_bounds.json")) as f:
        return json.load(f)


def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def http_date_to_iso(s):
    """Normalize an HTTP Last-Modified value to 'YYYY-MM-DD HH:MM UTC'.

    Returns the original string when unparsable (never raises).
    """
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return s


def download(url, dest, timeout=120, retries=3):
    """Download url -> dest (atomic write). Returns size + last_modified.

    Creates missing parent directories (fresh CI checkouts start empty).
    Retries transient failures (timeouts, resets, HTTP 5xx/408/429) with
    backoff; client errors such as HTTP 404 fail immediately. Partial
    downloads never replace dest (written to .part, then renamed).
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                total = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        total += len(chunk)
                last_modified = r.headers.get("Last-Modified")
            os.replace(tmp, dest)
            return {"size_bytes": total, "http_last_modified": last_modified}
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 404 or (400 <= e.code < 500 and e.code not in (408, 429)):
                raise  # permanent: retrying cannot help
            # transient 5xx/408/429 -> fall through to retry
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        if attempt < retries:
            time.sleep(2 * attempt)
    raise last_err


# ---------------------------------------------------------------- canvas map
def canvas_indices(lats, lons, bounds):
    """Map WGS84 lat/lon arrays to common-canvas pixel indices.

    Returns (rows, cols, valid) where valid marks points inside the bounds.
    Row 0 = north (lat_max), as required by PNG/KML LatLonBox.
    """
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()
    valid = (
        np.isfinite(lats) & np.isfinite(lons)
        & (lats >= lat_min) & (lats <= lat_max)
        & (lons >= lon_min) & (lons <= lon_max)
    )
    cols = np.clip(((lons - lon_min) / (lon_max - lon_min) * W).astype(int), 0, W - 1)
    rows = np.clip(((lat_max - lats) / (lat_max - lat_min) * H).astype(int), 0, H - 1)
    return rows, cols, valid


def bin_to_canvas(rows, cols, values, valid, shape, splat_radius=0):
    """Nearest-neighbour binning: mean of source values falling in each pixel.

    Source grids are often coarser than the canvas (e.g. 2.5 km wave grid
    vs ~1.2 km canvas pixels), which would leave polka-dot holes with
    single-pixel binning. splat_radius>0 paints each source point onto a
    (2r+1)x(2r+1) neighbourhood with mean-averaging, so lakes render solid
    without inventing precision (documented as display smoothing only).

    Returns (mean_grid, count_grid). Pixels with no source points are NaN/0.
    """
    H, W = shape
    r = rows[valid].astype(np.int64)
    c = cols[valid].astype(np.int64)
    v = values.ravel()[valid].astype(float)
    flat = r * W + c
    sums = np.bincount(flat, weights=v, minlength=H * W).reshape(H, W)
    counts = np.bincount(flat, minlength=H * W).reshape(H, W)
    if splat_radius > 0:
        rr = splat_radius
        padded_s = np.pad(sums, rr)
        padded_c = np.pad(counts, rr)
        acc_s = np.zeros_like(sums)
        acc_c = np.zeros_like(counts)
        for dr in range(2 * rr + 1):
            for dc in range(2 * rr + 1):
                acc_s += padded_s[dr:dr + H, dc:dc + W]
                acc_c += padded_c[dr:dr + H, dc:dc + W]
        sums, counts = acc_s, acc_c
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / np.where(counts > 0, counts, np.nan)
    return mean, counts


# ---------------------------------------------------------------- colormaps
def _interp_stops(stops, t):
    """stops: list of (pos, (r,g,b)) sorted by pos; t in [0,1]."""
    if t <= stops[0][0]:
        return stops[0][1]
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if t <= p1:
            f = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            return tuple(int(round(a + (b - a) * f)) for a, b in zip(c0, c1))
    return stops[-1][1]


def _lut(stops, n=256):
    return [_interp_stops(stops, i / (n - 1)) for i in range(n)]


WAVE_STOPS = [  # calm deep blue -> cyan -> green -> yellow -> orange -> red
    (0.00, (16, 52, 140)), (0.20, (20, 110, 200)), (0.40, (20, 170, 200)),
    (0.60, (120, 200, 60)), (0.75, (250, 210, 40)), (0.88, (240, 120, 20)),
    (1.00, (200, 20, 20)),
]
TEMP_STOPS = [  # cold blue -> cyan -> green -> yellow -> orange -> red
    (0.00, (30, 60, 180)), (0.25, (30, 150, 220)), (0.45, (60, 190, 150)),
    (0.60, (240, 220, 60)), (0.80, (240, 130, 30)), (1.00, (190, 30, 30)),
]
ICE_STOPS = [  # low ice pale cyan -> blue -> deep navy -> near-total white
    (0.00, (190, 235, 255)), (0.10, (140, 210, 255)), (0.25, (70, 160, 240)),
    (0.45, (35, 100, 215)), (0.65, (20, 55, 150)), (0.85, (10, 25, 90)),
    (1.00, (245, 250, 255)),
]


def apply_colormap(field, vmin, vmax, stops, alpha, transparent_value=None):
    """Render a 2D float field (NaN = transparent) to an RGBA uint8 image."""
    H, W = field.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    ok = np.isfinite(field)
    if transparent_value is not None:
        ok = ok & (field != transparent_value)
    if not np.any(ok):
        return rgba  # fully transparent (legit for ice-free days)
    span = vmax - vmin if vmax > vmin else 1.0
    t = np.clip((field - vmin) / span, 0.0, 1.0)
    lut = np.array(_lut(stops), dtype=np.uint8)  # 256x3
    idx = (t[ok] * 255).astype(int)
    rgba[ok, 0:3] = lut[idx]
    rgba[ok, 3] = alpha
    return rgba


def save_png(rgba, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


# ------------------------------------------------------------------- legend
LEGEND_W, LEGEND_H = 640, 210


def _legend_font(size):
    """Legible TTF legend font with graceful fallback.

    Tries bare name, then well-known system locations (Ubuntu GH runners
    ship fonts-dejavu-core; macOS ships Helvetica), else PIL's bitmap font.
    """
    from PIL import ImageFont
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_legend(path, title, subtitle, unit_label, vmin, vmax, stops,
                source_line, fmt="{:.0f}", transparent_note=None):
    """Draw a standalone legend PNG (used by KML ScreenOverlay)."""
    W, H = LEGEND_W, LEGEND_H
    img = Image.new("RGBA", (W, H), (255, 255, 255, 235))
    d = ImageDraw.Draw(img)
    f_title, f_body, f_small = _legend_font(22), _legend_font(15), _legend_font(13)
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=2)
    d.text((14, 8), title, font=f_title, fill=(10, 10, 10))
    d.text((14, 36), subtitle, font=f_body, fill=(40, 40, 40))
    bx, by, bw, bh = 14, 70, W - 28, 30
    lut = _lut(stops, bw)
    for i, c in enumerate(lut):
        d.line([(bx + i, by), (bx + i, by + bh)], fill=c + (255,))
    d.rectangle([bx, by, bx + bw - 1, by + bh], outline=(40, 40, 40))
    for frac, val in ((0.0, vmin), (0.5, (vmin + vmax) / 2), (1.0, vmax)):
        x = bx + int(frac * (bw - 1))
        d.text((min(max(x - 18, 2), W - 70), by + bh + 4),
               fmt.format(val), font=f_body, fill=(10, 10, 10))
    d.text((bx + bw - 66, by + bh + 26), unit_label, font=f_body, fill=(10, 10, 10))
    d.text((14, H - 40), source_line, font=f_small, fill=(60, 60, 60))
    if transparent_note:
        d.text((14, H - 22), transparent_note, font=f_small, fill=(60, 60, 60))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def fmt_ticks(vmin, vmax):
    span = vmax - vmin
    if span <= 12:
        return "{:.1f}"
    return "{:.0f}"


# ------------------------------------------------------------------ metadata
def write_metadata(product_dir, meta):
    os.makedirs(product_dir, exist_ok=True)
    path = os.path.join(product_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def base_metadata(product, title, freshness_label, source_name, source_url,
                  variable, data_time_utc, source_last_modified_utc,
                  units, source_resolution, color_min, color_max,
                  color_units, missing_data_treatment):
    bounds = load_bounds()
    return {
        "product": product,
        "title": title,
        "freshness": freshness_label,
        "noaa_source": source_name,
        "product_name": source_name,
        "variable": variable,
        "source_url": source_url,
        "data_time_utc": data_time_utc,
        "source_last_modified_utc": source_last_modified_utc,
        "processing_time_utc": utcnow_iso(),
        "units": units,
        "spatial_resolution_source": source_resolution,
        "spatial_resolution_rendered": (
            f"{bounds['canvas_width']}x{bounds['canvas_height']} canvas over "
            f"lon [{bounds['lon_min']},{bounds['lon_max']}], "
            f"lat [{bounds['lat_min']},{bounds['lat_max']}] (mean-value binning, no upsampling of source precision)"
        ),
        "crs": bounds["crs"],
        "render_bounds": {
            "lon_min": bounds["lon_min"], "lon_max": bounds["lon_max"],
            "lat_min": bounds["lat_min"], "lat_max": bounds["lat_max"],
        },
        "color_scale_min": color_min,
        "color_scale_max": color_max,
        "color_scale_units": color_units,
        "missing_data_treatment": missing_data_treatment,
        "attribution": ("Data: US NOAA. This project is not endorsed by NOAA. "
                        "See DATA_SOURCES.md for exact products and endpoints."),
    }


# ------------------------------------------------------------------ state
def state_path(product):
    return os.path.join(REPO_ROOT, "output", "state", f"{product}.json")


def read_state(product):
    p = state_path(product)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}  # corrupt state must trigger a rebuild, not a crash
    return {}


def write_state(product, state):
    os.makedirs(os.path.dirname(state_path(product)), exist_ok=True)
    with open(state_path(product), "w") as f:
        json.dump(state, f, indent=2)


# ------------------------------------------------------- atomic staging
def stage_dir(product):
    """Scratch dir mirroring repo-relative output paths for one product."""
    return os.path.join(REPO_ROOT, "output", "stage", product)


def promote_stage(product):
    """Move every staged file into its live repo location (makedirs as needed).

    Builders render into the stage first so a mid-build crash can never
    leave a half-updated product set behind for the artifact uploader.
    Returns the list of promoted repo-relative paths.
    """
    import shutil
    root = stage_dir(product)
    promoted = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, root)
            dest = os.path.join(REPO_ROOT, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(src, dest)
            promoted.append(rel)
    shutil.rmtree(root, ignore_errors=True)
    return promoted


# ------------------------------------------------------------------ NDBC QC
def fetch_buoy_obs(buoy_ids):
    """Fetch latest NDBC realtime obs for given buoys. Returns {id: dict}."""
    out = {}
    for bid in buoy_ids:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{bid}.txt"
        try:
            req = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(req, timeout=30) as r:
                lines = r.read().decode(errors="replace").splitlines()
            if len(lines) < 3:
                continue
            cols = lines[0].split()
            vals = lines[2].split()
            row = dict(zip(cols, vals))
            out[bid] = {
                "time_utc": (f"{row.get('#YY')}-{row.get('MM')}-{row.get('DD')} "
                             f"{row.get('hh')}:{row.get('mm')} UTC"),
                "WVHT_m": _f(row.get("WVHT")),
                "WTMP_C": _f(row.get("WTMP")),
            }
        except Exception as e:  # buoy fetch must never break a pipeline
            out[bid] = {"error": str(e)[:160]}
    return out


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None
