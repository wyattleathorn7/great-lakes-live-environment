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
WATERMASK_PATH = os.path.join(REPO_ROOT, "assets", "great_lakes_watermask.png")
PROD_BASE_URL = "https://wyattleathorn7.github.io/great-lakes-live-environment"

WATERMASK_4X_PATH = os.path.join(REPO_ROOT, "assets", "great_lakes_watermask_4x.png")

_WATERMASK = None
_WATERMASK_4X = None


def tile_layout():
    """LOD pyramid: [(level, divisions_per_axis, min_lod_pixels)]."""
    return [(1, 2, 256), (2, 4, 512)]


def tile_bounds(level, ix, iy):
    """Bounds dict (with 1800x1175 canvas) for one tile of the pyramid."""
    base = load_bounds()
    n = {1: 2, 2: 4}[level]
    assert 0 <= ix < n and 0 <= iy < n
    lon_min = base["lon_min"] + (base["lon_max"] - base["lon_min"]) * ix / n
    lon_max = base["lon_min"] + (base["lon_max"] - base["lon_min"]) * (ix + 1) / n
    # iy=0 is the NORTH row (row 0 of the image)
    lat_max = base["lat_max"] - (base["lat_max"] - base["lat_min"]) * iy / n
    lat_min = base["lat_max"] - (base["lat_max"] - base["lat_min"]) * (iy + 1) / n
    return {"crs": base["crs"], "lon_min": lon_min, "lon_max": lon_max,
            "lat_min": lat_min, "lat_max": lat_max,
            "canvas_width": 1800, "canvas_height": 1175,
            "overlay_alpha": base["overlay_alpha"]}


def tile_rel_path(product, level, ix, iy):
    return f"{product}/tiles/z{level}_{ix}_{iy}.png"


def load_watermask_4x():
    """7200x4700 tile-source mask (float 0..1). Committed asset, loaded once."""
    global _WATERMASK_4X
    if _WATERMASK_4X is None:
        if not os.path.exists(WATERMASK_4X_PATH):
            raise FileNotFoundError(
                f"tile shoreline mask missing: {WATERMASK_4X_PATH}")
        m = np.array(Image.open(WATERMASK_4X_PATH).convert("L"))
        if m.shape != (4700, 7200):
            raise ValueError(f"4x mask shape {m.shape} != (4700, 7200)")
        _WATERMASK_4X = m.astype(np.float32) / 255.0
    return _WATERMASK_4X


def mask_crop_for_tile(tb):
    """Crop of the 4x mask exactly covering one tile's lon/lat box."""
    base = load_bounds()
    m = load_watermask_4x()
    H4, W4 = m.shape
    x0 = int(round((tb["lon_min"] - base["lon_min"])
                   / (base["lon_max"] - base["lon_min"]) * W4))
    x1 = int(round((tb["lon_max"] - base["lon_min"])
                   / (base["lon_max"] - base["lon_min"]) * W4))
    y0 = int(round((base["lat_max"] - tb["lat_max"])
                   / (base["lat_max"] - base["lat_min"]) * H4))
    y1 = int(round((base["lat_max"] - tb["lat_min"])
                   / (base["lat_max"] - base["lat_min"]) * H4))
    crop = m[y0:y1, x0:x1]
    return np.array(Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (tb["canvas_width"], tb["canvas_height"]), Image.LANCZOS)
    ).astype(np.float32) / 255.0


def load_watermask():
    """Shared GSHHG water mask (8-bit; 255 = water). Loaded once.

    Raises if the committed asset is missing or the wrong size: shipping
    unmasked rasters would silently regress the shoreline, so this is a
    loud failure (product keeps its previous raster via exit codes).
    """
    global _WATERMASK
    if _WATERMASK is None:
        bounds = load_bounds()
        if not os.path.exists(WATERMASK_PATH):
            raise FileNotFoundError(
                f"shared shoreline mask missing: {WATERMASK_PATH}")
        m = np.array(Image.open(WATERMASK_PATH).convert("L"))
        if m.shape != (bounds["canvas_height"], bounds["canvas_width"]):
            raise ValueError(
                f"shoreline mask shape {m.shape} != canvas "
                f"({bounds['canvas_height']}, {bounds['canvas_width']})")
        _WATERMASK = m.astype(np.float32) / 255.0
    return _WATERMASK


def apply_shoreline_mask(rgba):
    """Multiply overlay alpha by the shared water mask (antialiased edges).

    Same mask object for every product, so all layers share one shoreline.
    """
    mask = load_watermask()
    out = rgba.copy()
    out[:, :, 3] = np.round(out[:, :, 3].astype(np.float32) * mask).astype(np.uint8)
    return out

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


WAVE_STOPS = [  # continuous anchor gradient, ft -> color (piecewise-linear).
    # Anchors bunch toward low values, so each successive range gets
    # progressively less color resolution (0-9 highest, 27-30+ compressed).
    # Values above 30 ft clamp into the dark-purple extreme end.
    (0.0 / 30, (13, 42, 120)),    # 0 dark blue
    (1.0 / 30, (20, 100, 215)),   # 1 blue
    (2.0 / 30, (20, 170, 225)),   # 2 blue -> cyan
    (3.0 / 30, (20, 200, 200)),   # 3 cyan
    (5.0 / 30, (80, 195, 120)),   # 5 cyan -> green
    (6.0 / 30, (120, 200, 60)),   # 6 green
    (9.0 / 30, (250, 220, 40)),   # 9 green -> yellow
    (10.0 / 30, (250, 215, 40)),  # 10 yellow
    (12.0 / 30, (250, 185, 30)),  # 12 yellow -> yellow-orange
    (13.0 / 30, (250, 160, 30)),  # 13 yellow-orange -> orange
    (15.0 / 30, (240, 120, 20)),  # 15 orange
    (16.0 / 30, (240, 115, 25)),  # 16 orange
    (20.0 / 30, (210, 30, 30)),   # 20 orange -> red
    (21.0 / 30, (200, 25, 45)),   # 21 red
    (23.0 / 30, (170, 20, 90)),   # 23 red -> red-violet
    (24.0 / 30, (150, 20, 110)),  # 24 red-violet
    (26.0 / 30, (120, 30, 150)),  # 26 red-violet -> violet
    (27.0 / 30, (110, 25, 150)),  # 27 violet
    (30.0 / 30, (60, 10, 90)),    # 30+ violet -> purple -> dark purple
]
WAVE_TICKS = [(0, "0 ft"), (2, "2 ft"), (5, "5 ft"), (9, "9 ft"),
              (12, "12 ft"), (15, "15 ft"), (20, "20 ft"), (23, "23 ft"),
              (26, "26 ft"), (30, "30+ ft")]
TEMP_STOPS = [  # cold blue -> cyan -> green -> yellow -> orange -> red
    (0.00, (30, 60, 180)), (0.25, (30, 150, 220)), (0.45, (60, 190, 150)),
    (0.60, (240, 220, 60)), (0.80, (240, 130, 30)), (1.00, (190, 30, 30)),
]
ICE_STOPS = [  # low ice pale cyan -> blue -> deep navy -> near-total white
    (0.00, (190, 235, 255)), (0.10, (140, 210, 255)), (0.25, (70, 160, 240)),
    (0.45, (35, 100, 215)), (0.65, (20, 55, 150)), (0.85, (10, 25, 90)),
    (1.00, (245, 250, 255)),
]
THICK_STOPS = [  # thinner light blue -> blue -> deeper blue -> purple thicker
    (0.00, (190, 235, 255)), (0.20, (140, 210, 255)), (0.40, (70, 160, 240)),
    (0.60, (90, 80, 200)), (0.80, (130, 50, 180)), (1.00, (90, 20, 140)),
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
                source_line, fmt="{:.0f}", transparent_note=None,
                tick_labels=None):
    """Draw a standalone legend PNG (used by KML ScreenOverlay).
    Returns (W, H). tick_labels = optional [(value, label)] drawn at their
    scale positions (used by the anchored wave gradient)."""
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
    if tick_labels:
        span = (vmax - vmin) or 1.0
        for val, label in tick_labels:
            frac = min(max((val - vmin) / span, 0.0), 1.0)
            x = bx + int(frac * (bw - 1))
            tw = d.textlength(label, font=f_small)
            d.text((min(max(x - tw / 2, 2), W - tw - 2), by + bh + 4),
                   label, font=f_small, fill=(10, 10, 10))
    else:
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
    return W, H


def fmt_ticks(vmin, vmax):
    span = vmax - vmin
    if span <= 12:
        return "{:.1f}"
    return "{:.0f}"


def draw_category_legend(path, title, subtitle, rows, source_line, note=None):
    """Categorical key: every row is (color, label); colors must equal the
    exact raster colors (callers use the same color table). Returns (W, H)."""
    row_h, sw, pad = 26, 30, 14
    W = 640
    H = 96 + row_h * len(rows) + (30 if note else 12)
    img = Image.new("RGBA", (W, H), (255, 255, 255, 235))
    d = ImageDraw.Draw(img)
    f_title, f_body, f_small = _legend_font(22), _legend_font(15), _legend_font(13)
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=2)
    d.text((pad, 8), title, font=f_title, fill=(10, 10, 10))
    d.text((pad, 36), subtitle, font=f_body, fill=(40, 40, 40))
    y = 66
    for color, label in rows:
        if isinstance(color, str):
            h = color.lstrip("#")
            rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        else:
            rgb = tuple(color)
        d.rectangle([pad, y, pad + sw, y + row_h - 6], fill=rgb + (255,),
                    outline=(40, 40, 40))
        d.text((pad + sw + 10, y - 1), label, font=f_body, fill=(10, 10, 10))
        y += row_h
    d.text((pad, y + 4), source_line, font=f_small, fill=(60, 60, 60))
    if note:
        d.text((pad, y + 22), note, font=f_small, fill=(60, 60, 60))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return W, H


def rasterize_polygons(polys, bounds, value_fn):
    """Burn SIGRID-style polygons onto the common canvas.

    value_fn(poly) -> (value, ct_tenths) or (None, ct). Pixels inside a
    polygon take its value; overlapping polygons resolve last-wins and are
    counted. Returns (value_grid, ct_grid, n_overlaps).
    """
    from shapely import contains_xy
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    values = np.full((H, W), np.nan)
    cts = np.zeros((H, W), dtype=np.int8)
    filled = np.zeros((H, W), dtype=bool)
    overlaps = 0
    for poly in polys:
        val, ct = value_fn(poly)
        if val is None:
            continue
        geom = poly["geom"]
        try:
            minx, miny, maxx, maxy = geom.bounds
        except Exception:
            continue
        c0 = int((minx - lon_min) / (lon_max - lon_min) * W)
        c1 = int((maxx - lon_min) / (lon_max - lon_min) * W) + 1
        r0 = int((lat_max - maxy) / (lat_max - lat_min) * H)
        r1 = int((lat_max - miny) / (lat_max - lat_min) * H) + 1
        c0, c1 = max(c0, 0), min(c1, W)
        r0, r1 = max(r0, 0), min(r1, H)
        if r0 >= r1 or c0 >= c1:
            continue
        xs = lon_min + (np.arange(c0, c1) + 0.5) / W * (lon_max - lon_min)
        ys = lat_max - (np.arange(r0, r1) + 0.5) / H * (lat_max - lat_min)
        xx, yy = np.meshgrid(xs, ys)
        try:
            mask = contains_xy(geom, xx, yy)
        except Exception:
            continue
        if not np.any(mask):
            continue
        sub = filled[r0:r1, c0:c1]
        overlaps += int((sub & mask).sum())
        values[r0:r1, c0:c1][mask] = val
        cts[r0:r1, c0:c1][mask] = ct if ct is not None else 0
        sub |= mask
    return values, cts, overlaps


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
        "shoreline_mask": ("assets/great_lakes_watermask.png — shared GSHHG "
                           "v2.3.7 water mask (L1 high-res land, L2 full-res lakes, "
                           "L3 islands, L4 ponds; 3x supersampled, antialiased). "
                           "Overlay alpha is multiplied by this mask, so every "
                           "product shares one shoreline."),
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


COORDS_URL = "https://www.glerl.noaa.gov/data/ice/glicd/grids/coords.zip"


def ensure_coords(raw_dir):
    """GLSEA-family WGS84 LUTs + lake mask (1024 grids). Downloads once per
    runner (gitignored); shared by the temperature and coverage builders."""
    import zipfile
    dest = os.path.join(raw_dir, "coords")
    need = ["1024_latgrid.txt", "1024_longrid.txt", "1024_lake_ids.txt"]
    if all(os.path.exists(os.path.join(dest, n)) for n in need):
        return dest
    os.makedirs(dest, exist_ok=True)
    zpath = os.path.join(raw_dir, "coords.zip")
    download(COORDS_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    return dest


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
                "WDIR_deg": _f(row.get("WDIR")),
                "WSPD_ms": _f(row.get("WSPD")),
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
