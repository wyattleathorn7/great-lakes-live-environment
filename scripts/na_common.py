"""Shared helpers for the four North American live layers.

All NA products use config/north_america_bounds.json (NOT the Great Lakes
rectangle) via scripts/na_bounds.py. Common canvas 3300x1600, EPSG:4326.
"""

import os
import urllib.request

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from na_bounds import load_na_bounds

UA = {"User-Agent": "great-lakes-live-environment/1.0 (north-america)"}


def wms_getmap(url, dest, timeout=180):
    """GET a WMS GetMap PNG -> dest (atomic). Returns (width, height).

    Raises on non-PNG response (WMS XML exceptions fail the magic check).
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"WMS non-PNG ({len(data)}B head={data[:120]!r}) url={url[:160]}")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    with Image.open(dest) as im:
        return im.size


def load_rgba(path, expect_wh=None):
    im = Image.open(path).convert("RGBA")
    if expect_wh is not None and im.size != expect_wh:
        raise ValueError(f"WMS size {im.size} != canvas {expect_wh}")
    return np.array(im)


def bin_points_to_canvas(vals, lats, lons, bounds, agg="mean"):
    """Bin scattered WGS84 points onto the NA canvas. Returns (grid, count).

    grid holds agg value per cell, NaN where no points. Separate count
    array tracks support so valid-zero (e.g. 0% cloud, calm) is never
    confused with missing.
    """
    h, w = bounds["canvas_height"], bounds["canvas_width"]
    col = ((np.asarray(lons) - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"]) * w).astype(int)
    row = ((bounds["lat_max"] - np.asarray(lats)) / (bounds["lat_max"] - bounds["lat_min"]) * h).astype(int)
    vals = np.asarray(vals, dtype=float)
    ok = np.isfinite(vals) & (col >= 0) & (col < w) & (row >= 0) & (row < h)
    col, row, vals = col[ok], row[ok], vals[ok]
    grid = np.full((h, w), np.nan)
    count = np.zeros((h, w), dtype=np.int32)
    if vals.size == 0:
        return grid, count
    idx = row * w + col
    order = np.argsort(idx)
    idx_s, vals_s = idx[order], vals[order]
    bounds_idx = np.concatenate([[0], np.nonzero(np.diff(idx_s))[0] + 1, [idx_s.size]])
    cells = idx_s[bounds_idx[:-1]]
    if agg == "mean":
        sums = np.add.reduceat(vals_s, bounds_idx[:-1])
        n = np.diff(bounds_idx)
        grid.ravel()[cells] = sums / n
    elif agg == "max":
        mx = np.maximum.reduceat(vals_s, bounds_idx[:-1])
        grid.ravel()[cells] = mx
    else:
        raise ValueError(agg)
    count.ravel()[cells] = np.diff(bounds_idx)
    return grid, count


def fill_pinholes(grid, radius=2):
    """NaN-aware pinhole fill for coarse-model -> fine-canvas resampling.

    Fills ONLY NaN cells within `radius` cells of valid data (interior
    resampling gaps); the true domain edge and large unobserved areas
    stay NaN (never extrapolated, never fabricated).
    """
    out = grid.copy()
    valid = np.isfinite(out)
    if valid.all() or not valid.any():
        return out
    for _ in range(radius):
        missing = ~np.isfinite(out)
        # neighbor-mean over currently-valid cells (4-neighbourhood)
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_valid = np.zeros_like(valid)
            shifted_val = np.zeros_like(out)
            src = valid[max(0, -dy):out.shape[0] - max(0, dy),
                        max(0, -dx):out.shape[1] - max(0, dx)]
            shifted_valid[max(0, dy):out.shape[0] - max(0, -dy),
                          max(0, dx):out.shape[1] - max(0, -dx)] = src
            shifted_val[shifted_valid] = out[max(0, -dy):out.shape[0] - max(0, dy),
                                            max(0, -dx):out.shape[1] - max(0, dx)][
                shifted_valid[max(0, dy):out.shape[0] - max(0, -dy),
                              max(0, dx):out.shape[1] - max(0, -dx)]]
            acc += np.where(shifted_valid, shifted_val, 0.0)
            cnt += shifted_valid.astype(float)
        fillable = missing & (cnt > 0)
        if not fillable.any():
            break
        out[fillable] = acc[fillable] / cnt[fillable]
    return out


def gdps_wcs_grid(coverage_id, bounds, raw_dir, timeout=300):
    """Fetch ECCC GDPS WCS GetCoverage GeoTIFF for the NA bbox.

    Returns (vals, lats, lons) in source units, or raises. GDPS is the
    documented last-resort fallback: callers fill ONLY cells with no
    HRRR/RRFS/GOES value and record the fill count honestly.
    """
    import urllib.parse
    lon0, lat0, lon1, lat1 = (bounds["lon_min"], bounds["lat_min"],
                              bounds["lon_max"], bounds["lat_max"])
    q = (f"service=WCS&version=2.0.1&request=GetCoverage&coverageId={coverage_id}"
         f"&format=image%2Ftiff&subset=lat({lat0},{lat1})&subset=long({lon0},{lon1})")
    url = f"https://geo.weather.gc.ca/geomet?{q}"
    dest = os.path.join(raw_dir, f"na_gdps_{coverage_id}.tif")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if data[:4] not in (b"II*\x00", b"MM\x00*"):
        raise ValueError(f"WCS non-TIFF ({len(data)}B head={data[:160]!r})")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    import rasterio
    with rasterio.open(dest) as ds:
        vals = ds.read()
        west, south, east, north = ds.bounds
        h, w = vals.shape[1], vals.shape[2]
        lons = west + (np.arange(w) + 0.5) * (east - west) / w
        lats = north - (np.arange(h) + 0.5) * (north - south) / h
        lo2, la2 = np.meshgrid(lons, lats)
    return vals, la2, lo2


def sanitize_model_vals(vals, lo=-100.0, hi=400.0):
    """Map GRIB missing flags (9999, -9999, huge) and unphysical values
    to NaN. Keeps valid-zero distinct from missing."""
    v = np.asarray(vals, dtype=float)
    bad = (~np.isfinite(v)) | (v == 9999.0) | (v == -9999.0) | (v < lo) | (v > hi)
    return np.where(bad, np.nan, v)


def eccodes_read_idx_ranges(idx_text, wanted, step="0"):
    """Parse NOMADS .idx text for wanted [(shortName, level_substr)] at step.

    Returns {shortName: (start, end)} byte ranges. Raises if missing.
    """
    lines = idx_text.splitlines()
    anl = [l for l in lines if f":{step}:" in l or ":anl:" in l]
    found = {}
    for short, level in wanted:
        for l in anl:
            p = l.split(":")
            if len(p) > 4 and p[3] == short and level in l:
                a = int(p[1])
                j = lines.index(l) + 1
                if j >= len(lines):
                    continue
                b = int(lines[j].split(":")[1])
                if b > a:
                    found[short] = (a, b)
                    break
        if short not in found:
            raise ValueError(f"{short}/{level} not in .idx")
    return found


def fetch_http(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()
