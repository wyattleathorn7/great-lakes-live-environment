"""Pipeline V4 — LIVE LIGHT POLLUTION (independent, slow-varying composite).

Source: NASA VIIRS Black Marble annual nighttime-lights composite
(Suomi NPP VIIRS Day/Night Band, Román et al.) served as public WMTS
tiles by NASA GIBS. Ten level-6 tiles covering the basin are mosaicked
and mapped onto the common canvas as a RELATIVE brightness index 0-100
(linear mapping of the published composite visualization — documented
as imagery-derived, never fabricated radiance units). The composite is
annual (currently 2016 on GIBS), so the hourly workflow check is a
single probe-tile content hash: the raster republishes only when NASA
publishes a newer composite year. Freshness wording is LATEST AVAILABLE
COMPOSITE, never real-time. FIXED 0-100 continuous spectrum -> FULL
BASIN RECTANGLE (lon -93..-73.5, lat 40.5..49.5, the exact LIVE LEAF
COLOR footprint) -> key + metadata -> Folder live KML + stable entry
KML. Continuous environmental surface (no preserve points). Distinguishes
artificial glow from natural night-sky brightness in the folder text.
Exit 0/2/1 per contract.
"""

import hashlib
import io
import json
import os
import sys
import traceback
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_field import (SKIP_NOTE, finish, refresh_kml, should_skip)
from geospatial_utils import (REPO_ROOT, SITE_DIR, USER_AGENT, bin_to_canvas,
                              canvas_indices, load_bounds, now_det_str)

PRODUCT = "light_pollution"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
KML_FILE = "Great_Lakes_Live_Light_Pollution.kml"
OVERLAY_NAME = "💡 LIVE LIGHT POLLUTION"

TMS = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/VIIRS_Black_Marble/default/default/500m"
# GIBS 500m TileMatrixSet geometry (from WMTSCapabilities): level 6 is
# 80 cols x 40 rows of 512 px tiles -> 4.5 x 4.5 degrees per tile
# (~113 px/deg, matching the common canvas). Basin lon -93..-73.5 needs
# cols 19..23, lat 40.5..49.5 needs rows 9..10.
LEVEL = 6
NW, NH = 80, 40
COLS = (19, 20, 21, 22, 23)
ROWS = (9, 10)
PROBE = (21, 9)

STOPS = [
    (0.0, (5, 8, 20)),        # pristine dark: near-black
    (10.0, (16, 52, 140)),    # dark site: navy
    (25.0, (20, 110, 200)),   # rural: blue
    (40.0, (40, 190, 160)),   # fringe: teal
    (55.0, (120, 200, 90)),   # suburban: green
    (70.0, (245, 215, 50)),   # bright suburb: yellow
    (85.0, (245, 130, 25)),   # urban: orange
    (100.0, (245, 245, 240)), # urban core: near-white
]
LABELS = [(0.0, "LOWEST 0 pristine"), (25.0, "25"), (50.0, "50"),
          (75.0, "75"), (100.0, "HIGHEST+ 100 urban")]
SCALE_HTML = ("Artificial night-sky brightness as a relative index 0-100 "
              "(NASA VIIRS Black Marble annual composite, fixed absolute "
              "scale): <b>LOWEST 0</b> near-black pristine dark sky &rarr; "
              "navy &rarr; blue rural &rarr; teal fringe &rarr; green "
              "suburban &rarr; yellow &rarr; orange urban &rarr; "
              "<b>HIGHEST+ 100</b> near-white urban core. Same brightness "
              "always shows the same color. Composite visualization index — "
              "not raw radiance units; natural sky brightness (moonlight, "
              "airglow) is not separated per-pixel, city glow dominates the "
              "signal.")


def _tile(col, row, timeout=90):
    from PIL import Image
    url = f"{TMS}/{LEVEL}/{row}/{col}.png"
    req = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data, np.array(Image.open(io.BytesIO(data)).convert("RGB"),
                          dtype=float)


def _probe_hash():
    data, _arr = _tile(*PROBE)
    return hashlib.sha256(data).hexdigest()[:16]


def _mosaic():
    """Mosaic tiles -> canvas-space relative-brightness index 0-100.

    Three documented display steps turn the point-like composite into a
    readable basin-scale environmental surface (all monotonic, ordering
    preserved):
    1. Floor subtraction: GIBS pads no-light areas at a uniform navy
       (~luminance 5.8); the mosaic's 0.5th percentile estimates that
       floor so pristine dark reads 0.
    2. Skyglow bloom: NaN-aware broad blur (radius 10, 3 passes,
       ~9 km sigma) spreads point-like city cores into the diffuse
       metro glow every published light-pollution map shows — the same
       display-smoothing family the repo's solar product uses.
    3. Square-root perceptual map to 0-100 against the 99.99th
       percentile, so suburbs and town glow own mid-scale while the
       brightest cores still saturate.
    """
    from gradient_scale import _box_sum
    bounds = load_bounds()
    lats_list, lons_list, lum_list = [], [], []
    for col in COLS:
        for row in ROWS:
            _data, arr = _tile(col, row)
            H, W = arr.shape[:2]
            # GIBS EPSG:4326 tile geo-registration at this level.
            lon0 = -180.0 + col * 360.0 / NW
            lon1 = -180.0 + (col + 1) * 360.0 / NW
            lat1 = 90.0 - row * 180.0 / NH
            lat0 = 90.0 - (row + 1) * 180.0 / NH
            yy, xx = np.mgrid[0:H, 0:W]
            lons = lon0 + (xx + 0.5) / W * (lon1 - lon0)
            lats = lat1 - (yy + 0.5) / H * (lat1 - lat0)
            lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1]
                   + 0.114 * arr[:, :, 2])
            lats_list.append(lats.ravel())
            lons_list.append(lons.ravel())
            lum_list.append(lum.ravel())
    lats = np.concatenate(lats_list)
    lons = np.concatenate(lons_list)
    lum = np.concatenate(lum_list)
    rows, cols, valid = canvas_indices(lats, lons, bounds)
    shape = (bounds["canvas_height"], bounds["canvas_width"])
    # Tiles sit at ~canvas resolution, so single-pixel binning leaves
    # polka-dot holes; splat_radius=1 closes them with neighborhood
    # averaging (display smoothing only, per repo convention).
    lum_field, _counts = bin_to_canvas(rows, cols, lum, valid, shape,
                                       splat_radius=1)
    have = np.isfinite(lum_field)
    floor = float(np.percentile(lum_field[have], 0.5)) if have.any() else 0.0
    base = np.where(have, np.maximum(lum_field - floor, 0.0), np.nan)
    # NaN-aware broad bloom for skyglow.
    cur = np.where(have, base, 0.0)
    w = have.astype(float)
    radius, passes = 10, 3
    for _ in range(passes):
        sw = _box_sum(w, radius)
        sv = _box_sum(cur, radius)
        with np.errstate(invalid="ignore", divide="ignore"):
            cur = np.where(sw > 0, sv / np.where(sw > 0, sw, 1.0), 0.0)
        w = np.minimum(sw / ((2 * radius + 1) ** 2), 1.0)
    bloomed = np.where(have, cur, np.nan)
    pos = bloomed[have & (bloomed > 0)]
    ref = float(np.percentile(pos, 99.99)) if pos.size else 1.0
    ref = ref if ref > 0 else 1.0
    field = np.where(have, 100.0 * np.sqrt(
        np.clip(bloomed / ref, 0.0, 1.0)), np.nan)
    return field


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    try:
        phash = _probe_hash()
    except Exception as e:
        print(f"[{PRODUCT}] PROBE FAILED (keeping previous): {e}")
        return 2
    source_id = f"gibs-black-marble-2016-{phash}"
    if should_skip(PRODUCT, source_id):
        print(f"[{PRODUCT}] composite unchanged ({phash}); keeping raster.")
        return refresh_kml(PRODUCT, KML_FILE, OVERLAY_NAME,
                           CONFIG["title"],
                           CONFIG["refresh_interval_seconds"])
    try:
        return _build(source_id, phash)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(source_id, phash):
    field = _mosaic()
    ok = np.isfinite(field)
    if int(ok.sum()) < 50_000:
        raise ValueError(f"too few valid canvas cells ({int(ok.sum())})")
    data_time_utc = "2016 annual composite"
    subtitle = (f"Artificial night brightness (index 0-100, Black Marble "
                f"annual)  |  {data_time_utc}")
    return finish(
        PRODUCT, CONFIG, KML_FILE, OVERLAY_NAME, field, STOPS, LABELS,
        "index", subtitle,
        f"Source: NASA VIIRS Black Marble 2016 via GIBS  |  "
        f"Processed {now_det_str()}", SCALE_HTML,
        [CONFIG["what"], CONFIG["field"],
         "Composite checked hourly via tile probe; republishes only when "
         "NASA publishes a newer annual composite."],
        {"model_cycle": f"Black-Marble-2016 probe {phash}",
         "stats": {"composite_year": "2016", "probe_hash": phash,
                   "data_nature": "annual composite visualization"}},
        source_id, data_time_utc, "n/a (WMTS tiles)",
        "relative brightness index 0-100 (display; tile-background floor "
        "subtraction + skyglow bloom + square-root stretch of the "
        "published composite visualization; monotonic, ordering "
        "preserved)",
        "NASA GIBS 500m WMTS tiles (level 5) mosaicked to the common canvas",
        "tile-background floor subtraction, NaN-aware skyglow bloom "
        "(radius 10 x 3 passes), then square-root 100*sqrt(bloomed/ref) "
        "mapping against the 99.99th percentile (monotonic perceptual "
        "mapping so metro glow reads at basin scale); full basin "
        "rectangle, no shoreline cut; untiled canvas pixels transparent; "
        "never zero-filled.")


if __name__ == "__main__":
    sys.exit(main())
