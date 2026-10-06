"""Pipeline V4 — LIVE LIGHT POLLUTION (independent, slow-varying composite).

Source: NASA VIIRS Black Marble annual nighttime-lights composite
(Suomi NPP VIIRS Day/Night Band, Román et al.) served as public WMTS
tiles by NASA GIBS. Eight level-5 tiles covering the basin are mosaicked
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
LEVEL = 5
# Level-5 EPSG:4326 grid: 64 cols x 32 rows of 512 px tiles.
COLS = (15, 16, 17, 18)
ROWS = (7, 8)
PROBE = (16, 8)

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
    """Mosaic tiles -> canvas-space relative-brightness index 0-100."""
    bounds = load_bounds()
    lats_list, lons_list, lum_list = [], [], []
    for col in COLS:
        for row in ROWS:
            _data, arr = _tile(col, row)
            H, W = arr.shape[:2]
            # GIBS EPSG:4326 tile geo-registration at this level.
            lon0 = -180.0 + col * 360.0 / 64.0
            lon1 = -180.0 + (col + 1) * 360.0 / 64.0
            lat1 = 90.0 - row * 180.0 / 32.0
            lat0 = 90.0 - (row + 1) * 180.0 / 32.0
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
    idx = 100.0 * np.clip(lum / 255.0, 0.0, 1.0)
    # Tiles sit at ~canvas resolution, so single-pixel binning leaves
    # polka-dot holes; splat_radius=1 closes them with neighborhood
    # averaging (display smoothing only, per repo convention).
    field, _counts = bin_to_canvas(rows, cols, idx, valid, shape,
                                   splat_radius=1)
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
        "relative brightness index 0-100 (display; linear map of the "
        "published composite visualization)",
        "NASA GIBS 500m WMTS tiles (level 5) mosaicked to the common canvas",
        "linear 100*luminance/255 mapping of the published composite; "
        "full basin rectangle, no shoreline cut; untiled canvas pixels "
        "transparent; never zero-filled.")


if __name__ == "__main__":
    sys.exit(main())
