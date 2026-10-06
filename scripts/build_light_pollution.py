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
    (0.0, (6, 8, 22)),        # B1 pristine: near-black
    (5.0, (64, 70, 84)),      # B2 true dark: slate gray (reads apart from black)
    (10.0, (30, 80, 190)),    # B3 rural: strong blue
    (16.0, (35, 175, 135)),   # B4 transition: teal-green
    (24.0, (150, 190, 70)),   # B5 suburban: olive yellow-green
    (36.0, (235, 195, 55)),   # B6 bright suburban: gold
    (52.0, (240, 130, 30)),   # B7 transition: orange
    (72.0, (220, 45, 55)),    # B8 city: red
    (82.0, (245, 150, 120)),  # B8/B9 shoulder: salmon (fast ramp to white)
    (90.0, (250, 232, 222)),  # B9 inner city: near-white well before the top
    (100.0, (245, 245, 240)), # B9 extreme core: white
]
LABELS = [(0.0, "LOWEST 0 pristine"), (25.0, "25"), (50.0, "50"),
          (75.0, "75"), (100.0, "HIGHEST+ 100 urban")]
# The Bortle dark-sky scale (John E. Bortle, Sky & Telescope, Feb 2001):
# nine classes from pristine to inner-city night skies. Titles + NELM
# (naked-eye limiting magnitude) are Bortle's own; SQM ranges are the
# widely-circulated third-party retrofit approximations (Bortle's original
# article contains no SQM values) — documented as approximate here and in
# metadata, never presented as calibration.
# Index breakpoints are log-spaced (100*(10^(k/9)-1)/9): brightness is
# log-distributed, so uniform splits would crush every town into the
# darkest band (the classic night-lights binning mistake). Short labels
# are condensed Bortle titles that fit the even key cells.
BORTLE_CLASSES = [
    (1, "Excellent dark-sky site", "Excellent", "7.6–8.0", "21.76–22.0"),
    (2, "Typical truly dark site", "True dark", "7.1–7.5", "21.60–21.75"),
    (3, "Rural sky", "Rural", "6.6–7.0", "21.30–21.59"),
    (4, "Rural/suburban transition", "Rural/sub.", "6.1–6.5", "20.40–21.29"),
    (5, "Suburban sky", "Suburban", "5.6–6.0", "19.10–20.39"),
    (6, "Bright suburban sky", "Bright sub.", "~5.5", "18.50–19.09"),
    (7, "Suburban/urban transition", "Sub./urban", "5.0", "18.00–18.49"),
    (8, "City sky", "City", "4.5", "<18.00"),
    (9, "Inner-city sky", "Inner city", "≤4.0", "<18.00"),
]


def bortle_breakpoints():
    """Log-spaced 0-100 index boundaries between the 9 Bortle classes."""
    return [100.0 * (10.0 ** (k / 9.0) - 1.0) / 9.0 for k in range(1, 9)]


def draw_bortle_key(path, title, subtitle, stops, source_line):
    """Nine even cells, one per Bortle class, each painted with the
    gradient color at its class midpoint (guaranteed raster<->key match).
    Class numbers on row 1, condensed Bortle titles on row 2 — even
    spacing, no overlaps. Returns (W, H)."""
    from PIL import Image, ImageDraw
    from geospatial_utils import _legend_font
    from gradient_scale import color_for
    bounds = bortle_breakpoints()
    edges = [0.0] + bounds + [100.0]
    W, H = 640, 252
    img = Image.new("RGBA", (W, H), (255, 255, 255, 235))
    d = ImageDraw.Draw(img)
    f_title, f_body, f_small = _legend_font(22), _legend_font(15), _legend_font(13)
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=2)
    d.text((14, 8), title, font=f_title, fill=(10, 10, 10))
    d.text((14, 36), subtitle, font=f_body, fill=(40, 40, 40))
    bx, by, bw, bh = 14, 66, W - 28, 34
    cw = bw / 9.0
    for i, cls in enumerate(BORTLE_CLASSES):
        lo, hi = edges[i], edges[i + 1]
        mid = (lo + hi) / 2.0
        rgb = color_for(mid, stops)
        x0 = bx + i * cw
        d.rectangle([x0, by, x0 + cw, by + bh], fill=rgb + (255,),
                    outline=(40, 40, 40))
        num = f"Class {cls[0]}"
        tw = d.textlength(num, font=f_small)
        d.text((x0 + (cw - tw) / 2, by + bh + 4), num, font=f_small,
               fill=(10, 10, 10))
        sw = d.textlength(cls[2], font=f_small)
        d.text((x0 + (cw - sw) / 2, by + bh + 22), cls[2], font=f_small,
               fill=(60, 60, 60))
    d.text((14, H - 44), source_line, font=f_small, fill=(60, 60, 60))
    d.text((14, H - 26), "Bortle classes: log-spaced index bands; "
           "approximate relative classification, not SQM-measured.",
           font=f_small, fill=(60, 60, 60))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return W, H


SCALE_HTML = ("Artificial night-sky brightness on the <b>Bortle dark-sky "
              "scale Classes 1–9</b> (John E. Bortle, Sky & Telescope 2001; "
              "keyed from the NASA VIIRS Black Marble annual composite): "
              "<b>LOWEST Class 1 Excellent dark-sky site</b> near-black "
              "(NELM 7.6–8.0) &rarr; Class 2 True dark slate-gray &rarr; "
              "Class 3 Rural blue &rarr; Class 4 Rural/suburban teal-green "
              "&rarr; Class 5 Suburban olive &rarr; Class 6 Bright suburban "
              "gold &rarr; Class 7 Suburban/urban orange &rarr; Class 8 "
              "City-sky red &rarr; "
              "<b>HIGHEST+ Class 9 Inner-city sky</b> near-white "
              "(NELM ≤4.0). Classes 1–5 use widely separated hues so dark "
              "skies stay readable. Class "
              "boundaries are log-spaced on the brightness index (3.2 / 7.4 "
              "/ 12.8 / 19.8 / 28.8 / 40.5 / 55.5 / 74.9), shaped like the "
              "published SQM retrofit table — an approximate relative "
              "classification from composite imagery, not a calibrated "
              "SQM measurement. Same sky always shows the same class.")


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
    # NaN-aware broad bloom for skyglow. Kept deliberately tight
    # (radius 6, 2 passes, ~4 km sigma): enough to render point-like city
    # cores as readable metro glow, small enough to stay crisp.
    cur = np.where(have, base, 0.0)
    w = have.astype(float)
    radius, passes = 6, 2
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
    subtitle = (f"Night-sky brightness (Bortle Classes 1-9, Black Marble "
                f"annual)  |  {data_time_utc}")

    def _bortle_legend(legend_path):
        return draw_bortle_key(
            legend_path, CONFIG["title"], subtitle, STOPS,
            f"Source: NASA VIIRS Black Marble 2016 via GIBS  |  "
            f"Processed {now_det_str()}")

    return finish(
        PRODUCT, CONFIG, KML_FILE, OVERLAY_NAME, field, STOPS, LABELS,
        "Bortle class", subtitle,
        f"Source: NASA VIIRS Black Marble 2016 via GIBS  |  "
        f"Processed {now_det_str()}", SCALE_HTML,
        [CONFIG["what"], CONFIG["field"],
         "Composite checked hourly via tile probe; republishes only when "
         "NASA publishes a newer annual composite."],
        {"model_cycle": f"Black-Marble-2016 probe {phash}",
         "stats": {"composite_year": "2016", "probe_hash": phash,
                   "data_nature": "annual composite visualization"},
         "fields": {
             "bortle_classes": [
                 {"class": c[0], "title": c[1], "nelm": c[3],
                  "sqm_approx": c[4]} for c in BORTLE_CLASSES],
             "bortle_index_breakpoints": bortle_breakpoints(),
             "bortle_methodology": (
                 "Approximate relative Bortle classification. Class titles "
                 "and NELM are Bortle's own (Sky & Telescope, Feb 2001); "
                 "SQM ranges are the widely-circulated third-party retrofit "
                 "approximations (Bortle's original contains no SQM values). "
                 "Index breakpoints are log-spaced "
                 "(100*(10^(k/9)-1)/9), shaped like the published SQM "
                 "retrofit table, because night-light brightness is "
                 "log-distributed. NOT a calibrated SQM measurement; the "
                 "input is GIBS composite-imagery luminance, not "
                 "nW/cm2/sr radiance.")}},
        source_id, data_time_utc, "n/a (WMTS tiles)",
        "relative brightness index 0-100 (display; tile-background floor "
        "subtraction + skyglow bloom + square-root stretch of the "
        "published composite visualization; monotonic, ordering "
        "preserved)",
        "NASA GIBS 500m WMTS tiles (level 6, 4.5-degree) mosaicked to the common canvas",
        "tile-background floor subtraction, NaN-aware skyglow bloom "
        "(radius 10 x 3 passes), then square-root 100*sqrt(bloomed/ref) "
        "mapping against the 99.99th percentile (monotonic perceptual "
        "mapping so metro glow reads at basin scale); full basin "
        "rectangle, no shoreline cut; untiled canvas pixels transparent; "
        "never zero-filled.",
        custom_legend=_bortle_legend)


if __name__ == "__main__":
    sys.exit(main())
