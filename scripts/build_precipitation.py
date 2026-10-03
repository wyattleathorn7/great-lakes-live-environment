"""Pipeline P1 — LIVE PRECIPITATION (independent).

Source: NOAA NCEP MRMS MergedReflectivityQC (quality-controlled composite
reflectivity), operational NSSL science distributed by NCEP, new files
about every 2 minutes. Professional + credible: the same QC mosaic NWS
forecasters use.

Why QC instead of raw NEXRAD N0Q: the unfiltered N0Q mosaic passes
non-precipitation echoes straight through — nighttime anomalous
propagation and migrating birds/insects render as bright-green rings and
blobs centered on radar sites, sitting in clear air where the cloud
layer (and the sky) show nothing. Verified 2026-10-02: N0Q painted
~16% of the basin while MRMS QC showed clear air at the same cells and
kept only real echoes (max 34.5 dBZ, eastern lakes). QC filtering keeps
rain, snow, and thunderstorms in all seasons and drops the rest.

Method: list the MRMS directory, take the newest file, decode its dBZ
grid (eccodes), bin the basin window onto the exact common canvas
(lon -93..-73.5, lat 40.5..49.5, 1800x1175, EPSG:4326), and render the
NWS-family palette (cyan/blue = Very Light, green = Light/Moderate,
yellow/orange = Heavy, red = Severe, magenta/purple = Intense,
white = Extreme) so the live gradient IS real-time source intensity —
never a recolor. No-echo and no-coverage cells stay fully transparent
(alpha 0); nothing is invented where the source has none.

Raster-only GroundOverlay (never a video: Google Earth has no live
video primitive — the overlay refreshes). Google Earth side refreshes
every 30 s (fastest practical cadence: each poll re-fetches the live
KML, and the overlay Icon re-requests its PNG, so a newly published
scan appears within about half a minute; polling faster would only
re-download an unchanged image since the source itself rebuilds about
every 2 minutes). The 30 s poll guarantees no
extra staleness is added client-side.

Exit codes: 0 updated (or skipped); 2 source/validation failure
(previous kept); 1 unexpected error.
"""

import datetime as dt
import gzip
import json
import os
import re
import sys
import traceback
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (REPO_ROOT, SITE_DIR, USER_AGENT, base_metadata,
                              load_bounds, now_det_str, promote_stage,
                              read_state, save_png, source_token, stage_dir,
                              write_metadata, write_state)
from geospatial_utils import _legend_font, draw_category_legend  # noqa: F401 (shared helpers)
from gradient_scale import render_rgba

PRODUCT = "precipitation"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Precipitation.kml"
OVERLAY_NAME = "\U0001f327\ufe0f LIVE PRECIPITATION"
SKIP_NOTE = "Turn on/off independently of all other layers."

# Canonical N0Q key colors (match the reference DBZ key image AND the
# IEM WMS rendering): cyan/blue Very Light, green Light/Moderate,
# yellow/orange Heavy, red Severe, magenta/purple Intense, white Extreme.
KEY_ROWS = [
    ((0, 236, 236), "Very Light ~5-15 dBZ — drizzle, very light rain/snow"),
    ((0, 144, 240), "Very Light ~15-20 dBZ — light returns, mist"),
    ((0, 240, 0), "Light ~20-30 dBZ — light rain / snow"),
    ((0, 128, 0), "Moderate ~30-40 dBZ — moderate rain / snow"),
    ((255, 255, 0), "Heavy ~40-45 dBZ — heavy rain"),
    ((255, 165, 0), "Heavy ~45-50 dBZ — very heavy rain"),
    ((255, 0, 0), "Severe ~50-60 dBZ — severe storms, hail possible"),
    ((192, 0, 0), "Severe ~55-60 dBZ — intense downpours, hail likely"),
    ((255, 0, 255), "Intense ~60-70 dBZ — violent storms, large hail"),
    ((139, 0, 255), "Intense ~65-70 dBZ — damaging hail, flash flooding"),
    ((255, 255, 255), "Extreme 70+ dBZ — destructive hail / tornadic debris"),
]

SCALE_HTML = (
    "Precipitation intensity (MRMS quality-controlled composite "
    "reflectivity, dBZ, OBSERVED radar echo): "
    "<b>LOWEST 5 Very Light</b> cyan/blue (drizzle, mist) &rarr; "
    "<b>Light</b> greens (~20-30: light rain/snow) &rarr; "
    "<b>Moderate</b> dark green (~30-40: steady rain/snow) &rarr; "
    "<b>Heavy</b> yellow/orange (~40-50: heavy rain) &rarr; "
    "<b>Severe</b> red (~50-60: severe storms, hail possible) &rarr; "
    "<b>Intense</b> magenta/purple (~60-70: violent storms, large hail) &rarr; "
    "<b>HIGHEST+ 70+ Extreme</b> white (destructive hail, debris). "
    "Same dBZ always shows the same color in every season — rain, snow, "
    "lake-effect snow, thunderstorms. The quality-controlled mosaic "
    "filters non-precipitation echoes (ground clutter, migrating "
    "birds/insects), so clear air stays see-through and only real "
    "precipitation paints.")

# Continuous dBZ anchors through the key colors (band centers exact).
DBZ_STOPS = [
    (5.0, (0, 236, 236)),
    (17.0, (0, 144, 240)),
    (25.0, (0, 240, 0)),
    (35.0, (0, 128, 0)),
    (42.0, (255, 255, 0)),
    (47.0, (255, 165, 0)),
    (55.0, (255, 0, 0)),
    (65.0, (255, 0, 255)),
    (68.0, (139, 0, 255)),
    (72.0, (255, 255, 255)),
    (75.0, (255, 255, 255)),
]

# MRMS QC reflectivity below this is sensor floor/clear — never painted.
# (-99 = observed clear, -999 = no coverage; lowest real echo seen -10.)
ECHO_FLOOR_DBZ = -30.0

MRMS_DIR = "https://mrms.ncep.noaa.gov/2D/MergedReflectivityQC/"
MRMS_RE = re.compile(
    r"MRMS_MergedReflectivityQC_00\.50_(\d{8})-(\d{6})\.grib2\.gz")


def newest_mrms():
    """Newest MRMS QC file (name, scan datetime UTC). Raises on failure."""
    req = urllib.request.Request(MRMS_DIR, headers=USER_AGENT)
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", errors="replace")
    cands = []
    for m in MRMS_RE.finditer(html):
        try:
            scan = dt.datetime.strptime(m.group(1) + m.group(2),
                                        "%Y%m%d%H%M%S")
            scan = scan.replace(tzinfo=dt.timezone.utc)
            cands.append((scan, m.group(0)))
        except ValueError:
            continue
    if not cands:
        raise ValueError("no MRMS QC files in directory listing")
    cands.sort()
    return cands[-1][1], cands[-1][0]


def read_mrms_grid(path, bounds):
    """Decode MRMS QC GRIB2 and return dBZ field on the common canvas."""
    from eccodes import (codes_get, codes_get_values, codes_grib_new_from_file,
                         codes_release)
    with open(path, "rb") as f:
        g = codes_grib_new_from_file(f)
        if g is None:
            raise ValueError("no GRIB message in file")
        try:
            nx = codes_get(g, "Nx")
            ny = codes_get(g, "Ny")
            lat0 = codes_get(g, "latitudeOfFirstGridPointInDegrees")
            lon0 = codes_get(g, "longitudeOfFirstGridPointInDegrees")
            lat1 = codes_get(g, "latitudeOfLastGridPointInDegrees")
            lon1 = codes_get(g, "longitudeOfLastGridPointInDegrees")
            vals = np.asarray(codes_get_values(g), dtype=float).reshape(ny, nx)
        finally:
            codes_release(g)
    lon0 = lon0 - 360.0 if lon0 > 180 else lon0
    lon1 = lon1 - 360.0 if lon1 > 180 else lon1
    dx = (lon1 - lon0) / (nx - 1)
    dy = (lat0 - lat1) / (ny - 1)  # rows run north -> south
    if not (0.005 < dx < 0.02 and 0.005 < dy < 0.02):
        raise ValueError(f"unexpected MRMS grid step {dx}/{dy}")
    # Basin window slice (source ~1 km vs ~1.2 km canvas: cheap + exact).
    j0 = max(int((bounds["lon_min"] - lon0) / dx) - 1, 0)
    j1 = min(int((bounds["lon_max"] - lon0) / dx) + 2, nx)
    i1 = max(int((lat0 - bounds["lat_max"]) / dy) - 1, 0)
    i0 = min(int((lat0 - bounds["lat_min"]) / dy) + 2, ny)
    if j1 <= j0 or i0 <= i1:
        raise ValueError("basin window outside MRMS domain")
    win = vals[i1:i0, j0:j1]
    lons = lon0 + (np.arange(j0, j1) + 0.5) * dx
    lats = lat0 - (np.arange(i1, i0) + 0.5) * dy
    xx, yy = np.meshgrid(lons, lats)
    from geospatial_utils import bin_to_canvas, canvas_indices
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    rows, cols, valid = canvas_indices(yy, xx, bounds)
    wflat = np.ascontiguousarray(win).ravel()
    mask = valid & np.isfinite(wflat)
    field, _c = bin_to_canvas(rows, cols, wflat, mask,
                              (H, W), splat_radius=1)
    field = np.where(np.isfinite(field) & (field >= ECHO_FLOOR_DBZ),
                     field, np.nan)
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
    # Source id: newest MRMS QC file on NCEP (2-minute cadence). The
    # directory listing is the cheap pre-check: only a new filename
    # triggers the ~250 KB download + decode.
    try:
        fname, scan_dt = newest_mrms()
    except Exception as e:
        print(f"[{PRODUCT}] NO FILE AVAILABLE (keeping previous): {e}")
        return 2
    source_id = f"mrms-qc-{scan_dt.strftime('%Y%m%d-%H%M%S')}"
    prev = read_state(PRODUCT)
    if (prev.get("source_id") == source_id
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png"))
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE))):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        return _refresh_kml()
    try:
        return _build(fname, scan_dt, source_id)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _refresh_kml():
    try:
        refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                             CONFIG["title"], SKIP_NOTE,
                             CONFIG["refresh_interval_seconds"])
    except Exception as e:
        print(f"[{PRODUCT}] WARNING: KML refresh failed: {e}")
    return 0


def _build(fname, scan_dt, source_id):
    bounds = load_bounds()
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    os.makedirs(stage_prod, exist_ok=True)
    W, H = bounds["canvas_width"], bounds["canvas_height"]

    raw_path = os.path.join(RAW_DIR, "mrms_qc_current.grib2.gz")
    os.makedirs(RAW_DIR, exist_ok=True)
    req = urllib.request.Request(MRMS_DIR + fname, headers=USER_AGENT)
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read()
    if len(data) < 50_000:
        raise ValueError(f"MRMS download too small ({len(data)} bytes)")
    with open(raw_path, "wb") as f:
        f.write(data)
    gz_path = os.path.join(RAW_DIR, "mrms_qc_current.grib2")
    with gzip.open(raw_path, "rb") as zin, open(gz_path, "wb") as zout:
        zout.write(zin.read())
    if os.path.getsize(gz_path) < 100_000:
        raise ValueError("MRMS GRIB2 too small after gunzip")

    field = read_mrms_grid(gz_path, bounds)
    okv = np.isfinite(field)
    n_valid = int(okv.sum())
    cur_max = float(field[okv].max()) if n_valid else float("nan")
    print(f"[{PRODUCT}] MRMS QC {scan_dt:%Y-%m-%d %H:%M:%SZ}: "
          f"echo={n_valid} ({100.0 * n_valid / (W * H):.1f}% of canvas), "
          f"max={cur_max:.1f} dBZ")
    # Clear-sky over the basin is VALID (transparent raster is honest, not
    # a failure) — same semantics as cloud_cover on a clear day.

    rgba = render_rgba(field, list(DBZ_STOPS), bounds["overlay_alpha"])
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    if n_opaque != n_valid:
        raise ValueError("render/field transparency mismatch")

    data_time_utc = scan_dt.strftime("%Y-%m-%d %H:%M UTC")
    subtitle = (f"MRMS QC composite reflectivity (dBZ, ~2-min mosaic)  |  "
                f"{data_time_utc}")
    lw, lh = draw_category_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        KEY_ROWS,
        f"Source: MRMS MergedReflectivityQC {scan_dt:%Y-%m-%d %H:%MZ}  |  "
        f"Processed {now_det_str()}",
        note="No echo = transparent (no precipitation).")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc=fname,
        units="dBZ (display = observed dBZ)",
        source_resolution="~1 km MRMS QC grid (7000x3500 CONUS), 2-minute files; "
                          "basin window mean-binned to the 1800x1175 canvas",
        color_min=5.0, color_max=75.0, color_units="dBZ",
        missing_data_treatment=("only MRMS cells at/above the echo floor paint; "
                                "clear-air (-99) and no-coverage (-999) render "
                                "fully transparent (alpha 0); full basin "
                                "rectangle, no shoreline cut; never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = SCALE_HTML
    meta["model_cycle"] = f"MRMS QC {scan_dt:%Y-%m-%d %H:%M:%SZ}"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["mrms_file"] = fname
    meta["stats"] = {"opaque_pixels": n_opaque,
                     "transparent_pixels": int(W * H - n_opaque)}
    token = meta["source_version"]
    # Folder: key image FIRST (top), written gradient explanation
    # UNDERNEATH — exactly the requested layout.
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p><img src=\"{legend_src(token)}\" width=\"600\" "
        f"alt=\"precipitation gradient key\"></p>"
        f"<p>{SCALE_HTML}</p>"
        f"<p>How to read it in any season. <b>Light</b> greens are light "
        f"rain in summer — and light snow in winter. <b>Moderate</b> dark "
        f"green is steady rain or steady snow, including lake-effect bands. "
        f"<b>Heavy</b> yellow/orange is heavy rain (or very heavy snow "
        f"rates). <b>Severe</b> red means severe thunderstorms with hail "
        f"possible; <b>Intense</b> magenta/purple means violent storms with "
        f"large hail and flash flooding; <b>Extreme</b> white is reserved "
        f"for the most destructive echoes. Radar sees precipitation "
        f"particles, not air temperature, so one scale covers rain, snow, "
        f"and thunderstorms all year. Where you see through the overlay, "
        f"there is no precipitation right now.</p>"
        f"<p><b>Variable:</b> {CONFIG['variable']}<br/>"
        f"<b>Units:</b> dBZ<br/>"
        f"<b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> QC files about every 2 minutes; Google Earth "
        f"re-polls every 30 seconds<br/>"
        f"<b>Data time:</b> {data_time_utc}<br/>"
        f"<b>Source version:</b> {source_id}<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Provenance:</b> <a href=\"{CONFIG['source_url']}\">NOAA NCEP "
        f"MRMS</a></p>")
    meta["folder_html"] = folder_html
    write_metadata(stage_prod, meta)
    block = legend_block(f"{PRODUCT}/legend.png", token, SCALE_HTML)
    outs = live_out_dirs(stage, KML_FILE)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        folder=(CONFIG["title"], folder_html),
        out_dirs=outs["live"],
        meta=meta)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "source_id": source_id,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def legend_src(token=None):
    from build_kml import pages_base
    base = pages_base()
    v = f"?v={token}" if token else ""
    return f"{base}/{PRODUCT}/legend.png{v}"


if __name__ == "__main__":
    sys.exit(main())
