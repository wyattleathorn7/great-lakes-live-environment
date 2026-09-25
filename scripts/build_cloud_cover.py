"""Pipeline C1 — LIVE CLOUD COVER (independent).

NOAA/NCEP HRRR 3 km TCDC entire-atmosphere analysis (hourly cycles,
15-min subhourly steps available) -> validate % -> bin native Lambert
grid onto the common canvas -> NO shoreline cut (basin rectangle:
lon -93..-73.5, lat 40.5..49.5, land stays visible) -> spectrum
gradient dark-blue (thin) to dark-purple (overcast), clear (<1%) fully
transparent -> PNG -> key image + metadata -> Folder live KML + stable
entry KML (individual files).

Exit codes: 0 updated (or skipped); 2 source/validation failure
(previous kept); 1 unexpected error.
"""

import datetime as dt
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR,
                              base_metadata, bin_to_canvas, canvas_indices,
                              grib_stamp_to_det, load_bounds, now_det_str,
                              promote_stage, read_state, save_png,
                              source_token, stage_dir, write_metadata,
                              write_state)
from gradient_scale import draw_scale_legend, render_rgba
import hrrr

PRODUCT = "cloud_cover"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Cloud_Cover.kml"
OVERLAY_NAME = "☁️ LIVE CLOUD COVER"
SKIP_NOTE = "Turn on/off independently of all other layers."

DARK_BLUE = (16, 52, 140)
BLUE = (20, 110, 200)
CYAN = (20, 190, 200)
GREEN = (90, 190, 80)
YELLOW = (245, 215, 50)
ORANGE = (240, 130, 25)
RED = (205, 30, 35)
MAGENTA = (225, 40, 130)
VIOLET = (130, 40, 170)
DARK_PURPLE = (59, 10, 90)

CLOUD_STOPS = [
    (0.0, DARK_BLUE),
    (12.0, BLUE),
    (25.0, CYAN),
    (40.0, GREEN),
    (55.0, YELLOW),
    (70.0, ORANGE),
    (82.0, RED),
    (90.0, MAGENTA),
    (95.0, VIOLET),
    (100.0, DARK_PURPLE),
]
CLOUD_LABELS = [
    (0.0, "0 clear"),
    (10.0, "10"),
    (25.0, "25"),
    (50.0, "50"),
    (75.0, "75"),
    (100.0, "100 overcast"),
]


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
        file_url, dd, cc = hrrr.latest_cycle()
    except Exception as e:
        print(f"[{PRODUCT}] NO CYCLE AVAILABLE (keeping previous): {e}")
        return 2
    source_id = f"hrrr-{dd}-t{cc}z-tcdc-anl"
    prev = read_state(PRODUCT)
    if (prev.get("source_id") == source_id
            and prev.get("render_version") == RENDER_VERSION
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png"))
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE))):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        return _refresh_kml()
    try:
        return _build(file_url, dd, cc, source_id)
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


def _build(file_url, dd, cc, source_id):
    bounds = load_bounds()
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    raw_path = os.path.join(RAW_DIR, "hrrr_cloud_current.grib2")
    hrrr.fetch_messages(file_url, raw_path, [("TCDC", "entire atmosphere")])
    got = hrrr.read_messages(raw_path, {"TCDC": 1})["TCDC"]
    vals, lats, lons, data_date, data_time = got
    data_time_utc = grib_stamp_to_det(data_date, data_time)
    vals = np.asarray(vals, dtype=float).ravel()
    lats = np.asarray(lats, dtype=float).ravel()
    lons = ((np.asarray(lons, dtype=float).ravel() + 180) % 360) - 180

    ok_src = np.isfinite(vals) & (vals >= 0.0) & (vals <= 100.0)
    if int(ok_src.sum()) < 5_000:
        raise ValueError(f"too few valid source cells ({int(ok_src.sum())})")
    print(f"[{PRODUCT}] {dd} t{cc}z TCDC: valid={int(ok_src.sum())} "
          f"range=[{vals[ok_src].min():.1f},{vals[ok_src].max():.1f}] %")

    rows, cols, valid = canvas_indices(lats, lons, bounds)
    field, _counts = bin_to_canvas(rows, cols, vals, valid, (H, W),
                                   splat_radius=2)
    okv = np.isfinite(field) & (field >= 0.0) & (field <= 100.0)
    n_valid = int(okv.sum())
    if n_valid < 50_000:
        raise ValueError(f"too few canvas cells ({n_valid})")
    cur_max = float(field[okv].max())

    rgba = render_rgba(field, CLOUD_STOPS, bounds["overlay_alpha"])
    # Basin rectangle (no shoreline cut) + clear-sky transparency:
    # anything under 1% is clear -> fully transparent.
    clear = ~np.isfinite(field) | (field < CONFIG["clear_threshold"])
    rgba[clear, 3] = 0
    # NOTE: no apply_shoreline_mask here — layers 1-2 stay a full rectangle.
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    print(f"[{PRODUCT}] opaque={n_opaque} (cloudy pixels)")
    if n_opaque < 1_000 and cur_max >= 1.0:
        raise ValueError("empty raster despite cloudy source")

    unit = CONFIG["display_units"]
    subtitle = (f"Cloud cover ({unit}, HRRR hourly)  |  {data_time_utc}  |  "
                f"max {cur_max:.0f}%")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        "%", CLOUD_STOPS, CLOUD_LABELS,
        f"Source: NOAA HRRR {dd} t{cc}z (analysis)  |  Processed {now_det_str()}",
        note="Clear sky (<1%) is transparent, not colored.")
    scale_html = ("Cloud cover (%, as filed by HRRR): <b>LOWEST 0 clear</b> "
                  "transparent &rarr; thin cloud <b>dark-blue/blue/cyan</b> &rarr; "
                  "<b>40 green</b> &rarr; <b>55 yellow</b> &rarr; <b>70 orange</b> "
                  "&rarr; <b>82 red</b> &rarr; <b>90 magenta</b> &rarr; "
                  "<b>HIGHEST+ 100 overcast</b> dark-purple. Same % always shows the same "
                  "color; clear areas stay see-through.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (NOMADS)",
        units="% (display = source %)",
        source_resolution="~3 km HRRR CONUS grid (1799x1059), mean-binned to canvas",
        color_min=0.0, color_max=100.0, color_units="%",
        missing_data_treatment=("only [0,100] admitted; clear (<1%) and missing "
                                "rendered fully transparent; full basin rectangle, "
                                "no shoreline cut; never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{dd} t{cc}z"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["stats"] = {"valid_cells": n_valid, "current_max_pct": cur_max,
                     "opaque_pixels": n_opaque}
    token = meta["source_version"]
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{CONFIG['what']}</p>"
        f"<p><img src=\"{legend_src(token)}\" width=\"600\" alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Units:</b> % cloud cover<br/>"
        f"<b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> HRRR hourly cycles (15-min steps feed the hourly view)<br/>"
        f"<b>Data time:</b> {data_time_utc}<br/>"
        f"<b>Source version:</b> {source_id}<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Provenance:</b> <a href=\"{CONFIG['source_url']}\">NOAA HRRR</a></p>")
    meta["folder_html"] = folder_html
    write_metadata(stage_prod, meta)
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    outs = live_out_dirs(stage, KML_FILE)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        folder=(CONFIG["title"], folder_html),
        out_dirs=outs["live"])
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    from geospatial_utils import promote_stage
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "source_id": source_id,
                          "render_version": RENDER_VERSION,
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
