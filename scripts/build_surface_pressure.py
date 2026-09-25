"""Pipeline C2 — LIVE SURFACE PRESSURE (independent).

NOAA/NCEP HRRR 3 km MSLMA mean sea-level pressure analysis (hourly
cycles) -> Pa/100 = hPa (the only conversion) -> bin native Lambert
grid onto the common canvas -> NO shoreline cut (basin rectangle:
lon -93..-73.5, lat 40.5..49.5) -> FIXED absolute spectrum with the
everyday average 1013.25 hPa in the MIDDLE (yellow): lows run
dark-blue->green left, highs run orange->dark-purple right ->
transparent PNG (only missing data transparent) -> key image +
metadata -> Folder live KML + stable entry KML (individual files).

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

PRODUCT = "surface_pressure"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Surface_Pressure.kml"
OVERLAY_NAME = "🧭 LIVE SURFACE PRESSURE"
SKIP_NOTE = "Turn on/off independently of all other layers."

AVG_HPA = 1013.25
P_STOPS = [
    (980.0, (16, 52, 140)),    # dark blue: deep low
    (992.0, (20, 110, 200)),   # blue
    (1000.0, (20, 190, 200)),  # cyan
    (1006.0, (90, 190, 80)),   # green
    (1010.0, (180, 200, 60)),  # green-yellow approach
    (1013.25, (245, 215, 50)),  # yellow: AVERAGE middle
    (1017.0, (240, 130, 25)),  # orange
    (1023.0, (205, 30, 35)),   # red
    (1029.0, (225, 40, 130)),  # magenta/pink
    (1034.0, (130, 40, 170)),  # violet
    (1040.0, (59, 10, 90)),    # dark purple: strong high
]
P_LABELS = [
    (980.0, "LOWEST 980"),
    (1000.0, "1000"),
    (1013.25, "AVERAGE 1013.25"),
    (1025.0, "1025"),
    (1040.0, "HIGHEST+ 1040"),
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
    source_id = f"hrrr-{dd}-t{cc}z-mslma-anl"
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
    raw_path = os.path.join(RAW_DIR, "hrrr_pres_current.grib2")
    hrrr.fetch_messages(file_url, raw_path, [("MSLMA", "mean sea level")])
    vals_pa, lats, lons, data_date, data_time = _read_mslma(raw_path)
    data_time_utc = grib_stamp_to_det(data_date, data_time)
    hpa = np.asarray(vals_pa, dtype=float).ravel() / 100.0
    lats = np.asarray(lats, dtype=float).ravel()
    lons = ((np.asarray(lons, dtype=float).ravel() + 180) % 360) - 180

    lo, hi = CONFIG["valid_min_hpa"], CONFIG["valid_max_hpa"]
    ok_src = np.isfinite(hpa) & (hpa >= lo) & (hpa <= hi)
    if int(ok_src.sum()) < 5_000:
        raise ValueError(f"too few valid source cells ({int(ok_src.sum())})")
    print(f"[{PRODUCT}] {dd} t{cc}z MSLMA: valid={int(ok_src.sum())} "
          f"range=[{hpa[ok_src].min():.1f},{hpa[ok_src].max():.1f}] hPa")

    rows, cols, valid = canvas_indices(lats, lons, bounds)
    field, _counts = bin_to_canvas(rows, cols, hpa, valid, (H, W),
                                   splat_radius=2)
    okv = np.isfinite(field) & (field >= lo) & (field <= hi)
    n_valid = int(okv.sum())
    if n_valid < 50_000:
        raise ValueError(f"too few canvas cells ({n_valid})")
    cur_min, cur_max = float(field[okv].min()), float(field[okv].max())

    rgba = render_rgba(field, P_STOPS, bounds["overlay_alpha"])
    # Basin rectangle (no shoreline cut): only missing data is transparent.
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    if n_opaque < 50_000:
        raise ValueError("empty raster")

    unit = CONFIG["display_units"]
    side = "above average (high)" if cur_max > AVG_HPA else "near average"
    subtitle = (f"Sea-level pressure ({unit}, HRRR hourly)  |  {data_time_utc}  |  "
                f"range {cur_min:.0f}-{cur_max:.0f}")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, P_STOPS, P_LABELS,
        f"Source: NOAA HRRR {dd} t{cc}z (analysis)  |  Processed {now_det_str()}",
        note="Middle yellow = average 1013.25 hPa.")
    scale_html = ("Sea-level pressure (hPa, HRRR analysis, fixed absolute scale): "
                  "<b>LOWEST 980</b> dark-blue deep low &rarr; <b>1000</b> cyan &rarr; "
                  "<b>1006</b> green &rarr; <b>AVERAGE 1013.25</b> yellow middle "
                  "&rarr; <b>1017</b> orange &rarr; <b>1023</b> red &rarr; "
                  "<b>1029</b> magenta &rarr; <b>HIGHEST+ 1040</b> dark-purple. "
                  "Same pressure always shows the same color; the everyday "
                  f"average sits in the middle. Current basin range: "
                  f"<b>{cur_min:.0f}-{cur_max:.0f} hPa</b> ({side}).")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (NOMADS)",
        units="hPa (display; source Pa / 100)",
        source_resolution="~3 km HRRR CONUS grid (1799x1059), mean-binned to canvas",
        color_min=CONFIG["color_min_hpa"], color_max=CONFIG["color_max_hpa"],
        color_units="hPa",
        missing_data_treatment=("only [900,1100] admitted; missing rendered fully "
                                "transparent; full basin rectangle, no shoreline cut; "
                                "never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{dd} t{cc}z"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["standard_atmosphere_hpa"] = AVG_HPA
    meta["stats"] = {"valid_cells": n_valid, "current_min_hpa": cur_min,
                     "current_max_hpa": cur_max}
    token = meta["source_version"]
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{CONFIG['what']}</p>"
        f"<p><img src=\"{legend_src(token)}\" width=\"600\" alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Units:</b> hPa (hectopascals, sea-level)<br/>"
        f"<b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> HRRR hourly cycles<br/>"
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


def _read_mslma(path):
    """MSLMA (shortName 'mslma', step 0) -> (Pa, lats, lons, date, time)."""
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    with open(path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                if (str(codes_get(h, "shortName")).lower() == "mslma"
                        and str(codes_get(h, "step")) == "0"):
                    vals = codes_get_values(h).astype(float)
                    lats = codes_get_array(h, "latitudes").astype(float)
                    lons = codes_get_array(h, "longitudes").astype(float)
                    lons = ((lons + 180) % 360) - 180
                    return (vals, lats, lons, str(codes_get(h, "dataDate")),
                            str(codes_get(h, "dataTime")).zfill(4))
            finally:
                codes_release(h)
    raise ValueError("MSLMA analysis message (step=0) not found in GRIB2")


def legend_src(token=None):
    from build_kml import pages_base
    base = pages_base()
    v = f"?v={token}" if token else ""
    return f"{base}/{PRODUCT}/legend.png{v}"


if __name__ == "__main__":
    sys.exit(main())
