"""Pipeline L — LIVE SNOW COVERAGE (independent, Michigan-only).

NOAA/NCEP HRRR 3 km snow analysis (hourly cycles): SNOD snow depth (m)
gated by SNOWC snow-cover % — COLOR means CONFIRMED SNOW ON THE GROUND.
Falling snow, forecasts, precipitation, radar, clouds, and uncertain or
invalid observations are transparent, as are no-snow land, water, and
everything outside Michigan. Masks: authoritative Michigan boundary +
shared shoreline mask. Continuous OKLab-friendly gradient (silver to
white), historical LOWEST/HIGHEST+ scale, Folder KML, no ScreenOverlay.

Depth in inches is a direct unit conversion of the source metres (no
depth source beyond HRRR analysis is operable without credentials; see
DATA_SOURCES.md). Exit codes: 0 updated (or skipped); 2 failure (previous
kept); 1 unexpected error.
"""

import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_kml,
                       description_html, legend_block)
from geospatial_utils import (REPO_ROOT, SITE_DIR, apply_shoreline_mask,
                              base_metadata, bin_to_canvas, canvas_indices,
                              load_bounds, load_michigan_mask,
                              promote_stage, read_state, save_png,
                              stage_dir, utcnow_iso, write_metadata,
                              write_state, bleed_rgb_into_transparent)
from gradient_scale import (SNOW_FAMILY, build_linear_stops,
                            draw_scale_legend, fmt_val, load_record,
                            record_tick_labels, render_rgba, save_record,
                            update_record)
from hrrr import fetch_messages, latest_cycle, read_messages

PRODUCT = "snow_coverage"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Snow_Coverage.kml"
OVERLAY_NAME = "\U00002744\uFE0F LIVE SNOW COVERAGE"
SKIP_NOTE = "Turn on/off independently of all other layers."
M_TO_IN = 39.3701
LEGEND_TEXT = ("Shows snow currently on the ground. Falling snow, clouds, "
               "precipitation without confirmed ground snow, and uncertain "
               "observations are transparent.")


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
        base, datestr, cycle = latest_cycle(30)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    prev = read_state(PRODUCT)
    if prev.get("model_cycle") == f"{datestr} t{cycle}z" \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")):
        print(f"[{PRODUCT}] model cycle unchanged ({datestr} t{cycle}z); keeping.")
        return _refresh_kml()
    try:
        return _build(base, datestr, cycle)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _refresh_kml():
    try:
        with open(os.path.join(SITE_DIR, PRODUCT, "metadata.json")) as f:
            meta = json.load(f)
        token = meta["processing_time_utc"].replace(" ", "_").replace(":", "")
        block = legend_block(f"{PRODUCT}/legend.png", token,
                             meta.get("legend_scale_html", ""))
        kml_text = build_kml(
            PRODUCT, KML_FILE, OVERLAY_NAME,
            f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
            description_html(CONFIG["title"], meta, SKIP_NOTE, block),
            CONFIG["refresh_interval_seconds"], token,
            folder=(CONFIG["title"], meta.get("folder_html", block)),
            out_dirs=[os.path.join(REPO_ROOT, "kml", KML_FILE),
                      os.path.join(SITE_DIR, "kml", KML_FILE)])
        assert_no_vector_geometry(kml_text)
        print(f"[{PRODUCT}] KML base URLs refreshed.")
    except Exception as e:
        print(f"[{PRODUCT}] WARNING: KML refresh failed: {e}")
    return 0


def _build(base, datestr, cycle):
    bounds = load_bounds()
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    raw_path = os.path.join(RAW_DIR, "hrrr_snow_current.grib2")
    fetch_messages(base, raw_path, [("SNOD", "surface"), ("SNOWC", "surface")])
    msgs = read_messages(raw_path, {"SNOD": 1, "SNOWC": 1})
    (sd, lats, lons, data_date, data_time) = msgs["SNOD"]
    (sc, _l, _o, _d, _t) = msgs["SNOWC"]
    data_time_utc = (f"{data_date[0:4]}-{data_date[4:6]}-{data_date[6:8]} "
                     f"{data_time[0:2]}:{data_time[2:4]} UTC")
    lf, lo_n = np.asarray(lats).ravel(), np.asarray(lons).ravel()
    inside = (np.isfinite(lf) & np.isfinite(lo_n)
              & (lf >= bounds["lat_min"]) & (lf <= bounds["lat_max"])
              & (lo_n >= bounds["lon_min"]) & (lo_n <= bounds["lon_max"]))
    rows, cols, _v = canvas_indices(lf, lo_n, bounds)
    ok_src = inside & np.isfinite(sd) & np.isfinite(sc)
    fld_d, _c = bin_to_canvas(rows, cols, np.asarray(sd).ravel(), ok_src,
                              (H, W), splat_radius=2)
    fld_c, _c = bin_to_canvas(rows, cols, np.asarray(sc).ravel(), ok_src,
                              (H, W), splat_radius=2)
    mich = load_michigan_mask()
    gate = (np.isfinite(fld_d) & np.isfinite(fld_c)
            & (fld_d > CONFIG["gate_depth_m"]) & (fld_c > CONFIG["gate_cover_pct"])
            & (fld_d <= CONFIG["valid_max_m"]) & mich)
    n_snow = int(gate.sum())
    print(f"[{PRODUCT}] {datestr} t{cycle}z: snow pixels={n_snow}")
    inches = np.where(gate, fld_d * M_TO_IN, np.nan)

    rec, res = load_record(PRODUCT)
    if n_snow:
        sample = inches[gate][::max(1, n_snow // 20000)][:20000]
        rec, res = update_record(rec, res, sample)
    if rec is None:
        print(f"[{PRODUCT}] no snow observed yet: provisional empty output.")
        return _build_empty(stage, stage_prod, bounds, W, H, data_time_utc,
                            datestr, cycle)
    stops = build_linear_stops(rec["hist_min"], rec["hist_max"],
                               family=SNOW_FAMILY)
    rgba = render_rgba(inches, stops, bounds["overlay_alpha"])
    rgba = apply_shoreline_mask(rgba)
    rgba[~mich, 3] = 0  # Michigan hard clip (state polygon covers lake water)
    save_png(rgba, os.path.join(stage_prod, "current.png"))

    p = rec["percentiles"]
    unit = CONFIG["display_units"]
    labels = record_tick_labels(rec)
    subtitle = (f"Snow depth on the ground ({unit})  |  {data_time_utc}")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, stops, labels,
        f"Source: NOAA HRRR {datestr} t{cycle}z snow analysis  |  "
        f"Processed {utcnow_iso()}",
        note="NO SNOW — TRANSPARENT. White = extreme end only.")
    return _finish(stage, stage_prod, bounds, W, H, rec, res, stops,
                   (lw, lh), subtitle, data_time_utc, datestr, cycle,
                   n_snow, inches)


def _build_empty(stage, stage_prod, bounds, W, H, data_time_utc, datestr,
                 cycle):
    """Valid-empty output (no snow anywhere): transparent raster, provisional
    legend clearly marked, honest metadata. NOT a failure."""
    stops = build_linear_stops(0.0, 24.0, family=SNOW_FAMILY)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba = bleed_rgb_into_transparent(rgba)
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    unit = CONFIG["display_units"]
    subtitle = (f"Snow depth on the ground ({unit})  |  {data_time_utc}  |  "
                f"no snow in Michigan")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, stops,
        [(0.0, "LOWEST 0"), (24.0, "HIGHEST+ 24 (provisional)")],
        f"Source: NOAA HRRR {datestr} t{cycle}z snow analysis  |  "
        f"Processed {utcnow_iso()}",
        note="NO SNOW — TRANSPARENT. Provisional scale until first snow.")
    rec = {"hist_min": 0.0, "hist_max": 24.0, "provisional": True,
           "percentiles": {"p5": 1.0, "p25": 4.0, "p50": 8.0, "p75": 12.0,
                           "p95": 16.0, "p99": 20.0},
           "n_obs": 0, "extends": []}
    res = np.array([], dtype=float)
    return _finish(stage, stage_prod, bounds, W, H, rec, res, stops,
                   (lw, lh), subtitle, data_time_utc, datestr, cycle,
                   0, np.full((H, W), np.nan))


def _finish(stage, stage_prod, bounds, W, H, rec, res, stops, legend_wh,
            subtitle, data_time_utc, datestr, cycle, n_snow, inches):
    import math
    lw, lh = legend_wh
    p = rec["percentiles"]
    unit = CONFIG["display_units"]
    vmax = rec["hist_max"]
    scale_html = (f"Snow depth on the ground ({unit}), continuous: "
                  f"<b>LOWEST {fmt_val(rec['hist_min'])}</b> (silver-gray) → "
                  f"<b>HIGHEST+ {fmt_val(vmax)}</b> (white extreme). "
                  f"{LEGEND_TEXT}")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"],
        CONFIG["variable"] + f"; {datestr} t{cycle}z",
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (NOMADS)",
        units=f"{unit} (display); source m",
        source_resolution="~3 km HRRR Lambert grid (1799x1059)",
        color_min=rec["hist_min"], color_max=vmax, color_units=unit,
        missing_data_treatment=("only SNOD>0.002 m with SNOWC>0 inside "
                                "Michigan admitted; no-snow/cloud/invalid/"
                                "out-of-state transparent; never interpolated."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{datestr} t{cycle}z"
    meta["historical"] = {"low": rec["hist_min"], "high": vmax,
                          "percentiles": p, "n_obs": rec["n_obs"],
                          "provisional": rec.get("provisional", False),
                          "extends": rec.get("extends", [])}
    with np.errstate(invalid="ignore"):
        mx = float(np.nanmax(inches)) if n_snow else 0.0
        mn = float(np.nanmin(inches)) if n_snow else 0.0
    meta["stats"] = {"snow_pixels": n_snow, "current_min": mn,
                     "current_max": mx}
    token = meta["processing_time_utc"].replace(" ", "_").replace(":", "")
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{LEGEND_TEXT}</p>"
        f"<p><img src=\"{legend_block_src(PRODUCT, token)}\" width=\"600\" "
        f"alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Units:</b> {unit} (source metres)<br/>"
        f"<b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> hourly analysis cycles<br/>"
        f"<b>Data time:</b> {data_time_utc}<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Provenance:</b> <a href=\"{CONFIG['source_url']}\">NOMADS HRRR</a></p>")
    meta["folder_html"] = folder_html
    write_metadata(stage_prod, meta)
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        folder=(CONFIG["title"], folder_html),
        out_dirs=[os.path.join(stage, "kml", KML_FILE),
                  os.path.join(stage, "site", "kml", KML_FILE)])
    assert_no_vector_geometry(kml_text)

    save_record(PRODUCT, rec, res)
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def legend_block_src(product, token=None):
    from build_kml import pages_base
    base = pages_base()
    v = f"?v={token}" if token else ""
    return f"{base}/{product}/legend.png{v}"


if __name__ == "__main__":
    sys.exit(main())
