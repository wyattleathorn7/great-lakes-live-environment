"""Pipeline K — LIVE AIR TEMPERATURE (independent).

NOAA/NCEP HRRR 3 km 2 m temperature analysis (hourly cycles; Kelvin ->
Fahrenheit; Apple-style spectrum with a purple negative extension that is
continuous through freezing) -> validate -> clip to Great Lakes environment
-> continuous historical-range gradient -> transparent PNG (full domain;
only missing data transparent) -> key image + metadata -> Folder KML.

Negative values use dark blue -> blue-violet -> violet -> deep purple at
the record-cold end (spec negative rule); non-negative products never get
purple lows. Exit codes: 0 updated (or skipped); 2 failure (previous kept);
1 unexpected error.
"""

import json
import math
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_kml,
                       description_html, legend_block)
from geospatial_utils import (REPO_ROOT, SITE_DIR, base_metadata,
                              bin_to_canvas, canvas_indices, fetch_buoy_obs,
                              load_bounds, promote_stage, read_state,
                              save_png, stage_dir, utcnow_iso,
                              write_metadata, write_state,
                              bleed_rgb_into_transparent)
from gradient_scale import (anchor_values, build_stops, draw_scale_legend,
                            fmt_val, load_record, render_rgba, save_record,
                            update_record)
from hrrr import fetch_messages, latest_cycle, read_messages

PRODUCT = "air_temperature"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Air_Temperature.kml"
OVERLAY_NAME = "\U0001F321\uFE0F LIVE AIR TEMPERATURE"
SKIP_NOTE = "Turn on/off independently of all other layers."
K2F = lambda k: (k - 273.15) * 9.0 / 5.0 + 32.0  # noqa: E731

BUOY_POS = {
    "45001": (-87.793, 48.061),
    "45002": (-86.411, 45.344),
    "45132": (-81.220, 42.460),
    "45012": (-77.383, 43.619),
    "45005": (-82.398, 41.677),
}


def ff(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


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
    raw_path = os.path.join(RAW_DIR, "hrrr_airt_current.grib2")
    fetch_messages(base, raw_path, [("TMP", "2 m above ground")])
    (vals, lats, lons, data_date, data_time) = read_messages(
        raw_path, {"TMP": 1})["TMP"]
    data_time_utc = (f"{data_date[0:4]}-{data_date[4:6]}-{data_date[6:8]} "
                     f"{data_time[0:2]}:{data_time[2:4]} UTC")
    lf, lo_n = np.asarray(lats).ravel(), np.asarray(lons).ravel()
    inside = (np.isfinite(lf) & np.isfinite(lo_n)
              & (lf >= bounds["lat_min"]) & (lf <= bounds["lat_max"])
              & (lo_n >= bounds["lon_min"]) & (lo_n <= bounds["lon_max"]))
    rows, cols, _v = canvas_indices(lf, lo_n, bounds)
    field_k, _c = bin_to_canvas(rows, cols, np.asarray(vals).ravel(),
                                inside & np.isfinite(np.asarray(vals).ravel()),
                                (H, W), splat_radius=2)
    lok, hik = CONFIG["valid_min_c"] + 273.15, CONFIG["valid_max_c"] + 273.15
    okv = np.isfinite(field_k) & (field_k >= lok) & (field_k <= hik)
    n_valid = int(okv.sum())
    field = np.where(okv, K2F(field_k), np.nan)
    print(f"[{PRODUCT}] {datestr} t{cycle}z: valid={n_valid} "
          f"range=[{field[okv].min():.1f},{field[okv].max():.1f}] F")
    if n_valid < 50_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few valid cells. Keeping previous.")
        return 2
    vals = field[okv]
    cur_min, cur_max = float(vals.min()), float(vals.max())

    rec, res = load_record(PRODUCT)
    if rec is None:
        print(f"[{PRODUCT}] cold start: seeding history from this analysis.")
    sample = vals[::max(1, vals.size // 20000)][:20000]
    rec, res = update_record(rec, res, sample)
    stops = build_stops(anchor_values(rec), CONFIG["allow_negative"])
    rgba = render_rgba(field, stops, bounds["overlay_alpha"])
    rgba = bleed_rgb_into_transparent(rgba)  # full-domain product: no water mask
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    if int((rgba[:, :, 3] > 0).sum()) < 50_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: empty raster. Keeping previous.")
        return 2

    # buoy QC: ATMP (reference only; never alters the grid)
    qc = fetch_buoy_obs(CONFIG["buoys"])
    buoy_qc = {}
    for bid, pos in BUOY_POS.items():
        obs = qc.get(bid, {})
        at = ff(obs.get("ATMP_C"))
        if at is None:
            print(f"[{PRODUCT}] buoy {bid}: no ATMP obs (skipped)")
            buoy_qc[bid] = {**obs, "note": "no ATMP obs"}
            continue
        at_f = at * 9.0 / 5.0 + 32.0
        col = int((pos[0] - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"]) * W)
        row = int((bounds["lat_max"] - pos[1]) / (bounds["lat_max"] - bounds["lat_min"]) * H)
        cell = None
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                rr, cc = row + dr, col + dc
                if 0 <= rr < H and 0 <= cc < W and np.isfinite(field[rr, cc]):
                    cell = float(field[rr, cc])
                    break
            if cell is not None:
                break
        diff = None if cell is None else round(abs(cell - at_f), 2)
        flag = ("OK" if (diff is not None and diff <= CONFIG["buoy_qc_tolerance_f"])
                else "CHECK" if diff is not None else "NO_GRID_CELL")
        print(f"[{PRODUCT}] buoy {bid}: obs {at_f:.1f}F grid {cell} diff {diff} -> {flag}")
        buoy_qc[bid] = {
            "obs_F": round(at_f, 2), "grid_F": cell, "absdiff_F": diff,
            "verdict": flag, "obs_time": obs.get("time_utc")}

    p = rec["percentiles"]
    unit = CONFIG["display_units"]
    labels = [(rec["hist_min"], f"LOWEST {fmt_val(rec['hist_min'])}"),
              (p["p25"], fmt_val(p["p25"])),
              (p["p50"], fmt_val(p["p50"])),
              (p["p75"], fmt_val(p["p75"])),
              (rec["hist_max"], f"HIGHEST+ {fmt_val(rec['hist_max'])}")]
    subtitle = (f"2 m air temperature ({unit})  |  {data_time_utc}")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, stops, labels,
        f"Source: NOAA HRRR {datestr} t{cycle}z analysis  |  "
        f"Processed {utcnow_iso()}",
        note="Apple-style spectrum; extreme cold continues into violet.")
    scale_html = (f"2 m air temperature ({unit}), continuous Apple-style "
                  f"spectrum: <b>LOWEST {fmt_val(rec['hist_min'])}</b> "
                  f"(deep purple/violet extreme cold) → cold blue → cyan → "
                  f"green → yellow → orange → red "
                  f"<b>HIGHEST+ {fmt_val(rec['hist_max'])}</b>. Freezing has "
                  f"no color break.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (NOMADS)",
        units=f"{unit} (display); source K",
        source_resolution="~3 km HRRR Lambert grid (1799x1059)",
        color_min=rec["hist_min"], color_max=rec["hist_max"], color_units=unit,
        missing_data_treatment=("only [-60,55] C admitted pre-conversion; "
                                "missing analysis transparent; never interpolated."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{datestr} t{cycle}z"
    meta["historical"] = {"low": rec["hist_min"], "high": rec["hist_max"],
                          "percentiles": p, "n_obs": rec["n_obs"],
                          "extends": rec.get("extends", [])}
    meta["stats"] = {"valid_cells": n_valid, "current_min": cur_min,
                     "current_max": cur_max}
    meta["buoy_qc"] = buoy_qc
    token = meta["processing_time_utc"].replace(" ", "_").replace(":", "")
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{CONFIG['what']}</p>"
        f"<p><img src=\"{legend_block_src(PRODUCT, token)}\" width=\"600\" "
        f"alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Units:</b> {unit} (source Kelvin)<br/>"
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
