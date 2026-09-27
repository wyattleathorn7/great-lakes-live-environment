"""Pipeline K — LIVE AIR TEMPERATURE (independent).

NOAA/NCEP HRRR 3 km 2 m temperature analysis (hourly cycles; Kelvin ->
Fahrenheit; FIXED key, -60..150 F, after the reference gradient zones,
BRIGHTENED throughout so no color renders dark; the MAP paints a steady
continuous gradient passing exactly through every key color) -> validate -> FULL
BASIN RECTANGLE (lon -93..-73.5, lat 40.5..49.5: land and water both
paint; only missing data is transparent) -> transparent PNG -> key image
+ metadata -> Folder KML.

The color scale is FIXED (same colors for the same temperatures every
day); the historical record still tracks LOWEST/HIGHEST+ and its values
are reported in the description. Exit codes: 0 updated (or skipped);
2 failure (previous kept); 1 unexpected error.
"""

import json
import math
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
                               fetch_buoy_obs, grib_stamp_to_det, load_bounds,
                               now_det_str, promote_stage,
                               read_state, save_png, source_token, stage_dir,
                               utcnow_iso, write_metadata, write_state)
from gradient_scale import (fmt_val,
                            load_record, render_rgba,
                            save_record, update_record)
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

# Banded 5 F-step key (-60..150 F) following the reference gradient zones,
# BRIGHTENED so no band renders dark on the map or in the key image:
# blue-white extreme cold -> gray-blue subfreezing -> royal-blue freezing
# wall -> turquoise/blue frigid -> lime/teal transition -> mellow yellow ->
# orange/gold warming -> vivid scorching pink/red (lifted maroons).
TEMP_BANDS = [
    (-60, -55, (232, 241, 250)),
    (-55, -50, (220, 233, 247)),
    (-50, -45, (207, 224, 245)),
    (-45, -40, (194, 215, 242)),
    (-40, -35, (180, 205, 239)),
    (-35, -30, (166, 195, 236)),
    (-30, -25, (152, 185, 233)),
    (-25, -20, (138, 175, 230)),
    (-20, -15, (124, 165, 226)),
    (-15, -10, (110, 155, 222)),
    (-10, -5, (96, 145, 218)),
    (-5, 0, (82, 127, 208)),
    (0, 5, (91, 135, 214)),
    (5, 10, (84, 120, 194)),
    (10, 15, (78, 106, 175)),
    (15, 20, (71, 92, 156)),
    (20, 25, (65, 78, 137)),
    (25, 30, (58, 66, 119)),
    (30, 35, (46, 79, 214)),
    (35, 40, (43, 86, 227)),
    (40, 45, (31, 127, 208)),
    (45, 50, (31, 160, 216)),
    (50, 55, (37, 184, 200)),
    (55, 60, (47, 191, 168)),
    (60, 65, (95, 196, 137)),
    (65, 70, (168, 212, 106)),
    (70, 75, (214, 222, 95)),
    (75, 80, (242, 225, 76)),
    (80, 85, (245, 201, 58)),
    (85, 90, (245, 168, 46)),
    (90, 95, (240, 126, 34)),
    (95, 100, (232, 90, 40)),
    (100, 105, (240, 64, 106)),
    (105, 110, (238, 36, 88)),
    (110, 115, (224, 22, 64)),
    (115, 120, (211, 18, 70)),
    (120, 125, (196, 15, 62)),
    (125, 130, (178, 13, 56)),
    (130, 135, (160, 12, 50)),
    (135, 140, (142, 11, 44)),
    (140, 145, (124, 10, 38)),
    (145, 150, (106, 9, 32)),
]

# Step-function stops: identical colors on both edges of each band, so the
# KEY IMAGE paints flat 5-degree bars (no blending between bars). Built
# band by band (NOT globally sorted) so each shared edge keeps the order
# (hi, old-color), (lo, new-color) with a zero-width transition segment.
TEMP_STOPS = []
for _lo, _hi, _rgb in TEMP_BANDS:
    TEMP_STOPS += [(_lo, _rgb), (_hi, _rgb)]

# Smooth raster stops: one anchor at each band CENTER in its band color, so
# the MAP paints a STEADY continuous gradient that passes exactly through
# every key color (band centers match the key pixel-for-pixel; only the
# 5-degree spans across boundaries blend). Same key colors, same order,
# strictly increasing values: a faithful monotonic function of source
# temperature, like the previous continuous spectrum.
TEMP_SMOOTH_STOPS = [((lo + hi) / 2.0, rgb) for lo, hi, rgb in TEMP_BANDS]

SCALE_HTML = (
    "2 m air temperature (degF), FIXED key (-60..150 F) with a STEADY "
    "gradient on the map: the raster blends smoothly through blue-white "
    "extreme cold &rarr; gray-blue subfreezing &rarr; <b>royal-blue "
    "freezing wall (30-40)</b> &rarr; frigid turquoise/blues (40-55) "
    "&rarr; transitional lime/teals (55-70) &rarr; mellow yellows "
    "(70-85) &rarr; warming orange/golds (85-100) &rarr; <b>scorching "
    "pink/reds (100+) to HIGHEST+ 150</b>, passing exactly through every "
    "key color (key bars show the <b>LOWEST -60</b> to <b>HIGHEST+ 150</b> "
    "band centers). "
    "Same temperature always shows the same color; every color is "
    "brightened so nothing renders dark. Freezing (32 F) sits inside "
    "the royal-blue wall.")


def ff(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# Zone callouts for the key image (mirrors the reference gradient layout):
# (band_lo, band_hi, [text lines], text rgb darkened for white background).
KEY_ZONES = [
    (100, 150, ["Scorching PINK/REDS", "past 100\u00b0F \u2014 really hot!"], (194, 20, 68)),
    (85, 100, ["Warming ORANGE/GOLDS", "past 85\u00b0F"], (194, 94, 16)),
    (70, 85, ["Mellow YELLOWS,", "70\u201385\u00b0"], (154, 130, 0)),
    (55, 70, ["Transitional LIME/TEALS,", "55\u201370\u00b0"], (30, 122, 90)),
    (40, 55, ["Frigid TURQUOISE/BLUES,", "40\u201355\u00b0F"], (11, 111, 164)),
    (30, 40, ["Royal-BLUE wall", "at freezing (32\u00b0F)"], (30, 64, 214)),
    (0, 30, ["Light GRAY/BLUES", "set in at 0\u00b0F"], (61, 78, 115)),
    (-60, 0, ["Barely-there BLUE/WHITE \u2014", "extremely cold!"], (74, 111, 165)),
]


def draw_banded_key(path, title, subtitle, bands, zones, source_line):
    """Key image recreating the reference gradient layout on white.

    Left column: one labeled bar per 5 F band (hottest on top), painted
    with the EXACT band colors used on the raster. Right column: bracket
    callouts grouping the bands into named zones. Returns (W, H).
    """
    from PIL import Image, ImageDraw
    from geospatial_utils import _legend_font
    W = 640
    pad = 12
    label_w, bar_w = 92, 196
    bar_x0 = pad + label_w + 8
    bar_x1 = bar_x0 + bar_w
    bracket_x = bar_x1 + 14
    text_x = bracket_x + 10
    row_h, gap = 20, 3
    stride = row_h + gap
    y0 = 64
    n = len(bands)
    rows_end = y0 + n * stride
    H = rows_end + 44
    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    f_title, f_body, f_small = _legend_font(20), _legend_font(14), _legend_font(12)
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=2)
    d.text((pad, 8), title, font=f_title, fill=(10, 10, 10))
    d.text((pad, 34), subtitle, font=f_small, fill=(40, 40, 40))
    # bands hottest-on-top like the reference
    ordered = sorted(bands, key=lambda b: b[0], reverse=True)
    row_of_lo = {}
    for i, (lo, hi, rgb) in enumerate(ordered):
        y = y0 + i * stride
        row_of_lo[lo] = y
        d.text((pad, y + 2), f"{lo} to {hi}", font=f_small, fill=(10, 10, 10))
        d.rectangle([bar_x0, y, bar_x1, y + row_h], fill=rgb + (255,),
                    outline=(90, 90, 90))
    # zone brackets + callouts
    for zlo, zhi, lines, color in zones:
        top_lo = zhi - 5  # topmost band's lo edge in this zone
        y_top = row_of_lo[top_lo] + row_h // 2
        y_bot = row_of_lo[zlo] + row_h // 2
        d.line([(bar_x1 + 4, y_top), (bracket_x, y_top)], fill=(80, 80, 80), width=1)
        d.line([(bar_x1 + 4, y_bot), (bracket_x, y_bot)], fill=(80, 80, 80), width=1)
        d.line([(bracket_x, y_top), (bracket_x, y_bot)], fill=(80, 80, 80), width=1)
        ym = (y_top + y_bot) / 2 - (len(lines) * 17) / 2
        for k, line in enumerate(lines):
            d.text((text_x, ym + k * 17), line, font=f_body, fill=color)
    d.text((pad, rows_end + 8), source_line, font=f_small, fill=(60, 60, 60))
    d.text((pad, rows_end + 24), "No-data transparent. Same color = same temperature.",
           font=f_small, fill=(60, 60, 60))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return W, H
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
            and prev.get("render_version") == RENDER_VERSION \
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
    # Source unchanged: deterministically rewrite entry + live KMLs from
    # committed metadata (byte-identical when nothing changed).
    try:
        refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                             CONFIG["title"], SKIP_NOTE,
                             CONFIG["refresh_interval_seconds"])
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
    data_time_utc = grib_stamp_to_det(data_date, data_time)
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
    # Map: steady gradient through the key colors (band centers exact).
    # Key image uses TEMP_STOPS (flat bars) and is NOT touched.
    stops = list(TEMP_SMOOTH_STOPS)
    rgba = render_rgba(field, stops, bounds["overlay_alpha"])
    # NOTE: full basin rectangle (no shoreline cut). Only missing data
    # is transparent.
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
    subtitle = (f"2 m air temperature ({unit})  |  {data_time_utc}")
    lw, lh = draw_banded_key(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        TEMP_BANDS, KEY_ZONES,
        f"Source: NOAA HRRR {datestr} t{cycle}z analysis  |  "
        f"Processed {now_det_str()}")
    scale_html = SCALE_HTML + (
        f" Record <b>LOWEST {fmt_val(rec['hist_min'])}</b> / "
        f"<b>HIGHEST+ {fmt_val(rec['hist_max'])}</b>.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (NOMADS)",
        units=f"{unit} (display); source K",
        source_resolution="~3 km HRRR Lambert grid (1799x1059)",
        color_min=rec["hist_min"], color_max=rec["hist_max"], color_units=unit,
         missing_data_treatment=("only [-60,55] C admitted pre-conversion; "
                                 "full basin rectangle, no shoreline cut; "
                                 "missing analysis transparent; never interpolated."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{datestr} t{cycle}z"
    source_id = f"hrrr-{datestr}-t{cycle}z"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["historical"] = {"low": rec["hist_min"], "high": rec["hist_max"],
                          "percentiles": p, "n_obs": rec["n_obs"],
                          "extends": rec.get("extends", [])}
    meta["stats"] = {"valid_cells": n_valid, "current_min": cur_min,
                     "current_max": cur_max}
    meta["buoy_qc"] = buoy_qc
    token = meta["source_version"]
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

    save_record(PRODUCT, rec, res)
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "source_id": source_id,
                          "render_version": RENDER_VERSION,
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
