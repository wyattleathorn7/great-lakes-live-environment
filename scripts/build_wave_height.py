"""Pipeline A — LIVE WAVE HEIGHT (independent).

NCEP GLWU GRIB2 analysis -> validate -> extract HTSGW -> clip to Great Lakes
-> m->ft -> wave-height gradient -> transparent PNG -> metadata -> KML.
NDBC buoys are QC reference only, never the rendering source.

Exit codes: 0 = updated; 2 = source/validation failure (previous valid raster
left untouched); 1 = unexpected error.
"""

import json
import math
import os
import re
import sys
import traceback
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR,
                              apply_shoreline_mask, base_metadata,
                              bin_to_canvas, canvas_indices, download,
                              fetch_buoy_obs, grib_stamp_to_det,
                              load_bounds, now_det_str, promote_stage,
                              read_state, save_png, source_token,
                              stage_dir, utcnow_iso, write_metadata,
                              write_state)
from render_gradient import render_field

PRODUCT = "wave_height"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
M_TO_FT = 3.28084

BUOY_POS = {
    "45001": (-87.793, 48.061),
    "45002": (-86.411, 45.344),
    "45132": (-81.220, 42.460),
    "45012": (-77.383, 43.619),
    "45005": (-82.398, 41.677),
}


def _hex_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# Fixed 18-bin discrete wave-height scale (user-supplied hexes VERBATIM).
# Each row is (label, hex, lo_inclusive_ft, hi_exclusive_ft). Bin edges
# partition [0, +inf) with no gaps: a label like "4-5 ft" covers [4, 6)
# (its 2-ft step up to the next row's lower edge), so values such as
# 5.5 ft, 7.5 ft, etc. always land in exactly one fixed color. Every
# rendered pixel wears its bin's exact hex -- never blended.
# Low-end unwritten rule: [0, 0.2) shows the "0 ft" color (covers 0
# and 0.1); [0.2, 0.7) shows the "0.5 ft" color (covers the 0.2-0.6
# rule); [0.7, 3.0) shows the "1-2 ft" color (covers the 0.7-1.0 rule,
# then the whole step up to 3 ft).
WAVE_FIXED_BINS = [
    ("0 ft", "#3156A0", 0.0, 0.2),
    ("0.5 ft", "#2675B8", 0.2, 0.7),
    ("1-2 ft", "#20A5C2", 0.7, 3.0),
    ("3-4 ft", "#32B878", 3.0, 4.0),
    ("4-5 ft", "#55A83A", 4.0, 6.0),
    ("6-7 ft", "#D6C43A", 6.0, 8.0),
    ("8-9 ft", "#D0A83A", 8.0, 10.0),
    ("10-11 ft", "#E07832", 10.0, 12.0),
    ("12-13 ft", "#C9573C", 12.0, 14.0),
    ("14-15 ft", "#C6283D", 14.0, 16.0),
    ("16-17 ft", "#9F3F68", 16.0, 18.0),
    ("18-19 ft", "#8E3FB3", 18.0, 20.0),
    ("20-21 ft", "#693D8C", 20.0, 22.0),
    ("22-23 ft", "#4F3475", 22.0, 24.0),
    ("24-25 ft", "#75344F", 24.0, 26.0),
    ("26-27 ft", "#8A245F", 26.0, 28.0),
    ("28-29 ft", "#A05F45", 28.0, 30.0),
    ("30+ ft", "#FFFFFF", 30.0, float("inf")),
]
WAVE_FIXED_MIN = 0.0
WAVE_FIXED_MAX = 30.0
WAVE_FIXED_EDGES = [b[2] for b in WAVE_FIXED_BINS[1:]] + [30.0]
WAVE_FIXED_RGB = [_hex_rgb(hx) for _, hx, _, _ in WAVE_FIXED_BINS]
# Never-again guards: user hexes verbatim, edges gap-free and increasing.
assert [hx for _, hx, _, _ in WAVE_FIXED_BINS] == [
    "#3156A0", "#2675B8", "#20A5C2", "#32B878", "#55A83A", "#D6C43A",
    "#D0A83A", "#E07832", "#C9573C", "#C6283D", "#9F3F68", "#8E3FB3",
    "#693D8C", "#4F3475", "#75344F", "#8A245F", "#A05F45", "#FFFFFF"], \
    "wave-height fixed hexes must match the user table verbatim"
assert all(WAVE_FIXED_BINS[i][2] < WAVE_FIXED_BINS[i][3]
           and WAVE_FIXED_BINS[i][3] == WAVE_FIXED_BINS[i + 1][2]
           for i in range(len(WAVE_FIXED_BINS) - 2)), \
    "wave-height bins must tile [0, 30) with no gaps or overlaps"


def wave_bin_index(values_ft):
    """Bin index 0..17 for each value; NaN/negative -> -1 (transparent)."""
    import numpy as _np
    v = _np.asarray(values_ft, dtype=float)
    idx = _np.searchsorted(
        _np.array([0.2, 0.7, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
                   16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]),
        v, side="right")
    idx = _np.clip(idx, 0, 17).astype(int)
    bad = ~( _np.isfinite(v) & (v >= 0))
    idx = _np.where(bad, -1, idx)
    return idx


def render_wave_fixed(field, alpha):
    """Exact preset render: every pixel takes its bin's verbatim hex."""
    import numpy as _np
    H, W = field.shape
    rgba = _np.zeros((H, W, 4), dtype=_np.uint8)
    idx = wave_bin_index(field)
    ok = idx >= 0
    if not ok.any():
        return rgba
    lut = _np.array(WAVE_FIXED_RGB, dtype=_np.uint8)
    rgba[ok, 0:3] = lut[idx[ok]]
    rgba[ok, 3] = alpha
    return rgba


def build_scale_html(run_max):
    """Legend description HTML: fixed-bin list, unwritten rules, buoy
    pointer + how buoys measure wave height. Shared by the builder and
    any offline re-render so the text can never drift between them."""
    return (f"Every pixel shows its bin's exact color; the legend is the full "
            f"table. Unwritten rules: <b>0 and 0.1 ft</b> show as "
            f"<b>0 ft</b>; <b>0.2 to 0.6 ft</b> "
            f"show as <b>0.5 ft</b>; <b>0.7 to 1 ft"
            f"</b> show as <b>1-2 ft</b>. Each 2-ft "
            f"label covers its full step up to the next row (e.g. the "
            f"4-5 ft color covers 4 up to 6 ft), so every value lands "
            f"in exactly one color. Model maximum this run: "
            f"<b>{run_max} ft</b>. Values above 30 ft stay white. "
            f"Significant height = average of highest third of waves. "
            f"For more precise wave height data, turn on the NOAA Great "
            f"Lakes Observation Network layer: its buoys report observed "
            f"wave height at their exact locations, while this layer shows "
            f"the wave model everywhere. How the buoys measure it: "
            f"significant wave height (WVHT) is the average of the highest "
            f"one-third of the waves recorded during a 20-minute sampling "
            f"period each hour. Accelerometers on the hull "
            f"record its heave motion, an onboard FFT converts that motion "
            f"into a wave-energy spectrum, and the height is derived from "
            f"that spectrum; buoy reports cover combined wind waves and "
            f"swell. Great Lakes buoys are seasonal and are hauled out "
            f"before ice season, so winter gaps are normal.")


def draw_wave_height_table(path, title, subtitle, source_line, note=None):
    """Categorical legend: the 18-row user table itself (label + swatch).

    No gradient slider, no hex text -- the swatches are the exact raster
    colors. Returns (W, H)."""
    import os as _os
    from PIL import Image, ImageDraw
    from geospatial_utils import _legend_font
    f_title, f_body, f_small = _legend_font(22), _legend_font(15), _legend_font(13)
    pad, row_h, head_h = 14, 24, 30
    W = 640
    y_top = 64
    y_src = y_top + head_h + row_h * len(WAVE_FIXED_BINS)
    H = y_src + (58 if note else 40) + 8
    img = Image.new("RGBA", (W, H), (255, 255, 255, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=2)
    d.text((pad, 8), title, font=f_title, fill=(10, 10, 10))
    d.text((pad, 36), subtitle, font=f_body, fill=(40, 40, 40))
    y = y_top
    d.rectangle([pad, y, W - pad, y + head_h], fill=(235, 235, 235, 255),
                outline=(40, 40, 40))
    d.text((pad + 6, y + 5), "Wave height", font=f_body, fill=(10, 10, 10))
    d.text((520, y + 5), "Color", font=f_body, fill=(10, 10, 10))
    y += head_h
    for i, (label, hx, _lo, _hi) in enumerate(WAVE_FIXED_BINS):
        if i % 2 == 1:
            d.rectangle([pad, y, W - pad, y + row_h],
                        fill=(245, 245, 245, 255))
        d.text((pad + 6, y + 3), label, font=f_body, fill=(10, 10, 10))
        d.rectangle([520, y + 2, W - pad - 2, y + row_h - 2],
                    fill=_hex_rgb(hx) + (255,), outline=(40, 40, 40))
        d.line([(pad, y), (W - pad, y)], fill=(200, 200, 200, 255))
        y += row_h
    d.rectangle([pad, y_top, W - pad, y], outline=(40, 40, 40))
    y += 6
    d.text((pad, y + 6), source_line, font=f_small, fill=(60, 60, 60))
    if note:
        d.text((pad, y + 24), note, font=f_small, fill=(60, 60, 60))
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    img.save(path)
    return W, H


def ff(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def candidate_urls(now):
    urls = []
    for d in (now, now - timedelta(days=1)):
        datestr = d.strftime("%Y%m%d")
        for cycle in CONFIG["cycles_try_order"]:
            urls.append((
                CONFIG["file_pattern"].format(
                    date_dir=f"glwu.{datestr}", cycle=cycle),
                datestr, cycle))
    return urls


GLWU_UA = {"User-Agent": "great-lakes-live-environment/1.0"}


def newest_available_cycle(now, var="HTSGW", label=None):
    """Newest available GLWU analysis from cheap .idx probes.

    Each candidate's .idx (KBs, not the 66 MB GRIB2) carries the analysis
    stamp (``d=YYYYMMDDHH``) on its ``:anl:`` lines. Returns
    (url, datestr, cycle, stamp) for the newest stamp, or None. The stamp
    -- not the workflow schedule -- is what identifies the source cycle.
    """
    best = None
    tag = label or PRODUCT
    for url, dd, cc in candidate_urls(now):
        try:
            req = urllib.request.Request(url + ".idx", headers=GLWU_UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                idx = r.read().decode("utf-8", errors="replace").splitlines()
            for line in idx:
                p = line.split(":")
                if len(p) > 5 and p[3] == var and "surface" in line \
                        and ":anl:" in line:
                    m = re.search(r"d=(\d{10})", line)
                    if m and (best is None or m.group(1) > best[3]):
                        best = (url, dd, cc, m.group(1))
                    break
        except Exception as e:
            print(f"[{tag}] idx probe {dd} t{cc}z: {str(e)[:100]}")
    return best


def cycle_source_id(stamp):
    return f"glwu-{stamp[:8]}-{stamp[8:]}00Z"


def extract_htsgw_analysis(grib_path):
    """Return (values_m, lats, lons, dataDate, dataTime) for HTSGW step 0."""
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    with open(grib_path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                if codes_get(h, "shortName") == "swh" \
                        and str(codes_get(h, "step")) == "0":
                    vals = codes_get_values(h).astype(float)
                    lats = codes_get_array(h, "latitudes").astype(float)
                    lons = codes_get_array(h, "longitudes").astype(float)
                    lons = ((lons + 180) % 360) - 180
                    date = str(codes_get(h, "dataDate"))
                    time = str(codes_get(h, "dataTime")).zfill(4)
                    return vals, lats, lons, date, time
            finally:
                codes_release(h)
    raise ValueError("HTSGW analysis message (step=0) not found in GRIB2")


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    now = datetime.now(timezone.utc)
    raw_path = os.path.join(RAW_DIR, "glwu_current.grib2")
    # Detect the newest AVAILABLE cycle first (KB .idx probes, no bulk
    # download). Unchanged stamp -> keep the published raster without
    # downloading the 66 MB file again.
    pick = newest_available_cycle(now, "HTSGW")
    if pick is None:
        print(f"[{PRODUCT}] NO CYCLE AVAILABLE (keeping previous).")
        return 2
    url, datestr, cycle, stamp = pick
    source_id = cycle_source_id(stamp)
    prev = read_state(PRODUCT)
    if prev.get("source_id") == source_id \
            and prev.get("render_version") == RENDER_VERSION \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live", "Great_Lakes_Live_Wave_Height.kml")):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        refresh_kml_base_url(PRODUCT, "Great_Lakes_Live_Wave_Height.kml",
                             "\U0001F30A LIVE WAVE HEIGHT", CONFIG["title"],
                             "Turn on/off independently of temperature and ice layers.",
                             CONFIG["refresh_interval_seconds"])
        return 0
    try:
        got = download(url, raw_path, timeout=300)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED for {datestr} t{cycle}z "
              f"(keeping previous): {str(e)[:140]}")
        return 2
    if got["size_bytes"] < 100_000:
        print(f"[{PRODUCT}] candidate {datestr} t{cycle}z too small; "
              f"keeping previous.")
        return 2

    # Render into a stage dir; promote to live site/ + kml/ only on full
    # success. Any data-dependent failure returns 2 (keep previous).
    try:
        return _build(got, url, datestr, cycle, raw_path)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(got, used_url, datestr, cycle, raw_path):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    try:
        vals_m, lats, lons, data_date, data_time = extract_htsgw_analysis(raw_path)
    except Exception as e:
        print(f"[{PRODUCT}] VALIDATION FAILED: GRIB2 decode: {e}. Keeping previous.")
        return 2

    valid = np.isfinite(vals_m) & (vals_m < 9000) & (vals_m >= 0)
    n_valid = int(valid.sum())
    print(f"[{PRODUCT}] HTSGW analysis {data_date} {data_time}Z: "
          f"valid={n_valid}/{vals_m.size}")
    if n_valid < 5_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few valid cells. Keeping previous.")
        return 2
    vmax_m = float(vals_m[valid].max())
    if vmax_m > 12:
        print(f"[{PRODUCT}] VALIDATION FAILED: implausible max {vmax_m} m.")
        return 2

    vals_ft = vals_m * M_TO_FT
    data_time_utc = grib_stamp_to_det(data_date, data_time)
    # Source-aware gate: the GRIB analysis stamp (not the workflow run time)
    # controls regeneration. Same source -> keep the published raster and
    # only ensure the entry/live KMLs are current (deterministic rewrite).
    source_id = f"glwu-{data_date}-{data_time}Z"
    prev = read_state(PRODUCT)
    if prev.get("source_id") == source_id \
            and prev.get("render_version") == RENDER_VERSION \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live", "Great_Lakes_Live_Wave_Height.kml")):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        refresh_kml_base_url(PRODUCT, "Great_Lakes_Live_Wave_Height.kml",
                             "\U0001F30A LIVE WAVE HEIGHT", CONFIG["title"],
                             "Turn on/off independently of temperature and ice layers.",
                             CONFIG["refresh_interval_seconds"])
        return 0

    # Fixed scientific scale 0-30 ft: Great Lakes storms can exceed 25 ft,
    # so the legend always explains the full credible range. Today's
    # maximum is reported in the subtitle and metadata instead.
    p995 = float(np.percentile(vals_ft[valid], 99.5))
    vmax, vmin = 30.0, 0.0
    run_max = round(float(vals_ft[valid].max()), 1)
    print(f"[{PRODUCT}] run max={run_max} ft p99.5={p995:.2f} ft "
          f"-> fixed scale [{vmin},{vmax}] ft")

    values_ft = np.full(vals_m.shape, np.nan)
    values_ft[valid] = vals_ft[valid]

    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], used_url, CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc=got.get("http_last_modified") or "n/a (NOMADS)",
        units="ft (display); source m",
        source_resolution="~2.5 km NCEP GLWU Lambert grid (581x361)",
        color_min=0.0, color_max=vmax, color_units="ft",
        missing_data_treatment=("GRIB2 missing value 9999 and off-water grid "
                                "points rendered fully transparent; never zero-filled."))
    meta["model_cycle"] = f"{datestr} t{cycle}z"
    meta["source_id"] = source_id
    # Cache-buster carries the render generation (same pattern as
    # live_field RENDER_TAGS): identical source re-rendered with new
    # colors/legend/description must change the ?v= token, or Google
    # Earth and browsers keep showing the stale cached PNGs.
    meta["source_version"] = source_token(f"{source_id}-r{RENDER_VERSION}")
    meta["stats"] = {
        "valid_cells": n_valid,
        "max_ft": round(float(vals_ft[valid].max()), 2),
        "mean_ft": round(float(vals_ft[valid].mean()), 3),
        "p99_5_ft": round(p995, 2),
    }

    field, rgba, meta = render_field(
        PRODUCT, lats, lons, values_ft, vmin, vmax, meta,
        title=CONFIG["title"],
        subtitle=(f"{CONFIG['freshness_label']}  |  Model time: {data_time_utc}  |  "
                  f"Model max this run: {run_max} ft"),
        source_line=(f"Source: NCEP GLWU v2.1 (WAVEWATCH III) {datestr} t{cycle}z  |  "
                     f"Processed {now_det_str()}"),
        unit_label="feet", transparent_value=None, fmt="{:.0f}",
        splat_radius=2, product_dir=stage_prod, tick_labels=None,
        discrete_bins=WAVE_FIXED_BINS, discrete_render=render_wave_fixed,
        discrete_legend=draw_wave_height_table)

    if int((rgba[:, :, 3] > 0).sum()) < 10_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: raster has no water pixels.")
        return 2

    # ---- buoy QC (reference only) ----
    qc = fetch_buoy_obs(CONFIG["buoys"])
    bounds = json.load(open(os.path.join(REPO_ROOT, "config", "great_lakes_bounds.json")))
    H, W = rgba.shape[:2]
    for bid, pos in BUOY_POS.items():
        obs = qc.get(bid, {})
        wv = ff(obs.get("WVHT_m"))
        if wv is None:
            print(f"[{PRODUCT}] buoy {bid}: no WVHT obs (skipped)")
            meta.setdefault("buoy_qc", {})[bid] = {**obs, "note": "no WVHT obs"}
            continue
        wv_ft = wv * M_TO_FT
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
        diff = None if cell is None else round(abs(cell - wv_ft), 2)
        flag = ("OK" if (diff is not None and diff <= CONFIG["buoy_qc_tolerance_ft"])
                else "CHECK" if diff is not None else "NO_GRID_CELL")
        print(f"[{PRODUCT}] buoy {bid}: obs {wv_ft:.1f}ft grid {cell} diff {diff} -> {flag}")
        meta.setdefault("buoy_qc", {})[bid] = {
            "obs_ft": round(wv_ft, 2), "grid_ft": cell, "absdiff_ft": diff,
            "verdict": flag, "obs_time": obs.get("time_utc")}

    np.savez_compressed(os.path.join(RAW_DIR, f"{PRODUCT}_field.npz"),
                        lats=lats, lons=lons, values=values_ft)
    scale_html = build_scale_html(run_max)
    meta["legend_scale_html"] = scale_html
    write_metadata(stage_prod, meta)  # re-write incl. buoy QC + legend text

    token = meta["source_version"]
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    outs = live_out_dirs(stage, "Great_Lakes_Live_Wave_Height.kml")
    kml_text = build_kml(
        PRODUCT, "Great_Lakes_Live_Wave_Height.kml",
        "\U0001F30A LIVE WAVE HEIGHT",
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta,
                         "Turn on/off independently of temperature and ice layers.",
                         block),
        CONFIG["refresh_interval_seconds"], token,
        out_dirs=outs["live"],
        meta=meta)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, "Great_Lakes_Live_Wave_Height.kml",
        "\U0001F30A LIVE WAVE HEIGHT",
        entry_description_html(
            CONFIG["title"], meta,
            "Turn on/off independently of temperature and ice layers."),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "source_id": source_id,
                          "render_version": RENDER_VERSION,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
