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
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR, WAVE_TICKS,
                              base_metadata, download, fetch_buoy_obs,
                              grib_stamp_to_det, now_det_str,
                              promote_stage, read_state, source_token,
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
    meta["source_version"] = source_token(source_id)
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
        splat_radius=2, product_dir=stage_prod, tick_labels=WAVE_TICKS)

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
    scale_html = (f"Wave height (feet, one continuous gradient): <b>0</b> dark "
                  f"blue → <b>2</b> blue/cyan → <b>5</b> cyan/green → <b>9</b> "
                  f"green/yellow → <b>12–15</b> yellow-orange → orange → "
                  f"<b>20</b> red → <b>23–26</b> red-violet → violet → "
                  f"<b>30+</b> dark purple extreme. Model maximum this run: "
                  f"<b>{run_max} ft</b>. Values above 30 ft stay dark purple. "
                  f"Significant height = average of highest third of waves.")
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
        out_dirs=outs["live"])
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
