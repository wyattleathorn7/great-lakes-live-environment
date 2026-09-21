"""Pipeline B — LIVE WATER TEMPERATURE (independent).

NOAA GLSEA daily ASCII -> validate -> extract SST -> clip to Great Lakes
-> degC->degF -> temperature gradient -> transparent PNG -> metadata -> KML.

Exit codes: 0 = updated (or skipped, source unchanged); 2 = source/validation
failure (previous valid raster is left untouched); 1 = unexpected error.
"""

import json
import math
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_kml,
                       description_html, legend_block,
                       refresh_kml_base_url)
from geospatial_utils import (REPO_ROOT, SITE_DIR, base_metadata,
                              download, ensure_coords, fetch_buoy_obs,
                              http_date_to_iso, promote_stage, read_state,
                              stage_dir, utcnow_iso, write_metadata,
                              write_state)
from render_gradient import render_field

PRODUCT = "water_temperature"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
GLSEA_URL = CONFIG["source_url"]
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")

BUOY_POS = {  # NDBC (lon, lat) — QC reference only
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
    raw_path = os.path.join(RAW_DIR, "glsea_cur.asc")
    try:
        info = download(GLSEA_URL, raw_path)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    if info["size_bytes"] < 1_000_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: file too small "
              f"({info['size_bytes']} bytes). Keeping previous.")
        return 2

    # No skip: every run rebuilds (tiles must deploy); identical
    # bytes simply produce no commit. Failures still keep previous.

    # Everything below renders into a stage dir first; the live site/ + kml/
    # tree is touched only by promote_stage() on full success, so a crash
    # can never publish a half-updated product set. Any data-dependent
    # failure returns 2 (keep previous); only unexpected engine errors
    # escape to exit 1 via main().
    try:
        return _build(info, raw_path)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(info, raw_path):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    with open(raw_path) as f:
        header = [f.readline() for _ in range(6)]
    try:
        ncols = int(float(header[0].split()[-1]))
        nrows = int(float(header[1].split()[-1]))
    except (ValueError, IndexError):
        print(f"[{PRODUCT}] VALIDATION FAILED: bad header {header}.")
        return 2
    if (ncols, nrows) != (1024, 1024):
        print(f"[{PRODUCT}] VALIDATION FAILED: dims {ncols}x{nrows} != 1024x1024.")
        return 2
    data = np.loadtxt(raw_path, skiprows=6)
    if data.shape != (1024, 1024):
        print(f"[{PRODUCT}] VALIDATION FAILED: shape {data.shape}.")
        return 2

    cdir = ensure_coords(RAW_DIR)
    lats = np.loadtxt(os.path.join(cdir, "1024_latgrid.txt"))
    lons = np.loadtxt(os.path.join(cdir, "1024_longrid.txt"))
    lake_ids = np.loadtxt(os.path.join(cdir, "1024_lake_ids.txt"))
    mask_water = (lake_ids >= 1) & (lake_ids <= 6)

    water = (data != CONFIG["land_code"]) & np.isfinite(data) \
        & (data > -3) & (data < 45)
    n_valid = int(water.sum())
    n_mask = int(mask_water.sum())
    if n_valid == 0:
        print(f"[{PRODUCT}] VALIDATION FAILED: no valid SST cells. "
              f"Keeping previous.")
        return 2
    print(f"[{PRODUCT}] valid SST cells={n_valid} mask water cells={n_mask} "
          f"range_C=[{data[water].min():.2f},{data[water].max():.2f}]")
    if n_valid < 80_000 or n_valid < 0.5 * n_mask or n_valid > 1.6 * n_mask:
        print(f"[{PRODUCT}] VALIDATION FAILED: implausible coverage. "
              f"Keeping previous.")
        return 2

    c = data[water]
    f_vals = c * 9.0 / 5.0 + 32.0
    p1, p99 = float(np.percentile(f_vals, 1)), float(np.percentile(f_vals, 99))
    vmin = max(32.0, math.floor(p1))
    vmax = min(86.0, math.ceil(p99))
    if vmax < vmin + 2:
        vmin, vmax = 32.0, 75.0
    print(f"[{PRODUCT}] color scale F=[{vmin},{vmax}]")

    values_f = np.full(data.shape, np.nan)
    values_f[water] = data[water] * 9.0 / 5.0 + 32.0

    data_time_iso = http_date_to_iso(info["http_last_modified"])
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], GLSEA_URL, CONFIG["variable"],
        data_time_utc=data_time_iso or "unknown (no source timestamp)",
        source_last_modified_utc=info["http_last_modified"] or "unknown",
        units="degF (display); source degC",
        source_resolution="~1.8 km GLSEA grid (1024x1024)",
        color_min=vmin, color_max=vmax, color_units="degF",
        missing_data_treatment=(f"land code {CONFIG['land_code']} and -9999/no-data "
                                "rendered fully transparent; never interpolated."))
    meta["stats"] = {
        "valid_cells": n_valid,
        "lakewide_mean_F": round(float(np.mean(f_vals)), 2),
        "min_F": round(float(np.min(f_vals)), 2),
        "max_F": round(float(np.max(f_vals)), 2),
    }

    field, rgba, meta = render_field(
        PRODUCT, lats, lons, values_f, vmin, vmax, meta,
        title=CONFIG["title"],
        subtitle=(f"{CONFIG['freshness_label']}  |  Data time: "
                  f"{data_time_iso or 'see metadata'}"),
        source_line=(f"Source: NOAA/GLERL CoastWatch GLSEA  |  "
                     f"Processed {utcnow_iso()}"),
        unit_label="\u00b0F", transparent_value=None, fmt="{:.0f}",
        splat_radius=1, product_dir=stage_prod)

    if int((rgba[:, :, 3] > 0).sum()) < 10_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: raster has no water pixels.")
        return 2

    # ---- buoy QC (reference only; never alters the grid) ----
    qc = fetch_buoy_obs(CONFIG["buoys"])
    H, W = rgba.shape[:2]
    b = json.load(open(os.path.join(REPO_ROOT, "config", "great_lakes_bounds.json")))
    for bid, pos in BUOY_POS.items():
        obs = qc.get(bid, {})
        wt = ff(obs.get("WTMP_C"))
        if wt is None:
            print(f"[{PRODUCT}] buoy {bid}: no WTMP obs (skipped)")
            meta.setdefault("buoy_qc", {})[bid] = {**obs, "note": "no WTMP obs"}
            continue
        wt_f = wt * 9.0 / 5.0 + 32.0
        col = int((pos[0] - b["lon_min"]) / (b["lon_max"] - b["lon_min"]) * W)
        row = int((b["lat_max"] - pos[1]) / (b["lat_max"] - b["lat_min"]) * H)
        cell = None
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = row + dr, col + dc
                if 0 <= rr < H and 0 <= cc < W and np.isfinite(field[rr, cc]):
                    cell = float(field[rr, cc])
                    break
            if cell is not None:
                break
        diff = None if cell is None else round(abs(cell - wt_f), 2)
        flag = ("OK" if (diff is not None and diff <= CONFIG["buoy_qc_tolerance_f"])
                else "CHECK" if diff is not None else "NO_GRID_CELL")
        print(f"[{PRODUCT}] buoy {bid}: obs {wt_f:.1f}F grid {cell} diff {diff} -> {flag}")
        meta.setdefault("buoy_qc", {})[bid] = {
            "obs_F": round(wt_f, 2), "grid_F": cell, "absdiff_F": diff,
            "verdict": flag, "obs_time": obs.get("time_utc")}

    np.savez_compressed(os.path.join(RAW_DIR, f"{PRODUCT}_field.npz"),
                        lats=lats, lons=lons, values=values_f)
    from render_gradient import build_grid_tiles
    tiles = build_grid_tiles(PRODUCT, stage_prod, lats, lons, values_f,
                             vmin, vmax, None, 1, True)
    print(f"[{PRODUCT}] LOD tiles: {len(tiles)}")
    mid_f = round((vmin + vmax) / 2, 1)
    scale_html = (f"Surface water temperature (°F): cold <b>{vmin:g}°F</b> "
                  f"(deep blue) → <b>{mid_f:g}°F</b> → warm <b>{vmax:g}°F</b> "
                  f"(red). Lakewide mean this run: "
                  f"<b>{round(float(np.mean(f_vals)), 1)}°F</b>.")
    meta["legend_scale_html"] = scale_html
    write_metadata(stage_prod, meta)  # re-write incl. buoy QC + legend text

    token = meta["processing_time_utc"].replace(" ", "_").replace(":", "")
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    kml_text = build_kml(
        PRODUCT, "Great_Lakes_Live_Water_Temperature.kml",
        "\U0001F321\uFE0F LIVE WATER TEMPERATURE",
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta,
                         "Turn on/off independently of wave and ice layers.",
                         block),
        CONFIG["refresh_interval_seconds"], token,
        tiles=tiles,
        out_dirs=[os.path.join(stage, "kml", "Great_Lakes_Live_Water_Temperature.kml"),
                  os.path.join(stage, "site", "kml", "Great_Lakes_Live_Water_Temperature.kml")])
    assert_no_vector_geometry(kml_text)

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"source_last_modified": info["http_last_modified"],
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
