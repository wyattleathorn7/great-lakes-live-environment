"""Pipeline C — LIVE ICE COVERAGE (independent).

USNIC NAIS daily ASCII grid -> validate -> extract concentration (%) ->
clip to Great Lakes -> ice gradient (0% open water = transparent) ->
transparent PNG -> metadata -> KML.

An all-zero grid in the warm season is VALID data. Corruption is caught by
file-structure, dimension, value-domain, and water-fraction checks.

Exit codes: 0 = updated (or skipped, source unchanged); 2 = source/validation
failure (previous valid raster left untouched); 1 = unexpected error.
"""

import hashlib
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR, base_metadata,
                              download, ensure_coords, http_date_to_det,
                              now_det_str, promote_stage,
                              read_state, source_token, stage_dir, utcnow_iso,
                              write_metadata, write_state)
from render_gradient import render_field

PRODUCT = "ice_coverage"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
ICE_URL = CONFIG["source_url"]
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
COORDS_DIR = os.path.join(RAW_DIR, "coords")


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    raw_path = os.path.join(RAW_DIR, "nic_ice_1800.asc")
    try:
        info = download(ICE_URL, raw_path)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    if info["size_bytes"] < 1_000_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: file too small "
              f"({info['size_bytes']} bytes). Keeping previous.")
        return 2

    # NIC sends no Last-Modified header, so change detection uses a
    # SHA-256 content hash (off-season grids are re-published identically
    # for weeks; hashing avoids pointless rebuilds AND avoids a stuck
    # None == None skip that would freeze updates forever).
    with open(raw_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    # Source-aware gate: NIC sends no Last-Modified, so the content
    # hash is the observation id (off-season grids repeat identically for
    # weeks). Same hash -> keep the published raster, refresh KMLs only.
    source_id = f"nic1800-{digest[:16]}"
    prev = read_state(PRODUCT)
    if prev.get("source_id") == source_id \
            and prev.get("render_version") == RENDER_VERSION \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live", "Great_Lakes_Live_Ice_Coverage.kml")):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        refresh_kml_base_url(PRODUCT, "Great_Lakes_Live_Ice_Coverage.kml",
                             "\U0001F9CA LIVE ICE COVERAGE", CONFIG["title"],
                             "Turn on/off independently of wave and temperature layers.",
                             CONFIG["refresh_interval_seconds"])
        return 0

    # Render into a stage dir; promote to live site/ + kml/ only on full
    # success. Any data-dependent failure returns 2 (keep previous).
    try:
        return _build(info, raw_path, digest, source_id)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(info, raw_path, digest, source_id=None):
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
        print(f"[{PRODUCT}] VALIDATION FAILED: dims {ncols}x{nrows}.")
        return 2
    data = np.loadtxt(raw_path, skiprows=6)
    if data.shape != (1024, 1024):
        print(f"[{PRODUCT}] VALIDATION FAILED: shape {data.shape}.")
        return 2

    land = CONFIG["land_code"]
    is_land = data == land
    is_water = (data >= 0.0) & (data <= 100.0)
    bad = ~(is_land | is_water) & np.isfinite(data)
    n_bad = int((bad | ~np.isfinite(data)).sum())
    n_water = int(is_water.sum())
    print(f"[{PRODUCT}] land={int(is_land.sum())} water={n_water} "
          f"out_of_domain={n_bad} max_ice={data[is_water].max() if n_water else 'n/a'}")
    if n_bad > 100:
        print(f"[{PRODUCT}] VALIDATION FAILED: {n_bad} out-of-domain values. "
              f"Keeping previous.")
        return 2
    if n_water < 80_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: implausible water coverage. "
              f"Keeping previous.")
        return 2

    # lake-mask cross-check (coords LUTs shared with the temperature layer;
    # downloaded on demand so fresh checkouts work)
    try:
        _cdir = ensure_coords(RAW_DIR)
        lake_ids = np.loadtxt(os.path.join(_cdir, "1024_lake_ids.txt"))
        lats = np.loadtxt(os.path.join(_cdir, "1024_latgrid.txt"))
        lons = np.loadtxt(os.path.join(_cdir, "1024_longrid.txt"))
        n_mask = int(((lake_ids >= 1) & (lake_ids <= 6)).sum())
        print(f"[{PRODUCT}] mask water cells={n_mask}")
        if not (0.5 * n_mask < n_water < 1.6 * n_mask):
            print(f"[{PRODUCT}] VALIDATION FAILED: water fraction disagrees "
                  f"with lake mask. Keeping previous.")
            return 2
    except FileNotFoundError:
        print(f"[{PRODUCT}] WARNING: lake-mask LUTs not present; skipping "
              f"mask cross-check.")
        lats = lons = None

    values = np.full(data.shape, np.nan)
    values[is_water] = data[is_water]
    if lats is None:  # cannot georeference -> fail safe, keep previous
        print(f"[{PRODUCT}] VALIDATION FAILED: no georeference LUTs.")
        return 2

    ice_frac = float((data[is_water] > 0).mean()) if n_water else 0.0
    retrieved = now_det_str()
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], ICE_URL, CONFIG["variable"],
        data_time_utc=(f"retrieved {retrieved} (NIC daily analysis; "
                     "NIC publishes no per-file timestamp)"),
        source_last_modified_utc=(http_date_to_det(info["http_last_modified"])
        or "unknown"),
        units="%",
        source_resolution="1.8 km NIC NAIS daily grid (1024x1024)",
        color_min=0.0, color_max=100.0, color_units="%",
        missing_data_treatment=("land code -1 and any value < -1 (e.g. -9999) "
                                "rendered fully transparent and NEVER treated as "
                                "0% ice; 0% open water is also transparent by "
                                "design so base layers stay visible."))
    if source_id is None:
        source_id = f"nic1800-{digest[:16]}"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["stats"] = {
        "water_cells": n_water,
        "ice_covered_fraction": round(ice_frac, 5),
        "max_concentration_pct": round(float(data[is_water].max()), 2) if n_water else 0.0,
        "note": ("ice-free conditions are normal outside Dec-Apr; "
                 "an all-zero grid is valid data, not corruption."),
    }

    field, rgba, meta = render_field(
        PRODUCT, lats, lons, values, 0.0, 100.0, meta,
        title=CONFIG["title"],
        subtitle=(f"{CONFIG['freshness_label']}  |  retrieved {retrieved}"),
        source_line=(f"Source: US National Ice Center daily Great Lakes analysis  |  "
                     f"Processed {now_det_str()}"),
        unit_label="%", transparent_value=0.0, fmt="{:.0f}",
        splat_radius=1, product_dir=stage_prod)

    np.savez_compressed(os.path.join(RAW_DIR, f"{PRODUCT}_field.npz"),
                        lats=lats, lons=lons, values=values)
    maxc = round(float(data[is_water].max()), 1) if n_water else 0.0
    scale_html = (f"Ice concentration (% of water area covered): open water "
                  f"(transparent) → <b>0%</b> → <b>50%</b> → <b>100%</b> "
                  f"total cover (near-white). Maximum this run: "
                  f"<b>{maxc}%</b>. Ice-free water in summer is normal.")
    meta["legend_scale_html"] = scale_html
    write_metadata(stage_prod, meta)

    token = meta["source_version"]
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    outs = live_out_dirs(stage, "Great_Lakes_Live_Ice_Coverage.kml")
    kml_text = build_kml(
        PRODUCT, "Great_Lakes_Live_Ice_Coverage.kml",
        "\U0001F9CA LIVE ICE COVERAGE",
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta,
                         "Turn on/off independently of wave and temperature layers.",
                         block),
        CONFIG["refresh_interval_seconds"], token,
        out_dirs=outs["live"])
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, "Great_Lakes_Live_Ice_Coverage.kml",
        "\U0001F9CA LIVE ICE COVERAGE",
        entry_description_html(
            CONFIG["title"], meta,
            "Turn on/off independently of wave and temperature layers."),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"source_id": source_id,
                          "render_version": RENDER_VERSION,
                          "content_sha256": digest,
                          "source_last_modified": info["http_last_modified"],
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK (ice-covered fraction={ice_frac:.4f}, "
          f"{len(promoted)} files promoted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
