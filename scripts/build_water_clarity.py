"""Pipeline I — LIVE WATER CLARITY / TURBIDITY (independent).

NOAA CoastWatch S-NPP VIIRS Kd(PAR) diffuse attenuation (Near Real-Time,
Global 4 km, Daily; ERDDAP nesdisVHNkdparDaily; kd_par, m^-1, NOAA MECB
algorithm, product status Experimental) -> validate -> newest-valid mosaic
of the latest 7 daily composites (daily ocean color is cloud-sparse:
clouds and orbit gaps leave most water pixels empty on any single day,
so each pixel shows its newest valid observation within the window) ->
clip to Great Lakes -> balanced LINEAR historical-range gradient (every
part of the value scale owns an equal share of the color resolution) ->
transparent PNG (water only, shared shoreline
mask) -> key image + metadata -> Folder KML.

The source variable IS diffuse attenuation: larger values mean MORE turbid
water (more light attenuation); clearer water sits at the blue/cyan end.
Source values are never altered for color convenience. Exit codes:
0 updated (or skipped); 2 source/validation failure (previous kept);
1 unexpected error.
"""

import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from erddap_coastwatch import fetch_csv, latest_time
from geospatial_utils import (REPO_ROOT, SITE_DIR, apply_shoreline_mask,
                              base_metadata, load_bounds, promote_stage,
                              RENDER_VERSION, iso_to_det, read_state, save_png, source_token, stage_dir,
                              now_det_str, utcnow_iso, write_metadata, write_state)
from gradient_scale import (KDPAR_LOG_MAX, KDPAR_LOG_MIN, KDPAR_LOG_STOPS,
                            KDPAR_LOG_TICKS, draw_scale_legend, load_record,
                            render_rgba, save_record, update_record)

PRODUCT = "water_clarity"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
# Live source priority (verified 2026-09-23): the Science Quality
# KdPAR daily is the operational stream (newest 2026-09-13, ~10 d
# production latency); the legacy NRT id is kept as fallback but is
# currently intermittent/gone (404 on its time axis).
DATASETS = ["nesdisVHNSQkdparDaily", "nesdisVHNkdparDaily"]
VAR = "kd_par"
KML_FILE = "Great_Lakes_Live_Water_Clarity_Turbidity.kml"
OVERLAY_NAME = "\U0001F30A LIVE WATER CLARITY / TURBIDITY"
SKIP_NOTE = "Turn on/off independently of all other layers."
MOSAIC_DAYS = 7


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def recent_times(n):
    """Newest n daily timestamps (ISO) from the freshest live dataset."""
    import datetime as dt
    end, dataset = None, None
    last = None
    for cand in DATASETS:
        try:
            end = latest_time(cand)
            dataset = cand
            break
        except Exception as e:
            print(f"[{PRODUCT}] dataset {cand} time-axis probe failed: "
                  f"{str(e)[:100]}")
            last = e
    if end is None:
        raise last or RuntimeError("no KdPAR dataset reachable")
    base = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
    return ([((base - dt.timedelta(days=i)).strftime("%Y-%m-%dT12:00:00Z"))
             for i in range(n)], dataset)


def run():
    bounds = load_bounds()
    try:
        times, dataset = recent_times(MOSAIC_DAYS)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    # v2 marker forces one rebuild to deploy the resample fix.
    ds = dataset if "dataset" in dir() else DATASET
    # v2 marker forces one rebuild to deploy the resample fix.
    source_id = f"{dataset}-v3-{times[0][:10]}"
    prev = read_state(PRODUCT)
    if prev.get("source_id") == source_id \
            and prev.get("render_version") == RENDER_VERSION \
            and prev.get("data_times") == times \
            and prev.get("dataset") == dataset \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live", KML_FILE)):
        print(f"[{PRODUCT}] source unchanged ({dataset} {times[0]}); "
              f"keeping current raster.")
        return _refresh_kml()
    try:
        return _build(bounds, times, dataset)
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
        print(f"[{PRODUCT}] KML base URLs refreshed.")
    except Exception as e:
        print(f"[{PRODUCT}] WARNING: KML refresh failed: {e}")
    return 0


def _build(bounds, times, dataset):
    from build_chlorophyll import _bin_grid
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    lo, hi = CONFIG["valid_min"], CONFIG["valid_max"]
    acc = None
    grids = {}
    for t in times:
        try:
            la, lo_n, g = fetch_csv(dataset, VAR, t, bounds["lat_min"],
                                    bounds["lat_max"], bounds["lon_min"],
                                    bounds["lon_max"])
        except Exception as e:
            print(f"[{PRODUCT}] WARNING: {t} unavailable: {str(e)[:120]}")
            continue
        v = np.where((g >= lo) & (g <= hi), g, np.nan)
        acc = v if acc is None else np.where(np.isfinite(acc), acc, v)
        grids[t] = (la, lo_n)
    if acc is None:
        print(f"[{PRODUCT}] VALIDATION FAILED: no daily files. Keeping previous.")
        return 2
    _la, _lo = next(iter(grids.values()))
    n_valid = int(np.isfinite(acc).sum())
    print(f"[{PRODUCT}] mosaic {times[0]}..{times[-1]}: valid={n_valid}")
    if n_valid < 500:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few valid cells. Keeping previous.")
        return 2

    field = _bin_grid(_la, _lo, acc, bounds, W, H)
    vals = field[np.isfinite(field)]
    cur_min, cur_max = float(vals.min()), float(vals.max())
    print(f"[{PRODUCT}] current range=[{cur_min:.4g},{cur_max:.4g}]")

    rec, res = load_record(PRODUCT)
    if rec is None:
        print(f"[{PRODUCT}] cold start: seeding history from this mosaic.")
    sample = vals[::max(1, vals.size // 20000)][:20000]
    rec, res = update_record(rec, res, sample)
    if not (lo <= rec["hist_min"] and rec["hist_max"] <= hi):
        raise ValueError("record extrema outside source valid range")
    # Fixed log-spaced absolute scale (not the drifting historical record):
    # the same KdPAR always shows the same color. Record stats are still
    # tracked below for QC.
    stops = KDPAR_LOG_STOPS
    rgba = render_rgba(field, stops, bounds["overlay_alpha"])
    rgba = apply_shoreline_mask(rgba)  # water-only product
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    if int((rgba[:, :, 3] > 0).sum()) < 100:
        print(f"[{PRODUCT}] VALIDATION FAILED: empty raster. Keeping previous.")
        return 2

    p = rec["percentiles"]
    unit = CONFIG["display_units"]
    labels = KDPAR_LOG_TICKS
    subtitle = (f"Kd(PAR) ({unit}) — larger = more turbid  |  "
                f"{times[0][:10]} (+{MOSAIC_DAYS - 1}d mosaic)")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, stops, labels,
        f"Source: NOAA CoastWatch VIIRS KdPAR  |  Processed {now_det_str()}",
        note="Transparent = land/cloud/missing.")
    scale_html = (f"Diffuse attenuation coefficient for PAR ({unit}), "
                  f"FIXED log-spaced scale <b>LOWEST 0.02</b> "
                  f"clearest (dark blue) → 0.1 → 0.3 → <b>1.0</b> turbid (red) → "
                  f"<b>HIGHEST+ 5+</b> most turbid (violet). "
                  f"Larger values always mean murkier water; "
                  f"source values are never altered.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"],
        CONFIG["variable"] + f"; mosaic {times[0][:10]}..{times[-1][:10]}",
        data_time_utc=f"{iso_to_det(times[0])} (newest of {MOSAIC_DAYS}-day mosaic)",
        source_last_modified_utc="n/a (ERDDAP)",
        units=f"{unit} (display); source m^-1",
        source_resolution="~4 km VIIRS L3, bilinear-resampled to canvas",
        color_min=KDPAR_LOG_MIN, color_max=KDPAR_LOG_MAX, color_units=unit,
        missing_data_treatment=("cloud/land/fill (NaN) transparent; only "
                                f"[{lo},{hi}] values admitted; never interpolated."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["historical"] = {"low": rec["hist_min"], "high": rec["hist_max"],
                          "percentiles": p, "n_obs": rec["n_obs"],
                          "extends": rec.get("extends", [])}
    meta["stats"] = {"valid_cells": n_valid, "current_min": cur_min,
                     "current_max": cur_max}
    meta["dataset"] = dataset
    # v2 = bilinear canvas resample (same dashed-stripe fix as
    # chlorophyll; shared _bin_grid). One-time rotation.
    source_id = f"{dataset}-v3-{times[0][:10]}"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    token = meta["source_version"]
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{CONFIG['what']}</p>"
        f"<p><img src=\"{legend_block_src(PRODUCT, token)}\" width=\"600\" "
        f"alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Exact variable:</b> {CONFIG['variable']}<br/>"
        f"<b>Units:</b> {unit}<br/><b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> daily composites<br/>"
        f"<b>Data time:</b> {iso_to_det(times[0])} (mosaic {times[0][:10]}..{times[-1][:10]})<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Why turbid water turns red/purple:</b> {CONFIG['why_extreme']}<br/>"
        f"<b>Provenance:</b> <a href=\"{CONFIG['source_url']}\">ERDDAP dataset</a></p>")
    meta["folder_html"] = folder_html
    write_metadata(stage_prod, meta)
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        folder=(CONFIG["title"], folder_html),
        out_dirs=live_out_dirs(stage, KML_FILE)["live"])
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"],
        out_dirs=live_out_dirs(stage, KML_FILE)["entry"])

    save_record(PRODUCT, rec, res)
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"data_times": times,
                          "dataset": dataset,
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
