"""Pipeline H — LIVE CHLOROPHYLL / ALGAL ACTIVITY (independent).

NOAA CoastWatch S-NPP VIIRS chlorophyll-a (Science Quality, Global 4 km,
Daily; ERDDAP nesdisVHNSQchlaDaily; chlor_a, mg/m^3, OC3 algorithm) ->
validate -> newest-valid mosaic of the latest 3 daily composites (daily
ocean color is cloud-sparse) -> clip to Great Lakes -> continuous
historical-range gradient -> transparent PNG (water only, shared shoreline
mask) -> key image + metadata -> Folder KML.

Chlorophyll-a is a phytoplankton biomass/activity proxy, NOT a toxin
measurement and NOT a HAB diagnosis. Exit codes: 0 updated (or skipped);
2 source/validation failure (previous kept); 1 unexpected error.
"""

import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_kml,
                       description_html, legend_block,
                       refresh_kml_base_url)
from erddap_coastwatch import fetch_csv, latest_time
from geospatial_utils import (REPO_ROOT, SITE_DIR, apply_shoreline_mask,
                              base_metadata, load_bounds, promote_stage,
                              read_state, save_png, stage_dir, utcnow_iso,
                              write_metadata, write_state)
from gradient_scale import (anchor_values, build_stops, draw_scale_legend,
                            fmt_val, load_record, render_rgba, save_record,
                            update_record)

PRODUCT = "chlorophyll"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
DATASET = "nesdisVHNSQchlaDaily"
VAR = "chlor_a"
KML_FILE = "Great_Lakes_Live_Chlorophyll.kml"
OVERLAY_NAME = "\U0001F33F LIVE CHLOROPHYLL / ALGAL ACTIVITY"
SKIP_NOTE = "Turn on/off independently of all other layers."
MOSAIC_DAYS = 3


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def recent_times(n):
    """Newest n daily timestamps (ISO) from the dataset axis."""
    import datetime as dt
    end = latest_time(DATASET)
    base = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
    return [((base - dt.timedelta(days=i)).strftime("%Y-%m-%dT12:00:00Z"))
            for i in range(n)]


def run():
    bounds = load_bounds()
    try:
        times = recent_times(MOSAIC_DAYS)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    sig_src = "|".join(times)
    prev = read_state(PRODUCT)
    if prev.get("data_times") == times \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")):
        print(f"[{PRODUCT}] source unchanged ({times[0]}); keeping current raster.")
        return _refresh_kml()
    try:
        return _build(bounds, times)
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


def _build(bounds, times):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    lo, hi = CONFIG["valid_min"], CONFIG["valid_max"]
    # newest-valid-wins mosaic over the latest daily composites
    acc = None
    for t in times:
        try:
            la, lo_n, g = fetch_csv(DATASET, VAR, t, bounds["lat_min"],
                                    bounds["lat_max"], bounds["lon_min"],
                                    bounds["lon_max"])
        except Exception as e:
            print(f"[{PRODUCT}] WARNING: {t} unavailable: {str(e)[:120]}")
            continue
        v = np.where((g >= lo) & (g <= hi), g, np.nan)
        acc = v if acc is None else np.where(np.isfinite(acc), acc, v)
        if acc is not None:
            _la, _lo = la, lo_n
    if acc is None:
        print(f"[{PRODUCT}] VALIDATION FAILED: no daily files. Keeping previous.")
        return 2
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
    stops = build_stops(anchor_values(rec), CONFIG["allow_negative"])
    rgba = render_rgba(field, stops, bounds["overlay_alpha"])
    rgba = apply_shoreline_mask(rgba)  # water-only product
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    if int((rgba[:, :, 3] > 0).sum()) < 100:
        print(f"[{PRODUCT}] VALIDATION FAILED: empty raster. Keeping previous.")
        return 2

    p = rec["percentiles"]
    unit = CONFIG["display_units"]
    labels = [(rec["hist_min"], f"LOWEST {fmt_val(rec['hist_min'])}"),
              (p["p25"], fmt_val(p["p25"])),
              (p["p50"], fmt_val(p["p50"])),
              (p["p75"], fmt_val(p["p75"])),
              (rec["hist_max"], f"HIGHEST+ {fmt_val(rec['hist_max'])}")]
    subtitle = (f"Chlorophyll-a ({unit})  |  {times[0][:10]} (+{MOSAIC_DAYS - 1}d mosaic)")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, stops, labels,
        f"Source: NOAA CoastWatch VIIRS chlorophyll  |  Processed {utcnow_iso()}",
        note="Transparent = land/cloud/missing. Not a toxin measurement.")
    scale_html = (f"Chlorophyll-a concentration ({unit}), continuous: "
                  f"<b>LOWEST {fmt_val(rec['hist_min'])}</b> (dark blue) → "
                  f"common {fmt_val(p['p50'])} (green/yellow) → "
                  f"<b>HIGHEST+ {fmt_val(rec['hist_max'])}</b> (deep purple). "
                  f"High values compress into red/purple; they indicate "
                  f"biomass/activity, not toxins.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"],
        CONFIG["variable"] + f"; mosaic {times[0][:10]}..{times[-1][:10]}",
        data_time_utc=f"{times[0]} (newest of {MOSAIC_DAYS}-day mosaic)",
        source_last_modified_utc="n/a (ERDDAP)",
        units=f"{unit} (display); source mg m^-3",
        source_resolution="~4 km VIIRS L3 (0.0375 deg), binned to canvas (nearest)",
        color_min=rec["hist_min"], color_max=rec["hist_max"], color_units=unit,
        missing_data_treatment=("cloud/land/fill (NaN) transparent; only "
                                f"[{lo},{hi}] values admitted; never interpolated."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["historical"] = {"low": rec["hist_min"], "high": rec["hist_max"],
                          "percentiles": p, "n_obs": rec["n_obs"],
                          "extends": rec.get("extends", [])}
    meta["stats"] = {"valid_cells": n_valid, "current_min": cur_min,
                     "current_max": cur_max}
    token = meta["processing_time_utc"].replace(" ", "_").replace(":", "")
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{CONFIG['what']}</p>"
        f"<p><img src=\"{legend_block_src(PRODUCT, token)}\" width=\"600\" "
        f"alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Units:</b> {unit}<br/><b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> daily composites<br/>"
        f"<b>Data time:</b> {times[0]} (mosaic {times[0][:10]}..{times[-1][:10]})<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Why high values turn red/purple:</b> {CONFIG['why_extreme']}<br/>"
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
        out_dirs=[os.path.join(stage, "kml", KML_FILE),
                  os.path.join(stage, "site", "kml", KML_FILE)])
    assert_no_vector_geometry(kml_text)

    save_record(PRODUCT, rec, res)
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"data_times": times,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def legend_block_src(product, token=None):
    from build_kml import pages_base
    base = pages_base()
    v = f"?v={token}" if token else ""
    return f"{base}/{product}/legend.png{v}"


def _bin_grid(lats1d, lons1d, grid, bounds, W, H):
    """Bin a regular ERDDAP grid onto the canvas (nearest + splat)."""
    from geospatial_utils import bin_to_canvas, canvas_indices
    yy, xx = np.meshgrid(lats1d, lons1d, indexing="ij")
    rows, cols, valid = canvas_indices(yy.ravel(), xx.ravel(), bounds)
    field, _c = bin_to_canvas(rows, cols, np.asarray(grid).ravel(), valid,
                              (H, W), splat_radius=3)
    return field


if __name__ == "__main__":
    sys.exit(main())
