"""Pipeline H — LIVE CHLOROPHYLL / ALGAL ACTIVITY (independent).

NOAA CoastWatch S-NPP VIIRS chlorophyll-a (Science Quality, Global 4 km,
Daily; ERDDAP nesdisVHNSQchlaDaily; chlor_a, mg/m^3, OC3 algorithm) ->
validate -> 7-day MEDIAN mosaic of the latest daily composites (daily
ocean color is cloud-sparse and single days carry row striping, so each
pixel shows the median valid observation within the window, which
cancels day-calibration steps and swath seams) -> clip to Great Lakes ->
balanced LINEAR historical-range gradient (every part of the value scale
owns an equal share of the color resolution) -> light display smoothing
(NaN-aware local mean, valid pixels only, same documented treatment as
the live wind splat) -> transparent PNG (water only, shared shoreline
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
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from erddap_coastwatch import fetch_csv, latest_time
from geospatial_utils import (REPO_ROOT, SITE_DIR, apply_shoreline_mask,
                               base_metadata, load_bounds, load_watermask, promote_stage,
                               RENDER_VERSION, iso_to_det, read_state, save_png, source_token, stage_dir,
                               now_det_str, utcnow_iso, write_metadata, write_state)
from gradient_scale import (build_linear_stops, draw_scale_legend,
                            fmt_val, load_record, record_tick_labels,
                            render_rgba, save_record, update_record)

PRODUCT = "chlorophyll"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
# Live source priority (verified 2026-09-23): the NRT gapfilled
# S-NPP+NOAA-20 daily product is the freshest operational stream
# (newest 2026-09-20); the Science Quality daily is the slower fallback
# (newest 2026-09-13, ~10 d production latency). NOAA-21 is not yet in
# CoastWatch ERDDAP; the S-NPP+NOAA-20 NRT stream is the operational
# coverage until it is (no fabricated NOAA-21 data).
DATASETS = ["nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily",
            "nesdisVHNSQchlaDaily"]
VARS = {"nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily": "chlor_a",
        "nesdisVHNSQchlaDaily": "chlor_a"}
KML_FILE = "Great_Lakes_Live_Chlorophyll.kml"
OVERLAY_NAME = "\U0001F33F LIVE CHLOROPHYLL / ALGAL ACTIVITY"
SKIP_NOTE = "Turn on/off independently of all other layers."
MOSAIC_DAYS = 7
STRIDE = 1  # Full ERDDAP source resolution (0.0833 deg): stride 2 threw
            # away 3/4 of the cells and rendered real gradients as 18 km
            # tall stripes/blocks. Fetches stay small (~109x235 x 7 days).


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def recent_times(n):
    """Newest n daily timestamps (ISO) + the dataset that owns the newest.

    Newest-wins across DATASETS: every candidate's time axis is probed and
    the freshest end date wins, so the mosaic can never strand on a stale
    primary while a fallback has newer data. Unreachable candidates are
    skipped; per-day gaps fall back across datasets in _build.
    """
    import datetime as dt
    best = None  # (end_dt, dataset)
    errors = []
    for cand in DATASETS:
        try:
            end = latest_time(cand)
            end_dt = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
            if best is None or end_dt > best[0]:
                best = (end_dt, cand)
        except Exception as e:
            print(f"[{PRODUCT}] dataset {cand} time-axis probe failed: "
                  f"{str(e)[:100]}")
            errors.append(e)
    if best is None:
        raise errors[0] if errors else RuntimeError(
            "no chlorophyll dataset reachable")
    base, dataset = best
    print(f"[{PRODUCT}] freshest source: {dataset} "
          f"({base.strftime('%Y-%m-%dT12:00:00Z')})")
    return ([((base - dt.timedelta(days=i)).strftime("%Y-%m-%dT12:00:00Z"))
             for i in range(n)], dataset)


def run():
    bounds = load_bounds()
    try:
        times, dataset = recent_times(MOSAIC_DAYS)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    # v4 marker forces one rebuild to deploy the stripe fix (stride 1 +
    # 7-day median mosaic + display smoothing).
    source_id = f"{dataset}-v4-{times[0][:10]}"
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
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    lo, hi = CONFIG["valid_min"], CONFIG["valid_max"]
    # 7-day median mosaic over the latest daily composites: per-pixel
    # median of the valid observations in the window. Single days carry
    # row striping and one-day clear patches; newest-wins stitching
    # printed those as stripes and rectangular seams, while the median
    # cancels day-calibration steps (same days in, typical value out).
    import warnings as _warnings
    stack = []
    day_sources = {}
    # Newest dataset first, then the rest: a day missing on the primary is
    # filled from whichever candidate has it (cloud gaps / short outages).
    order = [dataset] + [d for d in DATASETS if d != dataset]
    for t in times:
        g = None
        for ds in order:
            try:
                la, lo_n, g = fetch_csv(ds, VARS[ds], t, bounds["lat_min"],
                                         bounds["lat_max"], bounds["lon_min"],
                                         bounds["lon_max"], stride=STRIDE)
                day_sources[t] = ds
                if ds != dataset:
                    print(f"[{PRODUCT}] {t} filled from fallback {ds}")
                break
            except Exception as e:
                print(f"[{PRODUCT}] WARNING: {t} on {ds} unavailable: "
                      f"{str(e)[:120]}")
        if g is None:
            continue
        v = np.where((g >= lo) & (g <= hi), g, np.nan)
        stack.append(v)
        _la, _lo = la, lo_n
    if not stack:
        print(f"[{PRODUCT}] VALIDATION FAILED: no daily files. Keeping previous.")
        return 2
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        acc = np.nanmedian(np.array(stack), axis=0)
    n_valid = int(np.isfinite(acc).sum())
    print(f"[{PRODUCT}] median mosaic {times[0]}..{times[-1]} "
          f"({len(stack)}/{len(times)} days): valid={n_valid}")
    if n_valid < 500:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few valid cells. Keeping previous.")
        return 2

    field = _bin_grid(_la, _lo, acc, bounds, W, H)
    vals = field[np.isfinite(field)]
    cur_min, cur_max = float(vals.min()), float(vals.max())
    print(f"[{PRODUCT}] current range=[{cur_min:.4g},{cur_max:.4g}]")
    # Display smoothing AFTER the record sample below is drawn from the
    # raw field: the history keeps true observed values, only the paint
    # is softened (no invented precision, same treatment as wind splat).

    rec, res = load_record(PRODUCT)
    if rec is None:
        print(f"[{PRODUCT}] cold start: seeding history from this mosaic.")
    sample = vals[::max(1, vals.size // 20000)][:20000]
    rec, res = update_record(rec, res, sample)
    if not (lo <= rec["hist_min"] and rec["hist_max"] <= hi):
        raise ValueError("record extrema outside source valid range")
    # Log-scale color mapping (ocean-color standard): chlorophyll spans
    # orders of magnitude, so a linear scale would paint 99% of lake
    # water one blue. Colors are linear in log10(concentration) -- equal
    # color per decade -- while every displayed value stays the true
    # measured concentration (no value is altered for color).
    import math as _math
    log_min = _math.log10(max(lo, rec["hist_min"]))
    log_max = _math.log10(max(rec["hist_max"],
                              10.0 * max(lo, rec["hist_min"])))
    if not (log_max > log_min):
        log_max = log_min + 1.0
    stops = build_linear_stops(log_min, log_max)
    field = _display_smooth(field)
    with np.errstate(invalid="ignore", divide="ignore"):
        logfield = np.where(np.isfinite(field) & (field > 0),
                            np.log10(np.maximum(field, lo)), np.nan)
    rgba = render_rgba(logfield, stops, bounds["overlay_alpha"])
    # Chlorophyll-only shoreline treatment: ocean-color pixels adjacent
    # to land (and sub-cell inland ponds) are land-contaminated by the
    # sensor's footprint, so the shared mask is eroded one canvas pixel
    # (~1.2 km) before the standard hard cut. This drops the isolated
    # red specks on inland ponds and the adjacency-brightened shore ring
    # while keeping every lake interior and nearshore bloom intact.
    rgba[:, :, 3] = np.where(_eroded_water_mask(), rgba[:, :, 3], 0.0)
    # Hard shoreline clip: majority-land pixels go fully transparent so no
    # fringe blocks sit on shore at high zoom (water-only product).
    rgba = apply_shoreline_mask(rgba, hard_cut=True)
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    if int((rgba[:, :, 3] > 0).sum()) < 100:
        print(f"[{PRODUCT}] VALIDATION FAILED: empty raster. Keeping previous.")
        return 2

    p = rec["percentiles"]
    unit = CONFIG["display_units"]
    # Ticks show true concentrations, positioned at their log10 places.
    labels = [(float(_math.log10(max(v, lo))), t)
              for v, t in record_tick_labels(rec)]
    subtitle = (f"Chlorophyll-a ({unit})  |  {times[0][:10]} (7-day median mosaic)")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, stops, labels,
        f"Source: NOAA CoastWatch VIIRS chlorophyll  |  Processed {now_det_str()}",
        note="Transparent = land/cloud/missing. Not a toxin measurement.")
    scale_html = (f"Chlorophyll-a concentration ({unit}), logarithmic "
                  f"(base-10) scale: <b>LOWEST {fmt_val(rec['hist_min'])}</b> "
                  f"(dark blue) → common {fmt_val(p['p50'])} (green/yellow) → "
                  f"<b>HIGHEST+ {fmt_val(rec['hist_max'])}</b> (deep purple). "
                  f"Equal color per decade of concentration (ocean-color "
                  f"standard); displayed values are true concentrations, "
                  f"never altered. High values indicate biomass/activity, "
                  f"not toxins.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"],
        CONFIG["variable"] + f"; mosaic {times[0][:10]}..{times[-1][:10]}",
        data_time_utc=f"{iso_to_det(times[0])} (7-day median mosaic ending {times[0][:10]})",
        source_last_modified_utc="n/a (ERDDAP)",
        units=f"{unit} (display); source mg m^-3",
        source_resolution="~9 km VIIRS NRT (0.0833 deg), 7-day median mosaic, NaN-aware display smoothing",
        color_min=rec["hist_min"], color_max=rec["hist_max"], color_units=unit,
        missing_data_treatment=("cloud/land/fill (NaN) transparent; only "
                                f"[{lo},{hi}] values admitted; never interpolated; "
                                "colors follow log10(concentration), values shown raw."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["historical"] = {"low": rec["hist_min"], "high": rec["hist_max"],
                          "percentiles": p, "n_obs": rec["n_obs"],
                          "extends": rec.get("extends", [])}
    meta["stats"] = {"valid_cells": n_valid, "current_min": cur_min,
                     "current_max": cur_max}
    meta["dataset"] = dataset
    meta["mosaic_sources"] = day_sources
    meta["mosaic_method"] = f"per-pixel median of {len(stack)} daily composites"
    # v4 = stripe fix: full-resolution source, 7-day median mosaic,
    # display smoothing. One-time rotation to deploy the fixed rendering;
    # afterwards the id tracks source dataset+date only.
    source_id = f"{dataset}-v4-{times[0][:10]}"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    token = meta["source_version"]
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{CONFIG['what']}</p>"
        f"<p><img src=\"{legend_block_src(PRODUCT, token)}\" width=\"600\" "
        f"alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Units:</b> {unit}<br/><b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> daily composites<br/>"
        f"<b>Data time:</b> {iso_to_det(times[0])} (7-day median mosaic)<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Why high values turn red/purple:</b> {CONFIG['why_extreme']}<br/>"
        f"<b>Provenance:</b> <a href=\"{CONFIG['source_url']}\">ERDDAP dataset</a></p>")
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
        out_dirs=outs["live"],
        meta=meta)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    save_record(PRODUCT, rec, res)
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"data_times": times,
                          "dataset": dataset,
                          "mosaic_sources": day_sources,
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


def _display_smooth(field, radius=6, min_count=3):
    """NaN-aware local-mean smoothing for display only (numpy-only).

    Each valid pixel becomes the mean of the valid pixels in its
    (2r+1)x(2r+1) neighbourhood; NaN pixels stay NaN, so coverage never
    grows and no value is invented. Radius 6 (~7 km) matches the ~9 km
    source cells: smaller windows leave the cell-row banding visible as
    horizontal lines, while 13x13 still preserves every real bloom
    (Green Bay, western Erie). Same documented display treatment as the
    live wind splat (bin_to_canvas splat_radius).
    """
    f = np.asarray(field, dtype=float)
    m = np.isfinite(f)
    if not m.any():
        return f
    r = int(radius)
    fw = np.where(m, f, 0.0)
    mw = m.astype(float)
    pad_f = np.pad(fw, r)
    pad_m = np.pad(mw, r)
    acc_f = np.zeros_like(fw)
    acc_m = np.zeros_like(mw)
    for dr in range(2 * r + 1):
        for dc in range(2 * r + 1):
            acc_f += pad_f[dr:dr + f.shape[0], dc:dc + f.shape[1]]
            acc_m += pad_m[dr:dr + f.shape[0], dc:dc + f.shape[1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = acc_f / np.where(acc_m > 0, acc_m, np.nan)
    return np.where(m & (acc_m >= min_count), sm, f)


def _eroded_water_mask():
    """Shared water mask eroded one pixel (3x3 minimum, numpy-only).

    Chlorophyll-only: the 9 km sensor footprint contaminates water pixels
    touching land, so paint retreats one canvas pixel (~1.2 km) from every
    shore. Isolated sub-cell ponds vanish; lake interiors are untouched.
    """
    m = load_watermask() >= 0.5
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    e[1:, 1:] &= m[:-1, :-1]
    e[:-1, :-1] &= m[1:, 1:]
    e[1:, :-1] &= m[:-1, 1:]
    e[:-1, 1:] &= m[1:, :-1]
    return e


def _bin_grid(lats1d, lons1d, grid, bounds, W, H):
    """Resample a regular ERDDAP grid onto the canvas (bilinear).

    The old nearest-neighbour scatter + splat(radius=3) painted 7x7
    blocks on ~9.8px source spacing, leaving the horizontal dashed
    stripe gaps visible in Google Earth. Bilinear resampling of the
    native grid is continuous by construction at any source spacing;
    source NaNs (clouds/gaps) propagate as transparent, never filled.
    """
    from geospatial_utils import resample_gridded
    la = np.asarray(lats1d, dtype=float)
    lo = np.asarray(lons1d, dtype=float)
    g = np.asarray(grid, dtype=float)
    if g.shape != (la.size, lo.size):
        raise ValueError(f"ERDDAP grid shape {g.shape} != "
                         f"(lats {la.size}, lons {lo.size})")
    LO, LA = np.meshgrid(lo, la)
    return resample_gridded(g, LA, LO, bounds, (H, W))


if __name__ == "__main__":
    sys.exit(main())
