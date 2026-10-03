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
# Live source candidates (verified 2026-09-28): the legacy NRT id is back
# online (newest 2026-09-21) and currently FRESHER than the Science Quality
# daily (newest 2026-09-18, ~10 d production latency). Selection is
# newest-wins across candidates (see recent_times), never fixed priority,
# so the mosaic always tracks the freshest operational stream.
DATASETS = ["nesdisVHNkdparDaily", "nesdisVHNSQkdparDaily"]
VAR = "kd_par"
# Daily-fresh fallback of last resort (verified live 2026-09-28, newest
# 2026-09-27, publishes ~daily): MODIS Aqua Kd490 NRT
# (erdMH1kd4901day_R2022NRT, Kd_490, KD2 algorithm, valid 0.01-6.0 m^-1,
# no altitude dimension). DIFFERENT variable from KdPAR (attenuation at
# 490 nm, not PAR broadband) but same physical direction (larger = more
# turbid) and same units, so it shares the fixed log display scale.
# Used ONLY for mosaic days missing on BOTH KdPAR ids, and every such
# day is labeled in mosaic_sources + metadata fallback note.
K490_DATASET = "erdMH1kd4901day_R2022NRT"
K490_VAR = "Kd_490"
K490_MIN, K490_MAX = 0.01, 6.0
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
    """Newest n daily timestamps (ISO) + the dataset that owns the newest.

    Newest-wins across DATASETS: every candidate's time axis is probed and
    the freshest end date wins (a fixed priority order stranded the mosaic
    on SQ 2026-09-18 while NRT had 2026-09-21). Unreachable candidates are
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
        raise errors[0] if errors else RuntimeError("no KdPAR dataset reachable")
    base_kd, dataset = best
    # K490 fallback clock: MODIS Aqua NRT publishes ~daily and may be newer
    # than any KdPAR id (e.g. 2026-09-27 vs KdPAR 2026-09-21). The mosaic
    # anchors at the freshest of the two so new clarity information ships
    # every 24 h even while KdPAR stalls; KdPAR days still win wherever
    # both variables have the day (see _build).
    try:
        k490_end = latest_time(K490_DATASET)
        base_k490 = dt.datetime.fromisoformat(k490_end.replace("Z", "+00:00"))
    except Exception as e:
        print(f"[{PRODUCT}] K490 time-axis probe failed: {str(e)[:100]}")
        base_k490 = None
    if base_k490 is not None and base_k490 > base_kd:
        base, anchor = base_k490, K490_DATASET
    else:
        base, anchor = base_kd, dataset
    print(f"[{PRODUCT}] freshest KdPAR: {dataset} "
          f"({base_kd.strftime('%Y-%m-%dT12:00:00Z')}); "
          f"mosaic anchor: {anchor} ({base.strftime('%Y-%m-%dT12:00:00Z')})")
    return ([((base - dt.timedelta(days=i)).strftime("%Y-%m-%dT12:00:00Z"))
             for i in range(n)], anchor)


def run():
    bounds = load_bounds()
    try:
        times, anchor = recent_times(MOSAIC_DAYS)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    prev = read_state(PRODUCT)
    # Skip only when the anchor window is unchanged AND the previous mosaic
    # covered the whole window: transient per-day gaps (503s, missing
    # publication) are retried on the next hourly run instead of freezing
    # in. The final source_id (newest day actually present) is decided in
    # _build; comparing it here would rebuild every run.
    if prev.get("anchor") == anchor \
            and prev.get("render_version") == RENDER_VERSION \
            and prev.get("data_times") == times \
            and set(prev.get("mosaic_sources", {})) >= set(times) \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live", KML_FILE)):
        print(f"[{PRODUCT}] source unchanged ({anchor} {times[0]}); "
              f"keeping current raster.")
        return _refresh_kml()
    try:
        return _build(bounds, times, anchor)
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
    day_sources = {}
    day_vars = {}
    # Per-day source order: KdPAR ids newest-first (anchor first), then the
    # K490 daily fallback. KdPAR wins every day it has; K490 fills days
    # KdPAR lacks entirely (stalls/outages) so the mosaic still advances
    # ~daily. Different admission windows per variable.
    kdpar_order = ([dataset] + [d for d in DATASETS if d != dataset]
                   if dataset in DATASETS else list(DATASETS))
    for t in times:
        g = None
        for ds in kdpar_order:
            try:
                la, lo_n, g = fetch_csv(ds, VAR, t, bounds["lat_min"],
                                        bounds["lat_max"], bounds["lon_min"],
                                        bounds["lon_max"])
                day_sources[t] = ds
                day_vars[t] = VAR
                if ds != kdpar_order[0]:
                    print(f"[{PRODUCT}] {t} filled from fallback {ds}")
                break
            except Exception as e:
                print(f"[{PRODUCT}] WARNING: {t} on {ds} unavailable: "
                      f"{str(e)[:120]}")
        vmin, vmax = lo, hi
        if g is None:
            try:
                # Stride 1: this L3SMI dataset returns empty grids on
                # strided requests; the basin subset is small enough.
                la, lo_n, g = fetch_csv(
                    K490_DATASET, K490_VAR, t, bounds["lat_min"],
                    bounds["lat_max"], bounds["lon_min"], bounds["lon_max"],
                    altitude=False)
                day_sources[t] = K490_DATASET
                day_vars[t] = K490_VAR
                vmin, vmax = K490_MIN, K490_MAX
                print(f"[{PRODUCT}] {t} filled from K490 fallback "
                      f"({K490_DATASET})")
            except Exception as e:
                print(f"[{PRODUCT}] WARNING: {t} on {K490_DATASET} "
                      f"unavailable: {str(e)[:120]}")
        if g is None:
            continue
        v = np.where((g >= vmin) & (g <= vmax), g, np.nan)
        # Resample each day to the canvas BEFORE mosaicking: sources have
        # different native grids (KdPAR 241x521 vs K490 217x469) that
        # cannot be combined raw. Newest-valid-wins then runs on canvas.
        try:
            fday = _bin_grid(la, lo_n, v, bounds, W, H)
        except Exception as e:
            print(f"[{PRODUCT}] WARNING: {t} resample failed: "
                  f"{str(e)[:120]}")
            continue
        acc = fday if acc is None else np.where(np.isfinite(acc), acc, fday)
        grids[t] = True
    if acc is None:
        print(f"[{PRODUCT}] VALIDATION FAILED: no daily files. Keeping previous.")
        return 2
    # Honest newest: the anchor date may hold no data on any source (e.g.
    # K490 time axis newer than its retrievable days). All stamps below use
    # the newest day actually present in the mosaic, and the source id
    # tracks that day's owner — so the id only advances when real data does.
    eff_newest = max(day_sources)
    eff_owner = day_sources[eff_newest]
    # acc is already on-canvas (per-day resample, see above).
    field = acc
    _la, _lo = None, None
    n_valid = int(np.isfinite(field).sum())
    print(f"[{PRODUCT}] mosaic {eff_newest[:10]}..{times[-1][:10]} "
          f"(requested {times[0][:10]}..{times[-1][:10]}): "
          f"canvas-valid={n_valid}")
    if n_valid < 500:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few valid cells. Keeping previous.")
        return 2

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
    # Hard shoreline clip: majority-land pixels go fully transparent so no
    # fringe blocks sit on shore at high zoom (water-only product).
    rgba = apply_shoreline_mask(rgba, hard_cut=True)  # water-only product
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    if int((rgba[:, :, 3] > 0).sum()) < 100:
        print(f"[{PRODUCT}] VALIDATION FAILED: empty raster. Keeping previous.")
        return 2

    p = rec["percentiles"]
    unit = CONFIG["display_units"]
    labels = KDPAR_LOG_TICKS
    k490_days = sorted(t[:10] for t, s in day_sources.items()
                       if K490_DATASET in s)
    k490_note = (""
                 if not k490_days else
                 f" Mosaic day(s) {', '.join(k490_days)} from MODIS Aqua "
                 f"Kd490 NRT ({K490_VAR}, KD2, attenuation at 490 nm — "
                 f"same m^-1 units and larger=more-turbid direction as "
                 f"KdPAR; KdPAR stalled upstream).")
    subtitle = (f"Kd(PAR) ({unit}) — larger = more turbid  |  "
                f"{eff_newest[:10]} (+{MOSAIC_DAYS - 1}d mosaic)"
                f"{' +Kd490' if k490_days else ''}")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, stops, labels,
        f"Source: NOAA CoastWatch VIIRS KdPAR"
        f"{' + MODIS Kd490 fallback' if k490_days else ''}  |  "
        f"Processed {now_det_str()}",
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
        CONFIG["variable"] + f"; mosaic {eff_newest[:10]}..{times[-1][:10]}",
        data_time_utc=f"{iso_to_det(eff_newest)} (newest of {MOSAIC_DAYS}-day mosaic)",
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
    meta["dataset"] = eff_owner
    meta["mosaic_sources"] = day_sources
    meta["mosaic_variables"] = day_vars
    if k490_days:
        meta["fallback_note"] = (
            f"Mosaic day(s) {', '.join(k490_days)} use MODIS Aqua Kd490 NRT "
            f"({K490_DATASET}, {K490_VAR}, KD2 algorithm, valid "
            f"[{K490_MIN},{K490_MAX}] m^-1) because no KdPAR id published "
            f"those days. Kd490 is attenuation at 490 nm (not PAR "
            f"broadband) but shares units and larger=more-turbid "
            f"direction; values shown exactly as observed.")
    # v3 = fixed log-spaced absolute scale (same-value-same-color).
    # The id tracks the newest day ACTUALLY present and its owner, so it
    # advances exactly when real data does (never on empty anchor days).
    source_id = f"{eff_owner}-v3-{eff_newest[:10]}"
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
        f"<b>Data time:</b> {iso_to_det(eff_newest)} (mosaic {eff_newest[:10]}..{times[-1][:10]})<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"{'<b>Source fallback:</b> ' + meta['fallback_note'] + '<br/>' if k490_days else ''}"
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
        out_dirs=live_out_dirs(stage, KML_FILE)["live"],
        meta=meta)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"],
        out_dirs=live_out_dirs(stage, KML_FILE)["entry"])

    save_record(PRODUCT, rec, res)
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"data_times": times,
                          "anchor": dataset,
                          "dataset": eff_owner,
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


if __name__ == "__main__":
    sys.exit(main())
