"""Pipeline I — LIVE WATER CLARITY / TURBIDITY (independent).

NOAA CoastWatch S-NPP VIIRS Kd(PAR) diffuse attenuation (Near Real-Time,
Global 4 km, Daily; ERDDAP nesdisVHNkdparDaily; kd_par, m^-1, NOAA MECB
algorithm, product status Experimental) -> validate -> newest-valid mosaic
of the latest 7 daily composites (daily ocean color is cloud-sparse:
clouds and orbit gaps leave most water pixels empty on any single day,
so each pixel shows its newest valid observation within the window) ->
clip to Great Lakes -> fixed absolute KdPAR scale (same pattern as the
other working products: tuned anchors where lake water lives, raw values
render directly, no transforms) ->
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
from gradient_scale import (draw_scale_legend, load_record, render_rgba,
                            save_record, update_record)

PRODUCT = "water_clarity"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
# Live source candidates (verified 2026-09-28): the legacy NRT id is back
# online (newest 2026-09-21) and currently FRESHER than the Science Quality
# daily (newest 2026-09-18, ~10 d production latency). Selection is
# newest-wins across candidates (see recent_times), never fixed priority,
# so the mosaic always tracks the freshest operational stream.
DATASETS = ["nesdisVHNkdparDaily", "nesdisVHNSQkdparDaily"]
VAR = "kd_par"
# NOTE (2026-10-03): a MODIS Aqua Kd490 NRT daily fallback lived here briefly.
# Removed: Kd490 (attenuation at 490 nm) reads numerically LOWER than KdPAR
# (broadband PAR) for the same water, so mixed mosaics pinned whole lakes
# to the scale floor (dark navy) and drew visible square seams between the
# variables. One variable, one scale: KdPAR only, even when that means an
# older newest-day during KdPAR stalls. Honest staleness beats false color.
KML_FILE = "Great_Lakes_Live_Water_Clarity_Turbidity.kml"
OVERLAY_NAME = "\U0001F30A LIVE WATER CLARITY / TURBIDITY"
SKIP_NOTE = "Turn on/off independently of all other layers."
MOSAIC_DAYS = 7

# Fixed absolute KdPAR scale, built the same way as the other working
# products (cf. UV_STOPS, P_STOPS): hand-placed anchors in the master
# blue->purple family, tuned to where lake water actually lives
# (open-lake ~0.05-0.35 owns blue through yellow; plumes own orange/red;
# rare >1.3 extremes own magenta/deep-purple). Same value -> same color,
# always. Raw values render directly: no log transform, no smoothing.
# The floor EQUALS valid_min (0.016), so no valid observation can ever
# clamp into the floor color (the old above-valid floor painted valid
# clear water as dark-navy "holes" in Ontario/Superior).
CLARITY_STOPS = [
    (0.016, (16, 52, 140)),    # clearest: dark blue
    (0.060, (20, 110, 200)),   # blue
    (0.120, (20, 190, 200)),   # cyan
    (0.200, (90, 190, 80)),    # green
    (0.300, (180, 200, 60)),   # green-yellow
    (0.450, (245, 215, 50)),   # yellow
    (0.650, (240, 130, 25)),   # orange
    (0.900, (205, 30, 35)),    # red: turbid
    (1.300, (150, 25, 110)),   # red-violet
    (2.000, (90, 40, 160)),    # violet
    (5.000, (59, 10, 90)),     # HIGHEST+ deep purple (5+ clamps here)
]
CLARITY_MIN = 0.016
CLARITY_MAX = 5.0
CLARITY_LABELS = [
    (0.016, "LOWEST 0.016"),
    (0.200, "0.2"),
    (0.450, "0.45 typical"),
    (0.900, "0.9 turbid"),
    (5.000, "HIGHEST+ 5"),
]
# Never-again floor guard (mirrors the validate_outputs fixed-scale check):
# the scale floor must admit the lowest valid observation.
assert CLARITY_STOPS[0][0] <= float(CONFIG["valid_min"]), \
    "clarity floor above valid_min would re-create floor-color holes"
assert all(CLARITY_STOPS[i][0] < CLARITY_STOPS[i + 1][0]
           for i in range(len(CLARITY_STOPS) - 1)), \
    "clarity stops must strictly increase"


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
    base, anchor = best
    print(f"[{PRODUCT}] freshest KdPAR: {anchor} "
          f"({base.strftime('%Y-%m-%dT12:00:00Z')})")
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
    # Per-day source order: KdPAR ids newest-first (anchor first). One
    # variable only (see note at DATASETS): days missing on both ids stay
    # transparent rather than borrowing a different variable.
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
            continue
        v = np.where((g >= vmin) & (g <= vmax), g, np.nan)
    # Resample each day to the canvas BEFORE mosaicking: sources have
    # different native grids that cannot be combined raw. Newest-valid-wins
    # then runs on canvas.
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
    # Honest newest: the anchor date may hold no data on any source.
    # All stamps below use
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
    # Fixed absolute scale (same pattern as UV/pressure/cloud/aurora):
    # raw KdPAR values render directly through CLARITY_STOPS — no log
    # transform, no smoothing. Record stats are still tracked below for QC.
    stops = CLARITY_STOPS
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
    labels = CLARITY_LABELS
    subtitle = (f"Kd(PAR) ({unit}) — larger = more turbid  |  "
                f"{eff_newest[:10]} (+{MOSAIC_DAYS - 1}d mosaic)")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, stops, labels,
        f"Source: NOAA CoastWatch VIIRS KdPAR  |  "
        f"Processed {now_det_str()}",
        note="Transparent = land/cloud/missing.")
    scale_html = (f"Diffuse attenuation coefficient for PAR ({unit}), "
                  f"FIXED absolute scale <b>LOWEST 0.016</b> "
                  f"clearest (dark blue) → 0.2 → 0.45 → <b>0.9</b> turbid (red) → "
                  f"<b>HIGHEST+ 5</b> most turbid (deep purple). "
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
        color_min=CLARITY_MIN, color_max=CLARITY_MAX, color_units=unit,
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
    # v5 = working-product pattern: fixed absolute scale with tuned
    # anchors (UV/pressure style), raw-value rendering, floor == valid_min
    # so low-clamp holes are impossible. Forces one clean redeploy.
    # The id tracks the newest day ACTUALLY present and its owner, so it
    # advances exactly when real data does (never on empty anchor days).
    source_id = f"{eff_owner}-v5-{eff_newest[:10]}"
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
