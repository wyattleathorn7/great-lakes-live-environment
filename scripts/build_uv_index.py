"""Pipeline M — LIVE UV INDEX (independent).

NOAA/NCEP CPC operational Global UV Index GRIB2, from the 12 UTC GFS run
(hourly forecast steps ``uv.t12z.grbfHH.grib2``, HH=01..120, under daily
``uvi.YYYYMMDD/`` dirs) -> verify parameter metadata -> convert the filed
erythemal flux (W/m^2) to UV Index (x40, exactly once) -> select the
forecast hour nearest the current time -> bilinear resample of the native
regular lat/lon grid onto the canvas (continuous, no holes) -> FULL BASIN
RECTANGLE (lon -93..-73.5, lat 40.5..49.5: land and water both paint, same
footprint as cloud cover and surface pressure; only missing data is
transparent) -> fixed absolute 0..11+ EPA-style gradient (same index
always shows the same color) -> transparent PNG -> key image + metadata
-> Folder live KML + stable entry KML.

The UV source is a FORECAST field, not independent hourly satellite
observations: one 12Z run per day provides all 120 hourly steps. A new
raster is published when (a) a new 12Z run appears, or (b) the selected
forecast hour advances. Identical (run, hour) -> keep previous raster.

Exit codes: 0 updated (or skipped); 2 source/validation failure (previous
kept); 1 unexpected error.
"""

import datetime as dt
import json
import os
import sys
import traceback
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR,
                              base_metadata, download, fmt_det, load_bounds,
                              now_det_str,
                              promote_stage, read_state, resample_gridded,
                              save_png, source_token, stage_dir, utcnow_iso,
                              write_metadata, write_state)
from gradient_scale import draw_scale_legend, render_rgba

PRODUCT = "uv_index"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_UV_Index.kml"
OVERLAY_NAME = "\U0001F323 LIVE UV INDEX"
SKIP_NOTE = "Turn on/off independently of all other layers."
UA = {"User-Agent": "great-lakes-live-environment/1.0"}

# Fixed absolute EPA-style scale: same index -> same color, always.
UV_STOPS = [
    (0.0, (46, 125, 50)),    # 0 low green
    (2.0, (102, 187, 106)),  # 2 low green
    (3.0, (249, 168, 37)),   # 3 moderate yellow
    (5.0, (251, 192, 45)),   # 5 moderate yellow
    (6.0, (239, 108, 0)),    # 6 high orange
    (7.0, (230, 81, 0)),     # 7 high orange
    (8.0, (229, 57, 53)),    # 8 very high red
    (10.0, (198, 40, 40)),   # 10 very high red
    (11.0, (106, 27, 154)),  # 11+ extreme violet
    (12.0, (59, 10, 90)),    # 12+ extreme deep purple
]
UV_MAX = 12.0
UV_LABELS = [(0.0, "LOWEST 0"), (2.0, "2 low"), (3.0, "3 mod."),
             (5.0, "5 mod."), (6.0, "6 high"), (7.0, "7 high"),
             (8.0, "8 v.high"), (10.0, "10 v.high"),
             (11.0, "11+ extreme"), (12.0, "HIGHEST+ 12")]


def candidate_runs(now):
    """Newest-first (datestr, run_datetime) pairs for the daily 12Z run."""
    cands = []
    for back in range(0, 4):
        day = now - dt.timedelta(days=back)
        datestr = day.strftime("%Y%m%d")
        cands.append((datestr, dt.datetime(day.year, day.month, day.day,
                                           12, tzinfo=dt.timezone.utc)))
    return cands


def select_hour(now, run_dt):
    """Forecast hour nearest the current time, clamped to [1, 120]."""
    fh = int(round((now - run_dt).total_seconds() / 3600.0))
    return max(1, min(120, fh))


def read_uvi(path):
    """Return (flux_2d, lats_2d, lons_2d, dataDate, dataTime, step)
    for the UV field, verifying GRIB2 parameter metadata first."""
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    with open(path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                name = codes_get(h, "name")
                if "UV" not in str(name).upper():
                    continue
                units = str(codes_get(h, "units"))
                nx, ny = int(codes_get(h, "Nx")), int(codes_get(h, "Ny"))
                vals = codes_get_values(h).astype(float).reshape(ny, nx)
                lats = codes_get_array(h, "latitudes").astype(
                    float).reshape(ny, nx)
                lons = codes_get_array(h, "longitudes").astype(
                    float).reshape(ny, nx)
                lons = ((lons + 180) % 360) - 180
                return (vals, lats, lons, str(codes_get(h, "dataDate")),
                        str(codes_get(h, "dataTime")).zfill(4),
                        str(codes_get(h, "stepRange")), units,
                        str(codes_get(h, "shortName")))
            finally:
                codes_release(h)
    raise ValueError("UV field not found in GRIB2")


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    now = dt.datetime.now(dt.timezone.utc)
    raw_path = os.path.join(RAW_DIR, "uvi_current.grib2")
    got = None
    for datestr, run_dt in candidate_runs(now):
        fh = select_hour(now, run_dt)
        # A run published later the same day supersedes earlier hours:
        # prefer the newest run whose selected hour is already published
        # (files appear progressively; a missing file means not yet out).
        url = CONFIG["file_pattern"].format(date_dir=datestr,
                                            hour=fh)
        try:
            info = download(url, raw_path, timeout=120)
            if info["size_bytes"] < 20_000:
                print(f"[{PRODUCT}] {datestr} f{fh:02d} too small; "
                      f"trying older run.")
                continue
            got = (info, url, datestr, fh, run_dt)
            break
        except Exception as e:
            print(f"[{PRODUCT}] {datestr} f{fh:02d} unavailable: "
                  f"{str(e)[:120]}")
    if got is None:
        print(f"[{PRODUCT}] DOWNLOAD FAILED for all runs (keeping previous).")
        return 2
    try:
        return _build(got, raw_path, now)
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


def _build(got, raw_path, now):
    info, url, datestr, fh, run_dt = got
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    bounds = load_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]

    flux, lats, lons, data_date, data_time, step, units, short = \
        read_uvi(raw_path)
    print(f"[{PRODUCT}] GRIB2 shortName={short} units={units} "
          f"step={step} flux range=[{np.nanmin(flux):.4f},"
          f"{np.nanmax(flux):.4f}]")
    # The filed values are erythemal flux (W/m^2, global max < 1);
    # UV Index = flux x 40, applied exactly once. If a future file ever
    # carries the finished index already (values >> 1), do NOT scale again.
    fmax = float(np.nanmax(flux))
    factor = CONFIG["flux_to_index"]
    if fmax > 2.0:
        factor = 1.0
        print(f"[{PRODUCT}] values look like finished UV Index already "
              f"(max {fmax}); NOT scaling.")
    uvi = flux * factor
    valid = np.isfinite(uvi) & (uvi >= CONFIG["valid_min"]) \
        & (uvi <= CONFIG["valid_max"])
    if int(valid.sum()) < 1000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few valid cells. "
              f"Keeping previous.")
        return 2
    field_all = np.where(valid, uvi, np.nan)
    field = resample_gridded(field_all, lats, lons, bounds, (H, W))

    valid_time = run_dt + dt.timedelta(hours=fh)
    valid_str = fmt_det(valid_time)
    run_str = fmt_det(run_dt)
    # Source-aware gate: (run, forecast hour) is the observation id. The
    # selected hour advances ~hourly as time passes, so this product
    # legitimately rebuilds up to 24x/day; identical ids never rebuild.
    source_id = f"uvi-{datestr}-t12z-f{fh:02d}"
    prev = read_state(PRODUCT)
    if prev.get("source_id") == source_id \
            and prev.get("render_version") == RENDER_VERSION \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live", KML_FILE)):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        return _refresh_kml()

    okv = np.isfinite(field)
    n_valid = int(okv.sum())
    cur_max = float(field[okv].max()) if n_valid else 0.0
    print(f"[{PRODUCT}] run {run_str} f{fh:02d} valid {valid_str}: "
          f"canvas valid={n_valid} max UVI={cur_max:.1f}")
    if n_valid < 50_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few canvas cells. "
              f"Keeping previous.")
        return 2
    if cur_max > CONFIG["valid_max"]:
        print(f"[{PRODUCT}] VALIDATION FAILED: implausible max {cur_max}.")
        return 2

    rgba = render_rgba(field, UV_STOPS, bounds["overlay_alpha"])
    # NOTE: full basin rectangle (no shoreline cut) — same footprint as
    # cloud cover and surface pressure. Only missing data is transparent.
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    if n_opaque < 50_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: empty raster. Keeping previous.")
        return 2

    unit = CONFIG["display_units"]
    subtitle = (f"{unit} (forecast)  |  Valid {valid_str}  |  "
                f"Run {run_str} +{fh}h")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, UV_STOPS, UV_LABELS,
        f"Source: NOAA/NCEP CPC global UV (12Z GFS run)  |  "
        f"Processed {now_det_str()}",
        note="Forecast field, not a satellite observation. "
             "0-2 low, 3-5 moderate, 6-7 high, 8-10 very high, 11+ extreme.")
    scale_html = (f"UV Index ({unit}), fixed absolute scale: "
                  f"<b>LOWEST 0</b> (green, low) → 3-5 (yellow, moderate) → "
                  f"6-7 (orange, high) → 8-10 (red, very high) → "
                  f"<b>HIGHEST+ 11+</b> (violet, extreme). Same index always "
                  f"shows the same color. Current maximum over the lakes: "
                  f"<b>{cur_max:.1f}</b>.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=(f"valid {valid_str} (run {run_str}, forecast hour "
                       f"+{fh})"),
        source_last_modified_utc=info.get("http_last_modified")
        or "n/a (NOMADS)",
        units=f"{unit} (display); source erythemal flux W m^-2 x 40",
        source_resolution="~0.25 deg regular lat/lon (1440x721), "
                          "bilinear-resampled to canvas",
        color_min=0.0, color_max=UV_MAX, color_units=unit,
         missing_data_treatment=("source grid fully valid (global forecast); "
                                 "full basin rectangle, no shoreline cut; "
                                 "only missing data transparent; "
                                 "never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["source_run"] = run_str
    meta["forecast_hour"] = fh
    meta["valid_time_utc"] = valid_str
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["uv_scale_factor_applied"] = factor
    meta["stats"] = {"valid_cells": n_valid, "current_max_uvi": cur_max}
    token = meta["source_version"]
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{CONFIG['what']}</p>"
        f"<p><img src=\"{legend_block_src(PRODUCT, token)}\" width=\"600\" "
        f"alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Units:</b> {unit}<br/>"
        f"<b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Source run:</b> {run_str}<br/>"
        f"<b>Forecast hour:</b> +{fh}h<br/>"
        f"<b>Valid time:</b> {valid_str}<br/>"
        f"<b>Source version:</b> {source_id}<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Why high values turn red/purple:</b> {CONFIG['why_extreme']}<br/>"
        f"<b>Provenance:</b> "
        f"<a href=\"{CONFIG['source_url']}\">CPC UV documentation</a></p>")
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

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"source_id": source_id,
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
