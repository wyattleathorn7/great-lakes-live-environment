"""Pipeline J — LIVE SOLAR RADIATION (independent).

NOAA/NCEP GFS operational surface analysis, DSWRF (downward short-wave
radiation flux, W/m^2, analysis step f000 of the newest available 00/06/
12/18Z cycle) -> validate -> bilinear resample of the native Gaussian
grid onto the canvas (continuous field, no splat holes) -> clip to Great
Lakes water via the shared NOAA shoreline mask (water only; land is
transparent) -> balanced LINEAR historical-range gradient -> transparent
PNG -> key image + metadata -> Folder live KML + stable entry KML.

This is BROADBAND surface shortwave flux (W/m^2), not the UV index; the
scale shows measured flux exactly as observed.

Valid zero nighttime solar radiation is a VALID numerical value (deep
blue), never confused with NoData (transparent). Missing source pixels
are transparent; they are never zero-filled and never hidden with color.

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
from geospatial_utils import (REPO_ROOT, SITE_DIR, apply_shoreline_mask,
                              base_metadata, load_bounds, promote_stage,
                              read_state, resample_gridded, save_png,
                              source_token, stage_dir, utcnow_iso,
                              write_metadata, write_state)
from gradient_scale import (build_linear_stops, draw_scale_legend,
                            fmt_val, load_record, record_tick_labels,
                            render_rgba, save_record, update_record)

PRODUCT = "solar_radiation"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Solar_Radiation.kml"
OVERLAY_NAME = "\U0001F31E LIVE SOLAR RADIATION"
SKIP_NOTE = "Turn on/off independently of all other layers."

GFS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
GFS_CYCLES = ("18", "12", "06", "00")
UA = {"User-Agent": "great-lakes-live-environment/1.0"}


def gfs_sflux_url(datestr, cycle, fhour=0):
    return (f"{GFS_BASE}/gfs.{datestr}/{cycle}/atmos/"
            f"gfs.t{cycle}z.sfluxgrbf{fhour:03d}.grib2")


def latest_gfs_cycle(max_back_days=3):
    """Newest GFS cycle whose f000 sflux .idx carries DSWRF:surface:anl:.

    Returns (file_url, datestr, cycle). Raises if none found.
    """
    now = dt.datetime.now(dt.timezone.utc)
    last = None
    for back_day in range(max_back_days + 1):
        day = now - dt.timedelta(days=back_day)
        datestr = day.strftime("%Y%m%d")
        for cycle in GFS_CYCLES:
            url = gfs_sflux_url(datestr, cycle)
            try:
                req = urllib.request.Request(url + ".idx", headers=UA)
                with urllib.request.urlopen(req, timeout=40) as r:
                    idx = r.read().decode("utf-8",
                                          errors="replace").splitlines()
                if any(":DSWRF:surface:" in l and ":anl:" in l for l in idx):
                    return url, datestr, cycle
            except Exception as e:
                last = e
    raise last or RuntimeError("no GFS DSWRF analysis cycle found")


def fetch_dswrf(file_url, dest):
    """Download only the DSWRF:surface:anl: GRIB2 message via .idx range."""
    req = urllib.request.Request(file_url + ".idx", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        idx = r.read().decode("utf-8", errors="replace").splitlines()
    start = None
    for i, line in enumerate(idx):
        p = line.split(":")
        if len(p) > 5 and p[3] == "DSWRF" and "surface" in line \
                and ":anl:" in line:
            start = int(p[1])
            end = int(idx[i + 1].split(":")[1]) if i + 1 < len(idx) else None
            break
    if start is None:
        raise ValueError("DSWRF surface analysis message not in .idx")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    req = urllib.request.Request(
        file_url, headers={**UA, "Range": f"bytes={start}-"
                           f"{end - 1 if end else ''}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(body)
    os.replace(tmp, dest)
    return start, end


def read_dswrf(path):
    """Return (vals_2d, lats_2d, lons_2d, dataDate, dataTime, stepRange)."""
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    with open(path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                if codes_get(h, "shortName") == "sdswrf" \
                        and str(codes_get(h, "step")) == "0":
                    nx, ny = int(codes_get(h, "Nx")), int(codes_get(h, "Ny"))
                    vals = codes_get_values(h).astype(float).reshape(ny, nx)
                    lats = codes_get_array(h, "latitudes").astype(
                        float).reshape(ny, nx)
                    lons = codes_get_array(h, "longitudes").astype(
                        float).reshape(ny, nx)
                    lons = ((lons + 180) % 360) - 180
                    return (vals, lats, lons, str(codes_get(h, "dataDate")),
                            str(codes_get(h, "dataTime")).zfill(4),
                            str(codes_get(h, "stepRange")))
            finally:
                codes_release(h)
    raise ValueError("DSWRF analysis message (step=0) not found in GRIB2")


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    try:
        base, datestr, cycle = latest_gfs_cycle()
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    # Source-aware gate: GFS cycle id (not the run time) controls rebuilds.
    source_id = f"gfs-{datestr}-t{cycle}z-f000"
    prev = read_state(PRODUCT)
    if prev.get("source_id") == source_id \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live", KML_FILE)):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping.")
        return _refresh_kml()
    try:
        return _build(base, datestr, cycle, source_id)
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


def _build(base, datestr, cycle, source_id):
    bounds = load_bounds()
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    lo, hi = CONFIG["valid_min"], CONFIG["valid_max"]
    raw_path = os.path.join(RAW_DIR, "gfs_solar_current.grib2")
    fetch_dswrf(base, raw_path)
    vals, lats, lons, data_date, data_time, step_range = read_dswrf(raw_path)
    data_time_utc = (f"{data_date[0:4]}-{data_date[4:6]}-{data_date[6:8]} "
                     f"{data_time[0:2]}:{data_time[2:4]} UTC")
    # Bilinear resample of the native grid: continuous field by
    # construction (no nearest-neighbour splat gaps / stripe holes).
    field = resample_gridded(vals, lats, lons, bounds, (H, W))
    okv = np.isfinite(field) & (field >= lo) & (field <= hi)
    n_valid = int(okv.sum())
    with np.errstate(invalid="ignore"):
        print(f"[{PRODUCT}] {datestr} t{cycle}z f000: valid={n_valid} "
              f"range=[{field[okv].min():.1f},{field[okv].max():.1f}] W/m^2")
    if n_valid < 100_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few valid cells. "
              f"Keeping previous.")
        return 2
    vals_ok = field[okv]
    cur_min, cur_max = float(vals_ok.min()), float(vals_ok.max())

    rec, res = load_record(PRODUCT)
    if rec is None:
        print(f"[{PRODUCT}] cold start: seeding history from this analysis.")
    sample = vals_ok[::max(1, vals_ok.size // 20000)][:20000]
    rec, res = update_record(rec, res, sample)
    if not (lo <= rec["hist_min"] and rec["hist_max"] <= hi):
        raise ValueError("record extrema outside source valid range")
    stops = build_linear_stops(rec["hist_min"], rec["hist_max"])
    rgba = render_rgba(field, stops, bounds["overlay_alpha"])
    rgba = apply_shoreline_mask(rgba)  # water-only product
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    # Interior-hole guard: transparent pixels fully enclosed by opaque
    # water (stripe decoding artefacts) fail the run instead of shipping.
    n_holes = _interior_holes(rgba)
    print(f"[{PRODUCT}] opaque={n_opaque} interior_holes={n_holes}")
    if n_opaque < 100_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: empty raster. Keeping previous.")
        return 2
    if n_holes > 500:
        print(f"[{PRODUCT}] VALIDATION FAILED: {n_holes} enclosed transparent "
              f"holes inside water (decoding artefact). Keeping previous.")
        return 2

    p = rec["percentiles"]
    unit = CONFIG["display_units"]
    labels = record_tick_labels(rec)
    night = cur_max < 5.0
    subtitle = (f"Downward shortwave flux ({unit}, GFS f000 analysis)  |  "
                f"{data_time_utc}" + ("  |  nighttime" if night else ""))
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, stops, labels,
        f"Source: NOAA GFS {datestr} t{cycle}z analysis (f000)  |  "
        f"Processed {utcnow_iso()}",
        note="Zero is valid nighttime data (deep blue), not missing.")
    scale_html = (f"Surface downward shortwave solar radiation ({unit}, "
                  f"GFS f000 analysis step), balanced linear "
                  f"scale: <b>LOWEST {fmt_val(rec['hist_min'])}</b> (dark blue) → "
                  f"common {fmt_val(p['p50'])} → "
                  f"<b>HIGHEST+ {fmt_val(rec['hist_max'])}</b> (deep purple). "
                  f"Values shown exactly as observed; night reads zero; "
                  f"clouds reduce values quantitatively. Broadband flux, "
                  f"not the UV index.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (NOMADS)",
        units=f"{unit} (display); source W m^-2",
        source_resolution="~13 km GFS Gaussian grid (3072x1536), "
                          "bilinear-resampled to canvas",
        color_min=rec["hist_min"], color_max=rec["hist_max"], color_units=unit,
        missing_data_treatment=("only [0,1400] admitted; land outside the "
                                "NOAA shoreline and missing analysis "
                                "transparent; never interpolated across "
                                "land/water."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{datestr} t{cycle}z"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["gfs_step_range"] = step_range
    meta["historical"] = {"low": rec["hist_min"], "high": rec["hist_max"],
                          "percentiles": p, "n_obs": rec["n_obs"],
                          "extends": rec.get("extends", [])}
    meta["stats"] = {"valid_cells": n_valid, "current_min": cur_min,
                     "current_max": cur_max, "nighttime": night,
                     "interior_holes": n_holes}
    token = meta["source_version"]
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{CONFIG['what']}</p>"
        f"<p><img src=\"{legend_block_src(PRODUCT, token)}\" width=\"600\" "
        f"alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Variable:</b> {CONFIG['variable']}<br/>"
        f"<b>Units:</b> {unit} ({CONFIG['temporal']})<br/>"
        f"<b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> GFS 6-hourly analysis cycles (00/06/12/18Z)<br/>"
        f"<b>Data time:</b> {data_time_utc}<br/>"
        f"<b>Source version:</b> {source_id}<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Provenance:</b> <a href=\"{CONFIG['source_url']}\">NOMADS GFS</a></p>")
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

    save_record(PRODUCT, rec, res)
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "source_id": source_id,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def _interior_holes(rgba):
    """Transparent WATER pixels fully enclosed by opaque pixels.

    Flood-fills transparency from the image border on a 4x-downsampled
    grid. Only watermask-water pixels count: legitimate islands (Isle
    Royale, Manitoulin, ...) are mask-land and excluded. Enclosed
    transparent water is a decoding/resampling artefact (the stripe bug),
    not a shoreline edge (edges always connect to the border at lake
    scale).
    """
    from collections import deque
    from geospatial_utils import load_watermask
    wm = load_watermask()
    water = wm[::4, ::4] > 0.5
    a = (rgba[::4, ::4, 3] > 0)
    h, w = a.shape
    seen = np.zeros((h, w), dtype=bool)
    dq = deque()
    for c in range(w):
        for r in (0, h - 1):
            if not a[r, c] and not seen[r, c]:
                seen[r, c] = True
                dq.append((r, c))
    for r in range(h):
        for c in (0, w - 1):
            if not a[r, c] and not seen[r, c]:
                seen[r, c] = True
                dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not a[rr, cc] \
                    and not seen[rr, cc]:
                seen[rr, cc] = True
                dq.append((rr, cc))
    return int((~a & ~seen & water).sum())


def legend_block_src(product, token=None):
    from build_kml import pages_base
    base = pages_base()
    v = f"?v={token}" if token else ""
    return f"{base}/{product}/legend.png{v}"


if __name__ == "__main__":
    sys.exit(main())
