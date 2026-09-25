"""Pipeline J — LIVE SOLAR RADIATION (independent).

NOAA/NCEP RAP (Rapid Refresh) hourly surface DSWRF (downward short-wave
radiation flux, W/m^2): newest available hourly cycle with the value
valid for the current hour (analysis step when present, else the shortest
forecast lead) -> validate -> bin the native 13 km Lambert grid onto the
canvas -> FULL BASIN RECTANGLE (lon -93..-73.5, lat 40.5..49.5: land and
water both paint; only missing data is transparent) -> FIXED absolute
sequential gradient
(near-black night -> near-white extreme) -> transparent PNG -> key image
+ metadata -> Folder live KML + stable entry KML.

Modeled incoming sunlight energy at the surface (broadband: includes
ultraviolet, visible, and near-infrared solar energy). This is not a UV
Index and not a visible-light meter reading. An hourly weather-model
estimate, not a ground sensor measurement at every location.

Valid zero nighttime solar radiation is a VALID numerical value
(near-black), never confused with NoData (transparent). Missing source
pixels are transparent; they are never zero-filled and never hidden
with color.

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
                              base_metadata, bin_to_canvas, canvas_indices,
                              grib_stamp_to_det, load_bounds,
                              now_det_str, promote_stage,
                              read_state, save_png,
                              source_token, stage_dir, utcnow_iso,
                              write_metadata, write_state)
from gradient_scale import (SOLAR_FLUX_MAX, SOLAR_FLUX_STOPS,
                             SOLAR_FLUX_TICKS, draw_scale_legend, load_record,
                             render_rgba, save_record, update_record)

PRODUCT = "solar_radiation"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Solar_Radiation.kml"
OVERLAY_NAME = "\U0001F31E LIVE SOLAR RADIATION"
SKIP_NOTE = "Turn on/off independently of all other layers."

RAP_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod"
RAP_GRID = "awp252bgrbf"  # RAP native 13 km grid covering North America
UA = {"User-Agent": "great-lakes-live-environment/1.0"}


def rap_file_url(datestr, cycle, lead):
    return (f"{RAP_BASE}/rap.{datestr}/rap.t{cycle}z."
            f"{RAP_GRID}{lead:02d}.grib2")


def _idx_lines(url, timeout=60):
    req = urllib.request.Request(url + ".idx", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace").splitlines()


def latest_rap_valid_now(max_back_hours=10):
    """Newest RAP cycle whose DSWRF surface field is valid this hour.

    Walks back from the current hour: cycle = now - lead, file f{lead:02d},
    so nominal valid time (cycle + lead) is always the current hour and the
    lead is the shortest available. Returns
    (file_url, datestr, cycle, lead). Raises if none found.
    """
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0,
                                                   microsecond=0)
    last = None
    for back in range(max_back_hours + 1):
        cyc = now - dt.timedelta(hours=back)
        dd, cc, lead = (cyc.strftime("%Y%m%d"), cyc.strftime("%H"), back)
        url = rap_file_url(dd, cc, lead)
        try:
            idx = _idx_lines(url)
        except Exception as e:
            last = e
            continue
        if any(len(l.split(":")) > 4 and l.split(":")[3] == "DSWRF"
               and ":surface:" in l for l in idx):
            return url, dd, cc, lead
    raise last or RuntimeError("no RAP DSWRF field valid for current hour")


def fetch_rap_dswrf(file_url, dest):
    """Download only the DSWRF surface message via .idx byte range.
    Returns (start, end, step_hint)."""
    idx = _idx_lines(file_url)
    start = None
    for i, line in enumerate(idx):
        q = line.split(":")
        if len(q) > 4 and q[3] == "DSWRF" and ":surface:" in line:
            start = int(q[1])
            end = int(idx[i + 1].split(":")[1]) if i + 1 < len(idx) else None
            break
    if start is None:
        raise ValueError("DSWRF surface message not in .idx")
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


def read_rap_dswrf(path):
    """Return (vals_1d, lats_1d, lons_1d, dataDate, dataTime, stepRange)."""
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    with open(path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                if codes_get(h, "shortName") == "sdswrf":
                    vals = codes_get_values(h).astype(float)
                    lats = codes_get_array(h, "latitudes").astype(float)
                    lons = codes_get_array(h, "longitudes").astype(float)
                    lons = ((lons + 180) % 360) - 180
                    return (vals, lats, lons, str(codes_get(h, "dataDate")),
                            str(codes_get(h, "dataTime")).zfill(4),
                            str(codes_get(h, "stepRange")))
            finally:
                codes_release(h)
    raise ValueError("DSWRF message not found in RAP GRIB2")


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
        base, datestr, cycle, lead = latest_rap_valid_now()
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    # Source-aware gate: RAP cycle + lead (not the run time) controls
    # rebuilds. A new hourly cycle valid for the new hour always rebuilds.
    source_id = f"rap-{datestr}-t{cycle}z-f{lead:02d}"
    prev = read_state(PRODUCT)
    if prev.get("source_id") == source_id \
            and prev.get("render_version") == RENDER_VERSION \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live", KML_FILE)):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping.")
        return _refresh_kml()
    try:
        return _build(base, datestr, cycle, lead, source_id)
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


def _build(base, datestr, cycle, lead, source_id):
    bounds = load_bounds()
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    lo, hi = CONFIG["valid_min"], CONFIG["valid_max"]
    raw_path = os.path.join(RAW_DIR, "rap_solar_current.grib2")
    fetch_rap_dswrf(base, raw_path)
    vals, lats, lons, data_date, data_time, step_range = read_rap_dswrf(raw_path)
    data_time_utc = grib_stamp_to_det(data_date, data_time)
    # Bin the native 13 km Lambert grid onto the canvas (same approach as
    # the HRRR products): mean of source points per pixel, display
    # smoothing only.
    rows, cols, valid = canvas_indices(np.asarray(lats).ravel(),
                                       np.asarray(lons).ravel(), bounds)
    # RAP points land ~0.22 deg apart (~21 canvas px): wide splat for
    # solid coverage (display smoothing only; source precision unchanged).
    field, _counts = bin_to_canvas(rows, cols, np.asarray(vals).ravel(),
                                   valid, (H, W), splat_radius=11)
    okv = np.isfinite(field) & (field >= lo) & (field <= hi)
    n_valid = int(okv.sum())
    with np.errstate(invalid="ignore"):
        print(f"[{PRODUCT}] rap-{datestr} t{cycle}z f{lead:02d}: valid={n_valid} "
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
    # Fixed absolute sequential scale (not the drifting historical
    # record): the same flux always shows the same color, so intensity
    # reads at a glance. Record stats are still tracked below for QC.
    stops = SOLAR_FLUX_STOPS
    rgba = render_rgba(field, stops, bounds["overlay_alpha"])
    # NOTE: full basin rectangle (no shoreline cut). Only missing data
    # is transparent.
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
    labels = SOLAR_FLUX_TICKS
    night = cur_max < 5.0
    subtitle = (f"Incoming sunlight at the surface ({unit}, RAP hourly)  |  "
                f"{data_time_utc}" + ("  |  nighttime" if night else ""))
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        unit, stops, labels,
        f"Source: NOAA RAP {datestr} t{cycle}z (lead f{lead:02d})  |  "
        f"Processed {now_det_str()}",
        note="Night reads 0 (near-black), valid data. Hourly weather-model "
        f"estimate, not a ground sensor reading.")
    scale_html = (f"Surface downward shortwave solar radiation ({unit}), "
                  f"RAP hourly estimate, FIXED absolute scale 0&ndash;1000+: "
                  f"night <b>0</b> (near-black) &rarr; <b>150</b> &rarr; <b>300</b> "
                  f"&rarr; <b>450</b> &rarr; <b>600</b> &rarr; <b>750</b> &rarr; "
                  f"extreme <b>1000+</b> (near-white). <b>LOWEST 0</b>, "
                  f"<b>HIGHEST+ 1000+</b>. Same flux always shows the same "
                  f"color. Values shown exactly as observed; night reads zero; "
                  f"clouds reduce values quantitatively. Broadband sunlight: "
                  f"includes ultraviolet, visible, and near-infrared energy. "
                  f"This is not a UV Index and not a visible-light meter reading. "
                  f"Hourly weather-model estimate, not a ground sensor "
                  f"measurement at every location.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (NOMADS)",
        units=f"{unit} (display); source W m^-2",
        source_resolution="~13 km RAP native grid (awp252), "
                          "mean-binned to canvas (display smoothing only)",
        color_min=0.0, color_max=SOLAR_FLUX_MAX, color_units=unit,
         missing_data_treatment=("only [0,1400] admitted; full basin rectangle, "
                                 "no shoreline cut; missing analysis "
                                 "transparent; never interpolated across "
                                 "space."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{datestr} t{cycle}z"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["source_model"] = "NOAA/NCEP RAP (Rapid Refresh)"
    meta["forecast_lead_hours"] = lead
    meta["rap_step_range"] = step_range
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
        f"<b>Update:</b> RAP hourly cycles; value valid for the current hour<br/>"
        f"<b>Data time:</b> {data_time_utc}<br/>"
        f"<b>Source version:</b> {source_id}<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Provenance:</b> <a href=\"{CONFIG['source_url']}\">NOMADS RAP</a></p>")
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
                          "render_version": RENDER_VERSION,
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
