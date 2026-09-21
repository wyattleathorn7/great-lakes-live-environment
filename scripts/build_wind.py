"""Pipeline F — LIVE WIND (independent).

NCEP GLWU GRIB2 UGRD/VGRD surface analysis (same operational files as the
wave product, wind fields only) -> validate -> speed = sqrt(U^2+V^2) ->
Beaufort Force 0-12 -> blue->...->red->dark-purple(F12) gradient with
tiny movement-direction arrows rendered INTO the raster -> transparent PNG
(shared GSHHG shoreline mask) -> metadata -> KML.

Direction convention (critical): GRIB U/V components point in the direction
the air moves TOWARD (U eastward, V northward), unlike the meteorological
"FROM" convention, so arrows are drawn with NO reversal. See beaufort.py
and scripts/selftest.py (known-vector test).

Exit codes: 0 = updated; 2 = source/validation failure (previous valid raster
left untouched); 1 = unexpected error.
"""

import json
import math
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beaufort import (BEAUFORT, FORCE_COLORS, MS_TO_KT, force_from_kt,
                      force_name, force_range_text)
from build_kml import (assert_no_vector_geometry, build_kml,
                       description_html, legend_block,
                       refresh_kml_base_url)
from build_wave_height import candidate_urls
from geospatial_utils import (REPO_ROOT, SITE_DIR, apply_shoreline_mask,
                              base_metadata, download, draw_category_legend,
                              fetch_buoy_obs, load_bounds, promote_stage,
                              read_state, save_png, stage_dir, utcnow_iso,
                              write_metadata, write_state)

PRODUCT = "wind"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Wind.kml"
OVERLAY_NAME = "\U0001F4A8 LIVE WIND"
SKIP_NOTE = "Turn on/off independently of the ice, wave and temperature layers."

BUOY_POS = {  # NDBC (lon, lat) — QC reference only
    "45001": (-87.793, 48.061),
    "45002": (-86.411, 45.344),
    "45132": (-81.220, 42.460),
    "45012": (-77.383, 43.619),
    "45005": (-82.398, 41.677),
}


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ff(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def extract_uv_analysis(grib_path):
    """Return (u_ms, v_ms, lats, lons, nx, ny, dataDate, dataTime) for step 0."""
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    u = v = None
    lats = lons = date = time = None
    nx = ny = None
    with open(grib_path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                sn = codes_get(h, "shortName")
                if str(codes_get(h, "step")) == "0" and sn in ("u", "v"):
                    vals = codes_get_values(h).astype(float)
                    if lats is None:
                        lats = codes_get_array(h, "latitudes").astype(float)
                        lons = codes_get_array(h, "longitudes").astype(float)
                        lons = ((lons + 180) % 360) - 180
                        nx, ny = int(codes_get(h, "Nx")), int(codes_get(h, "Ny"))
                        date = str(codes_get(h, "dataDate"))
                        time = str(codes_get(h, "dataTime")).zfill(4)
                    if sn == "u":
                        u = vals
                    else:
                        v = vals
                    if u is not None and v is not None:
                        return u, v, lats, lons, nx, ny, date, time
            finally:
                codes_release(h)
    raise ValueError("UGRD/VGRD analysis messages (step=0) not found in GRIB2")


def draw_arrows(rgba, uu, vv, valid_src, rows, cols, bounds, step):
    """Draw tiny movement-direction arrows into the raster (in place).

    uu/vv are source-grid movement components (east/north +). Arrows are
    sampled every `step` source points and drawn at their canvas pixels.
    """
    H, W = rgba.shape[:2]
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    n = 0
    ny, nx = valid_src.shape
    for iy in range(0, ny, step):
        for ix in range(0, nx, step):
            if not valid_src[iy, ix]:
                continue
            u, v = float(uu[iy, ix]), float(vv[iy, ix])
            if not (math.isfinite(u) and math.isfinite(v)):
                continue
            sp = math.hypot(u, v)
            if sp < 0.5:
                continue  # calm: no arrow
            r, c = int(rows[iy, ix]), int(cols[iy, ix])
            if not (8 <= r < H - 8 and 8 <= c < W - 8):
                continue
            # movement direction on canvas: east+right, north+up(screen y down)
            dx, dy = u / sp, -v / sp
            L = 9.0
            x0, y0 = c - dx * L / 2, r - dy * L / 2
            x1, y1 = c + dx * L / 2, r + dy * L / 2
            ang = math.atan2(dy, dx)
            for ext, w, col in ((2, 3, (20, 20, 20, 230)),
                                (0, 1, (255, 255, 255, 235))):
                d.line([(x0, y0), (x1, y1)], fill=col, width=2 + ext)
                for s in (1, -1):
                    ha = ang + s * (math.pi - 0.5)
                    d.line([(x1, y1),
                            (x1 + math.cos(ha) * 5, y1 + math.sin(ha) * 5)],
                           fill=col, width=2 + ext)
            n += 1
    del d
    out = np.array(img)
    return out, n


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    now = datetime.now(timezone.utc)
    raw_path = os.path.join(RAW_DIR, "glwu_wind_current.grib2")
    got, used_url, datestr, cycle = None, None, None, None
    for url, dd, cc in candidate_urls(now):
        try:
            info = download(url, raw_path, timeout=300)
            if info["size_bytes"] < 100_000:
                print(f"[{PRODUCT}] candidate {dd} t{cc}z too small; trying older.")
                continue
            got, used_url, datestr, cycle = info, url, dd, cc
            break
        except Exception as e:
            print(f"[{PRODUCT}] candidate {dd} t{cc}z failed: {str(e)[:140]}")
    if got is None:
        print(f"[{PRODUCT}] DOWNLOAD FAILED for all candidates (keeping previous).")
        return 2

    prev = read_state(PRODUCT)
    # Waves and wind share GRIB2 files but track independent state, so one
    # product's skip can never suppress the other.
    if prev.get("model_cycle") == f"{datestr} t{cycle}z" \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")):
        print(f"[{PRODUCT}] model cycle unchanged ({datestr} t{cycle}z); skipping.")
        try:
            refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                                 CONFIG["title"], SKIP_NOTE,
                                 CONFIG["refresh_interval_seconds"])
            print(f"[{PRODUCT}] KML base URLs refreshed.")
        except Exception as e:
            print(f"[{PRODUCT}] WARNING: KML refresh failed: {e}")
        return 0

    try:
        return _build(got, used_url, datestr, cycle, raw_path)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(got, used_url, datestr, cycle, raw_path):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    bounds = load_bounds()

    u_ms, v_ms, lats, lons, gnx, gny, data_date, data_time = \
        extract_uv_analysis(raw_path)
    valid = (np.isfinite(u_ms) & np.isfinite(v_ms)
             & (np.abs(u_ms) < 75) & (np.abs(v_ms) < 75))
    n_valid = int(valid.sum())
    print(f"[{PRODUCT}] UV analysis {data_date} {data_time}Z: "
          f"valid={n_valid}/{u_ms.size}")
    if n_valid < 5_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few valid cells. Keeping previous.")
        return 2

    spd_ms = np.hypot(u_ms, v_ms)
    if float(spd_ms[valid].max()) > 75:
        print(f"[{PRODUCT}] VALIDATION FAILED: implausible max wind.")
        return 2
    spd_kt = spd_ms * MS_TO_KT
    forces = np.full(spd_kt.shape, np.nan)
    fvec = np.vectorize(force_from_kt, otypes=[float])
    forces[valid] = fvec(spd_kt[valid])
    data_time_utc = (f"{data_date[0:4]}-{data_date[4:6]}-{data_date[6:8]} "
                     f"{data_time[0:2]}:{data_time[2:4]} UTC")
    fmax = int(np.nanmax(forces))
    print(f"[{PRODUCT}] max {float(spd_kt[valid].max()):.1f} kt = Beaufort {fmax}")

    # bin force field + keep source UV on the canvas grid for arrows
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    lf, lo = np.asarray(lats).ravel(), np.asarray(lons).ravel()
    inside = (np.isfinite(lf) & np.isfinite(lo)
              & (lf >= lat_min) & (lf <= lat_max)
              & (lo >= lon_min) & (lo <= lon_max))
    cols = np.clip(((lo - lon_min) / (lon_max - lon_min) * W).astype(int), 0, W - 1)
    rows = np.clip(((lat_max - lf) / (lat_max - lat_min) * H).astype(int), 0, H - 1)

    # force field: mean per canvas pixel (forces are stepwise constant),
    # splat-filled like the wave layer (2.5 km source vs ~0.85 km pixels)
    from geospatial_utils import bin_to_canvas
    ffield, _cnt = bin_to_canvas(rows, cols, forces.ravel(), inside & valid,
                                 (H, W), splat_radius=2)

    # UV on source grid (for arrows): reshape to grid form
    ny, nx = gny, gnx
    if ny * nx != u_ms.size:
        raise ValueError(f"UV grid shape {nx}x{ny} != {u_ms.size} points")
    uu = np.where(valid, u_ms, np.nan).reshape(ny, nx)
    vv = np.where(valid, v_ms, np.nan).reshape(ny, nx)
    okg = valid.reshape(ny, nx)
    rgrid = rows.reshape(ny, nx)
    cgrid = cols.reshape(ny, nx)

    # colors: continuous LUT over force 0..12 (dark purple ONLY at force 12)
    lut = np.zeros((13, 3), dtype=np.uint8)
    for f in range(13):
        lut[f] = _rgb(FORCE_COLORS[f])
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    ook = np.isfinite(ffield)
    fi = np.clip(np.round(ffield[ook]).astype(int), 0, 12)
    rgba[ook, 0:3] = lut[fi]
    rgba[ook, 3] = bounds["overlay_alpha"]

    rgba, n_arrows = draw_arrows(rgba, uu, vv, okg, rgrid, cgrid, bounds,
                                 CONFIG["arrow_subsample"])
    print(f"[{PRODUCT}] arrows drawn: {n_arrows}")
    if n_arrows < 50:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few arrows. Keeping previous.")
        return 2
    rgba = apply_shoreline_mask(rgba)  # one shared GSHHG shoreline for all
    save_png(rgba, os.path.join(stage_prod, "current.png"))

    beaufort_rows = "".join(
        f"<b>F{f}</b> — {name} ({rng})<br/>"
        for f, name, rng in
        [(f, force_name(f), force_range_text(f)) for f, _, _, _ in BEAUFORT])
    scale_html = (f"WIND — BEAUFORT SCALE (colors follow force, dark purple = "
                  f"Force 12 only):<br/>{beaufort_rows}"
                  f"Maximum this run: <b>{float(spd_kt[valid].max()):.0f} kt</b> "
                  f"(Beaufort {fmax}). Arrows point where the air moves.")
    lw, lh = draw_category_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"],
        f"{CONFIG['freshness_label']}  |  Model time: {data_time_utc}",
        [(FORCE_COLORS[f], f"F{f} — {force_name(f)} ({force_range_text(f)})")
         for f in range(13)],
        f"Source: NCEP GLWU v2.1 U/V analysis {datestr} t{cycle}z  |  "
        f"Processed {utcnow_iso()}",
        note="Dark purple = Force 12 hurricane-force (≥64 kt) ONLY.")

    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], used_url, CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc=got.get("http_last_modified") or "n/a (NOMADS)",
        units="kt + Beaufort Force 0-12 (display); source m/s",
        source_resolution="~2.5 km NCEP GLWU Lambert grid (581x361)",
        color_min=0, color_max=12, color_units="Beaufort Force",
        missing_data_treatment=("GRIB2 missing value 9999 and off-water grid "
                                "points rendered fully transparent; never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{datestr} t{cycle}z"
    meta["beaufort_table"] = [
        {"force": f, "description": force_name(f), "range_kt": force_range_text(f),
         "color": FORCE_COLORS[f]} for f in range(13)]
    meta["methodology"] = (
        "Wind speed = sqrt(U^2+V^2) from GLWU UGRD/VGRD surface analysis "
        "(m/s, direction of motion), x1.94384 -> knots, mapped to WMO Beaufort "
        "Force 0-12 with unmodified thresholds. Arrows point along the (U,V) "
        "movement vector (no FROM->TOWARD reversal needed for GRIB components); "
        "calm (<0.5 m/s) gets no arrow. Overlay alpha cut by the shared GSHHG "
        "shoreline mask.")
    meta["stats"] = {
        "valid_cells": n_valid,
        "max_kt": round(float(spd_kt[valid].max()), 1),
        "max_beaufort": fmax,
        "arrows_drawn": n_arrows,
    }

    field = ffield  # for buoy QC sampling below
    # ---- buoy QC: WSPD (reference only) ----
    qc = fetch_buoy_obs(CONFIG["buoys"])
    for bid, pos in BUOY_POS.items():
        obs = qc.get(bid, {})
        ws = ff(obs.get("WSPD_ms"))
        if ws is None:
            print(f"[{PRODUCT}] buoy {bid}: no WSPD obs (skipped)")
            meta.setdefault("buoy_qc", {})[bid] = {**obs, "note": "no WSPD obs"}
            continue
        ws_kt = ws * MS_TO_KT
        col = int((pos[0] - lon_min) / (lon_max - lon_min) * W)
        row = int((lat_max - pos[1]) / (lat_max - lat_min) * H)
        cell = None
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                rr, cc = row + dr, col + dc
                if 0 <= rr < H and 0 <= cc < W and np.isfinite(field[rr, cc]):
                    cell = float(field[rr, cc])
                    break
            if cell is not None:
                break
        cell_kt = None
        if cell is not None:
            # map sampled force back to representative kt (range midpoint)
            _f, _n, kmin, kmax = BEAUFORT[int(round(cell))]
            cell_kt = (kmin + kmax) / 2 if kmax is not None else 70.0
        diff = None if cell_kt is None else round(abs(cell_kt - ws_kt), 1)
        flag = ("OK" if (diff is not None and diff <= CONFIG["buoy_qc_tolerance_kt"])
                else "CHECK" if diff is not None else "NO_GRID_CELL")
        print(f"[{PRODUCT}] buoy {bid}: obs {ws_kt:.1f}kt grid~{cell_kt} diff {diff} -> {flag}")
        meta.setdefault("buoy_qc", {})[bid] = {
            "obs_kt": round(ws_kt, 1), "grid_kt_approx": cell_kt,
            "absdiff_kt": diff, "verdict": flag, "obs_time": obs.get("time_utc")}
    write_metadata(stage_prod, meta)  # re-write incl. buoy QC

    token = meta["processing_time_utc"].replace(" ", "_").replace(":", "")
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        out_dirs=[os.path.join(stage, "kml", KML_FILE),
                  os.path.join(stage, "site", "kml", KML_FILE)])
    assert_no_vector_geometry(kml_text)

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())