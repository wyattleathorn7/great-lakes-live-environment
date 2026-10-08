"""Pipeline F — LIVE WIND (independent).

NCEP GLWU GRIB2 UGRD/VGRD surface analysis (same operational files as the
wave product, wind fields only) OVER WATER + NOAA/NCEP HRRR 3 km 10 m
UGRD/VGRD analysis as the land/background fill -> validate ->
speed = sqrt(U^2+V^2) -> Beaufort Force 0-12 -> blue->...->red->
dark-purple(F12) gradient with tiny movement-direction arrows rendered
INTO the raster -> transparent PNG (FULL BASIN RECTANGLE lon -93..-73.5 /
lat 40.5..49.5, the exact LIVE LEAF COLOR footprint; only missing data
is transparent, no shoreline cut) -> metadata -> KML.

GLWU has no valid values over land by design (off-water grid points file
as missing), so land would stay transparent on GLWU alone. HRRR fills
every GLWU gap (land + canvas edges beyond the GLWU mesh), giving the
wind layer the same geographic coverage as the leaf layer. Where both
are valid (open water) GLWU wins at its native ~2.5 km lake resolution.

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
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from build_wave_height import cycle_source_id, newest_available_cycle
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR,
                               base_metadata, bin_to_canvas, canvas_indices,
                               bleed_rgb_into_transparent,
                               download, draw_category_legend,
                               fetch_buoy_obs, load_bounds,
                               grib_stamp_to_det, now_det_str, promote_stage,
                               read_state, save_png, source_token, stage_dir,
                               utcnow_iso, write_metadata, write_state)

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

# Full-basin render generation: bump to force one redeploy of the expanded
# (GLWU+HRRR, no shoreline cut) raster even when the GLWU cycle is unchanged.
# Afterwards the source id tracks both model cycles.
RENDER_TAG = "fullbasin-g4"

# Dual-opacity finish: open-water pixels render at the original ~80% lake
# opacity; land pixels stay semi-transparent so the base-map terrain shows
# through (arrows paint at their own near-opaque alphas and stay legible).
WIND_ALPHA_WATER = 205
WIND_ALPHA_LAND = 140

HRRR_MSGS = [("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")]
CANVAS_ARROW_STEP_PX = 45  # canvas-space sampling for land+water arrows


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _vivid_for_land(h):
    """Saturation-boosted twin of a water hex for the semi-transparent land.

    Land pixels paint at alpha 140 over bright terrain, which washes the
    fill toward gray-green; pre-intensifying (same hue, higher saturation,
    slight value lift) makes the perceived land color match the water
    color at alpha 205. Single legend still applies (it shows the water
    colors) — no second key is created.
    """
    import colorsys
    r, g, b = (int(h.lstrip("#")[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hh, s, v = colorsys.rgb_to_hsv(r, g, b)
    r2, g2, b2 = colorsys.hsv_to_rgb(
        hh, min(1.0, s * 1.4), min(1.0, v * 1.08 + 0.03))
    return (round(r2 * 255), round(g2 * 255), round(b2 * 255))


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


def arrow_points(uu, vv, valid_src, rows, cols, step):
    """Sample (row, col, u, v) movement vectors on a regular sub-grid."""
    pts = []
    ny, nx = valid_src.shape
    for iy in range(0, ny, step):
        for ix in range(0, nx, step):
            if not valid_src[iy, ix]:
                continue
            u, v = float(uu[iy, ix]), float(vv[iy, ix])
            if not (math.isfinite(u) and math.isfinite(v)):
                continue
            if math.hypot(u, v) < 0.5:
                continue  # calm: no arrow
            pts.append((int(rows[iy, ix]), int(cols[iy, ix]), u, v))
    return pts


def paint_arrows(rgba, points):
    """Paint tiny movement-direction arrows (in place -> (rgba, count))."""
    H, W = rgba.shape[:2]
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    n = 0
    for r, c, u, v in points:
        if not (8 <= r < H - 8 and 8 <= c < W - 8):
            continue
        sp = math.hypot(u, v)
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
    return np.array(img), n


def draw_arrows(rgba, uu, vv, valid_src, rows, cols, bounds, step):
    """Legacy wrapper (kept for selftest): sample + paint. `bounds` unused."""
    return paint_arrows(rgba, arrow_points(uu, vv, valid_src, rows, cols, step))


def arrow_points_canvas(uu_canvas, vv_canvas, step_px):
    """Sample (row, col, u, v) movement vectors on a uniform canvas lattice.

    Full-basin coverage needs arrows over land too, so sampling happens in
    canvas space on the combined (GLWU-over-water + HRRR fill) UV field
    instead of on either source grid.
    """
    pts = []
    H, W = np.shape(uu_canvas)[:2]
    for r in range(0, H, step_px):
        for c in range(0, W, step_px):
            u, v = float(uu_canvas[r, c]), float(vv_canvas[r, c])
            if not (math.isfinite(u) and math.isfinite(v)):
                continue
            if math.hypot(u, v) < 0.5:
                continue  # calm: no arrow
            pts.append((r, c, u, v))
    return pts


def fetch_hrrr_uv():
    """Latest HRRR 10 m U/V analysis binned helpers.

    Returns (u_vals, v_vals, lats, lons, dd, cc, data_date, data_time)
    on the native HRRR grid (1-D raveled values + lat/lon). Raises on any
    fetch/decode failure (caller keeps the previous raster via exit 2).
    """
    import hrrr
    base, dd, cc = hrrr.latest_cycle()
    raw_hrrr = os.path.join(RAW_DIR, "hrrr_wind_current.grib2")
    hrrr.fetch_messages(base, raw_hrrr, HRRR_MSGS)
    got = hrrr.read_messages(raw_hrrr, {"UGRD": 1, "VGRD": 1})
    (u_vals, u_lats, u_lons, u_date, u_time) = got["UGRD"]
    (v_vals, _v_lats, _v_lons, _v_date, _v_time) = got["VGRD"]
    return (np.asarray(u_vals, dtype=float), np.asarray(v_vals, dtype=float),
            np.asarray(u_lats, dtype=float), np.asarray(u_lons, dtype=float),
            dd, cc, u_date, u_time)


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
    # Same newest-available-cycle detection as wave height (UGRD line),
    # so the wind raster tracks the actual posted cycle, never the
    # assumed schedule -- and unchanged cycles skip without download.
    pick = newest_available_cycle(now, "UGRD", label=PRODUCT)
    if pick is None:
        print(f"[{PRODUCT}] NO CYCLE AVAILABLE (keeping previous).")
        return 2
    url, datestr, cycle, stamp = pick
    # Combined source id: the raster now depends on BOTH cycles (GLWU over
    # water + HRRR land fill), so a roll of either must rebuild. The
    # RENDER_TAG forces one redeploy of the expanded full-basin raster.
    try:
        import hrrr as _hrrr_probe
        _hbase, _hdd, _hcc = _hrrr_probe.latest_cycle()
        hrrr_tag = f"hrrr-{_hdd}-t{_hcc}z"
    except Exception as e:
        print(f"[{PRODUCT}] HRRR PROBE FAILED (keeping previous): {str(e)[:140]}")
        return 2
    source_id = f"{cycle_source_id(stamp)}+{hrrr_tag}-{RENDER_TAG}"
    prev = read_state(PRODUCT)
    if prev.get("source_id") == source_id \
            and prev.get("render_version") == RENDER_VERSION \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE)):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                             CONFIG["title"], SKIP_NOTE,
                             CONFIG["refresh_interval_seconds"])
        return 0
    try:
        got = download(url, raw_path, timeout=300)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED for {datestr} t{cycle}z "
              f"(keeping previous): {str(e)[:140]}")
        return 2
    if got["size_bytes"] < 100_000:
        print(f"[{PRODUCT}] candidate {datestr} t{cycle}z too small; "
              f"keeping previous.")
        return 2

    # No skip: every run rebuilds (tiles must deploy); identical
    # bytes simply produce no commit. Failures still keep previous.

    try:
        return _build(got, url, datestr, cycle, raw_path)
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
    # HRRR land/background fill (full-basin coverage). Required: without it
    # land has no wind data (GLWU off-water points file as missing), so a
    # failed HRRR fetch keeps the previous raster instead of regressing.
    try:
        (hrrr_u, hrrr_v, hlats, hlons,
         hrrr_dd, hrrr_cc, hrrr_date, hrrr_time) = fetch_hrrr_uv()
    except Exception as e:
        print(f"[{PRODUCT}] HRRR FILL FAILED (keeping previous): {e}")
        return 2
    # Source-aware gate (both model stamps control regeneration, never the
    # workflow run time) + one-time full-basin render tag.
    source_id = (f"glwu-{data_date}-{data_time}Z"
                 f"+hrrr-{hrrr_dd}-t{hrrr_cc}z-{RENDER_TAG}")
    prev = read_state(PRODUCT)
    if prev.get("source_id") == source_id \
            and prev.get("render_version") == RENDER_VERSION \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live", KML_FILE)):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                             CONFIG["title"], SKIP_NOTE,
                             CONFIG["refresh_interval_seconds"])
        return 0
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
    data_time_utc = grib_stamp_to_det(data_date, data_time)
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
    ffield, _cnt = bin_to_canvas(rows, cols, forces.ravel(), inside & valid,
                                 (H, W), splat_radius=2)
    # GLWU UV on the canvas grid (for the combined arrow field below)
    gu_canvas, _ = bin_to_canvas(rows, cols, np.asarray(u_ms).ravel(),
                                 inside & valid, (H, W), splat_radius=2)
    gv_canvas, _ = bin_to_canvas(rows, cols, np.asarray(v_ms).ravel(),
                                 inside & valid, (H, W), splat_radius=2)
    n_glwu_canvas = int(np.isfinite(ffield).sum())
    print(f"[{PRODUCT}] GLWU canvas cells: {n_glwu_canvas}")

    # HRRR 10 m fill binned onto the same canvas (full CONUS coverage:
    # land + water + canvas edges beyond the GLWU mesh)
    hrrr_time_utc = grib_stamp_to_det(hrrr_date, hrrr_time)
    h_lons = ((np.asarray(hlons).ravel() + 180) % 360) - 180
    h_lats = np.asarray(hlats).ravel()
    h_u = np.asarray(hrrr_u).ravel()
    h_v = np.asarray(hrrr_v).ravel()
    h_ok_src = (np.isfinite(h_u) & np.isfinite(h_v)
                & (np.abs(h_u) < 75) & (np.abs(h_v) < 75))
    h_rows, h_cols, _hv = canvas_indices(h_lats, h_lons, bounds)
    h_inside = (np.isfinite(h_lats) & np.isfinite(h_lons)
                & (h_lats >= lat_min) & (h_lats <= lat_max)
                & (h_lons >= lon_min) & (h_lons <= lon_max))
    h_spd = np.hypot(np.where(h_ok_src, h_u, np.nan),
                     np.where(h_ok_src, h_v, np.nan))
    h_kt = h_spd * MS_TO_KT
    h_forces = np.full(h_kt.shape, np.nan)
    h_forces[h_ok_src] = fvec(h_kt[h_ok_src])
    h_ffield, _hcnt = bin_to_canvas(h_rows, h_cols, h_forces,
                                    h_inside & h_ok_src, (H, W),
                                    splat_radius=2)
    hu_canvas, _ = bin_to_canvas(h_rows, h_cols, h_u,
                                 h_inside & h_ok_src, (H, W), splat_radius=2)
    hv_canvas, _ = bin_to_canvas(h_rows, h_cols, h_v,
                                 h_inside & h_ok_src, (H, W), splat_radius=2)
    n_hrrr_canvas = int(np.isfinite(h_ffield).sum())
    print(f"[{PRODUCT}] HRRR {hrrr_dd} t{hrrr_cc}z canvas cells: {n_hrrr_canvas}")
    if n_hrrr_canvas < 50_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few HRRR fill cells. "
              f"Keeping previous.")
        return 2
    hrrr_max_kt = float(np.nanmax(h_kt[h_ok_src])) if int(h_ok_src.sum()) else 0.0

    # Composite (same Beaufort colors, wider footprint): GLWU wins wherever
    # it is valid (open water at lake resolution); HRRR fills everything
    # else (land + GLWU gaps + canvas edges). Same leaf-color footprint.
    glwu_ok = np.isfinite(ffield)
    field = np.where(glwu_ok, ffield, h_ffield)
    comb_u = np.where(glwu_ok, gu_canvas, hu_canvas)
    comb_v = np.where(glwu_ok, gv_canvas, hv_canvas)
    n_combined = int(np.isfinite(field).sum())
    n_fill = int((~glwu_ok & np.isfinite(field)).sum())
    print(f"[{PRODUCT}] combined canvas cells: {n_combined} "
          f"(GLWU {n_glwu_canvas} + HRRR fill {n_fill})")
    if n_combined < 50_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few combined cells. "
              f"Keeping previous.")
        return 2
    fmax_fill = int(np.nanmax(field))

    # colors: water LUT is the exact Beaufort table (dark purple ONLY at
    # force 12); land LUT is the saturation-boosted twin so the same force
    # reads the same through the extra land transparency. One legend.
    from geospatial_utils import load_watermask
    _wm = load_watermask()
    lut_water = np.zeros((13, 3), dtype=np.uint8)
    lut_land = np.zeros((13, 3), dtype=np.uint8)
    for f in range(13):
        lut_water[f] = _rgb(FORCE_COLORS[f])
        lut_land[f] = _vivid_for_land(FORCE_COLORS[f])
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    ook = np.isfinite(field)
    fi = np.clip(np.round(field[ook]).astype(int), 0, 12)
    _is_water = (_wm >= 0.5)[ook]
    rgba[ook, 0:3] = np.where(_is_water[:, None], lut_water[fi], lut_land[fi])
    rgba[ook, 3] = WIND_ALPHA_LAND

    rgba, n_arrows = paint_arrows(
        rgba, arrow_points_canvas(comb_u, comb_v, CANVAS_ARROW_STEP_PX))
    print(f"[{PRODUCT}] arrows drawn: {n_arrows}")
    if n_arrows < 50:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few arrows. Keeping previous.")
        return 2
    # Dual-opacity shoreline merge (the original water-only look over the
    # lakes, see-through over land): the shared NOAA shoreline mask drives
    # per-pixel alpha — 205 over open water, 140 over land, antialiased
    # blend along the shore. Arrows keep their own near-opaque alphas.
    _dual = np.round(WIND_ALPHA_LAND + (WIND_ALPHA_WATER - WIND_ALPHA_LAND)
                     * _wm).astype(np.uint8)
    _painted = rgba[:, :, 3] > 0
    rgba[_painted, 3] = _dual[_painted]
    # Full basin rectangle (atmospheric layer, like leaf footprint): only
    # missing source data is transparent. RGB bleed keeps
    # the anti-fringe contract for bilinear clients (Google Earth).
    rgba = bleed_rgb_into_transparent(rgba)
    save_png(rgba, os.path.join(stage_prod, "current.png"))


    beaufort_rows = "".join(
        f"<b>F{f}</b> — {name} ({rng})<br/>"
        for f, name, rng in
        [(f, force_name(f), force_range_text(f)) for f, _, _, _ in BEAUFORT])
    scale_html = (f"WIND — BEAUFORT SCALE (colors follow force, dark purple = "
                  f"Force 12 only):<br/>{beaufort_rows}"
                  f"Maximum this run: <b>{max(float(spd_kt[valid].max()), hrrr_max_kt):.0f} kt</b> "
                  f"(Beaufort {max(fmax, fmax_fill)}). Full basin coverage: "
                  f"GLWU over water, HRRR 10 m fill over land. "
                  f"Arrows point where the air moves.")
    lw, lh = draw_category_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"],
        f"{CONFIG['freshness_label']}  |  Water: {data_time_utc}  |  "
        f"Land fill: {hrrr_time_utc}",
        [(FORCE_COLORS[f], f"F{f} — {force_name(f)} ({force_range_text(f)})")
         for f in range(13)],
        f"Source: GLWU v2.1 U/V {datestr} t{cycle}z (water) + HRRR 10 m "
        f"{hrrr_dd} t{hrrr_cc}z (land fill)  |  "
        f"Processed {now_det_str()}",
        note="Dark purple = Force 12 hurricane-force (≥64 kt) ONLY. "
             "Lakes ~80% opacity; land ~55% with intensified colors so the "
             "same force reads the same (single key).")

    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], used_url, CONFIG["variable"],
        data_time_utc=f"GLWU {data_time_utc} (water); HRRR {hrrr_time_utc} "
                      f"(land fill)",
        source_last_modified_utc=got.get("http_last_modified") or "n/a (NOMADS)",
        units="kt + Beaufort Force 0-12 (display); source m/s",
        source_resolution=("~2.5 km NCEP GLWU Lambert grid (581x361) over "
                           "water + ~3 km NCEP HRRR CONUS grid as land fill"),
        color_min=0, color_max=12, color_units="Beaufort Force",
        missing_data_treatment=("full basin rectangle lon -93..-73.5 / lat "
                                "40.5..49.5 (same footprint as Live Leaf "
                                "Color): GLWU over water, HRRR 10 m wind "
                                "fills land + GLWU gaps + canvas edges; "
                                "dual opacity from the shared NOAA shoreline "
                                "mask (water alpha 205 ~80%, land alpha 140 "
                                "~55%); transparent only where both sources "
                                "lack data; never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{datestr} t{cycle}z + {hrrr_dd} t{hrrr_cc}z"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["beaufort_table"] = [
        {"force": f, "description": force_name(f), "range_kt": force_range_text(f),
         "color": FORCE_COLORS[f]} for f in range(13)]
    meta["methodology"] = (
        "Wind speed = sqrt(U^2+V^2) from GLWU UGRD/VGRD surface analysis "
        "over water (m/s, direction of motion) + HRRR UGRD/VGRD 10 m "
        "analysis as the land/background fill (same formula), x1.94384 -> "
        "knots, mapped to WMO Beaufort Force 0-12 with unmodified "
        "thresholds; GLWU wins wherever valid, HRRR fills land + GLWU "
        "gaps + canvas edges. Arrows point along the (U,V) movement "
        "vector (no FROM->TOWARD reversal needed for GRIB components); "
        "calm (<0.5 m/s) gets no arrow. Full basin rectangle "
        "(lon -93..-73.5, lat 40.5..49.5, the Live Leaf Color footprint) "
        "with dual opacity from the shared NOAA shoreline mask: open water "
        "~80% (alpha 205, the original lakes look) with the exact Beaufort "
        "table colors, land ~55% (alpha 140) with saturation-boosted twins "
        "of the same hues so each force reads the same through the extra "
        "transparency (single legend, no second key); edge RGB bled into "
        "transparent pixels (anti-fringe for bilinear clients).")
    meta["stats"] = {
        "valid_cells": n_combined,
        "glwu_canvas_cells": n_glwu_canvas,
        "hrrr_fill_cells": n_fill,
        "hrrr_cycle": f"{hrrr_dd} t{hrrr_cc}z",
        "max_kt": round(float(spd_kt[valid].max()), 1),
        "max_beaufort": fmax,
        "max_beaufort_combined": fmax_fill,
        "arrows_drawn": n_arrows,
    }

    field = field  # combined GLWU+HRRR force grid for buoy QC sampling below
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

    token = meta["source_version"]
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    outs = live_out_dirs(stage, KML_FILE)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        out_dirs=outs["live"],
        meta=meta)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "source_id": source_id,
                          "render_version": RENDER_VERSION,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())