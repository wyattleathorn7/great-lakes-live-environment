"""Pipeline C4 — LIVE SURFACE CURRENTS (independent).

NOAA/NOS Great Lakes Operational Forecast System (GLOFS) surface-current
NOWCAST analysis + NOAA/GLERL experimental GLCFS-FVCOM corridor fill:

  primary (operational, 6-hourly cycles, hourly nowcast hours):
    LSOFS  -> Lake Superior + upper St. Marys River
    LMHOFS -> Lakes Michigan + Huron + Mackinac Straits
    LEOFS  -> Lake Erie (+ upper Niagara River mouth)
    LOOFS  -> Lake Ontario (+ lower Niagara River)
  supplement (experimental GLERL GLCFS FVCOM nowcast, 12-hourly, hourly steps):
    HEC    -> St. Clair River + Lake St. Clair + Detroit River
    mih    -> lower St. Marys River reach only (restricted bbox)

Gradient = CURRENT SPEED (fixed absolute 0-100 cm/s, source m/s x100).
Arrows = CURRENT DIRECTION, rasterized INTO the PNG (zero KML Placemarks):
resampled block-median movement vectors on a regular canvas grid (dense
step inside connecting-channel boxes, coarse step on open lakes).

Direction convention (critical, verified from file metadata):
  u_eastward = CF eastward_sea_water_velocity (m/s)
  v_northward = CF northward_sea_water_velocity (m/s)
i.e. the direction the water moves TOWARD. FVCOM u/v are the same
eastward/northward motion components. Arrows are drawn along (U,V) with
NO FROM->TOWARD reversal. Sanity anchor: St. Clair / Detroit river
cells render southward (v<0), matching real lake-to-lake flow.

Precedence: operational GLOFS wins everywhere it has valid water; the
experimental FVCOM fields fill ONLY canvas cells with no operational
vector (connecting channels the operational regular grids mask out).
No blending, no interpolation across sources.

Documented gaps (no authoritative vectors): St. Marys rapids/locks
reach (~46.35-46.42), Niagara Falls/gorge reach (~42.905-43.23),
Welland Canal / Trent-Severn / other minor canals.

Exit codes: 0 = updated (or source-unchanged skip); 2 = source/validation
failure (previous valid raster left untouched); 1 = unexpected error.
"""

import json
import math
import os
import re
import sys
import traceback
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR,
                              apply_shoreline_mask, base_metadata,
                              bin_to_canvas, canvas_indices, download,
                              load_bounds, load_watermask, now_det_str,
                              promote_stage, read_state, save_png,
                              source_token, stage_dir, write_metadata,
                              write_state)
from gradient_scale import draw_scale_legend, render_rgba

PRODUCT = "surface_currents"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Surface_Currents.kml"
OVERLAY_NAME = "\U0001F30A LIVE SURFACE CURRENTS"
SKIP_NOTE = "Turn on/off independently of all other layers."
UA = {"User-Agent": "great-lakes-live-environment/1.0"}

COOPS = "https://opendap.co-ops.nos.noaa.gov"
GLERL = "https://apps.glerl.noaa.gov"

# Operational GLOFS models: (catalog dir, file prefix, role)
GLOFS_MODELS = [
    ("LSOFS", "lsofs", "Lake Superior + upper St. Marys River"),
    ("LMHOFS", "lmhofs", "Lakes Michigan-Huron + Mackinac Straits"),
    ("LEOFS", "leofs", "Lake Erie + upper Niagara River mouth"),
    ("LOOFS", "loofs", "Lake Ontario + lower Niagara River"),
]
# Experimental FVCOM fill: (lake dir, role, restrict bbox or None)
# mih is restricted to the lower St. Marys reach so experimental vectors
# never override operational open-lake water.
MIH_BOX = (-84.6, -84.05, 46.10, 46.45)  # lon0,lon1,lat0,lat1
FVCOM_FILL = [
    ("hec", "St. Clair River + Lake St. Clair + Detroit River", None),
    ("michgan-huron", "lower St. Marys River reach", MIH_BOX),
]

FILL_VALUE = -99999.0
MAX_PLAUSIBLE_MS = 10.0  # fastest observed ~3.9 m/s; 10 rejects corruption

# Fixed absolute speed scale (cm/s), derived from the observed 2026-10-04
# distribution: open-lake median ~6-18, p95 ~12-39, p99 ~20-53, channel
# jets 100-390. Lakes own the color resolution; >=100 clamps dark-purple.
CUR_STOPS = [
    (0.0, (13, 42, 120)),     # dark blue: stagnant
    (3.0, (20, 110, 200)),    # blue
    (6.0, (20, 170, 220)),    # blue-cyan
    (10.0, (45, 190, 150)),   # teal-green
    (15.0, (150, 200, 60)),   # yellow-green
    (20.0, (245, 215, 50)),   # yellow
    (30.0, (240, 140, 25)),   # orange
    (50.0, (205, 35, 35)),    # red
    (75.0, (150, 25, 110)),   # red-violet
    (100.0, (59, 10, 90)),    # dark purple: channel jets
]
CUR_LABELS = [
    (0.0, "LOWEST 0"),
    (5.0, "5"),
    (10.0, "10"),
    (20.0, "20"),
    (30.0, "30"),
    (50.0, "50"),
    (100.0, "HIGHEST+ 100"),
]
# legend tick subset: "LOWEST 0" is wide, so the 5 tick would collide
CUR_LABELS_NO5 = [t for t in CUR_LABELS if t[0] != 5.0]
CUR_MAX = 100.0

# Dense arrow sampling inside connecting-channel boxes (rivers only, not
# the wider lakes — Lake St. Clair and open lakes use the coarse step)
# (lon0, lon1, lat0, lat1, name)
CHANNEL_BOXES = [
    (-84.70, -84.00, 46.10, 46.65, "St. Marys River"),
    (-82.62, -82.38, 42.70, 43.02, "St. Clair River"),
    (-83.25, -82.95, 42.00, 42.42, "Detroit River"),
    (-79.15, -78.90, 42.88, 43.35, "Niagara River"),
]


def _http_text(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def newest_glofs_file(model_dir, prefix, now):
    """Newest (date, cycle, nhour) GLOFS regulargrid nowcast file.

    Cheap catalog-XML probes only (KBs). Returns
    (dods_url, datestr, cycle, nhour) or None.
    """
    best = None
    for day in (now, now - timedelta(days=1)):
        ds = day.strftime("%Y%m%d")
        url = (f"{COOPS}/thredds/catalog/NOAA/{model_dir}/MODELS/"
               f"{day.strftime('%Y/%m/%d')}/catalog.xml")
        try:
            xml = _http_text(url)
        except Exception as e:
            print(f"[{PRODUCT}] catalog probe {model_dir} {ds}: {str(e)[:90]}")
            continue
        for m in re.finditer(
                re.escape(prefix) + r"\.(t\d\dz)\." + ds +
                r"\.regulargrid\.(n\d{3})\.nc", xml):
            key = (ds, m.group(1), m.group(2))
            if best is None or key > best[0]:
                durl = (f"{COOPS}/thredds/dodsC/NOAA/{model_dir}/MODELS/"
                        f"{day.strftime('%Y/%m/%d')}/{prefix}.{m.group(1)}."
                        f"{ds}.regulargrid.{m.group(2)}.nc")
                best = (key, durl)
    if best is None:
        return None
    (ds, cycle, nh), durl = best
    return durl, ds, cycle, nh


def newest_fvcom_file(lake_dir, now):
    """Newest GLERL GLCFS-FVCOM nowcast file (MMDDHH_0001.nc)."""
    url = f"{GLERL}/thredds/catalog/glcfs/{lake_dir}/nowcast/catalog.xml"
    try:
        xml = _http_text(url)
    except Exception as e:
        print(f"[{PRODUCT}] catalog probe glcfs/{lake_dir}: {str(e)[:90]}")
        return None
    names = sorted(set(re.findall(r"(\d{6})_0001\.nc", xml)))
    if not names:
        return None
    mmddhh = names[-1]
    return (f"{GLERL}/thredds/dodsC/glcfs/{lake_dir}/nowcast/"
            f"{mmddhh}_0001.nc", mmddhh)


def _dap(ds, var, *slices):
    """pydap hyperslab -> numpy (np.asarray(.data) avoids a numpy2/pydap
    __array__ incompatibility)."""
    return np.squeeze(np.asarray(ds[var][slices].data))


def fetch_glofs_surface(dods_url):
    """Surface-layer (u, v, lat, lon, mask, valid_time_iso) from a GLOFS
    regulargrid nowcast file. Units m/s, TOWARD convention."""
    from pydap.client import open_url
    ds = open_url(dods_url)
    depth0 = float(_dap(ds, "Depth", 0))
    if not np.isfinite(depth0) or depth0 > 1.0:
        raise ValueError(f"unexpected surface depth {depth0}")
    u = _dap(ds, "u_eastward", 0, 0, slice(None), slice(None)).astype(float)
    v = _dap(ds, "v_northward", 0, 0, slice(None), slice(None)).astype(float)
    lat = _dap(ds, "Latitude", slice(None), slice(None)).astype(float)
    lon = _dap(ds, "Longitude", slice(None), slice(None)).astype(float)
    lon = ((lon + 180.0) % 360.0) - 180.0
    mask = _dap(ds, "mask", slice(None), slice(None)).astype(float)
    raw_t = np.asarray(ds["Times"][:].data).ravel()[0]
    t = raw_t.decode() if isinstance(raw_t, bytes) else str(raw_t)
    ok = (np.isfinite(u) & np.isfinite(v) & (u > FILL_VALUE / 2)
          & (v > FILL_VALUE / 2) & (np.abs(u) < MAX_PLAUSIBLE_MS)
          & (np.abs(v) < MAX_PLAUSIBLE_MS) & (mask == 1.0))
    return {"u": u, "v": v, "lat": lat, "lon": lon, "ok": ok,
            "valid_iso": t, "depth_m": depth0}


def fetch_fvcom_surface(dods_url, restrict_box=None):
    """Surface-layer (siglay 0, last time step) from a GLCFS FVCOM nowcast.
    u/v are eastward/northward motion components (m/s, TOWARD)."""
    from pydap.client import open_url
    ds = open_url(dods_url)
    nt = int(ds["u"].shape[0])
    ti = nt - 1
    u = _dap(ds, "u", ti, 0, slice(None)).astype(float)
    v = _dap(ds, "v", ti, 0, slice(None)).astype(float)
    lon = _dap(ds, "lonc", slice(None)).astype(float)
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat = _dap(ds, "latc", slice(None)).astype(float)
    wet = _dap(ds, "wet_cells", ti, slice(None)).astype(float)
    raw = np.asarray(ds["Times"][:].data).ravel()
    last = raw[-1]
    t = last.decode() if isinstance(last, bytes) else str(last)
    ok = (np.isfinite(u) & np.isfinite(v) & (wet == 1.0)
          & (np.abs(u) < MAX_PLAUSIBLE_MS)
          & (np.abs(v) < MAX_PLAUSIBLE_MS))
    if restrict_box is not None:
        x0, x1, y0, y1 = restrict_box
        ok = ok & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
    return {"u": u, "v": v, "lat": lat, "lon": lon, "ok": ok,
            "valid_iso": t}


def valid_to_det(iso):
    """FVCOM/GLOFS 'YYYY-MM-DDTHH:MM:SS.ffffff' (UTC) -> Detroit display."""
    try:
        d = datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc)
        from geospatial_utils import fmt_det
        return fmt_det(d)
    except (TypeError, ValueError):
        return iso


def speed_dir(u_ms, v_ms):
    """speed (m/s) + toward-heading (deg clockwise from N) from components."""
    sp = math.hypot(float(u_ms), float(v_ms))
    hd = (math.degrees(math.atan2(float(u_ms), float(v_ms))) + 360.0) % 360.0
    return sp, hd


def compass(deg):
    pts = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    return pts[int(((float(deg) % 360) + 22.5) // 45)]


def paint_arrows(rgba, uu, vv, valid_frac, step, L=10.0):
    """Uniform-length toward-motion arrows from canvas-grid (U,V) in m/s.

    Block-median resampling over step x step windows: the displayed vector
    is the median of the underlying field (no invented directions).
    Skips stagnant (<1 cm/s) and poorly covered windows.
    Returns (rgba, n).
    """
    H, W = uu.shape
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    n = 0
    for r0 in range(0, H, step):
        for c0 in range(0, W, step):
            blk_u = uu[r0:r0 + step, c0:c0 + step]
            blk_v = vv[r0:r0 + step, c0:c0 + step]
            blk_f = valid_frac[r0:r0 + step, c0:c0 + step]
            m = np.isfinite(blk_u) & np.isfinite(blk_v) & (blk_f > 0)
            if m.sum() < max(4, int(0.2 * blk_u.size)):
                continue
            mu, mv = float(np.median(blk_u[m])), float(np.median(blk_v[m]))
            if math.hypot(mu, mv) < 0.01:  # <1 cm/s: stagnant, no arrow
                continue
            r = r0 + step // 2
            c = c0 + step // 2
            if not (10 <= r < H - 10 and 10 <= c < W - 10):
                continue
            if rgba[r, c, 3] == 0:
                continue
            sp = math.hypot(mu, mv)
            dx, dy = mu / sp, -mv / sp  # east+/north-up on canvas
            x0, y0 = c - dx * L / 2, r - dy * L / 2
            x1, y1 = c + dx * L / 2, r + dy * L / 2
            ang = math.atan2(dy, dx)
            head = L / 2.0
            for ext, w, col in ((2, 3, (20, 20, 20, 235)),
                                (0, 1, (255, 255, 255, 240))):
                d.line([(x0, y0), (x1, y1)], fill=col, width=2 + ext)
                for s in (1, -1):
                    ha = ang + s * (math.pi - 0.5)
                    d.line([(x1, y1),
                            (x1 + math.cos(ha) * head,
                             y1 + math.sin(ha) * head)],
                           fill=col, width=2 + ext)
            n += 1
    del d
    return np.array(img), n


def paint_sample_arrow(path):
    """Paint the direction-key row onto a finished scale legend (in place).

    Draws the note row itself (glyph + text) so nothing collides with the
    source line above it.
    """
    from geospatial_utils import _legend_font
    img = Image.open(path).convert("RGBA")
    d = ImageDraw.Draw(img)
    f_small = _legend_font(13)
    W, H = img.size
    y = H - 24
    # sample glyph, same style as the raster arrows, smaller
    cx, cy = 30, y + 2
    L = 22.0
    ang = math.radians(-35.0)
    dx, dy = math.cos(ang), math.sin(ang)
    x0, y0 = cx - dx * L / 2, cy - dy * L / 2
    x1, y1 = cx + dx * L / 2, cy + dy * L / 2
    ha = math.atan2(dy, dx)
    for ext, w, col in ((2, 3, (20, 20, 20, 235)),
                        (0, 1, (255, 255, 255, 240))):
        d.line([(x0, y0), (x1, y1)], fill=col, width=2 + ext)
        for s in (1, -1):
            a2 = ha + s * (math.pi - 0.5)
            d.line([(x1, y1), (x1 + math.cos(a2) * 6,
                               y1 + math.sin(a2) * 6)],
                   fill=col, width=2 + ext)
    d.text((52, y - 6),
           "COLOR = speed. White arrows = flow direction.",
           font=f_small, fill=(10, 10, 10))
    img.save(path)


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
    # ---- newest-source discovery (cheap catalog probes, no bulk fetch) ----
    glofs_pick = {}
    for model_dir, prefix, _role in GLOFS_MODELS:
        glofs_pick[prefix] = newest_glofs_file(model_dir, prefix, now)
    fvcom_pick = {}
    for lake_dir, _role, _box in FVCOM_FILL:
        fvcom_pick[lake_dir] = newest_fvcom_file(lake_dir, now)
    missing = [p for p, v in glofs_pick.items() if v is None]
    if missing:
        print(f"[{PRODUCT}] NO OPERATIONAL FIELDS AVAILABLE for {missing} "
              f"(keeping previous).")
        return 2
    parts = []
    for _md, prefix, _r in GLOFS_MODELS:
        _u, ds, cyc, nh = glofs_pick[prefix]
        parts.append(f"{prefix}-{ds}-{cyc}-{nh}")
    for lake_dir, _r, _b in FVCOM_FILL:
        pk = fvcom_pick[lake_dir]
        parts.append(f"{lake_dir}-{pk[1]}" if pk else f"{lake_dir}-unavailable")
    source_id = "+".join(parts)
    prev = read_state(PRODUCT)
    if (prev.get("source_id") == source_id
            and prev.get("render_version") == RENDER_VERSION
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png"))
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE))):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        try:
            refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                                 CONFIG["title"], SKIP_NOTE,
                                 CONFIG["refresh_interval_seconds"])
        except Exception as e:
            print(f"[{PRODUCT}] WARNING: KML refresh failed: {e}")
        return 0
    try:
        return _build(glofs_pick, fvcom_pick, source_id, now)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(glofs_pick, fvcom_pick, source_id, now):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    bounds = load_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]

    # ---- fetch operational fields (all four required) ----
    op_fields = {}
    for model_dir, prefix, role in GLOFS_MODELS:
        durl, ds, cyc, nh = glofs_pick[prefix]
        last_err = None
        for _try in (1, 2):
            try:
                f = fetch_glofs_surface(durl)
                break
            except Exception as e:
                last_err = e
                print(f"[{PRODUCT}] {prefix} fetch try{_try}: {str(e)[:120]}")
        else:
            raise ValueError(f"{prefix} operational fetch failed: {last_err}")
        n_valid = int(f["ok"].sum())
        print(f"[{PRODUCT}] {prefix} {ds} {cyc} {nh} valid {f['valid_iso']}: "
              f"{n_valid} water vectors")
        if n_valid < 3_000:
            raise ValueError(f"{prefix}: too few valid vectors ({n_valid})")
        op_fields[prefix] = {**f, "datestr": ds, "cycle": cyc, "nhour": nh,
                             "role": role}

    # ---- fetch experimental corridor fill (best-effort, flagged) ----
    fv_fields = {}
    for lake_dir, role, box in FVCOM_FILL:
        pk = fvcom_pick[lake_dir]
        if pk is None:
            print(f"[{PRODUCT}] {lake_dir} experimental unavailable "
                  f"(continuing operational-only there).")
            continue
        durl, mmddhh = pk
        try:
            f = fetch_fvcom_surface(durl, box)
        except Exception as e:
            print(f"[{PRODUCT}] {lake_dir} experimental fetch failed "
                  f"(continuing): {str(e)[:120]}")
            continue
        n_valid = int(f["ok"].sum())
        print(f"[{PRODUCT}] {lake_dir} {mmddhh} valid {f['valid_iso']}: "
              f"{n_valid} fill vectors")
        if n_valid < 100:
            print(f"[{PRODUCT}] {lake_dir}: too few fill vectors; skipping.")
            continue
        fv_fields[lake_dir] = {**f, "mmddhh": mmddhh, "role": role}

    # ---- bin to canvas: operational first, experimental fills holes ----
    spd_c = np.full((H, W), np.nan)   # cm/s
    uu_c = np.full((H, W), np.nan)    # m/s east
    vv_c = np.full((H, W), np.nan)    # m/s north
    for prefix, f in op_fields.items():
        rows, cols, valid = canvas_indices(f["lat"], f["lon"], bounds)
        src_ok = valid & f["ok"].ravel()
        spd = np.hypot(f["u"], f["v"]).ravel() * 100.0
        um = f["u"].ravel()
        vm = f["v"].ravel()
        for grid, vals in ((spd_c, spd), (uu_c, um), (vv_c, vm)):
            g, _cnt = bin_to_canvas(rows, cols, vals, src_ok, (H, W),
                                    splat_radius=2)
            fresh = np.isnan(grid) & np.isfinite(g)
            grid[fresh] = g[fresh]
    n_op = int(np.isfinite(spd_c).sum())
    op_valid = np.isfinite(spd_c).copy()  # operational-only mask for QC
    for lake_dir, f in fv_fields.items():
        rows, cols, valid = canvas_indices(f["lat"], f["lon"], bounds)
        src_ok = valid & f["ok"].ravel()
        spd = np.hypot(f["u"], f["v"]).ravel() * 100.0
        for grid, vals in ((spd_c, spd), (uu_c, f["u"].ravel()),
                           (vv_c, f["v"].ravel())):
            g, _cnt = bin_to_canvas(rows, cols, vals, src_ok, (H, W),
                                    splat_radius=1)
            hole = np.isnan(grid) & np.isfinite(g)
            grid[hole] = g[hole]
    n_all = int(np.isfinite(spd_c).sum())
    n_fill = n_all - n_op
    print(f"[{PRODUCT}] canvas: operational {n_op} + experimental fill "
          f"{n_fill} = {n_all} water vectors")
    if n_all < 5_000:
        raise ValueError(f"too few canvas vectors ({n_all})")
    if float(np.nanmax(spd_c)) > CUR_MAX * 5:
        raise ValueError(f"implausible max speed {float(np.nanmax(spd_c)):.0f}")

    okv = np.isfinite(spd_c)
    cur_max = float(spd_c[okv].max())
    cur_med = float(np.median(spd_c[okv]))
    # mean toward-heading of the displayed field
    _mu = float(np.nanmedian(uu_c[okv]))
    _mv = float(np.nanmedian(vv_c[okv]))
    _sp, mean_hd = speed_dir(_mu, _mv)

    # ---- overlap QC: experimental vs OPERATIONAL-ONLY cells ----
    # (compares the fill source against independent operational vectors;
    # fill cells themselves would match by construction)
    overlap_note = "no operational/experimental overlap sampled"
    for lake_dir, f in fv_fields.items():
        rows, cols, valid = canvas_indices(f["lat"], f["lon"], bounds)
        g, _c = bin_to_canvas(rows, cols,
                              (np.hypot(f["u"], f["v"]).ravel() * 100.0),
                              valid & f["ok"].ravel(), (H, W), splat_radius=1)
        both = np.isfinite(g) & op_valid
        if int(both.sum()) > 200:
            mad = float(np.median(np.abs(g[both] - spd_c[both])))
            overlap_note = (f"{lake_dir}: {int(both.sum())} overlap cells, "
                            f"median |dSpeed| {mad:.1f} cm/s")
            print(f"[{PRODUCT}] overlap QC {overlap_note}")

    # ---- render speed gradient ----
    rgba = render_rgba(spd_c, CUR_STOPS, bounds["overlay_alpha"])
    rgba = apply_shoreline_mask(rgba)  # lake water only
    wm = load_watermask()
    water = wm > 0.5
    # arrows from the combined vector field on water only
    uu_w = np.where(water, uu_c, np.nan)
    vv_w = np.where(water, vv_c, np.nan)
    val_w = np.where(water & np.isfinite(uu_c), 1.0, 0.0)
    n_arrows = 0
    lon_ax = (bounds["lon_min"] + (np.arange(W) + 0.5) / W
              * (bounds["lon_max"] - bounds["lon_min"]))
    lat_ax = (bounds["lat_max"] - (np.arange(H) + 0.5) / H
              * (bounds["lat_max"] - bounds["lat_min"]))
    ch_mask = np.zeros((H, W), bool)
    for x0, x1, y0, y1, _nm in CHANNEL_BOXES:
        ch_mask |= ((lon_ax[None, :] >= x0) & (lon_ax[None, :] <= x1)
                    & (lat_ax[:, None] >= y0) & (lat_ax[:, None] <= y1))
    # coarse pass everywhere EXCEPT river boxes (avoids double-painted
    # pile-ups where the dense pass lands on top of lake arrows)
    val_lake = np.where(~ch_mask, val_w, 0.0)
    tmp, n0 = paint_arrows(rgba, uu_w, vv_w, val_lake,
                           CONFIG["arrow_step_px"])
    n_arrows += n0
    rgba = tmp
    # dense pass inside river boxes only
    uu_ch = np.where(ch_mask, uu_w, np.nan)
    vv_ch = np.where(ch_mask, vv_w, np.nan)
    val_ch = np.where(ch_mask, val_w, 0.0)
    rgba, n1 = paint_arrows(rgba, uu_ch, vv_ch, val_ch,
                            CONFIG["arrow_channel_step_px"], L=6.0)
    n_arrows += n1
    # arrows near shore can spill 1-2 px onto land: clip alpha back to
    # the water mask (no second bleed — colors already bled once).
    rgba[:, :, 3] = np.round(
        rgba[:, :, 3].astype(np.float32) * wm).astype(np.uint8)
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    print(f"[{PRODUCT}] opaque={n_opaque} arrows={n_arrows} "
          f"(channels +{n1}) speed max={cur_max:.1f} med={cur_med:.1f} cm/s")
    if n_opaque < 5_000:
        raise ValueError("empty raster")
    if n_arrows < CONFIG["min_arrows"]:
        raise ValueError(f"too few arrows ({n_arrows})")

    # ---- legend + metadata + KML ----
    valid_isos = [f["valid_iso"] for f in op_fields.values()]
    data_iso = max(valid_isos)
    data_time_utc = valid_to_det(data_iso)
    unit = CONFIG["display_units"]
    subtitle = (f"Speed (cm/s) + flow arrows  |  {data_time_utc}  |  "
                f"max {cur_max:.0f}, median {cur_med:.1f} cm/s")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        "cm/s", CUR_STOPS, CUR_LABELS_NO5,
        f"Source: NOAA GLOFS nowcast + GLERL corridor fill  |  "
        f"Processed {now_det_str()}",
        note=None)
    paint_sample_arrow(os.path.join(stage_prod, "legend.png"))
    scale_html = (
        "Surface-current speed (centimeters per second, one continuous "
        "gradient): <b>LOWEST 0</b> dark-blue stagnant &rarr; <b>5</b> blue "
        "&rarr; <b>10</b> teal &rarr; <b>15-20</b> green-yellow &rarr; "
        "<b>30</b> orange &rarr; <b>50</b> red &rarr; <b>75</b> red-violet "
        "&rarr; <b>HIGHEST+ 100 cm/s</b> dark-purple channel jets. Same "
        "speed always shows the same color; above 100 stays dark-purple. "
        "White arrows point where the surface water is flowing "
        f"(filed U/V motion vector, no reversal). Now: max <b>{cur_max:.0f}"
        f"</b>, median <b>{cur_med:.1f} cm/s</b>, median flow "
        f"<b>{mean_hd:.0f}&deg; ({compass(mean_hd)})</b>. NO DATA stays "
        "transparent — never zero.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"],
        "https://opendap.co-ops.nos.noaa.gov/thredds/catalog.html",
        CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (THREDDS catalogs + OPeNDAP)",
        units="cm/s display (source m/s x100); direction deg TOWARD",
        source_resolution=("operational GLOFS regular grids ~0.5-1.1 km; "
                          "experimental FVCOM 30 m-2 km unstructured"),
        color_min=0.0, color_max=CUR_MAX, color_units="cm/s",
        missing_data_treatment=("fill value -99999, land mask, and "
                                "out-of-domain cells rendered fully "
                                "transparent; experimental fill only where "
                                "operational has no vector; never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["direction_convention"] = (
        "u_eastward = CF eastward_sea_water_velocity, v_northward = CF "
        "northward_sea_water_velocity (m/s, direction water moves TOWARD); "
        "FVCOM u/v are the same eastward/northward motion components. "
        "Arrows follow (U,V) directly with no FROM->TOWARD reversal; "
        "toward-heading = atan2(U,V) clockwise from north.")
    meta["field_type"] = ("model-generated nowcast guidance (FVCOM-based "
                         "GLOFS/GLCFS), NOT direct observations")
    meta["check_interval_seconds"] = 3600
    meta["source_cycles"] = {
        prefix: {"model": model_dir, "role": role,
                 "date": op_fields[prefix]["datestr"],
                 "cycle": op_fields[prefix]["cycle"],
                 "nowcast_hour": op_fields[prefix]["nhour"],
                 "valid_iso_utc": op_fields[prefix]["valid_iso"],
                 "valid_detroit": valid_to_det(op_fields[prefix]["valid_iso"]),
                 "vectors": int(op_fields[prefix]["ok"].sum())}
        for model_dir, prefix, role in GLOFS_MODELS
    }
    meta["corridor_fill"] = {
        lake_dir: {"role": role, "file": fv_fields[lake_dir].get("mmddhh"),
                   "valid_iso_utc": fv_fields[lake_dir]["valid_iso"],
                   "valid_detroit": valid_to_det(
                       fv_fields[lake_dir]["valid_iso"]),
                   "vectors": int(fv_fields[lake_dir]["ok"].sum())}
        for lake_dir, role, _b in FVCOM_FILL if lake_dir in fv_fields
    }
    meta["coverage_gaps"] = [
        "St. Marys rapids/locks reach (~46.35-46.42): no model vectors",
        "Niagara Falls/gorge reach (~42.905-43.23): no model vectors",
        "Welland Canal / Trent-Severn / other minor canals: no coverage",
    ]
    meta["overlap_qc"] = overlap_note
    meta["stats"] = {"canvas_vectors": n_all,
                     "operational_vectors": n_op,
                     "experimental_fill_vectors": n_fill,
                     "max_speed_cms": round(cur_max, 1),
                     "median_speed_cms": round(cur_med, 2),
                     "mean_toward_heading_deg": round(mean_hd, 1),
                     "mean_compass": compass(mean_hd),
                     "arrows_drawn": n_arrows}
    token = meta["source_version"]
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{CONFIG['what']}</p>"
        f"<p><img src=\"{legend_src(token)}\" width=\"600\" alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Units:</b> {unit}<br/>"
        f"<b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> GLOFS 6-hourly cycles + GLCFS 12-hourly, "
        f"checked hourly<br/>"
        f"<b>Data time:</b> {data_time_utc}<br/>"
        f"<b>Source version:</b> {source_id}<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Provenance:</b> <a href=\"{CONFIG['source_url']}\">CO-OPS "
        f"THREDDS</a> + "
        f"<a href=\"https://apps.glerl.noaa.gov/thredds/catalog/glcfs/"
        f"glcfs-catalog.html\">GLERL GLCFS THREDDS</a></p>")
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

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"source_id": source_id,
                          "render_version": RENDER_VERSION,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def legend_src(token=None):
    from build_kml import pages_base
    base = pages_base()
    v = f"?v={token}" if token else ""
    return f"{base}/{PRODUCT}/legend.png{v}"


if __name__ == "__main__":
    sys.exit(main())
