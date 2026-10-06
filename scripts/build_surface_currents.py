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
Flow marks = CURRENT DIRECTION, rasterized INTO the PNG (zero KML
Placemarks): NO preset lattice — thousands of 1 px streamlets trace RK2
streamlines integrated through the filed (U,V) field (brightness =
speed), with a tiny downstream chevron on every third streamlet. Every
mark position, path, and orientation is field-derived.

Narrow rivers vs the shared open-lake shoreline mask: the mask reads
land over sub-pixel rivers, so product-local channel water (source
vector present + inside RIVER_BOXES) restores river alpha. Shared mask
asset untouched; validators permit opaque-outside-mask only there.

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
                              base_metadata, bin_to_canvas,
                              bleed_rgb_into_transparent, canvas_indices,
                              download,
                              load_bounds, load_watermask, now_det_str,
                              promote_stage, read_state, save_png,
                              source_token, stage_dir, write_metadata,
                              write_state)
from gradient_scale import draw_scale_legend, render_rgba

PRODUCT = "surface_currents"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Surface_Currents.kml"
OVERLAY_NAME = "\U0001F504 LIVE SURFACE CURRENTS"
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
    (10.0, "10"),
    (20.0, "20"),
    (30.0, "30"),
    (40.0, "40"),
    (50.0, "50"),
    (60.0, "60"),
    (70.0, "70"),
    (80.0, "80"),
    (90.0, "90"),
    (100.0, "100+"),
]
CUR_MAX = 100.0

MPH_PER_CMS = 0.0223694  # key-image only: 1 cm/s = 0.0223694 mph.
# The SOURCE and all data/metadata stay cm/s; this converts display
# tick labels for the legend's MPH row alone.


def cms_to_mph(cms):
    """Key-image display conversion (legend MPH row only)."""
    return float(cms) * MPH_PER_CMS


def paint_mph_row(path, stops, labels):
    """MPH equivalents directly under each cm/s tick, key image only.

    Mirrors gradient_scale.draw_scale_legend geometry + de-collision, so
    MPH values sit under exactly the kept cm/s labels. The lone "cm/s"
    unit caption is removed (subtitle documents both units instead).
    """
    from geospatial_utils import _legend_font
    img = Image.open(path).convert("RGBA")
    d = ImageDraw.Draw(img)
    f_body, f_small = _legend_font(15), _legend_font(13)
    W, H = img.size
    bx, by, bw, bh = 14, 66, W - 28, 32
    vmin, vmax = stops[0][0], stops[-1][0]
    span = vmax - vmin if vmax > vmin else 1.0
    # erase the "cm/s" unit caption (units live in the subtitle now)
    ux, uy = bx + bw - 70, by + bh + 24
    bb = d.textbbox((ux, uy), "cm/s", font=f_body)
    d.rectangle([bb[0] - 2, bb[1] - 2, bb[2] + 2, bb[3] + 2],
                fill=(255, 255, 255, 235))
    # same de-collision as draw_scale_legend: endpoints kept, middles
    # only when clear of kept neighbors
    placed = []
    for val, text in labels:
        frac = min(max((val - vmin) / span, 0.0), 1.0)
        x = bx + int(frac * (bw - 1))
        tw = d.textlength(text, font=f_small)
        placed.append((val, x, tw, text))

    def _clear(c, boxes):
        return all(c[1] - c[2] / 2 > b[1] + b[2] / 2 + 2
                   or c[1] + c[2] / 2 < b[1] - b[2] / 2 - 2 for b in boxes)
    kept = [placed[0]] if placed else []
    last = placed[-1] if len(placed) > 1 else None
    for cand in placed[1:-1]:
        if _clear(cand, kept) and (last is None or _clear(cand, [last])):
            kept.append(cand)
    if last is not None:
        kept.append(last)
    for val, x, tw, text in kept:
        mph = f"{cms_to_mph(val):.2f}"
        if val >= stops[-1][0]:
            mph += "+"  # open-ended top bin, matches the "100+" tick
        mw = d.textlength(mph, font=f_small)
        d.text((min(max(x - mw / 2, 2), W - mw - 2), by + bh + 19),
               mph, font=f_small, fill=(10, 10, 10))
    img.save(path)
    return [v for v, _x, _tw, _t in kept]

# Narrow product-local river-water boxes. The shared NOAA shoreline mask
# (medium-resolution, built for open-lake coastlines) reads 0 over most
# of these sub-pixel rivers, which would erase their vectors. Alpha is
# restored ONLY where a source vector strictly lands (no-splat coverage)
# inside one of these boxes — and the shoreline is a HARD clip (mask
# majority or strict channel water: opaque; everything else alpha 0),
# so no feathered gradient fringe survives on land at any zoom. The
# shared mask asset itself is never touched. Validators import
# RIVER_BOXES and permit opaque pixels outside the shared mask solely
# inside these boxes.
RIVER_BOXES = [
    (-84.60, -84.10, 46.15, 46.62),   # St. Marys River
    (-82.64, -82.36, 42.56, 43.02),   # St. Clair River
    (-83.28, -82.96, 41.96, 42.42),   # Detroit River
    (-79.13, -78.90, 42.86, 43.36),   # Niagara River
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


def river_box_mask(bounds):
    """Bool canvas mask of RIVER_BOXES (for validators: opaque outside
    the shared shoreline is permitted only here)."""
    import numpy as _np
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    lon_ax = (bounds["lon_min"] + (_np.arange(W) + 0.5) / W
              * (bounds["lon_max"] - bounds["lon_min"]))
    lat_ax = (bounds["lat_max"] - (_np.arange(H) + 0.5) / H
              * (bounds["lat_max"] - bounds["lat_min"]))
    m = _np.zeros((H, W), bool)
    for x0, x1, y0, y1 in RIVER_BOXES:
        m |= ((lon_ax[None, :] >= x0) & (lon_ax[None, :] <= x1)
              & (lat_ax[:, None] >= y0) & (lat_ax[:, None] <= y1))
    return m


def nanmean3(grid):
    """NaN-aware 3x3 mean (display smoothing only; NaN stays NaN)."""
    m = np.isfinite(grid).astype(float)
    gf = np.where(np.isfinite(grid), grid, 0.0)
    mp, gp = np.pad(m, 1), np.pad(gf, 1)
    tot = np.zeros_like(gf)
    cnt = np.zeros_like(gf)
    for dr in range(3):
        for dc in range(3):
            tot += gp[dr:dr + gf.shape[0], dc:dc + gf.shape[1]]
            cnt += mp[dr:dr + gf.shape[0], dc:dc + gf.shape[1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1.0),
                        np.nan)


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


def sample_uv(y, x, uu, vv):
    """Bilinear (U,V) at fractional canvas coords; (nan, nan) when any
    corner is non-finite or out of bounds (stops traces at land/no-data)."""
    H, W = uu.shape
    if not (0.0 <= y <= H - 1.001 and 0.0 <= x <= W - 1.001):
        return (float("nan"), float("nan"))
    y0, x0 = int(y), int(x)
    y1, x1 = min(y0 + 1, H - 1), min(x0 + 1, W - 1)
    fy, fx = y - y0, x - x0
    try:
        u = (uu[y0, x0] * (1 - fx) + uu[y0, x1] * fx) * (1 - fy) + \
            (uu[y1, x0] * (1 - fx) + uu[y1, x1] * fx) * fy
        v = (vv[y0, x0] * (1 - fx) + vv[y0, x1] * fx) * (1 - fy) + \
            (vv[y1, x0] * (1 - fx) + vv[y1, x1] * fx) * fy
    except IndexError:
        return (float("nan"), float("nan"))
    if not (np.isfinite(u) and np.isfinite(v)):
        return (float("nan"), float("nan"))
    return (float(u), float(v))


def trace_streamline(sy, sx, uu, vv, ds=3.0, max_steps=24, min_speed=0.01):
    """Integrate one downstream path (RK2 midpoint) through the filed
    vector field. Canvas coords: +x east, +y SOUTH (row down); northward
    +V steps y down. Returns [(y, x, dx, dy)] canvas-unit tangents.
    Stops at land/no-data, stagnant water, or sharp hairpins (noisy
    cells must not sling arrows across the map). Pure function of the
    source field — every position and tangent is field-derived."""
    path = []
    y, x = float(sy), float(sx)
    px, py = None, None
    for _ in range(max_steps):
        u, v = sample_uv(y, x, uu, vv)
        sp = math.hypot(u, v)
        if not math.isfinite(sp) or sp < min_speed:
            break
        dx, dy = u / sp, -v / sp
        mx, my = x + dx * ds / 2.0, y + dy * ds / 2.0
        u2, v2 = sample_uv(my, mx, uu, vv)
        sp2 = math.hypot(u2, v2)
        if not math.isfinite(sp2) or sp2 < min_speed:
            break
        ax, ay = dx + u2 / sp2, dy + (-v2 / sp2)
        n = math.hypot(ax, ay)
        if n < 1e-9:
            break
        ax, ay = ax / n, ay / n
        if px is not None and (ax * px + ay * py) < 0.3:
            break  # hairpin: noisy/convergent cells, stop the trace
        x, y = x + ax * ds, y + ay * ds
        if not (8 <= y < uu.shape[0] - 8 and 8 <= x < uu.shape[1] - 8):
            break
        path.append((y, x, ax, ay))
        px, py = ax, ay
    return path


def paint_flow_arrows(rgba, uu, vv, seed_step=7, river_seed_step=4,
                      river_mask=None, ds=3.0, max_steps=24,
                      speed_ref_cms=50.0, min_speed_cms=1.0,
                      sep_px=1.5, river_sep_px=1.0,
                      head_every=3, head_sep_px=5.0,
                      river_head_sep_px=3.5,
                      line_base_alpha=90, line_bright_alpha=110,
                      head_alpha=230):
    """Dense flow-streak field with micro direction heads, rasterized.

    Jittered seeds (seeded RNG: deterministic per source field) advect
    downstream along RK2 streamlines through the filed (U,V) field; every
    accepted path is drawn as a 1 px streamlet, and every head_every-th
    path gets a tiny downstream chevron. Position, orientation, AND path
    of every mark come from integrating the vector field — never a
    preset lattice. Separation-aware seeding (Jobard-Lefer style) keeps
    converging flow from piling into blobs.
    Streamlet BRIGHTNESS encodes speed (dim drift -> bright jets), the
    background gradient carries the absolute scale. Returns
    (rgba, n_streamlets, n_heads).
    """
    H, W = uu.shape
    if river_mask is None:
        river_mask = np.zeros((H, W), bool)
    rng = np.random.default_rng(7)
    base = Image.fromarray(rgba, mode="RGBA")
    base_alpha = rgba[:, :, 3]
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    n_lines = n_heads = 0
    accepted = 0
    covered = np.zeros((H, W), bool)
    rr, cc = np.ogrid[:H, :W]

    def claim(path, sep):
        s = int(math.ceil(sep))
        for (y, x, _ax, _ay) in path[::3]:
            r0, r1 = max(0, int(y) - s), min(H, int(y) + s + 1)
            c0, c1 = max(0, int(x) - s), min(W, int(x) + s + 1)
            dy = rr[r0:r1, 0:1] - y
            dx = cc[0:1, c0:c1] - x
            covered[r0:r1, c0:c1][(dy * dy + dx * dx) <= sep * sep] = True

    def seeds(step):
        pts = []
        for r0 in range(0, H, step):
            for c0 in range(0, W, step):
                r = r0 + step / 2.0 + (rng.random() - 0.5) * step * 0.66
                c = c0 + step / 2.0 + (rng.random() - 0.5) * step * 0.66
                pts.append((r, c))
        return pts

    def brightness(sp):
        return int(round(line_base_alpha + line_bright_alpha
                         * min(1.0, sp * 100.0 / speed_ref_cms)))

    min_speed = min_speed_cms / 100.0
    placed = np.zeros((H, W), bool)

    def clear_for_head(y, x, sep):
        s = int(math.ceil(sep))
        r0, r1 = max(0, int(y) - s), min(H, int(y) + s + 1)
        c0, c1 = max(0, int(x) - s), min(W, int(x) + s + 1)
        dy = rr[r0:r1, 0:1] - y
        dx = cc[0:1, c0:c1] - x
        win = placed[r0:r1, c0:c1]
        hit = (dy * dy + dx * dx) <= sep * sep
        if bool((win & hit).any()):
            return False
        win[hit] = True
        return True

    for step, in_river, sep, hsep in ((seed_step, False, sep_px,
                                       head_sep_px),
                                      (river_seed_step, True, river_sep_px,
                                       river_head_sep_px)):
        for (r, c) in seeds(step):
            if not (8 <= r < H - 8 and 8 <= c < W - 8):
                continue
            if bool(river_mask[int(r), int(c)]) != in_river:
                continue  # lake pass skips rivers; river pass only rivers
            if covered[int(r), int(c)]:
                continue  # too close to an accepted path: skip the seed
            u0, v0 = sample_uv(r, c, uu, vv)
            sp0 = math.hypot(u0, v0)
            if not math.isfinite(sp0) or sp0 < min_speed:
                continue
            path = trace_streamline(r, c, uu, vv, ds=ds,
                                    max_steps=max_steps,
                                    min_speed=min_speed)
            if len(path) < 6:
                continue  # stagnant pocket: no mark, no coverage claim
            claim(path, sep)
            pts = [(x, y) for (y, x, _ax, _ay) in path[::2]]
            if len(pts) < 2:
                continue
            if base_alpha[int(path[-1][0]), int(path[-1][1])] == 0:
                continue  # path ran off water: no mark
            u, v = sample_uv(path[-1][0], path[-1][1], uu, vv)
            sp = math.hypot(u, v)
            if not math.isfinite(sp):
                continue
            d.line(pts, fill=(255, 255, 255, brightness(max(sp, min_speed))),
                   width=1)
            n_lines += 1
            accepted += 1
            if accepted % head_every == 0 and len(path) >= 10:
                y, x, ax, ay = path[-1]
                if not clear_for_head(y, x, hsep):
                    continue  # a head already owns this spot
                ang = math.atan2(ay, ax)
                hl = 3.5
                for s in (1, -1):
                    ha = ang + s * (math.pi - 0.6)
                    d.line([(x, y),
                            (x + math.cos(ha) * hl,
                             y + math.sin(ha) * hl)],
                           fill=(255, 255, 255, head_alpha), width=1)
                n_heads += 1
    del d
    out = Image.alpha_composite(base, overlay)
    return np.array(out), n_lines, n_heads


def paint_sample_arrow(path):
    """Paint the direction-key row onto a finished scale legend (in place).

    Draws the note row itself (streamlet + chevron sample, matching the
    raster marks) so nothing collides with the source line above it.
    """
    from geospatial_utils import _legend_font
    img = Image.open(path).convert("RGBA")
    d = ImageDraw.Draw(img)
    f_small = _legend_font(13)
    W, H = img.size
    y = H - 24
    x0, x1 = 16, 120
    yy = y + 2
    # dark slate sample (the raster streaks are white-on-gradient; on the
    # white card the same marks are shown dark for legibility)
    d.line([(x0, yy), (x1, yy)], fill=(30, 60, 120, 255), width=2)
    ang = 0.0
    for s in (1, -1):
        ha = ang + s * (math.pi - 0.6)
        d.line([(x1, yy),
                (x1 + math.cos(ha) * 8, yy + math.sin(ha) * 8)],
               fill=(30, 60, 120, 255), width=2)
    d.text((132, y - 6),
           "COLOR = speed. Streaks ride the flow (brighter = faster).",
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
    strict_water = np.zeros((H, W), bool)  # cells a source vector lands in
    for prefix, f in op_fields.items():    # (no splat: hard water edge)
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
        _one = np.ones(um.size)
        _cov, _c0 = bin_to_canvas(rows, cols, _one, src_ok, (H, W),
                                  splat_radius=0)
        strict_water |= _c0 > 0
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
        _one = np.ones(f["u"].size)
        _cov, _c0 = bin_to_canvas(rows, cols, _one, src_ok, (H, W),
                                  splat_radius=0)
        strict_water |= _c0 > 0
    n_all = int(np.isfinite(spd_c).sum())
    n_fill = n_all - n_op
    print(f"[{PRODUCT}] canvas: operational {n_op} + experimental fill "
          f"{n_fill} = {n_all} water vectors")
    if n_all < 5_000:
        raise ValueError(f"too few canvas vectors ({n_all})")
    # river display smoothing: sub-pixel channels bin into harsh
    # stair-steps; a NaN-aware 3x3 mean inside RIVER_BOXES only softens
    # the steps (documented display resampling; streaks trace the same
    # smoothed field so marks match colors).
    _rb = river_box_mask(bounds)
    for _grid in (spd_c, uu_c, vv_c):
        _sm = nanmean3(_grid)
        _use = _rb & np.isfinite(_grid) & np.isfinite(_sm)
        _grid[_use] = _sm[_use]
    # basin display smoothing: one NaN-aware 3x3 pass over speed only, so
    # deep-zoom magnification degrades into a smooth wash rather than hard
    # mosaic squares (vectors stay unsmoothed; traces keep full direction
    # fidelity). Same documented display-resampling class as HRRR layers.
    _sm_all = nanmean3(spd_c)
    _use_all = np.isfinite(spd_c) & np.isfinite(_sm_all)
    spd_c[_use_all] = _sm_all[_use_all]
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
    vec_ok = np.isfinite(spd_c)  # authoritative vector present (pre-mask)
    wm = load_watermask()
    river_boxes = river_box_mask(bounds)
    # HARD shoreline: water pixels fully opaque, everything else fully
    # transparent — no feathered fringe on land at any zoom. Open lakes
    # follow the shared mask majority (>=0.5); rivers (sub-pixel for the
    # mask) render only where a source vector strictly lands (no splat
    # smear, no bleed halo).
    river_allow = strict_water & river_boxes  # product-local channel water
    keep = ((wm >= 0.5) | river_allow).astype(np.float32)
    # arrows ride the flow (painted BEFORE masking so river glyphs land)
    uu_w = np.where(vec_ok, uu_c, np.nan)
    vv_w = np.where(vec_ok, vv_c, np.nan)
    rgba, n_lines, n_heads = paint_flow_arrows(
        rgba, uu_w, vv_w,
        seed_step=CONFIG["seed_step_px"],
        river_seed_step=CONFIG["river_seed_step_px"],
        river_mask=river_boxes,
        ds=CONFIG["stream_ds_px"], max_steps=CONFIG["stream_max_steps"],
        speed_ref_cms=CONFIG["arrow_speed_ref_cms"],
        min_speed_cms=CONFIG["arrow_min_speed_cms"])
    # shoreline: hard clip — shared mask majority on lakes, strict
    # channel water in rivers; land alpha is exactly 0 everywhere
    # (keep is binary, so no feathered fringe survives on land).
    rgba[:, :, 3] = (rgba[:, :, 3].astype(np.float32) * keep).round().astype(
        np.uint8)
    rgba = bleed_rgb_into_transparent(rgba)  # anti-fringe for GE bilinear
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    n_river = int((rgba[:, :, 3] > 0).sum() - (np.round(
        rgba[:, :, 3].astype(np.float32) * wm) > 0).sum())
    print(f"[{PRODUCT}] opaque={n_opaque} (river-restored ~{n_river}) "
          f"streamlets={n_lines} heads={n_heads} "
          f"speed max={cur_max:.1f} med={cur_med:.1f} cm/s")
    if n_opaque < 5_000:
        raise ValueError("empty raster")
    if n_lines < CONFIG["min_streamlets"]:
        raise ValueError(f"too few streamlets ({n_lines})")
    if n_heads < CONFIG["min_heads"]:
        raise ValueError(f"too few direction heads ({n_heads})")

    # ---- legend + metadata + KML ----
    valid_isos = [f["valid_iso"] for f in op_fields.values()]
    data_iso = max(valid_isos)
    data_time_utc = valid_to_det(data_iso)
    unit = CONFIG["display_units"]
    subtitle = (f"Speed: cm/s top, mph bottom  |  {data_time_utc}  |  "
                f"max {cur_max:.0f} cm/s")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        "cm/s", CUR_STOPS, CUR_LABELS,
        f"Source: NOAA GLOFS nowcast + GLERL corridor fill  |  "
        f"Processed {now_det_str()}",
        note=None)
    paint_mph_row(os.path.join(stage_prod, "legend.png"),
                  CUR_STOPS, CUR_LABELS)
    paint_sample_arrow(os.path.join(stage_prod, "legend.png"))
    scale_html = (
        "Surface-current speed (centimeters per second, one continuous "
        "gradient): <b>LOWEST 0</b> dark-blue stagnant &rarr; <b>10</b> teal "
        "&rarr; <b>20</b> yellow &rarr; <b>30</b> orange &rarr; <b>50</b> red "
        "&rarr; <b>75</b> red-violet &rarr; <b>HIGHEST+ 100 cm/s</b> "
        "dark-purple channel jets. Same speed always shows the same color; "
        "above 100 stays dark-purple. Thousands of fine white streaks RIDE "
        "the flow — each traces a short RK2 streamline of the filed U/V "
        "field (brighter streak = faster water), with tiny chevrons "
        "pointing downstream. Now: max "
        f"<b>{cur_max:.0f}</b>, median <b>{cur_med:.1f} cm/s</b>, median "
        f"flow <b>{mean_hd:.0f}&deg; ({compass(mean_hd)})</b>. NO DATA stays "
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
    meta["arrow_method"] = (
        "No preset lattice. Deterministic jittered seeds (seeded RNG) "
        "advected downstream along RK2-midpoint streamlines (3 px steps, "
        "up to 24 steps ≈ 72 px paths) integrated through the filed U/V "
        "field with bilinear sampling; each accepted path is drawn as a "
        "1 px streamlet, every third path carries a tiny downstream "
        "chevron. Traces stop at land/no-data, stagnant water (<1 cm/s), "
        "or hairpins. Streamlet brightness encodes speed (dim drift, "
        "bright jets); rivers get denser seeds. Glyphs trace the same "
        "field as the gradient colors.")
    meta["river_treatment"] = (
        "The shared open-lake shoreline mask reads land over sub-pixel "
        "rivers, so river alpha is restored product-locally ONLY where an "
        "authoritative source vector exists inside the documented "
        "RIVER_BOXES (St. Marys, St. Clair, Detroit, Niagara). Shared mask "
        "asset untouched; no other layer affected.")
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
                     "streamlets_drawn": n_lines,
                     "direction_heads_drawn": n_heads,
                     "arrows_drawn": n_heads}
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
