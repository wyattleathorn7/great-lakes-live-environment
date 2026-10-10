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
# g7 = streamlet direction streaks (surface-current technique, brightness =
# wind speed) + actual knot values in the same halo style (was: lattice
# arrows, no values). New token by construction so caches cannot serve
# the old pixels under the old URL.
# g8 = free (data-seeded, non-grid) label placement + whiter streaks/heads.
# g9 = kt value labels DISABLED by default (gradient + streamlets only);
#      re-enable via --show-kt-labels. New token so caches cannot serve
#      the old labeled pixels under the old URL.
RENDER_TAG = "fullbasin-g9-labels-disabled"

# Live kt value labels (white-halo "12kt" numbers painted into the raster).
# Disabled by default: gradient + streamlet streaks remain, numeric labels
# are skipped. Function paint_wind_labels() is kept intact so this can be
# re-enabled via CLI (--show-kt-labels) without restoring deleted code.
SHOW_KT_LABELS = False

# Inland lakes that read as LAND in the shared NOAA shoreline mask (which
# covers the five Great Lakes) but get the full 80% water treatment here:
# (name, lon_min, lon_max, lat_min, lat_max). Lake pixels are identified by
# the committed landcover water class (code 9) inside each box — surrounding
# land is never affected. GLWU has no valid mesh cells here, so the HRRR
# fill supplies the (live) values; only the opacity + palette follow water.
INLAND_LAKES = [
    ("nipigon", -89.2, -87.9, 49.0, 49.5),
    ("nipissing", -80.4, -79.3, 46.15, 46.45),
    ("simcoe", -79.7, -79.1, 44.15, 44.65),
    ("winnebago", -88.8, -88.2, 43.7, 44.3),
]

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
        hh, min(1.0, s * 1.3), min(1.0, v * 1.10 + 0.02))
    return (round(r2 * 255), round(g2 * 255), round(b2 * 255))


def inland_lake_mask(bounds, shape):
    """Bool canvas mask of the INLAND_LAKES water surfaces (wind-local).

    The shared NOAA shoreline mask excludes these lakes, so this draws on
    the committed landcover water class (code 9 = open water) clipped to
    tight per-lake boxes. Raises on missing/wrong-sized asset (loud
    failure; previous raster kept via exit codes).
    """
    import numpy as _np
    from PIL import Image as _Im
    from geospatial_utils import REPO_ROOT as _RR
    H, W = shape
    p = os.path.join(_RR, "assets", "leaf_landcover.png")
    lc = _np.array(_Im.open(p).convert("L"))
    if lc.shape != (H, W):
        raise ValueError(f"landcover shape {lc.shape} != canvas {(H, W)}")
    out = _np.zeros((H, W), dtype=bool)
    for _name, lo0, lo1, la0, la1 in INLAND_LAKES:
        c0 = max(int((lo0 - bounds["lon_min"])
                     / (bounds["lon_max"] - bounds["lon_min"]) * W), 0)
        c1 = min(int((lo1 - bounds["lon_min"])
                     / (bounds["lon_max"] - bounds["lon_min"]) * W) + 1, W)
        r0 = max(int((bounds["lat_max"] - la1)
                     / (bounds["lat_max"] - bounds["lat_min"]) * H), 0)
        r1 = min(int((bounds["lat_max"] - la0)
                     / (bounds["lat_max"] - bounds["lat_min"]) * H) + 1, H)
        out[r0:r1, c0:c1] |= (lc[r0:r1, c0:c1] == 9)
    return out


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


# ------------------------------------------- streamlet direction + KT labels
# Same treatment as the combined wave layer (ported technique, verbal style
# match): RK2 streamlets follow the (U,V) movement field (GRIB components
# already point toward-motion, no reversal), streak brightness encodes wind
# speed, micro-chevrons mark travel; actual knot values ride in the same
# white-halo mono style as the wave period labels. Fully rasterized.


def _wind_label_font(size=7):
    """DejaVu Sans Mono Bold (same face as the wave labels)."""
    import glob as _glob
    from PIL import ImageFont as _IF
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ]
    try:
        import matplotlib as _mpl
        cands.append(os.path.join(os.path.dirname(_mpl.__file__),
                                  "mpl-data", "fonts", "ttf",
                                  "DejaVuSansMono-Bold.ttf"))
    except Exception:
        pass
    cands += _glob.glob("/System/Library/Fonts/Supplemental/DejaVuSansMono-Bold*.ttf")
    cands += _glob.glob(os.path.expanduser("~/Library/Fonts/DejaVuSansMono-Bold.ttf"))
    for path in cands:
        try:
            return _IF.truetype(path, size)
        except (OSError, IOError):
            continue
    from geospatial_utils import _legend_font
    return _legend_font(max(size, 8))


def _draw_wind_label(d, x, y, text, font):
    """White core + 1 px outline stroke (same as the wave period labels)."""
    d.text((x, y), text, font=font, fill=(255, 255, 255, 255),
           stroke_width=1, stroke_fill=(20, 20, 20, 235))


def _wsample_uv(y, x, uu, vv):
    """Bilinear (U,V) at fractional canvas coords; (nan, nan) at no-data."""
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


def _wtrace(sy, sx, uu, vv, ds=3.0, max_steps=40, min_speed=0.5):
    path = []
    y, x = float(sy), float(sx)
    px, py = None, None
    for _ in range(max_steps):
        u, v = _wsample_uv(y, x, uu, vv)
        sp = math.hypot(u, v)
        if not math.isfinite(sp) or sp < min_speed:
            break
        dx, dy = u / sp, -v / sp
        mx, my = x + dx * ds / 2.0, y + dy * ds / 2.0
        u2, v2 = _wsample_uv(my, mx, uu, vv)
        sp2 = math.hypot(u2, v2)
        if not math.isfinite(sp2) or sp2 < min_speed:
            break
        ax, ay = dx + u2 / sp2, dy + (-v2 / sp2)
        n = math.hypot(ax, ay)
        if n < 1e-9:
            break
        ax, ay = ax / n, ay / n
        if px is not None and (ax * px + ay * py) < 0.3:
            break
        x, y = x + ax * ds, y + ay * ds
        if not (8 <= y < uu.shape[0] - 8 and 8 <= x < uu.shape[1] - 8):
            break
        path.append((y, x, ax, ay))
        px, py = ax, ay
    return path


def paint_wind_streamlets(rgba, uu, vv, kt_field, seed_step=20,
                          ds=3.0, max_steps=40, min_speed=0.5,
                          sep_px=3.0, head_every=2, head_sep_px=5.0,
                          line_width=1, head_alpha=230, head_len=4.0):
    """RK2 streamlets through the wind movement field, brightness = knots.

    Dim streaks = light air, bright streaks = gale; micro-chevrons mark
    travel. Fully rasterized (no KML vectors). Returns
    (rgba, n_streamlets, n_heads). Paint AFTER the dual-opacity merge so
    the speed brightness survives (the merge flattens glyph alphas).
    """
    H, W = uu.shape
    rng = np.random.default_rng(7)
    base = Image.fromarray(rgba, mode="RGBA")
    base_alpha = rgba[:, :, 3]
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    n_lines = n_heads = accepted = 0
    covered = np.zeros((H, W), bool)
    placed = np.zeros((H, W), bool)
    rr, cc = np.ogrid[:H, :W]

    def claim(path, sep):
        s = int(math.ceil(sep))
        for (y, x, _ax, _ay) in path[::3]:
            r0, r1 = max(0, int(y) - s), min(H, int(y) + s + 1)
            c0, c1 = max(0, int(x) - s), min(W, int(x) + s + 1)
            dy = rr[r0:r1, 0:1] - y
            dx = cc[0:1, c0:c1] - x
            covered[r0:r1, c0:c1][(dy * dy + dx * dx) <= sep * sep] = True

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

    for r0 in range(0, H, seed_step):
        for c0 in range(0, W, seed_step):
            r = r0 + seed_step / 2.0 + (rng.random() - 0.5) * seed_step * 0.66
            c = c0 + seed_step / 2.0 + (rng.random() - 0.5) * seed_step * 0.66
            if not (8 <= r < H - 8 and 8 <= c < W - 8):
                continue
            if covered[int(r), int(c)]:
                continue
            u0, v0 = _wsample_uv(r, c, uu, vv)
            if not (math.isfinite(u0) and math.hypot(u0, v0) >= min_speed):
                continue
            path = _wtrace(r, c, uu, vv, ds=ds, max_steps=max_steps,
                           min_speed=min_speed)
            if len(path) < 6:
                continue
            claim(path, sep_px)
            pts = [(x, y) for (y, x, _ax, _ay) in path[::2]]
            if len(pts) < 2:
                continue
            if base_alpha[int(path[-1][0]), int(path[-1][1])] == 0:
                continue
            kts = [float(kt_field[int(y), int(x)])
                   for (y, x, _a, _b) in path[::4]
                   if np.isfinite(kt_field[int(y), int(x)])]
            kt = sum(kts) / len(kts) if kts else 0.0
            alpha = int(round(110 + 110 * min(1.0, max(0.0, kt) / 35.0)))
            d.line(pts, fill=(255, 255, 255, alpha), width=line_width)
            n_lines += 1
            accepted += 1
            if accepted % head_every == 0 and len(path) >= 10:
                y, x, ax, ay = path[-1]
                if not clear_for_head(y, x, head_sep_px):
                    continue
                ang = math.atan2(ay, ax)
                for s in (1, -1):
                    ha = ang + s * (math.pi - 0.6)
                    d.line([(x, y),
                            (x + math.cos(ha) * head_len,
                             y + math.sin(ha) * head_len)],
                           fill=(255, 255, 255, 255), width=1)
                n_heads += 1
    del d
    out = Image.alpha_composite(base, overlay)
    return np.array(out), n_lines, n_heads


def paint_wind_labels(rgba, kt_field, label_every=2, extra_step=64,
                      extra_thresh_kt=4.0, font_size=7, max_labels=260):
    """Actual knot values in the wave-label style: white mono + dark stroke.

    NO fixed grid: candidates come from a data-seeded jittered lattice, so
    positions move with the weather instead of sitting on a rigid lattice.
    High-change cells (fronts/gust gradients) go first with tight spacing;
    the rest fill a quota with wide spacing. Box-collision avoidance
    throughout. Returns (rgba, n_priority, n_fill).
    """
    H, W = kt_field.shape
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    font = _wind_label_font(font_size)
    ok = np.isfinite(kt_field)
    if not ok.any():
        del d
        return np.array(img), 0, 0
    # deterministic-per-data seed: same field -> same layout (no flicker
    # between rebuilds of one cycle); new weather -> new positions.
    _fin = kt_field[ok]
    seed = int(abs(float(_fin.mean())) * 1000 + float(_fin.std()) * 97) \
        % (2 ** 32 - 1)
    rng = np.random.default_rng(seed)
    # local change map (12 px neighborhood max abs diff)
    grad = np.zeros(kt_field.shape)
    for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
        shifted = np.roll(kt_field, shift=(-dr, -dc), axis=(0, 1))
        diff = np.abs(kt_field - shifted)
        both = ok & np.isfinite(shifted)
        grad = np.maximum(grad, np.where(both, diff, 0.0))
    # jittered candidates on a fine lattice (no visible grid)
    step = 32
    cand = []
    for r0 in range(0, H, step):
        for c0 in range(0, W, step):
            r = r0 + step / 2.0 + (rng.random() - 0.5) * step * 0.9
            c = c0 + step / 2.0 + (rng.random() - 0.5) * step * 0.9
            r, c = int(round(r)), int(round(c))
            if not (8 <= r < H - 8 and 8 <= c < W - 8):
                continue
            if not ok[r, c] or rgba[r, c, 3] == 0:
                continue
            cand.append((grad[r, c], rng.random(), r, c))
    # priority first (sharp change), then fill in random order
    cand.sort(key=lambda t: (-t[0], t[1]))
    placed = []
    n_priority = n_fill = 0

    def _collides(x0, y0, x1, y1, pad=3):
        for (a0, b0, a1, b1) in placed:
            if not (x1 + pad < a0 or x0 - pad > a1 or y1 + pad < b0 or y0 - pad > b1):
                return True
        return False

    def _put(cx, cy, text, sep):
        tw = d.textlength(text, font=font)
        th = font_size + 4
        x = cx - tw / 2
        y = cy - th / 2
        if not (4 <= x < W - tw - 4 and 4 <= y < H - th - 4):
            return False
        if _collides(x, y, x + tw, y + th):
            return False
        # separation from other labels (wide for fill, tight for fronts)
        for (a0, b0, a1, b1) in placed:
            acx, acy = (a0 + a1) / 2, (b0 + b1) / 2
            if math.hypot(acx - cx, acy - cy) < sep:
                return False
        pts = [(cx, cy), (x + 2, y + 2), (x + tw - 2, y + 2)]
        for px, py in pts:
            ix, iy = int(np.clip(round(px), 0, W - 1)), int(np.clip(round(py), 0, H - 1))
            if rgba[iy, ix, 3] == 0:
                return False
        if not np.isfinite(kt_field[int(np.clip(round(cy), 0, H - 1)),
                                    int(np.clip(round(cx), 0, W - 1))]):
            return False
        _draw_wind_label(d, x, y, text, font)
        placed.append((x, y, x + tw, y + th))
        return True

    for g, _, r, c in cand:
        if n_priority + n_fill >= max_labels:
            break
        if g < extra_thresh_kt:
            break  # remaining candidates are all low-change: fill pass next
        v = float(kt_field[r, c])
        if not (0.0 <= v <= 120.0):
            continue
        if _put(c, r - 12, f"{v:.0f}kt", sep=30):
            n_priority += 1
    rest = [t for t in cand if t[0] < extra_thresh_kt]
    rng.shuffle(rest)
    for _, _, r, c in rest:
        if n_priority + n_fill >= max_labels:
            break
        v = float(kt_field[r, c])
        if not (0.0 <= v <= 120.0):
            continue
        if _put(c, r - 12, f"{v:.0f}kt", sep=48):
            n_fill += 1

    del d
    return np.array(img), n_priority, n_fill


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
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--direction-style", default="stream",
                        choices=["arrows", "stream"],
                        help="direction glyphs: streamlets or classic arrows")
        ap.add_argument("--preview-dir", default=None,
                        help="preview: write current.png here, skip promote")
        ap.add_argument("--font-size", type=int, default=7)
        ap.add_argument("--label-every", type=int, default=2)
        ap.add_argument("--extra-step", type=int, default=64)
        ap.add_argument("--extra-thresh", type=float, default=4.0,
                        help="kt-change threshold for extra labels")
        ap.add_argument("--show-kt-labels", action="store_true",
                        default=SHOW_KT_LABELS,
                        help="re-enable live kt value labels "
                             "(default: disabled)")
        args = ap.parse_args()
        return run(direction_style=args.direction_style,
                   preview_dir=args.preview_dir, font_size=args.font_size,
                   label_every=args.label_every, extra_step=args.extra_step,
                   extra_thresh=args.extra_thresh,
                   show_kt_labels=args.show_kt_labels)
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1

def run(direction_style="stream", preview_dir=None, font_size=7,
        label_every=2, extra_step=64, extra_thresh=4.0,
        show_kt_labels=SHOW_KT_LABELS):
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
        return _build(got, url, datestr, cycle, raw_path, direction_style,
                      preview_dir, font_size, label_every, extra_step,
                      extra_thresh, show_kt_labels)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(got, used_url, datestr, cycle, raw_path, direction_style="stream",
           preview_dir=None, font_size=7, label_every=2, extra_step=64,
           extra_thresh=4.0, show_kt_labels=SHOW_KT_LABELS):
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
    _lake = inland_lake_mask(bounds, (H, W))
    _water = (_wm >= 0.5) | _lake
    n_inland = int((_lake & np.isfinite(field)).sum())
    print(f"[{PRODUCT}] inland-lake water cells "
          f"(nipigon/nipissing/simcoe/winnebago): {n_inland}")
    lut_water = np.zeros((13, 3), dtype=np.uint8)
    lut_land = np.zeros((13, 3), dtype=np.uint8)
    for f in range(13):
        lut_water[f] = _rgb(FORCE_COLORS[f])
        lut_land[f] = _vivid_for_land(FORCE_COLORS[f])
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    ook = np.isfinite(field)
    fi = np.clip(np.round(field[ook]).astype(int), 0, 12)
    _is_water = _water[ook]
    rgba[ook, 0:3] = np.where(_is_water[:, None], lut_water[fi], lut_land[fi])
    rgba[ook, 3] = WIND_ALPHA_LAND

    kt_canvas = np.hypot(comb_u, comb_v) * MS_TO_KT  # m/s -> kt on canvas
    n_lines = n_heads = n_lab = n_extra = 0
    if direction_style == "stream":
        n_arrows = 0  # replaced below by stream heads (validator continuity)
    else:
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
    # Inland lakes (nipigon/nipissing/simcoe/winnebago) join the 205 group
    # via the combined water test below.
    _wfrac = np.maximum(_wm, _lake.astype(float))
    _dual = np.round(WIND_ALPHA_LAND + (WIND_ALPHA_WATER - WIND_ALPHA_LAND)
                     * _wfrac).astype(np.uint8)
    _painted = rgba[:, :, 3] > 0
    rgba[_painted, 3] = _dual[_painted]
    # Full basin rectangle (atmospheric layer, like leaf footprint): only
    # missing source data is transparent. RGB bleed keeps
    # the anti-fringe contract for bilinear clients (Google Earth).
    rgba = bleed_rgb_into_transparent(rgba)
    if direction_style == "stream":
        # Painted AFTER the merge so streak speed-brightness survives (the
        # merge flattens glyph alphas) — same order as the wave layer.
        rgba, n_lines, n_heads = paint_wind_streamlets(
            rgba, comb_u, comb_v, kt_canvas)
        if show_kt_labels:
            rgba, n_lab, n_extra = paint_wind_labels(
                rgba, kt_canvas, label_every=label_every,
                extra_step=extra_step, extra_thresh_kt=extra_thresh,
                font_size=font_size)
        else:
            # Live kt value display DISABLED: keep gradient + streamlets,
            # skip numeric labels (paint_wind_labels kept for re-enable).
            n_lab, n_extra = 0, 0
        n_arrows = n_heads
        print(f"[{PRODUCT}] streamlets={n_lines}+{n_heads}heads "
              f"kt_labels={n_lab}+{n_extra}extra "
              f"(show_kt_labels={show_kt_labels})")
        if n_arrows < 50:
            print(f"[{PRODUCT}] VALIDATION FAILED: too few stream heads. "
                  f"Keeping previous.")
            return 2
        if show_kt_labels and n_lab + n_extra < 20:
            print(f"[{PRODUCT}] VALIDATION FAILED: too few kt labels. "
                  f"Keeping previous.")
            return 2
    if preview_dir is not None:
        os.makedirs(preview_dir, exist_ok=True)
        save_png(rgba, os.path.join(preview_dir, "current.png"))
        print(f"[{PRODUCT}] PREVIEW {direction_style} written.")
        return 0
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
                  f"White streaks trace travel (brightness = knots, dim calm "
                  f"to bright gale)"
                  + ("; white halo numbers are actual kt at "
                     "that spot." if show_kt_labels else "."))
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
             "same force reads the same (single key). White streaks trace "
             "travel (brightness = knots)"
             + ("; white numbers are actual kt." if show_kt_labels else "."))

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
                                "~55%) with inland lakes nipigon/nipissing/"
                                "simcoe/winnebago (landcover water class, "
                                "wind-local boxes) at full water opacity; "
                                "transparent only where both sources lack "
                                "data; never zero-filled."))
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
        "transparency (single legend, no second key); inland lakes "
        "nipigon/nipissing/simcoe/winnebago render at full water opacity "
        "via the wind-local landcover water mask; edge RGB bled into "
        "transparent pixels (anti-fringe for bilinear clients).")
    meta["stats"] = {
        "valid_cells": n_combined,
        "glwu_canvas_cells": n_glwu_canvas,
        "hrrr_fill_cells": n_fill,
        "inland_lake_cells": n_inland,
        "hrrr_cycle": f"{hrrr_dd} t{hrrr_cc}z",
        "max_kt": round(float(spd_kt[valid].max()), 1),
        "max_beaufort": fmax,
        "max_beaufort_combined": fmax_fill,
        "arrows_drawn": n_arrows,
        "direction_style": direction_style,
        "streamlets_drawn": n_lines,
        "stream_heads_drawn": n_heads,
        "kt_labels_on_grid": n_lab,
        "kt_labels_extra": n_extra,
    }
    if direction_style == "stream":
        meta["methodology"] += (
            " Direction glyphs are RK2 streamlets through the (U,V) "
            "movement field (surface-current technique; streak brightness "
            "encodes knots, dim calm to bright gale; micro-chevrons mark "
            "travel)"
            + (", plus actual knot values (e.g. 12kt) in white mono "
               "with a dark outline, placed sparsely with extras where the "
               "wind changes sharply." if show_kt_labels
               else " (live kt value labels currently disabled)."))
    meta["stats"]["show_kt_labels"] = show_kt_labels

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