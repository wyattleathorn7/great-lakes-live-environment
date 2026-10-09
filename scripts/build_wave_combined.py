"""Pipeline C4 — LIVE WAVES COMBINED (new, independent, non-destructive).

Combines the two existing wave layers into ONE raster without editing them:

* BASE = existing wave-height gradient, verbatim:
  WAVE_FIXED_BINS (18 bins, user hexes) + wave_bin_index + render_wave_fixed
  copied exactly from scripts/build_wave_height.py.
* ARROWS = existing wave-direction arrows, verbatim:
  paint_arrows (toward = filed FROM + 180, white shaft + dark outline,
  step=16, L=10) copied exactly from scripts/build_wave_direction.py.
  The period COLOR WASH from that layer is deleted (not rendered).
* LABELS = actual peak-period values rendered JUST LIKE the arrows:
  same white-fill + dark-halo rasterized PIL text, placed right above a
  decimated subset of arrows, plus extras where period changes sharply
  even where no arrow sits nearby.

All three fields (HTSGW height, PERPW period, WVDIR direction) come from
the SAME GLWU cycle file (single download, step-0 analysis), so the three
channels can never disagree by cycle.

RASTER ONLY: everything (colors, arrows, numbers) is burned into one RGBA
PNG GroundOverlay. Zero LineString / Polygon / Placemark / Point /
ScreenOverlay (asserted). Existing wave_height / wave_direction products,
configs, and KMLs are never touched.

Exit codes: 0 updated (or skipped); 2 source/validation failure
(previous kept); 1 unexpected error.
"""

import argparse
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
                              grib_stamp_to_det, load_bounds, now_det_str,
                              promote_stage, read_state, save_png,
                              source_token, stage_dir, write_metadata,
                              write_state)

PRODUCT = "wave_combined"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = CONFIG.get("kml_file", "Great_Lakes_Live_Waves_Combined.kml")
_OVERLAY_RAW = CONFIG.get("overlay_name", "LIVE WAVE HEIGHT/PERIOD/DIRECTION")
OVERLAY_NAME = _OVERLAY_RAW if _OVERLAY_RAW.startswith("\U0001F30A") else "\U0001F30A " + _OVERLAY_RAW
SKIP_NOTE = "Turn on/off independently of all other layers."
UA = {"User-Agent": "great-lakes-live-environment/1.0"}
M_TO_FT = 3.28084

# ---------------------------------------------------------------- height base
# VERBATIM copy of the fixed 18-bin wave-height scale from
# scripts/build_wave_height.py (user-supplied hexes, gap-free edges).
# The combined layer MUST wear these exact colors — never blended.


def _hex_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


WAVE_FIXED_BINS = [
    ("0 ft", "#3156A0", 0.0, 0.2),
    ("0.5 ft", "#2675B8", 0.2, 0.7),
    ("1-2 ft", "#20A5C2", 0.7, 3.0),
    ("3-4 ft", "#32B878", 3.0, 4.0),
    ("4-5 ft", "#55A83A", 4.0, 6.0),
    ("6-7 ft", "#D6C43A", 6.0, 8.0),
    ("8-9 ft", "#D0A83A", 8.0, 10.0),
    ("10-11 ft", "#E07832", 10.0, 12.0),
    ("12-13 ft", "#C9573C", 12.0, 14.0),
    ("14-15 ft", "#C6283D", 14.0, 16.0),
    ("16-17 ft", "#9F3F68", 16.0, 18.0),
    ("18-19 ft", "#8E3FB3", 18.0, 20.0),
    ("20-21 ft", "#693D8C", 20.0, 22.0),
    ("22-23 ft", "#4F3475", 22.0, 24.0),
    ("24-25 ft", "#75344F", 24.0, 26.0),
    ("26-27 ft", "#8A245F", 26.0, 28.0),
    ("28-29 ft", "#A05F45", 28.0, 30.0),
    ("30+ ft", "#FFFFFF", 30.0, float("inf")),
]
WAVE_FIXED_RGB = [_hex_rgb(hx) for _, hx, _, _ in WAVE_FIXED_BINS]
assert [hx for _, hx, _, _ in WAVE_FIXED_BINS] == [
    "#3156A0", "#2675B8", "#20A5C2", "#32B878", "#55A83A", "#D6C43A",
    "#D0A83A", "#E07832", "#C9573C", "#C6283D", "#9F3F68", "#8E3FB3",
    "#693D8C", "#4F3475", "#75344F", "#8A245F", "#A05F45", "#FFFFFF"], \
    "combined base hexes must match wave_height verbatim"


def wave_bin_index(values_ft):
    """Bin index 0..17; NaN/negative -> -1 (transparent). Verbatim."""
    import numpy as _np
    v = _np.asarray(values_ft, dtype=float)
    idx = _np.searchsorted(
        _np.array([0.2, 0.7, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
                   16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]),
        v, side="right")
    idx = _np.clip(idx, 0, 17).astype(int)
    bad = ~(_np.isfinite(v) & (v >= 0))
    idx = _np.where(bad, -1, idx)
    return idx


def render_wave_fixed(field, alpha):
    """Exact preset render: every pixel takes its bin's verbatim hex."""
    import numpy as _np
    H, W = field.shape
    rgba = _np.zeros((H, W, 4), dtype=_np.uint8)
    idx = wave_bin_index(field)
    ok = idx >= 0
    if not ok.any():
        return rgba
    lut = _np.array(WAVE_FIXED_RGB, dtype=_np.uint8)
    rgba[ok, 0:3] = lut[idx[ok]]
    rgba[ok, 3] = alpha
    return rgba


# ------------------------------------------------------------------ arrows
# VERBATIM copy of paint_arrows + compass from build_wave_direction.py.
# Toward-travel vector (filed FROM + 180), white shaft + dark outline.

def paint_arrows(rgba, field_deg, step=16):
    """Toward-travel arrows from filed compass degrees (FROM + 180)."""
    H, W = field_deg.shape
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    n = 0
    positions = []
    for r in range(step // 2, H, step):
        for c in range(step // 2, W, step):
            if not np.isfinite(field_deg[r, c]):
                continue
            if rgba[r, c, 3] == 0:
                continue
            toward = (float(field_deg[r, c]) + 180.0) % 360.0
            rad = math.radians(toward)
            dx, dy = math.sin(rad), -math.cos(rad)  # east+, north-up
            L = 10.0
            x0, y0 = c - dx * L / 2, r - dy * L / 2
            x1, y1 = c + dx * L / 2, r + dy * L / 2
            if not (8 <= x0 < W - 8 and 8 <= y0 < H - 8
                    and 8 <= x1 < W - 8 and 8 <= y1 < H - 8):
                continue
            ang = math.atan2(dy, dx)
            for ext, w, col in ((2, 3, (20, 20, 20, 235)),
                                (0, 1, (255, 255, 255, 240))):
                d.line([(x0, y0), (x1, y1)], fill=col, width=2 + ext)
                for s in (1, -1):
                    ha = ang + s * (math.pi - 0.5)
                    d.line([(x1, y1),
                            (x1 + math.cos(ha) * 5, y1 + math.sin(ha) * 5)],
                           fill=col, width=2 + ext)
            n += 1
            positions.append((r, c))
    del d
    return np.array(img), n, positions


def compass(deg):
    pts = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    return pts[int(((float(deg) % 360) + 22.5) // 45)]


# ------------------------------------------------------------------- labels
# Period values in the SAME arrow style: white fill + dark halo, rasterized
# into the PNG (no KML vectors). Placed right above arrows + extras where
# the period field changes even with no arrow nearby.


def _label_font(size=13):
    from geospatial_utils import _legend_font
    return _legend_font(size)


def _draw_halo_text(d, x, y, text, font):
    """Arrow-style glyph: dark halo passes, then white core — like paint_arrows."""
    halo = (20, 20, 20, 235)
    core = (255, 255, 255, 245)
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, -1), (-1, 1), (1, 1)):
        d.text((x + ox, y + oy), text, font=font, fill=halo)
    d.text((x, y), text, font=font, fill=core)


def paint_period_labels(rgba, period_field, arrow_positions,
                        label_every=3, extra_step=48,
                        extra_thresh_s=0.6, font_size=13):
    """Burn PERPW values into the raster, arrow-style.

    Pass 1: label every `label_every`-th arrow, text centered right above
      the arrow shaft (dy = -16 px).
    Pass 2: extras on a coarse grid where the period changes sharply
      (|local gradient| >= extra_thresh_s) even with no arrow nearby,
      with box-collision avoidance so numbers never stack.

    Returns (rgba, n_arrow_labels, n_extra_labels).
    """
    H, W = period_field.shape
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    font = _label_font(font_size)
    placed = []  # (x0, y0, x1, y1) occupied boxes
    n_arrow = 0
    n_extra = 0

    def _collides(x0, y0, x1, y1, pad=3):
        for (a0, b0, a1, b1) in placed:
            if not (x1 + pad < a0 or x0 - pad > a1 or y1 + pad < b0 or y0 - pad > b1):
                return True
        return False

    def _put(cx, cy, text):
        nonlocal n_arrow, n_extra
        tw = d.textlength(text, font=font)
        th = font_size + 4
        x = cx - tw / 2
        y = cy - th / 2
        if not (4 <= x < W - tw - 4 and 4 <= y < H - th - 4):
            return False
        if _collides(x, y, x + tw, y + th):
            return False
        # whole glyph must sit on water: center + top corners opaque,
        # else shoreline alpha-clip would slice the text (e.g. ".2s").
        pts = [(cx, cy), (x + 2, y + 2), (x + tw - 2, y + 2)]
        for px, py in pts:
            ix, iy = int(np.clip(round(px), 0, W - 1)), int(np.clip(round(py), 0, H - 1))
            if rgba[iy, ix, 3] == 0:
                return False
        if rgba[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1)), 3] == 0:
            return False
        if not np.isfinite(period_field[int(np.clip(cy, 0, H - 1)),
                                        int(np.clip(cx, 0, W - 1))]):
            # allow label anchor slightly off-water only if its arrow cell
            # itself is valid; extras require a valid cell.
            return False
        _draw_halo_text(d, x, y, text, font)
        placed.append((x, y, x + tw, y + th))
        return True

    # Pass 1 — above arrows (decimated).
    for i, (r, c) in enumerate(arrow_positions):
        if (i % label_every) != 0:
            continue
        v = float(period_field[r, c])
        if not (np.isfinite(v) and 0.0 <= v <= 20.0):
            continue
        if _put(c, r - 17, f"{v:.1f}s"):
            n_arrow += 1

    # Pass 2 — extras where period changes, no arrow required.
    # Local change = max abs diff vs 4 neighbors 12 px away.
    if np.isfinite(period_field).sum() > 0:
        grad = np.full(period_field.shape, np.nan)
        ok = np.isfinite(period_field)
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            shifted = np.roll(period_field, shift=(-dr, -dc), axis=(0, 1))
            diff = np.abs(period_field - shifted)
            both = ok & np.isfinite(shifted)
            g = np.where(both, diff, 0.0)
            grad = np.where(np.isnan(grad), g, np.maximum(grad, g))
        for r in range(extra_step // 2, H, extra_step):
            for c in range(extra_step // 2, W, extra_step):
                if not np.isfinite(period_field[r, c]):
                    continue
                if rgba[r, c, 3] == 0:
                    continue
                if not (grad[r, c] >= extra_thresh_s):
                    continue
                # skip if an arrow label already covers this cell
                if _collides(c - 20, r - 26, c + 20, r - 8):
                    continue
                v = float(period_field[r, c])
                if _put(c, r - 10, f"{v:.1f}s"):
                    n_extra += 1

    del d
    return np.array(img), n_arrow, n_extra


# ------------------------------------------------------------------- legend
# Height table copied from build_wave_height.py, extended with one footer
# block explaining arrows + period labels (same 640 width).


def draw_combined_table(path, title, subtitle, source_line, note=None,
                        arrow_note=None):
    import os as _os
    from PIL import Image, ImageDraw
    from geospatial_utils import _legend_font
    f_title, f_body, f_small = _legend_font(22), _legend_font(15), _legend_font(13)
    pad, row_h, head_h = 14, 24, 30
    W = 640
    y_top = 64
    y_src = y_top + head_h + row_h * len(WAVE_FIXED_BINS)
    footer_lines = []
    if arrow_note:
        # crude wrap at ~78 chars for the 13px font
        words, line = arrow_note.split(), ""
        for w in words:
            if len(line) + 1 + len(w) > 78:
                footer_lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            footer_lines.append(line)
    y_src += 16 * len(footer_lines)
    H = y_src + (58 if note else 40) + 8
    img = Image.new("RGBA", (W, H), (255, 255, 255, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=2)
    d.text((pad, 8), title, font=f_title, fill=(10, 10, 10))
    d.text((pad, 36), subtitle, font=f_body, fill=(40, 40, 40))
    y = y_top
    d.rectangle([pad, y, W - pad, y + head_h], fill=(235, 235, 235, 255),
                outline=(40, 40, 40))
    d.text((pad + 6, y + 5), "Wave height", font=f_body, fill=(10, 10, 10))
    d.text((520, y + 5), "Color", font=f_body, fill=(10, 10, 10))
    y += head_h
    for i, (label, hx, _lo, _hi) in enumerate(WAVE_FIXED_BINS):
        if i % 2 == 1:
            d.rectangle([pad, y, W - pad, y + row_h],
                        fill=(245, 245, 245, 255))
        d.text((pad + 6, y + 3), label, font=f_body, fill=(10, 10, 10))
        d.rectangle([520, y + 2, W - pad - 2, y + row_h - 2],
                    fill=_hex_rgb(hx) + (255,), outline=(40, 40, 40))
        d.line([(pad, y), (W - pad, y)], fill=(200, 200, 200, 255))
        y += row_h
    d.rectangle([pad, y_top, W - pad, y], outline=(40, 40, 40))
    y += 6
    for ln in footer_lines:
        d.text((pad, y), ln, font=f_small, fill=(20, 20, 20))
        y += 16
    y += 2
    d.text((pad, y + 6), source_line, font=f_small, fill=(60, 60, 60))
    if note:
        d.text((pad, y + 24), note, font=f_small, fill=(60, 60, 60))
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    img.save(path)
    return W, H


def build_scale_html(run_max_ft, run_max_s, run_mean_dir):
    return (f"Base color = wave height, exact bin colors (see table). "
            f"Model height max this run: <b>{run_max_ft} ft</b>. "
            f"Arrows point where the waves travel (WVDIR + 180&deg;). "
            f"White halo numbers are the actual peak period at that spot "
            f"(seconds between crests, e.g. 4.2s), drawn in the same style "
            f"as the arrows and placed just above them; extra numbers mark "
            f"spots where the period changes sharply even between arrows. "
            f"Period max now <b>{run_max_s:.1f}s</b>, mean travel "
            f"<b>{run_mean_dir:.0f}&deg; ({compass(run_mean_dir)})</b>. "
            f"Significant height = average of highest third of waves.")


# ------------------------------------------------------------------- source

def candidate_urls(now):
    urls = []
    for d in (now, now - timedelta(days=1)):
        datestr = d.strftime("%Y%m%d")
        for cycle in CONFIG["cycles_try_order"]:
            urls.append((
                CONFIG["file_pattern"].format(
                    date_dir=f"glwu.{datestr}", cycle=cycle),
                datestr, cycle))
    return urls


def newest_available_cycle(now, var="HTSGW"):
    best = None
    for url, dd, cc in candidate_urls(now):
        try:
            req = urllib.request.Request(url + ".idx", headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                idx = r.read().decode("utf-8", errors="replace").splitlines()
            for line in idx:
                p = line.split(":")
                if len(p) > 5 and p[3] == var and "surface" in line \
                        and ":anl:" in line:
                    m = re.search(r"d=(\d{10})", line)
                    if m and (best is None or m.group(1) > best[3]):
                        best = (url, dd, cc, m.group(1))
                    break
        except Exception as e:
            print(f"[{PRODUCT}] idx probe {dd} t{cc}z: {str(e)[:100]}")
    return best


def extract_trio(grib_path):
    """Return dict HTSGW/PERPW/WVDIR step-0 (vals, lats, lons, date, time)."""
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    out = {}
    with open(grib_path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                sn = str(codes_get(h, "shortName")).lower()
                step = str(codes_get(h, "step"))
                if step != "0":
                    continue
                key = {"swh": "HTSGW", "perpw": "PERPW",
                       "wvdir": "WVDIR", "dirpw": "WVDIR",
                       "mwdir": "WVDIR"}.get(sn)
                if key is not None and key not in out:
                    vals = codes_get_values(h).astype(float)
                    lats = codes_get_array(h, "latitudes").astype(float)
                    lons = codes_get_array(h, "longitudes").astype(float)
                    lons = ((lons + 180) % 360) - 180
                    out[key] = (vals,
                                lats, lons,
                                str(codes_get(h, "dataDate")),
                                str(codes_get(h, "dataTime")).zfill(4))
            finally:
                codes_release(h)
    missing = [k for k in ("HTSGW", "PERPW", "WVDIR") if k not in out]
    if missing:
        raise ValueError(f"trio incomplete, missing {missing} step=0")
    return out


# ------------------------------------------------------------------- build

def main():
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--local-file", default=None,
                        help="use a cached GRIB2 instead of downloading "
                             "(e.g. output/raw/glwu_current.grib2)")
        ap.add_argument("--label-every", type=int, default=2,
                        help="label every Nth arrow (default 2)")
        ap.add_argument("--extra-step", type=int, default=32)
        ap.add_argument("--extra-thresh", type=float, default=0.5,
                        help="period-change threshold (s) for extra labels")
        ap.add_argument("--font-size", type=int, default=10,
                        help="period label font size px (default 10)")
        args = ap.parse_args()
        return run(local_file=args.local_file, label_every=args.label_every,
                   extra_step=args.extra_step, extra_thresh=args.extra_thresh,
                   font_size=args.font_size)
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run(local_file=None, label_every=2, extra_step=32, extra_thresh=0.5,
        font_size=10):
    now = datetime.now(timezone.utc)
    raw_path = os.path.join(RAW_DIR, "glwu_combined_current.grib2")
    used_url, datestr, cycle, stamp = None, None, None, None

    if local_file:
        if not os.path.exists(local_file):
            print(f"[{PRODUCT}] local file missing: {local_file}")
            return 2
        import shutil
        if os.path.abspath(local_file) != os.path.abspath(raw_path):
            shutil.copyfile(local_file, raw_path)
        used_url = f"local:{local_file}"
        datestr, cycle, stamp = "local", "local", "0000000000"
        source_id = source_token(f"local-{os.path.getsize(raw_path)}")
    else:
        pick = newest_available_cycle(now, "HTSGW")
        if pick is None:
            print(f"[{PRODUCT}] NO CYCLE AVAILABLE (keeping previous).")
            return 2
        used_url, datestr, cycle, stamp = pick
        source_id = f"glwu-{stamp[:8]}-{stamp[8:]}00Z-combined"
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
            got = download(used_url, raw_path, timeout=300)
        except Exception as e:
            print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {str(e)[:140]}")
            return 2
        if got["size_bytes"] < 100_000:
            print(f"[{PRODUCT}] candidate too small; keeping previous.")
            return 2

    try:
        return _build(raw_path, used_url, datestr, cycle, source_id,
                      label_every, extra_step, extra_thresh, font_size)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(raw_path, used_url, datestr, cycle, source_id,
           label_every, extra_step, extra_thresh, font_size=10):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    bounds = load_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]

    trio = extract_trio(raw_path)
    h_m, h_lats, h_lons, data_date, data_time = trio["HTSGW"]
    p_s, p_lats, p_lons, _, _ = trio["PERPW"]
    d_deg, d_lats, d_lons, _, _ = trio["WVDIR"]
    data_time_utc = grib_stamp_to_det(data_date, data_time)

    h_m = np.asarray(h_m, dtype=float).ravel()
    h_lats = np.asarray(h_lats, dtype=float).ravel()
    h_lons = np.asarray(h_lons, dtype=float).ravel()
    p_s = np.asarray(p_s, dtype=float).ravel()
    d_deg = np.asarray(d_deg, dtype=float).ravel()

    ok_h = np.isfinite(h_m) & (h_m >= 0) & (h_m < 9000)
    if int(ok_h.sum()) < 5_000:
        raise ValueError("too few valid height cells")
    rows, cols, valid = canvas_indices(h_lats, h_lons, bounds)
    h_ft = h_m * M_TO_FT
    hfield, _ = bin_to_canvas(rows, cols, h_ft, valid & ok_h, (H, W),
                              splat_radius=2)

    ok_p = (np.isfinite(p_s) & (p_s >= 0.0) & (p_s <= 20.0))
    rows_p, cols_p, valid_p = canvas_indices(p_lats, p_lons, bounds)
    pfield, _ = bin_to_canvas(rows_p, cols_p, p_s, valid_p & ok_p, (H, W),
                              splat_radius=2)
    rows_d, cols_d, valid_d = canvas_indices(d_lats, d_lons, bounds)
    dfield, _ = bin_to_canvas(rows_d, cols_d, d_deg,
                              valid_d & np.isfinite(d_deg)
                              & (d_deg >= 0) & (d_deg <= 360), (H, W),
                              splat_radius=0)

    okv = np.isfinite(hfield) & (hfield >= 0)
    if int(okv.sum()) < 5_000:
        raise ValueError(f"too few canvas height cells ({int(okv.sum())})")
    run_max_ft = round(float(hfield[okv].max()), 1)
    okpv = np.isfinite(pfield) & (pfield >= 0) & (pfield <= 20)
    run_max_s = float(pfield[okpv].max()) if okpv.any() else float("nan")
    dokv = np.isfinite(dfield) & (dfield >= 0) & (dfield <= 360)
    mean_deg = float(np.arctan2(np.sin(np.radians(dfield[dokv])).mean(),
                                np.cos(np.radians(dfield[dokv])).mean())
                     * 180.0 / math.pi % 360.0) if dokv.any() else 0.0

    # base: height colors verbatim, then shoreline mask
    rgba = render_wave_fixed(hfield, bounds["overlay_alpha"])
    rgba = apply_shoreline_mask(rgba)
    from geospatial_utils import load_watermask
    wm = load_watermask()
    water = wm > 0.5
    d_water = np.where(water & np.isfinite(dfield), dfield, np.nan)
    p_water = np.where(water & np.isfinite(pfield), pfield, np.nan)

    # arrows: existing style, verbatim
    rgba, n_arrows, arrow_pos = paint_arrows(rgba, d_water, step=16)
    # labels: actual period values, same arrow style, above arrows + extras
    rgba, n_lab, n_extra = paint_period_labels(
        rgba, p_water, arrow_pos, label_every=label_every,
        extra_step=extra_step, extra_thresh_s=extra_thresh,
        font_size=font_size)

    # arrows near shore can spill 1-2 px onto land: clip alpha back
    rgba[:, :, 3] = np.round(
        rgba[:, :, 3].astype(np.float32) * wm).astype(np.uint8)
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    print(f"[{PRODUCT}] opaque={n_opaque} arrows={n_arrows} "
          f"period_labels={n_lab}+{n_extra}extra "
          f"hmax={run_max_ft}ft pmax={run_max_s:.1f}s meandir={mean_deg:.0f}")
    if n_opaque < 5_000:
        raise ValueError("empty raster")
    if n_arrows < 50:
        raise ValueError(f"too few arrows ({n_arrows})")
    if n_lab + n_extra < 20:
        raise ValueError(f"too few period labels ({n_lab}+{n_extra})")

    subtitle = (f"Height color + period numbers + arrows  |  "
                f"max {run_max_ft} ft / {run_max_s:.1f}s")
    arrow_note = ("Arrows show travel direction (WVDIR + 180). "
                  "White halo numbers are the actual peak period in seconds "
                  "at that spot (e.g. 4.2s), placed just above arrows; extra "
                  "numbers mark sharp period changes even between arrows.")
    lw, lh = draw_combined_table(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        f"Source: NCEP GLWU v2.1 {datestr} t{cycle}z  |  Processed {now_det_str()}",
        note="Transparent outside valid water data.", arrow_note=arrow_note)
    scale_html = build_scale_html(run_max_ft, run_max_s, mean_deg)

    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], used_url, CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (NOMADS)",
        units="ft base (display); s labels (source s); degrees arrows",
        source_resolution="~2.5 km NCEP GLWU Lambert grid (581x361), single cycle",
        color_min=0.0, color_max=30.0, color_units="ft",
        missing_data_treatment=("only finite HTSGW admitted; off-water grid "
                                "points and land outside the NOAA shoreline "
                                "transparent; never zero-filled. Period/dir "
                                "sampled only on water."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{datestr} t{cycle}z"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(f"{source_id}-r{RENDER_VERSION}")
    meta["label_style"] = ("white fill (255,255,255,245) + dark halo "
                           "(20,20,20,235), same as direction arrows; "
                           f"every {label_every}th arrow + extras at "
                           f">= {extra_thresh}s change")
    meta["stats"] = {"valid_cells": int(okv.sum()),
                     "max_ft": run_max_ft,
                     "max_period_s": round(float(run_max_s), 2),
                     "mean_direction_deg": round(float(mean_deg), 1),
                     "mean_compass": compass(mean_deg),
                     "arrows_drawn": n_arrows,
                     "period_labels_on_arrows": n_lab,
                     "period_labels_extra": n_extra}
    token = meta["source_version"]
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>Wave height color + actual peak-period numbers + travel arrows, "
        f"all from one GLWU cycle.</p>"
        f"<p><img src=\"{pages_base()}/{PRODUCT}/legend.png?v={token}\" "
        f"width=\"600\" alt=\"key\"></p>"
        f"<p>{scale_html}</p>")
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
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "source_id": source_id,
                          "render_version": RENDER_VERSION,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def pages_base():
    from build_kml import pages_base as _pb
    return _pb()


if __name__ == "__main__":
    sys.exit(main())
