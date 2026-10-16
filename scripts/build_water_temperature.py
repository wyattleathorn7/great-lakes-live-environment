"""Pipeline B — LIVE WATER TEMPERATURE (independent).

NOAA GLSEA daily ASCII -> validate -> extract SST -> clip to Great Lakes
-> degC->degF -> temperature gradient -> transparent PNG -> metadata -> KML.

Exit codes: 0 = updated (or skipped, source unchanged); 2 = source/validation
failure (previous valid raster is left untouched); 1 = unexpected error.
"""

import hashlib
import json
import math
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR, base_metadata,
                               download, ensure_coords, fetch_buoy_obs,
                               head_last_modified,
                               http_date_to_det, now_det_str, promote_stage,
                               read_state,
                               source_token, stage_dir, utcnow_iso,
                               write_metadata, write_state)
from render_gradient import render_field
from na_bounds import load_global_bounds
from na_common import apply_coverage_mask
from erddap_coastwatch import BASE, BASE_CENTRAL, fetch_csv, latest_time
from build_kml import coverage_selector_html

PRODUCT = "water_temperature"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
GLSEA_URL = CONFIG["source_url"]
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")

# Coverage identities within the ONE preserved user-facing file
# (Great_Lakes_Live_Water_Temperature.kml). Basin stays the default ON
# layer, preserving the current default view. Dir suffixes reuse the
# shared COVERAGE_DIR_SUFFIX map (landonly -> continental).
WT_LABELS = {
    "basin": "Great Lakes Coverage",
    "landonly": "Continental Lake Coverage",
    "global": "Extended Oceanic Coverage",
}
WT_DIRS = {"landonly": "water_temperature_continental",
           "global": "water_temperature_global"}
OCEAN_DATASET = CONFIG.get("ocean_dataset_id", "noaacwBLENDEDsstDLDaily")
OCEAN_STRIDE = int(CONFIG.get("ocean_erddap_stride", 5))
MUR_DATASET = CONFIG.get("mur_dataset_id", "jplMURSST41")
MUR_STRIDE = int(CONFIG.get("mur_erddap_stride", 20))
MUR_LAKE_MASK = 5  # verified live: open_lake cells read 5 (seas 1, land 2)
# Basin canvas (preserved) vs worldwide canvas (new coverages share the
# authoritative global 3600x1800 grid from config/global_bounds.json).
WT_DEFAULT_COVERAGE = CONFIG.get("coverage_mode", "basin")
if WT_DEFAULT_COVERAGE not in ("basin",):
    WT_DEFAULT_COVERAGE = "basin"  # Basin default is part of the identity

BUOY_POS = {  # NDBC (lon, lat) — QC reference only
    "45001": (-87.793, 48.061),
    "45002": (-86.411, 45.344),
    "45132": (-81.220, 42.460),
    "45012": (-77.383, 43.619),
    "45005": (-82.398, 41.677),
}


def ff(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _temp_label_font(size=7):
    """DejaVu Sans Mono Bold (same face as the wave/wind value labels)."""
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


def _draw_temp_label(d, x, y, text, font):
    """White core + 1 px outline stroke (same as the wave/wind labels)."""
    from PIL import ImageDraw as _ID
    d.text((x, y), text, font=font, fill=(255, 255, 255, 255),
           stroke_width=1, stroke_fill=(20, 20, 20, 235))


def paint_temp_labels(rgba, field_f, font_size=7, extra_thresh_f=1.5,
                      max_labels=180, sparse_water=False, priority_cap=None):
    """Actual °F values in the wave/wind label style, free placement.

    NO fixed grid: data-seeded jittered candidates (positions move with the
    water, never a rigid lattice); thermal-front cells go first, then a
    spaced fill up to quota. Returns (rgba, n_priority, n_fill).

    priority_cap optionally reserves most of the quota for the scattered
    fill (fronts take at most cap labels), so open water far from any
    front still gets values instead of clustering only on gradients.
    sparse_water=True adapts the SAME font/size/style/gradient-first
    ordering to small scattered lakes: candidates are drawn from valid
    water cells themselves (a step-32 sweep almost never lands on a
    lake), and the glyph-corner water check is relaxed to the label
    anchor (a lake narrower than its own glyph can never host all three
    probe points, yet its temperature is still reportable). Anchor,
    collision, bounds, finite-value, and 20–95°F filters are unchanged.
    """
    from PIL import Image, ImageDraw
    H, W = field_f.shape
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    font = _temp_label_font(font_size)
    ok = np.isfinite(field_f)
    if not ok.any():
        del d
        return np.array(img), 0, 0
    _fin = field_f[ok]
    seed = int(abs(float(_fin.mean())) * 1000 + float(_fin.std()) * 97) \
        % (2 ** 32 - 1)
    rng = np.random.default_rng(seed)
    grad = np.zeros(field_f.shape)
    for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
        shifted = np.roll(field_f, shift=(-dr, -dc), axis=(0, 1))
        diff = np.abs(field_f - shifted)
        both = ok & np.isfinite(shifted)
        grad = np.maximum(grad, np.where(both, diff, 0.0))
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
    cand.sort(key=lambda t: (-t[0], t[1]))
    placed = []
    n_priority = n_fill = 0

    def _collides(x0, y0, x1, y1, pad=3):
        for (a0, b0, a1, b1) in placed:
            if not (x1 + pad < a0 or x0 - pad > a1 or y1 + pad < b0 or y0 - pad > b1):
                return True
        return False

    def _put(cx, cy, text, sep, need_corners=True, check_anchor=True,
             check_alpha=True):
        tw = d.textlength(text, font=font)
        th = font_size + 4
        x = cx - tw / 2
        y = cy - th / 2
        if not (4 <= x < W - tw - 4 and 4 <= y < H - th - 4):
            return False
        if _collides(x, y, x + tw, y + th):
            return False
        for (a0, b0, a1, b1) in placed:
            acx, acy = (a0 + a1) / 2, (b0 + b1) / 2
            if math.hypot(acx - cx, acy - cy) < sep:
                return False
        pts = [(cx, cy), (x + 2, y + 2), (x + tw - 2, y + 2)]
        if not need_corners:
            pts = [(cx, cy)]
        if check_alpha:
            for px, py in pts:
                ix, iy = int(np.clip(round(px), 0, W - 1)), int(np.clip(round(py), 0, H - 1))
                if rgba[iy, ix, 3] == 0:
                    return False
        if check_anchor and not np.isfinite(
                field_f[int(np.clip(round(cy), 0, H - 1)),
                        int(np.clip(round(cx), 0, W - 1))]):
            return False
        _draw_temp_label(d, x, y, text, font)
        placed.append((x, y, x + tw, y + th))
        return True

    for g, _, r, c in cand:
        if n_priority + n_fill >= max_labels:
            break
        if priority_cap is not None and n_priority >= priority_cap:
            break
        if g < extra_thresh_f:
            break
        v = float(field_f[r, c])
        if not (20.0 <= v <= 95.0):
            continue
        if _put(c, r - 12, f"{v:.0f}\u00b0F", sep=30):
            n_priority += 1
    rest = [t for t in cand if t[0] < extra_thresh_f]
    rng.shuffle(rest)
    for _, _, r, c in rest:
        if n_priority + n_fill >= max_labels:
            break
        v = float(field_f[r, c])
        if not (20.0 <= v <= 95.0):
            continue
        if _put(c, r - 12, f"{v:.0f}\u00b0F", sep=48):
            n_fill += 1

    if sparse_water and n_priority + n_fill < max_labels:
        # Small scattered lakes: one candidate per ~36 px bucket (the
        # highest-gradient valid cell), gradient-first across buckets, so
        # attempts spread over every lake instead of piling onto the
        # biggest one. Anchor-only water check (see docstring).
        rr, cc = np.nonzero(ok)
        _bkt = (rr // 36) * (W // 36 + 1) + (cc // 36)
        _best = {}
        for i in range(rr.size):
            b = int(_bkt[i])
            g = float(grad[rr[i], cc[i]]) if np.isfinite(
                grad[rr[i], cc[i]]) else -1.0
            if b not in _best or g > _best[b][0]:
                _best[b] = (g, i)
        tie = rng.random(len(_best))
        _keys = list(_best.keys())
        _order = sorted(range(len(_keys)),
                        key=lambda k: (-_best[_keys[k]][0], tie[k]))
        for k in _order:
            if n_priority + n_fill >= max_labels:
                break
            i = _best[_keys[k]][1]
            r, c = int(rr[i]), int(cc[i])
            v = float(field_f[r, c])
            if not (20.0 <= v <= 95.0):
                continue
            if _put(c, r - 12, f"{v:.0f}\u00b0F", sep=40,
                    need_corners=False, check_anchor=False,
                    check_alpha=False):
                n_fill += 1

    del d
    return np.array(img), n_priority, n_fill


def _endpoints_f(f_vals, lo_clamp=32.0, hi_clamp=86.0):
    """Existing dynamic-range rule, shared by all coverages.

    1st/99th percentiles of the coverage's own valid °F population,
    floored/ceiled to whole degrees, clamped to [lo_clamp, hi_clamp]°F
    (either clamp may be None to leave that end unclamped), with the
    32/75 fallback when the span collapses. Linear temperature
    normalization between these endpoints (NOT percentile ranks).

    Basin + Continental keep (32, 86). Extended Oceanic leaves the low
    end unclamped (None, 86): polar water (~28–32°F, valid in-source)
    would otherwise hide below the legend floor.
    """
    p1, p99 = float(np.percentile(f_vals, 1)), float(np.percentile(f_vals, 99))
    vmin = math.floor(p1) if lo_clamp is None else max(lo_clamp,
                                                       math.floor(p1))
    vmax = math.ceil(p99) if hi_clamp is None else min(hi_clamp,
                                                       math.ceil(p99))
    if vmax < vmin + 2:
        vmin, vmax = 32.0, 75.0
    return vmin, vmax


def _scale_html(vmin, vmax, mean_f, coverage_label, scope_note):
    """Legend sentence: same structure as the long-standing basin text."""
    mid_f = round((vmin + vmax) / 2, 1)
    return (f"Surface water temperature (°F) — {coverage_label}: cold "
            f"<b>{vmin:g}°F</b> (deep blue) → <b>{mid_f:g}°F</b> → warm "
            f"<b>{vmax:g}°F</b> (red). Mean this run: "
            f"<b>{round(float(mean_f), 1)}°F</b>. White halo "
            f"numbers are actual °F at that spot. {scope_note}")


BASIN_SCOPE = ("The bar ends are recomputed every run from that day's "
               "Great Lakes water (1st/99th percentiles, clamped 32–86°F): "
               "blue always means the coldest water present and red the "
               "warmest, but the same color can stand for a different °F "
               "on different days — only the printed numbers are absolute.")


def _overlay_legend_html(prod_dir_name, token, scale_html, status_label,
                         source_name, source_url, coverage_label,
                         timing_line):
    """Self-contained per-coverage description: own versioned legend image
    + own endpoints + source/timing + gradient/label semantics. Never
    references another coverage's assets, so hiding/unhiding layers or
    refreshing one coverage cannot confuse it with another."""
    from build_kml import legend_block
    block = legend_block(f"{prod_dir_name}/legend.png", token, scale_html)
    return (
        f"{block}"
        f"<p><b>Coverage:</b> {coverage_label}</p>"
        f"<p><b>Status:</b> {status_label}</p>"
        f"<p><b>Source:</b> {source_name}<br/>"
        f"<a href=\"{source_url}\">{source_url}</a><br/>{timing_line}</p>"
        f"<p>Colors run cold (deep blue) → warm (red) across this "
        f"coverage's current water-temperature distribution; white halo "
        f"numbers are actual water temperatures in °F at those spots. "
        f"Missing or unavailable observations stay transparent and are "
        f"never fabricated.</p>"
    )


def _discover_blended_time():
    """Newest available blended-SST analysis time (ERDDAP time axis).

    Raises on failure (caller maps to exit 2: previous raster kept).
    """
    return latest_time(OCEAN_DATASET, base=BASE_CENTRAL)


def _discover_mur_time():
    """Newest available MUR analysis time (ERDDAP time axis, PFEG node).

    Raises on failure (caller maps to exit 2: previous raster kept).
    """
    return latest_time(MUR_DATASET, base=BASE)


def _fetch_blended_sst(time_iso):
    """Global blended SST at configured stride. Land cells arrive as
    NaN (verified live: only true water carries values). Returns
    (lats_1d_asc, lons_1d, sst_c_2d_southfirst). Raises."""
    la, lo, sst = fetch_csv(OCEAN_DATASET, "analysed_sst", time_iso,
                            -89.975, 89.975, -179.975, 179.975,
                            stride=OCEAN_STRIDE, altitude=False,
                            base=BASE_CENTRAL)
    return la, lo, sst


def _fetch_mur(time_iso):
    """Global MUR SST + lake mask at configured stride. Returns
    (lats_1d_asc, lons_1d, sst_c_2d_southfirst, mask_2d). Raises."""
    la, lo, sst = fetch_csv(MUR_DATASET, "analysed_sst", time_iso,
                            -89.99, 89.99, -179.99, 179.99,
                            stride=MUR_STRIDE, altitude=False, base=BASE)
    _, _, msk = fetch_csv(MUR_DATASET, "mask", time_iso,
                          -89.99, 89.99, -179.99, 179.99,
                          stride=MUR_STRIDE, altitude=False, base=BASE)
    return la, lo, sst, msk


def _render_world_variant(mode, comp_f, gbounds, stage_prod_dir, token,
                          scale_html, status_label, source_name, source_url,
                          timing_line, title, font_size=7, extra_thresh=1.5,
                          max_labels=600, sparse_water=False,
                          lo_clamp=32.0, hi_clamp=86.0, legend_title=None,
                          priority_cap=None):
    """Colormap + labels + legend PNG + metadata for one world coverage.

    Same TEMP_STOPS LUT and linear (field-vmin)/(vmax-vmin) mapping as
    the basin path (geospatial_utils.apply_colormap); endpoints come
    from _endpoints_f on this coverage's own valid °F population, with
    the caller-supplied clamps (global leaves the low end unclamped).
    The legend is drawn from the same vmin/vmax that colors the raster,
    titled for its own coverage (never the basin title elsewhere).
    Continental gets a hard binary shoreline cut (crisp lake cutouts,
    no feathered fringe); labels paint after the cut.
    """
    from geospatial_utils import (TEMP_STOPS, apply_colormap,
                                  bleed_rgb_into_transparent, draw_legend,
                                  save_png, write_metadata as _wm,
                                  base_metadata as _bm, load_bounds as _lb)
    from na_common import load_continental
    H, W = comp_f.shape
    ok = np.isfinite(comp_f)
    f_vals = comp_f[ok]
    vmin, vmax = _endpoints_f(f_vals, lo_clamp, hi_clamp)
    print(f"[{PRODUCT}] variant {WT_LABELS[mode]}: n={f_vals.size} "
          f"F=[{f_vals.min():.2f},{f_vals.max():.2f}] scale=[{vmin},{vmax}]")
    rgba = apply_colormap(comp_f, vmin, vmax, TEMP_STOPS,
                          _lb()["overlay_alpha"], None)
    if mode == "landonly":
        # Crisp lake cutouts: the continental mask is strictly binary
        # water/land at canvas resolution, so cut hard (no feathered
        # fringe) before painting labels.
        _lakemask = load_continental()
        rgba[:, :, 3] = np.where(
            _lakemask >= 0.5, rgba[:, :, 3], 0).astype(np.uint8)
        rgba = bleed_rgb_into_transparent(rgba.copy())
    else:
        rgba = apply_coverage_mask(rgba, "global")
    rgba, n_pri, n_fill = paint_temp_labels(
        rgba, comp_f, font_size=font_size, extra_thresh_f=extra_thresh,
        max_labels=max_labels, sparse_water=sparse_water,
        priority_cap=priority_cap)
    print(f"[{PRODUCT}] variant {WT_LABELS[mode]}: "
          f"temp_labels={n_pri}+{n_fill}")
    os.makedirs(stage_prod_dir, exist_ok=True)
    save_png(rgba, os.path.join(stage_prod_dir, "current.png"))
    subtitle = (f"{timing_line}  |  ends = this coverage's "
                f"coldest/warmest water")
    lw, lh = draw_legend(os.path.join(stage_prod_dir, "legend.png"),
                         legend_title or title,
                         subtitle, "\u00b0F", vmin, vmax, TEMP_STOPS,
                         f"Source: {source_name}  |  "
                         f"Processed {now_det_str()}",
                         fmt="{:.0f}",
                         tick_labels=[(vmin, f"{vmin:g}"),
                                      (vmax, f"{vmax:g}")],
                         transparent_note=("Transparent where no valid "
                                           "water observation."))
    meta = _bm(WT_DIRS[mode], legend_title or f"{title} — {WT_LABELS[mode]}",
               status_label, source_name, source_url,
               ("analysed/fused lake surface + sea surface water "
                "temperature (satellite-derived composite/analysis)"),
               data_time_utc=timing_line,
               source_last_modified_utc=timing_line,
               units="degF (display); source degC",
               source_resolution=("blended 0.05 deg analysis resampled to "
                                  "0.10 deg global canvas (+ GLSEA ~1.8 km "
                                  "inside the Great Lakes rectangle)"),
               color_min=vmin, color_max=vmax, color_units="degF",
               missing_data_treatment=(
                   "non-water mask cells, fill values, out-of-range values "
                   "and Great Lakes-rectangle blended cells (GLSEA wins "
                   "there) rendered fully transparent; never interpolated, "
                   "never zero-filled."))
    meta["source_id"] = f"blended-{timing_line}"
    meta["source_version"] = token
    meta["na_bounds"] = dict(gbounds)
    meta["coverage_mode"] = mode
    meta["legend_scale_html"] = scale_html
    meta["legend_size"] = [lw, lh]
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    meta["rendered_nontransparent_pixels"] = n_opaque
    meta["rendered_canvas_pixels"] = int(H * W)
    meta["stats"] = {
        "valid_cells": int(f_vals.size),
        "mean_F": round(float(np.mean(f_vals)), 2),
        "min_F": round(float(np.min(f_vals)), 2),
        "max_F": round(float(np.max(f_vals)), 2),
        "temp_labels_priority": n_pri,
        "temp_labels_fill": n_fill,
    }
    _wm(stage_prod_dir, meta)
    return meta, vmin, vmax, n_opaque


def main():
    try:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--preview-dir", default=None,
                        help="preview: write current.png here, skip promote")
        ap.add_argument("--font-size", type=int, default=7)
        ap.add_argument("--extra-thresh", type=float, default=1.5,
                        help="°F-change threshold for priority labels")
        ap.add_argument("--max-labels", type=int, default=180)
        args = ap.parse_args()
        return run(preview_dir=args.preview_dir, font_size=args.font_size,
                   extra_thresh=args.extra_thresh, max_labels=args.max_labels)
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def _published_files_present():
    need = [os.path.join(SITE_DIR, PRODUCT, "current.png"),
            os.path.join(SITE_DIR, PRODUCT, "legend.png"),
            os.path.join(SITE_DIR, PRODUCT, "metadata.json"),
            os.path.join(SITE_DIR, "water_temperature_continental",
                         "current.png"),
            os.path.join(SITE_DIR, "water_temperature_continental",
                         "legend.png"),
            os.path.join(SITE_DIR, "water_temperature_continental",
                         "metadata.json"),
            os.path.join(SITE_DIR, "water_temperature_global", "current.png"),
            os.path.join(SITE_DIR, "water_temperature_global", "legend.png"),
            os.path.join(SITE_DIR, "water_temperature_global",
                         "metadata.json"),
            os.path.join(SITE_DIR, "kml", "live",
                         "Great_Lakes_Live_Water_Temperature.kml")]
    return all(os.path.exists(p) for p in need)


def run(preview_dir=None, font_size=7, extra_thresh=1.5, max_labels=180):
    raw_path = os.path.join(RAW_DIR, "glsea_cur.asc")
    # Cheap hourly gate: HEAD the Last-Modified stamp before the ~8 MB
    # download. Match with the committed state -> source unchanged, KMLs
    # only. Anything else falls through to download + content-hash compare.
    prev = read_state(PRODUCT)
    lm = head_last_modified(GLSEA_URL)
    try:
        blended_time_cached = _discover_blended_time()
    except Exception as e:
        print(f"[{PRODUCT}] blended time probe failed: {e}")
        blended_time_cached = None
    try:
        mur_time_cached = _discover_mur_time()
    except Exception as e:
        print(f"[{PRODUCT}] MUR time probe failed: {e}")
        mur_time_cached = None
    if lm and lm == prev.get("source_last_modified") \
            and blended_time_cached \
            and blended_time_cached == prev.get("blended_time") \
            and mur_time_cached \
            and mur_time_cached == prev.get("mur_time") \
            and prev.get("render_version") == RENDER_VERSION \
            and _published_files_present():
        print(f"[{PRODUCT}] HEAD unchanged ({lm}) + blended unchanged "
              f"({blended_time_cached}) + MUR unchanged ({mur_time_cached}); "
              f"keeping rasters.")
        refresh_kml_base_url(PRODUCT, "Great_Lakes_Live_Water_Temperature.kml",
                             "\U0001F321\uFE0F LIVE WATER TEMPERATURE",
                             CONFIG["title"],
                             "Turn on/off independently of wave and ice layers.",
                             CONFIG["refresh_interval_seconds"])
        return 0
    try:
        info = download(GLSEA_URL, raw_path)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    if info["size_bytes"] < 1_000_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: file too small "
              f"({info['size_bytes']} bytes). Keeping previous.")
        return 2

    # Source-aware gate: GLSEA publishes one file per day (HTTP
    # Last-Modified is the observation id; content hash is the fallback so
    # a missing header can never freeze updates via a None == None skip).
    # Same source -> keep the published raster, refresh KMLs only.
    with open(raw_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    raw_id = info["http_last_modified"] or f"sha256:{digest[:16]}"
    source_id = f"glsea-{raw_id}"
    prev = read_state(PRODUCT)
    if blended_time_cached is None:
        try:
            blended_time_cached = _discover_blended_time()
        except Exception as e:
            print(f"[{PRODUCT}] blended time probe failed: {e}")
            blended_time_cached = None
    if mur_time_cached is None:
        try:
            mur_time_cached = _discover_mur_time()
        except Exception as e:
            print(f"[{PRODUCT}] MUR time probe failed: {e}")
            mur_time_cached = None
    if prev.get("source_id") == source_id \
            and blended_time_cached \
            and prev.get("blended_time") == blended_time_cached \
            and mur_time_cached \
            and prev.get("mur_time") == mur_time_cached \
            and prev.get("render_version") == RENDER_VERSION \
            and _published_files_present():
        print(f"[{PRODUCT}] source unchanged ({source_id}) + blended "
              f"unchanged ({blended_time_cached}) + MUR unchanged "
              f"({mur_time_cached}); keeping rasters.")
        refresh_kml_base_url(PRODUCT, "Great_Lakes_Live_Water_Temperature.kml",
                             "\U0001F321\uFE0F LIVE WATER TEMPERATURE",
                             CONFIG["title"],
                             "Turn on/off independently of wave and ice layers.",
                             CONFIG["refresh_interval_seconds"])
        return 0

    # Everything below renders into a stage dir first; the live site/ + kml/
    # tree is touched only by promote_stage() on full success, so a crash
    # can never publish a half-updated product set. Any data-dependent
    # failure returns 2 (keep previous); only unexpected engine errors
    # escape to exit 1 via main().
    try:
        return _build(info, raw_path, source_id, preview_dir, font_size,
                      extra_thresh, max_labels,
                      blended_time=blended_time_cached,
                      mur_time=mur_time_cached)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(info, raw_path, source_id, preview_dir=None, font_size=7,
           extra_thresh=1.5, max_labels=180, blended_time=None,
           mur_time=None):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    with open(raw_path) as f:
        header = [f.readline() for _ in range(6)]
    try:
        ncols = int(float(header[0].split()[-1]))
        nrows = int(float(header[1].split()[-1]))
    except (ValueError, IndexError):
        print(f"[{PRODUCT}] VALIDATION FAILED: bad header {header}.")
        return 2
    if (ncols, nrows) != (1024, 1024):
        print(f"[{PRODUCT}] VALIDATION FAILED: dims {ncols}x{nrows} != 1024x1024.")
        return 2
    data = np.loadtxt(raw_path, skiprows=6)
    if data.shape != (1024, 1024):
        print(f"[{PRODUCT}] VALIDATION FAILED: shape {data.shape}.")
        return 2

    cdir = ensure_coords(RAW_DIR)
    lats = np.loadtxt(os.path.join(cdir, "1024_latgrid.txt"))
    lons = np.loadtxt(os.path.join(cdir, "1024_longrid.txt"))
    lake_ids = np.loadtxt(os.path.join(cdir, "1024_lake_ids.txt"))
    mask_water = (lake_ids >= 1) & (lake_ids <= 6)

    water = (data != CONFIG["land_code"]) & np.isfinite(data) \
        & (data > -3) & (data < 45)
    n_valid = int(water.sum())
    n_mask = int(mask_water.sum())
    if n_valid == 0:
        print(f"[{PRODUCT}] VALIDATION FAILED: no valid SST cells. "
              f"Keeping previous.")
        return 2
    print(f"[{PRODUCT}] valid SST cells={n_valid} mask water cells={n_mask} "
          f"range_C=[{data[water].min():.2f},{data[water].max():.2f}]")
    if n_valid < 80_000 or n_valid < 0.5 * n_mask or n_valid > 1.6 * n_mask:
        print(f"[{PRODUCT}] VALIDATION FAILED: implausible coverage. "
              f"Keeping previous.")
        return 2

    c = data[water]
    f_vals = c * 9.0 / 5.0 + 32.0
    vmin, vmax = _endpoints_f(f_vals)
    print(f"[{PRODUCT}] color scale F=[{vmin},{vmax}]")

    values_f = np.full(data.shape, np.nan)
    values_f[water] = data[water] * 9.0 / 5.0 + 32.0

    data_time_iso = http_date_to_det(info["http_last_modified"])
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], GLSEA_URL, CONFIG["variable"],
        data_time_utc=data_time_iso or "unknown (no source timestamp)",
        source_last_modified_utc=(http_date_to_det(info["http_last_modified"])
        or "unknown"),
        units="degF (display); source degC",
        source_resolution="~1.8 km GLSEA grid (1024x1024)",
        color_min=vmin, color_max=vmax, color_units="degF",
        missing_data_treatment=(f"land code {CONFIG['land_code']} and -9999/no-data "
                                "rendered fully transparent; never interpolated."))
    meta["source_id"] = source_id
    # Style rides the version token (same-cycle restyles change the ?v=
    # URL so caches cannot serve old pixels). TEXT_REV bumps on any
    # label change.
    meta["source_version"] = source_token(f"{source_id}-r{RENDER_VERSION}-tempvals-tx2")
    meta["label_method"] = (
        "Actual °F values (e.g. 45°F) in white mono with a dark outline, "
        "free placement: data-seeded jittered candidates (positions move "
        "with the water, never a fixed grid), thermal-front cells first, "
        "spaced fill up to quota.")
    meta["stats"] = {
        "valid_cells": n_valid,
        "lakewide_mean_F": round(float(np.mean(f_vals)), 2),
        "min_F": round(float(np.min(f_vals)), 2),
        "max_F": round(float(np.max(f_vals)), 2),
    }

    field, rgba, meta = render_field(
        PRODUCT, lats, lons, values_f, vmin, vmax, meta,
        title=CONFIG["title"],
        subtitle=(f"{data_time_iso or 'see metadata'}  |  ends = "
                 f"today's coldest/warmest water"),
        source_line=(f"Source: NOAA/GLERL CoastWatch GLSEA  |  "
                     f"Processed {now_det_str()}"),
        unit_label="\u00b0F", transparent_value=None, fmt="{:.0f}",
        splat_radius=1, product_dir=stage_prod,
        tick_labels=[(vmin, f"{vmin:g}"), (vmax, f"{vmax:g}")])

    if int((rgba[:, :, 3] > 0).sum()) < 10_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: raster has no water pixels.")
        return 2

    # Actual °F values in the wave/wind label style (free placement).
    rgba, n_tpri, n_tfill = paint_temp_labels(
        rgba, field, font_size=font_size, extra_thresh_f=extra_thresh,
        max_labels=max_labels)
    print(f"[{PRODUCT}] temp_labels={n_tpri}+{n_tfill}extra")
    if n_tpri + n_tfill < 20:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few temp labels.")
        return 2
    # render_field saved the BASE raster before painting above: persist the
    # labeled raster (without this the values never reach the file).
    # Glyphs near shore can spill 1-2 px onto land: clip alpha back to
    # the shared shoreline (same as the wave layer).
    from geospatial_utils import save_png as _sp2, load_watermask as _lwm
    _wm = _lwm()
    rgba[:, :, 3] = np.round(
        rgba[:, :, 3].astype(np.float32) * _wm).astype(np.uint8)
    _sp2(rgba, os.path.join(stage_prod, "current.png"))
    if preview_dir is not None:
        import os as _os
        from geospatial_utils import save_png as _sp
        _os.makedirs(preview_dir, exist_ok=True)
        _sp(rgba, os.path.join(preview_dir, "current.png"))
        print(f"[{PRODUCT}] PREVIEW written.")
        return 0

    # ---- world coverages (new; basin path above is byte-identical logic)
    # Precedence, one owner per cell (never double-counted):
    #   GLSEA (Great Lakes rectangle, where valid)
    #     > MUR open-lake cells (mask == 5, inland water worldwide)
    #     > blended SST (all remaining valid water: oceans + seas).
    # Blended SST alone was evaluated and covers enclosed seas but marks
    # true lakes as land (verified: Caspian/Black/Baltic water; Victoria,
    # Baikal, Tanganyika, Malawi, Great Bear/Slave, Ladoga all land), so
    # it is NOT used for lakes. MUR was verified lake-valid (Victoria,
    # Baikal, Superior, Caspian resolve). Copernicus CLMS LSWT was
    # investigated as a future inland source (see config lake_source_note)
    # and is NOT integrated: no anonymous-CI access path, 10-day
    # synthesis latency, unverified lake inventory.
    if not blended_time:
        try:
            blended_time = _discover_blended_time()
        except Exception as e:
            print(f"[{PRODUCT}] VALIDATION FAILED: blended SST time "
                  f"unavailable ({e}). Keeping previous.")
            return 2
    if not mur_time:
        try:
            mur_time = _discover_mur_time()
        except Exception as e:
            print(f"[{PRODUCT}] VALIDATION FAILED: MUR time unavailable "
                  f"({e}). Keeping previous.")
            return 2
    try:
        _bla, _blo, _bsst = _fetch_blended_sst(blended_time)
    except Exception as e:
        print(f"[{PRODUCT}] VALIDATION FAILED: blended SST fetch failed "
              f"({type(e).__name__}: {e}). Keeping previous.")
        return 2
    try:
        _mla, _mlo, _msst, _mmsk = _fetch_mur(mur_time)
    except Exception as e:
        print(f"[{PRODUCT}] VALIDATION FAILED: MUR fetch failed "
              f"({type(e).__name__}: {e}). Keeping previous.")
        return 2
    from geospatial_utils import (bin_to_canvas, canvas_indices,
                                  resample_gridded)
    gbounds = load_global_bounds()
    GH, GW = gbounds["canvas_height"], gbounds["canvas_width"]

    def _to_canvas(la1, lo1, grid):
        _lo2, _la2 = np.meshgrid(np.asarray(lo1, dtype=float),
                                 np.asarray(la1, dtype=float))
        return resample_gridded(np.asarray(grid, dtype=float),
                                _la2, _lo2, gbounds, (GH, GW))

    try:
        _bfield = _to_canvas(_bla, _blo, _bsst)
        _mfield = _to_canvas(_mla, _mlo, _msst)
        _kfield = _to_canvas(_mla, _mlo, _mmsk)
    except Exception as e:
        print(f"[{PRODUCT}] VALIDATION FAILED: world resample failed "
              f"({type(e).__name__}: {e}). Keeping previous.")
        return 2
    _bvalid = (np.isfinite(_bfield) & (np.abs(_bfield + 327.68) > 1.0)
               & (_bfield >= -2.0) & (_bfield <= 40.0))
    _mvalid = (np.isfinite(_mfield) & (np.abs(_mfield + 7.768) > 0.01)
               & (_mfield >= -3.0) & (_mfield <= 40.0))
    _lakebit = (np.isfinite(_kfield)
                & (np.round(_kfield).astype(np.int32) == MUR_LAKE_MASK))
    print(f"[{PRODUCT}] blended valid water cells={int(_bvalid.sum())}, "
          f"MUR lake cells={int((_mvalid & _lakebit).sum())} of {GH*GW}")
    _bf = np.full((GH, GW), np.nan)
    _bf[_bvalid] = _bfield[_bvalid] * 9.0 / 5.0 + 32.0
    _mf = np.full((GH, GW), np.nan)
    _mf[_mvalid & _lakebit] = _mfield[_mvalid & _lakebit] * 9.0 / 5.0 + 32.0
    del _bsst, _msst, _mmsk, _bfield, _mfield, _kfield
    # GLSEA wins everywhere valid (mean-binned onto the global canvas).
    _grows, _gcols, _gpos = canvas_indices(lats, lons, gbounds)
    _glmean, _glcnt = bin_to_canvas(_grows, _gcols,
                                    np.asarray(values_f, dtype=float),
                                    _gpos & np.isfinite(
                                        np.asarray(values_f).ravel()),
                                    (GH, GW), splat_radius=0)
    _glok = (_glcnt > 0) & np.isfinite(_glmean)
    _comp = _bf
    _lakeonly = _mf
    _comp[np.isfinite(_mf)] = _mf[np.isfinite(_mf)]
    _comp[_glok] = _glmean[_glok]
    _lakeonly[_glok] = _glmean[_glok]
    print(f"[{PRODUCT}] GLSEA override cells on world canvas="
          f"{int(_glok.sum())}")
    _n_ocn = int(np.isfinite(_comp).sum())
    _n_lak = int(np.isfinite(_lakeonly).sum())
    print(f"[{PRODUCT}] world populations: ocean={_n_ocn} lakes={_n_lak}")
    if _n_ocn < 50_000 or _n_lak < 500:
        print(f"[{PRODUCT}] VALIDATION FAILED: implausible world coverage "
              f"(ocean={_n_ocn}, lakes={_n_lak}). Keeping previous.")
        return 2
    from geospatial_utils import iso_to_det
    _blend_det = iso_to_det(blended_time) or blended_time
    _mur_det = iso_to_det(mur_time) or mur_time
    blended_id = f"blended-{blended_time}"
    mur_id = f"mur-{mur_time}"
    _wstatus = ("LATEST AVAILABLE (daily analyses; blended typically "
                "~3-day latency, MUR typically ~2-day latency)")
    _wsource = (f"{CONFIG.get('mur_source_name', 'MUR SST')} (lakes) + "
                f"{CONFIG.get('ocean_source_name', 'blended SST')} "
                f"(oceans/seas) + {CONFIG['source_name']} (Great Lakes)")
    _wurl = CONFIG.get("mur_source_url", CONFIG.get(
        "ocean_source_url", GLSEA_URL))
    _ocean_token = source_token(
        f"{blended_id}-{mur_id}-r{RENDER_VERSION}-tempvals-global-nofloor")
    _lake_token = source_token(
        f"{mur_id}-r{RENDER_VERSION}-tempvals-continental")
    _con_scope = ("Endpoints are recomputed every run from this coverage's "
                  "own tracked inland-lake water only (MUR open-lake cells "
                  "plus Great Lakes GLSEA cells; 1st/99th percentiles, "
                  "clamped 32–86°F, linear mapping between them). Ocean "
                  "water is excluded from this distribution, so the same "
                  "color stands for a different °F than in the other "
                  "coverages — only the printed numbers are absolute. "
                  "Lakes below the fetch sampling, frozen lakes outside "
                  "the open-lake mask state, and rivers stay transparent "
                  "and are never fabricated.")
    _glo_scope = ("Endpoints are recomputed every run from all valid water "
                  "across this coverage's supported global dataset (oceans "
                  "and seas from blended SST, inland lakes from MUR, Great "
                  "Lakes from GLSEA; each observation counted once; "
                  "1st/99th percentiles with floor/ceil rounding; low end "
                  "unclamped so polar water (~28°F) reads correctly, high "
                  "end clamped 86°F; linear mapping between them). The same "
                  "color stands for a different °F than in the other "
                  "coverages — only the printed numbers are absolute. "
                  "Unsupported or unobserved water stays transparent and is "
                  "never fabricated.")
    _con_vals = _lakeonly[np.isfinite(_lakeonly)]
    _glo_vals = _comp[np.isfinite(_comp)]
    _con_scale = _scale_html(*_endpoints_f(_con_vals, 32.0, 86.0),
                             float(np.mean(_con_vals)),
                             "Continental Lake Coverage", _con_scope)
    _glo_scale = _scale_html(*_endpoints_f(_glo_vals, None, 86.0),
                             float(np.mean(_glo_vals)),
                             "Extended Oceanic Coverage", _glo_scope)
    _con_meta, _con_vmin, _con_vmax, _con_px = _render_world_variant(
        "landonly", _lakeonly, gbounds,
        os.path.join(stage, "site", WT_DIRS["landonly"]), _lake_token,
        _con_scale, _wstatus, _wsource, _wurl,
        f"MUR {_mur_det}; blended {_blend_det}", CONFIG["title"],
        font_size=font_size, extra_thresh=extra_thresh, max_labels=300,
        sparse_water=True,
        legend_title="CONTINENTAL LAKE WATER TEMPERATURE")
    _glo_meta, _glo_vmin, _glo_vmax, _glo_px = _render_world_variant(
        "global", _comp, gbounds,
        os.path.join(stage, "site", WT_DIRS["global"]), _ocean_token,
        _glo_scale, _wstatus, _wsource, _wurl,
        f"MUR {_mur_det}; blended {_blend_det}", CONFIG["title"],
        font_size=font_size, extra_thresh=extra_thresh, max_labels=1000,
        lo_clamp=None, hi_clamp=86.0, priority_cap=300,
        legend_title="EXTENDED OCEAN WATER TEMPERATURE")
    if min(_con_px, _glo_px) < 5_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: world raster has too few "
              f"water pixels (continental={_con_px}, global={_glo_px}).")
        return 2

    # ---- buoy QC (reference only; never alters the grid) ----
    qc = fetch_buoy_obs(CONFIG["buoys"])
    H, W = rgba.shape[:2]
    b = json.load(open(os.path.join(REPO_ROOT, "config", "great_lakes_bounds.json")))
    for bid, pos in BUOY_POS.items():
        obs = qc.get(bid, {})
        wt = ff(obs.get("WTMP_C"))
        if wt is None:
            print(f"[{PRODUCT}] buoy {bid}: no WTMP obs (skipped)")
            meta.setdefault("buoy_qc", {})[bid] = {**obs, "note": "no WTMP obs"}
            continue
        wt_f = wt * 9.0 / 5.0 + 32.0
        col = int((pos[0] - b["lon_min"]) / (b["lon_max"] - b["lon_min"]) * W)
        row = int((b["lat_max"] - pos[1]) / (b["lat_max"] - b["lat_min"]) * H)
        cell = None
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = row + dr, col + dc
                if 0 <= rr < H and 0 <= cc < W and np.isfinite(field[rr, cc]):
                    cell = float(field[rr, cc])
                    break
            if cell is not None:
                break
        diff = None if cell is None else round(abs(cell - wt_f), 2)
        flag = ("OK" if (diff is not None and diff <= CONFIG["buoy_qc_tolerance_f"])
                else "CHECK" if diff is not None else "NO_GRID_CELL")
        print(f"[{PRODUCT}] buoy {bid}: obs {wt_f:.1f}F grid {cell} diff {diff} -> {flag}")
        meta.setdefault("buoy_qc", {})[bid] = {
            "obs_F": round(wt_f, 2), "grid_F": cell, "absdiff_F": diff,
            "verdict": flag, "obs_time": obs.get("time_utc")}

    np.savez_compressed(os.path.join(RAW_DIR, f"{PRODUCT}_field.npz"),
                        lats=lats, lons=lons, values=values_f)
    mid_f = round((vmin + vmax) / 2, 1)
    meta["stats"]["temp_labels_priority"] = n_tpri
    meta["stats"]["temp_labels_fill"] = n_tfill
    scale_html = (f"Surface water temperature (°F): cold <b>{vmin:g}°F</b> "
                  f"(deep blue) → <b>{mid_f:g}°F</b> → warm <b>{vmax:g}°F</b> "
                  f"(red). Lakewide mean this run: "
                  f"<b>{round(float(np.mean(f_vals)), 1)}°F</b>. White halo "
                  f"numbers are actual °F at that spot. The bar ends are "
                  f"recomputed every run from that day's water (1st/99th "
                  f"percentiles, clamped 32–86°F): blue always means the "
                  f"coldest water present and red the warmest, but the same "
                  f"color can stand for a different °F on different days — "
                  f"only the printed numbers are absolute.")
    meta["legend_scale_html"] = scale_html
    meta["coverage_mode"] = "basin"
    meta["blended_time_utc"] = blended_time
    meta["blended_source_id"] = blended_id
    meta["mur_time_utc"] = mur_time
    meta["mur_source_id"] = mur_id
    write_metadata(stage_prod, meta)  # re-write incl. buoy QC + legend text

    token = meta["source_version"]
    # Live Document/Folder descriptions carry NO legend images (same
    # contract as Live Precipitation): the only three legends live in
    # the three coverage overlay cards. The Document text explains the
    # foldering, toggling, and where the legends are.
    doc_explainer = (
        f"<p><b>This layer holds three coverages in one folder:</b> "
        f"{WT_LABELS['basin']} (default), {WT_LABELS['landonly']}, "
        f"{WT_LABELS['global']}. Open the "
        f"{CONFIG['title']} folder below and check exactly one coverage "
        f"at a time (Google Earth Pro switches them exclusively; Web/app "
        f"users check one box manually). {WT_LABELS['basin']} is always "
        f"the default: a fresh load or data refresh restores it (a saved "
        f"project may keep your own last-used checkboxes until you toggle "
        f"back). Each coverage carries its own "
        f"legend — click that coverage's row to open its card. There is "
        f"no shared legend at this level; a legend shown here would belong "
        f"to only one coverage and mislead about the others.</p>")
    doc_html = (doc_explainer
                + description_html(CONFIG["title"], meta,
                                   "Turn on/off independently of wave and "
                                   "ice layers.", "")
                + coverage_selector_html("basin"))
    meta["live_doc_html"] = doc_html
    # Self-contained per-coverage overlay legends: each GroundOverlay
    # description carries its own versioned legend image + endpoints +
    # source/timing, so visibility toggles can never mix them up.
    basin_odesc = _overlay_legend_html(
        PRODUCT, token, scale_html, CONFIG["freshness_label"],
        CONFIG["source_name"], GLSEA_URL, WT_LABELS["basin"],
        f"GLSEA {data_time_iso or 'see metadata'}")
    con_odesc = _overlay_legend_html(
        WT_DIRS["landonly"], _lake_token, _con_scale, _wstatus, _wsource,
        _wurl, WT_LABELS["landonly"],
        f"MUR {_mur_det}; Great Lakes GLSEA "
        f"{data_time_iso or 'see metadata'}")
    glo_odesc = _overlay_legend_html(
        WT_DIRS["global"], _ocean_token, _glo_scale, _wstatus, _wsource,
        _wurl, WT_LABELS["global"],
        f"MUR {_mur_det} (lakes) + blended {_blend_det} (oceans/seas); "
        f"Great Lakes GLSEA {data_time_iso or 'see metadata'}")
    folder_html = (
        f"<p><b>Sources:</b> {CONFIG['source_name']} (Great Lakes) + "
        f"{CONFIG.get('mur_source_name', 'MUR')} (lakes) + "
        f"{CONFIG.get('ocean_source_name', 'blended SST')} (oceans/seas)<br>"
        f"<b>Precedence:</b> {CONFIG.get('ocean_precedence', '')}<br>"
        f"<b>Coverage:</b> {WT_LABELS['basin']} (default shown)<br>"
        f"<b>Data time:</b> GLSEA {data_time_iso or 'see metadata'}; "
        f"MUR {_mur_det}; blended {_blend_det}</p>"
        f"<p>Endpoints: Great Lakes <b>{vmin:g}–{vmax:g}°F</b>; "
        f"Continental Lakes <b>{_con_vmin:g}–{_con_vmax:g}°F</b>; "
        f"Extended Oceanic <b>{_glo_vmin:g}–{_glo_vmax:g}°F</b>. Each "
        f"coverage's colors and endpoints come from its own water only."
        f"</p>"
        f"<p>Legends live only in the three coverage cards below — click "
        f"a coverage row to read its legend. View one coverage at a time."
        f"</p>") + coverage_selector_html("basin")
    meta["folder_html"] = folder_html
    meta["folder_radio"] = True
    meta["primary_name"] = WT_LABELS["basin"]
    meta["overlay_description"] = basin_odesc
    meta["extra_overlays"] = [
        {"name": WT_LABELS["landonly"],
         "png": f"{WT_DIRS['landonly']}/current.png",
         "bounds": dict(gbounds), "draw_order": 11, "visibility": 0,
         "overlay_description": con_odesc, "version_token": _lake_token},
        {"name": WT_LABELS["global"],
         "png": f"{WT_DIRS['global']}/current.png",
         "bounds": dict(gbounds), "draw_order": 12, "visibility": 0,
         "overlay_description": glo_odesc, "version_token": _ocean_token},
    ]
    write_metadata(stage_prod, meta)  # re-write incl. folder + overlays
    outs = live_out_dirs(stage, "Great_Lakes_Live_Water_Temperature.kml")
    kml_text = build_kml(
        PRODUCT, "Great_Lakes_Live_Water_Temperature.kml",
        "\U0001F321\uFE0F LIVE WATER TEMPERATURE",
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        doc_html,
        CONFIG["refresh_interval_seconds"], token,
        out_dirs=outs["live"],
        folder=(CONFIG["title"], folder_html), meta=meta,
        extra_overlays=[(WT_LABELS["landonly"],
                         f"{WT_DIRS['landonly']}/current.png",
                         dict(gbounds), 11, 0, con_odesc, _lake_token),
                        (WT_LABELS["global"],
                         f"{WT_DIRS['global']}/current.png",
                         dict(gbounds), 12, 0, glo_odesc, _ocean_token)],
        folder_radio=True, primary_name=WT_LABELS["basin"],
        overlay_descriptions=basin_odesc)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, "Great_Lakes_Live_Water_Temperature.kml",
        "\U0001F321\uFE0F LIVE WATER TEMPERATURE",
        entry_description_html(
            CONFIG["title"], meta,
            "Turn on/off independently of wave and ice layers."),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"source_last_modified": info["http_last_modified"],
                          "source_id": source_id,
                          "blended_time": blended_time,
                          "blended_source_id": blended_id,
                          "mur_time": mur_time,
                          "mur_source_id": mur_id,
                          "render_version": RENDER_VERSION,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
