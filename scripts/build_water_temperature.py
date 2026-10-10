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

PRODUCT = "water_temperature"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
GLSEA_URL = CONFIG["source_url"]
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")

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
                      max_labels=180):
    """Actual °F values in the wave/wind label style, free placement.

    NO fixed grid: data-seeded jittered candidates (positions move with the
    water, never a rigid lattice); thermal-front cells go first, then a
    spaced fill up to quota. Returns (rgba, n_priority, n_fill).
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

    def _put(cx, cy, text, sep):
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
        for px, py in pts:
            ix, iy = int(np.clip(round(px), 0, W - 1)), int(np.clip(round(py), 0, H - 1))
            if rgba[iy, ix, 3] == 0:
                return False
        if not np.isfinite(field_f[int(np.clip(round(cy), 0, H - 1)),
                                   int(np.clip(round(cx), 0, W - 1))]):
            return False
        _draw_temp_label(d, x, y, text, font)
        placed.append((x, y, x + tw, y + th))
        return True

    for g, _, r, c in cand:
        if n_priority + n_fill >= max_labels:
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

    del d
    return np.array(img), n_priority, n_fill


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


def run(preview_dir=None, font_size=7, extra_thresh=1.5, max_labels=180):
    raw_path = os.path.join(RAW_DIR, "glsea_cur.asc")
    # Cheap hourly gate: HEAD the Last-Modified stamp before the ~8 MB
    # download. Match with the committed state -> source unchanged, KMLs
    # only. Anything else falls through to download + content-hash compare.
    prev = read_state(PRODUCT)
    lm = head_last_modified(GLSEA_URL)
    if lm and lm == prev.get("source_last_modified") \
            and prev.get("render_version") == RENDER_VERSION \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live",
                "Great_Lakes_Live_Water_Temperature.kml")):
        print(f"[{PRODUCT}] HEAD unchanged ({lm}); keeping raster.")
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
    if prev.get("source_id") == source_id \
            and prev.get("render_version") == RENDER_VERSION \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live",
                "Great_Lakes_Live_Water_Temperature.kml")):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
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
                      extra_thresh, max_labels)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(info, raw_path, source_id, preview_dir=None, font_size=7,
           extra_thresh=1.5, max_labels=180):
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
    p1, p99 = float(np.percentile(f_vals, 1)), float(np.percentile(f_vals, 99))
    vmin = max(32.0, math.floor(p1))
    vmax = min(86.0, math.ceil(p99))
    if vmax < vmin + 2:
        vmin, vmax = 32.0, 75.0
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
    meta["source_version"] = source_token(f"{source_id}-r{RENDER_VERSION}-tempvals-tx1")
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
        subtitle=(f"{CONFIG['freshness_label']}  |  Data time: "
                  f"{data_time_iso or 'see metadata'}"),
        source_line=(f"Source: NOAA/GLERL CoastWatch GLSEA  |  "
                     f"Processed {now_det_str()}"),
        unit_label="\u00b0F", transparent_value=None, fmt="{:.0f}",
        splat_radius=1, product_dir=stage_prod)

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
                  f"numbers are actual °F at that spot.")
    meta["legend_scale_html"] = scale_html
    write_metadata(stage_prod, meta)  # re-write incl. buoy QC + legend text

    token = meta["source_version"]
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    outs = live_out_dirs(stage, "Great_Lakes_Live_Water_Temperature.kml")
    kml_text = build_kml(
        PRODUCT, "Great_Lakes_Live_Water_Temperature.kml",
        "\U0001F321\uFE0F LIVE WATER TEMPERATURE",
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta,
                         "Turn on/off independently of wave and ice layers.",
                         block),
        CONFIG["refresh_interval_seconds"], token,
        out_dirs=outs["live"],
        meta=meta)
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
                          "render_version": RENDER_VERSION,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
