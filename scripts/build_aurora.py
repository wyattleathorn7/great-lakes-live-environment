"""Pipeline — LIVE OVATION AURORA FORECAST (independent).

NOAA/SWPC OVATION Prime aurora model (the model family behind the NOAA
Aurora Viewline Tonight/Tomorrow-Night product):
  JSON grid  https://services.swpc.noaa.gov/json/ovation_aurora_latest.json
             ([lon, lat, aurora] 1-degree global grid + Observation/Forecast Time)
  Text run   https://services.swpc.noaa.gov/text/ovation_latest_aurora_n.txt
             (hemispheric power, forecast Kp)
  Kp context https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json

The webpage image is NEVER scraped; only these machine-readable
authoritative endpoints drive the raster.

Footprint: the EXACT rectangular WGS84 footprint of LIVE LEAF COLOR
(lon -93..-73.5, lat 40.5..49.5, 1800x1175 canvas). Aurora is
atmospheric, so the FULL basin rectangle paints (same treatment as the
solar / air-temperature / pressure basin-rectangle layers): no shoreline
cut, no cropping to the lakes, no basin polygon. LEAF files are never
touched.

Rendering: OVATION aurora values over the rectangle are bilinearly
resampled onto the canvas and painted through a FIXED absolute
NOAA-like intensity scale (deep green auroral oval -> yellow -> red ->
violet extreme). Below the detection floor the canvas stays
transparent. No placemarks, no polygons, no vector lines — raster only.

Viewline: the southernmost latitude (within the basin longitude band)
where OVATION intensity reaches the viewline threshold is reported in
the legend/metadata/folder text as the model-derived viewline. It marks
the southernmost region from which aurora may be visible low on the
northern horizon under appropriate (dark, clear, unobstructed)
conditions — never a guarantee. No KML line is drawn (raster-only rule).

Refresh: source checked hourly; entry NetworkLink + live Icon both
refreshMode=onInterval at 3600 s. Source-aware gate: rebuilds only when
the OVATION Observation/Forecast time changes.

Exit codes: 0 updated (or skipped); 2 source/validation failure
(previous kept); 1 unexpected error.
"""

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
                               base_metadata, load_bounds, now_det_str,
                               promote_stage, read_state, save_png,
                               source_token, stage_dir, write_metadata,
                               write_state)
from gradient_scale import draw_scale_legend, render_rgba

PRODUCT = "aurora"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Ovation_Aurora_Forecast.kml"
OVERLAY_NAME = "\U0001F30C LIVE OVATION AURORA FORECAST"
SKIP_NOTE = "Turn on/off independently of all other layers."
AURORA_ALPHA = 205  # same as the shared overlay_alpha
UA = {"User-Agent": "great-lakes-live-environment/1.0"}

# Fixed absolute OVATION-intensity scale (model units). Same intensity
# always shows the same color. Anchors follow the NOAA aurora-product
# visual language: faint green (oval edge) -> bright green -> yellow ->
# orange -> red -> magenta -> violet (extreme). Values >= 30 clamp violet.
AURORA_STOPS = [
    (0.0, (10, 60, 35)),     # near-black green (faint)
    (3.0, (20, 140, 60)),    # auroral green (oval edge / viewline band)
    (6.0, (60, 200, 80)),    # bright green
    (10.0, (235, 220, 60)),  # yellow
    (14.0, (245, 150, 30)),  # orange
    (18.0, (220, 50, 35)),   # red
    (23.0, (200, 30, 140)),  # magenta
    (30.0, (90, 20, 140)),   # violet extreme (30+ clamps here)
]
AURORA_MAX = 30.0
AURORA_TICKS = [
    (0.0, "LOWEST 0"),
    (6.0, "6 oval"),
    (14.0, "14 bright"),
    (23.0, "23 severe"),
    (30.0, "HIGHEST+ 30+"),
]


def _get(url, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _aur_label_font(size=7):
    """DejaVu Sans Mono Bold (same face as the wave/wind/temp labels)."""
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


def _draw_aur_label(d, x, y, text, font):
    """White core + 1 px outline stroke (same as the wave/wind/temp labels)."""
    d.text((x, y), text, font=font, fill=(255, 255, 255, 255),
           stroke_width=1, stroke_fill=(20, 20, 20, 235))


# Intensity bands (lo, hi, max labels): outer ring stays sparse (3-5),
# the visible center mass carries the bulk (8-15 across the mids), the
# extreme top stays sparse so the gradient stays readable. Painter is
# footprint-agnostic (takes any field) so a future full-North-America
# canvas can reuse it unchanged.
AURORA_BANDS = [
    (1.0, 3.0, 4),    # outer oval edge
    (3.0, 8.0, 4),    # faint body
    (8.0, 15.0, 9),   # bright center mass
    (15.0, 22.0, 6),  # severe
    (22.0, float("inf"), 3),  # extreme top
]


def paint_aurora_labels(rgba, field, font_size=7, extra_sep=45):
    """OVATION intensity numbers in the wave/wind/temp label style.

    NO preset locations: data-seeded jittered candidates. Within each
    intensity band, interior (locally uniform) cells go first so labels
    sit inside color regions instead of on ring edges; per-band caps keep
    any single measurement from repeating all over the map. Quiet runs
    (empty basin) simply place zero labels — never a failure.
    Returns (rgba, n_labels).
    """
    import math as _math
    from PIL import Image, ImageDraw
    H, W = field.shape
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    font = _aur_label_font(font_size)
    ok = np.isfinite(field)
    if not ok.any():
        del d
        return np.array(img), 0
    _fin = field[ok]
    seed = int(abs(float(_fin.mean())) * 1000 + float(_fin.std()) * 97) \
        % (2 ** 32 - 1)
    rng = np.random.default_rng(seed)
    grad = np.zeros(field.shape)
    for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
        shifted = np.roll(field, shift=(-dr, -dc), axis=(0, 1))
        diff = np.abs(field - shifted)
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
            cand.append((float(field[r, c]), float(grad[r, c]),
                         rng.random(), r, c))
    placed = []
    n_labels = 0

    def _collides(x0, y0, x1, y1, pad=3):
        for (a0, b0, a1, b1) in placed:
            if not (x1 + pad < a0 or x0 - pad > a1 or y1 + pad < b0 or y0 - pad > b1):
                return True
        return False

    def _put(cx, cy, text):
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
            if _math.hypot(acx - cx, acy - cy) < extra_sep:
                return False
        pts = [(cx, cy), (x + 2, y + 2), (x + tw - 2, y + 2)]
        for px, py in pts:
            ix = int(np.clip(round(px), 0, W - 1))
            iy = int(np.clip(round(py), 0, H - 1))
            if rgba[iy, ix, 3] == 0:
                return False
        _draw_aur_label(d, x, y, text, font)
        placed.append((x, y, x + tw, y + th))
        return True

    for lo, hi, cap in AURORA_BANDS:
        band = [t for t in cand if lo <= t[0] < hi]
        # interiors first (low local change), then jitter order
        band.sort(key=lambda t: (t[1], t[2]))
        n = 0
        for v, _g, _j, r, c in band:
            if n >= cap:
                break
            if _put(c, r - 12, f"{v:.0f}"):
                n += 1
                n_labels += 1

    del d
    return np.array(img), n_labels


def fetch_ovation():
    """Return (obs_time, fcst_time, lons, lats, vals, hemi_power, fcst_kp)."""
    raw = _get(CONFIG["data_endpoint"])
    doc = json.loads(raw.decode("utf-8"))
    obs = str(doc.get("Observation Time", "")).strip()
    fcst = str(doc.get("Forecast Time", "")).strip()
    coords = doc.get("coordinates", [])
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1000:
        raise ValueError(f"unexpected OVATION grid shape {arr.shape}")
    lons = ((arr[:, 0] + 180.0) % 360.0) - 180.0
    lats = arr[:, 1]
    vals = arr[:, 2]
    hemi, kpf = None, None
    try:
        txt = _get(CONFIG["data_endpoint_text"]).decode("utf-8", errors="replace")
        for line in txt.splitlines():
            s = line.strip()
            if s.startswith("Hemispheric Power"):
                hemi = s.split(":")[-1].strip().split()[0]
            if s.startswith("Forecast Kp"):
                kpf = s.split(":")[-1].strip().split()[0]
    except Exception:
        pass
    kp_now = None
    try:
        kj = json.loads(_get(CONFIG["kp_endpoint"]).decode("utf-8"))
        if isinstance(kj, list) and kj:
            kp_now = str(kj[-1].get("kp", ""))
    except Exception:
        pass
    return obs, fcst, lons, lats, vals, hemi, kpf, kp_now


def resample_to_canvas(lons, lats, vals, bounds, shape):
    """Bilinear resample of the 1-degree OVATION grid onto the canvas."""
    H, W = shape
    lon_ax = np.arange(-180.0, 180.0, 1.0)  # cell centres approx
    lat_ax = np.arange(-90.0, 90.0, 1.0)
    # Bin observations to the 1-degree grid first (mean per cell).
    j = np.clip(np.floor(lons + 180.0).astype(int), 0, 359)
    i = np.clip(np.floor(lats + 90.0).astype(int), 0, 179)
    sums = np.bincount(i * 360 + j, weights=vals, minlength=180 * 360).reshape(180, 360)
    cnts = np.bincount(i * 360 + j, minlength=180 * 360).reshape(180, 360)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = sums / np.where(cnts > 0, cnts, np.nan)
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    xs = (np.arange(W) + 0.5) / W * (lon_max - lon_min) + lon_min
    ys = lat_max - (np.arange(H) + 0.5) / H * (lat_max - lat_min)
    fx = np.clip(xs + 180.0, 0, 359.999)
    fy = np.clip(ys + 90.0, 0, 179.999)
    j0 = np.clip(np.floor(fx).astype(int), 0, 358)
    i0 = np.clip(np.floor(fy).astype(int), 0, 178)
    tx = (fx - j0)[None, :]
    ty = (fy - i0)[:, None]
    f00 = grid[i0[:, None], j0[None, :]]
    f10 = grid[i0[:, None], j0[None, :] + 1]
    f01 = grid[i0[:, None] + 1, j0[None, :]]
    f11 = grid[i0[:, None] + 1, j0[None, :] + 1]
    with np.errstate(invalid="ignore"):
        field = (f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty)
                 + f01 * (1 - tx) * ty + f11 * tx * ty)
    return field


def viewline_latitude(field, bounds, threshold):
    """Southernmost canvas latitude with intensity >= threshold, else None."""
    H = field.shape[0]
    rows = np.where((field >= threshold).any(axis=1))[0]
    if rows.size == 0:
        return None
    r = rows.max()  # row 0 = north
    lat_max, lat_min = bounds["lat_max"], bounds["lat_min"]
    return lat_max - (r + 0.5) / H * (lat_max - lat_min)


def global_viewline(lons, lats, vals, threshold):
    """NOAA-style viewline from the GLOBAL grid: southernmost northern-
    hemisphere latitude (within the basin longitude band) reaching
    `threshold`. This is the product's true visibility boundary even when
    the oval sits entirely north of the rectangle (the normal case).

    Latitudes below 30N are excluded: the OVATION grid carries isolated
    near-equatorial artifact cells (e.g. a lat-0 row reaching 5+) that
    are not aurora — real auroral precipitation is strictly high
    latitude. Documented guard, not data removal (the raster itself is
    an unmodified resample of the source grid).
    """
    m = (lons >= -93.0) & (lons <= -73.5) & (lats >= 30.0) & (vals >= threshold)
    if not m.any():
        return None
    return float(lats[m].min())


def main():
    try:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--preview-dir", default=None,
                        help="preview: write current.png here, skip promote")
        ap.add_argument("--font-size", type=int, default=7)
        args = ap.parse_args()
        return run(preview_dir=args.preview_dir, font_size=args.font_size)
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run(preview_dir=None, font_size=7):
    try:
        obs, fcst, lons, lats, vals, hemi, kpf, kp_now = fetch_ovation()
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    if not obs or not fcst:
        print(f"[{PRODUCT}] VALIDATION FAILED: missing timestamps. Keeping previous.")
        return 2
    source_id = f"ovation-{obs}-fcst-{fcst}"
    prev = read_state(PRODUCT)
    if (prev.get("source_id") == source_id
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png"))
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE))):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping.")
        return _refresh_kml()
    try:
        return _build(obs, fcst, lons, lats, vals, hemi, kpf, kp_now,
                      source_id, preview_dir, font_size)
    except Exception:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED. Keeping previous.")
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


def _build(obs, fcst, lons, lats, vals, hemi, kpf, kp_now, source_id,
           preview_dir=None, font_size=7):
    from geospatial_utils import iso_to_det
    bounds = load_bounds()
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    ok = np.isfinite(vals) & (vals >= 0) & (vals <= CONFIG["valid_max"])
    print(f"[{PRODUCT}] obs={obs} fcst={fcst} cells={ok.sum()} "
          f"max={vals[ok].max() if ok.any() else float('nan'):.1f}")
    if ok.sum() < 1000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few valid cells. Keeping previous.")
        return 2
    field = resample_to_canvas(lons[ok], lats[ok], vals[ok], bounds, (H, W))
    floor = CONFIG["detection_floor"]
    paint = np.where(field >= floor, field, np.nan)
    n_valid = int(np.isfinite(paint).sum())
    print(f"[{PRODUCT}] canvas valid>={floor}: {n_valid} px")
    rgba = render_rgba(paint, AURORA_STOPS, AURORA_ALPHA)
    # Full basin rectangle: no shoreline cut (atmospheric layer).
    # Intensity numbers in the wave/wind/temp label style (band-capped,
    # free placement). Quiet runs place zero — never a failure.
    rgba, n_aur_labels = paint_aurora_labels(rgba, paint, font_size=font_size)
    print(f"[{PRODUCT}] aurora_labels={n_aur_labels}")
    if preview_dir is not None:
        os.makedirs(preview_dir, exist_ok=True)
        save_png(rgba, os.path.join(preview_dir, "current.png"))
        print(f"[{PRODUCT}] PREVIEW written.")
        return 0
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    vl_in = viewline_latitude(field, bounds, CONFIG["viewline_threshold"])
    vl_global = global_viewline(lons[ok], lats[ok], vals[ok], CONFIG["viewline_threshold"])
    if vl_global is not None:
        if vl_global <= bounds["lat_max"] + 0.5 and vl_global >= bounds["lat_min"] - 2.0:
            vl_txt = (f"~{vl_global:.0f}°N (at/inside this map — northern Great Lakes may see aurora "
                      f"low on the northern horizon when dark)")
        elif vl_global > bounds["lat_max"] + 0.5:
            vl_txt = (f"~{vl_global:.0f}°N (north of this map — oval is poleward of the Great Lakes now)")
        else:
            vl_txt = (f"~{vl_global:.0f}°N (south of the Great Lakes — aurora may be visible from the basin when dark)")
    else:
        vl_txt = "no auroral activity reaching viewline threshold anywhere in the basin longitude band"
    obs_det = iso_to_det(obs) or obs
    fcst_det = iso_to_det(fcst) or fcst
    kp_txt = kpf or kp_now or "n/a"
    hemi_txt = hemi or "n/a"
    subtitle = (f"Fixed 0-30+ scale  |  {fcst_det}")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        "intensity", AURORA_STOPS, AURORA_TICKS,
        f"SWPC OVATION {obs_det} | {hemi_txt} GW | Kp {kp_txt}",
        note="Below-faint stays transparent. Forecast, not a sighting guarantee.")
    scale_html = (
        f"OVATION auroral intensity, FIXED absolute scale 0&ndash;30+: "
        f"<b>LOWEST 0</b> (transparent, no activity) &rarr; <b>6 bright green</b> "
        f"(auroral oval overhead) &rarr; "
        f"<b>14 orange</b> &rarr; <b>23 magenta</b> (severe) &rarr; "
         f"<b>HIGHEST+ 30+</b> (violet extreme). "
         f"Same intensity always shows the same color. White halo numbers "
         f"are actual OVATION intensity at that spot (sparse by band so "
         f"the gradient stays readable). Model-derived viewline in basin: "
        f"<b>{vl_txt}</b> (southernmost latitude reaching intensity 5+: aurora may be "
        f"visible low on the northern horizon from there under dark, clear skies).")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=f"obs {obs_det}; forecast {fcst_det}",
        source_last_modified_utc="n/a (SWPC JSON, polled hourly)",
        units=f"{CONFIG['display_units']}; source {CONFIG['source_units']}",
        source_resolution="1-degree global OVATION grid, bilinearly resampled to canvas (display only; source precision unchanged)",
        color_min=0.0, color_max=AURORA_MAX, color_units="OVATION intensity",
        missing_data_treatment=(
            f"intensity < {floor} transparent (no activity, valid); "
            "full basin rectangle lon -93..-73.5 / lat 40.5..49.5, no shoreline cut "
            "(atmospheric layer, like solar/air-temperature); missing source transparent, never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["source_id"] = source_id
    # Style rides the version token (same-cycle restyles change the ?v=
    # URL so caches cannot serve old pixels).
    meta["source_version"] = source_token(
        f"{source_id}-r{RENDER_VERSION}-aurvals-tx1")
    meta["noaa_product"] = "Aurora Viewline Tonight and Tomorrow Night (Experimental) product family"
    meta["noaa_model"] = "OVATION Prime (2013 version, real-time; Newell et al. / Machol & Redmon / Viereck implementation)"
    meta["data_endpoint"] = CONFIG["data_endpoint"]
    meta["observation_time_utc"] = obs
    meta["forecast_time_utc"] = fcst
    meta["hemispheric_power_gw"] = hemi_txt
    meta["forecast_kp"] = kp_txt
    meta["viewline_threshold"] = CONFIG["viewline_threshold"]
    meta["viewline_latitude_global"] = vl_global
    meta["viewline_latitude_in_canvas"] = vl_in
    meta["viewline_text"] = vl_txt
    meta["kp_latest_context"] = kp_now or "n/a"
    meta["footprint"] = CONFIG["footprint_note"]
    meta["stats"] = {"valid_cells": n_valid,
                     "current_min": float(np.nanmin(paint)) if n_valid else None,
                     "current_max": float(np.nanmax(paint)) if n_valid else None,
                     "aurora_labels": n_aur_labels}
    token = meta["source_version"]
    folder_html = (
        f"<p><img src=\"{legend_src(token)}\" width=\"600\" alt=\"Aurora gradient key\"></p>"
        f"<h2>What This Shows</h2>"
        f"<p>Expected auroral intensity and oval location over the Great Lakes from NOAA's "
        f"OVATION Prime model — the same model family behind NOAA's Aurora Viewline product. "
        f"Colors follow NOAA's visual concept: green auroral-oval band, warming through yellow/orange "
        f"to red/magenta/violet as predicted intensity rises. This is a Google Earth raster "
        f"implementation using NOAA's authoritative data, not a copy of NOAA's webpage image.</p>"
        f"<h2>Source / Study</h2>"
        f"<p><b>Product:</b> NOAA/SWPC Aurora Viewline product family (experimental viewline page).<br/>"
        f"<b>Model:</b> OVATION Prime (real-time auroral precipitation model driven by solar-wind observations; "
        f"forecast position/intensity of the auroral oval).<br/>"
        f"<b>Endpoint:</b> <a href=\"{CONFIG['data_endpoint']}\">{CONFIG['data_endpoint']}</a><br/>"
        f"<b>Run context:</b> hemispheric power {hemi_txt} GW; forecast Kp {kp_txt}.<br/>"
        f"<b>Observation time:</b> {obs_det}<br/><b>Forecast time:</b> {fcst_det}</p>"
        f"<h2>How to Read the Gradient</h2>"
        f"<p>{scale_html}</p>"
        f"<p><b>OVATION role:</b> OVATION converts upstream solar-wind energy into maps of auroral particle "
        f"precipitation; brighter regions = stronger predicted aurora overhead.</p>"
        f"<p><b>Viewline meaning:</b> the viewline is the southernmost region from which aurora may be seen "
        f"low on the northern horizon under appropriate conditions (dark, clear, unobstructed northern view). "
        f"Model-derived basin viewline now: <b>{vl_txt}</b>. Being south of the line means aurora is unlikely; "
        f"being near or north of it means look north when dark — it is an opportunity boundary, not a sighting line.</p>"
        f"<p><b>Great Lakes reading:</b> green over your lake = oval possibly overhead/near; yellow→red = strong "
        f"activity nearby; violet = extreme storm conditions. Daylight, clouds, moonlight and city lights can hide "
        f"even a strong forecast.</p>"
        f"<h2>Data / Scientific Limitations</h2>"
        f"<p>Auroral forecast/prediction only — it does not guarantee anyone will see aurora from a specific place "
        f"at a specific time. Intensity is model output in OVATION units, not a naked-eye brightness meter. Coverage is "
        f"the fixed project rectangle (lon −93…−73.5, lat 40.5…49.5, WGS84, 1800×1175), inherited from LIVE LEAF COLOR; "
        f"it is not cropped to shorelines. Kp context supplements but never replaces the OVATION field.</p>"
        f"<h2>Update Frequency</h2>"
        f"<p>Source checked hourly (refreshInterval 3600 s on both the entry NetworkLink and the overlay Icon). "
        f"SWPC issues new OVATION runs about every 5 minutes; the layer rebuilds only when the "
        f"Observation/Forecast time actually changes (source version <b>{source_id}</b>).</p>")
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
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def legend_src(token):
    from build_kml import pages_base
    return f"{pages_base()}/{PRODUCT}/legend.png?v={token}"


if __name__ == "__main__":
    sys.exit(main())
