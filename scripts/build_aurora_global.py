"""Pipeline — LIVE GLOBAL OVATION AURORA FORECAST (independent).

Full-coverage companion/replacement for the retired Great Lakes basin
aurora close-up: BOTH polar ovals from the same global OVATION JSON grid
(the northern file already spans -90..90), rendered on an equirectangular
world canvas (3600x1800, 0.1 deg/px) with the identical fixed 0-30+ scale.

Extra vs the basin layer: ONE rasterized viewline — a single straight
latitude line at the northern oval's farthest equatorward reach
(threshold crossings at/above the guard latitude), tagged once with its
latitude. Raster-burned (no LineString/Placemark: the raster-only rule
holds and validation stays green).

Intensity numbers ride in the wave/wind/temp label style (band-capped,
free placement; quiet runs place zero). The basin builder file is kept
(the painter + stops are imported from it); only its outputs are retired.

Exit codes: 0 updated (or skipped); 2 source/validation failure
(previous kept); 1 unexpected error.
"""

import json
import math
import os
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_aurora import AURORA_STOPS, fetch_ovation, paint_aurora_labels
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR,
                              base_metadata, load_bounds, now_det_str,
                              promote_stage, read_state, save_png,
                              source_token, stage_dir, write_metadata,
                              write_state)
from gradient_scale import draw_scale_legend, render_rgba

PRODUCT = "aurora_global"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = CONFIG["kml_file"]
OVERLAY_NAME = CONFIG["overlay_name"]
SKIP_NOTE = "Turn on/off independently of all other layers."
GW, GH = CONFIG["canvas_width"], CONFIG["canvas_height"]
GBOUNDS = {"lon_min": CONFIG["lon_min"], "lon_max": CONFIG["lon_max"],
           "lat_min": CONFIG["lat_min"], "lat_max": CONFIG["lat_max"]}


def _label_font_big(size=14):
    from build_aurora import _aur_label_font
    return _aur_label_font(size)


def resample_global(lons, lats, vals):
    """Bin OVATION obs to the 1-degree grid, then bilinear to 0.1 deg."""
    j = np.clip(np.floor(lons + 180.0).astype(int), 0, 359)
    i = np.clip(np.floor(lats + 90.0).astype(int), 0, 179)
    sums = np.bincount(i * 360 + j, weights=vals,
                       minlength=180 * 360).reshape(180, 360)
    cnts = np.bincount(i * 360 + j, minlength=180 * 360).reshape(180, 360)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = sums / np.where(cnts > 0, cnts, np.nan)
    xs = (np.arange(GW) + 0.5) / GW * 360.0 - 180.0
    ys = 90.0 - (np.arange(GH) + 0.5) / GH * 180.0
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
        return (f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty)
                + f01 * (1 - tx) * ty + f11 * tx * ty)


def farthest_viewline_lat(lons, lats, vals, threshold, guard):
    """Farthest equatorward reach of the NORTHERN oval: minimum latitude
    with intensity >= threshold at/above the guard (documented guard
    excludes near-equatorial grid artifacts, never real aurora)."""
    m = (lats >= guard) & (vals >= threshold)
    if not m.any():
        return None
    return float(lats[m].min())


def viewline_spans(lons, lats, vals, threshold, guard, vl, margin_deg=8.0):
    """Longitude ranges where the oval actually reaches near the viewline.

    The farthest reach is one latitude, but the oval never dips there at
    every longitude — drawing the line ocean-wide would imply visibility
    where the oval stays far north. Returns [(lon0, lon1)] merged spans
    (gaps <= 5 deg tolerated) where some cell within margin north of the
    line reaches threshold.
    """
    spans = []
    cur0 = cur1 = None
    for lo in range(-180, 180, 2):
        m = (lons >= lo) & (lons < lo + 2) & (lats >= guard) & (vals >= threshold)
        if not m.any():
            hit = False
        else:
            hit = float(lats[m].min()) <= vl + margin_deg
        if hit:
            if cur0 is None:
                cur0 = cur1 = lo
            elif lo - cur1 <= 5 + 2:
                cur1 = lo
            else:
                spans.append((cur0, cur1 + 2))
                cur0 = cur1 = lo
        else:
            if cur0 is not None:
                spans.append((cur0, cur1 + 2))
                cur0 = cur1 = None
    if cur0 is not None:
        spans.append((cur0, cur1 + 2))
    return spans


def paint_viewline(rgba, lat, bounds, spans=None):
    """Single straight raster latitude line + one mono tag (raster only).

    Spans (optional [(lon0, lon1)]) clip the line to longitudes where the
    oval actually reaches near it; None draws full-width (legacy).
    """
    H, W = rgba.shape[:2]
    row = int((bounds["lat_max"] - lat)
              / (bounds["lat_max"] - bounds["lat_min"]) * H)
    if not (4 <= row < H - 4):
        return rgba, False
    if spans:
        segs = []
        for lo0, lo1 in spans:
            c0 = int((lo0 - bounds["lon_min"])
                     / (bounds["lon_max"] - bounds["lon_min"]) * W)
            c1 = int((lo1 - bounds["lon_min"])
                     / (bounds["lon_max"] - bounds["lon_min"]) * W)
            c0, c1 = max(4, c0), min(W - 4, c1)
            if c1 - c0 >= 20:
                segs.append((c0, c1))
        if not segs:
            return rgba, False
    else:
        segs = [(4, W - 4)]
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    for c0, c1 in segs:
        for dy in (-1, 0, 1):
            d.line([(c0, row + dy), (c1, row + dy)],
                   fill=(255, 255, 255, 220))
    font = _label_font_big(14)
    tag = f"VIEWLINE ~{lat:.0f}\u00b0N"
    tx = segs[0][0] + 6
    d.text((tx, row - 22), tag, font=font, fill=(255, 255, 255, 255),
           stroke_width=1, stroke_fill=(20, 20, 20, 235))
    del d
    return np.array(img), True


def main():
    try:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--preview-dir", default=None)
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
        print(f"[{PRODUCT}] VALIDATION FAILED: missing timestamps.")
        return 2
    source_id = f"ovation-global-{obs}-fcst-{fcst}"
    prev = read_state(PRODUCT)
    if (prev.get("source_id") == source_id
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png"))
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE))):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping.")
        return 0
    try:
        return _build(obs, fcst, lons, lats, vals, hemi, kpf, kp_now,
                      source_id, preview_dir, font_size)
    except Exception:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED. Keeping previous.")
        return 2


def _build(obs, fcst, lons, lats, vals, hemi, kpf, kp_now, source_id,
           preview_dir=None, font_size=7):
    from geospatial_utils import iso_to_det
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    ok = np.isfinite(vals) & (vals >= 0) & (vals <= 100.0)
    print(f"[{PRODUCT}] obs={obs} fcst={fcst} cells={ok.sum()} "
          f"max={vals[ok].max() if ok.any() else float('nan'):.1f}")
    if ok.sum() < 1000:
        raise ValueError("too few valid cells")
    field = resample_global(lons[ok], lats[ok], vals[ok])
    # Tropical model-background mask (documented, not data removal of real
    # signal): OVATION is a high-latitude precipitation model; the faint
    # 1-4 equatorial/Americas-sector drizzle is background noise, so cells
    # below |40° paint only at/above the viewline threshold (storm signal
    # always survives; the viewline itself is computed on raw source).
    ys = 90.0 - (np.arange(GH) + 0.5) / GH * 180.0
    trop = (np.abs(ys) < 40.0)[:, None] & (field < 5.0)
    paint = np.where((field >= 1.0) & ~trop, field, np.nan)
    rgba = render_rgba(paint, AURORA_STOPS, 205)
    rgba, n_labels = paint_aurora_labels(rgba, paint, font_size=font_size)
    vl = farthest_viewline_lat(lons[ok], lats[ok], vals[ok],
                               CONFIG["viewline_threshold"],
                               CONFIG["viewline_guard_lat"])
    vl_ok = False
    vl_spans = []
    if vl is not None:
        vl_spans = viewline_spans(lons[ok], lats[ok], vals[ok],
                                  CONFIG["viewline_threshold"],
                                  CONFIG["viewline_guard_lat"], vl)
        rgba, vl_ok = paint_viewline(rgba, vl, GBOUNDS, spans=vl_spans)
    print(f"[{PRODUCT}] labels={n_labels} viewline={vl} drawn={vl_ok} "
          f"spans={vl_spans}")
    if preview_dir is not None:
        os.makedirs(preview_dir, exist_ok=True)
        save_png(rgba, os.path.join(preview_dir, "current.png"))
        print(f"[{PRODUCT}] PREVIEW written.")
        return 0
    save_png(rgba, os.path.join(stage_prod, "current.png"))

    obs_det = iso_to_det(obs) or obs
    fcst_det = iso_to_det(fcst) or fcst
    kp_txt = kpf or kp_now or "n/a"
    hemi_txt = hemi or "n/a"
    vl_txt = (f"~{vl:.0f}\u00b0N northern-oval farthest reach"
              if vl is not None else "no activity reaching threshold")
    subtitle = f"Global both ovals, fixed 0-30+  |  {fcst_det}"
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        "intensity", AURORA_STOPS,
        [(0.0, "LOWEST 0"), (6.0, "6 oval"), (14.0, "14 bright"),
         (23.0, "23 severe"), (30.0, "HIGHEST+ 30+")],
        f"SWPC OVATION {obs_det} | {hemi_txt} GW | Kp {kp_txt}",
        note="White line = farthest viewline. Below-faint transparent.")
    scale_html = (
        f"Global OVATION auroral intensity, FIXED absolute scale 0&ndash;30+: "
        f"<b>LOWEST 0</b> (transparent) &rarr; <b>6 bright green</b> "
        f"(oval overhead) &rarr; <b>14 orange</b> &rarr; <b>23 magenta</b> "
        f"(severe) &rarr; <b>HIGHEST+ 30+</b> (violet extreme). Same "
        f"intensity always shows the same color. White halo numbers are "
        f"actual intensity at that spot. Single white viewline: the "
        f"northern oval's farthest equatorward reach now "
        f"(<b>{vl_txt}</b>) — aurora may be visible low on the northern "
         f"horizon from there under dark, clear skies. The line spans "
         f"only longitudes where the oval actually reaches near it. Faint "
         f"equatorial speckles are a known OVATION grid artifact, not "
         f"aurora.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=f"obs {obs_det}; forecast {fcst_det}",
        source_last_modified_utc="n/a (SWPC JSON, polled hourly)",
        units="OVATION intensity 0-30+ (display = source)",
        source_resolution="1-degree global OVATION grid, bilinear to 0.1 deg canvas",
        color_min=0.0, color_max=30.0, color_units="OVATION intensity",
        missing_data_treatment=("intensity < 1.0 transparent; tropical "
                                "model-background cells (|lat| < 40° and "
                                "intensity < 5) masked as non-auroral noise — "
                                "storm-level signal always paints, and the "
                                "viewline is computed on unmasked source; "
                                "full world equirectangular canvas lon "
                                "-180..180 / lat -90..90 (3600x1800); missing "
                                "source transparent, never zero-filled."))
    meta["render_bounds"] = {"lon_min": -180.0, "lon_max": 180.0,
                             "lat_min": -90.0, "lat_max": 90.0}
    meta["spatial_resolution_rendered"] = (
        "3600x1800 world canvas over lon [-180,180], lat [-90,90] "
        "(0.1 deg/px bilinear from the 1-degree source grid)")
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["source_id"] = source_id
    meta["source_version"] = source_token(
        f"{source_id}-r{RENDER_VERSION}-aurglobal-tx2")
    meta["viewline_threshold"] = CONFIG["viewline_threshold"]
    meta["viewline_farthest_lat"] = vl
    meta["stats"] = {"valid_cells": int(np.isfinite(paint).sum()),
                     "aurora_labels": n_labels,
                     "viewline_drawn": vl_ok}
    token = meta["source_version"]
    folder_html = (
        f"<p><img src=\"{pages_base()}/{PRODUCT}/legend.png?v={token}\" "
        f"width=\"600\" alt=\"key\"></p><p>{scale_html}</p>")
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
        out_dirs=outs["live"], meta=meta, bounds={
            "lon_min": -180.0, "lon_max": 180.0,
            "lat_min": -90.0, "lat_max": 90.0, "crs": "EPSG:4326"})
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


def pages_base():
    from build_kml import pages_base as _pb
    return _pb()


if __name__ == "__main__":
    sys.exit(main())
