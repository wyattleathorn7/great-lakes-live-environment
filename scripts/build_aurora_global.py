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


def oval_edge_columns(paint, threshold, bounds, step=2):
    """Exact equatorward boundary of what is PAINTED, per canvas column.

    No smoothing, no margins, no guards beyond the raster's own documented
    rules: for each column, the southernmost pixel at/above threshold.
    Columns without reach split the curve. The line IS the displayed data
    edge — nothing interpretive.
    Returns ([(x, y)] pixel points, southernmost_lat or None).
    """
    H, W = paint.shape
    # Northern oval only: rows at/above the equator (row of lat 0).
    r_eq = int((bounds["lat_max"] - 0.0)
               / (bounds["lat_max"] - bounds["lat_min"]) * H)
    pts = []
    for x in range(0, W, step):
        col = paint[:r_eq, x]
        hits = np.where(np.isfinite(col) & (col >= threshold))[0]
        if hits.size:
            pts.append((x, int(hits.max())))
    southernmost = None
    if pts:
        rmax = max(r for _, r in pts)
        southernmost = bounds["lat_max"] - (rmax + 0.5) / H \
            * (bounds["lat_max"] - bounds["lat_min"])
    return pts, southernmost


def paint_viewline(rgba, edge_pts, bounds, tag_lat):
    """Raster viewline = the painted threshold edge itself (raster only).

    Raw pixel points, unsmoothed; gaps in the data split the curve. One
    mono tag at the southernmost dip. Cartographic linework only (dark
    underlay + white core so it reads on any color) — every position is
    a data pixel.
    """
    H, W = rgba.shape[:2]
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    # split on pixel gaps (missing columns break the run)
    runs, cur = [], []
    for x, r in sorted(edge_pts):
        if not (4 <= r < H - 4):
            continue
        if cur and x - cur[-1][0] > 3 * 2:
            runs.append(cur)
            cur = []
        cur.append((x, r))
    if cur:
        runs.append(cur)
    drew = False
    dip = None
    for run in runs:
        if len(run) < 3:
            continue
        flat = [(x, y) for x, y in run]
        d.line(flat, fill=(20, 20, 20, 170), width=5, joint="curve")
        d.line(flat, fill=(255, 255, 255, 235), width=2, joint="curve")
        drew = True
        cand = max(run, key=lambda t: t[1])
        if dip is None or cand[1] > dip[1]:
            dip = cand
    if drew and dip is not None:
        font = _label_font_big(14)
        lat = bounds["lat_max"] - (dip[1] + 0.5) / H \
            * (bounds["lat_max"] - bounds["lat_min"])
        d.text((min(max(dip[0] - 40, 6), W - 170), dip[1] + 8),
               f"VIEWLINE ~{lat:.0f}\u00b0N", font=font,
               fill=(255, 255, 255, 255),
               stroke_width=1, stroke_fill=(20, 20, 20, 235))
    del d
    return np.array(img), drew


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
    # Source-level floor mask (verified signature, not assumed): obs cells
    # at exactly the model floor (< 2.0) with no real activity (>= 2.0)
    # within 3 deg are fill, not precipitation — static across frames,
    # exactly uniform amid zeros (France/SAtlantic sectors verified).
    # Real faint oval cells always neighbor genuine activity. Masked cells
    # are dropped before resampling so no smear of them can paint.
    _jj = np.clip(np.floor(lons + 180.0).astype(int), 0, 359)
    _ii2 = np.clip(np.floor(lats + 90.0).astype(int), 0, 179)
    _gmax = np.full(180 * 360, -1.0)
    _vv = np.where(ok, vals, -1.0)
    np.maximum.at(_gmax, _ii2 * 360 + _jj, _vv)
    _gmax = _gmax.reshape(180, 360)
    _dil = _gmax.copy()
    for _di in (-3, -2, -1, 1, 2, 3):
        for _dj in (-3, -2, -1, 1, 2, 3):
            _dil = np.maximum(_dil, np.roll(np.roll(_gmax, _di, axis=0),
                                            _dj, axis=1))
    _cellmax = _dil[_ii2, _jj]
    _bg = ok & (vals < 2.0) & (_cellmax < 2.0)
    print(f"[{PRODUCT}] floor-fill masked obs cells: {int(_bg.sum())}")
    use = ok & ~_bg
    field = resample_global(lons[use], lats[use], vals[use])
    # Background-noise rule (documented; applies equally at ALL latitudes,
    # so it is not a geographic bias): OVATION carries a persistent,
    # static, exactly-uniform faint floor (verified: hundreds of cells at
    # exactly 1.0 amid zeros, unmoving across frames, far below the
    # product's own viewing-relevance threshold). When genuine activity
    # exists, faint green paints only attached to it (oval edges stay,
    # isolated floor cells never paint).
    core = np.isfinite(field) & (field >= 5.0)
    # Background rule (documented; verified against source, not assumed):
    # the persistent static exact-1.0 floor appears far from any activity,
    # while real faint oval sections sit poleward or adjacent to bright
    # cores. So: poleward of |50° everything >= floor paints (never erase
    # the oval); equatorward, faint paints only attached (within 3°,
    # covering bilinear smear plus genuine gradient fringe) to genuinely
    # active cells, so storm expansions survive and flat floor plateaus
    # never paint. Quiet regimes keep their poleward faint oval.
    _R, _S = 30, 61
    _pad = np.pad(core.astype(float), _R, mode="constant")
    _ii = np.zeros((np.asarray(_pad.shape) + 1), dtype=float)
    _ii[1:, 1:] = np.cumsum(np.cumsum(_pad, axis=0), axis=1)
    _support = (_ii[_S:_S + GH, _S:_S + GW] - _ii[0:GH, _S:_S + GW]
                - _ii[_S:_S + GH, 0:GW] + _ii[0:GH, 0:GW]) > 0
    _ys = 90.0 - (np.arange(GH) + 0.5) / GH * 180.0
    _high = (np.abs(_ys) >= 50.0)[:, None]
    paint = np.where((field >= 1.0) & (_high | _support), field, np.nan)
    rgba = render_rgba(paint, AURORA_STOPS, 205)
    rgba, n_labels = paint_aurora_labels(rgba, paint, font_size=font_size)
    # Viewline = the painted threshold edge itself (no smoothing, margins,
    # or separate guards: every line pixel is a displayed data pixel).
    edge_pts, vl = oval_edge_columns(paint, CONFIG["viewline_threshold"],
                                     GBOUNDS)
    vl_ok = False
    if vl is not None:
        rgba, vl_ok = paint_viewline(rgba, edge_pts, GBOUNDS, vl)
    print(f"[{PRODUCT}] labels={n_labels} viewline={vl} drawn={vl_ok}")
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
         f"actual intensity at that spot. White viewline hugs the oval's "
         f"farthest equatorward edge (dipping where coverage dips, like "
         f"NOAA's own view line) "
         f"(<b>{vl_txt}</b>) — aurora may be visible low on the northern "
         f"horizon from there under dark, clear skies. The line spans "
         f"only longitudes where the oval actually reaches near it. "
         f"Isolated faint cells paint only where connected to genuinely "
         f"active areas, so model background noise never renders as "
         f"aurora.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=f"obs {obs_det}; forecast {fcst_det}",
        source_last_modified_utc="n/a (SWPC JSON, polled hourly)",
        units="OVATION intensity 0-30+ (display = source)",
        source_resolution="1-degree global OVATION grid, bilinear to 0.1 deg canvas",
        color_min=0.0, color_max=30.0, color_units="OVATION intensity",
        missing_data_treatment=("intensity < 1.0 transparent; faint cells "
                                "paint only poleward of |50° or attached "
                                "(within 10°) to genuinely active areas, so "
                                "isolated model background never renders as "
                                "aurora while storm expansions and faint oval "
                                "sections survive; full world equirectangular "
                                "canvas lon -180..180 / lat -90..90 "
                                "(3600x1800); missing source transparent, "
                                "never zero-filled."))
    meta["render_bounds"] = {"lon_min": -180.0, "lon_max": 180.0,
                             "lat_min": -90.0, "lat_max": 90.0}
    meta["spatial_resolution_rendered"] = (
        "3600x1800 world canvas over lon [-180,180], lat [-90,90] "
        "(0.1 deg/px bilinear from the 1-degree source grid)")
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["source_id"] = source_id
    meta["source_version"] = source_token(
        f"{source_id}-r{RENDER_VERSION}-aurglobal-tx5")
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
