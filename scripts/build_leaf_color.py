"""Pipeline G — LIVE LEAF COLOR (NASA HLS v2 replacement engine).

ONE continuous Great Lakes leaf-color gradient built from real NASA HLS
Landsat 8/9 (HLSL30 v2.0) + Sentinel-2 (HLSS30 v2.0) 30 m surface
reflectance via Microsoft Planetary Computer STAC.

Why HLS: the previous engine stitched three independent MODIS tiles.  HLS
places Landsat and Sentinel-2 on one harmonized 30 m MGRS grid, and this
builder composites every scene into ONE basin canvas before deriving a
single phenology field.  There are no per-tile or per-lake gradients.

Preserved product contract (deliberately unchanged):
  * the same 1800x1175 canvas / geographic bounds,
  * the same 19-anchor OKLab circular LUT (deep blue -> spring -> green ->
    yellow/orange/red -> purple -> deep blue),
  * the same legend labels, phase positions, scale 0..1 and overlay alpha.

Freshness: the newest clear HLS observation wins at each pixel inside a
rolling lookback window.  The layer rides the existing weekly cadence;
NASA HLS latency is about two days, so every weekly run uses the newest
available scenes.

Exit codes: 0 = updated (or skipped, source unchanged); 2 = source or
validation failure (previous valid raster kept); 1 = unexpected.
"""

import datetime as dt
import hashlib
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       child_description_html, description_html,
                       entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR, base_metadata,
                              load_bounds, now_det_str, promote_stage,
                              read_state, save_png, source_token, stage_dir,
                              write_metadata, write_state)
from leaf_hls import (blend_tile_seams, mosaic_scenes, search_items,
                      select_scenes)
from leaf_phenology import build_leaf_lut, draw_leaf_legend

PRODUCT = "leaf_color"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
STATE_DIR = os.path.join(REPO_ROOT, "output", "state", PRODUCT)
KML_FILE = "Great_Lakes_Live_Leaf_Color.kml"
OVERLAY_NAME = "\U0001F342 LIVE LEAF COLOR"
SKIP_NOTE = "Turn on/off independently of all other layers."
HIST_SHAPE = (294, 450)  # quarter-res history grids
ENGINE = "hls-v2"


def main():
    try:
        return run()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    bounds = load_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    lookback = int(CONFIG.get("lookback_days", 45))
    max_scenes = int(CONFIG.get("max_scenes_per_tile", 6))
    max_cloud = float(CONFIG.get("max_cloud_cover", 90.0))
    try:
        items = search_items(bounds, lookback_days=lookback)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    if not items:
        print(f"[{PRODUCT}] VALIDATION FAILED: no HLS items in the "
              f"{lookback}-day window. Keeping previous.")
        return 2
    scenes = select_scenes(items, max_per_tile=max_scenes,
                           max_cloud=max_cloud)
    if not scenes:
        print(f"[{PRODUCT}] VALIDATION FAILED: no usable HLS scenes. "
              f"Keeping previous.")
        return 2
    sig = hashlib.sha256(
        ("|".join(sorted(i.id for i in scenes))).encode()).hexdigest()
    prev = read_state(PRODUCT)
    if prev.get("composite_sig") == sig \
            and prev.get("render_version") == RENDER_VERSION \
            and prev.get("engine") == ENGINE \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE)):
        print(f"[{PRODUCT}] HLS scenes unchanged; keeping current raster.")
        try:
            refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                                 CONFIG["title"], SKIP_NOTE,
                                 CONFIG["refresh_interval_seconds"])
            print(f"[{PRODUCT}] KML base URLs refreshed.")
        except Exception as e:
            print(f"[{PRODUCT}] WARNING: KML refresh failed: {e}")
        return 0
    try:
        return _build(bounds, W, H, scenes, sig, lookback)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(bounds, W, H, scenes, sig, lookback):
    from PIL import Image
    from geospatial_utils import apply_shoreline_mask, load_watermask

    lc = np.array(Image.open(os.path.join(REPO_ROOT, "assets",
                                          "leaf_landcover.png")).convert("L"))
    if lc.shape != (H, W):
        raise ValueError(f"landcover shape {lc.shape} != canvas")
    land = load_watermask() < 0.5
    veg = landing_target(lc, land)

    mosaic = mosaic_scenes(scenes, veg, bounds,
                           epochs=int(CONFIG.get("epochs", 4)))
    ndvi = mosaic["ndvi"]
    redness = mosaic["redness"]
    ndvi, redness = blend_tile_seams(ndvi, redness, mosaic["tile_grid"])

    n_veg = int(veg.sum())
    n_obs = int((mosaic["observed"] & veg).sum())
    if n_obs < 50_000:
        raise ValueError(f"too few clear HLS observations ({n_obs})")
    print(f"[{PRODUCT}] scenes selected={mosaic['stats']['scenes_selected']} "
          f"used={mosaic['stats']['scenes_used']} "
          f"tiles={mosaic['stats']['tiles_used']}")
    print(f"[{PRODUCT}] observed vegetated pixels={n_obs}/{n_veg} "
          f"({100.0 * n_obs / max(n_veg, 1):.2f}%)")

    # The NDVI trajectory comes from the same HLS pass: the selected scenes
    # are sliced into equal time bins, so the rise/fall rules compare this
    # week's 30 m field against HLS observations from the same lookback
    # window.  No MODIS-era history is mixed into a 30 m HLS phase.
    epochs = np.asarray(mosaic["ndvi_epochs"], dtype=np.float32)
    hist = {"ndvi": np.stack([
        downsample_quarter(np.where(np.isfinite(e), e, np.nan))
        for e in epochs]).astype(np.float32)}
    print(f"[{PRODUCT}] HLS epoch coverage (vegetated px)="
          f"{mosaic['stats'].get('epoch_pixels', [])}")

    snow = mosaic["snow_seen"] & veg & ~mosaic["observed"]
    bad = ~np.isfinite(ndvi)
    marginal = np.zeros((H, W), dtype=bool)
    phase = compute_phase_grid(ndvi, lc, redness, snow, bad, marginal, hist)
    # Snow-covered vegetation is dormant winter (the circular scale's deep
    # blue end), not a hole.  This keeps winter coverage continuous.
    phase = np.where(snow & veg, 0.02, phase)
    phase_obs = np.array(phase, dtype=float)

    prev_phase = load_prev_phase(W, H)
    phase = fill_phase(phase, veg, prev_phase, dormant=0.02)
    coverage = float((np.isfinite(phase) & veg).sum()) / max(n_veg, 1)
    print(f"[{PRODUCT}] final vegetated coverage={coverage * 100.0:.2f}%")
    if coverage < 0.999:
        raise ValueError(f"incomplete vegetation coverage ({coverage:.4f})")

    lut = np.array(build_leaf_lut(), dtype=np.uint8)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    ok = veg & np.isfinite(phase)
    rgba[ok, 0:3] = lut[np.clip((phase[ok] * 255).astype(int), 0, 255)]
    rgba[ok, 3] = bounds["overlay_alpha"]
    rgba = apply_shoreline_mask(rgba, invert=True)  # leaf grows on LAND

    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    print(f"[{PRODUCT}] opaque pixels={n_opaque}")
    if n_opaque < 50_000:
        raise ValueError("too little vegetation after shoreline mask")
    n_colors = len(np.unique(
        rgba[(rgba[:, :, 3] > 0)][..., :3].reshape(-1, 3), axis=0))
    print(f"[{PRODUCT}] distinct raster colors={n_colors}")

    comp_date = mosaic["stats"]["composite_date"]
    today = dt.datetime.now(dt.timezone.utc).date()
    age = (today - comp_date).days if comp_date else None
    stale = age is not None and age > CONFIG["stale_threshold_days"]
    subtitle = CONFIG["subtitle"]
    if age is not None:
        subtitle += f"  |  Composite: {comp_date} ({age}d old)"
    if stale:
        subtitle += "  |  STALE SOURCE"
    lw, lh = draw_leaf_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        f"Source: NASA HLS Landsat 8/9 + Sentinel-2 via Planetary Computer  "
        f"|  Processed {now_det_str()}", build_leaf_lut())

    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=(f"HLS newest-clear observation {comp_date} "
                       f"({lookback}-day rolling window); "
                       f"retrieved {now_det_str()}"),
        source_last_modified_utc=f"mosaic sig {sig[:12]}",
        units="phenology phase 0..1 (display color); source HLS NDVI/reflectance",
        source_resolution="30 m NASA HLS (MGRS/UTM), reprojected to canvas",
        color_min=0.0, color_max=1.0, color_units="phenology phase (circular)",
        missing_data_treatment=(
            "ONE continuous basin-wide HLS mosaic.  At every vegetated land "
            "pixel the newest clear Landsat/Sentinel-2 observation in the "
            "rolling window wins; persistent cloud/ocean-adjacent gaps fall "
            "back to the most recent valid observation, and snow-covered "
            "vegetation renders as dormant winter deep blue.  Cities, bare "
            "ground, non-vegetated land and open Great Lakes water stay "
            "transparent because leaves do not grow there."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = (
        "What each color means (one trip around the year, left to right on "
        "the key): <b>deep blue</b> = dormant winter; <b>light "
        "blue/cyan</b> = spring awakening and bud break; "
        "<b>teal-green</b> = leaf emergence; <b>green</b> = growing season; "
        "<b>yellow-green/yellow</b> = first color; <b>orange</b> = autumn "
        "coloring; <b>red/magenta</b> = peak color into leaf drop; "
        "<b>purple</b> = late drop returning to <b>deep-blue</b> dormant "
        "(the scale wraps: both ends are winter). Color is the "
        "satellite-derived growth stage, not the color of individual "
        "trees. Every vegetated land pixel is painted from the newest clear "
        "HLS observation in the rolling window; persistent gaps fall back "
        "to the most recent valid observation. <b>Transparent = cities, "
        "bare ground, open water, or non-vegetated land</b>.")
    meta["data_nature"] = CONFIG["data_nature"]
    meta["gradient"] = {"interpolation": "OKLab (perceptually uniform)",
                        "anchors": 19, "distinct_raster_colors": n_colors,
                        "wraparound": "first == last deep blue #123B73"}
    meta["provenance"] = {
        "provider": ("NASA HLS v2.0 (Landsat 8/9 + Sentinel-2) via "
                     "Microsoft Planetary Computer STAC"),
        "collections": ["hls2-l30", "hls2-s30"],
        "items": mosaic["item_map"],
        "selected_scene_count": mosaic["stats"]["scenes_selected"],
        "used_scene_count": mosaic["stats"]["scenes_used"],
        "composite_sig": sig,
        "composite_date": str(comp_date),
        "source_age_days": age, "stale": stale,
        "retrieved": now_det_str(),
        "native_resolution_m": 30,
        "native_crs": "NASA HLS MGRS / UTM",
        "resampling": ("average surface reflectance, nearest Fmask, into "
                       "the shared 1800x1175 EPSG:4326 canvas"),
        "qa_rules": ("HLS v2 Fmask: cloud (bit 1), adjacent to cloud "
                     "(bit 2), cloud shadow (bit 3), snow/ice (bit 4), "
                     "water (bit 5), aerosol level (bits 6-7); reflectance "
                     "fill -9999 rejected"),
        "trajectory": ("same-pass HLS epoch slices over the rolling "
                       "lookback window; no MODIS-era NDVI history"),
        "landcover": ("USGS NLCD 2021 + NRCan 2020 Land Cover of Canada "
                      "(NALCMS inputs) mosaic -> assets/leaf_landcover.png"),
        "water_mask": "assets/great_lakes_watermask.png (shared)",
        "algorithm": ("one HLS newest-clear temporal mosaic over the full "
                      "basin -> single leaf_phenology phase field -> "
                      "unchanged 19-anchor OKLab circular LUT; no per-tile "
                      "gradients"),
    }
    source_id = f"leaf-hls-v1-{sig[:12]}-r{RENDER_VERSION}"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["stats"] = {
        "observed_pixels": n_obs,
        "opaque_pixels": n_opaque,
        "vegetated_pixels": n_veg,
        "coverage_fraction": round(coverage, 6),
        "phase_mean": round(float(np.nanmean(phase)), 4),
    }
    token = meta["source_version"]
    scale_html = meta["legend_scale_html"]
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    child_html = child_description_html(
        block, CONFIG.get("data_nature", ""), CONFIG.get("display_units", ""),
        CONFIG["source_name"], CONFIG["source_url"],
        CONFIG["freshness_label"])
    meta["overlay_description"] = child_html
    write_metadata(stage_prod, meta)

    outs = live_out_dirs(stage, KML_FILE)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        out_dirs=outs["live"],
        overlay_descriptions=child_html,
        meta=meta)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    save_history(hist["ndvi"], phase_obs, mosaic["used_ids"], sig)
    save_prev_phase(phase, W, H)
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"composite_sig": sig,
                          "source_id": source_id,
                          "render_version": RENDER_VERSION,
                          "engine": ENGINE,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def landing_target(landcover, land):
    """Vegetated land classes that carry a leaf signal (1..6)."""
    return land & np.isin(landcover, [1, 2, 3, 4, 5, 6])


def fill_phase(phase, target, prev_phase=None, dormant=0.02):
    """Guarantee continuous vegetation coverage without inventing color.

    Priority: current HLS observation -> previous published phase for the
    same pixel -> nearest-valid interpolation for remaining gaps -> dormant
    winter blue as the final full-coverage backstop.  rasterio.fill is part
    of the existing rasterio dependency, so large cloud gaps fill from the
    closest valid phenology instead of becoming artificial blue blocks.
    """
    from rasterio.fill import fillnodata
    out = np.array(phase, dtype=float)
    if prev_phase is not None and prev_phase.shape == out.shape:
        use = target & ~np.isfinite(out) & np.isfinite(prev_phase)
        out[use] = prev_phase[use]
    known = np.isfinite(out)
    if (target & ~known).any() and known.any():
        seed = np.where(known, out, 0.0).astype(np.float64)
        filled = fillnodata(seed, mask=known.astype(np.uint8),
                            max_search_distance=10000,
                            smoothing_iterations=0)
        use = target & ~known
        out[use] = filled[use]
    out[target & ~np.isfinite(out)] = dormant
    return np.clip(out, 0.0, 0.999)


def downsample_quarter(a):
    """NaN-aware quarter-res block mean (used only by the history baseline)."""
    H, W = a.shape
    h2, w2 = HIST_SHAPE
    ys = (np.arange(h2) * H / h2).astype(int)
    xs = (np.arange(w2) * W / w2).astype(int)
    ye = np.clip(((np.arange(h2) + 1) * H / h2).astype(int), 0, H)
    xe = np.clip(((np.arange(w2) + 1) * W / w2).astype(int), 0, W)
    filled = np.where(np.isfinite(a), a, 0.0)
    mask = np.isfinite(a).astype(np.float64)
    ii = np.pad(filled, 1).cumsum(0).cumsum(1)
    im = np.pad(mask, 1).cumsum(0).cumsum(1)
    y0, y1 = ys[:, None], ye[:, None]
    x0, x1 = xs[None, :], xe[None, :]
    sums = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    cnts = im[y1, x1] - im[y0, x1] - im[y1, x0] + im[y0, x0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnts > 0, sums / np.maximum(cnts, 1e-9), np.nan)


def save_history(epoch_q, phase, item_ids, sig):
    """Persist the HLS lookback trajectory (quarter res) for diagnostics.

    The builder itself does not depend on saved NDVI history: each run
    derives its trajectory from the HLS scenes in the lookback window, so a
    source change can never mix sensor biases into the phenology rules.
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    q = np.asarray(epoch_q, dtype=np.float32)
    if q.ndim != 3 or q.shape[1:] != HIST_SHAPE:
        q = np.full((1,) + HIST_SHAPE, np.nan, dtype=np.float32)
    if q.shape[0] < HIST_SHAPE[0]:
        pad = np.full((HIST_SHAPE[0] - q.shape[0],) + HIST_SHAPE, np.nan,
                      dtype=np.float32)
        q = np.concatenate([pad, q], axis=0)
    q = q[-HIST_SHAPE[0]:]
    np.save(os.path.join(STATE_DIR, "hist_ndvi.npy"),
            np.clip(np.nan_to_num(q, nan=-1.28) * 100.0, -128, 127
                    ).astype(np.int8))
    np.save(os.path.join(STATE_DIR, "hist_phase.npy"),
            downsample_quarter(np.where(np.isfinite(phase), phase, np.nan)
                               ).astype(np.float32))
    json.dump({"dates": sorted(set(item_ids)), "sig": sig, "engine": ENGINE},
              open(os.path.join(STATE_DIR, "hist_meta.json"), "w"))


def load_prev_phase(W, H):
    """Full-resolution previous phase, used only to fill current gaps."""
    try:
        from PIL import Image
        p = os.path.join(STATE_DIR, "prev_phase.png")
        im = Image.open(p).convert("L")
        if im.size != (W, H):
            return None
        return np.asarray(im, dtype=np.float32) / 255.0 * 0.999
    except Exception:
        return None


def save_prev_phase(phase, W, H):
    from PIL import Image
    os.makedirs(STATE_DIR, exist_ok=True)
    arr = np.clip(np.nan_to_num(phase, nan=0.0) * 255.0, 0, 255
                  ).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(os.path.join(STATE_DIR, "prev_phase.png"))


def compute_phase_grid(ndvi, lc, redness, snow, bad, marginal, hist):
    """One vectorized phenology phase field for the whole basin canvas."""
    H, W = ndvi.shape
    qh, qw = HIST_SHAPE
    ys = np.clip((np.arange(H) * qh / H).astype(int), 0, qh - 1)
    xs = np.clip((np.arange(W) * qw / W).astype(int), 0, qw - 1)
    hn = hist["ndvi"]
    cur_q = downsample_quarter(np.where(np.isfinite(ndvi), ndvi, np.nan))
    stack = list(hn) + [cur_q]
    hcnt = sum(np.isfinite(s).astype(int) for s in stack[:-1])
    htot = sum(np.where(np.isfinite(s), s, 0.0) for s in stack[:-1])
    n_ref_q = np.where(hcnt > 0, htot / np.maximum(hcnt, 1), cur_q)
    n_max_q = np.maximum.reduce(
        [np.where(np.isfinite(s), s, -1) for s in stack])
    n_max_q = np.where(n_max_q < -0.5, np.nan, n_max_q)
    n_ref = n_ref_q[ys[:, None], xs]
    n_max = n_max_q[ys[:, None], xs]

    phase = np.full((H, W), np.nan)
    hb = bad | ~np.isfinite(ndvi)
    work = ~(hb | snow) & np.isfinite(ndvi)
    forest = np.isin(lc, [1, 2, 3])
    prior = np.where(forest, 0.72, 0.55)
    n_max = np.maximum(np.where(np.isnan(n_max), -1.0, n_max), prior)
    span = np.maximum(n_max - 0.10, 0.20)
    ngr = np.clip((ndvi - 0.10) / span, 0, 1)
    d = np.where(np.isfinite(n_ref), ndvi - n_ref, 0.0)
    red = np.clip(redness, 0, 1)

    m_dorm = work & (ngr <= 0.12) & (np.abs(d) <= 0.03) & (red < 0.3)
    phase[m_dorm] = 0.02 + 0.06 * (ngr[m_dorm] / 0.12)
    m_rise = work & ~m_dorm & (d > 0.02)
    phase[m_rise] = 0.15 + 0.30 * ngr[m_rise]
    m_fall = work & ~m_dorm & ((d <= -0.02) | (red > 0.5))
    t = np.maximum(1.0 - ngr[m_fall], red[m_fall])
    ph = 0.55 + 0.30 * t + 0.08 * red[m_fall] * t
    low = ngr[m_fall] < 0.22
    ph[low] = np.maximum(ph[low], 0.85 + 0.15 * (0.22 - ngr[m_fall][low]) / 0.22)
    phase[m_fall] = ph
    m_st = work & ~m_dorm & ~m_fall & (np.abs(d) <= 0.02)
    hi = m_st & (ngr > 0.6)
    phase[hi] = 0.45 + 0.10 * np.clip((ngr[hi] - 0.7) / 0.3, 0, 1)
    lo = m_st & (ngr <= 0.6)
    phase[lo] = 0.10 + 0.25 * (ngr[lo] / 0.6)

    m_ev = work & (lc == 3)
    phase[m_ev] = 0.40 + 0.16 * ngr[m_ev]
    for code, w, cap in ((2, 0.55, None), (4, 0.45, 0.66), (5, 0.45, 0.66),
                         (6, 0.35, 0.62)):
        m_c = work & (lc == code)
        phase[m_c] = 0.5 + (phase[m_c] - 0.5) * w
        if cap is not None:
            phase[m_c] = np.minimum(phase[m_c], cap)
    m_mask = (lc == 0) | (lc == 7) | (lc == 8) | (lc == 9)
    phase[m_mask] = np.nan
    phase[snow] = np.nan
    m_marg = work & np.asarray(marginal, dtype=bool) & np.isfinite(phase)
    phase[m_marg] = 0.5 + (phase[m_marg] - 0.5) * 0.6
    return np.clip(phase, 0.0, 0.999)


if __name__ == "__main__":
    sys.exit(main())
