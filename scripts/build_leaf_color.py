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
ENGINE = "hls-v5"


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
    print(f"[{PRODUCT}] HLS items found={len(items)} in {lookback}-day window")
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

    comp_date = mosaic["stats"]["composite_date"]
    today = dt.datetime.now(dt.timezone.utc).date()
    if os.environ.get("LEAF_DEBUG_DUMP"):
        np.savez_compressed(
            os.environ["LEAF_DEBUG_DUMP"],
            ndvi=mosaic["ndvi"], redness=mosaic["redness"],
            obs_day=mosaic["obs_day"], epochs=mosaic["ndvi_epochs"],
            redness_epochs=mosaic["redness_epochs"],
            snow_seen=mosaic["snow_seen"], veg=veg, lc=lc,
            comp_date=np.array(str(comp_date)))

    # A pixel whose newest clear overpass is older than the freshness cap is
    # not allowed to paint the current map; it becomes a gap and fills from
    # this week's neighbours.  This is what keeps the layer from showing an
    # October canopy over a November landscape.
    max_age = int(CONFIG.get("max_obs_age_days", 21))
    age_days = today.toordinal() - mosaic["obs_day"]
    stale_obs = (mosaic["obs_day"] > 0) & (age_days > max_age)
    if stale_obs.any():
        ndvi[stale_obs] = np.nan
        redness[stale_obs] = np.nan
        mosaic["observed"][stale_obs] = False

    n_veg = int(veg.sum())
    n_obs = int((mosaic["observed"] & veg).sum())
    if n_obs < 50_000:
        raise ValueError(f"too few clear HLS observations ({n_obs})")
    print(f"[{PRODUCT}] scenes selected={mosaic['stats']['scenes_selected']} "
          f"used={mosaic['stats']['scenes_used']} "
          f"tiles={mosaic['stats']['tiles_used']}")
    print(f"[{PRODUCT}] observed vegetated pixels={n_obs}/{n_veg} "
          f"({100.0 * n_obs / max(n_veg, 1):.2f}%) "
          f"(stale>{max_age}d dropped={int((stale_obs & veg).sum())})")

    # The four HLS epoch slices are the observed trajectory: the phase is
    # derived from how the canopy is actually changing right now, not from
    # the calendar.
    epochs = np.asarray(mosaic["ndvi_epochs"], dtype=np.float32)
    hist = {"ndvi": np.stack([
        downsample_quarter(np.where(np.isfinite(e), e, np.nan))
        for e in epochs]).astype(np.float32)}
    print(f"[{PRODUCT}] HLS epoch coverage (vegetated px)="
          f"{mosaic['stats'].get('epoch_pixels', [])}")

    snow = mosaic["snow_seen"] & veg & ~mosaic["observed"]
    phase, details = compute_phase_grid(ndvi, lc, redness, snow, epochs,
                                        return_details=True)
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
                       f"(observations <= {max_age} days old; "
                       f"{lookback}-day search window); "
                       f"retrieved {now_det_str()}"),
        source_last_modified_utc=f"mosaic sig {sig[:12]}",
        units="phenology phase 0..1 (display color); source HLS NDVI/reflectance",
        source_resolution="30 m NASA HLS (MGRS/UTM), reprojected to canvas",
        color_min=0.0, color_max=1.0, color_units="phenology phase (circular)",
        missing_data_treatment=(
            "ONE continuous basin-wide HLS mosaic.  At every vegetated land "
            f"pixel the newest clear Landsat/Sentinel-2 observation (no older "
            f"than {max_age} days) wins; older or cloud-covered gaps fill "
            "from this week's neighbours, then the previous phase, then "
            "dormant winter deep blue.  Snow-covered vegetation renders as "
            "dormant winter blue.  Cities, bare ground, non-vegetated land "
            "and open Great Lakes water stay transparent because leaves do "
            "not grow there."))
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
        "trajectory": ("observed HLS state: absolute newest-clear NDVI as a "
                       "fraction of each land-cover class's summer canopy "
                       "potential sets greenness, the measured NDVI change "
                       "across the four epoch slices confirms green-up or "
                       "suppresses it, and red foliage chromaticity gated by "
                       "canopy sets senescence colour; no calendar/seasonal "
                       "clock and no MODIS-era NDVI"),
        "freshness_cap_days": max_age,
        "calibration": calibration_report(
            ndvi, redness, details["trend"], phase_obs,
            details["observed"]),
        "landcover": ("USGS NLCD 2021 + NRCan 2020 Land Cover of Canada "
                      "(NALCMS inputs) mosaic -> assets/leaf_landcover.png"),
        "water_mask": "assets/great_lakes_watermask.png (shared)",
        "algorithm": ("one HLS newest-clear temporal mosaic over the full "
                      "basin -> spatially despiked NDVI/redness -> NDVI as a "
                      "fraction of the land-cover class summer potential + "
                      "observed multi-epoch NDVI change + red chromaticity "
                      "-> single continuous phase field -> unchanged "
                      "19-anchor OKLab circular LUT; no per-tile gradients"),
    }
    source_id = f"leaf-hls-v3-{sig[:12]}-r{RENDER_VERSION}"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["stats"] = {
        "observed_pixels": n_obs,
        "opaque_pixels": n_opaque,
        "vegetated_pixels": n_veg,
        "coverage_fraction": round(coverage, 6),
        "phase_mean": round(float(np.nanmean(phase)), 4),
        "stale_dropped_pixels": int((stale_obs & veg).sum()),
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

    Priority: current HLS observation -> nearest-valid interpolation from
    this week's field -> previous published phase -> dormant winter blue as
    the final full-coverage backstop.  Nearest-valid comes first so a stale
    green phase from an older run cannot leak into this week's gaps.
    rasterio.fill is part of the existing rasterio dependency, so large
    cloud gaps fill from the closest valid phenology instead of becoming
    artificial blue blocks.
    """
    from rasterio.fill import fillnodata
    out = np.array(phase, dtype=float)
    known = np.isfinite(out)
    if (target & ~known).any() and known.any():
        seed = np.where(known, out, 0.0).astype(np.float64)
        filled = fillnodata(seed, mask=known.astype(np.uint8),
                            max_search_distance=10000,
                            smoothing_iterations=0)
        use = target & ~known
        out[use] = filled[use]
        known = np.isfinite(out)
    if prev_phase is not None and prev_phase.shape == out.shape:
        use = target & ~known & np.isfinite(prev_phase)
        out[use] = prev_phase[use]
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
    # Keep the HLS trajectory in its own file.  The committed hist_ndvi.npy
    # is the legacy MODIS-era record and must not be overwritten.
    np.save(os.path.join(STATE_DIR, "hist_ndvi_hls.npy"),
            np.clip(np.nan_to_num(q, nan=-1.28) * 100.0, -128, 127
                    ).astype(np.int8))
    np.save(os.path.join(STATE_DIR, "hist_phase.npy"),
            downsample_quarter(np.where(np.isfinite(phase), phase, np.nan)
                               ).astype(np.float32))
    json.dump({"dates": sorted(set(item_ids)), "sig": sig, "engine": ENGINE},
              open(os.path.join(STATE_DIR, "hist_meta_hls.json"), "w"))


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


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / float(hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# Physical calibration that maps the observed HLS fields onto the fixed
# 19-anchor gradient.  These are optical constants (how green a canopy reads,
# how red senescing foliage reads), NOT calendar terms: the same NDVI and
# redness always map to the same phase regardless of the date.
CALIB = {
    "ndvi_bare": 0.10,      # NDVI of bare/dormant vegetated ground
    "canopy_lo": 0.32,      # canopy fraction where the green band starts
    "canopy_hi": 0.85,      # canopy fraction of peak green
    "red_gate_lo": 0.10,    # canopy fraction where red foliage starts counting
    "red_gate_hi": 0.45,    # canopy fraction that fully admits red foliage
    "spring_lo": 0.02,      # NDVI rise below this is compositing noise
    "spring_full": 0.12,    # NDVI rise over the window = full green-up
    "decline_green": 0.55,  # how much an active decline suppresses greenness
    "red_onset": 0.28,      # red chromaticity where senescence color starts
    "red_full": 0.62,       # red chromaticity of peak autumn color
    "autumn_floor": 0.58,   # phase of a senescing pixel with no red signal
    "autumn_max": 0.95,     # phase reached at a full red-foliage signal
    "trend_lo": 0.02,       # NDVI drop below this is compositing noise
    "trend_full": 0.12,     # NDVI change over the window = full direction
}

# Summer canopy potential by land-cover class: the NDVI a fully-leafed canopy
# of that class reaches, used to convert absolute NDVI into a canopy fraction.
# This is static land cover, not a clock.  A 45-day lookback never contains the
# summer peak, so normalizing against the window's own peak made mid-autumn
# NDVI look like full canopy; the class potential does not have that failure.
PRIOR = np.array([0.0, 0.82, 0.78, 0.65, 0.70, 0.72, 0.82, 0.0, 0.0, 0.0],
                 dtype=np.float32)


def _q(a, p):
    """Finite-only percentile (0..100), NaN when nothing is observed."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        return None
    return round(float(np.percentile(v, p)), 4)


def calibration_report(ndvi, redness, trend, phase, observed):
    """Observed distribution behind this run's data->gradient mapping.

    Written into metadata so the calibration is auditable: the constants
    above are optical, while these percentiles show where this week's
    observations actually landed on the scale.
    """
    nd = ndvi[observed]
    rd = redness[observed]
    tr = trend[observed]
    ph = phase[observed]
    return {
        "constants": dict(CALIB),
        "observed": {
            "ndvi_p05": _q(nd, 5), "ndvi_p50": _q(nd, 50),
            "ndvi_p95": _q(nd, 95),
            "redness_p50": _q(rd, 50), "redness_p90": _q(rd, 90),
            "trend_p10": _q(tr, 10), "trend_p50": _q(tr, 50),
            "trend_p90": _q(tr, 90),
            "phase_p10": _q(ph, 10), "phase_p50": _q(ph, 50),
            "phase_p90": _q(ph, 90),
        },
        "mapping": ("absolute NDVI as a fraction of each land-cover class's "
                    "summer canopy potential -> greenness (a mid-autumn NDVI "
                    "is a partial canopy, not full green); observed NDVI rise "
                    "confirms green-up, observed decline suppresses it; red "
                    "chromaticity gated by canopy -> senescence color; no "
                    "calendar/seasonal clock"),
    }


def _nanmedian3(a, passes=1):
    """NaN-aware 3x3 median, repeated to suppress compositing speckle.

    Multi-date HLS pixels leave isolated 1-2 px reflectance jumps.  A median
    removes them without inventing values or blurring real land-cover edges.
    """
    out = np.asarray(a, dtype=np.float32)
    for _ in range(max(1, int(passes))):
        H, W = out.shape
        p = np.pad(out, 1, mode="constant", constant_values=np.nan)
        stack = np.stack([
            p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.nanmedian(stack, axis=0).astype(np.float32)
    return out


def _epoch_trend(epochs):
    """Observed NDVI change, newest slice vs the two preceding slices.

    The four epoch slices are equal time steps across the lookback window.
    Comparing the newest slice with the immediately preceding ones measures
    what the canopy is doing NOW: greening (spring) or browning (autumn).
    A pixel that finished dropping a month ago reads ~0 here and therefore
    renders dormant instead of being painted as an active autumn.
    """
    arr = np.asarray(epochs, dtype=np.float32)
    if arr.ndim == 2:
        return np.zeros(arr.shape, dtype=np.float32)
    E = arr.shape[0]
    if E < 2:
        return np.zeros(arr.shape[1:], dtype=np.float32)
    has = np.isfinite(arr)
    idx = np.arange(E, dtype=np.float32)[:, None, None]
    newest = np.where(has, idx, -1.0).max(axis=0)
    before = has & (idx < newest[None])
    p1 = np.where(before, idx, -1.0).max(axis=0)
    before2 = before & (idx < p1[None])
    p2 = np.where(before2, idx, -1.0).max(axis=0)

    def _at(ix):
        return np.take_along_axis(
            arr, np.clip(ix, 0, E - 1).astype(np.int64)[None], axis=0)[0]

    y_last = _at(newest)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y_prev = np.nanmean(np.stack([_at(p1), _at(p2)]), axis=0)
    trend = np.where((newest >= 0) & (p1 >= 0), y_last - y_prev, 0.0)
    return np.where(np.isfinite(trend), trend, 0.0).astype(np.float32)


def compute_phase_grid(ndvi, lc, redness, snow, epochs, calib=None,
                       return_details=False):
    """One continuous, observation-driven phenology field for the basin.

    Nothing here uses the calendar or a per-pixel seasonal maximum.  Each
    pixel is placed on the circular phase scale from what the satellite
    actually saw:

      * absolute NDVI as a fraction of its land-cover class's summer
        canopy potential -> greenness.  A 45-day window never contains the
        summer peak, so normalizing by the window's own peak made every
        mid-autumn pixel look fully green; the static class potential does
        not have that failure.  Mid-autumn NDVI is a partial canopy.
      * observed NDVI rise across the epochs confirms green-up; observed
        decline suppresses greenness (a falling canopy is not peak green).
      * red foliage chromaticity (bare soil already maps to ~0 in the
        redness index) gated by canopy presence -> autumn colour.  A
        canopy that already dropped reads dormant, not active autumn.

    The formula is continuous everywhere, so adjacent pixels cannot jump
    between green and red (no rise/fall branches, no block artifacts).

    `calib` overrides CALIB (physical data->gradient calibration).
    """
    c = dict(CALIB)
    if calib:
        c.update(calib)
    nd = _nanmedian3(ndvi, passes=2)
    rd = _nanmedian3(redness, passes=2)
    ep = np.asarray(epochs, dtype=np.float32)
    if ep.ndim == 3:
        ep = np.stack([_nanmedian3(e, passes=1) for e in ep])
    observed = np.isfinite(ndvi) & ~snow
    nd = np.where(observed, nd, np.nan)
    rd = np.where(observed, rd, np.nan)
    trend = _epoch_trend(ep)

    # 1. Canopy fraction from absolute NDVI against the class potential.
    #    Mid-autumn NDVI is a *partial* canopy, not full green.
    prior = PRIOR[np.clip(lc, 0, 9)]
    frac = np.clip((nd - c["ndvi_bare"]) /
                   np.maximum(prior - c["ndvi_bare"], 0.05), 0.0, 1.0)
    green = _smoothstep(frac, c["canopy_lo"], c["canopy_hi"])
    # An observed NDVI rise confirms green-up, but only where the canopy
    # fraction is already meaningful: sparse early-spring cover must not
    # read as peak green just because it is rising.
    spring = _smoothstep(trend, c["spring_lo"], c["spring_full"]) * \
        _smoothstep(frac, 0.30, 0.70)
    green = green + spring * (1.0 - green)
    # A canopy that is actively losing NDVI is not peak green even if its
    # absolute NDVI is still moderate.  This is what keeps late-autumn
    # deciduous forest out of the summer-green band.
    decline = _smoothstep(-trend, c["trend_lo"], c["trend_full"])
    green = green * (1.0 - c["decline_green"] * decline)
    phase_green = 0.50 * green

    # 2. Senescence colour.  The redness index is red chromaticity
    #    re-centred so bare soil is ~0; gating it by canopy keeps brown
    #    fields blue while letting red foliage colour.  Only observed red
    #    foliage produces autumn colour: a bare canopy that already
    #    dropped is dormant, not an active autumn.
    red = np.clip(np.nan_to_num(rd, nan=0.0), 0.0, 1.0)
    canopy_gate = _smoothstep(frac, c["red_gate_lo"], c["red_gate_hi"])
    red_norm = _smoothstep(red, c["red_onset"], c["red_full"])
    sen = np.clip(red_norm * canopy_gate, 0.0, 1.0)

    # Autumn targets yellow at the onset of senescence and deep red/purple
    # at full red foliage.  Yellow -> orange -> red as red rises.
    phase_autumn = c["autumn_floor"] + \
        (c["autumn_max"] - c["autumn_floor"]) * red_norm
    phase = phase_green + sen * (phase_autumn - phase_green)
    phase = np.where(observed, phase, np.nan)

    # Conifers keep a green band all year; a 0.35 NDVI conifer stand is not
    # dormant just because it is late autumn.
    ev = observed & (lc == 3)
    phase[ev] = 0.40 + 0.16 * green[ev]

    for code, w, cap in ((2, 0.55, None), (4, 0.45, 0.66), (5, 0.45, 0.66),
                         (6, 0.35, 0.62)):
        m_c = observed & (lc == code)
        # Subdue only the autumn excursion for mixed/shrub/grass/crop.
        # Dormancy must stay deep blue on every class: pulling the low end
        # toward 0.5 would paint dormant grassland teal in winter.
        phase[m_c] = np.where(phase[m_c] > 0.5,
                              0.5 + (phase[m_c] - 0.5) * w,
                              phase[m_c])
        if cap is not None:
            phase[m_c] = np.minimum(phase[m_c], cap)
    m_mask = (lc == 0) | (lc == 7) | (lc == 8) | (lc == 9)
    phase[m_mask] = np.nan
    phase[snow] = np.nan
    # Final single median pass: multi-date compositing leaves isolated
    # 1-2 px colour jumps (salt-and-pepper).  A median removes them without
    # flattening real field/land-cover edges or moving the values.
    med = _nanmedian3(phase, passes=1)
    phase = np.where(np.isfinite(phase) & np.isfinite(med), med, phase)
    phase = np.clip(phase, 0.0, 0.999)
    if return_details:
        return phase, {"trend": trend, "green": green,
                       "observed": observed}
    return phase


if __name__ == "__main__":
    sys.exit(main())
