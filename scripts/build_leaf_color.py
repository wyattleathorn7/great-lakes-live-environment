"""Pipeline G — LIVE LEAF COLOR (independent).

NASA MODIS Aqua global 500 m composites via Planetary Computer STAC
(no credentials; anonymous SAS) — the no-auth global VIIRS-class stream:
  MYD13A1.061: NDVI + pixel reliability (16-day; same VI family/scale as
    the VIIRS VNP13 series, global incl. Canada, QA'd). VNP13A4N itself
    requires Earthdata credentials unavailable to automation (see
    DATA_SOURCES.md for the deviation rationale + upgrade path).
  MYD09A1.061: surface reflectance red/green/blue/SWIR + state QA (8-day):
    autumn redness proxy + NDSI snow (no separate snow source needed).
Land classes: committed US-NLCD + Canada-NALCMS mosaic
  (assets/leaf_landcover.png; static ancillary, documented).
Michigan mask: committed assets/michigan_mask.png (authoritative state
boundary, both peninsulas). Water: shared mask (lakes transparent).

Per-pixel phenology phase (leaf_phenology.py) from NDVI trajectory +
baseline + direction + class + spectral gating; circular continuous LUT;
rolling quarter-res history in output/state/leaf_color/ for baseline,
direction and bad-observation hold-forward.

Exit codes: 0 = updated (or skipped, composites unchanged); 2 = source/
validation failure (previous valid raster left untouched); 1 = unexpected.
"""

import datetime as dt
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
from geospatial_utils import (REPO_ROOT, SITE_DIR, base_metadata,
                              download, load_bounds, promote_stage,
                              read_state, save_png, source_token, stage_dir,
                              utcnow_iso, write_metadata, write_state)
from leaf_phenology import (build_leaf_lut, draw_leaf_legend)

PRODUCT = "leaf_color"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
STATE_DIR = os.path.join(REPO_ROOT, "output", "state", PRODUCT)
KML_FILE = "Great_Lakes_Live_Leaf_Color.kml"
OVERLAY_NAME = "\U0001F342 LIVE LEAF COLOR"
SKIP_NOTE = "Turn on/off independently of all other layers."
TILES = ["h11v04", "h12v04", "h13v04"]
HIST_SHAPE = (294, 450)  # quarter-res history grids

LEAF_LUT = None


def latest_items(collection, days_back=120):
    """Newest STAC item per MODIS tile (pystac objects).

    Server-side SORT is not honored, so all items are collected and the
    newest composite per tile is selected client-side by start date.
    """
    import pystac_client
    end = dt.datetime.now(dt.timezone.utc).date()
    start = end - dt.timedelta(days=days_back)
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1")
    search = cat.search(
        collections=[collection], bbox=[-93, 40.5, -73.5, 49.5],
        datetime=f"{start}/{end}", limit=100)
    best = {}
    for f in search.items():
        tid = None
        for part in f.id.split("."):
            if part.startswith("h") and "v" in part:
                tid = part
        if tid not in TILES:
            continue
        day = composite_start_day(f.id)
        key = (day or dt.date(2000, 1, 1), f.id)
        if tid not in best or key > best[tid][0]:
            best[tid] = (key, f)
    return {t: f for t, (k, f) in best.items()}


def sign_item(item):
    """Attach anonymous SAS hrefs (proven PC flow)."""
    import planetary_computer as pc
    return pc.sign(item)


def composite_start_day(item_id):
    """A2026217 -> date(2026, 8, 5)."""
    import re
    m = re.search(r"\.A(\d{4})(\d{3})\.", item_id)
    if not m:
        return None
    return dt.date(int(m.group(1)), 1, 1) + dt.timedelta(int(m.group(2)) - 1)


def pick_asset(item, *cands):
    """(key, pystac Asset) by case-insensitive substring; (None, None)."""
    lower = {k.lower(): (k, v) for k, v in item.assets.items()}
    for c in cands:
        for k, (ok, v) in lower.items():
            if c in k:
                return ok, v
    return None, None


def read_window(signed_href, out_shape, resampling):
    """Read a COG's canvas window via rasterio WarpedVRT (no full download).

    WarpedVRT forbids boundless reads, so the window is clipped to the
    raster and pasted into a NaN canvas at the matching offset.
    """
    import rasterio
    from rasterio.enums import Resampling as RS
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds
    import numpy as _np
    b = load_bounds()
    rs = {"nearest": RS.nearest, "bilinear": RS.bilinear}[resampling]
    H, W = out_shape
    canvas = _np.full((H, W), _np.nan)
    with rasterio.open(signed_href) as src:
        with WarpedVRT(src, crs="EPSG:4326") as vrt:
            win = from_bounds(b["lon_min"], b["lat_min"], b["lon_max"],
                              b["lat_max"], vrt.transform)
            r0, r1 = max(int(win.row_off), 0), min(
                int(win.row_off + win.height), vrt.height)
            c0, c1 = max(int(win.col_off), 0), min(
                int(win.col_off + win.width), vrt.width)
            if r1 <= r0 or c1 <= c0:
                return canvas, dict(vrt.tags()), src.nodata
            from rasterio.windows import Window
            sub = vrt.read(1, window=Window(c0, r0, c1 - c0, r1 - r0),
                           out_shape=(
                               max(1, int((r1 - r0) / win.height * H)),
                               max(1, int((c1 - c0) / win.width * W))),
                           resampling=rs)
            # place sub into canvas at proportional offset
            or0 = int(round((r0 - win.row_off) / win.height * H))
            oc0 = int(round((c0 - win.col_off) / win.width * W))
            or1, oc1 = or0 + sub.shape[0], oc0 + sub.shape[1]
            cr0, cr1 = max(or0, 0), min(or1, H)
            cc0, cc1 = max(oc0, 0), min(oc1, W)
            canvas[cr0:cr1, cc0:cc1] = sub[cr0 - or0:cr1 - or0,
                                           cc0 - oc0:cc1 - oc0]
            return canvas, dict(vrt.tags()), src.nodata


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    bounds = load_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    try:
        vi = latest_items("modis-13A1-061")
        rf = latest_items("modis-09A1-061")
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    if not vi or not rf:
        print(f"[{PRODUCT}] VALIDATION FAILED: no STAC items "
              f"(vi={len(vi)} refl={len(rf)}). Keeping previous.")
        return 2
    # Full-coverage mosaic rule: Michigan spans three MODIS tiles. A
    # missing tile is a REGIONAL HOLE, never silently skipped -- fail the
    # run and keep the previous valid raster instead.
    missing_vi = [t for t in TILES if t not in vi]
    missing_rf = [t for t in TILES if t not in rf]
    if missing_vi or missing_rf:
        print(f"[{PRODUCT}] VALIDATION FAILED: incomplete tile mosaic "
              f"(missing VI {missing_vi} reflectance {missing_rf}). "
              f"Keeping previous.")
        return 2
    sig = hashlib.sha256(
        ("|".join(sorted(i.id for i in list(vi.values()) + list(rf.values())))
         ).encode()).hexdigest()
    source_id_hint = f"leaf-v3-{sig[:12]}"
    prev = read_state(PRODUCT)
    if prev.get("source_id") == source_id_hint \
            and prev.get("composite_sig") == sig \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(
                SITE_DIR, "kml", "live", KML_FILE)):
        print(f"[{PRODUCT}] composites unchanged; keeping current raster.")
        try:
            refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                                 CONFIG["title"], SKIP_NOTE,
                                 CONFIG["refresh_interval_seconds"])
            print(f"[{PRODUCT}] KML base URLs refreshed.")
        except Exception as e:
            print(f"[{PRODUCT}] WARNING: KML refresh failed: {e}")
        return 0
    try:
        return _build(bounds, W, H, vi, rf, sig)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _mosaic(pairs):
    """First-valid-wins mosaic over (array, fill_value) reads.

    Fill codes (which are finite numbers like 65535/-28672) are forced to
    NaN first — otherwise the first tile's fill would shadow later tiles'
    valid data and poison QA bit tests.
    """
    acc = None
    for a, f in pairs:
        a = np.asarray(a, dtype=float)
        if f is not None:
            try:
                a = np.where(a == float(f), np.nan, a)
            except (TypeError, ValueError):
                pass
        if acc is None:
            acc = a
        else:
            acc = np.where(np.isfinite(acc), acc,
                           np.where(np.isfinite(a), a, np.nan))
    return acc


def _build(bounds, W, H, vi, rf, sig):
    from PIL import Image
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)

    tok_vi = sign_item(next(iter(vi.values())))
    _ = tok_vi  # per-item signing below; this warms/validates credentials
    signed_vi = {t: sign_item(it) for t, it in vi.items()}
    signed_rf = {t: sign_item(it) for t, it in rf.items()}

    # per-tile reads (canvas window only); keep each SDS nodata for fill masking
    ndvi_r, rel_r, evi_r, comp_r = [], [], [], []
    b1_r, b3_r, b4_r, b6_r, st_r = [], [], [], [], []
    vi_ids, rf_ids = {}, {}
    for tid in TILES:
        if tid not in vi or tid not in rf:
            continue
        vitem, ritem = signed_vi[tid], signed_rf[tid]
        vi_ids[tid], rf_ids[tid] = vi[tid].id, rf[tid].id
        _nk, ndvi_a = pick_asset(vitem, "ndvi")
        _ek, evi_a = pick_asset(vitem, "evi")
        _rk, rel_a = pick_asset(vitem, "reliability")
        b1k, b1_a = pick_asset(ritem, "b01", "band01", "red")
        b3k, b3_a = pick_asset(ritem, "b03", "band03", "blue")
        b4k, b4_a = pick_asset(ritem, "b04", "band04", "green")
        b6k, b6_a = pick_asset(ritem, "b06", "band06", "swir")
        stk, st_a = pick_asset(ritem, "state")
        if not ndvi_a or not rel_a or not b1_a:
            raise ValueError(f"missing SDS assets for tile {tid}")
        ndvi_r.append(read_window(ndvi_a.href, (H, W), "nearest"))
        rel_r.append(read_window(rel_a.href, (H, W), "nearest"))
        evi_r.append(read_window(evi_a.href, (H, W), "nearest")
                     if evi_a else (np.full((H, W), np.nan), None))
        b1_r.append(read_window(b1_a.href, (H, W), "nearest"))
        b3_r.append(read_window(b3_a.href, (H, W), "nearest"))
        b4_r.append(read_window(b4_a.href, (H, W), "nearest"))
        b6_r.append(read_window(b6_a.href, (H, W), "nearest")
                     if b6_a else (np.full((H, W), np.nan), None))
        st_r.append(read_window(st_a.href, (H, W), "nearest")
                     if st_a else (np.full((H, W), np.nan), None))
    if not ndvi_r:
        print(f"[{PRODUCT}] VALIDATION FAILED: no tile reads. Keeping previous.")
        return 2

    def _a(r):
        return r[0]

    def _n(r):
        return r[2]

    ndvi = _mosaic([(_a(r), _n(r)) for r in ndvi_r]) * 0.0001
    rel = _mosaic([(_a(r), _n(r)) for r in rel_r])
    red = _mosaic([(_a(r), _n(r)) for r in b1_r]) * 0.0001
    blue = _mosaic([(_a(r), _n(r)) for r in b3_r]) * 0.0001
    green = _mosaic([(_a(r), _n(r)) for r in b4_r]) * 0.0001
    swir = _mosaic([(_a(r), _n(r)) for r in b6_r]) * 0.0001
    state = _mosaic([(_a(r), _n(r)) for r in st_r])
    print(f"[{PRODUCT}] tiles={sorted(vi_ids)} ndvi range "
          f"[{np.nanmin(ndvi):.3f},{np.nanmax(ndvi):.3f}]")

    # QA: pixel reliability 0 good; 1 marginal(hold); 2 snow; 3 cloud; else bad.
    # Plus MODLAND usefulness via state bit0 cloud as backup.
    rel_ok = np.isfinite(rel)
    good = rel_ok & (rel == 0)
    marginal = rel_ok & (rel == 1)
    snow_qa = rel_ok & (rel == 2)
    cloudy_qa = rel_ok & (rel == 3)
    ndsi = (green - swir) / (green + swir + 1e-6)
    snow = snow_qa | ((ndsi > 0.4) & (ndvi < 0.25) & np.isfinite(ndvi))
    st_int = np.where(np.isfinite(state), state, 0).astype(np.int64)
    cloudy = cloudy_qa | ((st_int & 1) == 1)
    bad = ~(good | marginal) & ~snow_qa
    n_good = int(good.sum())
    print(f"[{PRODUCT}] good={n_good} marginal={int(marginal.sum())} "
          f"snow={int(snow.sum())} cloudy={int(cloudy.sum())}")
    if n_good < 50_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too few good pixels. Keeping previous.")
        return 2

    # ancillary grids
    lc = np.array(Image.open(os.path.join(REPO_ROOT, "assets",
                                          "leaf_landcover.png")).convert("L"))
    from geospatial_utils import load_michigan_mask, load_watermask
    land = load_watermask() < 0.5  # NOT lake water (mask is float 0..1)
    mich = load_michigan_mask()  # Michigan-only hard clip
    if lc.shape != (H, W):
        raise ValueError(f"landcover shape {lc.shape} != canvas")

    # history (quarter-res): previous NDVI stack + phase for baseline/hold
    hist = load_history()
    with np.errstate(invalid="ignore", divide="ignore"):
        redness = np.clip(((red / (red + green + blue + 1e-6)) - 0.38) / 0.12, 0, 1)
        redness = np.where(np.isfinite(redness), redness, 0.0)
    hv = [hist["ndvi"][i] for i in range(hist["ndvi"].shape[0])]

    phase = compute_phase_grid(ndvi, lc, redness, snow,
                               bad | cloudy, marginal, hist)
    # FULL-COVERAGE RULE (v3): every Michigan land pixel (state mask minus
    # Great-Lakes water) must render opaque. Clouds, snow, masked landcover
    # classes (urban/barren/nodata/inland-water) and bad-QA pixels with no
    # history previously went transparent, leaving speckled holes across
    # the state. Gap-fill them: hold-forward history first, then
    # nearest-valid spatial propagation, then global circular-median
    # fallback -- never transparent on land.
    phase = fill_phase_full_coverage(phase, mich, land, prev_phase_grid(hist))
    lut = np.array(build_leaf_lut(), dtype=np.uint8)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    target = mich & land
    ok = np.isfinite(phase) & target
    # Safety net: if any target pixel is still NaN (should be impossible
    # after the fill), it is a bug -- fail loudly rather than ship holes.
    n_holes = int((target & ~ok).sum())
    if n_holes:
        raise ValueError(f"full-coverage fill left {n_holes} holes")
    rgba[ok, 0:3] = lut[np.clip((phase[ok] * 255).astype(int), 0, 255)]
    rgba[ok, 3] = bounds["overlay_alpha"]
    from geospatial_utils import apply_shoreline_mask
    rgba = apply_shoreline_mask(rgba, invert=True)  # leaf grows on LAND
    # Michigan edge: hard clip outside the state boundary
    rgba[~mich, 3] = 0
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    print(f"[{PRODUCT}] opaque pixels={n_opaque}")
    if n_opaque < 50_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: too little vegetation. Keeping previous.")
        return 2
    n_colors = len(np.unique(rgba[(rgba[:, :, 3] > 0)][..., :3].reshape(-1, 3),
                                axis=0))
    print(f"[{PRODUCT}] distinct raster colors={n_colors}")

    vstarts = [composite_start_day(i) for i in vi_ids.values()]
    vstarts = [d for d in vstarts if d]
    comp_date = max(vstarts) if vstarts else None
    today = dt.datetime.now(dt.timezone.utc).date()
    age = (today - comp_date).days if comp_date else None
    stale = age is not None and age > CONFIG["stale_threshold_days"]
    subtitle = (f"{CONFIG['subtitle']}  |  Composite: {comp_date} "
                f"({age}d old)" if age is not None else CONFIG["subtitle"])
    if stale:
        subtitle += "  |  STALE SOURCE"
    lw, lh = draw_leaf_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        f"Source: MODIS Aqua MYD13A1/MYD09A1 via Planetary Computer  |  "
        f"Processed {utcnow_iso()}", build_leaf_lut())

    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=(f"VI composite start {comp_date} (+16d window); "
                       f"reflectance {max([composite_start_day(i) for i in rf_ids.values() if composite_start_day(i)] or ['unknown'])} (+8d); "
                       f"retrieved {utcnow_iso()}"),
        source_last_modified_utc=f"composite sig {sig[:12]}",
        units="phenology phase 0..1 (display color); source NDVI/reflectance",
        source_resolution="500 m MODIS sinusoidal, reprojected to canvas (nearest)",
        color_min=0.0, color_max=1.0, color_units="phenology phase (circular)",
        missing_data_treatment=("Michigan land renders with full coverage: "
                                "cloud/bad-QA hold the previous phase, then "
                                "nearest-valid spatial fill; snow, urban/barren/"
                                "nodata and inland-water classes are gap-filled "
                                "from neighbors/history so no land holes remain. "
                                "Great-Lakes water transparent via shared mask."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = (
        "Satellite-derived seasonal vegetation state. The continuous color "
        "scale represents the changing seasonal condition of vegetated land, "
        "from winter dormancy through spring emergence, active growth, "
        "autumn coloration, leaf drop, and return to dormancy. Color "
        "represents a satellite-derived phenological state and should not "
        "be interpreted as the exact color of every individual tree. "
        "Every Michigan land pixel is painted: clouds hold the previous "
        "phase, and snow, urban, barren, nodata or briefly missing pixels "
        "are filled from surrounding valid land and recent history. Only "
        "Great-Lakes water stays transparent.")
    meta["data_nature"] = CONFIG["data_nature"]
    meta["gradient"] = {"interpolation": "OKLab (perceptually uniform)",
                        "anchors": 19, "distinct_raster_colors": n_colors,
                        "wraparound": "first == last deep blue #123B73"}
    meta["provenance"] = {
        "provider": "NASA MODIS (Aqua) via Microsoft Planetary Computer STAC",
        "collections": ["modis-13A1-061", "modis-09A1-061"],
        "vi_items": vi_ids, "reflectance_items": rf_ids,
        "composite_sig": sig,
        "composite_date": str(comp_date), "source_age_days": age,
        "stale": stale, "retrieved": utcnow_iso(),
        "native_resolution_m": 500, "native_crs": "MODIS sinusoidal",
        "resampling": "nearest (no invented values)",
        "qa_rules": ("MYD13A1 pixel reliability 0=good/1=hold/2=snow/3=cloud; "
                     "MYD09A1 state bit0 cloud; NDSI>0.4+snow test; "
                     "brightness backup"),
        "landcover": ("USGS NLCD 2021 + NRCan 2020 Land Cover of Canada "
                      "(NALCMS inputs) mosaic -> assets/leaf_landcover.png"),
        "water_mask": "assets/great_lakes_watermask.png (shared)",
        "michigan_mask": "assets/michigan_mask.png (authoritative state "
                          "boundary, both peninsulas; hard clip)", 
        "algorithm": "leaf_phenology v3 (trajectory + baseline + class + "
                     "spectral gating; 19-anchor OKLab circular gradient; "
                     "full-coverage Michigan gap-fill)",
    }
    # v3 = full-coverage Michigan gap-fill (no land holes) + full-coverage
    # mosaic rule (all 3 tiles required) + NaN-aware history means.
    # One-time version rotation to deploy the fixed rendering; afterwards
    # the id tracks source composites only.
    source_id = f"leaf-v3-{sig[:12]}"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["stats"] = {
        "good_pixels": n_good, "opaque_pixels": n_opaque,
        "phase_mean": round(float(np.nanmean(phase)), 4),
    }
    write_metadata(stage_prod, meta)

    # v3 = full-coverage Michigan gap-fill (no land holes) + full-coverage
    # mosaic rule (all 3 tiles required) + NaN-aware history means.
    # One-time version rotation to deploy the fixed rendering; afterwards
    # the id tracks source composites only.
    token = meta["source_version"]
    scale_html = meta["legend_scale_html"]
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    outs = live_out_dirs(stage, KML_FILE)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        out_dirs=outs["live"])
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    save_history(ndvi, phase, vi_ids, sig)
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"composite_sig": sig,
                          "source_id": source_id,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def downsample_quarter(a):
    """NaN-aware quarter-res block MEAN (not single-pixel sampling).

    Single-pixel sampling aliased cloud/QA holes into the history, so a
    cloudy 4x4 block poisoned hold-forward for the whole block on later
    runs. The block mean represents valid observations wherever any
    exist, which is what lets the rolling history legitimately fill
    transient observational gaps.
    """
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


def load_history():
    import glob
    base = {"ndvi": np.full((4,) + HIST_SHAPE, np.nan, dtype=np.float32),
            "phase": np.full(HIST_SHAPE, np.nan, dtype=np.float32),
            "dates": []}
    try:
        h = np.load(os.path.join(STATE_DIR, "hist_ndvi.npy"))
        p = np.load(os.path.join(STATE_DIR, "hist_phase.npy"))
        m = json.load(open(os.path.join(STATE_DIR, "hist_meta.json")))
        if h.shape == base["ndvi"].shape and p.shape == base["phase"].shape:
            base = {"ndvi": h.astype(np.float32) / 100.0
                    if h.dtype == np.int8 else h.astype(np.float32),
                    "phase": p.astype(np.float32),
                    "dates": m.get("dates", [])}
    except Exception:
        pass
    return base


def save_history(ndvi, phase, vi_ids, sig):
    os.makedirs(STATE_DIR, exist_ok=True)
    try:
        h = np.load(os.path.join(STATE_DIR, "hist_ndvi.npy"))
    except Exception:
        h = np.full((4,) + HIST_SHAPE, -128, dtype=np.int8)
    h = np.roll(h, -1, axis=0)
    h[-1] = np.clip(np.nan_to_num(
        downsample_quarter(np.where(np.isfinite(ndvi), ndvi, np.nan)),
        nan=-1.28) * 100.0, -128, 127).astype(np.int8)
    np.save(os.path.join(STATE_DIR, "hist_ndvi.npy"), h)
    np.save(os.path.join(STATE_DIR, "hist_phase.npy"),
            downsample_quarter(np.where(np.isfinite(phase), phase, np.nan)
                               ).astype(np.float32))
    json.dump({"dates": sorted(set(vi_ids.values())), "sig": sig},
              open(os.path.join(STATE_DIR, "hist_meta.json"), "w"))


def prev_phase_grid(hist):
    """Upsample quarter-res history phase to full canvas (nearest)."""
    qh, qw = HIST_SHAPE
    H, W = load_bounds()["canvas_height"], load_bounds()["canvas_width"]
    ys = np.clip((np.arange(H) * qh / H).astype(int), 0, qh - 1)
    xs = np.clip((np.arange(W) * qw / W).astype(int), 0, qw - 1)
    return hist["phase"][ys[:, None], xs]


def fill_phase_full_coverage(phase, mich, land, prev_phase=None):
    """Gap-fill phenology phase so Michigan land has zero holes.

    Target = mich & land (state mask minus Great-Lakes water; shoreline
    antialiasing applied later). Fill order for missing target pixels:
      1. hold-forward previous published phase (temporal continuity,
         works for clouds/snow/urban alike);
      2. nearest-valid spatial propagation (Voronoi fill; preserves local
         gradient texture, avoids circular-mean seam artifacts);
      3. global circular-median fallback (only if a component has no
         valid neighbor at all).
    Valid observed pixels are never altered.
    """
    target = np.asarray(mich, dtype=bool) & np.asarray(land, dtype=bool)
    out = np.array(phase, dtype=float)
    valid = np.isfinite(out) & target
    if not np.any(valid):
        return out
    missing = target & ~np.isfinite(out)
    if prev_phase is not None:
        pp = np.asarray(prev_phase, dtype=float)
        if pp.shape == out.shape:
            use = missing & np.isfinite(pp)
            out[use] = np.clip(pp[use], 0.0, 0.999)
            missing = target & ~np.isfinite(out)
    if np.any(missing):
        # Nearest-valid propagation: iteratively dilate filled region.
        filled = np.isfinite(out) & target
        # Seed: also allow valid pixels just outside target to donate
        # colors across the target edge (prevents edge darkening).
        donor = np.where(np.isfinite(out), out, np.nan)
        cur = np.where(filled, donor, np.nan)
        it = 0
        while np.any(target & ~np.isfinite(cur)) and it < 5000:
            it += 1
            nxt = cur.copy()
            have = np.isfinite(cur)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                shift_have = np.roll(np.roll(have, dy, axis=0), dx, axis=1)
                shift_val = np.roll(np.roll(cur, dy, axis=0), dx, axis=1)
                # np.roll wraps edges; mask wrapped rows/cols.
                if dy == 1:
                    shift_have[0, :] = False
                elif dy == -1:
                    shift_have[-1, :] = False
                if dx == 1:
                    shift_have[:, 0] = False
                elif dx == -1:
                    shift_have[:, -1] = False
                take = (~np.isfinite(nxt)) & shift_have
                nxt[take] = shift_val[take]
            if np.array_equal(np.isfinite(nxt), np.isfinite(cur)):
                break
            cur = nxt
        still = target & ~np.isfinite(cur)
        if np.any(still):
            # Global circular-median fallback (unit-circle median).
            v = out[valid]
            ang = v * 2.0 * math.pi
            mx, my = float(np.median(np.cos(ang))), float(np.median(np.sin(ang)))
            fb = (math.atan2(my, mx) / (2.0 * math.pi)) % 1.0
            cur[still] = min(max(fb, 0.0), 0.999)
        out = cur
    return np.clip(out, 0.0, 0.999)


def compute_phase_grid(ndvi, lc, redness, snow, bad, marginal, hist):
    """Vectorized phenology phase (mirrors leaf_phenology rules)."""
    H, W = ndvi.shape
    hq = HIST_SHAPE[1] / W  # quarter-res mapping factor
    qh, qw = HIST_SHAPE
    ys = np.clip((np.arange(H) * qh / H).astype(int), 0, qh - 1)
    xs = np.clip((np.arange(W) * qw / W).astype(int), 0, qw - 1)
    hn = hist["ndvi"]  # (4, qh, qw), NaN-aware, newest LAST after roll? we roll before save; in-memory: build stack
    # build valid stack: [h0,h1,h2,h3,current]
    cur_q = downsample_quarter(np.where(np.isfinite(ndvi), ndvi, np.nan))
    stack = list(hn) + [cur_q]
    n_prev = stack[-2]
    # median of valid history as the robust direction reference
    hcnt = sum(np.isfinite(s).astype(int) for s in stack[:-1])
    htot = sum(np.where(np.isfinite(s), s, 0.0) for s in stack[:-1])
    n_ref_q = np.where(hcnt > 0, htot / np.maximum(hcnt, 1), cur_q)
    n_max_q = np.maximum.reduce(
        [np.where(np.isfinite(s), s, -1) for s in stack])
    n_max_q = np.where(n_max_q < -0.5, np.nan, n_max_q)
    # upsample to full res (nearest)
    n_ref = n_ref_q[ys[:, None], xs]
    n_max = n_max_q[ys[:, None], xs]
    prev_phase = hist["phase"][ys[:, None], xs]

    phase = np.full((H, W), np.nan)
    hb = bad | ~np.isfinite(ndvi)
    has_prev = np.isfinite(prev_phase)
    phase[hb & has_prev] = prev_phase[hb & has_prev]
    work = ~(hb | snow) & np.isfinite(ndvi)
    # class priors for cold-start baseline (always applied)
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
    m_fall = work & ~m_dorm & (d <= -0.02)
    t = 1.0 - ngr[m_fall]
    ph = 0.55 + 0.30 * t + 0.08 * red[m_fall] * t
    low = ngr[m_fall] < 0.22
    ph[low] = np.maximum(ph[low], 0.85 + 0.15 * (0.22 - ngr[m_fall][low]) / 0.22)
    phase[m_fall] = ph
    m_st = work & ~m_dorm & (np.abs(d) <= 0.02)
    hi = m_st & (ngr > 0.6)
    phase[hi] = 0.45 + 0.10 * np.clip((ngr[hi] - 0.7) / 0.3, 0, 1)
    lo = m_st & (ngr <= 0.6)
    phase[lo] = 0.10 + 0.25 * (ngr[lo] / 0.6)

    # class modulation
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
    # marginal-QA pixels participate with damped response (documented)
    m_marg = work & np.asarray(marginal, dtype=bool) & np.isfinite(phase)
    phase[m_marg] = 0.5 + (phase[m_marg] - 0.5) * 0.6
    return np.clip(phase, 0.0, 0.999)


if __name__ == "__main__":
    sys.exit(main())
