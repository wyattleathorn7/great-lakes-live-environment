"""NASA HLS v2 acquisition + single-mosaic compositor for LIVE LEAF COLOR.

Harmonized Landsat Sentinel-2 (HLS) v2.0 is produced by NASA at 30 m on a
common MGRS/UTM grid: HLSL30 (Landsat 8/9 OLI) and HLSS30 (Sentinel-2A/B/C
MSI).  Both collections are queried through the Microsoft Planetary
Computer STAC API and feed ONE temporal mosaic on the shared Great Lakes
canvas.  There are no per-tile/per-lake gradients: every scene is read into
the same canvas, the newest clear observation wins at each pixel, and the
phenology phase is computed once over the whole field.

Fmask v2.0 bits (LP DAAC HLS User Guide):
  0 cirrus (reserved), 1 cloud, 2 adjacent to cloud/shadow, 3 cloud shadow,
  4 snow/ice, 5 water, 6-7 aerosol level.
Surface-reflectance bands are int16 scaled by 0.0001 with fill -9999.
"""

import datetime as dt
import os
import re

import numpy as np

from geospatial_utils import load_bounds

# A single stalled COG request must not hang a weekly build.  GDAL otherwise
# waits indefinitely on a dead S3/HTTP connection.
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "20")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")

COLLECTIONS = ("hls2-l30", "hls2-s30")
PLATFORM = {"hls2-l30": "l30", "hls2-s30": "s30"}

# HLS keeps native sensor band names.  L30: B05=NIR, B06=SWIR1.
# S30: B08=NIR, B11=SWIR1.  Blue/green/red are B02/B03/B04 on both.
BANDS = {
    "hls2-l30": {"blue": "B02", "green": "B03", "red": "B04",
                 "nir": "B05", "swir": "B06"},
    "hls2-s30": {"blue": "B02", "green": "B03", "red": "B04",
                 "nir": "B08", "swir": "B11"},
}

FILL = -9999
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
TILE_RE = re.compile(r"\.(T\d{2}[A-Z]{3})\.")


def _bbox(bounds):
    return [bounds["lon_min"], bounds["lat_min"],
            bounds["lon_max"], bounds["lat_max"]]


def tile_id(item):
    """MGRS tile id from an HLS item id (HLS.S30.T15TVH...)."""
    m = TILE_RE.search(item.id or "")
    return m.group(1) if m else None


def overlap_area(item, bounds):
    """Approximate lon/lat bbox overlap in square degrees (0 if disjoint)."""
    b = item.bbox
    if not b or len(b) < 4:
        return 0.0
    w = min(bounds["lon_max"], b[2]) - max(bounds["lon_min"], b[0])
    h = min(bounds["lat_max"], b[3]) - max(bounds["lat_min"], b[1])
    return max(0.0, w) * max(0.0, h)


def search_items(bounds, lookback_days=45, max_items=20000):
    """All HLS L30/S30 items intersecting the basin inside the window."""
    import pystac_client
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=int(lookback_days))
    cat = pystac_client.Client.open(STAC_URL)
    search = cat.search(
        collections=list(COLLECTIONS), bbox=_bbox(bounds),
        datetime=f"{start.isoformat()}/{end.isoformat()}", limit=1000)
    out = []
    for item in search.items():
        if item.collection_id not in COLLECTIONS:
            continue
        if not tile_id(item) or overlap_area(item, bounds) <= 0.0:
            continue
        keys = set(item.assets)
        if not {"Fmask"} <= keys:
            continue
        if not {"B02", "B03", "B04"} <= keys:
            continue
        out.append(item)
        if len(out) >= int(max_items):
            break
    return out


def _scene_date(item):
    try:
        return item.datetime.date()
    except Exception:
        return None


def select_scenes(items, max_per_tile=6, max_cloud=90.0):
    """Newest usable scenes per MGRS tile, spread across the window.

    HLS revisits every 2-3 days, but clouds are common.  The two newest
    scenes from each platform are kept, then the remaining slots are placed
    at even date targets so the phenology rules see an actual lookback
    trajectory instead of six copies of the same afternoon.
    """
    groups = {}
    for item in items:
        tid = tile_id(item)
        d = _scene_date(item)
        if not tid or d is None:
            continue
        groups.setdefault(tid, []).append(item)

    scenes = []
    for tid, group in groups.items():
        group.sort(key=lambda it: it.datetime, reverse=True)
        # Prefer low-cloud scenes, but never drop a tile entirely: a very
        # cloudy scene can still contain clear pixels the compositor needs.
        usable = [it for it in group
                  if float(it.properties.get("eo:cloud_cover", 0.0) or 0.0)
                  <= max_cloud]
        group = usable or group

        seen = set()
        deduped = []
        for it in group:
            key = (it.collection_id, _scene_date(it))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)

        chosen, picked = [], set()
        for col in COLLECTIONS:
            for it in [x for x in deduped if x.collection_id == col][:2]:
                if it.id not in picked:
                    chosen.append(it)
                    picked.add(it.id)

        want = int(max_per_tile) - len(chosen)
        remaining = [it for it in deduped if it.id not in picked]
        if want > 0 and remaining:
            dates = sorted({_scene_date(it) for it in deduped
                            if _scene_date(it)})
            if dates:
                lo, hi = dates[0], dates[-1]
                targets = [lo + dt.timedelta(
                    days=(hi - lo).days * (k + 1) / (want + 1))
                    for k in range(want)]
                for t in targets:
                    avail = [it for it in remaining if it.id not in picked]
                    if not avail:
                        break
                    pick = min(avail, key=lambda it: abs(
                        (_scene_date(it) - t).days))
                    chosen.append(pick)
                    picked.add(pick.id)
        for it in deduped:
            if len(chosen) >= int(max_per_tile):
                break
            if it.id not in picked:
                chosen.append(it)
                picked.add(it.id)
        scenes.extend(chosen[:int(max_per_tile)])

    # Globally newest-first so the first valid observation at every pixel is
    # the newest observation available for that pixel.
    scenes.sort(key=lambda it: it.datetime, reverse=True)
    return scenes


def decode_hls_fmask(fmask):
    """Decode HLS v2 Fmask bits into clear/cloud/snow/water/aerosol masks."""
    f = np.asarray(fmask)
    if f.dtype != np.uint8:
        f = np.clip(np.nan_to_num(f, nan=255), 0, 255).astype(np.uint8)
    valid = f != 255
    cloud = ((f & 0b00001110) != 0) | ~valid  # bit 0 reserved; 1-3 cloud
    snow = (f & 0b00010000) != 0
    water = (f & 0b00100000) != 0
    aerosol = (f >> 6) & 3
    clear = valid & ~cloud & ~snow & ~water & (aerosol < 3)
    return {"valid": valid, "cloud": cloud, "snow": snow,
            "water": water, "aerosol": aerosol, "clear": clear}


def _scene_geometry(item, bounds):
    """Canvas window/shape/offset for one HLS scene in EPSG:4326."""
    import rasterio
    from rasterio.enums import Resampling as RS
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window, from_bounds

    red_key = BANDS[item.collection_id]["red"]
    href = item.assets[red_key].href
    H, W = bounds["canvas_height"], bounds["canvas_width"]
    try:
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs="EPSG:4326",
                           resampling=RS.nearest) as vrt:
                win = from_bounds(bounds["lon_min"], bounds["lat_min"],
                                  bounds["lon_max"], bounds["lat_max"],
                                  vrt.transform)
                r0 = max(int(win.row_off), 0)
                r1 = min(int(win.row_off + win.height), vrt.height)
                c0 = max(int(win.col_off), 0)
                c1 = min(int(win.col_off + win.width), vrt.width)
                if r1 <= r0 or c1 <= c0 or win.height <= 0 or win.width <= 0:
                    return None
                out_h = max(1, int(round((r1 - r0) / win.height * H)))
                out_w = max(1, int(round((c1 - c0) / win.width * W)))
                or0 = int(round((r0 - win.row_off) / win.height * H))
                oc0 = int(round((c0 - win.col_off) / win.width * W))
                return {
                    "window": Window(c0, r0, c1 - c0, r1 - r0),
                    "shape": (out_h, out_w),
                    "offset": (or0, oc0),
                }
    except Exception:
        return None


def _read_patch(href, geom, resampling, nodata=None):
    import rasterio
    from rasterio.enums import Resampling as RS
    from rasterio.vrt import WarpedVRT
    rs = RS.average if resampling == "average" else RS.nearest
    with rasterio.open(href) as src:
        src_nodata = nodata if nodata is not None else src.nodata
        dst_nodata = np.nan if resampling == "average" else src_nodata
        kwargs = {"crs": "EPSG:4326", "src_nodata": src_nodata,
                  "dst_nodata": dst_nodata}
        if resampling == "average":
            kwargs["dtype"] = "float32"
        with WarpedVRT(src, **kwargs) as vrt:
            return vrt.read(1, window=geom["window"],
                            out_shape=geom["shape"], resampling=rs)


def _crop(full, offset, shape):
    h, w = shape
    or0, oc0 = offset
    H, W = full.shape
    out = np.zeros(shape, dtype=full.dtype)
    r0, c0 = max(or0, 0), max(oc0, 0)
    r1, c1 = min(or0 + h, H), min(oc0 + w, W)
    if r1 > r0 and c1 > c0:
        out[r0 - or0:r1 - or0, c0 - oc0:c1 - oc0] = full[r0:r1, c0:c1]
    return out


def _assign(full, patch, mask, offset):
    h, w = patch.shape
    or0, oc0 = offset
    H, W = full.shape
    r0, c0 = max(or0, 0), max(oc0, 0)
    r1, c1 = min(or0 + h, H), min(oc0 + w, W)
    if r1 <= r0 or c1 <= c0:
        return
    sub = full[r0:r1, c0:c1]
    ps = patch[r0 - or0:r1 - or0, c0 - oc0:c1 - oc0]
    ms = mask[r0 - or0:r1 - or0, c0 - oc0:c1 - oc0]
    sub[ms] = ps[ms]


def _seam_mask(tile_grid, dilate=2):
    m = np.zeros(tile_grid.shape, dtype=bool)
    for axis in (0, 1):
        for shift in (1, -1):
            other = np.roll(tile_grid, shift, axis=axis)
            m |= ((tile_grid >= 0) & (other >= 0) & (tile_grid != other))
    for _ in range(int(dilate)):
        grown = m.copy()
        for axis in (0, 1):
            for shift in (1, -1):
                grown |= np.roll(m, shift, axis=axis)
        m = grown
    m[0, :] = m[-1, :] = False
    m[:, 0] = m[:, -1] = False
    return m


def _nanmean3(a):
    import warnings
    stack = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            stack.append(np.roll(np.roll(a, dy, axis=0), dx, axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(stack), axis=0)


def blend_tile_seams(ndvi, redness, tile_grid):
    """Remove only the narrow MGRS tile-boundary bands.

    Pixels far from a tile edge keep their exact source value.  Boundary
    bands get a 3x3 nan-aware mean so adjacent tiles blend into one field
    instead of showing a mosaic seam.
    """
    seam = _seam_mask(tile_grid)
    if not seam.any():
        return ndvi, redness
    for arr in (ndvi, redness):
        mean = _nanmean3(arr)
        use = seam & np.isfinite(mean) & np.isfinite(arr)
        arr[use] = mean[use]
    return ndvi, redness


def _sign(item):
    import planetary_computer as pc
    return pc.sign(item)


def mosaic_scenes(scenes, target, bounds, epochs=4):
    """Composite all selected scenes into one canvas field.

    target: bool canvas mask of vegetated land.  Returns newest-clear NDVI,
    same-scene redness proxy, observation age, and provenance.  Snow is
    tracked separately so it can render as dormant winter (the legend's
    deep-blue end) instead of leaving a hole.

    Alongside the full newest-clear field, the same single pass accumulates
    `epochs` equal time slices of NDVI/redness across the selected scenes.
    Those slices are the HLS-only NDVI trajectory used by the phenology
    rules (rise/fall vs the recent mean), so no MODIS-era history is ever
    mixed into a 30 m HLS phase.  Each scene is still read exactly once.
    """
    H, W = bounds["canvas_height"], bounds["canvas_width"]
    ndvi = np.full((H, W), np.nan, dtype=np.float32)
    redness = np.full((H, W), np.nan, dtype=np.float32)
    obs_day = np.full((H, W), -1, dtype=np.int32)
    tile_grid = np.full((H, W), -1, dtype=np.int16)
    observed = np.zeros((H, W), dtype=bool)
    snow_seen = np.zeros((H, W), dtype=bool)

    days = [d for d in (_scene_date(it) for it in scenes) if d]
    end_day = max(days) if days else dt.date.today()
    start_day = min(days) if days else end_day
    span = max((end_day - start_day).days, 1)
    E = max(1, int(epochs))
    ep_ndvi = np.full((E, H, W), np.nan, dtype=np.float32)
    ep_red = np.full((E, H, W), np.nan, dtype=np.float32)
    ep_seen = np.zeros((E, H, W), dtype=bool)

    tile_codes = {}
    used_ids = []
    used_dates = []
    item_map = {}
    used_tiles = set()
    print(f"[leaf_hls] compositing {len(scenes)} scenes into the basin canvas",
          flush=True)

    for item in scenes:
        tid = tile_id(item)
        if not tid:
            continue
        try:
            signed = _sign(item)
            geom = _scene_geometry(signed, bounds)
        except Exception as exc:
            print(f"[leaf_hls] skip {item.id}: {exc}")
            continue
        if geom is None:
            continue
        shape, off = geom["shape"], geom["offset"]

        try:
            fm = _read_patch(signed.assets["Fmask"].href, geom, "nearest",
                             nodata=255)
        except Exception as exc:
            print(f"[leaf_hls] Fmask skip {item.id}: {exc}")
            continue
        qa = decode_hls_fmask(fm)

        target_patch = _crop(target, off, shape)
        seen_patch = _crop(observed, off, shape)
        new = qa["clear"] & target_patch & ~seen_patch
        # Track snow only where no clear observation exists yet.
        _assign(snow_seen, qa["valid"] & qa["snow"] & target_patch
                & ~seen_patch, np.ones(shape, dtype=bool), off)

        day = _scene_date(item)
        if day is None:
            e = E - 1
        else:
            frac = (day - start_day).days / span
            e = min(E - 1, max(0, int(frac * E)))
        e_seen_patch = _crop(ep_seen[e], off, shape)
        e_new = qa["clear"] & target_patch & ~e_seen_patch
        if not new.any() and not e_new.any():
            continue

        bands = BANDS[item.collection_id]
        try:
            red = _read_patch(signed.assets[bands["red"]].href,
                              geom, "average", nodata=FILL).astype(np.float32)
            nir = _read_patch(signed.assets[bands["nir"]].href,
                              geom, "average", nodata=FILL).astype(np.float32)
        except Exception as exc:
            print(f"[leaf_hls] band skip {item.id}: {exc}")
            continue
        red = np.where(red == FILL, np.nan, red * 0.0001)
        nir = np.where(nir == FILL, np.nan, nir * 0.0001)
        with np.errstate(invalid="ignore", divide="ignore"):
            n = (nir - red) / (nir + red + 1e-6)
        usable = (new | e_new) & np.isfinite(n) & (red > 0.0) & (nir > 0.0)
        if not usable.any():
            continue

        try:
            green = _read_patch(signed.assets[bands["green"]].href,
                                geom, "average", nodata=FILL).astype(np.float32)
            blue = _read_patch(signed.assets[bands["blue"]].href,
                               geom, "average", nodata=FILL).astype(np.float32)
        except Exception:
            green = np.full(shape, np.nan, dtype=np.float32)
            blue = np.full(shape, np.nan, dtype=np.float32)
        green = np.where(green == FILL, np.nan, green * 0.0001)
        blue = np.where(blue == FILL, np.nan, blue * 0.0001)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.clip(((red / (red + green + blue + 1e-6)) - 0.38) / 0.12,
                        0.0, 1.0)
        r = np.where(np.isfinite(r), r, 0.0)

        day_ord = day.toordinal() if day else 0
        code = tile_codes.setdefault(tid, len(tile_codes))

        _assign(ndvi, n, usable & new, off)
        _assign(redness, r, usable & new, off)
        _assign(obs_day, np.full(shape, day_ord, dtype=np.int32),
                usable & new, off)
        _assign(tile_grid, np.full(shape, code, dtype=np.int16),
                usable & new, off)
        _assign(observed, np.ones(shape, dtype=bool), usable & new, off)
        # Epoch slices feed the phenology trajectory, newest-first per slice.
        _assign(ep_ndvi[e], n, usable & e_new, off)
        _assign(ep_red[e], r, usable & e_new, off)
        _assign(ep_seen[e], np.ones(shape, dtype=bool), usable & e_new, off)

        used_ids.append(item.id)
        used_tiles.add(tid)
        if day:
            used_dates.append(day)
        if len(used_ids) % 50 == 0:
            print(f"[leaf_hls] {len(used_ids)}/{len(scenes)} scenes read, "
                  f"{int(observed.sum())} pixels observed", flush=True)
        slot = item_map.setdefault(tid, {})
        plat = PLATFORM.get(item.collection_id, item.collection_id)
        slot[plat] = item.id

        # Stop only when the current field is complete AND every epoch has
        # enough coverage to describe the lookback trajectory.
        need = 0.10 * float(target.sum())
        ep_ok = all(float((ep_seen[i] & target).sum()) >= need
                    for i in range(E))
        if ep_ok and int(observed.sum()) >= int(target.sum() * 0.999):
            break

    if not used_ids:
        raise RuntimeError("no usable HLS observations after QA")

    stats = {
        "scenes_selected": len(scenes),
        "scenes_used": len(used_ids),
        "tiles_used": len(used_tiles),
        "observed_pixels": int(observed.sum()),
        "target_pixels": int(target.sum()),
        "composite_date": max(used_dates) if used_dates else None,
        "epoch_pixels": [int((ep_seen[i] & target).sum()) for i in range(E)],
    }
    return {
        "ndvi": ndvi, "redness": redness, "obs_day": obs_day,
        "tile_grid": tile_grid, "observed": observed,
        "snow_seen": snow_seen, "used_ids": used_ids,
        "ndvi_epochs": ep_ndvi, "redness_epochs": ep_red,
        "item_map": item_map, "stats": stats,
    }
