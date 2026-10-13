"""Pipeline C1-NA — LIVE CLOUD COVER, NORTH AMERICA (independent).

Precedence (explicit, never substituted):
  1) Valid GOES-East (GOES-19) + GOES-West (GOES-18) full-disk ABI L2
     Clear Sky Mask (ABI-L2-ACMF) BCM gated DQF==0. % = cloudy fraction
     per NA canvas cell (true area density).
  2) RRFS TCDC entire-atmosphere analysis (%) where no valid GOES mask.
  3) ECCC GDPS TotalCloudCover ONLY for verified remaining gaps.
Missing satellite pixels are NEVER rendered as clear sky. One 0-100%
scale, identical styling (CLOUD_STOPS shared with the Great Lakes
layer); metadata distinguishes observed vs model pixels.

Exit codes: 0 updated; 2 source failure (previous kept); 1 error.
"""

import datetime as dt
import json
import os
import re
import sys
import traceback
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from na_bounds import load_na_bounds
from na_common import (apply_coast_mask, bin_points_to_canvas, load_landmask,
                         despeckle_valid, feather_merge, fetch_http,
                         fill_pinholes, gate_field_to_mask,
                         sanitize_model_vals, smooth_field)
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs)
from build_cloud_cover import CLOUD_STOPS, CLOUD_LABELS
from geospatial_utils import (REPO_ROOT, SITE_DIR, base_metadata, now_det_str,
                              promote_stage, read_state, save_png,
                              source_token, stage_dir, write_metadata,
                              write_state)
from gradient_scale import draw_scale_legend, render_rgba
import goes_acm

PRODUCT = "cloud_cover_na"
CONFIG_NA = json.load(open(os.path.join(REPO_ROOT, "config", "cloud_cover_na.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "North_America_Live_Cloud_Cover.kml"
OVERLAY_NAME = "☁️ LIVE CLOUD COVER"
SKIP_NOTE = "Turn on/off independently of all other layers."

FD_SOURCES = [
    ("GOES-19", "https://noaa-goes19.s3.amazonaws.com", "ABI-L2-ACMF", "G19"),
    ("GOES-18", "https://noaa-goes18.s3.amazonaws.com", "ABI-L2-ACMF", "G18"),
]
FD_RE = re.compile(r"OR_ABI-L2-ACMF-M6_G\d{2}_s(\d{4})(\d{3})(\d{2})(\d{2})\d{3}_e.*\.nc$")


def fd_newest(bucket, prefix, max_back_hours=6, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    floor = now - dt.timedelta(minutes=12)
    t = now.replace(minute=0, second=0, microsecond=0)
    last = None
    for _ in range(max_back_hours + 1):
        dd = t.strftime("%Y/%j/%H")
        try:
            url = f"{bucket}/?list-type=2&prefix={prefix}/{dd}/&max-keys=100"
            req = urllib.request.Request(url, headers={"User-Agent": "great-lakes-live-environment/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                root = ET.fromstring(r.read())
            ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
            keys = [e.text for e in root.findall("s:Contents/s:Key", ns)]
        except Exception as e:
            last = e
            t -= dt.timedelta(hours=1)
            continue
        cands = []
        for k in keys:
            m = FD_RE.search(k.split("/")[-1])
            if not m:
                continue
            try:
                scan = dt.datetime(int(m.group(1)), 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
                    days=int(m.group(2)) - 1, hours=int(m.group(3)), minutes=int(m.group(4)))
            except ValueError:
                continue
            if scan <= floor:
                cands.append((scan, k))
        if cands:
            scan, key = max(cands)
            return f"{bucket}/{key}", scan
        t -= dt.timedelta(hours=1)
    raise last or RuntimeError(f"no {prefix} FD file found")


def fd_read_mask(path):
    """FD variant of goes_acm.read_mask (no CONUS shape assertion).

    CRITICAL: scale/offset are read from the file's own x/y attributes.
    Full-disk files use add_offset ∓0.151844 (vs CONUS -0.101332 /
    +0.128212); using the CONUS constants shifts the whole disk by
    degrees (verified live 2026-10-11).
    """
    import h5netcdf
    f = h5netcdf.File(path, "r")
    try:
        xa, ya = f["x"].attrs, f["y"].attrs
        xs = float(xa["scale_factor"]) if "scale_factor" in xa else goes_acm.X_SCALE
        xo = float(xa["add_offset"]) if "add_offset" in xa else goes_acm.X_OFFSET
        ys = float(ya["scale_factor"]) if "scale_factor" in ya else goes_acm.Y_SCALE
        yo = float(ya["add_offset"]) if "add_offset" in ya else goes_acm.Y_OFFSET
        x = f["x"][:].astype(float) * xs + xo
        y = f["y"][:].astype(float) * ys + yo
        bcm = np.asarray(f["BCM"][:]).astype(float)
        dqf = np.asarray(f["DQF"][:]).astype(float)
        proj = f["goes_imager_projection"].attrs
        lon0 = float(proj["longitude_of_projection_origin"])
        h = float(proj["perspective_point_height"])
    finally:
        f.close()
    good = (dqf == 0) & np.isfinite(bcm) & ((bcm == 0) | (bcm == 1))
    cloudy = np.where(good, bcm, np.nan)
    lats, lons = goes_acm.fixed_grid_to_latlon(x, y, lon0, h)
    return cloudy.ravel(), lats.ravel(), lons.ravel()


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    bounds = load_na_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    try:
        return _build(bounds, W, H)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] FAILED: {type(e).__name__}: {e}. Keeping previous.")
        return 2


def _build(bounds, W, H):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    os.makedirs(stage_prod, exist_ok=True)
    # Fresh checkouts (CI runners) lack the gitignored output/raw dir.
    os.makedirs(RAW_DIR, exist_ok=True)
    prov = {}
    sat_grids = []

    for name, bucket, prefix, sat in FD_SOURCES:
        try:
            url, scan = fd_newest(bucket, prefix)
            dest = os.path.join(RAW_DIR, f"na_{sat.lower()}_acmf.nc")
            goes_acm.download(url, dest)
            cloudy, lats, lons = fd_read_mask(dest)
            # splat_radius=2 like the original CONUS layer: the rotated
            # GOES fixed grid is finer than the canvas and prints moire
            # without it.
            g, c = bin_points_to_canvas(cloudy, lats, lons, bounds, agg="mean",
                                        splat_radius=2)
            sat_grids.append(g)
            prov[name] = f"{scan:%Y-%m-%d %H:%MZ} valid-cells={int(np.isfinite(g).sum())}"
        except Exception as e:
            prov[name] = f"unavailable: {type(e).__name__}: {e}"

    field = np.full((H, W), np.nan)  # 0..100 %
    sat_valid = np.zeros((H, W), dtype=bool)
    if sat_grids:
        stack = np.stack([np.where(np.isfinite(g), g, np.nan) for g in sat_grids])
        with np.errstate(invalid="ignore"):
            frac = np.nanmean(stack, axis=0)
        m = np.isfinite(frac)
        field[m] = frac[m] * 100.0
        sat_valid[m] = True
    layers = [(field, sat_valid)]
    # RRFS TCDC fallback where GOES missing.
    rrfc = 0
    rrfs_grid = np.full((H, W), np.nan)
    rrfs_valid = np.zeros((H, W), dtype=bool)
    try:
        from na_common import eccodes_read_idx_ranges
        now = dt.datetime.now(dt.timezone.utc)
        got = False
        for back in range(0, 8):
            t = now - dt.timedelta(hours=back)
            dd = t.strftime("rrfs.%Y%m%d")
            for base in ("https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod",
                         "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/para"):
                fn = "rrfs.t00z.2dfld.13km.f000.na.grib2"
                idx_url = f"{base}/{dd}/00/{fn}.idx"
                try:
                    idx = fetch_http(idx_url, timeout=30).decode("utf-8", errors="replace")
                except Exception:
                    continue
                try:
                    rng = eccodes_read_idx_ranges(idx, [("TCDC", "entire atmosphere")])
                except ValueError:
                    continue
                a, b = rng["TCDC"]
                dest = os.path.join(RAW_DIR, "na_rrfs_tcdc.grib2")
                req = urllib.request.Request(f"{base}/{dd}/00/{fn}",
                                             headers={"User-Agent": "great-lakes-live-environment/1.0",
                                                      "Range": f"bytes={a}-{b-1}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                with open(dest + ".part", "wb") as f:
                    f.write(data)
                os.replace(dest + ".part", dest)
                from hrrr import read_messages
                vals, lats, lons, d1, d2 = read_messages(dest, {"TCDC": 1})["TCDC"]
                lons = ((np.asarray(lons) + 180) % 360) - 180
                vals = sanitize_model_vals(np.asarray(vals, float), 0.0, 100.0)
                g, c = bin_points_to_canvas(vals, np.asarray(lats, float),
                                            np.asarray(lons, float), bounds)
                g = fill_pinholes(g, radius=3)
                m = np.isfinite(g)
                rrfs_grid[m] = np.clip(g[m], 0, 100)
                rrfs_valid[m] = True
                # Display-smoothing on the MODEL layer only: the coarse
                # 13 km grid must not print cell edges. The GOES-observed
                # layer below is NEVER smoothed (its raw 2 km grain is the
                # original layer's look).
                rrfs_grid = smooth_field(rrfs_grid, size=5, passes=2)
                rrfc = int(m.sum())
                prov["RRFS-TCDC"] = f"{dd} n={rrfc}"
                got = True
                break
            if got:
                break
        if not got:
            prov["RRFS-TCDC"] = "unavailable: no analysis found"
    except Exception as e:
        prov["RRFS-TCDC"] = f"unavailable: {type(e).__name__}"

    layers.append((rrfs_grid, rrfs_valid))
    # GDPS TotalCloudCover last-resort fill ONLY where GOES + RRFS both lack
    # verified valid data (same fallback contract as air-temp/wind; the
    # config already promises it). Bilinear resample so coarse cells render
    # as gradient, never dots.
    try:
        from na_common import gdps_wcs_grid, resample_wcs_to_canvas
        arr, _la, _lo, slat, slon = gdps_wcs_grid(
            "GDPS_15km_TotalCloudCover", bounds, RAW_DIR)
        raw = np.asarray(arr[0], dtype=float)
        if np.isfinite(raw).any() and np.nanmax(raw) <= 1.0:
            raw = raw * 100.0  # fraction -> %
        g = resample_wcs_to_canvas(
            sanitize_model_vals(raw, 0.0, 100.0), slat, slon, bounds)
        g = smooth_field(fill_pinholes(np.clip(g, 0.0, 100.0), radius=3),
                         size=5, passes=2)
        layers.append((g, np.isfinite(g)))
        prov["GDPS-TCC"] = f"GDPS_15km_TotalCloudCover n={int(np.isfinite(g).sum())}"
    except Exception as e:
        prov["GDPS-TCC"] = f"unavailable: {type(e).__name__}: {e}"
    # Per-pixel provenance mask (same precedence as the merge: sat wins,
    # then RRFS, then GDPS) for accuracy auditing: 1=GOES observed,
    # 2=RRFS model, 3=GDPS model, 0=missing.
    try:
        _gdps_ok = np.isfinite(layers[2][0]) & np.asarray(layers[2][1], dtype=bool) \
            if len(layers) > 2 else np.zeros((H, W), dtype=bool)
        _rrfs_ok = np.isfinite(layers[1][0]) & np.asarray(layers[1][1], dtype=bool)
        _sat_ok = np.asarray(sat_valid, dtype=bool) & np.isfinite(layers[0][0])
        provmask = np.zeros((H, W), dtype=np.uint8)
        provmask[_gdps_ok] = 3
        provmask[_rrfs_ok] = 2
        provmask[_sat_ok] = 1
        np.savez_compressed(os.path.join(RAW_DIR, "na_cloud_prov.npz"),
                            prov=provmask,
                            built_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except Exception as e:
        print(f"[{PRODUCT}] WARNING: prov mask write failed: {e}")
    # Seamless cross-fade at the satellite/model boundary, then the coast
    # gate. NO global smoothing, NO despeckle: the original layer renders
    # the raw observed neighborhood fraction (its grain IS the product),
    # and model layers arrive pre-smoothed above. Gaps stay transparent.
    field, _v = feather_merge(layers, feather_px=12)
    # Exact air-temp footprint + 100mi expansion: selected-country feathered
    # mask (ends at Mexico, Caribbean exclusions carved).
    field = gate_field_to_mask(field, load_landmask())
    # Shared cache for the precipitation composite: the merged % field
    # (NaN = missing/clear) + build time, so precipitation_na can paint the
    # same realistic white-cloud underlay without refetching GOES/RRFS/GDPS.
    try:
        np.savez_compressed(os.path.join(RAW_DIR, "na_cloud_field.npz"),
                            field=field.astype(np.float32),
                            built_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except Exception as e:
        print(f"[{PRODUCT}] WARNING: cloud field cache write failed: {e}")
    n_sat = int(sat_valid.sum())
    n_mod = int(np.isfinite(field).sum() - (np.isfinite(field) & sat_valid).sum())
    if np.isfinite(field).sum() == 0:
        raise RuntimeError(f"no GOES-FD or RRFS cloud data ({prov})")
    print(f"[{PRODUCT}] sat px={n_sat} model-fill px={n_mod} prov={prov}")

    rgba = render_rgba(field, CLOUD_STOPS, 205)
    # Clear (<1%) and missing stay transparent (GL convention).
    clear = np.isfinite(field) & (field < 1.0)
    rgba[clear, 3] = 0
    rgba = apply_coast_mask(rgba, load_landmask())
    save_png(rgba, os.path.join(stage_prod, "current.png"))

    data_time = now_det_str()
    # Original continuous key: identical stops, labels, and units.
    lw, lh = draw_scale_legend(os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"],
                               f"GOES full-disk + RRFS fill | {data_time}", "%", CLOUD_STOPS, CLOUD_LABELS,
                               f"Source: NOAA GOES-East/GOES-West ABI clear-sky mask + RRFS TCDC fill  |  "
                               f"Processed {data_time}",
                               note="Clear sky (<1%) is transparent, not colored.")
    scale_html = ("Cloud cover (%, OBSERVED neighborhood fraction from "
                  "GOES: share of 2 km satellite pixels flagged cloudy "
                  "around each point): <b>LOWEST 0 clear</b> transparent "
                  "&rarr; thin cloud <b>dark-blue/blue/cyan</b> &rarr; "
                  "<b>40 green</b> &rarr; <b>55 yellow</b> &rarr; <b>70 orange</b> "
                  "&rarr; <b>82 red</b> &rarr; <b>90 magenta</b> &rarr; "
                  "<b>HIGHEST+ 100 overcast</b> dark-purple. Same % always shows the same "
                  "color; clear areas stay see-through. Satellite retrieval where "
                  "GOES is valid, model fill elsewhere (see provenance).")
    meta = base_metadata(
        PRODUCT, CONFIG_NA["title"], CONFIG_NA["freshness_label"],
        CONFIG_NA["source_name"], CONFIG_NA["source_url"], CONFIG_NA["variable"],
        data_time_utc=data_time, source_last_modified_utc=json.dumps(prov)[:300],
        units="%", source_resolution="GOES FD ~2km; RRFS 13km NA; canvas 3300x1600",
        color_min=0.0, color_max=100.0, color_units="%",
        missing_data_treatment="GOES BCM/DQF==0 valid > RRFS TCDC > GDPS (verified gaps) > transparent; missing never 0%.")
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    src_id = f"na-cloud-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M}"
    meta["source_id"] = src_id
    meta["source_version"] = source_token(src_id + json.dumps(prov, sort_keys=True))
    meta["na_bounds"] = bounds
    meta["stats"] = {"satellite_pixels": n_sat, "model_fill_pixels": n_mod}
    token = meta["source_version"]
    folder_html = (f"<h2>{CONFIG_NA['title']}</h2><p>{scale_html}</p>"
                   f"<p><b>Precedence:</b> {CONFIG_NA['precedence']}<br><b>Provenance:</b> {prov}</p>")
    meta["folder_html"] = folder_html
    write_metadata(stage_prod, meta)
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    outs = live_out_dirs(stage, KML_FILE)
    kml_text = build_kml(PRODUCT, KML_FILE, OVERLAY_NAME, f"{PRODUCT}/current.png",
                         f"{PRODUCT}/legend.png",
                         description_html(CONFIG_NA["title"], meta, SKIP_NOTE, block),
                         CONFIG_NA["refresh_interval_seconds"], token,
                         folder=(CONFIG_NA["title"], folder_html),
                         out_dirs=outs["live"], meta=meta, bounds=bounds)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(PRODUCT, KML_FILE, OVERLAY_NAME,
                    entry_description_html(CONFIG_NA["title"], meta, SKIP_NOTE),
                    CONFIG_NA["refresh_interval_seconds"], out_dirs=outs["entry"])
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": data_time, "source_id": src_id,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
