"""Pipeline C1-NA — LIVE CLOUD COVER, MULTI-SATELLITE IR (independent).

White infrared satellite imagery (day + night) on a transparent globe.
Mosaic precedence (explicit, never substituted); see scripts/sat_ir.py:
  1) GOES-19 + GOES-18 ABI L2 CMIPF Band-13 direct BT (Americas/Pacific).
  2) Himawari-9 Band-13 via GIBS WMTS LUT inversion (Asia-Pacific).
  3) Meteosat FES (0 deg) + IODC (45.5 E) SEVIRI IR10.8 via EUMETView WMS
     grayscale imagery (Europe/Africa/Indian Ocean; imagery-derived).
  4) VIIRS I5 Day/Night via GIBS (daily) ONLY where no GEO valid (poles).
Missing observations stay transparent -- never clear sky, never stretched,
never duplicated, never model fill. No percentage scale: brightness
temperature (or documented imagery) maps to white-cloud brightness/alpha.

RETIRED: the BCM-% gradient + RRFS/GDPS model-fill path (_build below) is
kept for reference only and is never selected as the production path.
Production uses _build_mosaic (white IR). See config/cloud_cover_na.json.

Exit codes: 0 updated (or skipped when scans unchanged); 2 source failure
(previous kept); 1 error.
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
from na_common import (apply_coverage_mask, bin_points_to_canvas,
                         despeckle_valid, feather_merge, fetch_http,
                         fill_pinholes, gate_field_to_coverage,
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


MOSAIC_VERSION = 6  # bump when sat_ir transfer/merge changes (forces rebuild)


def run():
    try:
        return _build_mosaic()
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
    from na_bounds import load_global_bounds
    gbounds = load_global_bounds()
    GW, GH = gbounds["canvas_width"], gbounds["canvas_height"]
    # EXPERIMENT (reversible): realistic white clouds instead of the
    # 0-100% gradient. The full gradient pipeline below is untouched;
    # flip config render_style to "gradient" to restore it (original
    # legend archived at archive/cloud_cover_gradient_legend.png).
    if CONFIG_NA.get("render_style", "realistic") == "realistic":
        return _build_realistic(bounds, W, H, gbounds, GW, GH,
                                stage, stage_prod)
    prov = {}
    sat_grids = []
    gsat_grids = []

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
            gg, _ = bin_points_to_canvas(cloudy, lats, lons, gbounds, agg="mean",
                                         splat_radius=2)
            gsat_grids.append(gg)
            prov[name] = f"{scan:%Y-%m-%d %H:%MZ} valid-cells={int(np.isfinite(g).sum())}/g={int(np.isfinite(gg).sum())}"
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
    gfield = np.full((GH, GW), np.nan)
    gsat_valid = np.zeros((GH, GW), dtype=bool)
    if gsat_grids:
        gstack = np.stack([np.where(np.isfinite(g), g, np.nan) for g in gsat_grids])
        with np.errstate(invalid="ignore"):
            gfrac = np.nanmean(gstack, axis=0)
        gm = np.isfinite(gfrac)
        gfield[gm] = gfrac[gm] * 100.0
        gsat_valid[gm] = True
    glayers = [(gfield, gsat_valid)]
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
                gg, _ = bin_points_to_canvas(vals, np.asarray(lats, float),
                                             np.asarray(lons, float), gbounds)
                gg = fill_pinholes(gg, radius=3)
                gm = np.isfinite(gg)
                grrfs_grid = np.full((GH, GW), np.nan)
                grrfs_grid[gm] = np.clip(gg[gm], 0, 100)
                grrfs_grid = smooth_field(grrfs_grid, size=5, passes=2)
                glayers.append((grrfs_grid, np.isfinite(grrfs_grid)))
                rrfs_grid[m] = np.clip(g[m], 0, 100)
                rrfs_valid[m] = True
                # Display-smoothing on the MODEL layer only: the coarse
                # 13 km grid must not print cell edges. The GOES-observed
                # layer below is NEVER smoothed (its raw 2 km grain is the
                # original layer's look).
                rrfs_grid = smooth_field(rrfs_grid, size=5, passes=2)
                rrfc = int(m.sum())
                prov["RRFS-TCDC"] = f"{dd} n={rrfc}/g={int(gm.sum())}"
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
    # config already promises it). Worldwide on the global canvas via a
    # second WCS bbox. Bilinear resample so coarse cells render
    # as gradient, never dots.
    def _gdps_tcc(target_bounds):
        from na_common import gdps_wcs_grid, resample_wcs_to_canvas
        arr, _la, _lo, slat, slon = gdps_wcs_grid(
            "GDPS_15km_TotalCloudCover", target_bounds, RAW_DIR)
        raw = np.asarray(arr[0], dtype=float)
        if np.isfinite(raw).any() and np.nanmax(raw) <= 1.0:
            raw = raw * 100.0  # fraction -> %
        g = resample_wcs_to_canvas(
            sanitize_model_vals(raw, 0.0, 100.0), slat, slon, target_bounds)
        return smooth_field(fill_pinholes(np.clip(g, 0.0, 100.0), radius=3),
                            size=7, passes=2)
    try:
        g = _gdps_tcc(bounds)
        layers.append((g, np.isfinite(g)))
        gg = _gdps_tcc(gbounds)
        glayers.append((gg, np.isfinite(gg)))
        prov["GDPS-TCC"] = f"GDPS_15km_TotalCloudCover n={int(np.isfinite(g).sum())}/g={int(np.isfinite(gg).sum())}"
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
    gfield_merged, _gv = feather_merge(glayers, feather_px=12)
    from na_common import blend_antimeridian as _blend_aml
    gfield_merged = _blend_aml(gfield_merged)
    default_cover = CONFIG_NA.get("coverage_mode", "global")
    if default_cover == "extended200":
        default_cover = "global"
    # Shared cache for the precipitation composite: the UNGATED merged %
    # field (NaN = missing/clear) + build time (precipitation_na applies
    # its own coverage mask, so the cache must not be pre-clipped).
    try:
        np.savez_compressed(os.path.join(RAW_DIR, "na_cloud_field.npz"),
                            field=field.astype(np.float32),
                            built_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except Exception as e:
        print(f"[{PRODUCT}] WARNING: cloud field cache write failed: {e}")
    n_sat = int(sat_valid.sum())
    if np.isfinite(field).sum() == 0:
        raise RuntimeError(f"no GOES-FD or RRFS cloud data ({prov})")

    from build_kml import (COVERAGE_DIR_SUFFIX, COVERAGE_LABELS,
                           COVERAGE_DISPLAY, coverage_selector_html)
    variant_dirs = {m: f"{PRODUCT}_{COVERAGE_DIR_SUFFIX[m]}"
                    for m in COVERAGE_DISPLAY}
    variant_titles = {m: f"{CONFIG_NA['title']} — {COVERAGE_LABELS[m]}"
                      for m in COVERAGE_DISPLAY}
    VARIANT_BOUNDS = {"basin": bounds, "landonly": gbounds, "global": gbounds,
                      "extended200": gbounds}
    data_time = now_det_str()
    scale_html = ("Cloud cover (%, OBSERVED neighborhood fraction from "
                  "GOES: share of 2 km satellite pixels flagged cloudy "
                  "around each point): <b>LOWEST 0 clear</b> transparent "
                  "&rarr; thin cloud <b>dark-blue/blue/cyan</b> &rarr; "
                  "<b>40 green</b> &rarr; <b>55 yellow</b> &rarr; <b>70 orange</b> "
                  "&rarr; <b>82 red</b> &rarr; <b>90 magenta</b> &rarr; "
                  "<b>HIGHEST+ 100 overcast</b> dark-purple. Same % always shows the same "
                  "color; clear areas stay see-through. Satellite retrieval where "
                  "GOES is valid, model fill elsewhere (see provenance).")
    src_id = f"na-cloud-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M}"
    token = source_token(src_id + json.dumps(prov, sort_keys=True))

    # Per-coverage rasters: basin on NA canvas; continental + global on
    # worldwide canvas (continental gated to worldwide land).
    for mode in COVERAGE_DISPLAY:
        src_field = gfield_merged if mode in ("global", "landonly") else field
        src_sat = gsat_valid if mode in ("global", "landonly") else sat_valid
        field_m = gate_field_to_coverage(src_field, mode)
        n_mod = int(np.isfinite(field_m).sum()
                    - (np.isfinite(field_m) & src_sat).sum())
        print(f"[{PRODUCT}] variant {mode}: model-fill px={n_mod}")
        rgba_m = render_rgba(field_m, CLOUD_STOPS, 205)
        clear_m = np.isfinite(field_m) & (field_m < 1.0)
        rgba_m[clear_m, 3] = 0
        rgba_m = apply_coverage_mask(rgba_m, mode)
        vprod = variant_dirs[mode]
        vtitle = variant_titles[mode]
        vdir = os.path.join(stage, "site", vprod)
        os.makedirs(vdir, exist_ok=True)
        save_png(rgba_m, os.path.join(vdir, "current.png"))
        vmeta = base_metadata(
            vprod, vtitle, CONFIG_NA["freshness_label"],
            CONFIG_NA["source_name"], CONFIG_NA["source_url"],
            CONFIG_NA["variable"],
            data_time_utc=data_time,
            source_last_modified_utc=json.dumps(prov)[:300],
            units="%",
            source_resolution="GOES FD ~2km; RRFS 13km NA; canvas 3300x1600",
            color_min=0.0, color_max=100.0, color_units="%",
            missing_data_treatment="GOES BCM/DQF==0 valid > RRFS TCDC > GDPS (verified gaps) > transparent; missing never 0%.")
        vmeta["legend_scale_html"] = scale_html
        vmeta["source_id"] = src_id
        vmeta["source_version"] = token
        vmeta["na_bounds"] = VARIANT_BOUNDS[mode]
        vmeta["coverage_mode"] = mode
        vmeta["stats"] = {"satellite_pixels": n_sat,
                          "model_fill_pixels": n_mod}
        vmeta["folder_html"] = (
            f"<h2>{vtitle}</h2>"
            f"<p><b>Coverage:</b> {COVERAGE_LABELS[mode]}<br>"
            f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br>"
            f"<b>Provenance:</b> {prov}</p>") + coverage_selector_html(mode)
        write_metadata(vdir, vmeta)

    src_field = gfield_merged if default_cover in ("global", "landonly") else field
    src_sat = gsat_valid if default_cover in ("global", "landonly") else sat_valid
    field = gate_field_to_coverage(src_field, default_cover)
    n_mod = int(np.isfinite(field).sum() - (np.isfinite(field) & src_sat).sum())
    n_sat = int(src_sat.sum())
    print(f"[{PRODUCT}] sat px={n_sat} model-fill px={n_mod} prov={prov}")

    rgba = render_rgba(field, CLOUD_STOPS, 205)
    # Clear (<1%) and missing stay transparent (GL convention).
    clear = np.isfinite(field) & (field < 1.0)
    rgba[clear, 3] = 0
    rgba = apply_coverage_mask(rgba, default_cover)
    save_png(rgba, os.path.join(stage_prod, "current.png"))

    # Original continuous key: identical stops, labels, and units.
    lw, lh = draw_scale_legend(os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"],
                               f"GOES full-disk + RRFS fill | {data_time}", "%", CLOUD_STOPS, CLOUD_LABELS,
                               f"Source: NOAA GOES-East/GOES-West ABI clear-sky mask + RRFS TCDC fill  |  "
                               f"Processed {data_time}",
                               note="Clear sky (<1%) is transparent, not colored.")
    meta = base_metadata(
        PRODUCT, CONFIG_NA["title"], CONFIG_NA["freshness_label"],
        CONFIG_NA["source_name"], CONFIG_NA["source_url"], CONFIG_NA["variable"],
        data_time_utc=data_time, source_last_modified_utc=json.dumps(prov)[:300],
        units="%", source_resolution="GOES FD ~2km; RRFS 13km NA; canvas 3300x1600",
        color_min=0.0, color_max=100.0, color_units="%",
        missing_data_treatment="GOES BCM/DQF==0 valid > RRFS TCDC > GDPS (verified gaps) > transparent; missing never 0%.")
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["source_id"] = src_id
    meta["source_version"] = token
    meta["na_bounds"] = VARIANT_BOUNDS[default_cover]
    meta["coverage_mode"] = default_cover
    meta["stats"] = {"satellite_pixels": n_sat, "model_fill_pixels": n_mod}
    folder_html = (f"<h2>{CONFIG_NA['title']}</h2>"
                   f"<p><b>Coverage:</b> {COVERAGE_LABELS[default_cover]}<br>"
                   f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br><b>Provenance:</b> {prov}</p>")
    meta["folder_html"] = folder_html + coverage_selector_html(default_cover)
    write_metadata(stage_prod, meta)
    outs = live_out_dirs(stage, KML_FILE)
    extras = [(COVERAGE_LABELS[m], f"{variant_dirs[m]}/current.png",
               VARIANT_BOUNDS[m], 11 + i, 0)
              for i, m in enumerate([m for m in COVERAGE_DISPLAY
                                     if m != default_cover])]
    kml_text = build_kml(PRODUCT, KML_FILE, OVERLAY_NAME, f"{PRODUCT}/current.png",
                         f"{PRODUCT}/legend.png",
                         description_html(CONFIG_NA["title"], meta, SKIP_NOTE, "")
                         + coverage_selector_html(default_cover),
                         CONFIG_NA["refresh_interval_seconds"], token,
                         folder=(CONFIG_NA["title"], meta["folder_html"]),
                         out_dirs=outs["live"], meta=meta,
                         bounds=VARIANT_BOUNDS[default_cover],
                         extra_overlays=extras, folder_radio=True,
                         primary_name=COVERAGE_LABELS[default_cover])
    assert_no_vector_geometry(kml_text)
    build_entry_kml(PRODUCT, KML_FILE, OVERLAY_NAME,
                    entry_description_html(CONFIG_NA["title"], meta, SKIP_NOTE),
                    CONFIG_NA["refresh_interval_seconds"], out_dirs=outs["entry"])
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": data_time, "source_id": src_id,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK realistic ({len(promoted)} files).")
    return 0


def _build_mosaic():
    """Production white-IR mosaic (multi-satellite; see sat_ir.py).

    One scan set is selected first and rendered to all three canvases, so
    basin/continental/global always show the same observations. Basin uses
    the true Great Lakes bounds (1800x1175); continental the NA canvas;
    global the worldwide canvas. Validity-driven transparency (no masks).
    Timestamps: observation_time_utc comes from the scans; processing and
    publication times are recorded separately; unchanged scans skip the
    rebuild without minting a new observation time or version token.
    """
    import sat_ir
    from build_kml import (COVERAGE_DIR_SUFFIX, COVERAGE_LABELS,
                           COVERAGE_DISPLAY, coverage_selector_html)
    from geospatial_utils import (draw_category_legend, load_bounds)
    from na_bounds import load_global_bounds

    bounds_na = load_na_bounds()
    gbounds = load_global_bounds()
    bounds_gl = load_bounds()
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)

    # 1) Select scans once for every coverage (cheap: listings + caps).
    scans, sel_diag = sat_ir.select_scans(raw_dir=RAW_DIR)
    src_id = sat_ir.source_identity(scans) + f"-v{MOSAIC_VERSION}"
    token = source_token(src_id)
    prev = read_state(PRODUCT)
    outputs_exist = all(
        os.path.exists(os.path.join(SITE_DIR, d, "current.png"))
        for d in [PRODUCT, f"{PRODUCT}_basin", f"{PRODUCT}_continental",
                  f"{PRODUCT}_global"]
    ) and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE))
    if (prev.get("source_id") == src_id and outputs_exist):
        print(f"[{PRODUCT}] scans unchanged ({src_id}); keeping rasters.")
        return _refresh_kml_mosaic()
    if not scans:
        raise RuntimeError(f"no IR source available ({sel_diag})")

    # 2) Render every canvas from the same scan set.
    from geospatial_utils import now_det_str as _now
    proc_time = _now()
    CANVASES = {
        # mode -> (bounds, variant dir suffix canvas)
        "basin": bounds_gl, "landonly": bounds_na, "global": gbounds,
    }
    rendered = {}
    for mode in COVERAGE_DISPLAY:
        bnds = CANVASES[mode]
        try:
            rendered[mode] = sat_ir.render_canvas(bnds, scans, RAW_DIR)
        except Exception as e:
            raise RuntimeError(f"canvas {mode} failed: {type(e).__name__}: {e}")

    def _render_white(res):
        from scipy.ndimage import distance_transform_edt
        b, a = res["bright"], res["alpha"]
        rgba = np.zeros((b.shape[0], b.shape[1], 4), dtype=np.uint8)
        m = np.isfinite(b) & np.isfinite(a) & (a > 0)
        # Dissolve valid-coverage edges (disks end mid-ocean); never
        # extends coverage, only fades existing pixels.
        d = distance_transform_edt(m)
        fade = np.clip(d / float(sat_ir.DISK_FADE_PX), 0.0, 1.0)
        rgba[m, 0:3] = np.clip(np.round(b[m]), 0, 255)[:, None].astype(np.uint8)
        rgba[m, 3] = np.clip(np.round(a[m] * fade[m]), 0, 255).astype(np.uint8)
        rgba[m & (rgba[:, :, 3] < 3), 3] = 0
        return rgba, int((rgba[:, :, 3] > 0).sum())

    # Observation time: newest GEO scan (VIIRS-daily excluded); per-source
    # times are all retained in source_observations (no single-time myth).
    geo_scans = [s["scan_dt"] for n, s in scans.items()
                 if n != "VIIRS" and s.get("scan_dt") is not None]
    obs_dt = max(geo_scans) if geo_scans else None
    obs_str = (obs_dt.strftime("%Y-%m-%d %H:%M UTC") if obs_dt
               else "unknown (supplementary only)")
    oldest = min(geo_scans) if geo_scans else None
    now_utc = dt.datetime.now(dt.timezone.utc)
    age_s = int((now_utc - obs_dt).total_seconds()) if obs_dt else None

    def _status(res):
        # "no coverage on this canvas" is nominal (disk footprint), not a
        # failure; only genuine FAILED sources degrade the status.
        bad = [o for o in res["observations"] if "FAILED" in o["detail"]]
        return "degraded" if bad else "current"

    update_status = _status(rendered["global"])
    primary_obs = rendered["global"]["observations"]
    prov_lines = "; ".join(
        f"{o['source']} {o['scan_utc']}" for o in primary_obs)[:600]

    scale_html = ("White = cloud tops (multi-satellite infrared, day + "
                  "night); transparent = clear sky or no satellite coverage. "
                  "Infrared brightness, not a cloud-cover percentage.")
    variant_dirs = {m: f"{PRODUCT}_{COVERAGE_DIR_SUFFIX[m]}"
                    for m in COVERAGE_DISPLAY}
    variant_titles = {m: f"{CONFIG_NA['title']} — {COVERAGE_LABELS[m]}"
                      for m in COVERAGE_DISPLAY}
    VARIANT_BOUNDS = {"basin": bounds_gl, "landonly": bounds_na,
                      "global": gbounds, "extended200": gbounds}

    # Shared white-cloud legend, drawn once and copied to every variant so
    # no stale percentage legend can survive from a previous rendering.
    import shutil as _shutil
    _legend_tmp = os.path.join(stage, "site", PRODUCT, "legend.png")
    os.makedirs(os.path.dirname(_legend_tmp), exist_ok=True)
    _lw, _lh = draw_category_legend(
        _legend_tmp, CONFIG_NA["title"],
        f"Realistic clouds (multi-satellite IR) | {obs_str}",
        [("#FFFFFF", "White = cloud tops (IR, day+night)")],
        f"Source: GOES-19/18 + Himawari-9 + Meteosat + VIIRS supplement  |  "
        f"Observed {obs_str} | Processed {proc_time}",
        note="Transparent = clear sky or no satellite coverage.")

    for mode in COVERAGE_DISPLAY:
        res = rendered[mode]
        rgba_m, n_px = _render_white(res)
        # Validity-driven transparency only: every valid pixel paints
        # (oceans included); no land/ocean masks on a cloud layer.
        vprod, vtitle = variant_dirs[mode], variant_titles[mode]
        vdir = os.path.join(stage, "site", vprod)
        os.makedirs(vdir, exist_ok=True)
        save_png(rgba_m, os.path.join(vdir, "current.png"))
        _shutil.copyfile(_legend_tmp, os.path.join(vdir, "legend.png"))
        print(f"[{PRODUCT}] mosaic variant {mode}: px={n_px} "
              f"skew={res['max_geo_skew_min']}min feather={res['feather_px']}")
        vmeta = base_metadata(
            vprod, vtitle, CONFIG_NA["freshness_label"],
            CONFIG_NA["source_name"], CONFIG_NA["source_url"],
            CONFIG_NA["variable"],
            data_time_utc=obs_str,
            source_last_modified_utc=prov_lines,
            units="IR brightness (white=cold cloud tops; not a percentage)",
            source_resolution=("multi-satellite IR: GOES-19/18 10.3um FD, "
                               "Himawari-9 10.3um, Meteosat SEVIRI 10.8um, "
                               "VIIRS I5 daily supplement"),
            color_min=0.0, color_max=100.0, color_units="display only",
            missing_data_treatment=("transparent = clear sky or no valid "
                                    "observation; never stretched, duplicated, "
                                    "or model-filled."))
        vmeta["legend_scale_html"] = scale_html
        vmeta["source_id"] = src_id
        vmeta["source_version"] = token
        vmeta["na_bounds"] = VARIANT_BOUNDS[mode]
        vmeta["coverage_mode"] = mode
        vmeta["observation_time_utc"] = obs_str
        vmeta["image_age_seconds"] = age_s
        vmeta["update_status"] = _status(res)
        vmeta["source_observations"] = res["observations"]
        vmeta["provenance"] = {o["source"]: o["scan_utc"]
                               for o in res["observations"]}
        vmeta["legend_size"] = [_lw, _lh]
        vmeta["stats"] = {"ir_pixels": n_px,
                           "max_geo_skew_min": res["max_geo_skew_min"],
                           "feather_px": res["feather_px"]}
        vmeta["folder_html"] = (
            f"<h2>{vtitle}</h2>"
            f"<p><b>Coverage:</b> {COVERAGE_LABELS[mode]}<br>"
            f"<b>Observed:</b> {obs_str}<br>"
            f"<b>Provenance:</b> {prov_lines}</p>") + coverage_selector_html(mode)
        # Correct the inherited basin render_bounds (base_metadata assumes
        # the Great Lakes canvas, which is only right for the basin layer).
        vb = VARIANT_BOUNDS[mode]
        vmeta["spatial_resolution_rendered"] = (
            f"{vb['canvas_width']}x{vb['canvas_height']} canvas over "
            f"lon [{vb['lon_min']},{vb['lon_max']}], "
            f"lat [{vb['lat_min']},{vb['lat_max']}]")
        vmeta["render_bounds"] = {"lon_min": vb["lon_min"],
                                  "lon_max": vb["lon_max"],
                                  "lat_min": vb["lat_min"],
                                  "lat_max": vb["lat_max"]}
        write_metadata(vdir, vmeta)

    # Provenance diagnostic (which satellite paints each region).
    try:
        _write_provenance_png(stage, rendered["global"]["prov"])
    except Exception as e:
        print(f"[{PRODUCT}] WARNING: provenance diagnostic failed: {e}")

    res0 = rendered["global"]
    rgba, n_px = _render_white(res0)
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    lw, lh = draw_category_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"],
        f"Realistic clouds (multi-satellite IR) | {obs_str}",
        [("#FFFFFF", "White = cloud tops (IR, day+night)")],
        f"Source: GOES-19/18 + Himawari-9 + Meteosat + VIIRS supplement  |  "
        f"Observed {obs_str} | Processed {proc_time}",
        note="Transparent = clear sky or no satellite coverage.")
    meta = base_metadata(
        PRODUCT, CONFIG_NA["title"], CONFIG_NA["freshness_label"],
        CONFIG_NA["source_name"], CONFIG_NA["source_url"], CONFIG_NA["variable"],
        data_time_utc=obs_str, source_last_modified_utc=prov_lines,
        units="IR brightness (white=cold cloud tops; not a percentage)",
        source_resolution=("multi-satellite IR: GOES-19/18 10.3um FD, "
                           "Himawari-9 10.3um, Meteosat SEVIRI 10.8um, "
                           "VIIRS I5 daily supplement"),
        color_min=0.0, color_max=100.0, color_units="display only",
        missing_data_treatment=("transparent = clear sky or no valid "
                                "observation; never stretched, duplicated, "
                                "or model-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["source_id"] = src_id
    meta["source_version"] = token
    meta["na_bounds"] = VARIANT_BOUNDS["global"]
    meta["coverage_mode"] = "global"
    meta["observation_time_utc"] = obs_str
    meta["image_age_seconds"] = age_s
    meta["update_status"] = update_status
    meta["source_observations"] = primary_obs
    meta["provenance"] = {o["source"]: o["scan_utc"] for o in primary_obs}
    meta["stats"] = {"ir_pixels": n_px,
                     "max_geo_skew_min": res0["max_geo_skew_min"],
                     "feather_px": res0["feather_px"]}
    meta["folder_html"] = (f"<h2>{CONFIG_NA['title']}</h2>"
                           f"<p><b>Coverage:</b> {COVERAGE_LABELS['global']}<br>"
                           f"<b>Observed:</b> {obs_str}<br>"
                           f"<b>Provenance:</b> {prov_lines}</p>")
    meta["folder_html"] = meta["folder_html"] + coverage_selector_html("global")
    gb = VARIANT_BOUNDS["global"]
    meta["spatial_resolution_rendered"] = (
        f"{gb['canvas_width']}x{gb['canvas_height']} canvas over "
        f"lon [{gb['lon_min']},{gb['lon_max']}], "
        f"lat [{gb['lat_min']},{gb['lat_max']}]")
    meta["render_bounds"] = {"lon_min": gb["lon_min"], "lon_max": gb["lon_max"],
                             "lat_min": gb["lat_min"], "lat_max": gb["lat_max"]}
    write_metadata(stage_prod, meta)
    outs = live_out_dirs(stage, KML_FILE)
    extras = [(COVERAGE_LABELS[m], f"{variant_dirs[m]}/current.png",
               VARIANT_BOUNDS[m], 11 + i, 0)
              for i, m in enumerate([m for m in COVERAGE_DISPLAY
                                     if m != "global"])]
    kml_text = build_kml(PRODUCT, KML_FILE, OVERLAY_NAME, f"{PRODUCT}/current.png",
                         f"{PRODUCT}/legend.png",
                         description_html(CONFIG_NA["title"], meta, SKIP_NOTE, "")
                         + coverage_selector_html("global"),
                         CONFIG_NA["refresh_interval_seconds"], token,
                         folder=(CONFIG_NA["title"], meta["folder_html"]),
                         out_dirs=outs["live"], meta=meta,
                         bounds=VARIANT_BOUNDS["global"],
                         extra_overlays=extras, folder_radio=True,
                         primary_name=COVERAGE_LABELS["global"])
    assert_no_vector_geometry(kml_text)
    build_entry_kml(PRODUCT, KML_FILE, OVERLAY_NAME,
                    entry_description_html(CONFIG_NA["title"], meta, SKIP_NOTE),
                    CONFIG_NA["refresh_interval_seconds"], out_dirs=outs["entry"])
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": obs_str, "source_id": src_id,
                          "mosaic_version": MOSAIC_VERSION,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK mosaic ({len(promoted)} files).")
    return 0


def _refresh_kml_mosaic():
    """Skip path: rewrite KMLs from committed metadata without touching
    observation timestamps or version tokens."""
    from geospatial_utils import SITE_DIR as _SITE
    import json as _json
    with open(os.path.join(_SITE, PRODUCT, "metadata.json")) as f:
        meta = _json.load(f)
    from build_kml import (COVERAGE_DIR_SUFFIX, COVERAGE_LABELS,
                           COVERAGE_DISPLAY, coverage_selector_html)
    from na_bounds import load_global_bounds as _glb
    from geospatial_utils import load_bounds as _bnd
    gbounds = _glb()
    VARIANT_BOUNDS = {"basin": _bnd(), "landonly": load_na_bounds(),
                      "global": gbounds, "extended200": gbounds}
    variant_dirs = {m: f"{PRODUCT}_{COVERAGE_DIR_SUFFIX[m]}"
                    for m in COVERAGE_DISPLAY}
    token = meta.get("source_version") or source_token(
        meta.get("source_id", ""))
    outs = live_out_dirs(stage_dir(PRODUCT), KML_FILE)
    extras = [(COVERAGE_LABELS[m], f"{variant_dirs[m]}/current.png",
               VARIANT_BOUNDS[m], 11 + i, 0)
              for i, m in enumerate([m for m in COVERAGE_DISPLAY
                                     if m != "global"])]
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME, f"{PRODUCT}/current.png",
        f"{PRODUCT}/legend.png",
        description_html(CONFIG_NA["title"], meta, SKIP_NOTE, "")
        + coverage_selector_html("global"),
        CONFIG_NA["refresh_interval_seconds"], token,
        folder=(CONFIG_NA["title"], meta.get("folder_html", "")),
        out_dirs=outs["live"], meta=meta, bounds=VARIANT_BOUNDS["global"],
        extra_overlays=extras, folder_radio=True,
        primary_name=COVERAGE_LABELS["global"])
    assert_no_vector_geometry(kml_text)
    build_entry_kml(PRODUCT, KML_FILE, OVERLAY_NAME,
                    entry_description_html(CONFIG_NA["title"], meta, SKIP_NOTE),
                    CONFIG_NA["refresh_interval_seconds"],
                    out_dirs=outs["entry"])
    promoted = promote_stage(PRODUCT)
    print(f"[{PRODUCT}] KML refreshed (scans unchanged, "
          f"{len(promoted)} files).")
    return 0


def _write_provenance_png(stage, prov):
    """Diagnostic: which satellite paints each global pixel."""
    from PIL import Image
    colors = {0: (0, 0, 0, 0), 1: (230, 57, 70, 255), 2: (241, 143, 1, 255),
              3: (46, 134, 193, 255), 4: (39, 174, 96, 255),
              5: (155, 89, 182, 255), 6: (149, 165, 166, 255)}
    h, w = prov.shape
    rgb = np.zeros((h, w, 4), dtype=np.uint8)
    for code, col in colors.items():
        rgb[prov == code] = col
    out = os.path.join(stage, "site", PRODUCT, "provenance.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(rgb, mode="RGBA").save(out)


if __name__ == "__main__":
    sys.exit(main())
