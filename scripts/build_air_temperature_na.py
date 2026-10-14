"""Pipeline K-NA — LIVE AIR TEMPERATURE, NORTH AMERICA (independent).

Sources (explicit precedence, never blended):
  1) NOAA/NCEP HRRR 3km TMP 2m analysis (verified CONUS/Alaska domain).
  2) NOAA RRFS North American output (parallel feed com/rrfs/para until
     prod HTTPS opens; 13km NA grid + 3km conus/ak nests; TMP 2m anl).
  3) ECCC GDPS 15km AirTemp 2m (GeoMet GDPS_15km_AirTemp_2m) ONLY where
     both above lack verified valid data.
  Transparent where no source is valid. Stale/missing never presented
  as current (timestamps verified before render).

Rendering identical to the Great Lakes layer: K->F, FIXED banded 5F key
-60..150F (TEMP_BANDS from build_air_temperature), same opacity rules.

Exit codes: 0 updated; 2 source failure (previous kept); 1 error.
"""

import datetime as dt
import json
import os
import sys
import traceback
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from na_bounds import load_na_bounds
from na_common import (apply_coverage_mask, bin_points_to_canvas,
                         despeckle_valid, feather_merge, fetch_http,
                         fill_pinholes, gate_field_to_coverage,
                         gdps_wcs_grid, sanitize_model_vals, smooth_field)
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs)
from build_air_temperature import (KEY_ZONES, SCALE_HTML, TEMP_BANDS,
                                     TEMP_SMOOTH_STOPS, draw_banded_key)
from gradient_scale import render_rgba
from geospatial_utils import (REPO_ROOT, SITE_DIR, base_metadata, now_det_str,
                              promote_stage, read_state, save_png,
                              source_token, stage_dir, write_metadata,
                              write_state)

PRODUCT = "air_temperature_na"
CONFIG_NA = json.load(open(os.path.join(REPO_ROOT, "config", "air_temperature_na.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Continental_Live_Air_Temperature.kml"
OVERLAY_NAME = "\U0001f321\ufe0f LIVE AIR TEMPERATURE"
SKIP_NOTE = "Turn on/off independently of all other layers."
K2F = lambda k: (k - 273.15) * 9.0 / 5.0 + 32.0

RRFS_PARA = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/para"
RRFS_PROD = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod"


def rrfs_newest_tmp(out_dir, grids=("na", "conus", "ak")):
    """Newest RRFS TMP-2m analysis from prod (preferred) else para.

    Returns list of (label, vals, lats, lons, cycle_id). Raises if none.
    """
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc)
    files = {"na": "rrfs.t{c}z.2dfld.13km.f000.na.grib2",
             "conus": "rrfs.t{c}z.2dfld.3km.f000.conus.grib2",
             "ak": "rrfs.t{c}z.2dfld.3km.f000.ak.grib2"}
    out = []
    for g in grids:
        got = False
        for back in range(0, 10):
            t = now - _dt.timedelta(hours=back)
            dd = t.strftime("rrfs.%Y%m%d")
            cc = t.strftime("%H") if g == "na" else f"{(int(t.strftime('%H'))//6)*6:02d}"
            for base in (RRFS_PROD, RRFS_PARA):
                fn = files[g].format(c=cc)
                idx_url = f"{base}/{dd}/{cc}/{fn}.idx"
                try:
                    idx = fetch_http(idx_url, timeout=30).decode("utf-8", errors="replace")
                except Exception:
                    continue
                from na_common import eccodes_read_idx_ranges
                try:
                    rng = eccodes_read_idx_ranges(idx, [("TMP", "2 m above ground")])
                except ValueError:
                    continue
                a, b = rng["TMP"]
                dest = os.path.join(out_dir, f"rrfs_{g}_tmp.grib2")
                os.makedirs(out_dir, exist_ok=True)
                req = urllib.request.Request(
                    f"{base}/{dd}/{cc}/{fn}",
                    headers={"User-Agent": "great-lakes-live-environment/1.0",
                             "Range": f"bytes={a}-{b-1}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, dest)
                from hrrr import read_messages
                vals, lats, lons, ddate, dtime = read_messages(dest, {"TMP": 1})["TMP"]
                lons = ((np.asarray(lons) + 180) % 360) - 180
                out.append((f"rrfs-{g} {dd}/{cc}", np.asarray(vals, dtype=float),
                            np.asarray(lats, dtype=float), np.asarray(lons, dtype=float),
                            f"{dd}-{cc}"))
                got = True
                break
            if got:
                break
    if not out:
        raise RuntimeError("no RRFS TMP-2m analysis found (prod+para)")
    return out


def hrrr_tmp():
    from hrrr import fetch_messages, latest_cycle, read_messages
    url, dd, cc = latest_cycle()
    dest = os.path.join(RAW_DIR, "na_hrrr_tmp.grib2")
    fetch_messages(url, dest, [("TMP", "2 m above ground")])
    vals, lats, lons, ddate, dtime = read_messages(dest, {"TMP": 1})["TMP"]
    lons = ((np.asarray(lons) + 180) % 360) - 180
    return (f"hrrr {dd}/{cc}", np.asarray(vals, dtype=float),
            np.asarray(lats, dtype=float), np.asarray(lons, dtype=float))


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
    prev = read_state(PRODUCT)
    try:
        return _build(bounds, W, H, prev)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] FAILED: {type(e).__name__}: {e}. Keeping previous.")
        return 2


def _build(bounds, W, H, prev):
    # Coverage switch (config coverage_mode, default landonly: the crisp
    # NA land zone with no sea fringe, exactly as published).
    cover = CONFIG_NA.get("coverage_mode", "landonly")
    from na_bounds import load_global_bounds
    gbounds = load_global_bounds()
    GW, GH = gbounds["canvas_width"], gbounds["canvas_height"]
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    os.makedirs(stage_prod, exist_ok=True)

    layers = []  # (grid, valid) in precedence order: HRRR > RRFS > GDPS
    glayers = []  # same precedence on the worldwide canvas
    prov = {}
    # 1) HRRR (highest precedence).
    try:
        label, v, la, lo = hrrr_tmp()
        v = sanitize_model_vals(v, 180.0, 330.0)
        g, c = bin_points_to_canvas(K2F(v), la, lo, bounds)
        g = fill_pinholes(g, radius=3)
        layers.append((g, np.isfinite(g)))
        gg, _ = bin_points_to_canvas(K2F(v), la, lo, gbounds)
        gg = fill_pinholes(gg, radius=3)
        glayers.append((gg, np.isfinite(gg)))
        prov["hrrr"] = f"{label} n={int(np.isfinite(g).sum())}/g={int(np.isfinite(gg).sum())}"
    except Exception as e:
        prov["hrrr"] = f"unavailable: {type(e).__name__}"
    # 2) RRFS nests + NA grid (same model family: first-valid among them).
    try:
        rg = np.full((H, W), np.nan)
        rgg = np.full((GH, GW), np.nan)
        for label, v, la, lo, cyc in rrfs_newest_tmp(RAW_DIR):
            v = sanitize_model_vals(v, 180.0, 330.0)
            g, c = bin_points_to_canvas(K2F(v), la, lo, bounds)
            m = np.isfinite(g) & ~np.isfinite(rg)
            rg[m] = g[m]
            gg2, _ = bin_points_to_canvas(K2F(v), la, lo, gbounds)
            mg = np.isfinite(gg2) & ~np.isfinite(rgg)
            rgg[mg] = gg2[mg]
            prov[f"rrfs_{label}"] = f"n={int(m.sum())}/g={int(mg.sum())}"
        rg = fill_pinholes(rg, radius=3)
        layers.append((rg, np.isfinite(rg)))
        rgg = fill_pinholes(rgg, radius=3)
        glayers.append((rgg, np.isfinite(rgg)))
    except Exception as e:
        prov["rrfs"] = f"unavailable: {type(e).__name__}"
    # 3) GDPS fallback: ECCC GDPS 15km AirTemp 2m via WCS — ONLY where
    #    HRRR+RRFS have no verified valid data. Worldwide on the global
    #    canvas (same WCS, global bbox), NA gaps on the NA canvas.
    def _gdps_tmp(target_bounds):
        from na_common import resample_wcs_to_canvas
        arr, la, lo, lat1d, lon1d = gdps_wcs_grid("GDPS_15km_AirTemp_2m", target_bounds, RAW_DIR)
        band0 = sanitize_model_vals(np.asarray(arr[0], dtype=float), -80.0, 60.0)
        g = resample_wcs_to_canvas(band0 + 273.15, lat1d, lon1d, target_bounds)
        g = K2F(np.where(np.isfinite(g), g, np.nan))
        return fill_pinholes(g, radius=3)
    try:
        g = _gdps_tmp(bounds)
        layers.append((g, np.isfinite(g)))
        gg = _gdps_tmp(gbounds)
        glayers.append((gg, np.isfinite(gg)))
        prov["gdps"] = f"GDPS_15km_AirTemp_2m n={int(np.isfinite(g).sum())}/g={int(np.isfinite(gg).sum())}"
    except Exception as e:
        prov["gdps"] = f"unavailable: {type(e).__name__}: {e}"
    if not layers or not any(np.isfinite(g).any() for g, _ in layers):
        raise RuntimeError(f"no HRRR/RRFS/GDPS temperature data ({prov})")
    # Seamless cross-fade where domains overlap (no rectangular seams),
    # then one shared treatment: despeckle, display smoothing, coast gate.
    f_grid, own = feather_merge(layers, feather_px=12)
    own = despeckle_valid(own, min_size=4) & np.isfinite(f_grid)
    f_grid[~own] = np.nan
    n_raw = int(own.sum())
    disp_full = smooth_field(f_grid, size=5, passes=2)
    print(f"[{PRODUCT}] raw px={n_raw} prov={prov}")
    # Worldwide merge on the global canvas (same precedence, tighter
    # feather at 0.10 deg/px).
    gf_grid, gown = feather_merge(glayers, feather_px=6)
    gown = despeckle_valid(gown, min_size=4) & np.isfinite(gf_grid)
    gf_grid[~gown] = np.nan
    disp_global = smooth_field(gf_grid, size=5, passes=2)

    from build_kml import (COVERAGE_DIR_SUFFIX, COVERAGE_LABELS,
                           COVERAGE_DISPLAY, coverage_selector_html)
    default_cover = CONFIG_NA.get("coverage_mode", "landonly")
    if default_cover == "extended200":
        default_cover = "global"
    variant_dirs = {m: f"{PRODUCT}_{COVERAGE_DIR_SUFFIX[m]}"
                    for m in COVERAGE_DISPLAY}
    variant_titles = {m: f"{CONFIG_NA['title']} — {COVERAGE_LABELS[m]}"
                      for m in COVERAGE_DISPLAY}
    VARIANT_BOUNDS = {"basin": bounds, "landonly": gbounds, "global": gbounds,
                      "extended200": gbounds}
    data_time = now_det_str()
    scale_html = SCALE_HTML
    src_id = f"na-tmp-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M}"
    token = source_token(src_id + json.dumps(prov, sort_keys=True))

    # Per-coverage rasters: basin on NA canvas; continental + global on
    # worldwide canvas (continental gated to worldwide land).
    for mode in COVERAGE_DISPLAY:
        src = disp_global if mode in ("global", "landonly") else disp_full
        disp_m = gate_field_to_coverage(src, mode)
        n_valid_m = int(np.isfinite(disp_m).sum())
        print(f"[{PRODUCT}] variant {mode}: px={n_valid_m}")
        rgba_m = render_rgba(np.clip(disp_m, -60.0, 150.0), TEMP_SMOOTH_STOPS, 205)
        rgba_m = apply_coverage_mask(rgba_m, mode)
        vprod = variant_dirs[mode]
        vtitle = variant_titles[mode]
        vdir = os.path.join(stage, "site", vprod)
        os.makedirs(vdir, exist_ok=True)
        save_png(rgba_m, os.path.join(vdir, "current.png"))
        vmeta = base_metadata(
            vprod, vtitle, CONFIG_NA["freshness_label"],
            CONFIG_NA["source_name"], CONFIG_NA["primary_url"],
            CONFIG_NA["variable"],
            data_time_utc=data_time,
            source_last_modified_utc=json.dumps(prov)[:300],
            units="degF",
            source_resolution="HRRR 3km; RRFS 13km NA / 3km nests; GDPS 15km fallback",
            color_min=-60.0, color_max=150.0, color_units="degF",
            missing_data_treatment="HRRR valid > RRFS valid > GDPS (verified gaps only) > transparent; never blended.")
        vmeta["legend_scale_html"] = scale_html
        vmeta["source_id"] = src_id
        vmeta["source_version"] = token
        vmeta["na_bounds"] = VARIANT_BOUNDS[mode]
        vmeta["coverage_mode"] = mode
        vmeta["stats"] = {"valid_pixels": n_valid_m, "raw_pixels": n_raw}
        vmeta["folder_html"] = (
            f"<h2>{vtitle}</h2>"
            f"<p><b>Coverage:</b> {COVERAGE_LABELS[mode]}<br>"
            f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br>"
            f"<b>Provenance:</b> {prov}</p>") + coverage_selector_html(mode)
        write_metadata(vdir, vmeta)

    disp_src = disp_global if default_cover in ("global", "landonly") else disp_full
    disp = gate_field_to_coverage(disp_src, default_cover)
    n_valid = int(np.isfinite(disp).sum())
    def_h, def_w = (GH, GW) if default_cover in ("global", "landonly") else (H, W)
    print(f"[{PRODUCT}] display px={n_valid} "
          f"({100.0*n_valid/(def_w*def_h):.2f}%) prov={prov}")

    rgba = render_rgba(np.clip(disp, -60.0, 150.0), TEMP_SMOOTH_STOPS, 205)
    rgba = apply_coverage_mask(rgba, default_cover)
    save_png(rgba, os.path.join(stage_prod, "current.png"))

    # Original banded key: identical layout, bars, and zone callouts.
    lw, lh = draw_banded_key(
        os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"],
        f"2 m air temperature (degF)  |  {data_time}",
        TEMP_BANDS, KEY_ZONES,
        f"Source: NOAA HRRR + RRFS-NA + ECCC GDPS (see provenance)  |  "
        f"Processed {now_det_str()}")
    meta = base_metadata(
        PRODUCT, CONFIG_NA["title"], CONFIG_NA["freshness_label"],
        CONFIG_NA["source_name"], CONFIG_NA["primary_url"], CONFIG_NA["variable"],
        data_time_utc=data_time, source_last_modified_utc=json.dumps(prov)[:300],
        units="degF", source_resolution="HRRR 3km; RRFS 13km NA / 3km nests; GDPS 15km fallback",
        color_min=-60.0, color_max=150.0, color_units="degF",
        missing_data_treatment="HRRR valid > RRFS valid > GDPS (verified gaps only) > transparent; never blended.")
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["source_id"] = src_id
    meta["source_version"] = token
    meta["na_bounds"] = VARIANT_BOUNDS[default_cover]
    meta["coverage_mode"] = default_cover
    meta["stats"] = {"valid_pixels": n_valid, "raw_pixels": n_raw}
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
                         extra_overlays=extras, folder_radio=True)
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
