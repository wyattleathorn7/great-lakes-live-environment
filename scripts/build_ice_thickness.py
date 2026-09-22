"""Pipeline D — LIVE ICE THICKNESS (independent).

USNIC NAIS daily SIGRID-3 shapefile -> validate -> concentration-weighted
WMO stage-range midpoints (DOCUMENTED DERIVED ESTIMATES, inches) ->
clip to Great Lakes -> continuous light-blue->purple gradient ->
transparent PNG (open water transparent; alpha scaled by concentration) ->
metadata -> KML.

Exit codes: 0 = updated (or skipped, source unchanged); 2 = source/validation
failure (previous valid raster left untouched); 1 = unexpected error.
"""

import hashlib
import json
import math
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_kml,
                       description_html, legend_block,
                       refresh_kml_base_url)
from geospatial_utils import (REPO_ROOT, SITE_DIR, THICK_STOPS, _lut,
                              apply_shoreline_mask, base_metadata, download,
                              draw_legend, load_bounds, promote_stage,
                              read_state, save_png, stage_dir, utcnow_iso,
                              write_metadata, write_state)
from nic_sigrid import (analysis_date_from_name, concentration_alpha,
                        download_nic_shapefile, load_polygons,
                        polygon_thickness, shapefile_base)

PRODUCT = "ice_thickness"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Ice_Thickness.kml"
OVERLAY_NAME = "\U0001F9CA LIVE ICE THICKNESS"
SKIP_NOTE = "Turn on/off independently of the other ice, wave and temperature layers."


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    try:
        zip_path, info = download_nic_shapefile(RAW_DIR, download)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    if info["size_bytes"] < 1_000:
        print(f"[{PRODUCT}] VALIDATION FAILED: file too small "
              f"({info['size_bytes']} bytes). Keeping previous.")
        return 2

    with open(zip_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    # No skip: every run rebuilds (tiles must deploy); identical
    # bytes simply produce no commit. Failures still keep previous.

    try:
        return _build(info, zip_path, digest)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(info, zip_path, digest):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    bounds = load_bounds()

    shp_base = shapefile_base(zip_path, os.path.join(RAW_DIR, "nic_shp_thickness"))
    analysis_date = (analysis_date_from_name(os.path.basename(shp_base))
                     or "unknown (not in NIC filename)")
    polys = load_polygons(shp_base)
    if not polys:
        print(f"[{PRODUCT}] VALIDATION FAILED: no polygons parsed.")
        return 2
    print(f"[{PRODUCT}] NIC analysis date {analysis_date}: "
          f"{len(polys)} polygons")

    thick, cts, overlaps = rasterize_thickness(polys, bounds)
    n_ice = int(np.isfinite(thick).sum())
    ice_frac = float(n_ice / thick.size)
    unknown_polys = sum(1 for p in polys
                        if (p["ct"] or 0) >= 1 and p["poly_type"] in ("I", "")
                        and polygon_thickness(p)[0] is None)
    print(f"[{PRODUCT}] ice pixels={n_ice} overlaps={overlaps} "
          f"polys-without-thickness-estimate={unknown_polys}")
    if n_ice:
        vmax = min(40.0, max(6.0, math.ceil(float(np.nanpercentile(thick, 99.5)))))
        print(f"[{PRODUCT}] thickness range in=[{np.nanmin(thick):.1f},"
              f"{np.nanmax(thick):.1f}] color max={vmax} in")
    else:
        vmax = 12.0
        print(f"[{PRODUCT}] no ice in current analysis (off-season valid-empty).")

    lut = np.array(_lut(THICK_STOPS), dtype=np.uint8)
    rgba = np.zeros((thick.shape[0], thick.shape[1], 4), dtype=np.uint8)
    ok = np.isfinite(thick)
    if np.any(ok):
        t = np.clip(thick[ok] / vmax, 0.0, 1.0)
        rgba[ok, 0:3] = lut[(t * 255).astype(int)]
        avec = np.vectorize(lambda c: concentration_alpha(int(c)))(cts[ok])
        rgba[ok, 3] = avec.astype(np.uint8)
    rgba = apply_shoreline_mask(rgba)  # one shared GSHHG shoreline for all

    save_png(rgba, os.path.join(stage_prod, "current.png"))

    subtitle = (f"{CONFIG['freshness_label']}  |  NIC analysis: {analysis_date}")
    lw, lh = draw_legend(os.path.join(stage_prod, "legend.png"),
                         CONFIG["title"], subtitle, "inches",
                         0.0, vmax, THICK_STOPS,
                         "Source: USNIC NAIS daily analysis (WMO stage-derived)  |  "
                         f"Processed {utcnow_iso()}",
                         fmt="{:.0f}",
                         transparent_note="Transparent where no ice (alpha ~ concentration).")

    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=(f"NIC analysis date {analysis_date} (from NIC filename; "
                       f"retrieved {utcnow_iso()})"),
        source_last_modified_utc=info["http_last_modified"] or "unknown (NIC sends none)",
        units="in (display); derived from WMO stage ranges in cm",
        source_resolution="NIC SIGRID-3 vector analysis (variable polygon size)",
        color_min=0.0, color_max=vmax, color_units="in",
        missing_data_treatment=("open water (CT<1) and stages without an "
                                "authoritative thickness (brash/unstaged/old/glacier/unknown) "
                                "rendered fully transparent; NEVER zero-filled or guessed."))
    meta["legend_size"] = [lw, lh]
    scale_html = (f"Ice thickness (inches, WMO stage-derived estimates): "
                  f"thinner <b>0 in</b> (light blue) → blue → deep blue → "
                  f"thicker <b>{vmax:g} in</b> (purple). No color is painted "
                  f"where there is no ice; fainter = partial concentration.")
    meta["legend_scale_html"] = scale_html
    meta["data_nature"] = CONFIG["data_nature"]
    meta["methodology"] = (
        "Per SIGRID-3 polygon: thickness = sum(partial_conc_i * WMO_stage_midpoint_i) "
        "/ sum(partial_conc_i) over SA/SB/SC stages with authoritative ranges "
        "(see nic_sigrid.STAGE_TABLE); cm/2.54 -> inches; pixel alpha scaled by "
        "total concentration CT/10. Off-season ice-free analyses are valid-empty.")
    meta["stats"] = {
        "analysis_date": analysis_date,
        "polygons": len(polys),
        "polygon_overlaps": overlaps,
        "ice_pixels": n_ice,
        "ice_covered_fraction": round(ice_frac, 6),
        "max_in": round(float(np.nanmax(thick)), 2) if n_ice else 0.0,
        "mean_in": round(float(np.nanmean(thick)), 3) if n_ice else 0.0,
        "polys_without_thickness_estimate": unknown_polys,
    }
    write_metadata(stage_prod, meta)

    token = meta["processing_time_utc"].replace(" ", "_").replace(":", "")
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        out_dirs=[os.path.join(stage, "kml", KML_FILE),
                  os.path.join(stage, "site", "kml", KML_FILE)])
    assert_no_vector_geometry(kml_text)

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"content_sha256": digest,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def rasterize_thickness(polys, bounds):
    from geospatial_utils import rasterize_polygons
    return rasterize_polygons(polys, bounds, polygon_thickness)


if __name__ == "__main__":
    sys.exit(main())
