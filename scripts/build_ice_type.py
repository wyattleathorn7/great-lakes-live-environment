"""Pipeline E — LIVE ICE TYPE (independent).

USNIC NAIS daily SIGRID-3 shapefile -> validate -> predominant WMO stage of
development (SA/SB/SC by partial concentration) -> clip to Great Lakes ->
categorical ice-type raster (one exact color per WMO category, full key) ->
transparent PNG (open water transparent; alpha scaled by concentration) ->
metadata -> KML.

Exit codes: 0 = updated (or skipped, source unchanged); 2 = source/validation
failure (previous valid raster left untouched); 1 = unexpected error.
"""

import hashlib
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_kml,
                       description_html, legend_block,
                       refresh_kml_base_url)
from geospatial_utils import (REPO_ROOT, SITE_DIR, apply_shoreline_mask,
                              base_metadata, download,
                              draw_category_legend, load_bounds,
                              promote_stage, rasterize_polygons, read_state,
                              save_png, stage_dir, utcnow_iso,
                              write_metadata, write_state)
from nic_sigrid import (TYPE_COLORS, TYPE_ORDER, analysis_date_from_name,
                        concentration_alpha, download_nic_shapefile,
                        hex_to_rgb, load_polygons, polygon_type,
                        shapefile_base)

PRODUCT = "ice_type"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Ice_Type.kml"
OVERLAY_NAME = "\U0001F9CA LIVE ICE TYPE"
SKIP_NOTE = "Turn on/off independently of the other ice, wave and temperature layers."
UNKNOWN_CODE = 999.0


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


def _encode(poly):
    code, ct = polygon_type(poly)
    if code == 55:
        return None, ct
    if code == "unknown":
        return UNKNOWN_CODE, ct
    return float(code), ct


def _build(info, zip_path, digest):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    bounds = load_bounds()

    shp_base = shapefile_base(zip_path, os.path.join(RAW_DIR, "nic_shp_type"))
    analysis_date = (analysis_date_from_name(os.path.basename(shp_base))
                     or "unknown (not in NIC filename)")
    polys = load_polygons(shp_base)
    if not polys:
        print(f"[{PRODUCT}] VALIDATION FAILED: no polygons parsed.")
        return 2
    print(f"[{PRODUCT}] NIC analysis date {analysis_date}: "
          f"{len(polys)} polygons")

    codes, cts, overlaps = rasterize_polygons(polys, bounds, _encode)
    n_ice = int(np.isfinite(codes).sum())
    seen = sorted(float(c) for c in np.unique(codes[np.isfinite(codes)]))
    n_unknown = int((codes == UNKNOWN_CODE).sum())
    print(f"[{PRODUCT}] ice pixels={n_ice} overlaps={overlaps} "
          f"observed codes={seen} unknown_pixels={n_unknown}")
    if not n_ice:
        print(f"[{PRODUCT}] no ice in current analysis (off-season valid-empty).")

    color_of = {float(c): hex_to_rgb(TYPE_COLORS[c]) for c in TYPE_ORDER}
    color_of[UNKNOWN_CODE] = hex_to_rgb(TYPE_COLORS["unknown"])
    rgba = np.zeros((codes.shape[0], codes.shape[1], 4), dtype=np.uint8)
    ok = np.isfinite(codes)
    for code in list(color_of):
        m = ok & (codes == code)
        if np.any(m):
            rgba[m, 0:3] = color_of[code]
            avec = np.vectorize(lambda c: concentration_alpha(int(c)))(cts[m])
            rgba[m, 3] = avec.astype(np.uint8)
    rgba = apply_shoreline_mask(rgba)  # one shared GSHHG shoreline for all
    save_png(rgba, os.path.join(stage_prod, "current.png"))


    from nic_sigrid import STAGE_TABLE
    rows = [(TYPE_COLORS[c],
             f"{c} — {STAGE_TABLE[c]['name']} ({STAGE_TABLE[c]['range']})")
            for c in TYPE_ORDER]
    rows.append((TYPE_COLORS["unknown"],
                 "?? — Unknown / Undetermined (codes 99, -9)"))
    subtitle = (f"{CONFIG['freshness_label']}  |  NIC analysis: {analysis_date}")
    lw, lh = draw_category_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        rows,
        "Source: USNIC NAIS daily Great Lakes analysis (SIGRID-3 stages)  |  "
        f"Processed {utcnow_iso()}",
        note="Transparent where no ice (alpha ~ concentration).")

    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=(f"NIC analysis date {analysis_date} (from NIC filename; "
                       f"retrieved {utcnow_iso()})"),
        source_last_modified_utc=info["http_last_modified"] or "unknown (NIC sends none)",
        units="WMO SIGRID-3 stage code (categorical)",
        source_resolution="NIC SIGRID-3 vector analysis (variable polygon size)",
        color_min="categorical (see ice_type_categories)",
        color_max="categorical (see ice_type_categories)",
        color_units="category",
        missing_data_treatment=("open water (CT<1) fully transparent; unknown/unstaged "
                                "ice gets the explicit Unknown color, never a real "
                                "type color."))
    meta["legend_size"] = [lw, lh]
    meta["ice_type_categories"] = [
        {"code": c, "name": STAGE_TABLE[c]["name"],
         "range": STAGE_TABLE[c]["range"], "color": TYPE_COLORS[c]}
        for c in TYPE_ORDER
    ] + [{"code": "unknown", "name": "Unknown / Undetermined",
          "range": "codes 99, -9", "color": TYPE_COLORS["unknown"]}]
    cat_rows = "".join(
        f"<span style=\"background:{c['color']};\">&nbsp;&nbsp;&nbsp;</span> "
        f"{c['code']} — {c['name']} ({c['range']})<br/>"
        for c in meta["ice_type_categories"])
    scale_html = (f"Ice type = predominant WMO stage of development per "
                  f"analysis polygon:<br/>{cat_rows}"
                  f"Open water is transparent; fainter = partial concentration.")
    meta["legend_scale_html"] = scale_html
    meta["data_nature"] = CONFIG["data_nature"]
    meta["methodology"] = (
        "Per SIGRID-3 polygon: predominant stage = highest partial concentration "
        "among SA/SB/SC (ties resolve to the thickest-listed stage); rendered in "
        "that stage's fixed key color with pixel alpha scaled by total "
                  "concentration CT/10. The key lists every WMO stage the dataset supports.")
    meta["stats"] = {
        "analysis_date": analysis_date,
        "polygons": len(polys),
        "polygon_overlaps": overlaps,
        "ice_pixels": n_ice,
        "observed_codes": seen,
        "unknown_pixels": n_unknown,
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


if __name__ == "__main__":
    sys.exit(main())
