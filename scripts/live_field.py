"""Shared finish/skip engine for the eight live environmental gradients.

All eight products inherit the exact LIVE LEAF COLOR coverage by reading
the single authoritative canvas in config/great_lakes_bounds.json via
load_bounds() (Leaf Color itself is never touched). Each builder fetches
its own authoritative source, bins it onto that canvas, and calls
finish() here for the identical render -> legend -> metadata -> KML ->
stage/promote -> state contract the proven HRRR products use.

KML rule (repo-wide, asserted): one GroundOverlay + one self-refreshing
entry NetworkLink; zero LineString/Polygon/Placemark/Point/ScreenOverlay.
Legends travel as legend.png in the product folder + inside the KML
description (ScreenOverlay is rejected by some Google Earth clients).
"""

import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR,
                              base_metadata, bin_to_canvas, canvas_indices,
                              load_bounds, now_det_str, promote_stage,
                              read_state, save_png, source_token, stage_dir,
                              write_metadata, write_state)
from gradient_scale import draw_scale_legend, render_rgba

SKIP_NOTE = "Turn on/off independently of all other layers."


def should_skip(product, source_id):
    """True when the source is unchanged and a full raster is deployed."""
    from glob import glob as _g
    prev = read_state(product)
    if prev.get("source_id") != source_id:
        return False
    if prev.get("render_version") != RENDER_VERSION:
        return False
    if not os.path.exists(os.path.join(SITE_DIR, product, "current.png")):
        return False
    return True


def refresh_kml(product, kml_file, overlay_name, title, refresh_interval):
    try:
        with open(os.path.join(REPO_ROOT, "config", f"{product}.json")) as f:
            cfg = json.load(f)
    except OSError:
        cfg = {"title": title}
    try:
        refresh_kml_base_url(product, kml_file, overlay_name,
                             cfg.get("title", title), SKIP_NOTE,
                             refresh_interval)
        print(f"[{product}] KML base URLs refreshed.")
    except Exception as e:
        print(f"[{product}] WARNING: KML refresh failed: {e}")
    return 0


def folder_html_3para(title, para_gradient, para_field, scale_html,
                      legend_src, source_name, source_url, update_line,
                      units_line):
    """Exactly 3 paragraphs: (1) gradient, (2) field of study, (3) key +
    scale + source/update/units. No volatile fetch timestamps."""
    return (
        f"<h2>{title}</h2>"
        f"<p>{para_gradient}</p>"
        f"<p>{para_field}</p>"
        f"<p><img src=\"{legend_src}\" width=\"600\" alt=\"key\"><br>"
        f"{scale_html}<br>"
        f"<b>Units:</b> {units_line}<br>"
        f"<b>Source:</b> {source_name}<br>"
        f"<b>Update:</b> {update_line}<br>"
        f"<b>Provenance:</b> <a href=\"{source_url}\">{source_url}</a></p>"
    )


def legend_src(product, token=None):
    from build_kml import pages_base
    base = pages_base()
    v = f"?v={token}" if token else ""
    return f"{base}/{product}/legend.png{v}"


def finish(product, config, kml_file, overlay_name, field, stops, labels,
           unit_label, subtitle, source_line, scale_html, folder_paras,
           meta_extra, source_id, data_time_utc, source_last_modified,
           units_desc, source_resolution, missing_treatment,
           min_opaque=50_000, extra_meta=None):
    """Render field -> validate -> legend/metadata/KML -> promote -> state."""
    bounds = load_bounds()
    stage = stage_dir(product)
    stage_prod = os.path.join(stage, "site", product)
    rgba = render_rgba(field, stops, bounds["overlay_alpha"])
    # Full basin rectangle (atmospheric layers are valid over land and
    # water alike, like air temperature / pressure / solar / aurora):
    # only missing source data is transparent. No shoreline cut, so no
    # land masking hides valid atmospheric data.
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    ok = np.isfinite(field)
    n_valid = int(ok.sum())
    if n_opaque < min_opaque or n_valid < min_opaque:
        raise ValueError(f"empty raster (opaque={n_opaque} valid={n_valid})")
    cur_min = float(field[ok].min()) if n_valid else float("nan")
    cur_max = float(field[ok].max()) if n_valid else float("nan")
    print(f"[{product}] valid={n_valid} "
          f"range=[{cur_min:.3g},{cur_max:.3g}] {unit_label}")

    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), config["title"], subtitle,
        unit_label, stops, labels, source_line,
        note="Missing source data transparent; never zero-filled.")
    meta = base_metadata(
        product, config["title"], config["freshness_label"],
        config["source_name"], config["source_url"],
        config["variable"], data_time_utc=data_time_utc,
        source_last_modified_utc=source_last_modified,
        units=units_desc, source_resolution=source_resolution,
        color_min=stops[0][0], color_max=stops[-1][0],
        color_units=unit_label,
        missing_data_treatment=missing_treatment)
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = meta_extra.get("model_cycle", source_id)
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["stats"] = {"valid_cells": n_valid, "current_min": cur_min,
                     "current_max": cur_max,
                     **meta_extra.get("stats", {})}
    for k, v in meta_extra.get("fields", {}).items():
        meta[k] = v
    if extra_meta:
        meta.update(extra_meta)
    token = meta["source_version"]
    folder = folder_html_3para(
        config["title"], folder_paras[0], folder_paras[1], scale_html,
        legend_src(product, token), config["source_name"],
        config["source_url"], folder_paras[2], units_desc)
    meta["folder_html"] = folder
    write_metadata(stage_prod, meta)
    block = legend_block(f"{product}/legend.png", token, scale_html)
    outs = live_out_dirs(stage, kml_file)
    kml_text = build_kml(
        product, kml_file, overlay_name,
        f"{product}/current.png", f"{product}/legend.png",
        description_html(config["title"], meta, SKIP_NOTE, block),
        config["refresh_interval_seconds"], token,
        folder=(config["title"], folder),
        out_dirs=outs["live"], meta=meta)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        product, kml_file, overlay_name,
        entry_description_html(config["title"], meta, SKIP_NOTE),
        config["refresh_interval_seconds"], out_dirs=outs["entry"])
    promoted = promote_stage(product)
    write_state(product, {"model_cycle": meta["model_cycle"],
                          "source_id": source_id,
                          "render_version": RENDER_VERSION,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{product}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def hrrr_field(canvas_values_fn, messages, raw_name, bounds, raw_dir):
    """Fetch HRRR messages then bin source points via canvas_values_fn.

    canvas_values_fn(vals_by_name) -> (values_1d, lats_1d, lons_1d,
    data_date, data_time). Returns (field, data_time_utc, n_src_valid).
    """
    import hrrr
    from geospatial_utils import grib_stamp_to_det
    base, dd, cc = hrrr.latest_cycle()
    raw_path = os.path.join(raw_dir, raw_name)
    hrrr.fetch_messages(base, raw_path, messages)
    wanted = {name: 1 for name, _lvl in messages}
    got = hrrr.read_messages(raw_path, wanted)
    vals, lats, lons, data_date, data_time = canvas_values_fn(got)
    data_time_utc = grib_stamp_to_det(data_date, data_time)
    H, W = bounds["canvas_height"], bounds["canvas_width"]
    lats = np.asarray(lats, dtype=float).ravel()
    lons = (((np.asarray(lons, dtype=float).ravel() + 180) % 360) - 180)
    v = np.asarray(vals, dtype=float).ravel()
    rows, cols, _v = canvas_indices(lats, lons, bounds)
    inside = (np.isfinite(lats) & np.isfinite(lons)
              & (lats >= bounds["lat_min"]) & (lats <= bounds["lat_max"])
              & (lons >= bounds["lon_min"]) & (lons <= bounds["lon_max"]))
    field, _c = bin_to_canvas(rows, cols, v, inside & np.isfinite(v),
                              (H, W), splat_radius=2)
    return field, data_time_utc, dd, cc, base
