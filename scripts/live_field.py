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

# Per-product render generation: bump a product's tag (and only that tag)
# to force Google Earth clients to refetch an otherwise identical source
# cycle after a rendering change (new ?v= without touching data or URLs).
RENDER_TAGS = {
    "visibility": "g3",
    "humidity": "g2",
    "light_pollution": "g4",
    "air_quality": "g2",
    "condensation": "g2",
    "fog": "g3",
    "dew_point": "g2",
}


def versioned_source_id(product, source_id):
    tag = RENDER_TAGS.get(product)
    return f"{source_id}-{tag}" if tag else source_id


def smooth_nan(field, radius=2, passes=2):
    """NaN-aware box-blur smoothing for TV-style display gradients.

    Softens razor-sharp model-grid edges and dissolves single-cell
    speckles into their surroundings while coherent features persist.
    Missing data stays missing (never zero-filled, never grown). Same
    display-smoothing contract as the solar product; statistics stay on
    raw values (callers override stats explicitly).
    """
    from gradient_scale import _box_sum
    cur = np.asarray(field, dtype=float)
    for _ in range(passes):
        have = np.isfinite(cur)
        if not have.any():
            return cur
        sw = _box_sum(have.astype(float), radius)
        sv = _box_sum(np.where(have, cur, 0.0), radius)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sv / sw
        cur = np.where(have & (sw > 0), mean, cur)
    return cur


def draw_two_row_legend(path, title, subtitle, unit_label, stops, ticks,
                        source_line, note=None):
    """Two-row evenly-spaced key: values on row 1, severity words on row 2.

    ticks = [(value, value_text, severity_text)] at TRUE linear scale
    positions — callers pass evenly-stepped values so labels can never
    pile onto each other (the failure mode of de-collided single-row
    keys). The bar itself is painted with the SAME stops as the raster.
    Returns (W, H).
    """
    from PIL import Image, ImageDraw
    from geospatial_utils import _legend_font
    W, H = 640, 252
    img = Image.new("RGBA", (W, H), (255, 255, 255, 235))
    d = ImageDraw.Draw(img)
    f_title, f_body, f_small = _legend_font(22), _legend_font(15), _legend_font(13)
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=2)
    d.text((14, 8), title, font=f_title, fill=(10, 10, 10))
    d.text((14, 36), subtitle, font=f_body, fill=(40, 40, 40))
    bx, by, bw, bh = 14, 66, W - 28, 32
    from gradient_scale import lut_from_stops
    lut = lut_from_stops(stops, bw)
    for i, c in enumerate(lut):
        d.line([(bx + i, by), (bx + i, by + bh)], fill=tuple(c) + (255,))
    d.rectangle([bx, by, bx + bw - 1, by + bh], outline=(40, 40, 40))
    vmin, vmax = stops[0][0], stops[-1][0]
    span = vmax - vmin if vmax > vmin else 1.0
    for val, vtext, stext in ticks:
        frac = min(max((val - vmin) / span, 0.0), 1.0)
        x = bx + int(frac * (bw - 1))
        tw = d.textlength(vtext, font=f_small)
        d.text((min(max(x - tw / 2, 2), W - tw - 2), by + bh + 4),
               vtext, font=f_small, fill=(10, 10, 10))
        sw = d.textlength(stext, font=f_small)
        d.text((min(max(x - sw / 2, 2), W - sw - 2), by + bh + 22),
               stext, font=f_small, fill=(60, 60, 60))
    d.text((bx + bw - 70, by + bh + 42), unit_label, font=f_body,
           fill=(10, 10, 10))
    d.text((14, H - 44), source_line, font=f_small, fill=(60, 60, 60))
    if note:
        d.text((14, H - 26), note, font=f_small, fill=(60, 60, 60))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return W, H


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
           min_opaque=50_000, extra_meta=None, alpha=None,
           key_ticks=None, custom_legend=None):
    """Render field -> validate -> legend/metadata/KML -> promote -> state."""
    bounds = load_bounds()
    if alpha is None:
        alpha = bounds["overlay_alpha"]
    source_id = versioned_source_id(product, source_id)
    stage = stage_dir(product)
    stage_prod = os.path.join(stage, "site", product)
    rgba = render_rgba(field, stops, alpha)
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

    if custom_legend is not None:
        lw, lh = custom_legend(os.path.join(stage_prod, "legend.png"))
    elif key_ticks is not None:
        lw, lh = draw_two_row_legend(
            os.path.join(stage_prod, "legend.png"), config["title"],
            subtitle, unit_label, stops, key_ticks, source_line,
            note="Missing source data transparent; never zero-filled.")
    else:
        lw, lh = draw_scale_legend(
            os.path.join(stage_prod, "legend.png"), config["title"],
            subtitle, unit_label, stops, labels, source_line,
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
