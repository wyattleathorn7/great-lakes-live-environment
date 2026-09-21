"""Shared gradient renderer + CLI.

Library: render_field() bins a WGS84-mapped source field onto the common
canvas and writes current.png / legend.png / metadata.json for one product.
CLI: re-render a cached intermediate field (*.npz with lats/lons/values)
without re-downloading, e.g.:
  python render_gradient.py --product wave_height --from-cache output/raw/wave_field.npz
"""

import argparse
import json
import math
import os

import numpy as np

from geospatial_utils import (ICE_STOPS, LEGEND_H, LEGEND_W, SITE_DIR,
                              TEMP_STOPS, WAVE_STOPS, apply_colormap,
                              base_metadata, bin_to_canvas, canvas_indices,
                              draw_legend, fmt_ticks, load_bounds, save_png,
                              utcnow_iso, write_metadata)

RENDER_SPECS = {
    "wave_height": {"stops": WAVE_STOPS, "unit": "ft"},
    "water_temperature": {"stops": TEMP_STOPS, "unit": "\u00b0F"},
    "ice_coverage": {"stops": ICE_STOPS, "unit": "%"},
}


def render_field(product, lats, lons, values_display, vmin, vmax, meta_extra,
                 title, subtitle, source_line, unit_label, transparent_value,
                 fmt, splat_radius=1, product_dir=None):
    bounds = load_bounds()
    rows, cols, valid = canvas_indices(lats, lons, bounds)
    field, counts = bin_to_canvas(
        rows, cols, np.asarray(values_display, dtype=float), valid,
        (bounds["canvas_height"], bounds["canvas_width"]),
        splat_radius=splat_radius)
    spec = RENDER_SPECS[product]
    rgba = apply_colormap(field, vmin, vmax, spec["stops"],
                          bounds["overlay_alpha"], transparent_value)
    if product_dir is None:
        product_dir = os.path.join(SITE_DIR, product)
    save_png(rgba, os.path.join(product_dir, "current.png"))
    draw_legend(os.path.join(product_dir, "legend.png"), title, subtitle,
                unit_label, vmin, vmax, spec["stops"], source_line,
                fmt=fmt,
                transparent_note="Transparent outside valid water data.")
    meta = dict(meta_extra)
    meta["color_scale_min"] = vmin
    meta["color_scale_max"] = vmax
    meta["color_scale_units"] = spec["unit"]
    meta["legend_size"] = [LEGEND_W, LEGEND_H]
    meta["rendered_nontransparent_pixels"] = int((rgba[:, :, 3] > 0).sum())
    meta["rendered_canvas_pixels"] = int(rgba.shape[0] * rgba.shape[1])
    write_metadata(product_dir, meta)
    return field, rgba, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True, choices=list(RENDER_SPECS))
    ap.add_argument("--from-cache", required=True)
    ap.add_argument("--meta", required=True,
                    help="JSON file with title/subtitle/source_line/vmin/vmax/meta_extra")
    ap.add_argument("--out-dir", default=None,
                    help="product output dir (default: site/<product>)")
    args = ap.parse_args()
    z = np.load(args.from_cache)
    with open(args.meta) as f:
        m = json.load(f)
    render_field(args.product, z["lats"], z["lons"], z["values"],
                 m["vmin"], m["vmax"], m["meta_extra"], m["title"],
                 m["subtitle"], m["source_line"], m["unit_label"],
                 m.get("transparent_value"), m.get("fmt", "{:.0f}"),
                 m.get("splat_radius", 1), product_dir=args.out_dir)
    print(f"re-rendered {args.product} from {args.from_cache}")


if __name__ == "__main__":
    main()
