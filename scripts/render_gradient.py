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
                              apply_shoreline_mask, base_metadata,
                              bin_to_canvas, canvas_indices, draw_legend,
                              fmt_ticks, load_bounds, save_png, utcnow_iso,
                              write_metadata)

RENDER_SPECS = {
    "wave_height": {"stops": WAVE_STOPS, "unit": "ft"},
    "water_temperature": {"stops": TEMP_STOPS, "unit": "\u00b0F"},
    "ice_coverage": {"stops": ICE_STOPS, "unit": "%"},
}


def render_field(product, lats, lons, values_display, vmin, vmax, meta_extra,
                 title, subtitle, source_line, unit_label, transparent_value,
                 fmt, splat_radius=1, product_dir=None, tick_labels=None):
    bounds = load_bounds()
    rows, cols, valid = canvas_indices(lats, lons, bounds)
    field, counts = bin_to_canvas(
        rows, cols, np.asarray(values_display, dtype=float), valid,
        (bounds["canvas_height"], bounds["canvas_width"]),
        splat_radius=splat_radius)
    spec = RENDER_SPECS[product]
    rgba = apply_colormap(field, vmin, vmax, spec["stops"],
                          bounds["overlay_alpha"], transparent_value)
    rgba = apply_shoreline_mask(rgba)  # one shared GSHHG shoreline for all
    if product_dir is None:
        product_dir = os.path.join(SITE_DIR, product)
    save_png(rgba, os.path.join(product_dir, "current.png"))
    draw_legend(os.path.join(product_dir, "legend.png"), title, subtitle,
                unit_label, vmin, vmax, spec["stops"], source_line,
                fmt=fmt, tick_labels=tick_labels,
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


TILE_HALO_DEG = 0.06  # source halo around tiles so splat windows match
# across seams (edge pixels average the same source points as neighbors)


def render_tile(product, lats, lons, values_display, vmin, vmax,
                transparent_value, splat_radius, tile_bounds, mask_crop):
    """Render one LOD tile (1800x1175) with the shared color table, masked by
    the tile's crop of the 4x shoreline mask. Returns RGBA uint8."""
    import math
    import numpy as np
    from geospatial_utils import (apply_colormap, bin_to_canvas,
                                  canvas_indices)
    spec = RENDER_SPECS[product]
    W, H = tile_bounds["canvas_width"], tile_bounds["canvas_height"]
    px = (tile_bounds["lon_max"] - tile_bounds["lon_min"]) / W
    pad = max(2, int(math.ceil(TILE_HALO_DEG / px)))
    eb = dict(tile_bounds,
              lon_min=tile_bounds["lon_min"] - pad * px,
              lon_max=tile_bounds["lon_max"] + pad * px,
              lat_min=tile_bounds["lat_min"] - pad * px,
              lat_max=tile_bounds["lat_max"] + pad * px,
              canvas_width=W + 2 * pad, canvas_height=H + 2 * pad)
    rows, cols, valid = canvas_indices(lats, lons, eb)
    field, _ = bin_to_canvas(rows, cols, np.asarray(values_display, dtype=float),
                             valid, (H + 2 * pad, W + 2 * pad),
                             splat_radius=splat_radius)
    field = field[pad:pad + H, pad:pad + W]
    rgba = apply_colormap(field, vmin, vmax, spec["stops"],
                          tile_bounds["overlay_alpha"], transparent_value)
    rgba[:, :, 3] = np.round(
        rgba[:, :, 3].astype(np.float32) * mask_crop).astype(np.uint8)
    return rgba, field


def build_tiles(product, stage_prod, render_one):
    """Render the LOD pyramid for one product.

    render_one(tile_bounds_dict, level) -> RGBA uint8 array, or None to skip
    the tile (e.g. no ice in an off-season polygon product). Tile PNGs are
    written under <stage_prod>/tiles/ (Pages-deployed, never committed).
    Returns [(rel_path, tile_bounds, min_lod)] for the KML, or [] when no
    tile was rendered (KML stays overview-only).
    """
    import os
    from geospatial_utils import (save_png, tile_bounds, tile_layout,
                                  tile_rel_path)
    tiles = []
    for level, n, min_lod in tile_layout():
        for iy in range(n):
            for ix in range(n):
                tb = tile_bounds(level, ix, iy)
                rgba = render_one(tb, level)
                if rgba is None:
                    continue
                rel = tile_rel_path(product, level, ix, iy)
                save_png(rgba, os.path.join(stage_prod, "tiles",
                                            os.path.basename(rel)))
                tiles.append((rel, tb, min_lod))
    return tiles


def build_grid_tiles(product, stage_prod, lats, lons, values, vmin, vmax,
                     transparent_value, splat_overview, has_data):
    """Tile pyramid for gridded (binned) products. Tile splat scales with
    level so coarse source grids stay hole-free at 2x/4x density."""
    from geospatial_utils import mask_crop_for_tile
    if not has_data:
        return []

    def one(tb, level):
        rgba, _ = render_tile(product, lats, lons, values, vmin, vmax,
                              transparent_value,
                              splat_overview * (2 ** level), tb,
                              mask_crop_for_tile(tb))
        return rgba

    return build_tiles(product, stage_prod, one)


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
                 m.get("splat_radius", 1), product_dir=args.out_dir,
                 tick_labels=m.get("tick_labels"))
    print(f"re-rendered {args.product} from {args.from_cache}")


if __name__ == "__main__":
    main()
