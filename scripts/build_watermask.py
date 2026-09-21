"""Build the ONE shared Great Lakes water mask (committed asset).

Source: GSHHG v2.3.7 (Global Self-consistent Hierarchical High-resolution
Geography; Wessel & Smith 1996; WVS + WDBII, public domain amalgamation,
LGPL-licensed database; maintained by P. Wessel / W.H.F. Smith, NOAA).
  L1 (high res): land/ocean boundary      -> land, unless overruled below
  L2 (full res): lake/land boundary       -> water (the Great Lakes + inland lakes)
  L3 (full res): island-in-lake boundary  -> land  (Isle Royale, etc.)
  L4 (full res): pond-in-island boundary  -> water (negligible, kept for correctness)

Rule per pixel (deepest level wins):
    water = L4 | (~L3 & (L2 | ~L1))

The mask is rasterized at 3x canvas resolution, then downsampled with
LANCZOS for antialiased shoreline edges, and committed as
assets/great_lakes_watermask.png (8-bit, 255 = water). CI never fetches
GSHHG; every product loads this identical file, so all layers align.

Regeneration (one-time, local):
  1. download https://github.com/GenericMappingTools/gshhg-gmt/releases/download/2.3.7/gshhg-shp-2.3.7.zip
  2. extract GSHHS_shp/{f/{L2,L3,L4},h/{L1}} shapefiles
  3. python scripts/build_watermask.py --l1 <GSHHS_h_L1 base> --l2 <GSHHS_f_L2 base> \\
       --l3 <GSHHS_f_L3 base> --l4 <GSHHS_f_L4 base>
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geospatial_utils import REPO_ROOT, load_bounds  # noqa: E402

SUPER = 3  # supersample factor for antialiased edges


def read_polys(shp_base):
    import shapefile
    from shapely.geometry import MultiPolygon, Polygon
    r = shapefile.Reader(shp_base)
    out = []
    for shape in r.iterShapes():
        pts = shape.points
        parts = list(shape.parts) + [len(pts)]
        geoms = []
        for i in range(len(parts) - 1):
            ring = pts[parts[i]:parts[i + 1]]
            if len(ring) < 4:
                continue
            try:
                g = Polygon(ring)
                if not g.is_valid:
                    g = g.buffer(0)
                if not g.is_empty:
                    geoms.append(g)
            except Exception:
                continue
        if geoms:
            out.append(geoms[0] if len(geoms) == 1 else MultiPolygon(geoms))
    return out


def inside_any(geoms, bounds, W, H):
    """Boolean grid: True where inside any geometry (bbox-cropped)."""
    from shapely import contains_xy
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    grid = np.zeros((H, W), dtype=bool)
    kept = 0
    for g in geoms:
        try:
            minx, miny, maxx, maxy = g.bounds
        except Exception:
            continue
        if maxx < lon_min or minx > lon_max or maxy < lat_min or miny > lat_max:
            continue
        kept += 1
        c0 = max(int((minx - lon_min) / (lon_max - lon_min) * W), 0)
        c1 = min(int((maxx - lon_min) / (lon_max - lon_min) * W) + 1, W)
        r0 = max(int((lat_max - maxy) / (lat_max - lat_min) * H), 0)
        r1 = min(int((lat_max - miny) / (lat_max - lat_min) * H) + 1, H)
        if r0 >= r1 or c0 >= c1:
            continue
        xs = lon_min + (np.arange(c0, c1) + 0.5) / W * (lon_max - lon_min)
        ys = lat_max - (np.arange(r0, r1) + 0.5) / H * (lat_max - lat_min)
        xx, yy = np.meshgrid(xs, ys)
        try:
            grid[r0:r1, c0:c1] |= contains_xy(g, xx, yy)
        except Exception:
            continue
    return grid, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", required=True)
    ap.add_argument("--l2", required=True)
    ap.add_argument("--l3", required=True)
    ap.add_argument("--l4", required=True)
    args = ap.parse_args()
    bounds = load_bounds()
    W, H = bounds["canvas_width"] * SUPER, bounds["canvas_height"] * SUPER

    levels = {}
    for name, base in (("L1", args.l1), ("L2", args.l2),
                       ("L3", args.l3), ("L4", args.l4)):
        geoms = read_polys(base)
        grid, kept = inside_any(geoms, bounds, W, H)
        levels[name] = grid
        print(f"{name}: {len(geoms)} polygons, {kept} intersect domain, "
              f"inside frac={grid.mean():.4f}")

    water = levels["L4"] | (~levels["L3"] & (levels["L2"] | ~levels["L1"]))
    print(f"water frac={water.mean():.4f}")
    img = Image.fromarray((water * 255).astype(np.uint8), mode="L")
    img = img.resize((bounds["canvas_width"], bounds["canvas_height"]),
                     Image.LANCZOS)
    out = os.path.join(REPO_ROOT, "assets", "great_lakes_watermask.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    arr = np.array(img)
    edge = ((arr > 0) & (arr < 255)).mean()
    print(f"wrote {out} water_frac={ (arr > 127).mean():.4f} "
          f"antialiased_edge_frac={edge:.4f}")


if __name__ == "__main__":
    main()
