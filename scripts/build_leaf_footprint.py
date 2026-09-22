"""Build the committed LEAF-COLOR footprint mask (one shared asset).

Footprint = Michigan (both peninsulas, whole state) UNION a geodesic
50-mile buffer of the NOAA medium-resolution Great Lakes water polygons,
dissolved and smoothed. Consequences, all verified below, not hand-drawn:
  - Michigan entirely inside (incl. entire UP: no land cutoff anywhere,
    so the "50 mi north of the UP line" extension is automatic — the line
    itself lies over water);
  - Bruce Peninsula fully inside (surrounded by buffered water);
  - Georgian Bay / Lake Ontario shores get 50-mi zones that merge with the
    Huron/Ontario zones into one natural footprint;
  - Wisconsin/Ohio/etc. only inside the 50-mi shoreline band.

Sources (committed output only; builders never fetch these):
  - Michigan polygon: Natural Earth 50m admin-1 (public domain).
  - Water polygons: same NOAA medium-res pipeline as the shoreline mask
    (scripts/build_watermask.py::build_water_geometry).
Metric work in EPSG:3175 (NAD83 / Great Lakes and St Lawrence Albers);
a 50-mile = 80467.2 m buffer; smoothing = simplify(1000 m) + buffer(0).

Regeneration (one-time, local):
  python scripts/build_leaf_footprint.py --states <ne_50m... base> \\
      --shoreline <us_medium_shoreline base>
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geospatial_utils import REPO_ROOT, load_bounds  # noqa: E402

M_TO_MI_BUFFER = 80467.2


def read_michigan(states_base):
    import shapefile
    from shapely.geometry import MultiPolygon, Polygon
    r = shapefile.Reader(states_base)
    flds = [f[0] for f in r.fields[1:]]
    geoms = []
    for shape, rec in zip(r.iterShapes(), r.iterRecords()):
        d = dict(zip(flds, rec))
        if d.get("name") == "Michigan" and d.get("iso_a2") == "US":
            pts = shape.points
            parts = list(shape.parts) + [len(pts)]
            for a, b in zip(parts[:-1], parts[1:]):
                ring = pts[a:b]
                if len(ring) >= 4:
                    try:
                        g = Polygon(ring)
                        if not g.is_valid:
                            g = g.buffer(0)
                        if not g.is_empty:
                            geoms.append(g)
                    except Exception:
                        continue
    if not geoms:
        raise ValueError("Michigan polygon not found")
    u = geoms[0] if len(geoms) == 1 else MultiPolygon(geoms)
    from shapely.ops import unary_union
    return unary_union(u)


def build_footprint(states_base, shoreline_base):
    import pyproj
    from shapely.ops import transform as shp_transform
    from shapely.ops import unary_union
    from build_watermask import build_water_geometry
    bounds = load_bounds()
    mich = read_michigan(states_base)
    water, _stats = build_water_geometry(shoreline_base, bounds)
    to_m = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3175", always_xy=True).transform
    to_deg = pyproj.Transformer.from_crs("EPSG:3175", "EPSG:4326", always_xy=True).transform
    mich_m = shp_transform(to_m, mich)
    water_m = shp_transform(to_m, water)
    env = unary_union([mich_m, water_m.buffer(M_TO_MI_BUFFER)])
    env = env.simplify(1000).buffer(0)
    return shp_transform(to_deg, env)


CHECKS_INSIDE = {  # must be inside footprint
    "Detroit": (-83.05, 42.33), "Ironwood_UP_west": (-90.17, 46.45),
    "CopperHarbor_UP_tip": (-87.89, 47.47), "Mackinaw": (-84.73, 45.78),
    "Bruce_tip": (-81.35, 45.25), "Toronto": (-79.38, 43.65),
    "GeorgianBay_east": (-79.8, 44.9), "Kingston": (-76.48, 44.23),
    "GreenBay_WI": (-88.0, 44.5), "Erie_PA": (-80.1, 42.15),
}
CHECKS_OUTSIDE = {  # must be outside footprint (>50 mi from water/Michigan)
    # NOTE: Ottawa falls inside via the genuine St. Lawrence connected
    # waterway + 50-mi rule, so it is not used as an exclusion probe.
    "Illinois_far": (-88.0, 40.5), "Columbus_OH": (-83.0, 39.96),
    "Ontario_north": (-80.0, 47.5), "Minnesota_far": (-92.9, 45.5),
}


def rasterize(geom, bounds, W, H):
    from shapely import contains_xy
    grid = np.zeros((H, W), dtype=bool)
    geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    for g in geoms:
        try:
            minx, miny, maxx, maxy = g.bounds
        except Exception:
            continue
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
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--shoreline", required=True)
    args = ap.parse_args()
    bounds = load_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    env = build_footprint(args.states, args.shoreline)
    from shapely.geometry import Point
    nparts = len(env.geoms) if env.geom_type == "MultiPolygon" else 1
    print(f"footprint parts: {nparts}, area deg2: {env.area:.3f}")
    for name, (x, y) in CHECKS_INSIDE.items():
        ok = env.contains(Point(x, y))
        print(f"  inside {name}: {ok}")
        if not ok:
            raise ValueError(f"footprint missing {name}")
    for name, (x, y) in CHECKS_OUTSIDE.items():
        ok = not env.contains(Point(x, y))
        print(f"  outside {name}: {ok}")
        if not ok:
            raise ValueError(f"footprint leaks to {name}")
    grid = rasterize(env, bounds, W, H)
    print(f"footprint frac={grid.mean():.4f}")
    out = os.path.join(REPO_ROOT, "assets", "leaf_footprint.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray((grid * 255).astype(np.uint8), mode="L").save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
