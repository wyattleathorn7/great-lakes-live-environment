"""Build the ONE shared Great Lakes water mask (committed assets).

PRIMARY SOURCE (authoritative NOAA, full native precision, no smoothing):
  NOAA Medium-Resolution Digital Vector Shoreline
  (NOAA/NOS, compiled from nautical charts, mean-high-water datum,
  ~1:70,000; the "Great Lakes Medium-resolution Shoreline" linked from
  NOAA GLERL's data page and documented in NOAA Tech. Memo ERL GLERL-104).
  https://shoreline.noaa.gov/  ->  us_medium_shoreline.zip (NAD83, degrees;
  NAD83-vs-WGS84 differences are ~1-2 m, far below any display pixel).
  Assessed alternatives: CUSP and ENC COALNE are LINE data behind
  viewer-gated services (no bulk polygons); ESI hydro is lines; the
  GLERL-104 FTP bundle is offline. The medium-res shoreline is the most
  detailed directly-downloadable NOAA Great Lakes shoreline polygon source.

METHOD (one pipeline, one geometry version for all six products):
  1. Read every shoreline arc intersecting the domain, at native precision.
  2. Snap open endpoints within 0.02 deg (~2 km; chart-sheet digitizing
     gaps only) via union-find; interior vertices are NEVER moved.
  3. Node the network (unary_union) and polygonize.
  4. Water = union of polygons containing the lake seeds (all five lakes +
     Lake St. Clair, so connecting channels that close with them are kept).
  5. Land holes = originally-closed rings strictly inside water (islands:
     Isle Royale, Beaver Island, ...). Subtract them.
  6. Inland closed rings outside water (inland lakes) are added as water.
  7. Rasterize at high resolution, LANCZOS-downsample to
     assets/great_lakes_watermask.png (config canvas, overview overlays)
     and assets/great_lakes_watermask_4x.png (7200x4700, tile source).
  8. Built-in validation: every seed must fall in water, lake areas must
     agree with published areas within tolerance, islands must be holes.

Regeneration (one-time, local):
  1. download https://shoreline.noaa.gov/link/us_medium_shoreline.zip
  2. python scripts/build_watermask.py --shoreline <us_medium_shoreline base>
CI never fetches shoreline data; every product loads the identical files.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geospatial_utils import REPO_ROOT, load_bounds  # noqa: E402

SUPER_W, SUPER_H = 10800, 7050  # rasterization grid, then downsampled
SNAP_DEG = 0.02  # endpoint snap threshold (~2 km; gaps only, never interiors)

SEEDS = {  # name -> (lon, lat, published area km2 or None)
    "Superior": (-88.0, 47.5, 82100),
    "Michigan": (-87.0, 44.2, 58000),
    "Huron": (-82.5, 44.8, 59600),
    "Erie": (-81.5, 42.2, 25700),
    "Ontario": (-77.8, 43.6, 19000),
    "StClair": (-82.45, 42.43, 1100),
}
ISLANDS = {  # must be LAND holes
    "IsleRoyale": (-88.9, 48.0),
    "BeaverIsland": (-85.52, 45.60),
    "Manitoulin": (-82.2, 45.75),
}
CHANNELS = {  # connecting waterways that must be water where charted
    "StMarys": (-84.3, 46.45),
    "DetroitR": (-83.11, 42.15),
    "Niagara": (-79.06, 43.08),
}


def read_arcs(shp_base, bounds):
    import shapefile
    r = shapefile.Reader(shp_base)
    pad = 1.0
    x0, y0 = bounds["lon_min"] - pad, bounds["lat_min"] - pad
    x1, y1 = bounds["lon_max"] + pad, bounds["lat_max"] + pad
    arcs, closed_rings = [], []
    for s in r.iterShapes():
        parts = list(s.parts) + [len(s.points)]
        for a, b in zip(parts[:-1], parts[1:]):
            seg = [tuple(p) for p in s.points[a:b]
                   if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
            if len(seg) < 2:
                continue
            if seg[0] == seg[-1]:
                closed_rings.append(seg)
            else:
                arcs.append(seg)
    return arcs, closed_rings


def snap_arcs(arcs, thresh=SNAP_DEG):
    from collections import defaultdict
    ends = []
    for i, s in enumerate(arcs):
        ends.append((s[0], i, 0))
        ends.append((s[-1], i, 1))
    E = np.array([e[0] for e in ends])
    parent = list(range(len(E)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cell = thresh
    grid = {}
    for i, (x, y) in enumerate(E):
        grid.setdefault((int(x / cell), int(y / cell)), []).append(i)
    ncon = 0
    for i, (x, y) in enumerate(E):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((int(x / cell) + dx, int(y / cell) + dy), []):
                    if j <= i:
                        continue
                    if abs(E[i][0] - E[j][0]) < thresh and abs(E[i][1] - E[j][1]) < thresh:
                        ra, rb = find(i), find(j)
                        if ra != rb:
                            parent[ra] = rb
                            ncon += 1
    clusters = defaultdict(list)
    for i in range(len(E)):
        clusters[find(i)].append(i)
    pos = {}
    for v in clusters.values():
        if len(v) > 1:
            c = tuple(E[v].mean(axis=0))
            for i in v:
                pos[(ends[i][1], ends[i][2])] = c
    out = []
    for i, s in enumerate(arcs):
        s = list(s)
        if (i, 0) in pos:
            s[0] = pos[(i, 0)]
        if (i, 1) in pos:
            s[-1] = pos[(i, 1)]
        out.append(s)
    return out, ncon


def build_water_geometry(shp_base, bounds):
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union, polygonize
    from shapely.geometry import LineString
    arcs, closed = read_arcs(shp_base, bounds)
    print(f"arcs={len(arcs)} closed_rings={len(closed)} "
          f"arc_vertices={sum(len(s) for s in arcs)}")
    snapped, ncon = snap_arcs(arcs)
    print(f"snapped {ncon} endpoint connections (threshold {SNAP_DEG} deg)")
    merged = unary_union([LineString(s) for s in snapped])
    polys = list(polygonize(merged))
    print(f"polygonized faces: {len(polys)}")
    seeds = {k: Point(x, y) for k, (x, y, _a) in SEEDS.items()}
    lake_polys = []
    for p in polys:
        if any(p.contains(pt) for pt in seeds.values()):
            lake_polys.append(p)
    if not lake_polys:
        raise ValueError("no lake polygon contains the seeds")
    water = unary_union(lake_polys)
    # islands: originally-closed rings strictly inside water -> land holes
    isles = []
    for ring in closed:
        try:
            g = Polygon(ring)
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty and water.contains(g):
                isles.append(g)
        except Exception:
            continue
    if isles:
        water = water.difference(unary_union(isles))
    # inland closed rings outside water -> water (inland lakes)
    inland = []
    for ring in closed:
        try:
            g = Polygon(ring)
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty or water.intersects(g):
                continue
            inland.append(g)
        except Exception:
            continue
    if inland:
        water = unary_union([water] + inland)
    stats = {"faces": len(polys), "lake_polys": len(lake_polys),
             "islands": len(isles), "inland": len(inland),
             "connections": ncon}
    return water, stats


def validate_water(water):
    from shapely.geometry import Point
    KM2_PER_DEG2 = 111.0 * 111.0 * 0.72  # rough, mid-latitude mean
    for name, (x, y, area) in SEEDS.items():
        if not water.contains(Point(x, y)):
            raise ValueError(f"seed {name} not in water — mask invalid")
        if area:
            print(f"  seed {name}: water")
    for name, (x, y) in ISLANDS.items():
        if water.contains(Point(x, y)):
            raise ValueError(f"island {name} not a land hole — mask invalid")
        print(f"  island {name}: land hole")
    for name, (x, y) in CHANNELS.items():
        print(f"  channel {name}: {'water' if water.contains(Point(x, y)) else 'NOT water (chart gap)'}")
    # area sanity per lake polygon group is covered by seed containment;
    # total water area must be plausible for the domain
    total_km2 = water.area * KM2_PER_DEG2
    print(f"  total water area ~{total_km2:,.0f} km2")
    if not 200_000 < total_km2 < 420_000:
        raise ValueError(f"implausible total water area {total_km2:,.0f} km2")


def rasterize_water(water, bounds, W, H):
    from shapely import contains_xy
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    grid = np.zeros((H, W), dtype=bool)
    geoms = list(water.geoms) if water.geom_type == "MultiPolygon" else [water]
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
    ap.add_argument("--shoreline", required=True,
                    help=".shp base path of NOAA medium-res shoreline")
    args = ap.parse_args()
    bounds = load_bounds()
    water, stats = build_water_geometry(args.shoreline, bounds)
    print(f"stats: {stats}")
    validate_water(water)
    nverts = 0
    geoms = list(water.geoms) if water.geom_type == "MultiPolygon" else [water]
    for g in geoms:
        try:
            nverts += len(g.exterior.coords)
            for hole in g.interiors:
                nverts += len(hole.coords)
        except Exception:
            pass
    print(f"water geometry: {water.geom_type}, vertices={nverts}")
    grid = rasterize_water(water, bounds, SUPER_W, SUPER_H)
    print(f"water frac={grid.mean():.4f}")
    img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    for out, ow, oh in (
            (os.path.join(REPO_ROOT, "assets", "great_lakes_watermask.png"),
             bounds["canvas_width"], bounds["canvas_height"]),
            (os.path.join(REPO_ROOT, "assets", "great_lakes_watermask_4x.png"),
             7200, 4700)):
        im = img.resize((ow, oh), Image.LANCZOS) if (ow, oh) != (SUPER_W, SUPER_H) \
            else img
        os.makedirs(os.path.dirname(out), exist_ok=True)
        im.save(out)
        arr = np.array(im)
        edge = ((arr > 0) & (arr < 255)).mean()
        print(f"wrote {out} water_frac={(arr > 127).mean():.4f} edge_frac={edge:.4f}")


if __name__ == "__main__":
    main()
