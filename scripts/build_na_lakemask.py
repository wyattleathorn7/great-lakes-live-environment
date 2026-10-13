"""Build the North American lake mask (wind layer only).

Water set = the five Great Lakes + Lake Nipigon (original inland-lake list)
+ the very large Canadian lakes running north-west of the Great Lakes and
other major Canadian lakes (Bear, Slave, Winnipeg, Athabasca, Manitoba,
Winnipegosis, Reindeer, Wollaston, Mistassini, Nettilling, Amadjuak,
Dubawnt, Cedar, Southern Indian, Woods). Natural lakes only — reservoirs
(Smallwood, Williston, Manicouagan) and US impoundments stay land-style,
exactly like the original INLAND_LAKES convention (natural lakes only).

Same geometry engine as scripts/build_na_coastmask.py (0.025 deg
rasterize, any-subcell max downsample to the 0.05 deg NA canvas), so
narrow-but-major lakes survive as water cells.

Outputs (committed assets, loaded by scripts/na_common.load_lakemask):
  assets/north_america_lakes.png  3300x1600 8-bit (255 = water treatment)
  assets/north_america_lakes.npy  3300x1600 float32 0/1
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_na_coastmask import H, LON_MAX, SUPER, W, rasterize_land

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_PNG = os.path.join(REPO_ROOT, "assets", "north_america_lakes.png")
ASSET_NPY = os.path.join(REPO_ROOT, "assets", "north_america_lakes.npy")

# Natural Earth ne_50m_lakes `name` values. Great Lakes + Nipigon (original
# basin set) + the large Canadian lakes NW of the basin and across Canada.
KEEP_LAKES = {
    "Lake Superior", "Lake Michigan", "Lake Huron", "Lake Erie",
    "Lake Ontario", "Lake Nipigon",
    "Great Bear Lake", "Great Slave Lake", "Lake Winnipeg",
    "Lake Athabasca", "Lake Manitoba", "Lake Winnipegosis",
    "Reindeer Lake", "Wollaston Lake", "Lake Mistassini",
    "Nettilling Lake", "Amadjuak Lake", "Dubawnt Lake",
    "Cedar Lake", "Southern Indian Lake", "Lake of the Woods",
}


def load_lake_shapes(shp_path, wanted):
    import shapefile
    r = shapefile.Reader(shp_path)
    shapes = r.shapes()
    recs = r.records()
    fields = [f[0] for f in r.fields[1:]]
    ni = fields.index("name")
    geoms = []
    seen = set()
    for s, rec in zip(shapes, recs):
        if rec[ni] not in wanted:
            continue
        seen.add(rec[ni])
        try:
            geoms.append(s.__geo_interface__)
        except Exception:
            continue
    missing = wanted - seen
    if missing:
        print(f"WARNING: lakes not in shapefile (skipped): {sorted(missing)}")
    print(f"selected {len(seen)} lakes, {len(geoms)} polygons")
    return geoms


def main(shp=None):
    if shp is None:
        shp = "/tmp/ne_50m_lakes/ne_50m_lakes.shp"
    if not os.path.exists(shp):
        raise SystemExit(f"lake shapefile missing: {shp} (download ne_50m_lakes.zip)")
    print("loading lake shapes...")
    geoms = load_lake_shapes(shp, KEEP_LAKES)
    w2, h2 = W * SUPER, H * SUPER
    print(f"rasterizing lakes {w2}x{h2}...")
    lake = rasterize_land(geoms, w2, h2)
    print(f"lake frac: {lake.mean():.5f}")
    # Any-subcell downsample: a canvas cell holding ANY kept-lake water is
    # water treatment (narrow lakes survive; shoreline stair-steps at most).
    small = lake.reshape(H, SUPER, W, SUPER).max(axis=(1, 3)).astype(np.float32)
    from PIL import Image
    Image.fromarray((small * 255).astype(np.uint8)).save(ASSET_PNG)
    np.save(ASSET_NPY, small)
    print(f"wrote {ASSET_PNG} + {ASSET_NPY}")
    checks = [(-88.0, 47.5, "Superior", 1.0), (-120.0, 66.0, "GreatBear", 1.0),
              (-114.0, 61.5, "GreatSlave", 1.0), (-97.5, 52.5, "Winnipeg", 1.0),
              (-110.0, 59.0, "Athabasca", 1.0), (-88.2, 43.0, "WI-inland", 0.0),
              (-100.0, 60.0, "Tundra", 0.0)]
    ok = True
    for lon, lat, name, expect in checks:
        from build_na_coastmask import LAT_MAX, LAT_MIN, LON_MIN
        x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * W)
        y = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * H)
        v = float(small[y, x])
        good = (v > 0.5) if expect > 0 else (v < 0.5)
        ok = ok and good
        print(f"  {name}: v={v:.1f} expect~{expect} {'OK' if good else 'FAIL'}")
    if not ok:
        raise SystemExit("lake checks failed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
