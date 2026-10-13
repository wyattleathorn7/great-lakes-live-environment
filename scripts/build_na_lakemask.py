"""Build the North American lake mask (wind layer only).

Water set = EVERY large natural lake in the NA bounds: the five Great
Lakes + Lake Nipigon (original inland-lake list) + all other major
Canadian lakes (Bear, Slave, Winnipeg, Athabasca, Manitoba, Winnipegosis,
Reindeer, Wollaston, Mistassini, Nettilling, Amadjuak, Dubawnt, Cedar,
Southern Indian, Woods, ...) + every other large natural lake on the
continent (Great Salt Lake, Iliamna, Nicaragua, Okeechobee, Champlain,
...). Rule-based selection from Natural Earth ne_50m_lakes geometry, so
no large lake is ever left on the land-style finish: any lake polygon
whose centroid falls in the NA bounds with a bbox area >= 0.10 deg2
(~700 km2 at mid-latitudes) is water treatment, except documented
reservoirs/impoundments (Hydro-Quebec/BC-Hydro/USBR/Missouri-system,
which stay land-style exactly like the original INLAND_LAKES convention
of natural lakes only).

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

# NA bounds (must match config/north_america_bounds.json).
LON_MIN, LAT_MIN, LAT_MAX = -180.0, 5.0, 85.0
# Bbox-area floor (deg2, cos(lat)-weighted): keeps Simcoe (0.142) and
# Nipissing (0.138) with margin; drops ponds. ne_50m is already the
# major-lake set, so this floor only trims the tail.
MIN_BBOX_DEG2 = 0.10

# Documented reservoirs/impoundments: land-style finish (original
# natural-lakes-only convention). Matched case-insensitively against the
# Natural Earth `name` (substring rules cover the Quebec Reservoire/Res.
# spellings and the Missouri-system impoundments).
EXCLUDE_SUBSTR = (
    "reservoir", "reservoire", "res.",
    "williston", "oahe", "sakakawea", "powell", "peck",
    "mead", "havasu", "shasta", "millerton", "amistad",
    "gatun", "kentucky lake", "wheeler lake", "francis case",
    "thurmond", "kinbasket", "arrow lake",
)


def _excluded(name):
    n = (name or "").lower()
    return any(s in n for s in EXCLUDE_SUBSTR)


def load_lake_shapes(shp_path):
    """Every large natural lake in the NA bounds (rule-based).

    Keeps any polygon whose centroid falls in the NA bounds with a
    cos(lat)-weighted bbox area >= MIN_BBOX_DEG2, unless its name matches
    a documented reservoir/impoundment. Unnamed polygons qualify on
    geometry alone (e.g. Alaska Peninsula lakes). Returns geojson-like
    geometries + the kept display names for logging.
    """
    import math
    import shapefile
    r = shapefile.Reader(shp_path)
    shapes = r.shapes()
    recs = r.records()
    fields = [f[0] for f in r.fields[1:]]
    ni = fields.index("name")
    geoms = []
    kept = []
    skipped_res = 0
    for s, rec in zip(shapes, recs):
        try:
            pts = s.points
        except Exception:
            continue
        if not pts:
            continue
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        cx = sum(lons) / len(lons)
        cy = sum(lats) / len(lats)
        if not (LON_MIN <= cx <= LON_MAX and LAT_MIN <= cy <= LAT_MAX):
            continue
        area = ((max(lons) - min(lons)) * (max(lats) - min(lats))
                * math.cos(math.radians(cy)))
        if area < MIN_BBOX_DEG2:
            continue
        name = rec[ni] or ""
        if _excluded(name):
            skipped_res += 1
            continue
        try:
            geoms.append(s.__geo_interface__)
        except Exception:
            continue
        kept.append(name or f"unnamed@{cx:.1f},{cy:.1f}")
    print(f"selected {len(geoms)} lake polygons "
          f"({skipped_res} reservoirs skipped)")
    return geoms, kept


def main(shp=None):
    if shp is None:
        shp = "/tmp/ne_50m_lakes/ne_50m_lakes.shp"
    if not os.path.exists(shp):
        raise SystemExit(f"lake shapefile missing: {shp} (download ne_50m_lakes.zip)")
    print("loading lake shapes...")
    geoms, kept = load_lake_shapes(shp)
    for must in ("Lake Superior", "Great Bear Lake", "Great Slave Lake",
                 "Lake Winnipeg", "Great Salt Lake", "Iliamna Lake",
                 "Lago de Nicaragua", "Lake Okeechobee", "Lake Champlain",
                 "Lake Simcoe", "Lake Nipissing"):
        if not any(must.lower() in (k or "").lower() for k in kept):
            print(f"WARNING: expected lake missing from selection: {must}")
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
              (-110.0, 59.0, "Athabasca", 1.0),
              (-112.45, 41.22, "GreatSaltLake", 1.0),
              (-85.42, 11.53, "Nicaragua", 1.0),
              (-154.63, 59.62, "Iliamna", 1.0),
              (-80.80, 26.93, "Okeechobee", 1.0),
              (-73.30, 44.50, "Champlain", 1.0),
              (-79.40, 44.44, "Simcoe", 1.0),
              (-65.12, 54.25, "SmallwoodRes", 0.0),
              (-114.19, 36.11, "MeadRes", 0.0),
              (-88.2, 43.0, "WI-inland", 0.0),
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
