"""Build the North American EXTENDED coverage mask (wind_na +
precipitation_na + cloud_cover_na).

Coverage = selected NA land + Central America, 200 statute miles
(321.869 km) offshore with a smooth feather, PLUS the full Gulf of
Mexico/America + Caribbean Sea water bodies, clipped exactly at the
South American + Russian shorelines, open Atlantic cut along the
operator's refined line (everything east of it stays transparent).

Land set = USA (incl. Alaska + Hawaii), Canada, Mexico, Greenland,
Central America (Guatemala, Belize, Honduras, El Salvador, Nicaragua,
Costa Rica, Panama), and the Caribbean island states. South America,
Russia, Europe, and Africa are excluded: their land is forced to
exactly 0 (hard shoreline cut), while Gulf/Caribbean water paints right
up to their coasts.

Same geometry engine as scripts/build_na_coastmask.py (0.025 deg
rasterize, banded geodesic EDT, smoothstep feather), but with a 200-mile
buffer, an explicit marginal-seas fill, the Atlantic cut, and
Central-America-in / South-America + Russia-out borders.

Outputs (committed assets, loaded by scripts/na_common.load_extended):
  assets/north_america_extended.png  3300x1600 8-bit (255 = keep)
  assets/north_america_extended.npy  3300x1600 float32 0..1 feather
"""

import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_na_coastmask import (H, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN,
                                SUPER, W, ocean_distance_km,
                                rasterize_land, smoothstep)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_PNG = os.path.join(REPO_ROOT, "assets", "north_america_extended.png")
ASSET_NPY = os.path.join(REPO_ROOT, "assets", "north_america_extended.npy")

SHAPE_URL = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"

BUFFER_KM = 321.8688  # 200 statute miles
FEATHER_KM = 370.0

# Kept land: North America + Central America + Caribbean island states.
# USA covers Alaska + Hawaii. Mainland runs through Panama; South
# America starts at Colombia.
NA_COUNTRIES = {
    "USA", "CAN", "MEX", "GRL", "SPM",
    "CUB", "BHS", "JAM", "HTI", "DOM", "PRI", "CYM", "TCA", "VIR",
    "ATG", "DMA", "GRD", "KNA", "LCA", "VCT", "BRB", "TTO",
    "ABW", "CUW", "SXM", "BMU", "AIA", "VGB", "MSR",
}
CAM_COUNTRIES = {"GTM", "BLZ", "HND", "SLV", "NIC", "CRI", "PAN"}
KEPT = NA_COUNTRIES | CAM_COUNTRIES
# France holds Martinique + Guadeloupe (same record as metropolitan
# France): keep ONLY the pieces inside this Caribbean box.
FRA_CARIBBEAN_BOX = (-70.0, 10.0, -58.0, 19.0)  # lon_min, lat_min, lon_max, lat_max

# Refined Atlantic cut line (operator's drawing, refined to keep the
# entire Leeward chain west of the cut): open Atlantic east of this
# polyline stays transparent (applied below lat 31 so Bermuda and the
# northern offshore keep their 200-mile rule).
CUT_LINE = [
    (-80.8, 32.3), (-78.5, 29.5), (-75.5, 26.0), (-72.0, 23.0),
    (-68.5, 20.8), (-64.5, 19.5), (-61.9, 18.2), (-60.5, 15.5),
    (-59.2, 12.5), (-60.0, 9.0), (-59.0, 6.0),
]
CUT_LAT_MAX = 31.0

# Marginal-seas fill (operator's coverage: full Gulf of Mexico/America +
# Caribbean Sea up to the South American shoreline). South-American and
# Russian land inside is hard-cut to 0 afterwards, so over-inclusion on
# land self-corrects; water stays fully painted.
SEA_FILL_POLY = [
    (-81.5, 33.5), (-78.5, 29.5), (-75.5, 26.0), (-72.0, 23.0),
    (-68.5, 20.8), (-64.5, 19.5), (-61.9, 18.2), (-60.5, 15.5),
    (-59.2, 12.5), (-60.0, 9.0), (-59.0, 5.8), (-60.5, 5.5),
    (-70.0, 8.5), (-75.0, 9.5), (-79.0, 9.0), (-83.0, 9.5),
    (-87.0, 13.0), (-88.0, 16.5), (-90.5, 19.0), (-94.0, 19.0),
    (-97.0, 22.0), (-97.0, 29.0), (-93.0, 29.5), (-89.0, 30.0),
    (-85.0, 30.0), (-82.5, 28.0), (-81.0, 25.5), (-80.0, 27.5),
    (-80.5, 30.5),
]


def load_country_shapes(shp_path, wanted):
    import shapefile
    r = shapefile.Reader(shp_path)
    shapes = r.shapes()
    recs = r.records()
    geoms = []
    seen = set()
    for s, rec in zip(shapes, recs):
        code = rec["ADM0_A3"]
        if code == "FRA":
            from shapely.geometry import box as _box, shape as _shape
            try:
                clip = _shape(s.__geo_interface__).intersection(
                    _box(*FRA_CARIBBEAN_BOX))
                if not clip.is_empty:
                    geoms.append(clip.__geo_interface__)
                    seen.add("FRA-Caribbean")
            except Exception:
                continue
            continue
        if code not in wanted:
            continue
        seen.add(code)
        try:
            geoms.append(s.__geo_interface__)
        except Exception:
            continue
    missing = wanted - seen
    if missing:
        print(f"WARNING: codes not in shapefile (skipped): {sorted(missing)}")
    print(f"selected {len(seen)} countries, {len(geoms)} polygons")
    return geoms


def excluded_land_geoms(shp_path):
    """Every non-kept country polygon in the canvas (South America,
    Russia, Europe, Africa, plus France outside the Caribbean box):
    their land cells are forced to exactly 0."""
    import shapefile
    r = shapefile.Reader(shp_path)
    from shapely.geometry import box as _box2, shape as _shape2
    other = []
    for s, rec in zip(r.shapes(), r.records()):
        code = rec["ADM0_A3"]
        if code in KEPT or code == "FRA":
            continue
        try:
            other.append(s.__geo_interface__)
        except Exception:
            continue
    for s, rec in zip(r.shapes(), r.records()):
        if rec["ADM0_A3"] != "FRA":
            continue
        try:
            g = _shape2(s.__geo_interface__)
            east = g.intersection(_box2(-60.0, 5.0, -15.0, 85.0))
            keep = east.difference(_box2(*FRA_CARIBBEAN_BOX))
            if not keep.is_empty:
                other.append(keep.__geo_interface__)
        except Exception:
            continue
    return other


def poly_mask(xs, ys, poly):
    from matplotlib.path import Path
    xx, yy = np.meshgrid(xs, ys)
    return Path(poly).contains_points(
        np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(xx.shape)


def east_of_line(xs, ys):
    """True where a cell is east of CUT_LINE (open Atlantic side), for
    latitudes below CUT_LAT_MAX. Above it the 200-mile rule owns."""
    import numpy as _np
    lons = _np.array([p[0] for p in CUT_LINE])
    lats = _np.array([p[1] for p in CUT_LINE])
    yy = ys[:, None]
    line_lon = _np.full((len(ys), 1), _np.nan)
    for i in range(len(lons) - 1):
        lo, hi = sorted((lats[i], lats[i + 1]))
        band = (yy[:, 0] >= lo) & (yy[:, 0] < hi)
        if not band.any():
            continue
        t = (yy[band, 0] - lats[i]) / (lats[i + 1] - lats[i])
        line_lon[band, 0] = lons[i] + t * (lons[i + 1] - lons[i])
    return (ys[:, None] < CUT_LAT_MAX) & _np.isfinite(line_lon) & (xs[None, :] > line_lon)


def main(shp=None):
    import urllib.request
    if shp is None:
        tmp = "/tmp/ne_50m_admin_0_countries.zip"
        if not os.path.exists(tmp):
            print("downloading Natural Earth 50m admin-0 countries...")
            req = urllib.request.Request(
                SHAPE_URL, headers={"User-Agent": "great-lakes-live-environment/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                f.write(r.read())
        d = "/tmp/ne_50m_admin"
        os.makedirs(d, exist_ok=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(d)
        shp = os.path.join(d, "ne_50m_admin_0_countries.shp")
    print("loading kept-country shapes (NA + Central America)...")
    geoms = load_country_shapes(shp, KEPT)
    w2, h2 = W * SUPER, H * SUPER
    print(f"rasterizing kept land {w2}x{h2}...")
    land = rasterize_land(geoms, w2, h2)
    print(f"kept-land frac: {land.mean():.4f}")
    print("distance transform (banded)...")
    res_deg = (LON_MAX - LON_MIN) / w2
    dist = ocean_distance_km(land, res_deg)
    alpha = 1.0 - smoothstep(BUFFER_KM, FEATHER_KM, dist)
    a2 = alpha.reshape(H, SUPER, W, SUPER).mean(axis=(1, 3)).astype(np.float32)

    xs = LON_MIN + (np.arange(W) + 0.5) * (LON_MAX - LON_MIN) / W
    ys = LAT_MAX - (np.arange(H) + 0.5) * (LAT_MAX - LAT_MIN) / H

    # Full Gulf + Caribbean water (operator's coverage, refined line).
    print("filling Gulf/Caribbean seas...")
    a2[poly_mask(xs, ys, SEA_FILL_POLY)] = 1.0

    # Open Atlantic east of the refined line stays transparent.
    print("cutting open Atlantic east of the line...")
    a2[east_of_line(xs, ys)] = 0.0

    # Hard shoreline cuts: South America, Russia, and every other
    # non-kept landmass read exactly 0 (water paints up to their
    # coasts, never onto them).
    print("rasterizing excluded land for hard border cuts...")
    excl = rasterize_land(excluded_land_geoms(shp), w2, h2)
    excl_small = excl.reshape(h2 // SUPER, SUPER, w2 // SUPER, SUPER).max(axis=(1, 3))
    a2[excl_small] = 0.0

    from PIL import Image
    Image.fromarray((a2 * 255 + 0.5).astype(np.uint8)).save(ASSET_PNG)
    np.save(ASSET_NPY, a2)
    print(f"wrote {ASSET_PNG} + {ASSET_NPY}")
    checks = [
        (-79.4, 43.7, "Toronto", 1.0), (-157.8, 21.3, "Honolulu", 1.0),
        (-99.1, 19.4, "MexicoCity", 1.0), (-90.5, 14.0, "Guatemala", 1.0),
        (-79.5, 9.0, "Panama", 1.0), (-84.1, 9.9, "SanJoseCR", 1.0),
        (-90.0, 25.0, "midGulf", 1.0), (-70.0, 15.0, "midCarib", 1.0),
        (-66.4, 18.2, "PuertoRico", 1.0), (-61.9, 17.1, "Antigua", 1.0),
        (-59.6, 13.1, "Barbados", 1.0), (-61.3, 10.4, "Trinidad", 1.0),
        (-112.5, 41.2, "GSLwater", 1.0),
        (-60.0, 8.0, "VenezuelaLand", 0.0), (-74.0, 6.0, "ColombiaLand", 0.0),
        (-175.0, 67.0, "Chukotka", 0.0), (-40.0, 20.0, "openAtlantic", 0.0),
        (-30.0, 50.0, "midAtlanticN", 0.0), (-64.8, 32.3, "Bermuda", 1.0),
    ]
    ok = True
    for lon, lat, name, expect in checks:
        x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * W)
        y = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * H)
        v = float(a2[y, x])
        good = (v > 0.9) if expect > 0 else (v < 0.05)
        ok = ok and good
        print(f"  {name}: alpha={v:.2f} expect~{expect} {'OK' if good else 'FAIL'}")
    if not ok:
        raise SystemExit("extended-mask checks failed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
