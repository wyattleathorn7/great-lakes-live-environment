"""Build the strict North American land mask (wind + air temperature only).

Land set = authoritative Natural Earth admin-0 country polygons for the
United States (incl. Alaska + Hawaii), Canada, Mexico, Greenland, and
the Caribbean island states. The mainland ends EXACTLY at Mexico's
southern land border (Guatemala/Belize) — Central America south of
Mexico, South America, Europe, and Africa are excluded, so no ocean
fringe around those continents can ever paint.

Same geometry engine as scripts/build_na_coastmask.py (0.025 deg
rasterize, banded geodesic EDT, 100 statute miles + smoothstep feather
161 -> 200 km), but the input land is the selected country set instead
of every continent.

Outputs (committed assets, loaded by scripts/na_common.load_landmask):
  assets/north_america_landmask.png  3300x1600 8-bit (255 = keep)
  assets/north_america_landmask.npy  3300x1600 float32 0..1 feather
"""

import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_na_coastmask import (BUFFER_KM, FEATHER_KM, H, LAT_MAX, LAT_MIN,
                                LON_MAX, LON_MIN, SUPER, W,
                                ocean_distance_km, rasterize_land,
                                smoothstep)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_PNG = os.path.join(REPO_ROOT, "assets", "north_america_landmask.png")
ASSET_NPY = os.path.join(REPO_ROOT, "assets", "north_america_landmask.npy")

SHAPE_URL = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"

# North America, mainland ending at Mexico. USA covers Alaska + Hawaii.
NA_COUNTRIES = {
    "USA", "CAN", "MEX", "GRL", "SPM",
    "CUB", "BHS", "JAM", "HTI", "DOM", "PRI", "CYM", "TCA", "VIR",
    "ATG", "DMA", "GRD", "KNA", "LCA", "VCT", "BRB", "TTO",
    "ABW", "CUW", "SXM", "BMU",
}


def load_country_shapes(shp_path, wanted):
    import shapefile
    r = shapefile.Reader(shp_path)
    shapes = r.shapes()
    recs = r.records()
    geoms = []
    seen = set()
    for s, rec in zip(shapes, recs):
        code = rec["ADM0_A3"]
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
    print("loading country shapes...")
    geoms = load_country_shapes(shp, NA_COUNTRIES)
    w2, h2 = W * SUPER, H * SUPER
    print(f"rasterizing NA land {w2}x{h2}...")
    land = rasterize_land(geoms, w2, h2)
    print(f"land frac: {land.mean():.4f}")
    print("distance transform (banded)...")
    res_deg = (LON_MAX - LON_MIN) / w2
    dist = ocean_distance_km(land, res_deg)
    alpha = 1.0 - smoothstep(BUFFER_KM, FEATHER_KM, dist)
    # Perfect country cutout: every land cell of a NON-selected country
    # (Central America south of Mexico, South America, Europe, Africa) is
    # forced to exactly 0 — the 100-mile ocean feather never bleeds across
    # a land border. Ocean keeps the shoreline-following feather.
    print("rasterizing excluded land for the hard border cut...")
    import shapefile as _shp
    r = _shp.Reader(shp)
    other = []
    for s, rec in zip(r.shapes(), r.records()):
        if rec["ADM0_A3"] not in NA_COUNTRIES:
            try:
                other.append(s.__geo_interface__)
            except Exception:
                continue
    excl = rasterize_land(other, w2, h2)
    excl_small = excl.reshape(h2 // SUPER, SUPER, w2 // SUPER, SUPER).max(axis=(1, 3))
    a2 = alpha.reshape(H, SUPER, W, SUPER).mean(axis=(1, 3)).astype(np.float32)
    a2[excl_small] = 0.0
    # Crisp land-only variant (wind + air temperature): selected-country
    # land exactly, NO offshore fringe. Majority vote per canvas cell, so
    # the shoreline (and the Mexico border) stays a hard cut, not a ramp.
    land_small = land.reshape(h2 // SUPER, SUPER, w2 // SUPER, SUPER).mean(axis=(1, 3))
    crisp = ((land_small >= 0.5).astype(np.float32))
    crisp[excl_small > 0.5] = 0.0
    # ...but never punch holes inside kept land (sub-cell islets/rocks in
    # the 50m data that read as water at 0.025 deg stay painted: they are
    # land, and speckling the interior would read as missing data).
    # binary_fill_holes touches ONLY fully enclosed gaps, so the outer
    # shoreline is untouched (no closing/dilation: those would push a
    # fringe of paint into the sea, which this variant forbids).
    from scipy.ndimage import binary_fill_holes
    crisp = binary_fill_holes(crisp > 0.5).astype(np.float32)
    crisp[excl_small > 0.5] = 0.0
    from PIL import Image
    Image.fromarray((a2 * 255 + 0.5).astype(np.uint8)).save(ASSET_PNG)
    np.save(ASSET_NPY, a2)
    print(f"wrote {ASSET_PNG} + {ASSET_NPY}")
    Image.fromarray((crisp * 255).astype(np.uint8)).save(
        os.path.join(REPO_ROOT, "assets", "north_america_landonly.png"))
    np.save(os.path.join(REPO_ROOT, "assets", "north_america_landonly.npy"), crisp)
    print("wrote assets/north_america_landonly.png/.npy (crisp, no sea fringe)")
    checks = [(-157.8, 21.3, "Honolulu", 1.0), (-79.4, 43.7, "Toronto", 1.0),
              (-99.1, 19.4, "MexicoCity", 1.0), (-92.5, 15.0, "S.Mexico", 1.0),
              (-90.5, 14.0, "Guatemala", 0.0), (-79.5, 9.0, "Panama", 0.0),
              (-60.0, 8.0, "Venezuela", 0.0), (-30.0, 50.0, "midAtlantic", 0.0),
              (-20.0, 37.0, "Iberia water", 0.0), (-52.0, 64.0, "Nuuk", 1.0),
              (-61.2, 14.6, "Martinique", 1.0)]
    ok = True
    for lon, lat, name, expect in checks:
        x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * W)
        y = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * H)
        v = float(a2[y, x])
        good = (v > 0.9) if expect > 0 else (v < 0.05)
        ok = ok and good
        print(f"  {name}: alpha={v:.2f} expect~{expect} {'OK' if good else 'FAIL'}")
    if not ok:
        raise SystemExit("cutout checks failed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
