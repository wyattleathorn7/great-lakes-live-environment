"""Build the shared North American coastline mask (all four NA layers).

Mask = Natural Earth 50m land polygons + every ocean cell within 100
statute miles (160.934 km) of land, with a smooth circular-style feather
to transparent beyond that. Hawaii, the Arctic islands, the Caribbean,
Central America, and Greenland are all inside the land set, so they are
all treated identically. No rectangular cuts anywhere: the boundary
follows the shoreline geometry at every point.

Outputs (committed assets, loaded by scripts/na_common.py):
  assets/north_america_coastmask.png  3300x1600 8-bit (255 = keep)
  assets/north_america_coastmask.npy  3300x1600 float32 0..1 feather

Method:
  1. Rasterize NE50m land at 0.025 deg/px (6600x3200).
  2. Euclidean distance-to-land over ocean with scipy EDT, run in 8
     overlapping latitude bands with per-band km/px sampling
     (2.78 km/px latitude; longitude scaled by cos(latitude)), stitched
     by minimum in the overlaps.
  3. alpha = 1 on land and water <= 160.934 km offshore, smoothstep ramp
     160.934 -> 200 km to 0 (soft circular edge, never a box).
  4. Downsample (mean) to the 0.05 deg NA canvas.

Reproduce: download ne_50m_land.zip (Natural Earth) and run this file.
"""

import os
import sys
import zipfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_PNG = os.path.join(REPO_ROOT, "assets", "north_america_coastmask.png")
ASSET_NPY = os.path.join(REPO_ROOT, "assets", "north_america_coastmask.npy")

LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -180.0, -15.0, 5.0, 85.0
W, H = 3300, 1600
SUPER = 2  # 0.025 deg working grid
BUFFER_KM = 160.9344  # 100 statute miles
FEATHER_KM = 200.0
SHAPE_URL = "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip"


def load_shapes(shp_path):
    import shapefile
    r = shapefile.Reader(shp_path)
    geoms = []
    for s in r.shapes():
        try:
            geoms.append(s.__geo_interface__)
        except Exception:
            continue
    return geoms


def rasterize_land(geoms, w, h):
    from rasterio.features import rasterize
    from affine import Affine
    res = (LON_MAX - LON_MIN) / w
    transform = Affine(res, 0.0, LON_MIN, 0.0, -res, LAT_MAX)
    land = rasterize(
        [(g, 1) for g in geoms],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return land > 0


def ocean_distance_km(land, res_deg, nbands=8, overlap=120):
    """Distance-to-land (km) over ocean cells; 0 on land."""
    from scipy.ndimage import distance_transform_edt
    h, w = land.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    rows = np.linspace(0, h, nbands + 1).astype(int)
    for b in range(nbands):
        r0 = max(0, rows[b] - overlap)
        r1 = min(h, rows[b + 1] + overlap)
        lat_top = LAT_MAX - r0 * res_deg
        lat_bot = LAT_MAX - r1 * res_deg
        lat_c = np.radians(0.5 * (lat_top + lat_bot))
        km_lat = 111.195 * res_deg
        km_lon = 111.195 * res_deg * max(0.2, float(np.cos(lat_c)))
        sub = land[r0:r1]
        d = distance_transform_edt(~sub, sampling=(km_lat, km_lon)).astype(np.float32)
        cur = dist[r0:r1]
        np.minimum(cur, d, out=cur)
    dist[land] = 0.0
    return dist


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# Hand-drawn exclusion (user polygon, southern Caribbean): every island
# inside goes — Trinidad & Tobago, Grenada, the Venezuelan offshore
# islands (Margarita/Coche/Cubagua, Los Roques), and the ABC islands.
# Vertices in (lon, lat), traced from the operator's drawing.
EXCLUDE_POLY = [
    (-70.4, 12.9), (-66.0, 12.7), (-62.5, 12.5), (-60.5, 11.8),
    (-59.4, 11.0), (-59.7, 10.2), (-60.8, 9.7), (-62.5, 9.4),
    (-64.5, 10.0), (-66.5, 10.6), (-68.2, 11.2), (-70.0, 11.7),
]


def carve_exclusions(mask):
    """Zero the mask inside EXCLUDE_POLY (user's drawn shape). Operates
    on the 3300x1600 canvas mask in place semantics (returns new array).
    Applies to every NA boundary so all gradients agree."""
    from matplotlib.path import Path
    Hh, Ww = mask.shape
    xs = LON_MIN + (np.arange(Ww) + 0.5) * (LON_MAX - LON_MIN) / Ww
    ys = LAT_MAX - (np.arange(Hh) + 0.5) * (LAT_MAX - LAT_MIN) / Hh
    xx, yy = np.meshgrid(xs, ys)
    inside = Path(EXCLUDE_POLY).contains_points(
        np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(Hh, Ww)
    out = mask.copy()
    out[inside] = 0.0
    return out


def main(shp=None):
    import urllib.request
    if shp is None:
        tmp = "/tmp/ne_50m_land.zip"
        if not os.path.exists(tmp):
            print("downloading Natural Earth 50m land...")
            req = urllib.request.Request(
                SHAPE_URL, headers={"User-Agent": "great-lakes-live-environment/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                f.write(r.read())
        d = "/tmp/ne_50m_land"
        os.makedirs(d, exist_ok=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(d)
        shp = os.path.join(d, "ne_50m_land.shp")
    print("loading shapes...")
    geoms = load_shapes(shp)
    print(f"{len(geoms)} polygons")
    w2, h2 = W * SUPER, H * SUPER
    print(f"rasterizing land {w2}x{h2}...")
    land = rasterize_land(geoms, w2, h2)
    print(f"land frac: {land.mean():.3f}")
    print("distance transform (banded)...")
    res_deg = (LON_MAX - LON_MIN) / w2
    dist = ocean_distance_km(land, res_deg)
    print(f"max ocean dist: {np.nanmax(dist):.0f} km")
    alpha = 1.0 - smoothstep(BUFFER_KM, FEATHER_KM, dist)
    # downsample to NA canvas (mean preserves the feather)
    a2 = alpha.reshape(H, SUPER, W, SUPER).mean(axis=(1, 3)).astype(np.float32)
    a2 = carve_exclusions(a2)
    from PIL import Image
    Image.fromarray((a2 * 255 + 0.5).astype(np.uint8)).save(ASSET_PNG)
    np.save(ASSET_NPY, a2)
    print(f"wrote {ASSET_PNG} + {ASSET_NPY}")
    for lon, lat, name in [(-157.8, 21.1, "Honolulu"), (-79.4, 43.7, "Toronto"),
                           (-68.5, 63.7, "Iqaluit"), (-84.1, 9.9, "SanJose"),
                           (-52.0, 64.0, "Nuuk"), (-61.2, 14.6, "Martinique")]:
        x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * W)
        y = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * H)
        print(f"  {name}: alpha={a2[y, x]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
