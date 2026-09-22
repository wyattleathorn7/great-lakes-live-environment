"""One-time land-cover asset build (committed output, CI never fetches).

Mosaic, reprojected to the common canvas and mode-resampled:
  US side   : USGS NLCD 2021 via MRLC GeoServer WMS (raw class values)
  Canada side: NRCan 2020 Land Cover of Canada (NALCMS input) via S3 + rasterio
              (Canada grid cells are authoritative north of the border;
              zeros = outside Canada and are filled from NLCD)
This mirrors the NALCMS harmonization (same two national inputs).

Internal class codes (assets/leaf_landcover.png, 8-bit):
  0 nodata  1 deciduous  2 mixed  3 evergreen  4 shrub  5 grass
  6 crop    7 urban      8 barren  9 water
Behavior weights live in leaf_phenology.py.

Regeneration (one-time, local):
  python scripts/build_leaf_landcover.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geospatial_utils import REPO_ROOT, load_bounds  # noqa: E402

# class -> internal code
NLCD_MAP = {11: 9, 21: 7, 22: 7, 23: 7, 24: 7, 31: 8, 41: 1, 42: 3, 43: 2,
            52: 4, 71: 5, 81: 5, 82: 6, 90: 4, 95: 4, 12: 0, 0: 0}
NALCMS_MAP = {0: 0, 1: 3, 2: 3, 5: 1, 6: 2, 8: 4, 10: 5, 11: 4, 12: 5,
              13: 8, 14: 4, 15: 6, 16: 8, 17: 7, 18: 9, 19: 0}

CANADA_S3 = ("https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com"
             "/store/land/landcover/landcover-2020-classification.tif")
NLCD_WMS = ("https://www.mrlc.gov/geoserver/wms?SERVICE=WMS&VERSION=1.3.0"
            "&REQUEST=GetMap&LAYERS=mrlc_download:NLCD_2021_Land_Cover_L48"
            "&CRS=EPSG:4326&FORMAT=image%2Fgeotiff")


def fetch_nlcd(W, H, bounds):
    import urllib.request
    url = (f"{NLCD_WMS}&BBOX={bounds['lat_min']},{bounds['lon_min']},"
           f"{bounds['lat_max']},{bounds['lon_max']}"
           f"&WIDTH={W}&HEIGHT={H}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    import io
    from PIL import Image
    return np.array(Image.open(io.BytesIO(data)))


def fetch_canada(W, H, bounds):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    with rasterio.open("/vsicurl/" + CANADA_S3) as src:
        wb = transform_bounds("EPSG:4326", src.crs, bounds["lon_min"],
                              bounds["lat_min"], bounds["lon_max"],
                              bounds["lat_max"])
        win = from_bounds(*wb, src.transform)
        return src.read(1, window=win, out_shape=(H, W),
                        resampling=Resampling.nearest)


def main():
    bounds = load_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    print("fetching NLCD (US)...")
    us = fetch_nlcd(W, H, bounds)
    print("fetching Canada NALCMS input...")
    ca = fetch_canada(W, H, bounds)
    umap = np.vectorize(lambda v: NLCD_MAP.get(int(v), 0), otypes=[np.uint8])
    cmap = np.vectorize(lambda v: NALCMS_MAP.get(int(v), 0), otypes=[np.uint8])
    us_c, ca_c = umap(us), cmap(ca)
    mosaic = np.where(ca_c != 0, ca_c, us_c)
    u, c = np.unique(mosaic, return_counts=True)
    names = {0: "nodata", 1: "deciduous", 2: "mixed", 3: "evergreen",
             4: "shrub", 5: "grass", 6: "crop", 7: "urban", 8: "barren",
             9: "water"}
    print("mosaic:", {names[int(k)]: int(v) for k, v in zip(u, c)})
    out = os.path.join(REPO_ROOT, "assets", "leaf_landcover.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from PIL import Image
    Image.fromarray(mosaic.astype(np.uint8), mode="L").save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
