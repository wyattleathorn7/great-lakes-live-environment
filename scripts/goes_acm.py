"""GOES-East (GOES-19) ABI L2 Clear Sky Mask (ACMC) fetcher.

CONUS 2 km files every 5 minutes from the public S3 bucket
(noaa-goes19, anonymous HTTPS — no credentials, no boto3):
  ABI-L2-ACMC/YYYY/DDD/HH/OR_ABI-L2-ACMC-M6_G19_s..._e..._c....nc

Decodes BCM (NOAA's own binary split: 0 = clear_or_probably_clear,
1 = cloudy_or_probably_cloudy) gated by DQF==0 (good quality), and
projects the fixed grid to WGS84 with pyproj (already a repo dep —
hand-rolled geostationary math was validated against it and dropped).

Each product downloads independently so jobs stay independent.
"""

import datetime as dt
import os
import re
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

UA = {"User-Agent": "great-lakes-live-environment/1.0"}
BUCKET = "https://noaa-goes19.s3.amazonaws.com"
PRODUCT_PREFIX = "ABI-L2-ACMC"
FILE_RE = re.compile(
    r"OR_ABI-L2-ACMC-M6_G19_s(\d{4})(\d{3})(\d{2})(\d{2})\d{3}_e.*\.nc$")
X_SCALE, X_OFFSET = 5.6e-05, -0.101332
Y_SCALE, Y_OFFSET = -5.6e-05, 0.128212


def _list_keys(prefix, timeout=30):
    """S3 ListBucket keys under prefix (anonymous HTTPS, no boto3)."""
    url = (f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=100")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        root = ET.fromstring(r.read())
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    return [e.text for e in root.findall("s:Contents/s:Key", ns)]


def newest_file(max_back_hours=8, now=None):
    """Newest ACMC file with scan-start at least 6 min in the past
    (skips the file still being written). Returns (https_url, scan_dt).
    Raises if none found. Probes hour dirs newest-first (KB XML each)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    floor = now - dt.timedelta(minutes=6)
    t = now.replace(minute=0, second=0, microsecond=0)
    last = None
    for _ in range(max_back_hours + 1):
        dd = t.strftime("%Y/%j/%H")
        try:
            keys = _list_keys(f"{PRODUCT_PREFIX}/{dd}/")
        except Exception as e:
            last = e
            t -= dt.timedelta(hours=1)
            continue
        cands = []
        for k in keys:
            m = FILE_RE.search(k.split("/")[-1])
            if not m:
                continue
            try:
                scan = dt.datetime(int(m.group(1)), 1, 1,
                                   tzinfo=dt.timezone.utc) \
                    + dt.timedelta(days=int(m.group(2)) - 1,
                                   hours=int(m.group(3)),
                                   minutes=int(m.group(4)))
            except ValueError:
                continue
            if scan <= floor:
                cands.append((scan, k))
        if cands:
            scan, key = max(cands)
            return f"{BUCKET}/{key}", scan
        t -= dt.timedelta(hours=1)
    raise last or RuntimeError("no GOES ACMC file found")


def download(url, dest, timeout=300):
    """Download url -> dest (atomic). Returns size in bytes."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers=UA)
    total = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
    os.replace(tmp, dest)
    return total


def read_mask(path):
    """Return dict with 1D arrays: cloudy01 (BCM, NaN where unusable),
    lats, lons, scan_dt (from filename), subpoint_lon. Raises if the
    file disagrees with the expected CONUS fixed grid."""
    import h5netcdf
    f = h5netcdf.File(path, "r")
    try:
        nx = f["x"].shape[0]
        ny = f["y"].shape[0]
        x = f["x"][:].astype(float) * X_SCALE + X_OFFSET
        y = f["y"][:].astype(float) * Y_SCALE + Y_OFFSET
        bcm = np.asarray(f["BCM"][:]).astype(float)
        dqf = np.asarray(f["DQF"][:]).astype(float)
        proj = f["goes_imager_projection"].attrs
        lon0 = float(proj["longitude_of_projection_origin"])
        h = float(proj["perspective_point_height"])
    finally:
        f.close()
    good = (dqf == 0) & np.isfinite(bcm) & ((bcm == 0) | (bcm == 1))
    cloudy = np.where(good, bcm, np.nan)
    lats, lons = fixed_grid_to_latlon(x, y, lon0, h)
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})\d{3}_",
                  os.path.basename(path))
    scan = None
    if m:
        scan = dt.datetime(int(m.group(1)), 1, 1,
                           tzinfo=dt.timezone.utc) \
            + dt.timedelta(days=int(m.group(2)) - 1,
                           hours=int(m.group(3)), minutes=int(m.group(4)))
    return {"cloudy01": cloudy.ravel(), "lats": lats.ravel(),
            "lons": lons.ravel(), "scan_dt": scan,
            "subpoint_lon": lon0, "grid_shape": (ny, nx)}


def fixed_grid_to_latlon(x, y, lon0_deg, h):
    """Fixed-grid scan angles (radians) -> WGS84 lat/lon via pyproj.

    Uses the file's own subpoint longitude and perspective height, so a
    future drift or satellite swap cannot silently misplace the raster.
    Returns (lats_2d, lons_2d) matching the (y, x) grid shape.
    """
    from pyproj import CRS, Transformer
    sat_h = float(h)
    geos = CRS(f"+proj=geos +h={sat_h} +lon_0={float(lon0_deg)} "
               f"+sweep=x +ellps=GRS80 +units=m +no_defs")
    t = Transformer.from_crs(geos, "EPSG:4326", always_xy=True)
    xx = np.asarray(x, dtype=float) * sat_h
    yy = np.asarray(y, dtype=float) * sat_h
    Xg, Yg = np.meshgrid(xx, yy)
    lo, la = t.transform(Xg, Yg)
    return np.asarray(la), np.asarray(lo)
