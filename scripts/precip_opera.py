"""OPERA European radar composite adapter (EUMETNET, CC BY 4.0).

Source: anonymous S3 listing + HTTPS object fetch, no credentials.
  Bucket: s3.waw3-1.cloudferro.com/openradar-24h
  Keys:   YYYY/MM/DD/OPERA/COMP/OPERA@<YYYYMMDDTHHMM>@0@DBZH.tiff
  Cadence: DBZH maximum reflectivity every 5 min (~3.5 MB/file).
  Grid: 1 km Lambert Azimuthal Equal Area (55N/10E), Europe + N. Atlantic.
  Band 1: dBZ float32; -9999000.0 (=-9.999e6) = undetect/dry (valid,
  suppresses lower layers); NaN = outside composite coverage.
  License: CC BY 4.0 (attribute "EUMETNET OPERA").

Only dBZ is consumed (no precipitation-type product exposed at this
endpoint), so OPERA echoes render undifferentiated on the N0Q key and
never as classified snow (rules: no snow inference from reflectivity).
"""

import datetime as dt
import os
import re
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUCKET = "https://s3.waw3-1.cloudferro.com/openradar-24h"
UA = {"User-Agent": "great-lakes-live-environment/1.0 (opera-eu-radar)"}
PRODUCT = "DBZH"
DRY_FLAG = -9999000.0
ECHO_FLOOR_DBZ = 5.0
LICENSE = "CC BY 4.0 (EUMETNET OPERA)"
FOOTPRINT = "Europe + N. Atlantic/Iceland (LAEA 1km composite)"


def _list(prefix, max_keys=1000):
    url = (f"{BUCKET}?list-type=2&prefix={prefix}"
           f"&max-keys={max_keys}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        xml = r.read().decode("utf-8", errors="replace")
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def latest_key(now=None):
    """Newest OPERA@<ts>@0@DBZH.tiff key at or before now (24h retention)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    for back in range(0, 3):
        day = now - dt.timedelta(days=back)
        prefix = day.strftime("%Y/%m/%d/OPERA/COMP/")
        try:
            keys = _list(prefix)
        except Exception:
            continue
        cands = []
        for k in keys:
            m = re.search(r"OPERA@(\d{8}T\d{4})@0@DBZH\.tiff$", k)
            if not m:
                continue
            try:
                ts = dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M").replace(
                    tzinfo=dt.timezone.utc)
            except ValueError:
                continue
            if ts <= now:
                cands.append((ts, k))
        if cands:
            return max(cands)
    raise RuntimeError("no OPERA DBZH composite found (3-day lookback)")


def fetch(key, raw_dir):
    """Download one composite TIFF to raw_dir (cached by key)."""
    import hashlib
    dest = os.path.join(
        raw_dir, "opera_" + hashlib.md5(key.encode()).hexdigest() + ".tiff")
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        return dest, True
    url = f"{BUCKET}/{key}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if data[:4] not in (b"II*\x00", b"MM\x00*"):
        raise ValueError(f"OPERA non-TIFF ({len(data)}B)")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return dest, False


def reproject_to_canvas(path, bounds, W, H):
    """Nearest-neighbour reproject of band-1 dBZ onto a canvas grid.

    Nearest (not bilinear) so flag values and storm cores are never
    blended into invented intermediates. Returns (dbz, covered, echo):
    covered = finite (dry flag counts as covered-dry, like IEM dry);
    echo = covered & >= ECHO_FLOOR_DBZ.
    """
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject
    dst_t = Affine((bounds["lon_max"] - bounds["lon_min"]) / W, 0.0,
                   bounds["lon_min"], 0.0,
                   -(bounds["lat_max"] - bounds["lat_min"]) / H,
                   bounds["lat_max"])
    grid = np.full((H, W), np.nan, dtype=np.float32)
    with rasterio.open(path) as ds:
        reproject(rasterio.band(ds, 1), grid, src_transform=ds.transform,
                  src_crs=ds.crs, dst_transform=dst_t, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest, src_nodata=np.nan,
                  dst_nodata=np.nan)
    covered = np.isfinite(grid) & (grid > DRY_FLAG - 1.0)
    echo = covered & (grid >= ECHO_FLOOR_DBZ)
    return grid, covered, echo
