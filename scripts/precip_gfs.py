"""GFS global model fallback adapter (NOAA/NCEP, public domain).

GFS 0.25° f000 analysis, 6-hourly cycles, fetched by GRIB byte-range
from NOMADS (same pattern as the RRFS fetches): PRATE (total
precipitation rate) + categorical CSNOW / CRAIN / CFRZR / CICEP flags.

Rendered ONLY where verified radar (IEM/ECCC/OPERA) has no coverage,
translucent and explicitly labeled model-ESTIMATED (never radar):
  CSNOW=1 & wet        -> estimated snow (blue ramp, translucent)
  CRAIN/CFRZR/CICEP=1 (CSNOW=0) & wet -> estimated rain (N0Q key, translucent)
Wet = PRATE >= 0.1 mm/h. Per snow rules: type comes from the model
flags themselves (never inferred from reflectivity or temperature
here — no temperature profile is consulted at all).
"""

import datetime as dt
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NOMADS = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
UA = {"User-Agent": "great-lakes-live-environment/1.0 (gfs-model-fill)"}
WET_MM_H = 0.1
# Display threshold for model estimates is intentionally stricter
# (0.25 mm/h ≈ light returns): model drizzle at the instrument floor
# renders as salt-and-pepper noise, not systems.
DISPLAY_WET_MM_H = 0.25
MODEL_ALPHA = 85
LICENSE = "Public domain (NOAA/NCEP GFS)"
FOOTPRINT = "Worldwide (0.25 deg global grid)"


def latest_cycle(now=None):
    """Most recent GFS cycle (00/06/12/18) with a published f000 idx."""
    from na_common import fetch_http
    now = now or dt.datetime.now(dt.timezone.utc)
    for back in range(0, 5):
        t = now - dt.timedelta(hours=6 * back)
        dd = t.strftime("gfs.%Y%m%d")
        for hh in (["%02d" % t.hour] if back == 0 else ["18", "12", "06", "00"]):
            idx_url = (f"{NOMADS}/{dd}/{hh}/atmos/"
                       f"gfs.t{hh}z.pgrb2.0p25.f000.idx")
            try:
                fetch_http(idx_url, timeout=30)
                return t.strftime("%Y%m%d"), hh
            except Exception:
                continue
    raise RuntimeError("no GFS f000 idx found (5-cycle lookback)")


def _byte_ranges(idx_text, wanted):
    from na_common import eccodes_read_idx_ranges
    # idx lines: "<n>:<start>:...:SHORT:..." — reuse shared parser with
    # level substring "surface"/"anl"-agnostic matching.
    return eccodes_read_idx_ranges(
        idx_text, [(s, "surface") for s in wanted])


def fetch_cycle(yyyymmdd, hh, raw_dir):
    """Download PRATE + type flags for one cycle (cached). Returns paths."""
    from na_common import fetch_http
    tag = f"gfs-{yyyymmdd}{hh}"
    dest = os.path.join(raw_dir, f"gfs_{tag}.grib2")
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        return dest, True
    base = f"{NOMADS}/gfs.{yyyymmdd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f000"
    idx = fetch_http(base + ".idx", timeout=60).decode("utf-8",
                                                       errors="replace")
    ranges = _byte_ranges(idx, ["PRATE", "CSNOW", "CRAIN", "CFRZR", "CICEP"])
    lo = min(a for a, _ in ranges.values())
    hi = max(b for _, b in ranges.values())
    req = urllib.request.Request(base, headers=dict(
        UA, **{"Range": f"bytes={lo}-{hi - 1}"}))
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return dest, False


def read_fields(path):
    """{name: (vals, lats, lons)} for PRATE + categorical flags."""
    from hrrr import read_messages
    got = read_messages(
        path, {"prate": 1, "csnow": 1, "crain": 1, "cfrzr": 1, "cicep": 1})
    out = {}
    for k, (vals, lats, lons, _d1, _d2) in got.items():
        out[k] = (np.asarray(vals, dtype=float),
                  np.asarray(lats, dtype=float),
                  np.asarray(lons, dtype=float))
    return out
