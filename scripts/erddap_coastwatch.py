"""NOAA CoastWatch ERDDAP griddap CSV reader (shared by chlorophyll/clarity).

Uses urllib (stdlib) with retries; no netCDF dependency. Parses the
(time,altitude,latitude,longitude,value) CSV into 2D lat/lon grids.
HTTP 403/429/5xx are retried (the front end rate-limits bursts);
other 4xx fail immediately.
"""

import math
import time
import urllib.error
import urllib.request

BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
UA = {"User-Agent": "great-lakes-live-environment/1.0"}

_RETRYABLE = {403, 408, 429, 500, 502, 503, 504}


def _retriable(e):
    return isinstance(e, urllib.error.HTTPError) and e.code in _RETRYABLE


def fetch_csv(dataset, var, time_str, lat0, lat1, lon0, lon1, retries=3):
    """Return (lats_1d, lons_1d, values_2d) for one time step.

    Raises on failure (caller maps to exit 2). time_str like
    '2026-09-12T12:00:00Z'. Works for Lon0360 datasets (pass 0-360 lons)
    and standard -180..180 ones.
    """
    q = (f"{dataset}.csv?{var}[({time_str})][(0.0)]"
         f"[({lat0}):({lat1})][({lon0}):({lon1})]")
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(BASE + "/" + q, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                text = r.read().decode("utf-8", errors="replace")
            return _parse(text, lat0, lat1, lon0, lon1)
        except Exception as e:
            if isinstance(e, urllib.error.HTTPError) and not _retriable(e):
                raise
            last = e
        if attempt < retries:
            time.sleep(6 * attempt)
    raise last


def _parse(text, lat0, lat1, lon0, lon1):
    import numpy as np
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].lower().startswith("error"):
        raise ValueError(f"ERDDAP error: {text[:200]}")
    lats, lons, vals = [], [], []
    for line in lines[2:]:
        p = line.split(",")
        if len(p) < 5:
            continue
        try:
            la, lo = float(p[2]), float(p[3])
            v = float(p[4]) if p[4] not in ("NaN", "") else math.nan
        except ValueError:
            continue
        lats.append(la)
        lons.append(lo)
        vals.append(v)
    lats = np.array(lats)
    lons = np.array(lons)
    vals = np.array(vals)
    ulat = np.unique(lats)
    ulon = np.unique(lons)
    if ulat.size < 2 or ulon.size < 2:
        raise ValueError(f"degenerate grid {ulat.size}x{ulon.size}")
    # ERDDAP returns lat descending or ascending? normalize ascending.
    lat_idx = np.searchsorted(ulat, lats)
    lon_idx = np.searchsorted(ulon, lons)
    grid = np.full((ulat.size, ulon.size), np.nan)
    grid[lat_idx, lon_idx] = vals
    # rows: index 0 = southernmost; canvas mapping uses lat values directly
    return ulat, ulon, grid


def latest_time(dataset, retries=3):
    """Newest time value (ISO string) of a dataset, via its time axis."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                f"{BASE}/{dataset}.csv?time", headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                lines = r.read().decode("utf-8", errors="replace").splitlines()
            times = [l.strip() for l in lines[1:] if l.strip()]
            if not times:
                raise ValueError("empty time axis")
            return times[-1]
        except Exception as e:
            if isinstance(e, urllib.error.HTTPError) and not _retriable(e):
                raise
            last = e
        if attempt < retries:
            time.sleep(6 * attempt)
    raise last
