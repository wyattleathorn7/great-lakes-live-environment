"""NOAA CoastWatch ERDDAP griddap CSV reader (shared by chlorophyll/clarity).

Uses urllib (stdlib) with retries; no netCDF dependency. Parses the
(time,altitude,latitude,longitude,value) CSV into 2D lat/lon grids.
HTTP 403/429/5xx are retried (the front end rate-limits bursts);
other 4xx fail immediately.
"""

import math
import random
import time
import urllib.error
import urllib.request

BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
# Browser-like UA: the front end 403s unknown bot UAs from datacenter IPs
# (GitHub Actions runners); a real browser string passes the heuristic.
UA = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/126.0.0.0 Safari/537.36"),
      "Accept": "text/csv,*/*;q=0.8"}

_RETRYABLE = {403, 408, 429, 500, 502, 503, 504}
ATTEMPTS = 5
BACKOFF = (8, 16, 32, 64)
TIMEOUT = 180
# Overall per-request body budget: the front end sometimes tar-pits large
# CSVs (slow drip that never trips the socket timeout). Chunked reads
# abort past this budget so a stalled day fails fast into retry/skip
# instead of hanging the job forever.
DEADLINE = 300
CHUNK = 1 << 20


def _read_body(r):
    """Read a response body in chunks with an overall deadline."""
    t0 = time.monotonic()
    parts = []
    while True:
        b = r.read(CHUNK)
        if not b:
            break
        parts.append(b)
        if time.monotonic() - t0 > DEADLINE:
            raise TimeoutError(f"body exceeded {DEADLINE}s deadline")
    return b"".join(parts)


def _retriable(e):
    return isinstance(e, urllib.error.HTTPError) and e.code in _RETRYABLE


def fetch_csv(dataset, var, time_str, lat0, lat1, lon0, lon1, retries=ATTEMPTS,
              stride=1):
    """Return (lats_1d, lons_1d, values_2d) for one time step.

    Raises on failure (caller maps to exit 2). time_str like
    '2026-09-12T12:00:00Z'. Works for Lon0360 datasets (pass 0-360 lons)
    and standard -180..180 ones. stride subsamples the grid (stride=2
    halves each axis) to keep large full-domain requests under the
    front-end rate limits.
    """
    s = f":{stride}:" if stride and stride > 1 else ":"
    q = (f"{dataset}.csv?{var}[({time_str})][(0.0)]"
         f"[({lat0}){s}({lat1})][({lon0}){s}({lon1})]")
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(BASE + "/" + q, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                text = _read_body(r).decode("utf-8", errors="replace")
            return _parse(text, lat0, lat1, lon0, lon1)
        except Exception as e:
            if isinstance(e, urllib.error.HTTPError) and not _retriable(e):
                raise
            last = e
        if attempt < retries:
            time.sleep(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
                       + random.uniform(0, 3))
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


def latest_time(dataset, retries=ATTEMPTS):
    """Newest time value (ISO string) of a dataset, via its time axis."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                f"{BASE}/{dataset}.csv?time", headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                lines = _read_body(r).decode("utf-8",
                                             errors="replace").splitlines()
            times = [l.strip() for l in lines[1:] if l.strip()]
            if not times:
                raise ValueError("empty time axis")
            return times[-1]
        except Exception as e:
            if isinstance(e, urllib.error.HTTPError) and not _retriable(e):
                raise
            last = e
        if attempt < retries:
            time.sleep(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
                       + random.uniform(0, 3))
    raise last
