"""Pipeline P1-NA — LIVE PRECIPITATION, NORTH AMERICA (independent).

Precedence (explicit, documented):
  1) NOAA/NSSL MRMS SeamlessHSR (official NCEP MRMS GRIB2 feed, measured
     dBZ onto the original N0Q key — rain and snow share one scale)
     INSIDE the IEM effective domain
     (lon -130..-65, lat 22..50): MRMS pixels win. Replaces the old IEM
     N0Q WMS source within the same footprint (the WSR-88D network
     domain); day and night identical (pure radar, no visible-satellite
     dependency, no site-centered rendering rings). Regional MRMS domains
     (Alaska / Hawaii / Caribbean) fill the old IEM regional footprint
     with the same measured dBZ scale.
  2) ECCC NA Radar Composite 1km rain (RADAR_1KM_RRAI, mm/h) + snow
     (RADAR_1KM_RSNO, cm/h) OUTSIDE the IEM domain, gated by the
     authoritative dynamic coverage layers (RADAR_COVERAGE_RRAI/RSNO) —
     never by alpha-zero (dry pixels are valid observations).
  3) OPERA EU DBZH composite where neither covers (Europe fringe).
  4) JMA Japan nowcast intensity where none of the above cover
     (Japan + vicinity box).
  5) Regional MRMS (Alaska / Hawaii / Caribbean) in verified non-NA
     radar gaps.
  6) Translucent GFS model estimates only in verified radar gaps.
  7) Transparent elsewhere (no source coverage: northern Arctic, oceans
     beyond radar range, Greenland interior).

ECCC RAIN rates (mm/h) map to ESTIMATED dBZ equivalents via the NWS
NEXRAD Z-R bridge (Z=300R^1.4) — ECCC publishes rates, not
reflectivity, so these colors must never be read as measured dBZ
(ECCC rain classes saturate near ~57 dBZ). ECCC SNOW echoes use the
snow product for echo detection only and render on the same dBZ key:
there is no separate snow ramp, snow legend or snow-only coloring.

Exit codes: 0 updated; 2 source/validation failure (previous kept);
1 unexpected error.
"""

import datetime as dt
import gzip
import json
import os
import re
import sys
import time
import traceback
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from na_bounds import load_na_bounds
from na_common import load_rgba, wms_getmap
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs)
from build_precipitation import KEY_ROWS, SCALE_HTML
from geospatial_utils import (REPO_ROOT, RENDER_VERSION, SITE_DIR,
                              base_metadata, now_det_str,
                              promote_stage, read_state, save_png,
                              source_token, stage_dir, write_metadata,
                              write_state)

PRODUCT = "precipitation_na"
# Basin config deleted 2026-10-14; only CONFIG_NA is used (KEY_ROWS/SCALE_HTML
# come from build_precipitation shared library).
CONFIG = {}
CONFIG_NA = json.load(open(os.path.join(REPO_ROOT, "config", "precipitation_na.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "North_America_Live_Precipitation.kml"
OVERLAY_NAME = "\U0001f327\ufe0f LIVE PRECIPITATION"
SKIP_NOTE = "Turn on/off independently of all other layers."

ECCC = "https://geo.weather.gc.ca/geomet"
# Official NOAA/NCEP operational MRMS feed (verified live 2026-10-20):
# SeamlessHSR hybrid-scan reflectivity composite (GRIB2, dBZ, -999
# missing / -99 valid no-echo, 2-min cadence) — same NOAA/NSSL product
# family for CONUS and the regional Alaska / Hawaii / Caribbean domains.
MRMS = "https://mrms.ncep.noaa.gov/data/2D"
MRMS_FILE_PREFIX = "MRMS_{product}_00.00_"
MRMS_MAX_AGE_MIN = 20
MRMS_MIN_VALID_PX = 50000
MRMS_GRID_SHAPE = (3500, 7000)  # (rows, cols) — CONUS
MRMS_CONUS_DOMAIN = (54.995, 20.005, 230.005, 299.995)  # lat_n,lat_s,lon_w,lon_e
# Regional MRMS domains: own grids; expected approximate footprint used
# only for validation (geometry is read from each GRIB2 message).
MRMS_DOMAINS = {
    "alaska": {"dir": "ALASKA", "lat": (50.0, 72.0), "lon": (-176.0, -126.0)},
    "hawaii": {"dir": "HAWAII", "lat": (15.0, 26.0), "lon": (-164.0, -151.0)},
    "carib": {"dir": "CARIB", "lat": (10.0, 25.0), "lon": (-90.0, -60.0)},
}


def na_wms_111(base, layers, bounds, w, h, extra="", style=""):
    q = (f"SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layers}"
         f"&STYLES={style}&SRS=EPSG:4326&BBOX={bounds['lon_min']},{bounds['lat_min']},"
         f"{bounds['lon_max']},{bounds['lat_max']}&WIDTH={w}&HEIGHT={h}"
         f"&FORMAT=image/png&TRANSPARENT=TRUE{extra}")
    return f"{base}?{q}"


_MRMS_LISTING_TTL_S = 120.0
_MRMS_LISTING_CACHE = {}


def mrms_newest_scan_ts(url=None, prefix=None, timeout=30):
    """Newest file timestamp from an MRMS directory listing (2-min
    cadence). None on any failure. Short-lived cache: the NA and global
    canvases share one run and re-read the same listings."""
    from na_common import UA
    url = url or f"{MRMS}/SeamlessHSR"
    prefix = prefix or MRMS_FILE_PREFIX.format(product="SeamlessHSR")
    key = (url, prefix)
    now = time.time()
    hit = _MRMS_LISTING_CACHE.get(key)
    if hit is not None and now - hit[0] < _MRMS_LISTING_TTL_S:
        return hit[1]
    req = urllib.request.Request(url + "/", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        html = r.read().decode("utf-8", errors="replace")
    best = None
    for m in re.finditer(re.escape(prefix) + r"(\d{8})-(\d{4})00\.grib2\.gz",
                         html):
        ts = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
        if best is None or ts > best:
            best = ts
    _MRMS_LISTING_CACHE[key] = (now, best)
    return best


def mrms_download(url, dest, timeout=180):
    """Fetch one MRMS GRIB2.gz -> dest (decompressed). Raises on
    non-gzip / non-GRIB responses."""
    from na_common import UA
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        blob = r.read()
    if blob[:2] != b"\x1f\x8b":
        raise ValueError(f"MRMS response not gzip ({len(blob)}B head={blob[:80]!r})")
    raw = gzip.decompress(blob)
    if raw[:4] != b"GRIB":
        raise ValueError(f"MRMS payload not GRIB ({len(raw)}B head={raw[:80]!r})")
    with open(dest, "wb") as f:
        f.write(raw)
    return url


def mrms_get_scan(base, product, dest, timeout=180, retries=4, start_ts=None):
    """Newest available scan of an MRMS product in a domain directory:
    tries the newest listed timestamps (2-min cadence) until one downloads
    — NCEP can 404 a file immediately after listing it. start_ts pins the
    first candidate. Returns (scan_ts, url). Raises after retries."""
    url_dir = f"{base}/{product}"
    prefix = MRMS_FILE_PREFIX.format(product=product)
    if start_ts is None:
        ts = mrms_newest_scan_ts(url_dir, prefix)
        if ts is None:
            raise RuntimeError(f"no {product} files listed at {url_dir}")
    else:
        ts = start_ts
    last = None
    for i in range(retries):
        t = ts - dt.timedelta(minutes=2 * i)
        u = f"{url_dir}/{prefix}{t:%Y%m%d-%H%M}00.grib2.gz"
        try:
            mrms_download(u, dest, timeout=timeout)
            return t, u
        except urllib.error.HTTPError as e:
            last = e
            if e.code != 404:
                raise
    raise last


def mrms_decode(dest, expect_ts, expect_shape=None, expect_domain=None,
                missing=-999.0, vmin=-100.0, vmax=100.0):
    """Decode + validate an MRMS SeamlessHSR GRIB2 message (dBZ).

    Verifies it is genuinely the official NOAA/NSSL product (centre 161
    = NOAA OAR, regular_ll grid, message timestamp == listing timestamp,
    plausible value range, real coverage support) and returns
    (vals2d, valid2d, scan_ts, geom) with geom = (nj, ni, lat_n, lat_s,
    lon_w, lon_e) and missing -> NaN. expect_shape/expect_domain are
    validated when given (CONUS); regional domains read their own
    geometry. Raises on any failed validation — a bad/empty/partial
    product is never published as valid data."""
    import eccodes
    with open(dest, "rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
        try:
            if int(eccodes.codes_get(gid, "centre")) != 161:
                raise ValueError("MRMS centre != 161 (US NOAA OAR)")
            if eccodes.codes_get(gid, "gridType") != "regular_ll":
                raise ValueError("MRMS gridType != regular_ll")
            nj, ni = (int(eccodes.codes_get(gid, "Nj")),
                      int(eccodes.codes_get(gid, "Ni")))
            lat_n = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
            lat_s = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
            lon_w = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
            lon_e = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")
            if expect_shape is not None and (nj, ni) != expect_shape:
                raise ValueError(f"MRMS grid {nj}x{ni} != {expect_shape[0]}x{expect_shape[1]}")
            if expect_domain is not None:
                e_lat_n, e_lat_s, e_lon_w, e_lon_e = expect_domain
                if not (abs(lat_n - e_lat_n) < 0.05 and abs(lat_s - e_lat_s) < 0.05
                        and abs(lon_w - e_lon_w) < 0.05 and abs(lon_e - e_lon_e) < 0.05):
                    raise ValueError(
                        f"MRMS domain lat {lat_n}..{lat_s} lon {lon_w}..{lon_e} unexpected")
            ts = dt.datetime.strptime("%08d%04d" % (
                eccodes.codes_get(gid, "dataDate"),
                eccodes.codes_get(gid, "dataTime")), "%Y%m%d%H%M")
            if ts != expect_ts:
                raise ValueError(
                    f"MRMS message {ts:%Y-%m-%d %H:%M} != listing {expect_ts:%Y-%m-%d %H:%M}")
            vals = eccodes.codes_get_values(gid)
        finally:
            eccodes.codes_release(gid)
    grid = np.asarray(vals, dtype=float).reshape(nj, ni)
    valid = grid > missing
    n_valid = int(valid.sum())
    if n_valid < MRMS_MIN_VALID_PX:
        raise ValueError(
            f"MRMS valid coverage {n_valid}px below floor {MRMS_MIN_VALID_PX}")
    vv = grid[valid]
    if vv.min() < vmin or vv.max() > vmax:
        raise ValueError(f"MRMS value range {vv.min()}..{vv.max()} implausible")
    geom = (nj, ni, lat_n, lat_s, lon_w, lon_e)
    return np.where(valid, grid, np.nan), valid, ts, geom


def _mrms_axes(geom):
    """Descending lat / ascending lon 1-D axes from a message geom, with
    the GRIB 0..360 longitudes shifted to the canvas -180..180 (all MRMS
    domains sit > 180, so a flat -360 shift is monotonic)."""
    nj, ni, lat_n, lat_s, lon_w, lon_e = geom
    lats1d = lat_n - (lat_n - lat_s) / nj * np.arange(nj)  # row 0 = north
    lons1d = lon_w + (lon_e - lon_w) / ni * np.arange(ni)
    if lons1d[0] > 180:
        lons1d = lons1d - 360.0
    return lats1d, lons1d


def _mrms_nn(vals2d, lats1d, lons1d, bounds):
    """Nearest-neighbour sample of a source grid onto the canvas; queries
    outside the source bbox -> NaN (never extrapolated). NaNs in the
    source propagate."""
    H, W = bounds["canvas_height"], bounds["canvas_width"]
    qlat = bounds["lat_max"] - (np.arange(H) + 0.5) * (bounds["lat_max"] - bounds["lat_min"]) / H
    qlon = bounds["lon_min"] + (np.arange(W) + 0.5) * (bounds["lon_max"] - bounds["lon_min"]) / W
    dlat = (lats1d[0] - lats1d[-1]) / max(len(lats1d) - 1, 1)
    ri = np.clip(np.round((lats1d[0] - qlat) / dlat).astype(int), 0, vals2d.shape[0] - 1)
    dlon = (lons1d[-1] - lons1d[0]) / max(len(lons1d) - 1, 1)
    ci = np.clip(np.round((qlon - lons1d[0]) / dlon).astype(int), 0, vals2d.shape[1] - 1)
    out = vals2d[np.ix_(ri, ci)].astype(float)
    out[(qlat < min(lats1d[0], lats1d[-1])) | (qlat > max(lats1d[0], lats1d[-1])), :] = np.nan
    out[:, (qlon < min(lons1d[0], lons1d[-1])) | (qlon > max(lons1d[0], lons1d[-1]))] = np.nan
    return out


def _mrms_aggregate(vals2d, valid2d, lats1d, lons1d, bounds):
    """Downsample a ~1 km MRMS grid to the coarser canvas cell (max
    composite). A canvas cell paints if ANY source echo falls in it, so
    fine echo is never dropped (point sampling loses ~60% of it), and it
    takes the echo-weighted mean dBZ of the cell. -99 no-echo and NaN
    missing never paint: the sentinel can never bleed into a low-dBZ
    halo, and interior no-echo holes stay transparent."""
    from scipy.ndimage import maximum_filter, uniform_filter
    H, W = bounds["canvas_height"], bounds["canvas_width"]
    cdlat = (bounds["lat_max"] - bounds["lat_min"]) / H
    cdlon = (bounds["lon_max"] - bounds["lon_min"]) / W
    sdlat = abs(lats1d[0] - lats1d[-1]) / max(len(lats1d) - 1, 1)
    sdlon = abs(lons1d[-1] - lons1d[0]) / max(len(lons1d) - 1, 1)
    fy = max(1, int(round(cdlat / sdlat))) if sdlat else 1
    fx = max(1, int(round(cdlon / sdlon))) if sdlon else 1
    echo = (valid2d & (vals2d > -99.0)).astype(float)
    if fy == 1 and fx == 1:
        any_echo = echo > 0
        mean = np.where(any_echo, vals2d, 0.0)
    else:
        num = uniform_filter(np.where(echo > 0, vals2d, 0.0), size=(fy, fx),
                             mode="nearest")
        cov = uniform_filter(echo, size=(fy, fx), mode="nearest")
        any_echo = maximum_filter(echo, size=(fy, fx), mode="nearest") > 0
        mean = np.divide(num, cov, out=np.zeros_like(num), where=cov > 0)
    return np.where(any_echo, mean, np.nan)


def _mrms_paint(dbz_c, mask):
    """Paint resampled MRMS reflectivity onto a canvas RGBA on the
    original N0Q key bins from measured dBZ. Rain and snow share this one
    scale (no snow-only coloring). Returns (rgba, n_echo)."""
    rgba = np.zeros((*dbz_c.shape, 4), dtype=np.uint8)
    echo = np.isfinite(dbz_c) & mask
    n_echo = int(echo.sum())
    if n_echo:
        keypal = np.array([c for c, _ in KEY_ROWS], dtype=np.uint8)
        rgba[echo, :3] = keypal[_bins_for_dbz(dbz_c[echo])]
        rgba[echo, 3] = 255
    return rgba, n_echo


def _mrms_domain_mask(bounds, W, H):
    """Documented IEM effective-domain box (config iem_effective_domain)
    on this canvas. The replacement source paints strictly inside the old
    source footprint — never outside it (no footprint enlargement)."""
    dom = CONFIG_NA.get("iem_effective_domain", {})
    m = np.zeros((H, W), dtype=bool)
    try:
        c0 = max(0, int((dom["lon_min"] - bounds["lon_min"])
                        / (bounds["lon_max"] - bounds["lon_min"]) * W))
        c1 = min(W, int((dom["lon_max"] - bounds["lon_min"])
                        / (bounds["lon_max"] - bounds["lon_min"]) * W))
        r0 = max(0, int((bounds["lat_max"] - dom["lat_max"])
                        / (bounds["lat_max"] - bounds["lat_min"]) * H))
        r1 = min(H, int((bounds["lat_max"] - dom["lat_min"])
                        / (bounds["lat_max"] - bounds["lat_min"]) * H))
        m[r0:r1, c0:c1] = True
    except Exception:
        pass
    return m


def _bins_for_dbz(z):
    """Vectorized twin of _bin_for_dbz (identical bins) for measured-dBZ
    arrays — the same normalization the OPERA dBZ path uses."""
    return np.clip(np.searchsorted(KEY_BINS, z, side="right") - 1,
                   0, len(KEY_ROWS) - 1)


def mrms_grib2_url(ts, product="SeamlessHSR"):
    """Provenance URL for a validated MRMS scan timestamp ('unavailable'
    on failure)."""
    if ts is None:
        return "unavailable"
    return (f"{MRMS}/{product}/"
            f"{MRMS_FILE_PREFIX.format(product=product)}{ts:%Y%m%d-%H%M}00.grib2.gz")


def _mrms_age_check(ts):
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    age_min = (now - ts).total_seconds() / 60.0
    if age_min > MRMS_MAX_AGE_MIN or age_min < -5.0:
        raise ValueError(f"MRMS scan {ts:%Y-%m-%d %H:%M} stale ({age_min:.0f} min old)")


def fetch_mrms_seamless_hsr(scan_dt, dest, bounds, W, H, timeout=180):
    """NOAA/NSSL MRMS SeamlessHSR (CONUS) — official NCEP operational
    GRIB2 feed (replaces the old IEM N0Q WMS source within the same
    footprint).

    Newest scan from the directory listing (2-min cadence), fully
    validated, resampled onto the canvas. Measured dBZ -> the ORIGINAL N0Q
    key bins (rain and snow share the scale). -99 = valid no-echo and
    -999 = missing stay transparent (fallback sources fill as before).
    Day and night identical: pure radar composite, no visible-satellite
    dependency.

    Returns (rgba_canvas, scan_ts, prov). Raises on any failure — the
    caller then keeps the old behavior (transparent; ECCC/OPERA/GFS
    paint exactly as before)."""
    ts, url = mrms_get_scan(MRMS, "SeamlessHSR", dest, timeout=timeout)
    dbz2d, valid2d, ts, geom = mrms_decode(
        dest, ts, expect_shape=MRMS_GRID_SHAPE,
        expect_domain=MRMS_CONUS_DOMAIN, missing=-999.0, vmin=-100.0, vmax=100.0)
    _mrms_age_check(ts)
    lats1d, lons1d = _mrms_axes(geom)
    # Max-composite downsample (~1 km MRMS -> ~0.1 deg canvas): any source
    # echo inside a canvas cell paints, so fine echo is not lost to point
    # sampling, and -99 no-echo never becomes a low-dBZ halo.
    dbz_c = _mrms_nn(_mrms_aggregate(dbz2d, valid2d, lats1d, lons1d, bounds),
                     lats1d, lons1d, bounds)
    rgba, n_echo = _mrms_paint(dbz_c, _mrms_domain_mask(bounds, W, H))
    prov = (f"MRMS SeamlessHSR {ts:%Y-%m-%d %H:%M} UTC echo "
            f"px={n_echo} valid px={int(valid2d.sum())} "
            f"({url})")
    return rgba, ts, prov


def fetch_mrms_region(domain, scan_dt, dest_dir, bounds, W, H, timeout=180):
    """Regional MRMS (Alaska / Hawaii / Caribbean): official NOAA/NSSL
    SeamlessHSR measured dBZ on the same N0Q key as CONUS, painted only
    inside the domain's own valid coverage, restoring the old IEM
    regional footprint.

    Returns (rgba_canvas, cov_mask, scan_ts, prov). Raises on any failure
    — the caller then keeps the old behavior (transparent; ECCC/OPERA/JMA/
    GFS paint exactly as before)."""
    d = MRMS_DOMAINS[domain]
    base = f"{MRMS}/{d['dir']}"
    hsr_dest = os.path.join(dest_dir, f"precip_region_{domain}_hsr.grib2")
    hsr_ts, hsr_url = mrms_get_scan(base, "SeamlessHSR", hsr_dest, timeout=timeout)
    hsr, hsr_valid, hsr_ts, geom = mrms_decode(
        hsr_dest, hsr_ts, missing=-999.0, vmin=-100.0, vmax=100.0)
    _mrms_age_check(hsr_ts)
    lats1d, lons1d = _mrms_axes(geom)
    _lat_lo, _lat_hi = d["lat"]
    _lon_lo, _lon_hi = d["lon"]
    if not (_lat_lo <= min(geom[2], geom[3]) and max(geom[2], geom[3]) <= _lat_hi
            and _lon_lo <= min(geom[4], geom[5]) - (360 if geom[4] > 180 else 0)
            and max(geom[4], geom[5]) - (360 if geom[4] > 180 else 0) <= _lon_hi):
        raise ValueError(f"{domain} MRMS footprint {geom[2:]} unexpected")
    # Max-composite downsample (see fetch_mrms_seamless_hsr): no 1 km echo
    # is dropped and -99 no-echo can never bleed into a low-dBZ halo.
    cov = _mrms_nn(hsr_valid.astype(float), lats1d, lons1d, bounds) > 0.5
    dbz_c = _mrms_nn(_mrms_aggregate(hsr, hsr_valid, lats1d, lons1d, bounds),
                     lats1d, lons1d, bounds)
    rgba, n_echo = _mrms_paint(dbz_c, cov)
    prov = (f"MRMS {domain} SeamlessHSR {hsr_ts:%Y-%m-%d %H:%M} UTC echo "
            f"px={n_echo} valid px={int(hsr_valid.sum())} "
            f"({hsr_url})")
    return rgba, cov, hsr_ts, prov


# ECCC Radar-Rain_14colors / Radar-Snow_14colors class colors, extracted
# from the official GetLegendGraphic ramps (same 14-color ramp for both;
# units differ: rain mm/h, snow cm/h). Class i spans [RATE, next RATE).
# Legend URLs:
#   https://geo.weather.gc.ca/geomet?...&layer=RADAR_1KM_RRAI&...&STYLE=Radar-Rain_14colors
#   https://geo.weather.gc.ca/geomet?...&layer=RADAR_1KM_RSNO&...&STYLE=Radar-Snow_14colors
ECCC_CLASS_RGB = [
    (0, 152, 254), (0, 229, 140), (0, 203, 0), (0, 152, 0), (0, 102, 0),
    (85, 152, 0), (254, 197, 0), (254, 152, 0), (254, 102, 0), (254, 0, 0),
    (254, 1, 108), (127, 25, 178), (80, 0, 121), (51, 0, 77),
]
ECCC_RAIN_RATES = [0.1, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 50.0,
                   64.0, 100.0, 125.0, 200.0]  # mm/h class lower bounds

# ECCC surface precipitation-type classes (Radar_1km_SfcPrecipType,
# style SfcPrecipType_Dis), exact legend RGB in legend order.
SFC_TYPE_RGB = [
    (254, 30, 13),    # 0 Moderate/Heavy Freezing Rain
    (254, 203, 200),  # 1 Light Freezing Rain
    (254, 192, 7),    # 2 Hail/Rain
    (0, 128, 0),      # 3 Heavy Rain
    (0, 203, 0),      # 4 Moderate Rain
    (154, 253, 154),  # 5 Light Rain
    (155, 39, 175),   # 6 Heavy Mixed Rain/Snow
    (195, 109, 210),  # 7 Moderate Mixed Rain/Snow
    (224, 189, 230),  # 8 Light Mixed Rain/Snow
    (0, 0, 224),      # 9 Heavy Snow
    (71, 164, 240),   # 10 Moderate Snow
    (127, 223, 235),  # 11 Light Snow
]
SFC_SNOW = (9, 10, 11)
SFC_TYPE_MAX_DIST = 60.0  # beyond this -> unclassified (background etc.)

_NEARNESS = {}  # (W, H) -> land+offshore bool mask (cached per canvas)


def _iem_nearshore_mask(bounds, W, H):
    """Land + ~200 km offshore on this canvas grid, from the committed
    worldwide land mask (row 0 = 90N, verified by spot check).

    The IEM effective-domain box claims WSR-88D network coverage, but
    NEXRAD sees ~200 km offshore at most: the box's open-ocean far
    field (offshore Pacific/Atlantic/Gulf with no radar) must not
    suppress GFS estimates there. Demonstrated: 1456 GFS-wet pixels in
    one scan sat inside the box with no ECCC/OPERA coverage and a fully
    transparent IEM mosaic, yet rendered transparent — and any system
    crossing the box edge (e.g. lon -130 offshore California) gets cut
    dead-straight. Land pixels always pass, so on-land behavior is
    bit-identical; ECCC/OPERA precedence is untouched."""
    key = (W, H, bounds["lon_min"], bounds["lon_max"],
           bounds["lat_min"], bounds["lat_max"])
    if key not in _NEARNESS:
        land = np.load(os.path.join(
            REPO_ROOT, "assets", "global_landonly.npy")) > 0
        xs = np.clip((((np.arange(W) + 0.5)
                       * (bounds["lon_max"] - bounds["lon_min"]) / W
                       + bounds["lon_min"] + 180.0) / 0.1).astype(int),
                     0, 3599)
        ys = np.clip(((90.0 - (bounds["lat_max"] - (np.arange(H) + 0.5)
                               * (bounds["lat_max"] - bounds["lat_min"])
                               / H)) / 0.1).astype(int), 0, 1799)
        g = land[np.ix_(ys, xs)]
        lat_px = (bounds["lat_max"] - bounds["lat_min"]) / H
        from scipy.ndimage import binary_dilation
        _NEARNESS[key] = binary_dilation(
            g, iterations=max(1, int(round(1.8 / lat_px))))
        # 1.8 deg latitude ~= 200 km WSR-88D offshore range
    return _NEARNESS[key]


def _rate_to_dbz(r_mmh):
    """NWS NEXRAD Z-R equivalence (Z=300R^1.4): the documented bridge that
    puts ECCC precipitation rates on the N0Q dBZ key. ECCC paints equal
    colors for equal severity in rain and snow, so the rain-rate column
    owns both products' mapping (no snow-ratio invention)."""
    import math
    return 10.0 * math.log10(300.0 * max(r_mmh, 1e-6) ** 1.4)


# dBZ value of each ECCC class lower bound under the Z-R bridge — places
# both rain and snow echoes on the N0Q key by class severity.
ECCC_CLASS_DBZ = [_rate_to_dbz(r) for r in ECCC_RAIN_RATES]

# GDPS model fill renders translucent so model guidance never reads as
# observed radar (vivid = radar echo, pale = model fill where blind).
GDPS_FILL_ALPHA = 120

# Original N0Q key bins (must match KEY_ROWS ranges in order).
KEY_BINS = [5.0, 15.0, 20.0, 30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]


# GFS model estimates render in the SAME key colors as radar
# (per user direction): no muted/pale estimate swatches, no separate
# estimate rows — one visual scale everywhere. Estimates keep the same
# severity mapping and are disclosed as model output in the subtitle,
# folder text, metadata source registry, and docs (never radar).


def _bin_for_dbz(z):
    b = 0
    for lo in KEY_BINS[1:]:
        if z >= lo:
            b += 1
        else:
            break
    return min(b, len(KEY_ROWS) - 1)


def eccc_time_default(layer, timeout=30):
    try:
        url = (f"{ECCC}?service=WMS&version=1.3.0&request=GetCapabilities&layer={layer}")
        req = urllib.request.Request(url, headers={"User-Agent": "great-lakes-live-environment/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            xml = r.read().decode("utf-8", errors="replace")
        m = re.search(r'<Dimension[^>]*name="time"[^>]*default="([^"]+)"', xml)
        return m.group(1) if m else "latest"
    except Exception:
        return "unprobed"


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


IR_SATS = [
    ("GOES-19", "https://noaa-goes19.s3.amazonaws.com", "G19"),
    ("GOES-18", "https://noaa-goes18.s3.amazonaws.com", "G18"),
]
IR_CACHE_S = 2700  # 45 min: satellite texture evolves slower than echoes


def _ir_underlay(bounds, W, H):
    """Real satellite cloud texture: GOES-19/18 full-disk Band-13 (10.3 um
    clean LWIR) brightness temperatures, coldest-wins merged, enhanced to
    grayscale (cold opaque white -> warm transparent so the base map shows
    through). Works day and night. Returns (bright, alpha, prov). Shared
    with the cloud-cover realistic renderer (same cache files); the
    precipitation composite itself is radar-only and never calls this."""
    import goes_acm
    from na_common import (bin_points_to_canvas, fill_pinholes,
                           gate_field_to_coverage)
    try:
        with np.load(os.path.join(
                RAW_DIR, f"na_cloud_ir_{W}x{H}.npz")) as z:
            cb = np.asarray(z["bright"], dtype=float)
            ca = np.asarray(z["alpha"], dtype=float)
            built = dt.datetime.fromisoformat(str(z["built_utc"]))
        age = abs((dt.datetime.now(dt.timezone.utc) - built).total_seconds())
        if (cb.shape == (H, W) and ca.shape == (H, W) and age < IR_CACHE_S
                and np.isfinite(cb).any()):
            return cb, ca, f"ir-cache age={age:.0f}s"
    except Exception:
        pass
    grids = []
    scans = []
    for name, bucket, sat in IR_SATS:
        try:
            url, scan = goes_acm.newest_cmi_c13(bucket)
            dest = os.path.join(RAW_DIR, f"na_{sat.lower()}_c13.nc")
            goes_acm.download(url, dest)
            got = goes_acm.read_cmi_bt(dest)
            g, _c = bin_points_to_canvas(
                np.asarray(got["bt"], dtype=float),
                np.asarray(got["lats"], dtype=float),
                np.asarray(got["lons"], dtype=float), bounds,
                splat_radius=2)  # rotated fixed grid prints moire w/o splat
            g = fill_pinholes(g, radius=3)
            if np.isfinite(g).any():
                grids.append(g)
                scans.append(f"{name} {scan:%H:%MZ}")
        except Exception as e:
            scans.append(f"{name} unavailable:{type(e).__name__}")
    if not grids:
        raise RuntimeError(f"no GOES-IR C13 data ({scans})")
    with np.errstate(invalid="ignore"):
        cold = np.nanmin(np.stack(grids), axis=0)  # coldest wins = clouds
    m = np.isfinite(cold)
    bright = np.full((H, W), np.nan)
    alpha = np.full((H, W), np.nan)
    bright[m] = np.clip((288.0 - cold[m]) * 4.0, 0.0, 255.0)
    alpha[m] = np.clip((282.0 - cold[m]) * 4.5, 0.0, 215.0)
    alpha[m & (alpha < 3.0)] = np.nan  # warm surface: fully transparent
    bright = gate_field_to_coverage(bright, "global")
    alpha = gate_field_to_coverage(alpha, "global")
    try:
        np.savez_compressed(os.path.join(
            RAW_DIR, f"na_cloud_ir_{W}x{H}.npz"),
                            bright=bright.astype(np.float32),
                            alpha=alpha.astype(np.float32),
                            built_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except Exception:
        pass
    return bright, alpha, f"IR {'+'.join(scans)} n={int(np.isfinite(alpha).sum())}"


def run():
    bounds = load_na_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    now = dt.datetime.now(dt.timezone.utc)
    bin5 = (now.minute // 5) * 5
    scan_dt = now.replace(minute=bin5, second=0, microsecond=0)
    source_id = f"na-precip-{scan_dt.strftime('%Y%m%d-%H%M')}-r{RENDER_VERSION}"
    prev = read_state(PRODUCT)
    if (prev.get("source_id") == source_id
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png"))):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        return 0
    try:
        return _build(scan_dt, source_id, bounds, W, H)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] FAILED: {type(e).__name__}: {e}. Keeping previous.")
        return 2


def _compose_radar(scan_dt, bounds, W, H, tag):
    """Radar composite on ANY canvas bbox (NA or worldwide): the exact
    same MRMS + ECCC fetch code, IR underlay, and one-legend compose as
    the original basin layer. Returns
    (base_rgba, n_iem, n_eccc, cloud_prov, t_rain, t_snow, t_type)."""
    # 1) MRMS SeamlessHSR (CONUS echoes land inside its domain). Any
    # fetch/decode/validation failure keeps the old behavior: fully
    # transparent source (ECCC/OPERA/GFS fallbacks paint exactly as
    # before) — never a failed raster, never false precipitation.
    mrms_path = os.path.join(RAW_DIR, f"precip_{tag}_mrms.grib2")
    try:
        iem, mrms_ts, mrms_prov = fetch_mrms_seamless_hsr(
            scan_dt, mrms_path, bounds, W, H)
    except Exception as e:
        iem = np.zeros((H, W, 4), dtype=np.uint8)
        mrms_ts = None
        mrms_prov = f"MRMS unavailable:{type(e).__name__}: {e}"
        print(f"[{PRODUCT}] {mrms_prov}")
    iem_echo = iem[:, :, 3] > 0

    # 2) ECCC rain + snow + coverage GetMaps (authoritative footprint).
    # Coverage semantics (verified live 2026-10-11 by echo containment:
    # 10147/10174 rain echoes fall OUTSIDE the .INV-opaque area): the
    # RADAR_COVERAGE_*.INV style fills areas with NO radar coverage, so
    # true coverage = INV-transparent pixels. The default (non-INV)
    # style paints only outlines and must NOT be used as a mask.
    # Rate products are fetched in their documented 14-color styles so
    # every rendered color inverts EXACTLY to its rate class (same ramp
    # for rain and snow; units differ). INTERPOLATION=TRUE is ECCC's
    # documented smoother resampling for GetMap.
    t_rain = eccc_time_default("RADAR_1KM_RRAI")
    t_snow = eccc_time_default("RADAR_1KM_RSNO")
    t_type = eccc_time_default("Radar_1km_SfcPrecipType")
    paths = {}
    for key, layer, style, interp in (
            ("rain", "RADAR_1KM_RRAI", "Radar-Rain_14colors", True),
            ("snow", "RADAR_1KM_RSNO", "Radar-Snow_14colors", True),
            ("ptype", "Radar_1km_SfcPrecipType", "SfcPrecipType_Dis", False),
            ("covr", "RADAR_COVERAGE_RRAI.INV", "", False),
            ("covs", "RADAR_COVERAGE_RSNO.INV", "", False)):
        p = os.path.join(RAW_DIR, f"precip_{tag}_eccc_{key}.png")
        wms_getmap(na_wms_111(ECCC, layer, bounds, W, H,
                              extra="&INTERPOLATION=TRUE" if interp else "",
                              style=style), p)
        paths[key] = load_rgba(p, (W, H))

    # Authoritative coverage: INV-transparent == radar-covered.
    cov = (paths["covr"][:, :, 3] == 0) | (paths["covs"][:, :, 3] == 0)
    rain_e = paths["rain"][:, :, 3] > 0
    snow_e = paths["snow"][:, :, 3] > 0
    # Precedence by verified coverage (never a box, never a border):
    # ECCC (rates -> dBZ -> original key) wins everywhere its dynamic
    # coverage is valid, including most of CONUS; MRMS SeamlessHSR
    # (measured dBZ -> original key bins) fills ECCC's gaps. MRMS dry
    # inside its network footprint still suppresses lower-priority data,
    # but ECCC-covered echoes are never hidden.
    eccc_echo = cov & (rain_e | snow_e)

    # 3) Compose on ONE legend: ECCC (rate -> dBZ -> original key bins)
    # everywhere covered; IEM native N0Q fills ECCC's gaps; a 10-px
    # cross-fade rides the coverage edge (both echo -> blend on one
    # scale, never a box seam, never double-painted).
    from scipy.ndimage import distance_transform_edt
    from na_common import apply_coverage_mask
    out = np.zeros((H, W, 4), dtype=np.uint8)
    # Radar-only composite: no cloud underlay (clouds live in the cloud
    # cover layer now). Transparent where no radar echo; radar echoes
    # overwrite below.
    cloud_prov = "none (radar-only composite)"
    eccc_avail = cov & (rain_e | snow_e)
    ecol = np.zeros((H, W, 3), dtype=np.uint8)
    ealp = np.zeros((H, W), dtype=np.uint8)
    n_snow = 0
    if eccc_avail.any():
        pal = np.array(ECCC_CLASS_RGB, dtype=float)
        rpx = paths["rain"][eccc_avail].astype(float)[:, :3]
        spx = paths["snow"][eccc_avail].astype(float)[:, :3]
        cls_r = ((rpx[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        cls_s = ((spx[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        # SfcPrecipType identifies snow echoes for provenance only. All
        # echoes — rain and snow alike — render on the N0Q dBZ key: there
        # is no snow-only coloring. Mixed/freezing rain/hail take their
        # severity from whichever product shows an echo.
        tpal = np.array(SFC_TYPE_RGB, dtype=float)
        tpx = paths["ptype"][eccc_avail].astype(float)[:, :3]
        topq = paths["ptype"][eccc_avail, 3] > 0
        _d2 = ((tpx[:, None, :] - tpal[None, :, :]) ** 2).sum(axis=2)
        tcls = _d2.argmin(axis=1)
        tcls[(_d2.min(axis=1) > SFC_TYPE_MAX_DIST ** 2) | ~topq] = -1
        snow_win = np.isin(tcls, SFC_SNOW)
        # Fallback for type-unclassified echoes: the snow product echoes
        # ONLY for snow, so a snow-only echo (rain product silent) counts
        # as snow here. Unknown type never defaults to snow.
        snow_win = snow_win | ((tcls < 0) & (paths["snow"][eccc_avail, 3] > 0)
                               & ~(paths["rain"][eccc_avail, 3] > 0))
        keypal = np.array([c for c, _ in KEY_ROWS], dtype=np.uint8)
        cols = np.zeros((rpx.shape[0], 3), dtype=np.uint8)
        # Echo-grounded severity: a product showing no echo here must not
        # contribute its (meaningless) background color to the rank.
        has_r = paths["rain"][eccc_avail, 3] > 0
        has_s = paths["snow"][eccc_avail, 3] > 0
        sev = np.maximum(np.where(has_r, cls_r, -1),
                         np.where(has_s, cls_s, -1))
        sev = np.clip(sev, 0, len(ECCC_CLASS_DBZ) - 1)
        dbz = np.array(ECCC_CLASS_DBZ)[sev]
        for i, z in enumerate(dbz):
            cols[i] = keypal[_bin_for_dbz(z)]
        ecol[eccc_avail] = cols
        ealp[eccc_avail] = np.maximum(paths["rain"][eccc_avail, 3],
                                      paths["snow"][eccc_avail, 3])
        n_snow = int(snow_win.sum())
    if eccc_echo.any():
        out[eccc_echo, :3] = ecol[eccc_echo]
        out[eccc_echo, 3] = ealp[eccc_echo]
    # MRMS gap fill: already decoded to the original N0Q key rows from
    # measured dBZ at the ingestion boundary (identical normalization to
    # the OPERA dBZ path) — no source-RGB repaint.
    iem_fill = iem_echo & ~eccc_echo
    out[iem_fill] = iem[iem_fill]
    d_edge = distance_transform_edt(~cov)
    band = cov & (d_edge <= 10) & eccc_echo & iem_echo
    if band.any():
        w = (d_edge[band] / 10.0)[:, None]
        out[band, :3] = np.round(w * out[band, :3].astype(float)
                                 + (1.0 - w) * iem[band, :3].astype(float)).astype(np.uint8)
        out[band, 3] = np.round(w[:, 0] * out[band, 3].astype(float)
                                + (1.0 - w[:, 0]) * iem[band, 3].astype(float)).astype(np.uint8)
    # ---- OPERA European radar composite (EUMETNET, CC BY 4.0): genuine
    # dBZ echoes fill only verified non-NA-radar gaps (e.g. Iceland,
    # N. Atlantic fringe). Undifferentiated type -> N0Q key. Dry OPERA
    # cells suppress model fill below.
    opera_prov = "OPERA unavailable"
    opera_cov = np.zeros((H, W), dtype=bool)
    n_opera = 0
    opera_ts = None
    try:
        from precip_opera import (fetch as _op_fetch, latest_key as _op_key,
                                  reproject_to_canvas as _op_reproj)
        opera_ts, opera_key = _op_key(scan_dt)
        _op_dest, _op_cached = _op_fetch(opera_key, RAW_DIR)
        _odbz, _ocov, _oecho = _op_reproj(_op_dest, bounds, W, H)
        allowed = ~cov & (out[:, :, 3] == 0)
        paint = _oecho & allowed
        if paint.any():
            _keypal = np.array([c for c, _ in KEY_ROWS], dtype=np.uint8)
            _cols = np.zeros((int(paint.sum()), 3), dtype=np.uint8)
            for _i, _z in enumerate(_odbz[paint]):
                _cols[_i] = _keypal[_bin_for_dbz(float(_z))]
            out[paint, :3] = _cols
            out[paint, 3] = 255
        opera_cov = _ocov
        n_opera = int(paint.sum())
        opera_prov = (f"OPERA DBZH {opera_ts:%H:%MZ} "
                      f"px={n_opera} (CC BY 4.0 EUMETNET)")
    except Exception as e:
        opera_prov = f"OPERA unavailable:{type(e).__name__}"
    # ---- JMA Japan nowcast (JMA, attribution required): radar-analysis
    # echoes fill only verified non-NA/EU-radar gaps (Japan + vicinity).
    # Intensity classes (mm/h) -> Z-R estimated dBZ, same bridge as ECCC
    # rain; undifferentiated type -> N0Q key. Dry
    # JMA cells suppress model fill below.
    jma_prov = "JMA unavailable"
    jma_cov = np.zeros((H, W), dtype=bool)
    n_jma = 0
    jma_ts = None
    try:
        from precip_jma import (JMA_RATES as _j_rates,
                                fetch as _j_fetch,
                                latest_time as _j_time,
                                mosaic_to_canvas as _j_mosaic)
        jma_ts = _j_time(scan_dt)
        _j_paths = _j_fetch(jma_ts, RAW_DIR)
        _jcls, _jcov, _jecho = _j_mosaic(_j_paths, bounds, W, H)
        allowed = ~cov & (out[:, :, 3] == 0)
        paint = _jecho & allowed
        if paint.any():
            _keypal = np.array([c for c, _ in KEY_ROWS], dtype=np.uint8)
            _cols = np.zeros((int(paint.sum()), 3), dtype=np.uint8)
            for _i, _c in enumerate(_jcls[paint]):
                _dbz = _rate_to_dbz(float(_j_rates[int(_c)]))
                _cols[_i] = _keypal[_bin_for_dbz(_dbz)]
            out[paint, :3] = _cols
            out[paint, 3] = 255
        jma_cov = _jcov
        n_jma = int(paint.sum())
        jma_prov = (f"JMA nowcast {jma_ts} px={n_jma} (JMA)")
    except Exception as e:
        jma_prov = f"JMA unavailable:{type(e).__name__}"
    # ---- Regional MRMS (Alaska / Hawaii / Caribbean): official NOAA/NSSL
    # SeamlessHSR measured dBZ — same scale as the CONUS source, restoring
    # the old IEM regional footprint (Alaska, Hawaii, Puerto Rico/
    # Caribbean). Echoes fill only verified non-NA-radar gaps (same gap
    # gate as OPERA/JMA); their valid coverage also suppresses GFS model
    # fill below.
    region_cov = np.zeros((H, W), dtype=bool)
    n_region = 0
    region_prov = []
    for _dom in MRMS_DOMAINS:
        try:
            _rrgba, _rcov, _rts, _rprov = fetch_mrms_region(
                _dom, scan_dt, RAW_DIR, bounds, W, H)
            # Gate on already-painted echo only. ECCC's RADAR_COVERAGE_*.INV
            # layer is transparent OUTSIDE its own extent, which the cov
            # test above reads as "covered"; over Alaska/Hawaii/Caribbean
            # that spurious cov would block the very gaps this source
            # exists to fill. A higher-priority echo is already in `out`,
            # so alpha==0 still protects it.
            _allowed = out[:, :, 3] == 0
            _paint = (_rrgba[:, :, 3] > 0) & _allowed
            if _paint.any():
                out[_paint] = _rrgba[_paint]
            region_cov |= _rcov
            n_region += int(_paint.sum())
            region_prov.append(_rprov)
        except Exception as e:
            region_prov.append(f"{_dom} unavailable:{type(e).__name__}")
    region_prov_s = "; ".join(region_prov)
    # ---- GFS model estimates (NOAA, public domain): translucent snow and
    # rain on the same N0Q key ONLY where no radar covers
    # (MRMS CONUS + regional domains, ECCC coverage, OPERA/JMA coverage
    # all false). Estimated, never radar; restorable/off-switch via config
    # model_fill.
    gfs_prov = "GFS unavailable"
    n_snow_est = 0
    n_rain_est = 0
    if CONFIG_NA.get("model_fill", True):
        try:
            from na_common import bin_points_to_canvas as _bin_canvas
            from na_common import sanitize_model_vals as _san
            from na_common import smooth_field as _smooth
            from precip_gfs import (DISPLAY_WET_MM_H, MODEL_ALPHA,
                                     fetch_cycle as _gfs_fetch,
                                     latest_cycle as _gfs_cyc,
                                     read_fields as _gfs_read)
            _yyyymmdd, _hh = _gfs_cyc(scan_dt)
            _gdest, _gcached = _gfs_fetch(_yyyymmdd, _hh, RAW_DIR)
            _F = _gfs_read(_gdest)
            # GFS is a complete regular 0.25 deg grid: place values
            # order-independently, then bilinear-resample (continuous
            # fields, no singleton speckles to clean up afterwards).
            from na_common import resample_wcs_to_canvas as _resamp
            def _grid2d(_name):
                _vv, _la, _lo = _F[_name]
                _g = np.full((721, 1440), np.nan)
                _ii = np.clip(np.round((90.0 - np.asarray(_la)) / 0.25),
                              0, 720).astype(int)
                _jj = np.clip(np.round((((np.asarray(_lo) + 360.0) % 360.0))
                                          / 0.25), 0, 1439).astype(int)
                _g[_ii, _jj] = np.asarray(_vv, dtype=float)
                return np.roll(_g, 720, axis=1)  # 0..360 -> -180..180
            _lats1d = np.arange(90.0, -90.001, -0.25)
            _lons1d = np.arange(-180.0, 180.0, 0.25)
            # Periodic wrap: the source grid's last node is 179.75, but
            # the canvas queries to 179.95 — the shared resampler clips
            # out-of-source queries to NaN, which killed the last two
            # canvas columns at the dateline (razor cut, demonstrated:
            # cols 3598-3599 always zero). GFS is periodic, so append
            # the -180 column as +180 (exact duplicate, no invention).
            _g_grid = _grid2d("prate")
            _g_lons = np.append(_lons1d, 180.0)
            _g_grid = np.hstack([_g_grid, _g_grid[:, :1]])
            _g_rate = _resamp(_san(_g_grid * 3600.0, 0.0, 500.0),
                              _lats1d, _g_lons, bounds)
            _flag = {}
            for _fn in ("csnow", "crain", "cfrzr", "cicep"):
                _g2 = _grid2d(_fn)
                _ii = np.clip(np.round(
                    (90.0 - (bounds["lat_max"] - (np.arange(H) + 0.5)
                             * (bounds["lat_max"] - bounds["lat_min"]) / H))
                    / 0.25), 0, 720).astype(int)
                # Periodic column (not clip): lon 179.95 must read grid
                # col 0 (lon -180 == +180), not the 179.75 cell.
                _jj = (np.round(
                    ((bounds["lon_min"] + (np.arange(W) + 0.5)
                      * (bounds["lon_max"] - bounds["lon_min"]) / W) + 180.0)
                    / 0.25).astype(int) % 1440)
                _flag[_fn] = np.isfinite(_g2[np.ix_(_ii, _jj)]) & (
                    _g2[np.ix_(_ii, _jj)] >= 0.5)
            _dom = CONFIG_NA.get("iem_effective_domain", {})
            _iem_dom = np.zeros((H, W), dtype=bool)
            try:
                _c0 = max(0, int((_dom["lon_min"] - bounds["lon_min"])
                                 / (bounds["lon_max"] - bounds["lon_min"]) * W))
                _c1 = min(W, int((_dom["lon_max"] - bounds["lon_min"])
                                 / (bounds["lon_max"] - bounds["lon_min"]) * W))
                _r0 = max(0, int((bounds["lat_max"] - _dom["lat_max"])
                                 / (bounds["lat_max"] - bounds["lat_min"]) * H))
                _r1 = min(H, int((bounds["lat_max"] - _dom["lat_min"])
                                 / (bounds["lat_max"] - bounds["lat_min"]) * H))
                _iem_dom[_r0:_r1, _c0:_c1] = True
            except Exception:
                pass
            # The box overclaims open ocean (no NEXRAD there): gate it to
            # land + ~200 km offshore so far-field ocean never suppresses
            # GFS. Land behavior is unchanged (land always passes).
            _iem_dom &= _iem_nearshore_mask(bounds, W, H)
            _radar_cov = _iem_dom | cov | opera_cov | jma_cov | region_cov
            _wet = (np.isfinite(_g_rate) & (_g_rate >= DISPLAY_WET_MM_H)
                    & ~_radar_cov)
            _snow_est = _wet & _flag["csnow"]
            _rain_est = _wet & ~_snow_est & (_flag["crain"] | _flag["cfrzr"]
                                            | _flag["cicep"])
            # QC on continuous model fields only: keep organized systems
            # (>=12 cells, ~1.2 deg across), drop drizzle speckles. Radar
            # echoes are never despeckled.
            from na_common import despeckle_valid as _desp
            _keep = _desp(_snow_est | _rain_est, min_size=12)
            _snow_est = _snow_est & _keep
            _rain_est = _rain_est & _keep
            _keypal = np.array([c for c, _ in KEY_ROWS], dtype=np.uint8)
            _paint_est = _rain_est | _snow_est
            if _paint_est.any():
                # GFS estimates (rain and snow alike) render on the N0Q key
                # from PRATE — one scale, no snow-only coloring.
                _dbz = np.array([_rate_to_dbz(float(x))
                                 for x in _g_rate[_paint_est]])
                _idx = np.array([_bin_for_dbz(z) for z in _dbz])
                out[_paint_est, :3] = _keypal[_idx]
                out[_paint_est, 3] = MODEL_ALPHA
                n_rain_est = int(_rain_est.sum())
                n_snow_est = int(_snow_est.sum())
            gfs_prov = (f"GFS {_yyyymmdd} {_hh}z est rain px={n_rain_est} "
                        f"snow px={n_snow_est} (public domain)")
        except Exception as e:
            gfs_prov = f"GFS unavailable:{type(e).__name__}"
    else:
        gfs_prov = "GFS model fill off (config)"
    # ---- Uniform overlay transparency (Goal 2): the KML GroundOverlay
    # carries NO <color> opacity element, so PNG alpha is the single and
    # only transparency control. Radar echoes used to paint at alpha 255
    # (MRMS / ECCC source / OPERA) while GFS estimates paint at
    # MODEL_ALPHA — that gap is exactly why North America looked more
    # opaque than the global layer. Cap every echo pixel at MODEL_ALPHA
    # (min, not assignment: WMS-feathered echo edges below the cap and
    # fully transparent no-data pixels pass through untouched, GFS pixels
    # already at the cap are untouched, and nothing is ever double-
    # scaled). Compositing above is overwrite-by-precedence (never alpha
    # blending of two echo layers), so overlaps cannot darken.
    _a = out[:, :, 3]
    out[_a > MODEL_ALPHA, 3] = MODEL_ALPHA
    base = out  # unmasked composite; each coverage masks its own copy
    n_iem = int(iem_fill.sum())
    n_eccc = int(eccc_echo.sum())
    print(f"[{PRODUCT}] MRMS px={n_iem} ECCC px={n_eccc} "
          f"(snow px={n_snow}) OPERA px={n_opera} JMA px={n_jma} "
          f"regional px={n_region} GFSest r/s={n_rain_est}/{n_snow_est} "
          f"({100.0*(n_iem+n_eccc)/(W*H):.2f}% canvas radar)")
    info = {"n_iem": n_iem, "n_eccc": n_eccc, "n_snow": n_snow,
            "n_opera": n_opera, "n_jma": n_jma, "n_region": n_region,
            "n_rain_est": n_rain_est,
            "n_snow_est": n_snow_est, "cloud_prov": cloud_prov,
            "t_rain": t_rain, "t_snow": t_snow, "t_type": t_type,
            "opera_prov": opera_prov, "jma_prov": jma_prov,
            "gfs_prov": gfs_prov, "mrms_prov": mrms_prov,
            "region_prov": region_prov_s, "mrms_ts": mrms_ts,
            "opera_ts": (opera_ts.strftime("%Y-%m-%d %H:%M UTC")
                         if opera_ts else "unavailable")}
    return out, info


def _gdps_precip_fill(base, bounds, W, H):
    """GDPS model precip fill on the worldwide composite ONLY where
    radar is blind: GDPS_15km_PrecipRate WCS -> mm/h (adaptive units,
    ECCC publishes a bogus uom) -> dBZ via the NWS Z=300R^1.4 bridge
    -> the ORIGINAL N0Q key bins (same colors as radar). Observed
    radar echoes always win; rates below 0.1 mm/h stay transparent
    like a no-echo. Returns (base, prov). Never raises."""
    import numpy as _np
    try:
        from na_common import (fill_pinholes, gdps_wcs_grid,
                               resample_wcs_to_canvas, sanitize_model_vals,
                               smooth_field)
        arr, _la, _lo, slat, slon = gdps_wcs_grid(
            "GDPS_15km_PrecipRate", bounds, RAW_DIR)
        raw = _np.asarray(arr[0], dtype=float)
        vmax = float(_np.nanmax(raw)) if _np.isfinite(raw).any() else float("nan")
        if not _np.isfinite(vmax):
            return base, "GDPS PrecipRate empty"
        if vmax < 0.5:
            rate, unit = raw * 3.6e6, "m/s->mm/h"
        else:
            rate, unit = raw, "mm/h native"
        rate = sanitize_model_vals(rate, 0.0, 500.0)
        g = resample_wcs_to_canvas(rate, slat, slon, bounds)
        g = smooth_field(fill_pinholes(g, radius=3), size=5, passes=2)
        wet = _np.isfinite(g) & (g >= 0.1)
        blind = wet & ~(base[:, :, 3] > 0)
        n = int(blind.sum())
        if n:
            dbz = _np.array([_rate_to_dbz(float(x)) for x in g[blind]])
            keypal = _np.array([c for c, _ in KEY_ROWS], dtype=_np.uint8)
            idx = _np.zeros(dbz.shape, dtype=int)
            for i, z in enumerate(dbz):
                b = 0
                for lo in KEY_BINS[1:]:
                    if z >= lo:
                        b += 1
                    else:
                        break
                idx[i] = min(b, len(keypal) - 1)
            base[blind, :3] = keypal[idx]
            base[blind, 3] = GDPS_FILL_ALPHA
        return base, f"GDPS_15km_PrecipRate ({unit}) fill px={n}"
    except Exception as e:
        return base, f"GDPS fill unavailable:{type(e).__name__}"


def _build(scan_dt, source_id, bounds, W, H):
    from geospatial_utils import _legend_font, draw_category_legend
    from na_bounds import load_global_bounds
    gbounds = load_global_bounds()
    GW, GH = gbounds["canvas_width"], gbounds["canvas_height"]
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    os.makedirs(stage_prod, exist_ok=True)
    base, info = _compose_radar(scan_dt, bounds, W, H, "na")
    n_iem, n_eccc = info["n_iem"], info["n_eccc"]
    n_snow = info["n_snow"]
    cloud_prov = info["cloud_prov"]
    t_rain, t_snow, t_type = info["t_rain"], info["t_snow"], info["t_type"]
    # Worldwide composite: MRMS + ECCC + OPERA radar where each covers,
    # GFS model estimates only in verified radar gaps.
    gbase, ginfo = _compose_radar(scan_dt, gbounds, GW, GH, "global")
    gcloud_prov = ginfo["cloud_prov"]

    from build_kml import (COVERAGE_DIR_SUFFIX, COVERAGE_LABELS,
                           COVERAGE_DISPLAY, coverage_selector_html)
    from na_common import apply_coverage_mask
    default_cover = CONFIG_NA.get("coverage_mode", "global")
    if default_cover == "extended200":
        default_cover = "global"
    variant_dirs = {m: f"{PRODUCT}_{COVERAGE_DIR_SUFFIX[m]}"
                    for m in COVERAGE_DISPLAY}
    variant_titles = {m: f"{CONFIG_NA['title']} — {COVERAGE_LABELS[m]}"
                      for m in COVERAGE_DISPLAY}
    VARIANT_BOUNDS = {"basin": bounds, "landonly": gbounds, "global": gbounds,
                      "extended200": gbounds}
    # Per-variant composite: basin uses the NA radar composite;
    # continental + global use the worldwide composite (MRMS+ECCC+OPERA
    # radar + GFS estimates (same colors) in verified radar gaps;
    # continental additionally gated to worldwide land).
    VARIANT_BASE = {"basin": base, "landonly": gbase, "global": gbase,
                    "extended200": gbase}
    VARIANT_DIMS = {"basin": (W, H), "landonly": (GW, GH), "global": (GW, GH),
                    "extended200": (GW, GH)}
    VARIANT_PROV = {"basin": cloud_prov, "landonly": gcloud_prov,
                    "global": gcloud_prov, "extended200": gcloud_prov}
    data_time = scan_dt.strftime("%Y-%m-%d %H:%M UTC")
    subtitle = (f"Radar mosaic (MRMS CONUS + regional AK/HI/CARIB SeamlessHSR "
                f"+ ECCC RRAI/RSNO/type + OPERA EU + JMA JP) + GFS estimates | {data_time}")
    token = source_token(source_id)
    # Global source registry (per-run health + provenance for the map).
    now_utc = dt.datetime.now(dt.timezone.utc)
    registry = [
        {"id": "mrms_hsr", "footprint": "CONUS (lon -130..-65, lat 22..50)",
         "product_type": "radar reflectivity (dBZ), MRMS SeamlessHSR",
         "data_time": (ginfo["mrms_ts"].strftime("%Y-%m-%d %H:%M UTC")
                       if ginfo["mrms_ts"] else data_time),
         "license": "Public domain (NOAA/NSSL MRMS)",
         "snow_capability": "none (reflectivity only; rain and snow share the dBZ key)",
         "status": "ok" if info["n_iem"] or ginfo["n_iem"] else "no-echo"},
        {"id": "mrms_regional",
         "footprint": "Alaska + Hawaii + Caribbean (MRMS regional domains)",
         "product_type": "radar reflectivity (dBZ), MRMS SeamlessHSR",
         "data_time": ginfo.get("region_prov", "unavailable"),
         "license": "Public domain (NOAA/NSSL MRMS)",
         "snow_capability": "none (reflectivity only; rain and snow share the dBZ key)",
         "status": ("ok" if ginfo.get("n_region", 0) or info.get("n_region", 0)
                    else "no-echo/unavailable")},
        {"id": "eccc_radar", "footprint": "North America (dynamic INV mask)",
         "product_type": "radar rain/snow rate + precipitation type",
         "data_time": f"rain {t_rain}; snow {t_snow}; type {t_type}",
         "license": "ECCC GeoMet (open data)",
         "snow_capability": "detected (SfcPrecipType) but rendered on the dBZ key",
         "status": "ok" if info["n_eccc"] or ginfo["n_eccc"] else "no-echo"},
        {"id": "opera_eu", "footprint": "Europe + N. Atlantic (LAEA 1km)",
         "product_type": "radar reflectivity (dBZ)",
         "data_time": ginfo["opera_prov"], "license": "CC BY 4.0 (EUMETNET OPERA)",
         "snow_capability": "none (reflectivity only)",
          "status": ("ok" if ginfo.get("n_opera", 0) or info.get("n_opera", 0)
                     else "no-echo/unavailable")},
        {"id": "jma_jp", "footprint": "Japan + vicinity (lon 122..148, lat 24..46)",
         "product_type": "radar-analysis intensity (mm/h classes)",
         "data_time": ginfo["jma_prov"], "license": "JMA (attribution required)",
         "snow_capability": "none (intensity only)",
         "status": ("ok" if ginfo.get("n_jma", 0) or info.get("n_jma", 0)
                    else "no-echo/unavailable")},
        {"id": "gfs_model", "footprint": "worldwide 0.25deg (radar gaps only)",
         "product_type": "model estimate (PRATE + CSNOW/CRAIN flags)",
         "data_time": ginfo["gfs_prov"], "license": "Public domain (NOAA GFS)",
         "snow_capability": "estimated (CSNOW) but rendered on the dBZ key",
         "status": ("ok" if ginfo.get("n_rain_est", 0) or ginfo.get("n_snow_est", 0)
                    else "no-estimates/unavailable")},
    ]

    # Per-coverage rasters: the primary live folder references these
    # PNGs as its three hideable overlays.
    for mode in COVERAGE_DISPLAY:
        vw, vh = VARIANT_DIMS[mode]
        out_m = apply_coverage_mask(VARIANT_BASE[mode], mode)
        vprov = VARIANT_PROV[mode]
        vprod = variant_dirs[mode]
        vtitle = variant_titles[mode]
        vdir = os.path.join(stage, "site", vprod)
        os.makedirs(vdir, exist_ok=True)
        save_png(out_m, os.path.join(vdir, "current.png"))
        n_opaque = int((out_m[:, :, 3] > 0).sum())
        print(f"[{PRODUCT}] variant {mode}: opaque={n_opaque}")
        vmeta = base_metadata(
            vprod, vtitle, CONFIG_NA["freshness_label"],
            CONFIG_NA["source_name"], CONFIG_NA["source_url"],
            CONFIG_NA["variable"],
            data_time_utc=data_time,
            source_last_modified_utc=f"ECCC rain {t_rain}; snow {t_snow}; type {t_type}",
            units="dBZ key: MRMS/OPERA measured; ECCC Z-R estimated equivalents; rain and snow share one scale (no snow-only coloring)",
            source_resolution="MRMS SeamlessHSR ~1km CONUS + regional AK/HI/CARIB composite 2-min (measured dBZ -> N0Q key); ECCC 1km rain/snow/type 6-min (class severity -> estimated dBZ via NWS Z=300R^1.4; snow echoes rendered on the same key)",
            color_min=5.0, color_max=75.0, color_units="dBZ",
            missing_data_treatment=("ECCC where RADAR_COVERAGE_*.INV is transparent (all precipitation types -> dBZ key); MRMS SeamlessHSR (CONUS + regional AK/HI/CARIB) fills ECCC gaps (measured dBZ -> N0Q key bins); "
                                     "10-px cross-fade on the coverage edge; OPERA EU + JMA JP radar fill remaining gaps; GFS model estimates "
                                     "only in verified radar gaps; transparent elsewhere; "
                                     "never zero-filled."))
        vmeta["legend_scale_html"] = SCALE_HTML
        vmeta["source_id"] = source_id
        vmeta["source_version"] = token
        vmeta["na_bounds"] = VARIANT_BOUNDS[mode]
        vmeta["coverage_mode"] = mode
        vmeta["stats"] = {"opaque_pixels": n_opaque,
                          "transparent_pixels": int(vw * vh - n_opaque),
                          "underlay": vprov}
        vmeta["provenance"] = {"mrms_grib2": mrms_grib2_url(ginfo["mrms_ts"]),
                               "mrms_regional": ginfo.get("region_prov", "unavailable"),
                               "eccc_wms": ECCC, "eccc_time_rain": t_rain,
                               "eccc_time_snow": t_snow, "eccc_time_type": t_type,
                               "underlay": vprov}
        vmeta["source_registry"] = registry
        vmeta["folder_html"] = (
            f"<h2>{vtitle}</h2>"
            f"<p><b>Sources:</b> {CONFIG_NA['source_name']}<br>"
            f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br>"
            f"<b>Coverage:</b> {COVERAGE_LABELS[mode]}<br>"
            f"<b>Radar note:</b> radar echoes (MRMS CONUS + regional AK/HI/CARIB/ECCC/OPERA/JMA); GFS model estimates "
            f"(not radar) in verified radar gaps, same colors<br>"
            f"<b>Data time:</b> {data_time}</p>") + coverage_selector_html(mode)
        write_metadata(vdir, vmeta)

    def_base = VARIANT_BASE[default_cover]
    def_prov = VARIANT_PROV[default_cover]
    def_w, def_h = VARIANT_DIMS[default_cover]
    out = apply_coverage_mask(def_base, default_cover)
    save_png(out, os.path.join(stage_prod, "current.png"))
    lw, lh = draw_category_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"], subtitle,
        KEY_ROWS,
        f"MRMS {scan_dt:%H:%MZ} + ECCC rain {t_rain} / snow {t_snow} / type {t_type} | {now_det_str()}",
        note="No echo = transparent. ECCC rain = est. dBZ; snow shares the key.")
    meta = base_metadata(
        PRODUCT, CONFIG_NA["title"], CONFIG_NA["freshness_label"],
        CONFIG_NA["source_name"], CONFIG_NA["source_url"], CONFIG_NA["variable"],
        data_time_utc=data_time, source_last_modified_utc=f"ECCC rain {t_rain}; snow {t_snow}; type {t_type}",
        units="dBZ key: MRMS/OPERA measured; ECCC Z-R estimated equivalents; rain and snow share one scale (no snow-only coloring)",
        source_resolution="MRMS SeamlessHSR ~1km CONUS + regional AK/HI/CARIB composite 2-min (measured dBZ -> N0Q key); ECCC 1km NA composite 6-min (class severity -> estimated dBZ via NWS Z=300R^1.4; snow echoes rendered on the same key); canvas 3300x1600 (0.05 deg/px)",
        color_min=5.0, color_max=75.0, color_units="dBZ",
        missing_data_treatment=("ECCC wins where RADAR_COVERAGE_*.INV is transparent "
                                "(verified authoritative coverage); MRMS SeamlessHSR (CONUS + regional AK/HI/CARIB) fills ECCC gaps "
                                "(measured dBZ -> N0Q key bins); "
                                "OPERA EU radar fills remaining non-NA gaps; JMA nowcast fills Japan gaps; "
                                "GFS model estimates only in verified radar gaps; transparent elsewhere; "
                                "never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = (SCALE_HTML + " GFS model estimates "
        "(not radar) render in the same colors where no radar covers.")
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["na_bounds"] = VARIANT_BOUNDS[default_cover]
    meta["stats"] = {"mrms_pixels": n_iem, "eccc_pixels": n_eccc,
                     "opera_pixels": ginfo.get("n_opera", 0),
                     "jma_pixels": ginfo.get("n_jma", 0),
                     "mrms_regional_pixels": ginfo.get("n_region", 0),
                     "gfs_est_rain": ginfo.get("n_rain_est", 0),
                     "gfs_est_snow": ginfo.get("n_snow_est", 0),
                     "transparent_pixels": int(def_w * def_h - n_iem - n_eccc),
                     "underlay": def_prov}
    meta["provenance"] = {"mrms_grib2": mrms_grib2_url(ginfo["mrms_ts"]),
                          "mrms_regional": ginfo.get("region_prov", "unavailable"),
                          "eccc_wms": ECCC, "eccc_time_rain": t_rain, "eccc_time_snow": t_snow, "eccc_time_type": t_type,
                          "underlay": def_prov}
    meta["source_registry"] = registry
    token = meta["source_version"]
    folder_html = (f"<h2>{CONFIG_NA['title']}</h2>"
                   f"<p><b>Sources:</b> {CONFIG_NA['source_name']}<br>"
                   f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br>"
                   f"<b>Coverage:</b> {COVERAGE_LABELS[default_cover]}<br>"
                   f"<b>Radar note:</b> radar echoes (MRMS CONUS + regional AK/HI/CARIB/ECCC/OPERA/JMA); GFS model estimates "
                   f"(not radar) in verified radar gaps, same colors<br>"
                   f"<b>Data time:</b> {data_time}</p>")
    meta["folder_html"] = folder_html + coverage_selector_html(default_cover)
    meta["coverage_mode"] = default_cover
    write_metadata(stage_prod, meta)
    outs = live_out_dirs(stage, KML_FILE)
    extras = [(COVERAGE_LABELS[m], f"{variant_dirs[m]}/current.png",
               VARIANT_BOUNDS[m], 11 + i, 0)
              for i, m in enumerate([m for m in COVERAGE_DISPLAY
                                     if m != default_cover])]
    kml_text = build_kml(PRODUCT, KML_FILE, OVERLAY_NAME, f"{PRODUCT}/current.png",
                         f"{PRODUCT}/legend.png",
                         description_html(CONFIG_NA["title"], meta, SKIP_NOTE, "")
                         + coverage_selector_html(default_cover),
                         CONFIG_NA["refresh_interval_seconds"], token,
                         folder=(CONFIG_NA["title"], meta["folder_html"]),
                         out_dirs=outs["live"], meta=meta,
                         bounds=VARIANT_BOUNDS[default_cover],
                         extra_overlays=extras, folder_radio=True,
                         primary_name=COVERAGE_LABELS[default_cover])
    assert_no_vector_geometry(kml_text)
    build_entry_kml(PRODUCT, KML_FILE, OVERLAY_NAME,
                    entry_description_html(CONFIG_NA["title"], meta, SKIP_NOTE),
                    CONFIG_NA["refresh_interval_seconds"], out_dirs=outs["entry"])
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": data_time, "source_id": source_id,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
