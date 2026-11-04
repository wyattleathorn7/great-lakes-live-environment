"""Multi-satellite infrared mosaic for LIVE CLOUD COVER (white IR imagery).

Sources (all verified live 2026-10-16, anonymous HTTPS, no credentials):
  GOES-19 / GOES-18  NOAA ABI L2 CMIPF Band-13 (10.3 um) full-disk NetCDF
                     via S3 listings. Direct brightness-temperature (K)
                     decode with per-file calibration + DQF==0 gate
                     (see goes_acm). 10-min FD scans. Covers Americas,
                     Atlantic, E/C Pacific.
  HIMAWARI-9        GIBS WMTS Himawari_AHI_Band13_Clean_Infrared GetMap.
                     Same 10.3 um window, 10-min. Server-mosaicked; RGB
                     inverted to BT through NASA's documented 1 degC
                     colormap (runtime-validated). Covers Asia + W/C Pacific.
                     Direct ISatSS download (88 tiles, ~26 MB/scan for Ch13
                     alone, 971 MB for all channels) was evaluated and
                     rejected for the 5-minute budget; the WMTS picture is
                     documented as imagery-derived with +/-0.5 C LUT
                     quantization (negligible for white-cloud rendering).
  METEOSAT FES/IODC  EUMETSAT EUMETView WMS msg_fes:ir108 (0 deg, prime)
                     + msg_iodc:ir108 (Meteosat-9, 45.5 E). SEVIRI IR10.8
                     L1.5, 15-min, grayscale rendering. No anonymous WCS
                     offers these layers, and the Data Store API needs a key,
                     so the WMS grayscale is used as DOCUMENTED IMAGERY
                     (bright = cold, verified live) mapped through its own
                     transfer -- never presented as calibrated BT.
  VIIRS (polar)      GIBS WMTS VIIRS_SNPP_Brightness_Temp_BandI5 Day+Night,
                     same LUT-inversion treatment, DAILY cadence. Used ONLY
                     where no geostationary source is valid (poles, seams).
                     Documented as daily supplementary, never 5-min fresh.

Common representation: brightness temperature (K). EUMETView grayscale is
converted to pseudo-BT by inverting the white-cloud display transfer, so
every source merges in one physical space with one rendering at the end.

Mosaic policy (documented, never coldest-wins):
  - validity masks only; missing is never clear sky;
  - geostationary preferred over VIIRS-daily everywhere;
  - among GEO, fresher scan wins; overlaps within MAX_SKEW_MIN (15) get a
    12-px linear cross-fade (feather_merge), larger skews resolve hard to
    the fresher observation (feather 0) -- never blind BT averaging;
  - antimeridian blended (blend_antimeridian); uncovered pixels stay NaN.

Each adapter exposes source id, satellite/instrument/product, acquisition
time, file/URL, quality info, coverage, BT grid + validity, diagnostics.
GOES-specific fixed-grid math is never reused for other satellites: WMS
sources arrive server-georeferenced to the requested canvas BBOX.
"""

import datetime as dt
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UA = {"User-Agent": "great-lakes-live-environment/1.0 (sat-ir)"}

# Display transfer (shared by every BT source; documented in metadata):
#   bright = clip((BT_HI - bt) * BT_GAIN, 0, 255)   white = cold
#   alpha  = clip((ALPHA_HI - bt) * ALPHA_GAIN, 0, ALPHA_MAX)
#   alpha < ALPHA_FLOOR -> transparent (warm surface / clear).
BT_HI, BT_GAIN = 288.0, 4.0
ALPHA_HI, ALPHA_GAIN, ALPHA_MAX, ALPHA_FLOOR = 282.0, 4.5, 215.0, 15.0
# Polar-surface ambiguity mitigation. Single-channel IR cannot separate a
# winter land/ice surface from cloud (a clear -30C ground reads every bit
# as warm as a cloud deck under the 282 K water transfer, and the old
# hard suppression (ALPHA_HI_LAND >= |lat| 60N) printed razor-sharp lines
# and blocky coast silhouettes exactly at the threshold boundary — ob-
# served live 2026-11-02 and reported by the user). From
# POLAR_BLEND_MIN_LAT the IR-derived alpha is blended into the GDPS
# model TotalCloudCover; past POLAR_BLEND_FULL_LAT the model owns the
# transfer, so the poles show a coherent line-free cloud picture.
POLAR_BLEND_MIN_LAT = 55.0
POLAR_BLEND_FULL_LAT = 65.0
# GDPS WCS cache ages: GDPS_15km layers refresh on the hourly runs.
GDPS_TCC_CACHE_MIN = 45

# Land-aware cloud detection: over land, the surface can be cold (Antarctica,
# Greenland, winter continents), so the IR transfer alone cannot distinguish
# cold cloud tops from cold surface. Over land, require a colder temperature
# for opacity to separate cloud from surface. Over ocean, the surface is warm
# and uniform, so the standard threshold works.
ALPHA_HI_LAND = 240.0
# Latitude band where ALPHA_HI_LAND applies. The land gate exists solely
# so polar ice-sheet surfaces (Antarctica, Greenland interior) do not
# render as phantom cloud. Mid-latitude land (all of the Great Lakes and
# the continental US) stays OUTSIDE this band and uses the full ocean
# threshold, so every cloud type is visible from the ground view.
COLD_PLATEAU_LAT = 60.0

# EUMETView grayscale imagery transfer (imagery-derived, documented as such):
# verified live 2026-10-16: warm Sahara ~= 24 (dark), cloud p95 ~= 213.
GRAY_WARM, GRAY_GAIN, GRAY_ALPHA_MAX = 20.0, 1.2, 215.0

MAX_SKEW_MIN = 15          # max GEO-GEO time skew for feathered blending
FEATHER_PX = 12            # cross-fade width where blending is permitted
DISK_FADE_PX = 12          # dissolve valid-coverage edges (no hard cut lines)

# Provenance codes for the diagnostic raster.
PROV = {"missing": 0, "GOES-19": 1, "GOES-18": 2, "HIMAWARI-9": 3,
        "METEOSAT-FES": 4, "METEOSAT-IODC": 5, "VIIRS": 6, "GDPS-TCC": 7}

GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
GIBS_WMTS_CAPS = ("https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
                  "wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities")
GIBS_CMAP_IR = ("https://gibs.earthdata.nasa.gov/colormaps/v1.3/"
                "Clean_Longwave_Infrared_Window_Band.xml")
GIBS_CMAP_VIIRS = ("https://gibs.earthdata.nasa.gov/colormaps/v1.3/"
                   "VIIRS_Brightness_Temp_BandI5.xml")
EUMET_WMS = "https://view.eumetsat.int/geoserver/wms"

GOES_SATS = [
    ("GOES-19", "https://noaa-goes19.s3.amazonaws.com", "G19"),
    ("GOES-18", "https://noaa-goes18.s3.amazonaws.com", "G18"),
]
GIBS_HIMAWARI_LAYER = "Himawari_AHI_Band13_Clean_Infrared"
GIBS_VIIRS_LAYERS = [
    "VIIRS_SNPP_Brightness_Temp_BandI5_Day",
    "VIIRS_SNPP_Brightness_Temp_BandI5_Night",
]
EUMET_LAYERS = [
    ("METEOSAT-FES", "msg_fes:ir108"),
    ("METEOSAT-IODC", "msg_iodc:ir108"),
]


# ------------------------------------------------------------------ helpers

def _http(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _parse_iso(s):
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s)


def bt_to_white(bt, land_mask=None):
    """BT(K) grid -> (bright 0..255, alpha 0..215) white-cloud rendering.

    When land_mask is provided (1.0 = land, 0.0 = ocean), land pixels use
    ALPHA_HI_LAND for the alpha transfer so cold surfaces (Antarctica,
    Greenland) are not rendered as opaque cloud. Ocean pixels use ALPHA_HI.
    """
    bt = np.asarray(bt, dtype=float)
    m = np.isfinite(bt)
    bright = np.full(bt.shape, np.nan)
    alpha = np.full(bt.shape, np.nan)
    bright[m] = np.clip((BT_HI - bt[m]) * BT_GAIN, 0.0, 255.0)
    alpha_hi = np.full(bt.shape, ALPHA_HI)
    if land_mask is not None:
        lm = np.asarray(land_mask)
        if lm.shape == bt.shape:
            alpha_hi = np.where(lm >= 0.5, ALPHA_HI_LAND, ALPHA_HI)
    alpha[m] = np.clip((alpha_hi[m] - bt[m]) * ALPHA_GAIN, 0.0, ALPHA_MAX)
    alpha[m & (alpha < ALPHA_FLOOR)] = np.nan
    bright[~np.isfinite(alpha)] = np.nan
    return bright, alpha


def gray_to_pseudo_bt(gray):
    """EUMETView grayscale (bright = cold) -> pseudo-BT(K) in the same
    space as the display transfer (inverts bt_to_white). Imagery-derived;
    documented as such in metadata -- not calibrated radiances."""
    g = np.asarray(gray, dtype=float)
    return BT_HI - g / BT_GAIN


def gray_to_white(gray, valid):
    """Direct imagery transfer for EUMETView grayscale."""
    g = np.asarray(gray, dtype=float)
    v = np.asarray(valid, dtype=bool) & np.isfinite(g)
    bright = np.full(g.shape, np.nan)
    alpha = np.full(g.shape, np.nan)
    bright[v] = np.clip(g[v], 0.0, 255.0)
    alpha[v] = np.clip((g[v] - GRAY_WARM) * GRAY_GAIN, 0.0, GRAY_ALPHA_MAX)
    alpha[v & (alpha < ALPHA_FLOOR)] = np.nan
    bright[~np.isfinite(alpha)] = np.nan
    return bright, alpha


def load_land_mask(shape, bounds=None):
    """Load the global land mask (assets/global_landonly.npy) sampled over
    `bounds` at the target resolution. Returns None if the asset is missing.

    The asset is the worldwide grid (3600x1800 @ 0.10 deg, -180..180 /
    -90..90, row 0 = 90N). Sampling each canvas pixel's lon/lat center into
    the world grid keeps land geographically aligned on every canvas.
    Resizing the whole world map to the canvas shape instead (the previous
    behavior when the shape differs) smears unrelated geography onto
    regional canvases: on the continental canvas the Great Lakes sampled
    equatorial Africa (78% misclassified as land), and on the basin canvas
    the entire world was squeezed into the 19.5x9 deg box -- suppressing
    real cloud wherever the shuffled mask claimed land (bt_to_white drops
    BT >= 240 K on land cells). Observed live 2026-11-01/02.
    """
    try:
        from na_common import load_continental
        mask = load_continental()
        h, w = int(shape[0]), int(shape[1])
        if (h, w) == mask.shape:
            return mask.astype(np.float32)
        if bounds is None:
            # No bounds available: legacy whole-map resize (kept only so the
            # function still works for callers that cannot supply bounds).
            from PIL import Image
            im = Image.fromarray((mask * 255).astype(np.uint8))
            im = im.resize((w, h), Image.NEAREST)
            return (np.array(im) >= 128).astype(np.float32)
        # Geographic sampling: canvas cell centers -> world grid indices.
        # Same scheme as the near-land dilation mask in build_precipitation_na.
        STEP_LON, STEP_LAT = 0.1, 0.1
        x = np.clip((((np.arange(w) + 0.5)
                      * (bounds["lon_max"] - bounds["lon_min"]) / w
                      + bounds["lon_min"] + 180.0) / STEP_LON).astype(int),
                    0, mask.shape[1] - 1)
        y = np.clip(((90.0 - (bounds["lat_max"] - (np.arange(h) + 0.5)
                               * (bounds["lat_max"] - bounds["lat_min"])
                               / h)) / STEP_LAT).astype(int),
                    0, mask.shape[0] - 1)
        return mask[np.ix_(y, x)].astype(np.float32)
    except Exception:
        return None


def cold_plateau_land_mask(land_mask, bounds):
    """Restrict the land mask to the cold-surface plateaus (see
    COLD_PLATEAU_LAT): mid-latitude land rows become 0 (ocean-like
    threshold in bt_to_white), so cloud detection is the same over land
    and water from the ground view. None/missing input passes through.
    NOTE: the mosaic render path no longer uses this gate -- the polar
    zone is handled by the GDPS model blend (polar_weights); the helper
    stays for legacy experiment callers and tests.
    """
    if land_mask is None:
        return None
    land_mask = np.asarray(land_mask, dtype=np.float32)
    h = land_mask.shape[0]
    lats = (bounds["lat_max"] - (np.arange(h) + 0.5)
            * (bounds["lat_max"] - bounds["lat_min"]) / h)
    midlat = (lats < COLD_PLATEAU_LAT) & (lats > -COLD_PLATEAU_LAT)
    return np.where(midlat[:, None], 0.0, land_mask)


def _smoothstep(t):
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def polar_weights(shape, bounds):
    """Per-ROW weight 0..1 for the polar GDPS blend: 0 through the
    mid-latitudes, smoothstep 0->1 across POLAR_BLEND_MIN_LAT..FULL_LAT
    of |lat| (North AND South). Returns (rows,) float."""
    h = int(shape[0])
    lats = (bounds["lat_max"] - (np.arange(h) + 0.5)
            * (bounds["lat_max"] - bounds["lat_min"]) / h)
    ramp = (POLAR_BLEND_FULL_LAT - POLAR_BLEND_MIN_LAT)
    up = _smoothstep((np.abs(lats) - POLAR_BLEND_MIN_LAT) / ramp)
    return up


def _gdps_tcc_alpha(bounds, raw_dir):
    """GDPS 15km TotalCloudCover -> (alpha 0..ALPHA_MAX, valid) at the
    canvas resolution. Worldwide fetch (cached ~GDPS run cadence), then
    bilinear-resampled to this canvas via the shared WCS resampler.
    Returns (alpha, valid): thin cover <1% maps to alpha 0 (honest
    'clear' from the model, NOT missing); missing model cells stay
    invalid so the IR value underneath survives."""
    import na_common as _nc
    from scipy.ndimage import gaussian_filter
    world = {"lon_min": -180.0, "lon_max": 180.0,
             "lat_min": -90.0, "lat_max": 90.0}
    vals, _la2, _lo2, lats1d, lons1d = _nc.gdps_wcs_grid(
        "GDPS_15km_TotalCloudCover", world, raw_dir,
        cache_min=GDPS_TCC_CACHE_MIN)
    v = np.asarray(vals, dtype=float)
    if v.ndim == 3:      # (bands, h, w) raster
        v = v[0]
    g = _nc.resample_wcs_to_canvas(v, lats1d, lons1d, bounds)
    # Model cells print through at fine canvases; soften by ~0.3 deg at
    # this canvas' pitch so the blend looks like weather, not lattice.
    res = (float(bounds["lon_max"]) - float(bounds["lon_min"])) / \
        float(bounds["canvas_width"])
    sigma_px = max(1.0, 0.3 / res)
    if np.isfinite(g).any():
        fill = float(np.nanmean(g))
        gs = gaussian_filter(np.where(np.isfinite(g), g, fill),
                             sigma_px, truncate=2.5)
        g = np.where(np.isfinite(g), gs, g)
    alpha = np.clip((np.nan_to_num(g, nan=0.0) / 100.0)
                    * ALPHA_MAX, 0.0, ALPHA_MAX)
    valid = np.isfinite(g)
    alpha[~valid] = np.nan
    return alpha, valid


def normalize_sources(layers):
    """Radiometric normalization: adjust each source's BT values so that
    the mean in the overlap region with the previous source matches. This
    accounts for different transfer functions (GOES true BT, Himawari
    LUT-inverted, Meteosat grayscale) before blending.

    Returns a new list of (grid, valid) with adjusted grids.
    """
    if len(layers) < 2:
        return layers
    out = [layers[0]]
    for i in range(1, len(layers)):
        prev_grid, prev_valid = out[-1]
        curr_grid, curr_valid = layers[i]
        both_valid = prev_valid & curr_valid & np.isfinite(prev_grid) & np.isfinite(curr_grid)
        if both_valid.sum() < 100:
            out.append(layers[i])
            continue
        bias = np.nanmean(prev_grid[both_valid]) - np.nanmean(curr_grid[both_valid])
        adjusted = curr_grid.astype(float) + bias
        out.append((adjusted, curr_valid))
    return out


# ------------------------------------------------------- GIBS LUT inversion

_LUT_CACHE = {}


def gibs_lut(cmap_url):
    """Build RGB->BT(C) lookup from NASA's documented colormap XML.

    Validates entry shape (1 degC bins, unique RGB keys); raises if the
    published colormap changes shape so we fail loudly instead of
    mis-inverting temperatures.
    """
    if cmap_url in _LUT_CACHE:
        return _LUT_CACHE[cmap_url]
    data = _http(cmap_url, timeout=60).decode("utf-8", errors="replace")
    # The Brightness Temperature block carries the physical scale. Prefer
    # value="(lo,hi]" (real units) over sourceValue (raw counts): the VIIRS
    # colormap's sourceValue is detector counts (e.g. 8547-8718) while its
    # value is Kelvin (180.0-180.6). The Clean-IR colormap uses identical
    # degree-C pairs in both attributes. Nodata entries (transparent=true)
    # never match the paren-open pattern and are skipped.
    block = data
    m = re.search(
        r'<ColorMap\s+title="Brightness Temperature"\s+units="([^"]+)"',
        data)
    units = m.group(1) if m else "°C"
    if m:
        start = m.start()
        end = data.find("</ColorMap>", start)
        block = data[start:end if end > 0 else len(data)]
    entries = re.findall(
        r'<ColorMapEntry\s+rgb="(\d+),(\d+),(\d+)"[^>]*?'
        r'value="\(([-\d.]+),([-\d.]+)\]"', block)
    if not entries:  # legacy files without a value attribute: °C assumed
        entries = re.findall(
            r'<ColorMapEntry\s+rgb="(\d+),(\d+),(\d+)"[^>]*?'
            r'sourceValue="\(([-\d.]+),([-\d.]+)\]"', block)
        units = "°C"
    if len(entries) < 50:
        raise ValueError(f"GIBS colormap changed shape ({len(entries)} "
                         f"entries): {cmap_url}")
    lut = {}
    for r, g, b, lo, _hi in entries:
        key = (int(r), int(g), int(b))
        mid = (float(lo) + float(_hi)) / 2.0
        mid_k = mid if units == "K" else mid + 273.15
        if key in lut:
            raise ValueError(f"GIBS colormap has duplicate RGB {key}: "
                             f"inversion ambiguous")
        lut[key] = mid_k
    _LUT_CACHE[cmap_url] = lut
    return lut


def invert_gibs_png(path, cmap_url):
    """GIBS Clean-IR / VIIRS I5 PNG -> BT(K) grid (NaN where transparent).

    Quantization is +/-0.5 C per the documented 1-degC bins -- negligible
    for white-cloud rendering; recorded in diagnostics.
    """
    from PIL import Image
    lut = gibs_lut(cmap_url)
    im = Image.open(path).convert("RGBA")
    a = np.array(im)
    opaque = a[:, :, 3] > 0
    bt = np.full(a.shape[:2], np.nan)
    if not opaque.any():
        return bt
    # WMS resamples the tiled source to our canvas, so most pixels carry
    # interpolated colors between documented bins. Exact matches invert to
    # the bin midpoint (+/-0.5 C); interpolants map to the nearest bin
    # (+/-1 C, documented quantization, negligible for cloud rendering).
    from scipy.spatial import cKDTree
    keys = np.array(sorted(lut), dtype=float)
    tree = cKDTree(keys)
    cols = a[opaque][:, :3].astype(float)
    uniq, inv = np.unique(cols, axis=0, return_inverse=True)
    dist, idx = tree.query(uniq, k=1)
    # Resampling blends data edges with the transparent background, making
    # halo colors far from any documented bin. Those pixels stay invalid
    # (transparent) instead of inventing BT; a render dominated by them is
    # rejected by the caller via the finite-fraction check.
    MAX_DIST = 50.0
    vals = np.full(len(uniq), np.nan)
    near = dist <= MAX_DIST
    vals[near] = np.array(
        [lut[tuple(map(int, keys[i]))] for i in idx[near]])  # Kelvin
    bt[opaque] = vals[inv]
    return bt


# ---------------------------------------------------------------- WMS fetch

# Probe memo: resolving the newest renderable GIBS granule (the walk of
# small probe renders behind gibs_newest_valid_time) is IDENTICAL for
# every canvas — data availability does not depend on the canvas. Both
# render passes share one result per (layer, requested TIME), which
# removes ~2/3 of probe traffic per layer; a NEGATIVE outcome ("no valid
# render", e.g. a canvas outside a disk footprint) is memoized too so the
# regional canvases never repeat the full 19-step walk. TTL bounds a
# memo's lifetime to half an hour.
_PROBE_MEMO_TTL_S = 1800.0
_PROBE_MEMO: dict = {}


def _probe_memo_get(layer, primary_iso):
    """Memoized probe outcome: iso string, False (no valid render), or None
    (no memo / expired)."""
    item = _PROBE_MEMO.get((layer, primary_iso))
    if item is None:
        return None
    value, at = item
    if (time.monotonic() - at) > _PROBE_MEMO_TTL_S:
        _PROBE_MEMO.pop((layer, primary_iso), None)
        return None
    return value


def _probe_memo_put(layer, primary_iso, value):
    _PROBE_MEMO[(layer, primary_iso)] = (value, time.monotonic())

def wms_getmap_png(base, layer, bounds, W, H, time_iso, dest, version="1.1.1"):
    """WMS GetMap -> dest PNG (atomic). Server georeferences to our canvas
    BBOX, so no client-side projection math is needed for WMS sources."""
    if version == "1.1.1":
        params = {"SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
                  "LAYERS": layer, "STYLES": "", "FORMAT": "image/png",
                  "TRANSPARENT": "TRUE", "SRS": "EPSG:4326",
                  "BBOX": f"{bounds['lon_min']},{bounds['lat_min']},"
                          f"{bounds['lon_max']},{bounds['lat_max']}",
                  "WIDTH": str(W), "HEIGHT": str(H)}
    else:
        params = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
                  "LAYERS": layer, "STYLES": "", "FORMAT": "image/png",
                  "TRANSPARENT": "TRUE", "CRS": "EPSG:4326",
                  "BBOX": f"{bounds['lat_min']},{bounds['lon_min']},"
                          f"{bounds['lat_max']},{bounds['lon_max']}",
                  "WIDTH": str(W), "HEIGHT": str(H)}
    if time_iso:
        params["TIME"] = time_iso
    url = base + "?" + urllib.parse.urlencode(params)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"WMS non-PNG ({len(data)}B head={data[:160]!r}) "
                         f"layer={layer}")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return url


def gibs_latest_time(layer, caps_cache=None):
    """Latest TIME value for a GIBS WMTS layer from GetCapabilities."""
    if caps_cache is None:
        caps_cache = _http(GIBS_WMTS_CAPS,
                           timeout=120).decode("utf-8", errors="replace")
    # Find the Layer block, then its Dimension default + trailing Value range.
    i = caps_cache.find("<ows:Identifier>" + layer + "</ows:Identifier>")
    if i < 0:
        raise ValueError(f"GIBS layer not in capabilities: {layer}")
    block = caps_cache[i:i + 20000]
    m = re.search(r"<Default>([^<]+)</Default>", block)
    if not m:
        raise ValueError(f"GIBS layer has no Default time: {layer}")
    return m.group(1).strip(), caps_cache


def eumet_latest_time(layer, caps_cache=None):
    """Latest TIME value for an EUMETView WMS layer."""
    if caps_cache is None:
        caps_cache = _http(EUMET_WMS + "?SERVICE=WMS&REQUEST=GetCapabilities",
                           timeout=120).decode("utf-8", errors="replace")
    i = caps_cache.find("<Name>" + layer + "</Name>")
    if i < 0:
        raise ValueError(f"EUMETView layer not in capabilities: {layer}")
    block = caps_cache[i:i + 8000]
    m = re.search(r'<Dimension[^>]*default="([^"]+)"[^>]*>', block)
    if not m:
        raise ValueError(f"EUMETView layer has no time dimension: {layer}")
    return m.group(1).strip(), caps_cache


# ---------------------------------------------------------- scan selection

def select_scans(now=None, raw_dir=None):
    """Choose the newest scan per source. Cheap (listings + capabilities);
    no bulk downloads. Returns (scans, diagnostics).

    scans: {name: {kind, url|bucket, scan_dt, ...}}; failures are recorded
    in diagnostics and the source is simply absent (never fabricated).
    """
    import goes_acm
    now = now or dt.datetime.now(dt.timezone.utc)
    scans = {}
    diag = {}

    for name, bucket, _sat in GOES_SATS:
        try:
            url, scan = goes_acm.newest_cmi_c13(bucket)
            scans[name] = {"kind": "goes-bt", "bucket": bucket, "url": url,
                           "scan_dt": scan, "file": url.split("/")[-1]}
            diag[name] = f"{scan:%Y-%m-%d %H:%MZ}"
        except Exception as e:
            diag[name] = f"unavailable: {type(e).__name__}: {e}"

    try:
        gibs_caps = _http(GIBS_WMTS_CAPS,
                          timeout=120).decode("utf-8", errors="replace")
    except Exception as e:
        gibs_caps = None
        diag["GIBS-caps"] = f"unavailable: {type(e).__name__}: {e}"
    if gibs_caps is not None:
        try:
            t, _ = gibs_latest_time(GIBS_HIMAWARI_LAYER, gibs_caps)
            scans["HIMAWARI-9"] = {"kind": "gibs-bt",
                                   "layer": GIBS_HIMAWARI_LAYER,
                                   "scan_dt": _parse_iso(t), "time": t,
                                   "cmap": GIBS_CMAP_IR}
            diag["HIMAWARI-9"] = t
        except Exception as e:
            diag["HIMAWARI-9"] = f"unavailable: {type(e).__name__}: {e}"
        for vl in GIBS_VIIRS_LAYERS:
            try:
                t, _ = gibs_latest_time(vl, gibs_caps)
                # Daily product: keep the calendar date as the scan id.
                scans.setdefault("VIIRS", {"kind": "gibs-viirs",
                                           "layers": [], "scan_dt": None,
                                           "cmap": GIBS_CMAP_VIIRS})
                scans["VIIRS"]["layers"].append({"layer": vl, "time": t})
                if scans["VIIRS"]["scan_dt"] is None:
                    day = _parse_iso(t[:10])
                    if day.tzinfo is None:
                        day = day.replace(tzinfo=dt.timezone.utc)
                    scans["VIIRS"]["scan_dt"] = day
                diag[vl] = t
            except Exception as e:
                diag[vl] = f"unavailable: {type(e).__name__}: {e}"

    try:
        eumet_caps = _http(EUMET_WMS + "?SERVICE=WMS&REQUEST=GetCapabilities",
                           timeout=120).decode("utf-8", errors="replace")
    except Exception as e:
        eumet_caps = None
        diag["EUMET-caps"] = f"unavailable: {type(e).__name__}: {e}"
    if eumet_caps is not None:
        for name, layer in EUMET_LAYERS:
            try:
                t, _ = eumet_latest_time(layer, eumet_caps)
                scans[name] = {"kind": "eumet-gray", "layer": layer,
                               "scan_dt": _parse_iso(t), "time": t}
                diag[name] = t
            except Exception as e:
                diag[name] = f"unavailable: {type(e).__name__}: {e}"
    return scans, diag


def source_identity(scans):
    """Deterministic source id from actual observation times (never wall
    clock). Same observations -> same id -> skip path, stable ?v=."""
    parts = []
    for name in sorted(scans):
        s = scans[name]
        scan = s.get("scan_dt")
        tag = scan.strftime("%Y%m%dT%H%M") if scan else "none"
        if s["kind"] == "gibs-viirs":
            tag = "+".join(sorted(l["time"][:10] for l in s["layers"]))
        parts.append(f"{name}@{tag}")
    return "cloudIR-" + "-".join(parts)


# ---------------------------------------------------------- per-canvas build

def land_clip_mask(shape, bounds):
    """Bilinearly interpolated worldwide land mask (na_common world
    grid) at an arbitrary canvas size/bounds.

    Used by the Continental Coverage variant to clip cloud alpha to
    land exactly per the family contract. Sampled INTERPOLATED (not
    nearest): nearest resize prints 0.1-degree coast blocks at the
    fine canvases, then reads as jagged 'odd lines' in Google Earth.
    Pixel centers map bijectively onto the 0.1-degree world cell
    centers; a scale-aware Gaussian (about 0.15 deg) removes the
    residual source-grid print-through so coast stays a smooth fade at
    any resolution. Raises if the asset is missing.
    """
    from na_common import load_continental
    from scipy.ndimage import map_coordinates, gaussian_filter
    mask = np.asarray(load_continental(), dtype=float)
    h, w = int(shape[0]), int(shape[1])
    lon_l, lon_r = bounds["lon_min"], bounds["lon_max"]
    lat_t, lat_b = bounds["lat_max"], bounds["lat_min"]
    lons = (lon_l + (np.arange(w) + 0.5) * (lon_r - lon_l) / w)
    lats = (lat_t - (np.arange(h) + 0.5) * (lat_t - lat_b) / h)
    # cell-center convention: world cell k center at -180 + (k+0.5)*0.1
    fy = ((90.0 - lats) / 0.1 - 0.5)[:, None] * np.ones((1, w))
    fx = ((lons + 180.0) / 0.1 - 0.5)[None, :] * np.ones((h, 1))
    smooth = map_coordinates(mask, [fy.ravel(), fx.ravel()], order=1,
                             mode="constant", cval=0.0,
                             prefilter=False).reshape(h, w)
    # ~0.15 deg of coast feather at this canvas' pixel pitch (>=1 px).
    res = max(abs(lon_r - lon_l) / w, abs(lat_t - lat_b) / h)
    sigma = max(1.0, 0.15 / res)
    if sigma > 1.05:
        smooth = gaussian_filter(smooth, sigma)
    return np.clip(smooth, 0.0, 1.0).astype(np.float32)


def _goes_inverse_transformer(lon0, sat_h):
    """pyproj lat/lon -> GOES fixed grid (radians), keyed to the file's
    own subpoint/height (same CRS as goes_acm.fixed_grid_to_latlon,
    inverted)."""
    from pyproj import CRS, Transformer
    geos = CRS(f"+proj=geos +h={float(sat_h)} +lon_0={float(lon0)} "
               f"+sweep=x +ellps=GRS80 +units=m +no_defs")
    return Transformer.from_crs("OGC:CRS84", geos, always_xy=True)


def _goes_sample_grid(sel, bounds, raw_dir):
    """Download (scan-cached) + inverse-georeference one GOES source to
    the full-resolution BT canvas grid.

    The previous path splatted the satellite's fixed-grid POINTS onto
    the canvas (nearest + pinhole fill). Wherever the fixed grid is
    coarser than the canvas pitch that sliced displaced fingerprints of
    concentric arc moire onto the extended coverages, and the unfillable
    high-latitude gaps printed as dotted rows (user report 2026-11-02).
    Resampling here goes the other way: every canvas pixel center is
    reprojected into the file's own fixed grid and the native CMI field
    is bilinearly sampled there with native DQF validity -- a continuous
    field with no lattice, so no moire, no speckle rows, no fill gaps.
    """
    import goes_acm
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    tag = sel["scan_dt"].strftime("%Y%m%d-%H%M")
    dest = os.path.join(raw_dir, f"mosaic_{sel['bucket'].split('//')[1].split('.')[0]}"
                                 f"_{tag}.nc")
    # Cache bad-file quarantine: any cached file that fails to parse is
    # removed and re-fetched once, so a truncated download can never
    # poison layers cycle after cycle (observed: >100 KB partial S3
    # bodies were accepted by the old size-only cache check for weeks).
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        try:
            got = goes_acm.read_cmi_2d(dest)
        except Exception as e:
            print(f"[cloud] WARN: cache {dest} unreadable "
                  f"({type(e).__name__}: {e}); re-fetching")
            try:
                os.remove(dest)
            except OSError:
                pass
            goes_acm.download(sel["url"], dest)
            got = goes_acm.read_cmi_2d(dest)
    else:
        goes_acm.download(sel["url"], dest)
        got = goes_acm.read_cmi_2d(dest)

    from scipy.ndimage import map_coordinates
    bt = got["bt2d"]
    dqf = got["dqf2d"]
    ny, nx = bt.shape
    xs, ys = got["x"], got["y"]
    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))
    lons_c = (bounds["lon_min"]
              + (np.arange(W) + 0.5) * (bounds["lon_max"] - bounds["lon_min"]) / W)
    lats_c = (bounds["lat_max"]
              - (np.arange(H) + 0.5) * (bounds["lat_max"] - bounds["lat_min"]) / H)
    t_inv = _goes_inverse_transformer(got["lon0"], got["h"])
    sat_h = got["h"]
    grid = np.full((H, W), np.nan)
    CH = 512
    for r0 in range(0, H, CH):
        r1 = min(H, r0 + CH)
        lons_t = np.broadcast_to(lons_c[None, :], (r1 - r0, W))
        lats_t = np.broadcast_to(lats_c[r0:r1, None], (r1 - r0, W))
        Xm, Ym = t_inv.transform(lons_t.ravel(), lats_t.ravel())
        cx = (np.asarray(Xm, dtype=float) / sat_h - xs[0]) / dx
        cy = (np.asarray(Ym, dtype=float) / sat_h - ys[0]) / dy
        with np.errstate(invalid="ignore"):
            inb = ((cx >= 0) & (cx <= nx - 1) & (cy >= 0) & (cy <= ny - 1))
            coords = np.stack([cy, cx])
            si = map_coordinates(bt, coords, order=1, mode="constant",
                                 cval=np.nan, prefilter=False)
            dq = map_coordinates(dqf, coords, order=0, mode="constant",
                                 cval=999.0)
        si = np.where(inb & (dq == 0) & np.isfinite(si), si, np.nan)
        grid[r0:r1] = si.reshape(r1 - r0, W)
    return grid


def _goes_bt_grid(sel, bounds, raw_dir):
    """Kept name (cache & call sites): inverse-georeference sampler.
    See _goes_sample_grid."""
    return _goes_sample_grid(sel, bounds, raw_dir)


def gibs_newest_valid_time(layer, bounds, primary_iso, cmap, raw_dir, tag,
                             max_back_min=180, step_min=10, min_frac=0.002):
    """Find the newest GIBS TIME with actual image content.

    The capabilities Default advertises the newest granule boundary, which
    sometimes has no rendered data behind it (fully transparent GetMap).
    Probe small (fast) renders stepping back until pixels appear; the
    chosen TIME becomes the honest scan time for provenance.
    """
    from PIL import Image
    memo = _probe_memo_get(layer, primary_iso)
    if memo is not None:
        if memo is False:
            raise RuntimeError(
                f"no valid GIBS render for {layer} "
                "(memo: no renderable data for this TIME)")
        return memo, _parse_iso(memo)
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    pw, ph = max(180, W // 10), max(90, H // 10)
    base = _parse_iso(primary_iso)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"probe_{tag}")[:80]
    tried = []
    t = base
    for _ in range(int(max_back_min / step_min) + 1):
        iso = t.strftime("%Y-%m-%dT%H:%M:00Z")
        dest = os.path.join(raw_dir, f"mosaic_{safe}_{pw}x{ph}.png")
        try:
            wms_getmap_png(GIBS_WMS, layer, bounds, pw, ph, iso, dest)
            im = Image.open(dest).convert("RGBA")
            a = np.array(im)
            if (a[:, :, 3] > 0).mean() >= min_frac:
                bt = invert_gibs_png(dest, cmap)
                if np.isfinite(bt).mean() >= min_frac:
                    _probe_memo_put(layer, primary_iso, iso)
                    return iso, _parse_iso(iso)
                tried.append(f"{iso}(unmapped-colors)")
            else:
                tried.append(f"{iso}(empty)")
        except Exception as e:
            tried.append(f"{iso}({type(e).__name__})")
        t -= dt.timedelta(minutes=step_min)
    # Negative memo: every canvas reproduces this walk otherwise.
    _probe_memo_put(layer, primary_iso, False)
    raise RuntimeError(f"no valid GIBS render for {layer} "
                       f"(tried: {'; '.join(tried[:6])})")


def _gibs_bt_grid(sel_or_layer, bounds, raw_dir, time_iso, cmap, tag):
    """WMS GetMap (time-cached) -> LUT-inverted BT grid.

    Resolves the newest TIME that actually carries image content (see
    gibs_newest_valid_time) and records it in the sidecar .time file, so
    provenance always reflects rendered data, never an empty granule.
    Returns (bt_grid, used_time_iso)."""
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    layer = sel_or_layer["layer"] if isinstance(sel_or_layer, dict) else sel_or_layer
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{tag}_{W}x{H}")[:120]
    dest = os.path.join(raw_dir, f"mosaic_{safe}.png")
    sidecar = dest + ".time"
    try:
        prev = open(sidecar).read().strip()
    except OSError:
        prev = ""
    if os.path.exists(dest) and prev == time_iso:
        # A cached render is ONLY reusable when its sidecar granule is the
        # granule still current for this request. The old check ("cached
        # file still inverts to real content") accepted ANY previous
        # granule, so long-lived processes served the first-fetched scan
        # forever: GIBS Times advance every 10 min but the file never
        # re-fetched (reproduced locally 2026-10-16..11-01; fixed there).
        try:
            bt = invert_gibs_png(dest, cmap)
            if np.isfinite(bt).mean() >= 0.002:
                return bt, prev
        except Exception:
            pass
    used_iso, _used_dt = gibs_newest_valid_time(layer, bounds, time_iso,
                                                cmap, raw_dir, tag)
    if os.path.exists(dest) and prev == used_iso:
        # The probe can re-resolve the same older granule it picked before;
        # a cache stamped with that exact Granule is still current.
        try:
            bt = invert_gibs_png(dest, cmap)
            if np.isfinite(bt).mean() >= 0.002:
                return bt, prev
        except Exception:
            pass
    wms_getmap_png(GIBS_WMS, layer, bounds, W, H, used_iso, dest)
    with open(sidecar, "w") as f:
        f.write(used_iso)
    return invert_gibs_png(dest, cmap), used_iso


def _fetch_dims(W, H, max_px):
    """Aspect-preserving WMS request size capped at `max_px` per side."""
    if max(W, H) <= max_px:
        return W, H
    s = float(max_px) / max(W, H)
    return (max(2, (int(W * s) // 2) * 2),
            max(2, (int(H * s) // 2) * 2))


def _upsample_to_canvas(grid, valid, shape):
    """Bilinear-resample a (small) fetched grid to the full canvas shape.

    WMS servers rasterize at fixed native scale: requesting a giant
    canvas mostly buys server-side upsampling (and risks HTTP failures
    past documented image-size caps, e.g. EUMETView). Fetch capped, then
    upscale the float field here (BILINEAR; mask NEAREST so validity
    edges stay crisp). Returns (grid, valid) at canvas resolution exactly"""
    from PIL import Image
    H, W = tuple(shape)
    gh, gw = grid.shape
    if (gh, gw) == (H, W):
        return grid, valid
    gi = Image.fromarray(grid.astype(np.float32), mode="F")
    gi = gi.resize((W, H), Image.BILINEAR)
    vi = Image.fromarray(valid.astype(np.uint8))
    vi = vi.resize((W, H), Image.NEAREST)
    out_g = np.asarray(gi, dtype=float)
    out_v = np.asarray(vi) > 0
    out_g[~out_v] = np.nan
    return out_g, out_v


# EUMETView (OWS) rejects or degrades very large GetMap requests; keep a
# margin under the ~4k server limit and upscale client-side instead.
EUMET_FETCH_MAX = 4096


def _eumet_gray_grid(sel, bounds, raw_dir):
    """WMS GetMap (time-cached) -> (grayscale 0..255, valid mask)."""
    from PIL import Image
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    rw, rh = _fetch_dims(W, H, EUMET_FETCH_MAX)
    safe = re.sub(r"[^A-Za-z0-9]+", "_",
                  f"{sel['layer']}_{rw}x{rh}")[:120]
    dest = os.path.join(raw_dir, f"mosaic_{safe}.png")
    prev = ""
    try:
        prev = open(dest + ".time").read().strip()
    except OSError:
        prev = ""
    if (not os.path.exists(dest)) or prev != sel["time"]:
        wms_getmap_png(EUMET_WMS, sel["layer"], bounds, rw, rh, sel["time"],
                       dest)
        with open(dest + ".time", "w") as f:
            f.write(sel["time"])
    im = Image.open(dest).convert("RGBA")
    a = np.array(im)
    opaque = a[:, :, 3] > 0
    gray = (a[:, :, 0].astype(float) + a[:, :, 1].astype(float)
            + a[:, :, 2].astype(float)) / 3.0
    gray[~opaque] = np.nan
    return _upsample_to_canvas(gray, opaque, (H, W))


def render_canvas(bounds, scans, raw_dir):
    """Render one canvas from the SAME scan set. Returns dict with bright,
    alpha, prov (uint8), observations, max_geo_skew_min, counts."""
    from na_common import blend_antimeridian, feather_merge
    W, H = bounds["canvas_width"], bounds["canvas_height"]

    layers = []   # [(bt_grid, valid)] GEO first (freshest first), VIIRS last
    geo_ages = []
    prov_layers = []
    observations = []
    now = dt.datetime.now(dt.timezone.utc)

    def _obs(name, sel, extra=""):
        scan = sel.get("scan_dt")
        if scan is not None and scan.tzinfo is None:
            scan = scan.replace(tzinfo=dt.timezone.utc)
        return {"source": name, "kind": sel["kind"],
                "scan_utc": scan.strftime("%Y-%m-%d %H:%M UTC")
                if scan else "unknown",
                "age_min": round((now - scan).total_seconds() / 60.0, 1)
                if scan else None,
                "detail": extra or sel.get("file") or sel.get("layer") or ""}

    # GEO direct-BT sources first (freshest first for the merge order).
    goes_present = []
    for name, _bucket, _sat in GOES_SATS:
        if name in scans:
            goes_present.append((scans[name]["scan_dt"], name))
    goes_present.sort(reverse=True)
    for _scan, name in goes_present:
        sel = scans[name]
        try:
            g = _goes_bt_grid(sel, bounds, raw_dir)
            valid = np.isfinite(g)
            layers.append((g, valid))
            prov_layers.append(PROV[name])
            geo_ages.append((now - sel["scan_dt"]).total_seconds() / 60.0)
            observations.append(_obs(name, sel))
        except Exception as e:
            observations.append(_obs(name, sel, f"FAILED: {type(e).__name__}: {e}"))

    # Himawari via GIBS LUT inversion.
    if "HIMAWARI-9" in scans:
        sel = scans["HIMAWARI-9"]
        try:
            g, used = _gibs_bt_grid(sel, bounds, raw_dir, sel["time"],
                                    sel["cmap"], "him")
            valid = np.isfinite(g)
            if not valid.any():
                raise RuntimeError("rendered grid has no valid pixels")
            used_dt = _parse_iso(used)
            sel_used = dict(sel, scan_dt=used_dt, time=used)
            layers.append((g, valid))
            prov_layers.append(PROV["HIMAWARI-9"])
            geo_ages.append((now - used_dt).total_seconds() / 60.0)
            observations.append(_obs("HIMAWARI-9", sel_used,
                                     f"GIBS {sel['layer']} LUT +/-1C"))
        except Exception as e:
            # An empty render means the disk does not cover this canvas
            # (nominal for regional canvases); anything else is a failure.
            if "no valid" in str(e):
                observations.append(_obs(
                    "HIMAWARI-9", sel,
                    "no coverage on this canvas (disk footprint)"))
            else:
                observations.append(_obs("HIMAWARI-9", sel,
                                         f"FAILED: {type(e).__name__}: {e}"))

    # Meteosat grayscale -> pseudo-BT (imagery-derived, documented).
    for name, _layer in EUMET_LAYERS:
        if name in scans:
            sel = scans[name]
            try:
                gray, opaque = _eumet_gray_grid(sel, bounds, raw_dir)
                g = gray_to_pseudo_bt(gray)
                g[~opaque] = np.nan
                valid = np.isfinite(g) & opaque
                layers.append((g, valid))
                prov_layers.append(PROV[name])
                geo_ages.append((now - sel["scan_dt"]).total_seconds() / 60.0)
                observations.append(_obs(name, sel, "EUMETView imagery"))
            except Exception as e:
                observations.append(_obs(
                    name, sel, f"FAILED: {type(e).__name__}: {e}"))

    # VIIRS daily supplement last (only fills where GEO has nothing).
    if "VIIRS" in scans:
        sel = scans["VIIRS"]
        try:
            grids = []
            for l in sel["layers"]:
                try:
                    g, _used = _gibs_bt_grid(
                        l, bounds, raw_dir, l["time"], sel["cmap"],
                        "viirs_" + re.sub(r"\W+", "", l["layer"])[:40])
                    if np.isfinite(g).any():
                        grids.append(g)
                except Exception:
                    continue
            if grids:
                if len(grids) == 1:
                    g = grids[0]
                else:
                    # Complementary combination: retain valid pixels from
                    # either Day or Night layer. Only average where both are
                    # valid; otherwise use whichever has data.
                    import warnings
                    stack = np.stack(grids)
                    with np.errstate(invalid="ignore"), \
                            warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        both_valid = np.isfinite(stack).all(axis=0)
                        g = np.where(both_valid,
                                     np.nanmean(stack, axis=0),
                                     np.nanmax(np.where(np.isfinite(stack), stack, np.nan), axis=0))
                        g = np.where(np.isfinite(g), g, np.nan)
                valid = np.isfinite(g)
                layers.append((g, valid))
                prov_layers.append(PROV["VIIRS"])
                observations.append(_obs("VIIRS", sel, "daily supplement"))
        except Exception as e:
            observations.append(_obs("VIIRS", sel,
                                     f"FAILED: {type(e).__name__}: {e}"))

    if not layers:
        raise RuntimeError(f"no IR source rendered ({observations})")

    max_skew = 0.0
    if len(geo_ages) >= 2:
        max_skew = max(geo_ages) - min(geo_ages)

    # Radiometric normalization: adjust each source to match the previous
    # source in the overlap region, accounting for different transfer functions.
    layers = normalize_sources(layers)

    # Gradual feather: reduce width as time skew increases, but never
    # eliminate blending entirely (unless skew is extreme).
    if max_skew <= MAX_SKEW_MIN:
        feather = FEATHER_PX
    else:
        feather = max(1, FEATHER_PX - int((max_skew - MAX_SKEW_MIN) // 5))

    merged, _valid = feather_merge(layers, feather_px=feather)
    merged = blend_antimeridian(merged)

    # Provenance: winner-takes-all in precedence order (pre-feather).
    prov = np.zeros(merged.shape, dtype=np.uint8)
    for (g, v), code in zip(layers, prov_layers):
        take = np.asarray(v, dtype=bool) & (prov == 0) & np.isfinite(g)
        prov[take] = code

    # Water-threshold IR transfer everywhere (land included). The old
    # high-latitude ALPHA_HI_LAND gate is gone from this path: it drew
    # razor-sharp latitude lines and blocky coast silhouettes in exactly
    # the zone where the floor transfer is ambiguous -- the polar band is
    # handled instead by the GDPS model blend below, which needs no
    # geographic hard edge and never needs the land mask asset.
    bright_full, alpha_full = bt_to_white(merged, land_mask=None)

    # Polar blend (|lat| 55..65 ramp): over cold surfaces single-channel
    # IR cannot tell clear ground from deck -- the old 240K land gate
    # "fixed" that by erasing everything north of ~60 (giant navy land
    # holes; user report: odd lines + weird-shaped coverage), and the
    # raw transfer renders sea ice as phantom cloud. GDPS
    # TotalCloudCover (a real overhead-cloud field, worldwide, hourly)
    # replaces the alpha there: genuine polar clouds still show (white)
    # and no hard geographic edge is introduced anywhere.
    pw_row = polar_weights(merged.shape, bounds)
    if pw_row.max() > 0.0:
        try:
            model_alpha, model_ok = _gdps_tcc_alpha(bounds, raw_dir)
        except Exception as e:
            print(f"[cloud] WARN: GDPS polar blend unavailable "
                  f"({type(e).__name__}: {e}); IR render stands alone")
            model_alpha = model_ok = None
        if model_alpha is not None:
            wgt = pw_row[:, None]
            ir_clean = np.nan_to_num(alpha_full, nan=0.0)
            blended = ((1.0 - wgt) * ir_clean + wgt * model_alpha)
            alpha_full = np.where(
                model_ok & (wgt > 0.0), blended,
                alpha_full)
            polar_owner = np.asarray(model_ok) & (wgt >= 0.5)
            prov[polar_owner] = PROV["GDPS-TCC"]
            observations.append(_obs(
                "GDPS-TCC",
                {"kind": "polar model blend", "file":
                 "GDPS_15km_TotalCloudCover"},
                extra=("replaces IR alpha across |lat| 55-65 (single-"
                       "channel IR vs cold-surface ambiguity)")))
            print(f"[cloud] polar blend: model ok px="
                  f"{int(np.asarray(model_ok).sum())}; "
                  f"prov owner px={int(polar_owner.sum())}")

    return {"bright": bright_full, "alpha": alpha_full, "prov": prov,
            "observations": observations, "max_geo_skew_min": round(max_skew, 1),
            "feather_px": feather,
            "n_geo": sum(1 for o in observations if "FAILED" not in o["detail"]
                         and o["source"] not in ("VIIRS", "GDPS-TCC"))}
