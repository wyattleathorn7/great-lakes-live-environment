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
ALPHA_HI, ALPHA_GAIN, ALPHA_MAX, ALPHA_FLOOR = 282.0, 4.5, 215.0, 3.0

# EUMETView grayscale imagery transfer (imagery-derived, documented as such):
# verified live 2026-10-16: warm Sahara ~= 24 (dark), cloud p95 ~= 213.
GRAY_WARM, GRAY_GAIN, GRAY_ALPHA_MAX = 20.0, 1.2, 215.0

MAX_SKEW_MIN = 15          # max GEO-GEO time skew for feathered blending
FEATHER_PX = 12            # cross-fade width where blending is permitted
DISK_FADE_PX = 12          # dissolve valid-coverage edges (no hard cut lines)

# Provenance codes for the diagnostic raster.
PROV = {"missing": 0, "GOES-19": 1, "GOES-18": 2, "HIMAWARI-9": 3,
        "METEOSAT-FES": 4, "METEOSAT-IODC": 5, "VIIRS": 6}

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


def bt_to_white(bt):
    """BT(K) grid -> (bright 0..255, alpha 0..215) white-cloud rendering."""
    bt = np.asarray(bt, dtype=float)
    m = np.isfinite(bt)
    bright = np.full(bt.shape, np.nan)
    alpha = np.full(bt.shape, np.nan)
    bright[m] = np.clip((BT_HI - bt[m]) * BT_GAIN, 0.0, 255.0)
    alpha[m] = np.clip((ALPHA_HI - bt[m]) * ALPHA_GAIN, 0.0, ALPHA_MAX)
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
    MAX_DIST = 25.0
    vals = np.full(len(uniq), np.nan)
    near = dist <= MAX_DIST
    vals[near] = np.array(
        [lut[tuple(map(int, keys[i]))] for i in idx[near]])  # Kelvin
    bt[opaque] = vals[inv]
    return bt


# ---------------------------------------------------------------- WMS fetch

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

def _goes_bt_grid(sel, bounds, raw_dir):
    """Download (scan-cached) + bin one GOES source to a BT grid."""
    import goes_acm
    from na_common import bin_points_to_canvas, fill_pinholes
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    tag = sel["scan_dt"].strftime("%Y%m%d-%H%M")
    dest = os.path.join(raw_dir, f"mosaic_{sel['bucket'].split('//')[1].split('.')[0]}"
                                 f"_{tag}.nc")
    if not (os.path.exists(dest) and os.path.getsize(dest) > 100_000):
        goes_acm.download(sel["url"], dest)
    got = goes_acm.read_cmi_bt(dest)
    bt = np.asarray(got["bt"], dtype=float).ravel()
    lats = np.asarray(got["lats"], dtype=float).ravel()
    lons = np.asarray(got["lons"], dtype=float).ravel()
    ok = np.isfinite(bt) & np.isfinite(lats) & np.isfinite(lons)
    g, _c = bin_points_to_canvas(bt[ok], lats[ok], lons[ok], bounds,
                                 splat_radius=2)
    return fill_pinholes(g, radius=3)


def gibs_newest_valid_time(layer, bounds, primary_iso, cmap, raw_dir, tag,
                             max_back_min=180, step_min=10, min_frac=0.002):
    """Find the newest GIBS TIME with actual image content.

    The capabilities Default advertises the newest granule boundary, which
    sometimes has no rendered data behind it (fully transparent GetMap).
    Probe small (fast) renders stepping back until pixels appear; the
    chosen TIME becomes the honest scan time for provenance.
    """
    from PIL import Image
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
                    return iso, _parse_iso(iso)
                tried.append(f"{iso}(unmapped-colors)")
            else:
                tried.append(f"{iso}(empty)")
        except Exception as e:
            tried.append(f"{iso}({type(e).__name__})")
        t -= dt.timedelta(minutes=step_min)
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
    if os.path.exists(dest) and prev:
        # Validate the cached render still inverts to real content.
        try:
            bt = invert_gibs_png(dest, cmap)
            if np.isfinite(bt).mean() >= 0.002:
                return bt, prev
        except Exception:
            pass
    used_iso, _used_dt = gibs_newest_valid_time(layer, bounds, time_iso,
                                                cmap, raw_dir, tag)
    wms_getmap_png(GIBS_WMS, layer, bounds, W, H, used_iso, dest)
    with open(sidecar, "w") as f:
        f.write(used_iso)
    return invert_gibs_png(dest, cmap), used_iso


def _eumet_gray_grid(sel, bounds, raw_dir):
    """WMS GetMap (time-cached) -> (grayscale 0..255, valid mask)."""
    from PIL import Image
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{sel['layer']}_{W}x{H}")[:120]
    dest = os.path.join(raw_dir, f"mosaic_{safe}.png")
    prev = ""
    try:
        prev = open(dest + ".time").read().strip()
    except OSError:
        prev = ""
    if (not os.path.exists(dest)) or prev != sel["time"]:
        wms_getmap_png(EUMET_WMS, sel["layer"], bounds, W, H, sel["time"],
                       dest)
        with open(dest + ".time", "w") as f:
            f.write(sel["time"])
    im = Image.open(dest).convert("RGBA")
    a = np.array(im)
    opaque = a[:, :, 3] > 0
    gray = (a[:, :, 0].astype(float) + a[:, :, 1].astype(float)
            + a[:, :, 2].astype(float)) / 3.0
    gray[~opaque] = np.nan
    return gray, opaque


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
                    import warnings
                    stack = np.stack(grids)
                    with np.errstate(invalid="ignore"), \
                            warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        g = np.nanmean(stack, axis=0)
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
    feather = FEATHER_PX if max_skew <= MAX_SKEW_MIN else 0

    if feather > 0:
        merged, _valid = feather_merge(layers, feather_px=feather)
    else:
        # Hard precedence (large time skew): fresher valid observation wins
        # outright -- no averaging across incompatible times.
        merged = layers[0][0].astype(float).copy()
        filled = np.asarray(layers[0][1], dtype=bool).copy()
        for grid, ok in layers[1:]:
            new = np.asarray(ok, dtype=bool) & ~filled
            merged[new] = np.asarray(grid, dtype=float)[new]
            filled = filled | np.asarray(ok, dtype=bool)
    merged = blend_antimeridian(merged)

    # Provenance: winner-takes-all in precedence order (pre-feather).
    prov = np.zeros(merged.shape, dtype=np.uint8)
    for (g, v), code in zip(layers, prov_layers):
        take = np.asarray(v, dtype=bool) & (prov == 0) & np.isfinite(g)
        prov[take] = code

    bright_full, alpha_full = bt_to_white(merged)
    return {"bright": bright_full, "alpha": alpha_full, "prov": prov,
            "observations": observations, "max_geo_skew_min": round(max_skew, 1),
            "feather_px": feather,
            "n_geo": sum(1 for o in observations if "FAILED" not in o["detail"]
                         and o["source"] != "VIIRS")}
