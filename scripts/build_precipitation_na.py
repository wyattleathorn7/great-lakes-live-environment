"""Pipeline P1-NA — LIVE PRECIPITATION, NORTH AMERICA (independent).

Precedence (explicit, documented):
  1) IEM CONUS NEXRAD N0Q (WMS nexrad-n0q-900913) INSIDE the IEM effective
     domain (lon -130..-65, lat 22..50): native IEM palette pixels win.
  2) ECCC NA Radar Composite 1km rain (RADAR_1KM_RRAI, mm/h) + snow
     (RADAR_1KM_RSNO, cm/h) OUTSIDE the IEM domain, gated by the
     authoritative dynamic coverage layers (RADAR_COVERAGE_RRAI/RSNO) —
     never by alpha-zero (dry pixels are valid observations).
  3) Transparent elsewhere (no source coverage: northern Arctic, oceans
     beyond radar range, Greenland interior).

ECCC rates are mapped to the same 11-step key QUALITATIVELY (echo-rank
-> key color); equal colors across the transition do NOT claim identical
dBZ. Rain/snow products are combined as intensity only — no precip-type
classification is added (per task scope).

Exit codes: 0 updated; 2 source/validation failure (previous kept);
1 unexpected error.
"""

import datetime as dt
import json
import os
import re
import sys
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
from geospatial_utils import (REPO_ROOT, SITE_DIR, base_metadata, now_det_str,
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
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q.cgi"


def na_wms_111(base, layers, bounds, w, h, extra="", style=""):
    q = (f"SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layers}"
         f"&STYLES={style}&SRS=EPSG:4326&BBOX={bounds['lon_min']},{bounds['lat_min']},"
         f"{bounds['lon_max']},{bounds['lat_max']}&WIDTH={w}&HEIGHT={h}"
         f"&FORMAT=image/png&TRANSPARENT=TRUE{extra}")
    return f"{base}?{q}"


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


def _rate_to_dbz(r_mmh):
    """NWS NEXRAD Z-R equivalence (Z=300R^1.4): the documented bridge that
    puts ECCC precipitation rates on the N0Q dBZ key. ECCC paints equal
    colors for equal severity in rain and snow, so the rain-rate column
    owns both products' mapping (no snow-ratio invention)."""
    import math
    return 10.0 * math.log10(300.0 * max(r_mmh, 1e-6) ** 1.4)


# dBZ value of each ECCC class lower bound under the Z-R bridge.
ECCC_CLASS_DBZ = [_rate_to_dbz(r) for r in ECCC_RAIN_RATES]

# Snow-only key (ECCC RSNO snow echoes): blue ramp, same severity order
# as the N0Q key so equal rank = equal severity across types.
SNOW_ROWS = [
    ("#E0F7FF", "Snow Very light — ice haze"),
    ("#A5E8FF", "Snow Light — pale cyan"),
    ("#67CEFF", "Snow Light–moderate — sky blue"),
    ("#329BFF", "Snow Moderate — azure"),
    ("#2463EB", "Snow Moderate–heavy — royal blue"),
    ("#5145CD", "Snow Heavy — deep indigo"),
    ("#873FDB", "Snow Very heavy — violet"),
    ("#D946EF", "Snow Extreme — magenta"),
]

# GDPS model fill renders translucent so model guidance never reads as
# observed radar (vivid = radar echo, pale = model fill where blind).
GDPS_FILL_ALPHA = 120

# Original N0Q key bins (must match KEY_ROWS ranges in order).
KEY_BINS = [5.0, 15.0, 20.0, 30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]


# Shared snow sentence for legend HTML (variant + primary metadata).
SNOW_SENT = (" ECCC snow echoes render blue ONLY where SfcPrecipType classifies snow "
    "(mixed/freezing rain render on the dBZ key at matched severity).")


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


SNOW_PAL = np.array([_hex_to_rgb(c) for c, _ in SNOW_ROWS], dtype=np.uint8)


def _bin_for_dbz(z):
    b = 0
    for lo in KEY_BINS[1:]:
        if z >= lo:
            b += 1
        else:
            break
    return min(b, len(KEY_ROWS) - 1)


def _snow_row_for_bin(b):
    return min(len(SNOW_PAL) - 1, b * len(SNOW_PAL) // len(KEY_ROWS))


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


def key_from_iem_wash(rgb):
    """Map IEM overview-scale pixels into N0Q key rows without inventing
    intensity. Vivid pixels (real chroma: storm cores survive overview
    minification) match by chromaticity to the exact key row. Washed-out
    pixels (overview blur destroyed their color) fall back to a
    luminance rank clamped to rows 0..3 (very light..moderate): a washed
    pixel can never paint as severe/intense/extreme."""
    import numpy as np
    keypal = np.array([c for c, _ in KEY_ROWS], dtype=float)
    knorm = keypal / np.maximum(keypal.max(axis=1, keepdims=True), 1.0)
    px = np.asarray(rgb, dtype=float).reshape(-1, 3)
    mx = px.max(axis=1, keepdims=True)
    pn = px / np.maximum(mx, 1.0)
    chroma = px.max(axis=1) - px.min(axis=1)
    d2 = ((pn[:, None, :] - knorm[None, :, :]) ** 2).sum(axis=2)
    cls = d2.argmin(axis=1)
    lum = (0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2]) / 255.0
    soft = np.clip((lum * 3.99).astype(int), 0, 3)
    cls = np.where(chroma > 60.0, cls, soft)
    return np.array([c for c, _ in KEY_ROWS], dtype=np.uint8)[cls].reshape(
        np.asarray(rgb).shape)


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
    source_id = f"na-precip-{scan_dt.strftime('%Y%m%d-%H%M')}"
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
    same IEM + ECCC WMS code, IR underlay, and one-legend compose as the
    original basin layer. Returns
    (base_rgba, n_iem, n_eccc, cloud_prov, t_rain, t_snow, t_type)."""
    # 1) IEM GetMap (CONUS echoes land inside its domain).
    iem_path = os.path.join(RAW_DIR, f"precip_{tag}_iem.png")
    wms_getmap(na_wms_111(IEM, "nexrad-n0q-900913", bounds, W, H), iem_path)
    iem = load_rgba(iem_path, (W, H))
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
    # coverage is valid, including most of CONUS; IEM N0Q (native
    # rendering, itself the original legend) fills ECCC's gaps. IEM dry
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
        # Authoritative type gate (ECCC SfcPrecipType): snow blues render
        # ONLY where the type product classifies snow. Mixed/freezing
        # rain/hail render on the N0Q key at matched severity; pixels
        # the type product leaves unclassified fall back to the severity
        # heuristic below. Never assume an echo is rain.
        tpal = np.array(SFC_TYPE_RGB, dtype=float)
        tpx = paths["ptype"][eccc_avail].astype(float)[:, :3]
        topq = paths["ptype"][eccc_avail, 3] > 0
        _d2 = ((tpx[:, None, :] - tpal[None, :, :]) ** 2).sum(axis=2)
        tcls = _d2.argmin(axis=1)
        tcls[(_d2.min(axis=1) > SFC_TYPE_MAX_DIST ** 2) | ~topq] = -1
        snow_win = np.isin(tcls, SFC_SNOW)
        # Fallback for type-unclassified echoes: snow product showing
        # equal-or-higher severity renders blue (previous behavior).
        snow_win = snow_win | ((tcls < 0) & (paths["snow"][eccc_avail, 3] > 0)
                               & (cls_s >= cls_r))
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
        dbz_s = np.array(ECCC_CLASS_DBZ)[sev[snow_win]]
        snow_cols = np.zeros((int(snow_win.sum()), 3), dtype=np.uint8)
        for i, z in enumerate(dbz_s):
            snow_cols[i] = SNOW_PAL[_snow_row_for_bin(_bin_for_dbz(z))]
        cols[snow_win] = snow_cols
        ecol[eccc_avail] = cols
        ealp[eccc_avail] = np.maximum(paths["rain"][eccc_avail, 3],
                                      paths["snow"][eccc_avail, 3])
        n_snow = int(snow_win.sum())
    if eccc_echo.any():
        out[eccc_echo, :3] = ecol[eccc_echo]
        out[eccc_echo, 3] = ealp[eccc_echo]
    # IEM gap fill repainted into the same key rows (vivid cores match by
    # chroma; washed pixels clamp to very-light..moderate, never intense).
    iem_fill = iem_echo & ~eccc_echo
    out[iem_fill] = iem[iem_fill]
    out[iem_fill, :3] = key_from_iem_wash(iem[iem_fill][:, :3])
    d_edge = distance_transform_edt(~cov)
    band = cov & (d_edge <= 10) & eccc_echo & iem_echo
    if band.any():
        w = (d_edge[band] / 10.0)[:, None]
        out[band, :3] = np.round(w * out[band, :3].astype(float)
                                 + (1.0 - w) * iem[band, :3].astype(float)).astype(np.uint8)
        out[band, 3] = np.round(w[:, 0] * out[band, 3].astype(float)
                                + (1.0 - w[:, 0]) * iem[band, 3].astype(float)).astype(np.uint8)
    base = out  # unmasked composite; each coverage masks its own copy
    n_iem = int(iem_fill.sum())
    n_eccc = int(eccc_echo.sum())
    print(f"[{PRODUCT}] IEM px={n_iem} ECCC px={n_eccc} snow px={n_snow} "
          f"({100.0*(n_iem+n_eccc)/(W*H):.2f}% canvas)")
    return out, n_iem, n_eccc, cloud_prov, t_rain, t_snow, t_type


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
    base, n_iem, n_eccc, cloud_prov, t_rain, t_snow, t_type = _compose_radar(
        scan_dt, bounds, W, H, "na")
    # Worldwide composite: the SAME WMS radar code at the global bbox
    # (echoes land where radar exists, transparent elsewhere) + the
    # worldwide IR underlay, then GDPS model fill where radar is blind.
    gbase, gn_iem, gn_eccc, gcloud_prov, _, _, _ = _compose_radar(
        scan_dt, gbounds, GW, GH, "global")
    # No GDPS model fill: radar-only composite (restorable via
    # _gdps_precip_fill). Transparent where radar is blind.
    gcloud_prov = f"{gcloud_prov}; GDPS fill off (radar-only)"

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
    # continental + global use the worldwide composite with GDPS model
    # fill (continental additionally gated to worldwide land).
    VARIANT_BASE = {"basin": base, "landonly": gbase, "global": gbase,
                    "extended200": gbase}
    VARIANT_DIMS = {"basin": (W, H), "landonly": (GW, GH), "global": (GW, GH),
                    "extended200": (GW, GH)}
    VARIANT_PROV = {"basin": cloud_prov, "landonly": gcloud_prov,
                    "global": gcloud_prov, "extended200": gcloud_prov}
    data_time = scan_dt.strftime("%Y-%m-%d %H:%M UTC")
    subtitle = f"NA radar mosaic (IEM N0Q + ECCC RRAI/RSNO/type) | {data_time}"
    token = source_token(source_id)

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
            units="dBZ (both sources on the original N0Q key)",
            source_resolution="IEM ~1km CONUS mosaic 5-min (native rendering); ECCC 1km rain/snow/type 6-min (rates -> dBZ via NWS Z=300R^1.4; snow blues at matched rank)",
            color_min=5.0, color_max=75.0, color_units="dBZ",
            missing_data_treatment=("ECCC type-gated (SfcPrecipType snow -> blue ramp; other types -> dBZ key) where RADAR_COVERAGE_*.INV is transparent; IEM N0Q fills ECCC gaps natively (type undifferentiated); "
                                    "10-px cross-fade on the coverage edge; radar-only composite (no cloud underlay, no model fill); transparent elsewhere; "
                                    "never zero-filled."))
        vmeta["legend_scale_html"] = (SCALE_HTML + SNOW_SENT)
        vmeta["source_id"] = source_id
        vmeta["source_version"] = token
        vmeta["na_bounds"] = VARIANT_BOUNDS[mode]
        vmeta["coverage_mode"] = mode
        vmeta["stats"] = {"opaque_pixels": n_opaque,
                          "transparent_pixels": int(vw * vh - n_opaque),
                          "underlay": vprov}
        vmeta["provenance"] = {"iem_wms": na_wms_111(IEM, "nexrad-n0q-900913", bounds, W, H)[:200],
                               "eccc_wms": ECCC, "eccc_time_rain": t_rain,
                               "eccc_time_snow": t_snow, "eccc_time_type": t_type,
                               "underlay": vprov}
        vmeta["folder_html"] = (
            f"<h2>{vtitle}</h2>"
            f"<p><b>Sources:</b> {CONFIG_NA['source_name']}<br>"
            f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br>"
            f"<b>Coverage:</b> {COVERAGE_LABELS[mode]}<br>"
            f"<b>Radar note:</b> radar echoes only; no cloud underlay, no model fill<br>"
            f"<b>Data time:</b> {data_time}</p>") + coverage_selector_html(mode)
        write_metadata(vdir, vmeta)

    def_base = VARIANT_BASE[default_cover]
    def_prov = VARIANT_PROV[default_cover]
    def_w, def_h = VARIANT_DIMS[default_cover]
    out = apply_coverage_mask(def_base, default_cover)
    save_png(out, os.path.join(stage_prod, "current.png"))
    lw, lh = draw_category_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"], subtitle,
        KEY_ROWS + SNOW_ROWS,
        f"IEM {scan_dt:%H:%MZ} + ECCC rain {t_rain} / snow {t_snow} / type {t_type} | {now_det_str()}",
        note="No echo = transparent. Blues = ECCC classified snow.")
    meta = base_metadata(
        PRODUCT, CONFIG_NA["title"], CONFIG_NA["freshness_label"],
        CONFIG_NA["source_name"], CONFIG_NA["source_url"], CONFIG_NA["variable"],
        data_time_utc=data_time, source_last_modified_utc=f"ECCC rain {t_rain}; snow {t_snow}; type {t_type}",
        units="dBZ (both sources on the original N0Q key)",
        source_resolution="IEM ~1km CONUS mosaic 5-min (native rendering); ECCC 1km NA composite 6-min (rates -> dBZ via NWS Z=300R^1.4); canvas 3300x1600 (0.05 deg/px)",
        color_min=5.0, color_max=75.0, color_units="dBZ",
        missing_data_treatment=("ECCC wins where RADAR_COVERAGE_*.INV is transparent "
                                "(verified authoritative coverage); IEM N0Q fills ECCC gaps natively; "
                                "10-px cross-fade on the coverage edge; transparent elsewhere; "
                                "never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = (SCALE_HTML + SNOW_SENT)
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["na_bounds"] = VARIANT_BOUNDS[default_cover]
    meta["stats"] = {"iem_pixels": n_iem, "eccc_pixels": n_eccc,
                      "transparent_pixels": int(def_w * def_h - n_iem - n_eccc),
                      "underlay": def_prov}
    meta["provenance"] = {"iem_wms": na_wms_111(IEM, "nexrad-n0q-900913", bounds, W, H)[:200],
                          "eccc_wms": ECCC, "eccc_time_rain": t_rain, "eccc_time_snow": t_snow, "eccc_time_type": t_type,
                          "underlay": def_prov}
    token = meta["source_version"]
    folder_html = (f"<h2>{CONFIG_NA['title']}</h2>"
                   f"<p><b>Sources:</b> {CONFIG_NA['source_name']}<br>"
                   f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br>"
                   f"<b>Coverage:</b> {COVERAGE_LABELS[default_cover]}<br>"
                   f"<b>Radar note:</b> radar echoes only; no cloud underlay, no model fill<br>"
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
