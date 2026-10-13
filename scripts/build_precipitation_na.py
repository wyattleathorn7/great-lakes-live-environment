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
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", "precipitation.json")))
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


def _rate_to_dbz(r_mmh):
    """NWS NEXRAD Z-R equivalence (Z=300R^1.4): the documented bridge that
    puts ECCC precipitation rates on the N0Q dBZ key. ECCC paints equal
    colors for equal severity in rain and snow, so the rain-rate column
    owns both products' mapping (no snow-ratio invention)."""
    import math
    return 10.0 * math.log10(300.0 * max(r_mmh, 1e-6) ** 1.4)


# dBZ value of each ECCC class lower bound under the Z-R bridge.
ECCC_CLASS_DBZ = [_rate_to_dbz(r) for r in ECCC_RAIN_RATES]

# Original N0Q key bins (must match KEY_ROWS ranges in order).
KEY_BINS = [5.0, 15.0, 20.0, 30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]


def eccc_class_to_key(eccc_rgb):
    """Map ECCC-rendered RGB -> original N0Q KEY_ROWS color via nearest
    documented class color -> rate -> dBZ -> key bin. Returns (N,3)."""
    import numpy as np
    pal = np.array(ECCC_CLASS_RGB, dtype=float)
    px = np.asarray(eccc_rgb, dtype=float).reshape(-1, 3)
    d2 = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    cls = d2.argmin(axis=1)
    dbz = np.array(ECCC_CLASS_DBZ)[cls]
    keypal = np.array([c for c, _ in KEY_ROWS], dtype=np.uint8)
    out = np.zeros((px.shape[0], 3), dtype=np.uint8)
    for i, z in enumerate(dbz):
        b = 0
        for lo in KEY_BINS[1:]:
            if z >= lo:
                b += 1
            else:
                break
        out[i] = keypal[min(b, len(keypal) - 1)]
    return out


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
    through). Works day and night. Returns (bright, alpha, prov). Raises
    only if NO sat yields data (caller falls back to the % underlay)."""
    import goes_acm
    from na_common import (bin_points_to_canvas, fill_pinholes,
                           gate_field_to_coverage)
    try:
        with np.load(os.path.join(RAW_DIR, "na_cloud_ir.npz")) as z:
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
    bright = gate_field_to_coverage(bright, CONFIG_NA.get("coverage_mode", "extended200"))
    alpha = gate_field_to_coverage(alpha, CONFIG_NA.get("coverage_mode", "extended200"))
    try:
        np.savez_compressed(os.path.join(RAW_DIR, "na_cloud_ir.npz"),
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


def _build(scan_dt, source_id, bounds, W, H):
    from geospatial_utils import _legend_font, draw_category_legend
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    os.makedirs(stage_prod, exist_ok=True)
    dom = CONFIG_NA["iem_effective_domain"]

    # 1) IEM full-NA GetMap (CONUS echoes land inside its domain).
    iem_path = os.path.join(RAW_DIR, "precip_na_iem.png")
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
    paths = {}
    for key, layer, style, interp in (
            ("rain", "RADAR_1KM_RRAI", "Radar-Rain_14colors", True),
            ("snow", "RADAR_1KM_RSNO", "Radar-Snow_14colors", True),
            ("covr", "RADAR_COVERAGE_RRAI.INV", "", False),
            ("covs", "RADAR_COVERAGE_RSNO.INV", "", False)):
        p = os.path.join(RAW_DIR, f"precip_na_eccc_{key}.png")
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
    # Exact air-temp footprint + 100mi expansion: selected-country feathered
    # mask (ends at Mexico, Caribbean exclusions carved), NOT the global
    # coastline mask.
    out = np.zeros((H, W, 4), dtype=np.uint8)
    # Real satellite cloud underlay (NOAA Science-On-a-Sphere look):
    # GOES-19/18 Band-13 infrared texture first (day + night); the merged
    # GOES+model % field as flat white only if IR is unavailable; radar
    # echoes overwrite either below. Underlay missing -> precip-only raster
    # (never a failure).
    cloud_prov = "unavailable"
    try:
        ib, ia, cloud_prov = _ir_underlay(bounds, W, H)
        m = np.isfinite(ib) & np.isfinite(ia) & (ia > 0)
        out[m, 0:3] = np.round(ib[m])[:, None].astype(np.uint8)
        out[m, 3] = np.round(ia[m]).astype(np.uint8)
    except Exception as e:
        cloud_prov = f"IR unavailable:{type(e).__name__}; trying % field"
        try:
            with np.load(os.path.join(RAW_DIR, "na_cloud_field.npz")) as z:
                cf = np.asarray(z["field"], dtype=float)
                built = dt.datetime.fromisoformat(str(z["built_utc"]))
            age = abs((dt.datetime.now(dt.timezone.utc) - built).total_seconds())
            if cf.shape == (H, W) and age < 1800 and np.isfinite(cf).any():
                m = np.isfinite(cf) & (cf >= 1.0)
                out[m, 0:3] = 255
                out[m, 3] = np.clip(50.0 + cf[m] * 1.4, 0, 190).astype(np.uint8)
                cloud_prov = f"% na_cloud_field age={age:.0f}s n={int(m.sum())}"
            else:
                cloud_prov = f"% stale/incompatible (age={age:.0f}s)"
        except Exception as e2:
            cloud_prov = f"unavailable: {type(e2).__name__}"
    eccc_avail = cov & (rain_e | snow_e)
    ecol = np.zeros((H, W, 3), dtype=np.uint8)
    ealp = np.zeros((H, W), dtype=np.uint8)
    if eccc_avail.any():
        both = np.maximum(paths["rain"][eccc_avail].astype(float),
                          paths["snow"][eccc_avail].astype(float))
        cols = eccc_class_to_key(both[:, :3])
        ecol[eccc_avail] = cols
        ealp[eccc_avail] = np.maximum(paths["rain"][eccc_avail, 3],
                                      paths["snow"][eccc_avail, 3])
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
    out = apply_coverage_mask(out, CONFIG_NA.get("coverage_mode", "extended200"))
    n_iem = int(iem_fill.sum())
    n_eccc = int(eccc_echo.sum())
    print(f"[{PRODUCT}] IEM px={n_iem} ECCC px={n_eccc} "
          f"({100.0*(n_iem+n_eccc)/(W*H):.2f}% canvas)")

    save_png(out, os.path.join(stage_prod, "current.png"))
    data_time = scan_dt.strftime("%Y-%m-%d %H:%M UTC")
    subtitle = f"NA radar mosaic (IEM N0Q + ECCC RRAI/RSNO) + cloud underlay | {data_time}"
    lw, lh = draw_category_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"], subtitle,
        KEY_ROWS,
        f"IEM {scan_dt:%H:%MZ} + ECCC rain {t_rain} / snow {t_snow} | {now_det_str()}",
        note="No echo = transparent (no precipitation). Gray shading = real "
             "GOES-19/18 infrared satellite cloud texture (Band 13, day+night).")
    meta = base_metadata(
        PRODUCT, CONFIG_NA["title"], CONFIG_NA["freshness_label"],
        CONFIG_NA["source_name"], CONFIG_NA["source_url"], CONFIG_NA["variable"],
        data_time_utc=data_time, source_last_modified_utc=f"ECCC rain {t_rain}; snow {t_snow}",
        units="dBZ (both sources on the original N0Q key)",
        source_resolution="IEM ~1km CONUS mosaic 5-min (native rendering); ECCC 1km NA composite 6-min (rates -> dBZ via NWS Z=300R^1.4); canvas 3300x1600 (0.05 deg/px)",
        color_min=5.0, color_max=75.0, color_units="dBZ",
        missing_data_treatment=("ECCC wins where RADAR_COVERAGE_*.INV is transparent "
                                "(verified authoritative coverage); IEM N0Q fills ECCC gaps natively; "
                                "10-px cross-fade on the coverage edge; transparent elsewhere; "
                                "never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = SCALE_HTML
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["na_bounds"] = bounds
    meta["stats"] = {"iem_pixels": n_iem, "eccc_pixels": n_eccc,
                      "transparent_pixels": int(W * H - n_iem - n_eccc),
                      "cloud_underlay": cloud_prov}
    meta["provenance"] = {"iem_wms": na_wms_111(IEM, "nexrad-n0q-900913", bounds, W, H)[:200],
                          "eccc_wms": ECCC, "eccc_time_rain": t_rain, "eccc_time_snow": t_snow,
                          "cloud_underlay": cloud_prov}
    token = meta["source_version"]
    folder_html = (f"<h2>{CONFIG_NA['title']}</h2><p>{SCALE_HTML}</p>"
                   f"<p><b>Sources:</b> {CONFIG_NA['source_name']}<br>"
                   f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br>"
                   f"<b>Cloud underlay:</b> real GOES-19/18 Band-13 infrared "
                   f"satellite texture ({cloud_prov})<br>"
                   f"<b>Data time:</b> {data_time}</p>")
    meta["folder_html"] = folder_html
    write_metadata(stage_prod, meta)
    block = legend_block(f"{PRODUCT}/legend.png", token, SCALE_HTML)
    outs = live_out_dirs(stage, KML_FILE)
    kml_text = build_kml(PRODUCT, KML_FILE, OVERLAY_NAME, f"{PRODUCT}/current.png",
                         f"{PRODUCT}/legend.png",
                         description_html(CONFIG_NA["title"], meta, SKIP_NOTE, block),
                         CONFIG_NA["refresh_interval_seconds"], token,
                         folder=(CONFIG_NA["title"], folder_html),
                         out_dirs=outs["live"], meta=meta, bounds=bounds)
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
