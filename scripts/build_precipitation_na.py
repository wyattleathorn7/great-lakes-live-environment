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


def na_wms_111(base, layers, bounds, w, h, extra=""):
    q = (f"SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layers}"
         f"&STYLES=&SRS=EPSG:4326&BBOX={bounds['lon_min']},{bounds['lat_min']},"
         f"{bounds['lon_max']},{bounds['lat_max']}&WIDTH={w}&HEIGHT={h}"
         f"&FORMAT=image/png&TRANSPARENT=TRUE{extra}")
    return f"{base}?{q}"


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


def eccc_rank_to_key(echo_strength):
    """Map 0..1 ECCC echo rank -> IEM KEY_ROWS color index (qualitative)."""
    idx = np.clip((echo_strength * len(KEY_ROWS)).astype(int), 0, len(KEY_ROWS) - 1)
    palette = np.array([c for c, _ in KEY_ROWS], dtype=np.uint8)
    return palette[idx]


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


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

    # IEM effective-domain mask (NOT the international border: the
    # network footprint from GetCapabilities + WSR-88D range).
    xs = bounds["lon_min"] + (np.arange(W) + 0.5) * (bounds["lon_max"] - bounds["lon_min"]) / W
    ys = bounds["lat_max"] - (np.arange(H) + 0.5) * (bounds["lat_max"] - bounds["lat_min"]) / H
    in_iem = ((xs[None, :] >= dom["lon_min"]) & (xs[None, :] <= dom["lon_max"])
              & (ys[:, None] >= dom["lat_min"]) & (ys[:, None] <= dom["lat_max"]))
    iem_wins = in_iem & iem_echo

    # 2) ECCC rain + snow + coverage GetMaps (authoritative footprint).
    # Coverage semantics (verified live 2026-10-11 by echo containment:
    # 10147/10174 rain echoes fall OUTSIDE the .INV-opaque area): the
    # RADAR_COVERAGE_*.INV style fills areas with NO radar coverage, so
    # true coverage = INV-transparent pixels. The default (non-INV)
    # style paints only outlines and must NOT be used as a mask.
    t_rain = eccc_time_default("RADAR_1KM_RRAI")
    t_snow = eccc_time_default("RADAR_1KM_RSNO")
    paths = {}
    for key, layer in (("rain", "RADAR_1KM_RRAI"), ("snow", "RADAR_1KM_RSNO"),
                       ("covr", "RADAR_COVERAGE_RRAI.INV"), ("covs", "RADAR_COVERAGE_RSNO.INV")):
        p = os.path.join(RAW_DIR, f"precip_na_eccc_{key}.png")
        wms_getmap(na_wms_111(ECCC, layer, bounds, W, H), p)
        paths[key] = load_rgba(p, (W, H))

    def luminance(a):
        return (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]) / 255.0

    # Authoritative coverage: INV-transparent == radar-covered.
    cov = (paths["covr"][:, :, 3] == 0) | (paths["covs"][:, :, 3] == 0)
    rain_e = paths["rain"][:, :, 3] > 0
    snow_e = paths["snow"][:, :, 3] > 0
    strength = np.maximum(luminance(paths["rain"]) * rain_e,
                          luminance(paths["snow"]) * snow_e)
    # ECCC is excluded EVERYWHERE inside the IEM effective domain — IEM
    # dry (no echo) is a valid observation there and must not expose
    # lower-priority ECCC echoes underneath (no duplicate rendering).
    eccc_echo = cov & (rain_e | snow_e) & ~in_iem

    # 3) Compose: IEM native pixels + ECCC rank-mapped pixels.
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[iem_wins] = iem[iem_wins]
    if eccc_echo.any():
        cols = eccc_rank_to_key(strength[eccc_echo])
        out[eccc_echo, :3] = cols
        out[eccc_echo, 3] = np.maximum(paths["rain"][eccc_echo, 3],
                                       paths["snow"][eccc_echo, 3])
    n_iem = int(iem_wins.sum())
    n_eccc = int(eccc_echo.sum())
    print(f"[{PRODUCT}] IEM px={n_iem} ECCC px={n_eccc} "
          f"({100.0*(n_iem+n_eccc)/(W*H):.2f}% canvas)")

    save_png(out, os.path.join(stage_prod, "current.png"))
    data_time = scan_dt.strftime("%Y-%m-%d %H:%M UTC")
    subtitle = f"NA radar mosaic (IEM N0Q + ECCC RRAI/RSNO) | {data_time}"
    lw, lh = draw_category_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"], subtitle,
        KEY_ROWS,
        f"IEM {scan_dt:%H:%MZ} + ECCC rain {t_rain} / snow {t_snow} | {now_det_str()}",
        note="No echo = transparent (no precipitation).")
    meta = base_metadata(
        PRODUCT, CONFIG_NA["title"], CONFIG_NA["freshness_label"],
        CONFIG_NA["source_name"], CONFIG_NA["source_url"], CONFIG_NA["variable"],
        data_time_utc=data_time, source_last_modified_utc=f"ECCC rain {t_rain}; snow {t_snow}",
        units="dBZ (IEM) + qualitative ECCC intensity",
        source_resolution="IEM ~1km CONUS mosaic 5-min; ECCC 1km NA composite 6-min; canvas 3300x1600 (0.05 deg/px)",
        color_min=5.0, color_max=75.0, color_units="dBZ (IEM domain); qualitative elsewhere",
        missing_data_treatment=("IEM wins inside lon -130..-65/lat 22..50; ECCC outside it where "
                                "RADAR_COVERAGE_*.INV is transparent (verified authoritative "
                                "coverage: echoes fall outside INV-opaque); transparent elsewhere; "
                                "never zero-filled; "
                                "ECCC colors are rank-mapped, not dBZ."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = SCALE_HTML
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["na_bounds"] = bounds
    meta["stats"] = {"iem_pixels": n_iem, "eccc_pixels": n_eccc,
                     "transparent_pixels": int(W * H - n_iem - n_eccc)}
    meta["provenance"] = {"iem_wms": na_wms_111(IEM, "nexrad-n0q-900913", bounds, W, H)[:200],
                          "eccc_wms": ECCC, "eccc_time_rain": t_rain, "eccc_time_snow": t_snow}
    token = meta["source_version"]
    folder_html = (f"<h2>{CONFIG_NA['title']}</h2><p>{SCALE_HTML}</p>"
                   f"<p><b>Sources:</b> {CONFIG_NA['source_name']}<br>"
                   f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br>"
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
