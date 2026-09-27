"""Pipeline P1 — LIVE PRECIPITATION (independent).

Source: Iowa Environmental Mesonet (Iowa State University) CONUS NEXRAD
N0Q Base Reflectivity mosaic, built from the NOAA NWS WSR-88D Level-III
network and regenerated every 5 minutes. Professional + credible:
Iowa State archives the operational NWS feed and serves it as an OGC WMS.

Method: WMS GetMap for the exact common canvas
(lon -93..-73.5, lat 40.5..49.5, 1800x1175, EPSG:4326, transparent PNG).
The WMS renders the N0Q palette itself (cyan/blue = Very Light,
green = Light/Moderate, yellow/orange = Heavy, red = Severe,
magenta/purple = Intense, white = Extreme) so the live gradient in
Google Earth IS the source's real-time gradient — never a recolor.
No-precipitation pixels arrive fully transparent (alpha 0) and are
kept transparent.

All seasons / all types: base reflectivity is echo strength from any
hydrometeor — spring/summer/fall rain, winter snow + lake-effect snow,
thunderstorms + hail whenever convection fires. Reflectivity does not
classify type; it shows intensity, which is exactly what the Light /
Moderate / Heavy / Severe / Intense / Extreme key explains.

Raster-only GroundOverlay (never a video: Google Earth has no live
video primitive — the overlay refreshes). Google Earth side refreshes
at 60 s (fastest sustainable cadence): each poll re-fetches the live
KML, and the overlay Icon re-requests its PNG, so a newly published
scan appears within about a minute. Source cadence is 5 minutes
(IEM rebuilds the mosaic every 5 min); the 60 s poll guarantees no
extra staleness is added client-side.

Exit codes: 0 updated (or skipped); 2 source/validation failure
(previous kept); 1 unexpected error.
"""

import datetime as dt
import json
import os
import sys
import traceback
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (REPO_ROOT, SITE_DIR, USER_AGENT, base_metadata,
                              load_bounds, now_det_str, promote_stage,
                              read_state, save_png, source_token, stage_dir,
                              write_metadata, write_state)
from geospatial_utils import _legend_font, draw_category_legend  # noqa: F401 (shared helpers)

PRODUCT = "precipitation"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Precipitation.kml"
OVERLAY_NAME = "\U0001f327\ufe0f LIVE PRECIPITATION"
SKIP_NOTE = "Turn on/off independently of all other layers."

# Canonical N0Q key colors (match the reference DBZ key image AND the
# IEM WMS rendering): cyan/blue Very Light, green Light/Moderate,
# yellow/orange Heavy, red Severe, magenta/purple Intense, white Extreme.
KEY_ROWS = [
    ((0, 236, 236), "Very Light ~5-15 dBZ — drizzle, very light rain/snow"),
    ((0, 144, 240), "Very Light ~15-20 dBZ — light returns, mist"),
    ((0, 240, 0), "Light ~20-30 dBZ — light rain / snow"),
    ((0, 128, 0), "Moderate ~30-40 dBZ — moderate rain / snow"),
    ((255, 255, 0), "Heavy ~40-45 dBZ — heavy rain"),
    ((255, 165, 0), "Heavy ~45-50 dBZ — very heavy rain"),
    ((255, 0, 0), "Severe ~50-60 dBZ — severe storms, hail possible"),
    ((192, 0, 0), "Severe ~55-60 dBZ — intense downpours, hail likely"),
    ((255, 0, 255), "Intense ~60-70 dBZ — violent storms, large hail"),
    ((139, 0, 255), "Intense ~65-70 dBZ — damaging hail, flash flooding"),
    ((255, 255, 255), "Extreme 70+ dBZ — destructive hail / tornadic debris"),
]

SCALE_HTML = (
    "Precipitation intensity (NEXRAD N0Q base reflectivity, dBZ, OBSERVED radar echo): "
    "<b>LOWEST 5 Very Light</b> cyan/blue (drizzle, mist) &rarr; "
    "<b>Light</b> greens (~20-30: light rain/snow) &rarr; "
    "<b>Moderate</b> dark green (~30-40: steady rain/snow) &rarr; "
    "<b>Heavy</b> yellow/orange (~40-50: heavy rain) &rarr; "
    "<b>Severe</b> red (~50-60: severe storms, hail possible) &rarr; "
    "<b>Intense</b> magenta/purple (~60-70: violent storms, large hail) &rarr; "
    "<b>HIGHEST+ 70+ Extreme</b> white (destructive hail, debris). "
    "Same dBZ always shows the same color in every season — rain, snow, "
    "lake-effect snow, thunderstorms. No echo (no precipitation) stays "
    "see-through so the map underneath shows.")


def wms_url():
    b = load_bounds()
    return (
        f"{CONFIG['wms_url']}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
        f"&LAYERS={CONFIG['wms_layer']}&STYLES=&SRS=EPSG:4326"
        f"&BBOX={b['lon_min']},{b['lat_min']},{b['lon_max']},{b['lat_max']}"
        f"&WIDTH={b['canvas_width']}&HEIGHT={b['canvas_height']}"
        f"&FORMAT=image/png&TRANSPARENT=TRUE")


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    # Source id: current UTC 5-minute bin (IEM rebuilds the mosaic every
    # 5 min). Cheap pre-check with no bulk state: the build itself IS the
    # probe — a failed download keeps the previous raster (exit 2).
    now = dt.datetime.now(dt.timezone.utc)
    minute_bin = (now.minute // 5) * 5
    scan_dt = now.replace(minute=minute_bin, second=0, microsecond=0)
    source_id = f"iem-n0q-{scan_dt.strftime('%Y%m%d-%H%M')}"
    prev = read_state(PRODUCT)
    # NOTE: no byte-stability skip here beyond the same-bin fast path:
    # precipitation moves fast, and the workflow runs on the 12-minute
    # cadence, so consecutive runs in one 5-min bin reuse the raster.
    if (prev.get("source_id") == source_id
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png"))
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE))):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        return _refresh_kml()
    try:
        return _build(scan_dt, source_id)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _refresh_kml():
    try:
        refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                             CONFIG["title"], SKIP_NOTE,
                             CONFIG["refresh_interval_seconds"])
    except Exception as e:
        print(f"[{PRODUCT}] WARNING: KML refresh failed: {e}")
    return 0


def _build(scan_dt, source_id):
    bounds = load_bounds()
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    os.makedirs(stage_prod, exist_ok=True)
    W, H = bounds["canvas_width"], bounds["canvas_height"]

    url = wms_url()
    raw_path = os.path.join(RAW_DIR, "precip_n0q_current.png")
    os.makedirs(RAW_DIR, exist_ok=True)
    req = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) < 20_000:
        raise ValueError(f"WMS download too small ({len(data)} bytes)")
    with open(raw_path, "wb") as f:
        f.write(data)

    im = Image.open(raw_path).convert("RGBA")
    if im.size != (W, H):
        # WMS was asked for the exact canvas; resampling here would blur
        # radar edges, so a size mismatch is a hard failure (keep previous).
        raise ValueError(f"WMS image size {im.size} != canvas {(W, H)}")
    rgba = np.array(im)
    # No-precipitation transparency: the WMS returns alpha 0 where there
    # is no echo. Enforce it (fully transparent stays transparent) and
    # never invent color where the source has none.
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    print(f"[{PRODUCT}] IEM N0Q {scan_dt:%Y-%m-%d %H:%MZ}: opaque={n_opaque} "
          f"({100.0 * n_opaque / (W * H):.1f}% of canvas)")
    # Clear-sky over the basin is VALID (transparent raster is honest, not
    # a failure) — same semantics as cloud_cover on a clear day.

    save_png(rgba, os.path.join(stage_prod, "current.png"))

    data_time_utc = scan_dt.strftime("%Y-%m-%d %H:%M UTC")
    subtitle = (f"NEXRAD N0Q base reflectivity (dBZ, 5-min mosaic)  |  "
                f"{data_time_utc}")
    lw, lh = draw_category_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        KEY_ROWS,
        f"Source: IEM N0Q {scan_dt:%Y-%m-%d %H:%MZ} (NOAA WSR-88D)  |  "
        f"Processed {now_det_str()}",
        note="No echo = transparent (no precipitation).")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (WMS GetMap mosaic)",
        units="dBZ (display = observed dBZ)",
        source_resolution="~1 km NEXRAD N0Q mosaic, 5-minute rebuilds; "
                          "WMS GetMap sampled at the 1800x1175 canvas",
        color_min=5.0, color_max=75.0, color_units="dBZ",
        missing_data_treatment=("only WMS echo pixels paint; no-echo and "
                                "missing render fully transparent (alpha 0); "
                                "full basin rectangle, no shoreline cut; "
                                "never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = SCALE_HTML
    meta["model_cycle"] = f"IEM N0Q mosaic {scan_dt:%Y-%m-%d %H:%MZ}"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["wms_request"] = url
    meta["stats"] = {"opaque_pixels": n_opaque,
                     "transparent_pixels": int(W * H - n_opaque)}
    token = meta["source_version"]
    # Folder: key image FIRST (top), written gradient explanation
    # UNDERNEATH — exactly the requested layout.
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p><img src=\"{legend_src(token)}\" width=\"600\" "
        f"alt=\"precipitation gradient key\"></p>"
        f"<p>{SCALE_HTML}</p>"
        f"<p>How to read it in any season. <b>Light</b> greens are light "
        f"rain in summer — and light snow in winter. <b>Moderate</b> dark "
        f"green is steady rain or steady snow, including lake-effect bands. "
        f"<b>Heavy</b> yellow/orange is heavy rain (or very heavy snow "
        f"rates). <b>Severe</b> red means severe thunderstorms with hail "
        f"possible; <b>Intense</b> magenta/purple means violent storms with "
        f"large hail and flash flooding; <b>Extreme</b> white is reserved "
        f"for the most destructive echoes. Radar sees precipitation "
        f"particles, not air temperature, so one scale covers rain, snow, "
        f"and thunderstorms all year. Where you see through the overlay, "
        f"there is no precipitation right now.</p>"
        f"<p><b>Variable:</b> {CONFIG['variable']}<br/>"
        f"<b>Units:</b> dBZ<br/>"
        f"<b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> mosaic rebuilt every 5 minutes; Google Earth "
        f"re-polls every 60 seconds<br/>"
        f"<b>Data time:</b> {data_time_utc}<br/>"
        f"<b>Source version:</b> {source_id}<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Provenance:</b> <a href=\"{CONFIG['source_url']}\">IEM / NOAA "
        f"NEXRAD</a></p>")
    meta["folder_html"] = folder_html
    write_metadata(stage_prod, meta)
    block = legend_block(f"{PRODUCT}/legend.png", token, SCALE_HTML)
    outs = live_out_dirs(stage, KML_FILE)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        folder=(CONFIG["title"], folder_html),
        out_dirs=outs["live"],
        meta=meta)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "source_id": source_id,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def legend_src(token=None):
    from build_kml import pages_base
    base = pages_base()
    v = f"?v={token}" if token else ""
    return f"{base}/{PRODUCT}/legend.png{v}"


if __name__ == "__main__":
    sys.exit(main())
