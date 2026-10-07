"""Pipeline V5 — LIVE AIR QUALITY (independent, PM2.5-modeled).

Source: ECMWF CAMS European air-quality analysis served hourly by the
Open-Meteo Air Quality API (free, no key): PM2.5 near-surface
concentration sampled on a regular grid over the basin, bilinearly
resampled onto the common canvas. The underlying variable is PM2.5
(particulate nf2.5 micrometers, ug/m3) — identified as PM2.5 in the
legend and metadata while the displayed title stays "LIVE AIR QUALITY".
CAMS output is MODEL analysis output, labeled as such everywhere; it is
never presented as station observations and never confused with aerosol
 optical depth. FIXED US EPA Air Quality Index 0-500 continuous
 spectrum (PM2.5 -> AQI via EPA breakpoints; muted professional palette)
 -> FULL BASIN RECTANGLE (lon -93..-73.5, lat 40.5..49.5, the
exact LIVE LEAF COLOR footprint) -> key + metadata -> Folder live KML +
stable entry KML. Hourly check; rebuild only when CAMS publishes a newer
valid hour. Exit 0/2/1 per contract.
"""

import json
import os
import sys
import traceback
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_field import (SKIP_NOTE, finish, refresh_kml, should_skip)
from geospatial_utils import (REPO_ROOT, SITE_DIR, USER_AGENT, load_bounds,
                              now_det_str, resample_gridded)

PRODUCT = "air_quality"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
KML_FILE = "Great_Lakes_Live_Air_Quality.kml"
OVERLAY_NAME = "\U0001F32B\uFE0F LIVE AIR QUALITY"

API = "https://air-quality-api.open-meteo.com/v1/air-quality"
# US EPA Air Quality Index (AirNow, 40 CFR Part 58) — the widely accepted
# professional standard for US public air-quality reporting. The source
# variable stays CAMS PM2.5 (ug/m3); it is converted to AQI via the EPA
# PM2.5 breakpoint table so the SAME index the public sees on AirNow is
# what the gradient paints. PRESET discrete bands (wind-style): each AQI
# category is one flat official EPA color with hard boundaries — no
# continuous slide for neighboring values to blur together.
AQI_BANDS = [  # (lo, hi, rep, name, official EPA rgb)
    (0.0, 50.0, 25.0, "Good", (0, 228, 0)),
    (51.0, 100.0, 75.0, "Moderate", (255, 255, 0)),
    (101.0, 150.0, 125.0, "Unhealthy for Sensitive Groups", (255, 126, 0)),
    (151.0, 200.0, 175.0, "Unhealthy", (255, 0, 0)),
    (201.0, 300.0, 250.0, "Very Unhealthy", (143, 63, 151)),
    (301.0, 500.0, 400.0, "Hazardous", (126, 0, 35)),
]
STOPS = [(rep, rgb) for _, _, rep, _, rgb in AQI_BANDS]


def aqi_band_rep(aqi):
    """Quantize an AQI value to its preset band representative value.

    Every pixel renders its band's exact official color (the field only
    ever holds the six rep values, so the LUT interpolator can never
    blend bands). Above 500 clamps Hazardous; NaN stays missing."""
    import math as _m
    try:
        v = float(aqi)
    except (TypeError, ValueError):
        return float("nan")
    if not _m.isfinite(v) or v < 0:
        return float("nan")
    for lo, hi, rep, _, _ in AQI_BANDS:
        if v <= hi:
            return rep
    return AQI_BANDS[-1][2]


AQ_BAND_ROWS = [  # (color, legend row label) — wind-style category key
    ((0, 228, 0), "0\u201350 Good"),
    ((255, 255, 0), "51\u2013100 Moderate"),
    ((255, 126, 0), "101\u2013150 Unhealthy for Sensitive Groups"),
    ((255, 0, 0), "151\u2013200 Unhealthy"),
    ((143, 63, 151), "201\u2013300 Very Unhealthy"),
    ((126, 0, 35), "301\u2013500 Hazardous"),
]
LABELS = [(25.0, "25 Good"), (75.0, "75 Moderate"),
          (125.0, "125 USG"), (175.0, "175 Unhealthy"),
          (250.0, "250 Very unhealthy"), (400.0, "400 Hazardous")]
KEY_TICKS = [(0.0, "0", "Good"), (50.0, "50", "Good"),
             (100.0, "100", "Moderate"), (150.0, "150", "USG"),
             (200.0, "200", "Unhealthy"), (300.0, "300", "Very unhealthy"),
             (500.0, "500", "Hazardous")]
SCALE_HTML = ("US EPA Air Quality Index (AQI, PM2.5-based): six preset "
              "official bands with hard boundaries — <b>0-50 Good</b> "
              "green &rarr; <b>51-100 Moderate</b> yellow &rarr; "
              "<b>101-150 Unhealthy for Sensitive Groups</b> orange "
              "&rarr; <b>151-200 Unhealthy</b> red &rarr; <b>201-300 Very "
              "Unhealthy</b> purple &rarr; <b>301-500 Hazardous</b> "
              "maroon. Every pixel shows its band's exact color; no "
              "blending between bands. CAMS PM2.5 model analysis "
              "converted via EPA PM2.5 breakpoints — not station "
              "observations.")
# EPA PM2.5 (ug/m3, 24-hr) -> AQI breakpoints: (c_lo, c_hi, aqi_lo, aqi_hi)
PM25_AQI_BP = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]


def pm25_to_aqi(c):
    """EPA piecewise-linear PM2.5 (ug/m3) -> AQI. Above 500.4 clamps 500."""
    import math as _m
    if c is None or (isinstance(c, float) and not _m.isfinite(c)):
        return float("nan")
    c = float(c)
    if c < 0:
        return float("nan")
    for c_lo, c_hi, a_lo, a_hi in PM25_AQI_BP:
        if c <= c_hi:
            return (a_lo + (c - c_lo) / (c_hi - c_lo) * (a_hi - a_lo))
    return 500.0

GRID_NLON, GRID_NLAT = 48, 30
BATCH = 100


def _get(url, timeout=60, retries=5):
    import time as _t
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 400:
                raise
            # 429/5xx: back off and retry (hourly job, never fatal)
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                OSError) as e:
            last = e
        if attempt < retries:
            _t.sleep(4 * attempt)
    raise last


def _probe_hour():
    """Cheapest staleness check: one point, current PM2.5 hour."""
    d = _get(API + "?latitude=43.5&longitude=-83.0&current=pm2_5&timezone=UTC")
    cur = d.get("current", {})
    return cur.get("time"), cur.get("pm2_5")


def _fetch_grid():
    """Sample the CAMS PM2.5 current-analysis hour on a regular grid.

    Uses the lightweight ``current=pm2_5`` endpoint (one value per point,
    tiny payloads) with pacing between batches to respect the free API
    rate limit. The returned ``current.time`` is the CAMS valid hour.
    """
    import time as _t
    bounds = load_bounds()
    xs = np.linspace(bounds["lon_min"], bounds["lon_max"], GRID_NLON)
    ys = np.linspace(bounds["lat_max"], bounds["lat_min"], GRID_NLAT)
    lo2d, la2d = np.meshgrid(xs, ys)
    flat_lat = la2d.ravel()
    flat_lon = lo2d.ravel()
    vals = np.full(flat_lat.shape, np.nan)
    hour = None
    for i in range(0, flat_lat.size, BATCH):
        sl = slice(i, i + BATCH)
        q = ("?latitude=" + ",".join(f"{v:.4f}" for v in flat_lat[sl])
             + "&longitude=" + ",".join(f"{v:.4f}" for v in flat_lon[sl])
             + "&current=pm2_5&timezone=UTC")
        d = _get(API + q)
        arr = d if isinstance(d, list) else [d]
        for k, item in enumerate(arr):
            cur = (item.get("current") or {})
            v = cur.get("pm2_5")
            if v is not None:
                vals[sl][k] = float(v)
                if hour is None and cur.get("time"):
                    hour = cur["time"]
        _t.sleep(1.5)  # stay under the free rate limit
    if hour is None:
        raise ValueError("no valid CAMS PM2.5 values returned")
    vals = np.where((vals >= CONFIG["valid_min"])
                    & (vals <= CONFIG["valid_max"]), vals, np.nan)
    if int(np.isfinite(vals).sum()) < 200:
        raise ValueError("too few valid CAMS grid points")
    return la2d, lo2d, vals.reshape(la2d.shape), hour


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    try:
        hour, _v = _probe_hour()
    except Exception as e:
        print(f"[{PRODUCT}] PROBE FAILED (keeping previous): {e}")
        return 2
    if not hour:
        print(f"[{PRODUCT}] PROBE FAILED: no current hour (keeping previous).")
        return 2
    source_id = f"cams-pm25-{hour}"
    if should_skip(PRODUCT, source_id):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        return refresh_kml(PRODUCT, KML_FILE, OVERLAY_NAME,
                           CONFIG["title"],
                           CONFIG["refresh_interval_seconds"])
    try:
        return _build(source_id)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(source_id):
    from datetime import datetime, timezone
    bounds = load_bounds()
    la2d, lo2d, grid, hour = _fetch_grid()
    field = resample_gridded(grid, la2d, lo2d, bounds,
                             (bounds["canvas_height"],
                              bounds["canvas_width"]))
    try:
        dt = datetime.fromisoformat(hour.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        from geospatial_utils import fmt_det
        data_time_utc = fmt_det(dt)
    except (TypeError, ValueError):
        data_time_utc = f"{hour} UTC"
    subtitle = (f"US EPA Air Quality Index (AQI, PM2.5-based)  |  {data_time_utc}")
    aqi_field = np.vectorize(pm25_to_aqi, otypes=[float])(field)
    band_field = np.vectorize(aqi_band_rep, otypes=[float])(aqi_field)

    def _aq_legend(legend_path):
        from geospatial_utils import draw_category_legend
        return draw_category_legend(
            legend_path, CONFIG["title"], subtitle, AQ_BAND_ROWS,
            "Source: CAMS via Open-Meteo, US EPA AQI bands (official colors)",
            note="Missing source data transparent; never zero-filled.")

    return finish(
        PRODUCT, CONFIG, KML_FILE, OVERLAY_NAME, band_field, STOPS, LABELS,
        "AQI", subtitle,
        f"Source: CAMS via Open-Meteo ({hour} UTC analysis)  |  "
        f"Processed {now_det_str()}", SCALE_HTML,
        [CONFIG["what"] + " The underlying variable is CAMS PM2.5, "
         "converted to the US EPA Air Quality Index via EPA PM2.5 "
         "breakpoints and identified as modeled output (CAMS analysis), "
         "not station observations.",
         CONFIG["field"],
         "Checked hourly; republishes only when CAMS publishes a newer "
         "valid hour."],
        {"model_cycle": f"CAMS {hour} UTC",
          "stats": {"cams_hour": hour,
                    "data_nature": "modeled analysis, not observed",
                    "index": "US EPA AQI (PM2.5-based, 0-500)",
                    "aqi_min": float(np.nanmin(aqi_field)),
                    "aqi_max": float(np.nanmax(aqi_field))}},
        source_id, data_time_utc, "n/a (API)",
        "AQI index 0-500 (display; source ug/m3 PM2.5 as filed)",
        f"CAMS PM2.5 analysis sampled on a {GRID_NLON}x{GRID_NLAT} basin grid, "
        "bilinearly resampled to the common canvas, converted PM2.5->AQI",
        "only [0,500.4] ug/m3 admitted; AQI above 500 clamps into maroon; "
        "pixels quantized to preset EPA bands (no inter-band blending); "
        "full basin rectangle, no shoreline cut; missing grid cells "
        "transparent; never zero-filled; modeled output, never labeled "
        "observed.",
        custom_legend=_aq_legend)


if __name__ == "__main__":
    sys.exit(main())
