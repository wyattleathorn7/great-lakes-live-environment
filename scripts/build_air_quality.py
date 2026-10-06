"""Pipeline V5 — LIVE AIR QUALITY (independent, PM2.5-modeled).

Source: ECMWF CAMS European air-quality analysis served hourly by the
Open-Meteo Air Quality API (free, no key): PM2.5 near-surface
concentration sampled on a regular grid over the basin, bilinearly
resampled onto the common canvas. The underlying variable is PM2.5
(particulate nf2.5 micrometers, ug/m3) — identified as PM2.5 in the
legend and metadata while the displayed title stays "LIVE AIR QUALITY".
CAMS output is MODEL analysis output, labeled as such everywhere; it is
never presented as station observations and never confused with aerosol
optical depth. FIXED EPA-breakpoint-anchored 0-150 ug/m3 continuous
spectrum -> FULL BASIN RECTANGLE (lon -93..-73.5, lat 40.5..49.5, the
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
# EPA PM2.5 breakpoints anchor the continuous scale (good 0-12, moderate
# 12.1-35.4, USG 35.5-55.4, unhealthy 55.5-150.4, very unhealthy above).
STOPS = [
    (0.0, (60, 180, 80)),      # good: green
    (12.0, (245, 215, 50)),    # moderate: yellow
    (35.4, (245, 150, 30)),    # USG: orange
    (55.4, (215, 45, 35)),     # unhealthy: red
    (150.0, (150, 25, 110)),   # very unhealthy: purple
]
LABELS = [(0.0, "LOWEST 0"), (12.0, "12 moderate"), (35.4, "35.4 USG"),
          (55.4, "55.4 unhealthy"), (150.0, "HIGHEST+ 150")]
SCALE_HTML = ("PM2.5 concentration in micrograms per cubic meter (CAMS model "
              "analysis, fixed EPA-anchored scale): <b>LOWEST 0</b> green "
              "good &rarr; <b>12 moderate</b> yellow &rarr; <b>35.4 "
              "unhealthy for sensitive groups</b> orange &rarr; <b>55.4 "
              "unhealthy</b> red &rarr; <b>HIGHEST+ 150</b> purple very "
              "unhealthy (above clamps into purple). Same concentration "
              "always shows the same color. Modeled PM2.5 — not aerosol "
              "optical depth, not station observations.")

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
    subtitle = (f"PM2.5 (ug/m3, CAMS analysis)  |  {data_time_utc}")
    return finish(
        PRODUCT, CONFIG, KML_FILE, OVERLAY_NAME, field, STOPS, LABELS,
        "ug/m3", subtitle,
        f"Source: CAMS via Open-Meteo ({hour} UTC analysis)  |  "
        f"Processed {now_det_str()}", SCALE_HTML,
        [CONFIG["what"] + " The underlying variable is PM2.5, identified "
         "as modeled output (CAMS analysis), not station observations.",
         CONFIG["field"],
         "Checked hourly; republishes only when CAMS publishes a newer "
         "valid hour."],
        {"model_cycle": f"CAMS {hour} UTC",
         "stats": {"cams_hour": hour,
                   "data_nature": "modeled analysis, not observed"}},
        source_id, data_time_utc, "n/a (API)",
        "ug/m3 PM2.5 (display; source ug/m3 as filed)",
        f"CAMS analysis sampled on a {GRID_NLON}x{GRID_NLAT} basin grid, "
        "bilinearly resampled to the common canvas",
        "only [0,500] ug/m3 admitted; values above 150 clamp into purple; "
        "full basin rectangle, no shoreline cut; missing grid cells "
        "transparent; never zero-filled; modeled output, never labeled "
        "observed.")


if __name__ == "__main__":
    sys.exit(main())
