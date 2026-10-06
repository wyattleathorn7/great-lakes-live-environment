"""Pipeline V2 — LIVE RELATIVE HUMIDITY (independent).

NOAA/NCEP HRRR 3 km 2 m relative humidity (RH) analysis, hourly cycles,
percent -> FIXED absolute 0-100 % continuous spectrum -> FULL BASIN
RECTANGLE (lon -93..-73.5, lat 40.5..49.5, the exact LIVE LEAF COLOR
footprint) -> key image + metadata -> Folder live KML + stable entry KML.
Near-surface moisture/saturation field; never PWAT or dew point.
Exit 0 updated/skipped; 2 source failure (previous kept); 1 unexpected.
"""

import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_field import (SKIP_NOTE, finish, hrrr_field, refresh_kml,
                        should_skip)
from geospatial_utils import (REPO_ROOT, SITE_DIR, load_bounds, now_det_str)

PRODUCT = "humidity"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Relative_Humidity.kml"
OVERLAY_NAME = "♨️ LIVE RELATIVE HUMIDITY"

STOPS = [
    (0.0, (150, 110, 40)),    # arid: amber
    (20.0, (240, 200, 60)),   # dry: yellow
    (40.0, (120, 200, 90)),   # comfortable: green
    (60.0, (40, 190, 150)),   # humid: teal
    (80.0, (30, 130, 220)),   # very humid: blue
    (100.0, (70, 15, 100)),   # saturated: deep purple
]
LABELS = [(0.0, "LOWEST 0"), (25.0, "25"), (50.0, "50"),
          (75.0, "75"), (100.0, "HIGHEST+ 100")]
SCALE_HTML = ("Near-surface relative humidity in percent (HRRR 2 m analysis, "
              "fixed absolute scale): <b>LOWEST 0</b> amber arid &rarr; "
              "yellow dry &rarr; green &rarr; teal humid &rarr; blue very "
              "humid &rarr; <b>HIGHEST+ 100</b> deep-purple saturated. Same "
              "humidity always shows the same color. Saturation percentage "
              "at current temperature — not precipitable water vapor, not "
              "dew point.")


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    import hrrr
    try:
        base, dd, cc = hrrr.latest_cycle()
    except Exception as e:
        print(f"[{PRODUCT}] NO CYCLE AVAILABLE (keeping previous): {e}")
        return 2
    source_id = f"hrrr-{dd}-t{cc}z-rh-anl"
    if should_skip(PRODUCT, source_id):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        return refresh_kml(PRODUCT, KML_FILE, OVERLAY_NAME,
                           CONFIG["title"],
                           CONFIG["refresh_interval_seconds"])
    try:
        return _build(dd, cc, source_id)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(dd, cc, source_id):
    bounds = load_bounds()

    def _vals(got):
        v, la, lo, ddate, dtime = got["RH"]
        v = np.asarray(v, dtype=float).ravel()
        ok = np.isfinite(v) & (v >= 0.0) & (v <= 100.0)
        if int(ok.sum()) < 5_000:
            raise ValueError(f"too few valid source cells ({int(ok.sum())})")
        return np.where(ok, v, np.nan), la, lo, ddate, dtime

    field, data_time_utc, _dd, _cc, _base = hrrr_field(
        _vals, [("RH", "2 m above ground")], "hrrr_rh_current.grib2",
        bounds, RAW_DIR)
    subtitle = (f"2 m relative humidity (%, HRRR hourly)  |  "
                f"{data_time_utc}")
    return finish(
        PRODUCT, CONFIG, KML_FILE, OVERLAY_NAME, field, STOPS, LABELS, "%",
        subtitle,
        f"Source: NOAA HRRR {dd} t{cc}z (analysis)  |  "
        f"Processed {now_det_str()}", SCALE_HTML,
        [CONFIG["what"], CONFIG["field"],
         "Checked hourly; republishes only on a newer HRRR cycle."],
        {"model_cycle": f"{dd} t{cc}z"}, source_id, data_time_utc,
        "n/a (NOMADS)", "% (display; source % as filed)",
        "~3 km HRRR CONUS grid, mean-binned to the common canvas",
        "only [0,100] % admitted; full basin rectangle, no shoreline cut; "
        "missing analysis transparent; never zero-filled.",
        alpha=170)


if __name__ == "__main__":
    sys.exit(main())
