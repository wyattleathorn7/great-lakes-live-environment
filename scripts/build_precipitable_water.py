"""Pipeline V3 — LIVE PRECIPITABLE WATER VAPOR (independent).

NOAA/NCEP HRRR 3 km precipitable water (PWAT) analysis, hourly cycles,
kg m-2 = mm liquid-water depth -> FIXED absolute 0-60 mm continuous
spectrum -> FULL BASIN RECTANGLE (lon -93..-73.5, lat 40.5..49.5, the
exact LIVE LEAF COLOR footprint) -> key image + metadata -> Folder live
KML + stable entry KML. Column-integrated moisture; never RH.
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

PRODUCT = "precipitable_water"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Precipitable_Water_Vapor.kml"
OVERLAY_NAME = "☁️ LIVE PRECIPITABLE WATER VAPOR"

STOPS = [
    (0.0, (16, 52, 140)),     # dry column: dark blue
    (10.0, (20, 110, 200)),   # blue
    (20.0, (20, 190, 200)),   # cyan
    (30.0, (90, 190, 80)),    # green
    (40.0, (245, 215, 50)),   # yellow
    (50.0, (240, 130, 25)),   # orange
    (60.0, (70, 15, 100)),    # very moist: deep purple
]
LABELS = [(0.0, "LOWEST 0"), (15.0, "15"), (30.0, "30"),
          (45.0, "45"), (60.0, "HIGHEST+ 60")]
SCALE_HTML = ("Column-integrated water vapor in millimeters of equivalent "
              "liquid-water depth (HRRR PWAT analysis, fixed absolute "
              "scale): <b>LOWEST 0</b> dark-blue dry column &rarr; blue "
              "&rarr; cyan &rarr; green &rarr; yellow &rarr; orange &rarr; "
              "<b>HIGHEST+ 60</b> deep-purple very moist. Same moisture "
              "always shows the same color; above 60 mm clamps into deep "
              "purple. Total atmospheric-column moisture — not relative "
              "humidity.")


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
    source_id = f"hrrr-{dd}-t{cc}z-pwat-anl"
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
        v, la, lo, ddate, dtime = got["PWAT"]
        v = np.asarray(v, dtype=float).ravel()
        ok = np.isfinite(v) & (v >= CONFIG["valid_min_mm"]) & (v <= CONFIG["valid_max_mm"])
        if int(ok.sum()) < 5_000:
            raise ValueError(f"too few valid source cells ({int(ok.sum())})")
        return np.where(ok, v, np.nan), la, lo, ddate, dtime

    field, data_time_utc, _dd, _cc, _base = hrrr_field(
        _vals, [("PWAT", "entire atmosphere")], "hrrr_pwat_current.grib2",
        bounds, RAW_DIR)
    subtitle = (f"Precipitable water (mm, HRRR hourly)  |  {data_time_utc}")
    return finish(
        PRODUCT, CONFIG, KML_FILE, OVERLAY_NAME, field, STOPS, LABELS, "mm",
        subtitle,
        f"Source: NOAA HRRR {dd} t{cc}z (analysis)  |  "
        f"Processed {now_det_str()}", SCALE_HTML,
        [CONFIG["what"], CONFIG["field"],
         "Checked hourly; republishes only on a newer HRRR cycle."],
        {"model_cycle": f"{dd} t{cc}z"}, source_id, data_time_utc,
        "n/a (NOMADS)", "mm liquid-water depth (display; source kg m-2 = mm)",
        "~3 km HRRR CONUS grid, mean-binned to the common canvas",
        "only [0,80] mm admitted; values above 60 mm clamp into deep "
        "purple; full basin rectangle, no shoreline cut; missing analysis "
        "transparent; never zero-filled.")


if __name__ == "__main__":
    sys.exit(main())
