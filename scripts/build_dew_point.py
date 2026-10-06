"""Pipeline V8 — LIVE DEW POINT (independent).

NOAA/NCEP HRRR 3 km 2 m dew point (DPT) analysis, hourly cycles,
Kelvin -> Fahrenheit -> FIXED absolute -20..90 F continuous spectrum ->
FULL BASIN RECTANGLE (lon -93..-73.5, lat 40.5..49.5, the exact LIVE LEAF
COLOR footprint) -> key image + metadata -> Folder live KML + stable
entry KML. Actual dew-point temperature; never spread/probability/RH.
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

PRODUCT = "dew_point"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Dew_Point.kml"
OVERLAY_NAME = "🧊 LIVE DEW POINT"

K2F = lambda k: (k - 273.15) * 9.0 / 5.0 + 32.0  # noqa: E731
STOPS = [
    (-20.0, (40, 10, 70)),     # very dry: deep purple
    (0.0, (16, 52, 140)),      # dry: blue
    (20.0, (20, 110, 200)),    # crisp: light blue
    (32.0, (45, 195, 178)),    # freezing: teal
    (45.0, (125, 205, 95)),    # pleasant: green
    (60.0, (240, 200, 45)),    # humid: yellow
    (75.0, (245, 155, 35)),    # muggy: amber
    (90.0, (220, 50, 30)),     # oppressive: red
]
LABELS = [(-20.0, "LOWEST -20"), (0.0, "0"), (32.0, "freezing 32"),
          (60.0, "60"), (90.0, "HIGHEST+ 90")]
SCALE_HTML = ("Dew-point temperature in degrees Fahrenheit (HRRR 2 m "
              "analysis, fixed absolute scale): <b>LOWEST -20</b> deep-purple "
              "very dry &rarr; blue &rarr; teal at <b>freezing 32</b> &rarr; "
              "green &rarr; yellow humid &rarr; amber muggy &rarr; "
              "<b>HIGHEST+ 90</b> red oppressive. Same dew point always shows "
              "the same color. Actual dew-point temperature — not spread, "
              "not probability, not relative humidity.")


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
    source_id = f"hrrr-{dd}-t{cc}z-dpt-anl"
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
        v, la, lo, ddate, dtime = got["DPT"]
        v = np.asarray(v, dtype=float).ravel()
        lok = CONFIG["valid_min_c"] + 273.15
        hik = CONFIG["valid_max_c"] + 273.15
        ok = np.isfinite(v) & (v >= lok) & (v <= hik)
        if int(ok.sum()) < 5_000:
            raise ValueError(f"too few valid source cells ({int(ok.sum())})")
        return np.where(ok, K2F(v), np.nan), la, lo, ddate, dtime

    field, data_time_utc, _dd, _cc, _base = hrrr_field(
        _vals, [("DPT", "2 m above ground")], "hrrr_dpt_current.grib2",
        bounds, RAW_DIR)
    subtitle = (f"2 m dew point (degF, HRRR hourly)  |  {data_time_utc}")
    return finish(
        PRODUCT, CONFIG, KML_FILE, OVERLAY_NAME, field, STOPS, LABELS,
        "degF", subtitle,
        f"Source: NOAA HRRR {dd} t{cc}z (analysis)  |  "
        f"Processed {now_det_str()}", SCALE_HTML,
        [CONFIG["what"], CONFIG["field"],
         "Checked hourly; republishes only on a newer HRRR cycle."],
        {"model_cycle": f"{dd} t{cc}z"}, source_id, data_time_utc,
        "n/a (NOMADS)", "degF (display; source K)",
        "~3 km HRRR CONUS grid, mean-binned to the common canvas",
        "only [-45,35] C admitted pre-conversion; full basin rectangle, "
        "no shoreline cut; missing analysis transparent; never "
        "zero-filled.")


if __name__ == "__main__":
    sys.exit(main())
