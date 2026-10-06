"""Pipeline V1 — LIVE ATMOSPHERIC VISIBILITY (independent).

NOAA/NCEP HRRR 3 km surface visibility (VIS) analysis, hourly cycles,
metres -> statute miles -> FIXED absolute 0-30 mi continuous spectrum ->
FULL BASIN RECTANGLE (lon -93..-73.5, lat 40.5..49.5, the exact LIVE LEAF
COLOR footprint; land and water both paint, only missing data is
transparent) -> key image + metadata -> Folder live KML + stable entry
KML. Meteorological visibility for navigation/aviation/shoreline haze;
never astronomical seeing. Exit 0 updated/skipped; 2 source failure
(previous kept); 1 unexpected error.
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

PRODUCT = "visibility"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Atmospheric_Visibility.kml"
OVERLAY_NAME = "\U0001F319 LIVE ATMOSPHERIC VISIBILITY"

M2MI = 1.0 / 1609.344
STOPS = [
    (0.0, (60, 10, 20)),     # obscured: maroon
    (1.0, (190, 25, 25)),    # dense fog/haze: red
    (3.0, (235, 90, 20)),    # poor: orange
    (5.0, (240, 200, 40)),    # moderate: yellow
    (10.0, (90, 190, 80)),   # good: green
    (15.0, (20, 190, 200)),   # very good: cyan
    (20.0, (20, 110, 200)),   # excellent: blue
    (30.0, (16, 52, 140)),    # crystal clear: deep blue
]
LABELS = [(0.0, "LOWEST 0"), (3.0, "3"), (10.0, "10"),
          (20.0, "20"), (30.0, "HIGHEST+ 30")]
SCALE_HTML = ("Meteorological visibility in statute miles (HRRR surface "
              "analysis, fixed absolute scale): <b>LOWEST 0</b> maroon "
              "obscured &rarr; red &rarr; orange &rarr; yellow moderate "
              "&rarr; green good &rarr; cyan &rarr; blue excellent &rarr; "
              "<b>HIGHEST+ 30</b> deep-blue crystal clear. Same visibility "
              "always shows the same color; above 30 mi clamps into deep "
              "blue. Human-observer/mariner/pilot visibility through haze, "
              "mist, precipitation, and smoke — not astronomical seeing.")


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
    source_id = f"hrrr-{dd}-t{cc}z-vis-anl"
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
        v, la, lo, ddate, dtime = got["VIS"]
        v = np.asarray(v, dtype=float).ravel()
        ok = np.isfinite(v) & (v >= CONFIG["valid_min_m"]) & (v <= CONFIG["valid_max_m"])
        if int(ok.sum()) < 5_000:
            raise ValueError(f"too few valid source cells ({int(ok.sum())})")
        return np.where(ok, v * M2MI, np.nan), la, lo, ddate, dtime

    raw, data_time_utc, _dd, _cc, _base = hrrr_field(
        _vals, [("VIS", "surface")], "hrrr_vis_current.grib2",
        bounds, RAW_DIR)
    from live_field import smooth_nan
    # TV-style display smoothing (razor model-grid edges -> soft gradients;
    # single-cell speckles dissolve, coherent fog/low-vis areas persist).
    # Statistics stay on raw values.
    field = smooth_nan(raw)
    ok = np.isfinite(raw)
    subtitle = (f"Surface visibility (statute miles, HRRR hourly)  |  "
                f"{data_time_utc}")
    return finish(
        PRODUCT, CONFIG, KML_FILE, OVERLAY_NAME, field, STOPS, LABELS, "mi", subtitle,
        f"Source: NOAA HRRR {dd} t{cc}z (analysis)  |  "
        f"Processed {now_det_str()}", SCALE_HTML,
        [CONFIG["what"], CONFIG["field"],
         "Checked hourly; republishes only on a newer HRRR cycle."],
        {"model_cycle": f"{dd} t{cc}z",
         "stats": {"current_min": float(raw[ok].min()),
                   "current_max": float(raw[ok].max()),
                   "display_smoothing": "NaN-aware 2-pass blur; stats on raw"}},
        source_id, data_time_utc,
        "n/a (NOMADS)",
        "mi, statute (display; source m / 1609.344)",
        "~3 km HRRR CONUS grid, mean-binned to the common canvas, "
        "NaN-aware display smoothing",
        "only [0,60000] m admitted pre-conversion; values above 30 mi "
        "clamp into deep blue; full basin rectangle, no shoreline cut; "
        "missing analysis transparent; never zero-filled.",
        alpha=165)


if __name__ == "__main__":
    sys.exit(main())
