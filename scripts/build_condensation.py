"""Pipeline V6 — LIVE CONDENSATION (independent, DERIVED).

Authoritative inputs: NOAA/NCEP HRRR 3 km 2 m TMP+DPT+RH analysis, hourly
cycles. Each input is binned onto the common canvas, then the documented
condensation-favorability index is computed in canvas space (see
scripts/derived_moisture.py and DATA_SOURCES.md): dew-point depression +
relative humidity only — no wind, no visibility (unlike fog risk), and
not a copy of RH or dew point. FIXED absolute 0-100 continuous spectrum
-> FULL BASIN RECTANGLE (lon -93..-73.5, lat 40.5..49.5, the exact LIVE
LEAF COLOR footprint) -> key image + metadata -> Folder live KML +
stable entry KML. Labeled DERIVED everywhere. Exit 0/2/1 per contract.
"""

import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derived_moisture import condensation_index
from live_field import (SKIP_NOTE, finish, refresh_kml, should_skip)
from geospatial_utils import (REPO_ROOT, SITE_DIR, bin_to_canvas,
                              canvas_indices, load_bounds, now_det_str)

PRODUCT = "condensation"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Condensation.kml"
OVERLAY_NAME = "💦 LIVE CONDENSATION"

MSGS = [("TMP", "2 m above ground"), ("DPT", "2 m above ground"),
        ("RH", "2 m above ground")]
# Fixed 0-100 scale in the derivation's own vocabulary
# (config/condensation.json derivation, scripts/derived_moisture.py):
# even value steps carry the source's own nouns — relative humidity
# (whose term alone reaches exactly 35), dew-point depression (needed
# past 35; alone reaches exactly 65), surfaces stay wet (high values),
# saturation (100). No invented severity system ("slight / moderate /
# favorable" retired). Muted palette in the existing hue family.
STOPS = [
    (0.0, (206, 177, 104)),   # no condensation: tan
    (20.0, (248, 224, 47)),   # relative humidity: gold
    (40.0, (99, 193, 77)),    # dew-point depression: green
    (60.0, (18, 192, 192)),   # dew-point depression: teal
    (80.0, (19, 105, 221)),   # surfaces stay wet: blue
    (100.0, (0, 22, 165)),    # saturation: deep blue
]
LABELS = [(0.0, "0"), (20.0, "20"), (40.0, "40"),
          (60.0, "60"), (80.0, "80"), (100.0, "100")]
# Two-row even key (step 20): values on row 1, the source's own nouns on
# row 2. Formula landmarks (35 = humidity term alone; 65 = depression
# term alone) live in the scale text so ticks stay even.
KEY_TICKS = [
    (0.0, "0", "No condensation"),
    (20.0, "20", "Relative humidity"),
    (40.0, "40", "Dew-point depression"),
    (60.0, "60", "Dew-point depression"),
    (80.0, "80", "Surfaces stay wet"),
    (100.0, "100", "Saturation"),
]
SCALE_HTML = ("Condensation-favorability index 0-100 (DERIVED from HRRR "
              "analysis, fixed absolute scale): <b>0 No condensation</b> "
              "tan (dew-point depression at/above 2.5 C with RH at/below "
              "60%) &rarr; <b>20 Relative humidity</b> gold &rarr; <b>40 "
              "Dew-point depression</b> green &rarr; <b>60 Dew-point "
              "depression</b> teal &rarr; <b>80 Surfaces stay wet</b> "
              "steel blue &rarr; <b>100 Saturation</b> deep blue (spread "
              "at/below 0 C with RH 100%). Relative humidity alone "
              "reaches 35; the depression term alone reaches 65; both "
              "together reach saturation. Same index always shows the "
              "same color — not relative humidity renamed, not dew point "
              "renamed, no fog visibility gate.")


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
    source_id = f"hrrr-{dd}-t{cc}z-condensation-anl"
    if should_skip(PRODUCT, source_id):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        return refresh_kml(PRODUCT, KML_FILE, OVERLAY_NAME,
                           CONFIG["title"],
                           CONFIG["refresh_interval_seconds"])
    try:
        return _build(base, dd, cc, source_id)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _bin_one(vals, lats, lons, bounds, shape, ok):
    H, W = shape
    v = np.asarray(vals, dtype=float).ravel()
    la = np.asarray(lats, dtype=float).ravel()
    lo = (((np.asarray(lons, dtype=float).ravel() + 180) % 360) - 180)
    rows, cols, _v = canvas_indices(la, lo, bounds)
    field, _c = bin_to_canvas(rows, cols, np.where(ok, v, np.nan),
                              (np.isfinite(la) & np.isfinite(lo)
                               & (la >= bounds["lat_min"])
                               & (la <= bounds["lat_max"])
                               & (lo >= bounds["lon_min"])
                               & (lo <= bounds["lon_max"])
                               & np.isfinite(np.where(ok, v, np.nan))),
                              (H, W), splat_radius=2)
    return field


def _build(base, dd, cc, source_id):
    import hrrr
    from geospatial_utils import grib_stamp_to_det
    bounds = load_bounds()
    H, W = bounds["canvas_height"], bounds["canvas_width"]
    raw_path = os.path.join(RAW_DIR, "hrrr_condensation_current.grib2")
    hrrr.fetch_messages(base, raw_path, MSGS)
    got = hrrr.read_messages(raw_path, {n: 1 for n, _l in MSGS})
    tv, tla, tlo, ddate, dtime = got["TMP"]
    dv = got["DPT"][0]
    rhv = got["RH"][0]
    data_time_utc = grib_stamp_to_det(ddate, dtime)
    ta = np.asarray(tv, dtype=float).ravel()
    da = np.asarray(dv, dtype=float).ravel()
    ra = np.asarray(rhv, dtype=float).ravel()
    tmp_c = _bin_one(ta - 273.15, tla, tlo, bounds, (H, W),
                     np.isfinite(ta) & (ta > 200) & (ta < 330))
    dpt_c = _bin_one(da - 273.15, tla, tlo, bounds, (H, W),
                     np.isfinite(da) & (da > 200) & (da < 330))
    rh = _bin_one(ra, tla, tlo, bounds, (H, W),
                  np.isfinite(ra) & (ra >= 0) & (ra <= 100))
    field = condensation_index(tmp_c, rh, dpt_c)
    ok = np.isfinite(field)
    if int(ok.sum()) < 50_000:
        raise ValueError(f"too few valid canvas cells ({int(ok.sum())})")
    subtitle = (f"Condensation index 0-100 (derived, HRRR hourly)  |  "
                f"{data_time_utc}")
    return finish(
        PRODUCT, CONFIG, KML_FILE, OVERLAY_NAME, field, STOPS, LABELS,
        "index", subtitle,
        f"Source: NOAA HRRR {dd} t{cc}z (analysis, derived index)  |  "
        f"Processed {now_det_str()}", SCALE_HTML,
        [CONFIG["what"] + " Derivation: " + CONFIG["derivation"],
         CONFIG["field"],
         "Checked hourly; republishes only on a newer HRRR cycle."],
        {"model_cycle": f"{dd} t{cc}z",
         "stats": {"derivation": CONFIG["derivation"]}},
        source_id, data_time_utc, "n/a (NOMADS)",
        "condensation index 0-100 (display; derived, see derivation)",
        "~3 km HRRR CONUS grid fields, mean-binned to the common canvas, "
        "index computed in canvas space",
        "inputs gated to physical ranges pre-binning; index computed only "
        "where TMP+DPT+RH valid; full basin rectangle, no shoreline cut; "
        "missing analysis transparent; never zero-filled.",
        alpha=165, key_ticks=KEY_TICKS)


if __name__ == "__main__":
    sys.exit(main())
