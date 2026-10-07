"""Pipeline V7 — LIVE FOG RISK/ACTIVE FOG (independent, DERIVED).

Authoritative inputs: NOAA/NCEP HRRR 3 km analysis, hourly cycles —
TMP+DPT+RH (2 m) + UGRD/VGRD (10 m wind) + VIS (surface). Each input is
binned onto the common canvas, then the documented fog-risk index is
computed in canvas space (see scripts/derived_moisture.py and
DATA_SOURCES.md): thermodynamic saturation base x calm-air factor, raised
only by observed low visibility (active-fog gate). FIXED absolute 0-100
continuous spectrum -> FULL BASIN RECTANGLE (lon -93..-73.5, lat
40.5..49.5, the exact LIVE LEAF COLOR footprint) -> key image + metadata
-> Folder live KML + stable entry KML. Distinct from raw visibility:
calm saturated air scores high before visibility collapses.
Labeled DERIVED everywhere. Exit 0/2/1 per contract.
"""

import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derived_moisture import fog_risk_index
from live_field import (SKIP_NOTE, finish, refresh_kml, should_skip)
from geospatial_utils import (REPO_ROOT, SITE_DIR, bin_to_canvas,
                              canvas_indices, load_bounds, now_det_str)

PRODUCT = "fog"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Fog_Risk.kml"
OVERLAY_NAME = "\U0001F32B\uFE0F LIVE FOG RISK/ACTIVE FOG"

MSGS = [("TMP", "2 m above ground"), ("DPT", "2 m above ground"),
        ("RH", "2 m above ground"), ("UGRD", "10 m above ground"),
        ("VGRD", "10 m above ground"), ("VIS", "surface")]
# v2 scale in the derivation's own vocabulary: risk owns 0-35, active
# fog by density owns 35-100 (config/fog.json derivation,
# scripts/derived_moisture.py). Even value steps carry the source's own
# zone words — "no observed fog" / "fog risk" on the risk side, "active
# fog" wherever the corroborated-visibility gate holds — so no invented
# severity system ("low/moderate/dense/extreme" retired). Muted palette
# in the existing hue family, never neon.
STOPS = [
    (0.0, (92, 182, 99)),      # no observed fog: green
    (20.0, (201, 217, 67)),    # fog risk: yellow-green
    (40.0, (252, 170, 25)),    # active fog: gold
    (60.0, (220, 49, 45)),     # active fog: red
    (80.0, (156, 41, 115)),   # active fog: violet
    (100.0, (61, 19, 91)),     # active fog: deep purple
]
LABELS = [(0.0, "0"), (20.0, "20"), (40.0, "40"),
          (60.0, "60"), (80.0, "80"), (100.0, "100")]
# Two-row even key (step 20): values on row 1, the source's own zone
# words on row 2. Fog-side ticks resolve to their exact source
# visibilities in the scale text (fog_zone = 35+65*d).
KEY_TICKS = [
    (0.0, "0", "No observed fog"),
    (20.0, "20", "Fog risk"),
    (40.0, "40", "Active fog"),
    (60.0, "60", "Active fog"),
    (80.0, "80", "Active fog"),
    (100.0, "100", "Active fog"),
]
SCALE_HTML = ("Fog index 0-100 (DERIVED from HRRR analysis, fixed absolute "
              "scale): <b>0 No observed fog</b> sage green &rarr; <b>20 "
              "Fog risk</b> yellow-green (risk owns 0-35: thermodynamic "
              "base x calm-air factor) &rarr; <b>40 Active fog</b> gold "
              "(VIS 4797 m / 2.98 mi) &rarr; <b>60 Active fog</b> brick "
              "red (VIS 3486 m / 2.17 mi) &rarr; <b>80 Active fog</b> "
              "violet (VIS 1842 m / 1.14 mi) &rarr; <b>100 Active fog</b> "
              "deep purple (VIS 0 m). Active fog by corroborated density "
              "owns 35-100 (gate: VIS below 5000 m with RH at/above 90%); "
              "dry-air speckles can never paint active fog. Same index "
              "always shows the same color — not a copy of the visibility "
              "layer.")


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
    source_id = f"hrrr-{dd}-t{cc}z-fog-anl"
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


def _bin_one(name, vals, lats, lons, bounds, shape, ok):
    H, W = shape
    v = np.asarray(vals, dtype=float).ravel()
    la = np.asarray(lats, dtype=float).ravel()
    lo = (((np.asarray(lons, dtype=float).ravel() + 180) % 360) - 180)
    rows, cols, _v = canvas_indices(la, lo, bounds)
    inside = (np.isfinite(la) & np.isfinite(lo)
              & (la >= bounds["lat_min"]) & (la <= bounds["lat_max"])
              & (lo >= bounds["lon_min"]) & (lo <= bounds["lon_max"]))
    field, _c = bin_to_canvas(rows, cols, np.where(ok, v, np.nan),
                              inside & np.isfinite(np.where(ok, v, np.nan)),
                              (H, W), splat_radius=2)
    return field


def _build(base, dd, cc, source_id):
    import hrrr
    from geospatial_utils import grib_stamp_to_det
    bounds = load_bounds()
    H, W = bounds["canvas_height"], bounds["canvas_width"]
    raw_path = os.path.join(RAW_DIR, "hrrr_fog_current.grib2")
    hrrr.fetch_messages(base, raw_path, MSGS)
    got = hrrr.read_messages(raw_path, {n: 1 for n, _l in MSGS})
    tv, tla, tlo, ddate, dtime = got["TMP"]
    dv = got["DPT"][0]
    rhv = got["RH"][0]
    uv = got["UGRD"][0]
    vv = got["VGRD"][0]
    visv = got["VIS"][0]
    data_time_utc = grib_stamp_to_det(ddate, dtime)
    tmp_c = _bin_one("TMP", np.asarray(tv, dtype=float).ravel() - 273.15,
                     tla, tlo, bounds, (H, W),
                     np.isfinite(np.asarray(tv, dtype=float).ravel())
                     & (np.asarray(tv, dtype=float).ravel() > 200)
                     & (np.asarray(tv, dtype=float).ravel() < 330))
    dpt_c = _bin_one("DPT", np.asarray(dv, dtype=float).ravel() - 273.15,
                     tla, tlo, bounds, (H, W),
                     np.isfinite(np.asarray(dv, dtype=float).ravel())
                     & (np.asarray(dv, dtype=float).ravel() > 200)
                     & (np.asarray(dv, dtype=float).ravel() < 330))
    rh = _bin_one("RH", rhv, tla, tlo, bounds, (H, W),
                  np.isfinite(np.asarray(rhv, dtype=float).ravel())
                  & (np.asarray(rhv, dtype=float).ravel() >= 0)
                  & (np.asarray(rhv, dtype=float).ravel() <= 100))
    uu = _bin_one("UGRD", uv, tla, tlo, bounds, (H, W),
                  np.isfinite(np.asarray(uv, dtype=float).ravel())
                  & (np.abs(np.asarray(uv, dtype=float).ravel()) < 100))
    vv2 = _bin_one("VGRD", vv, tla, tlo, bounds, (H, W),
                   np.isfinite(np.asarray(vv, dtype=float).ravel())
                   & (np.abs(np.asarray(vv, dtype=float).ravel()) < 100))
    vis = _bin_one("VIS", visv, tla, tlo, bounds, (H, W),
                   np.isfinite(np.asarray(visv, dtype=float).ravel())
                   & (np.asarray(visv, dtype=float).ravel() >= 0))
    wspd = np.sqrt(np.where(np.isfinite(uu), uu, np.nan) ** 2
                   + np.where(np.isfinite(vv2), vv2, np.nan) ** 2)
    from live_field import smooth_nan
    raw = fog_risk_index(tmp_c, rh, dpt_c, vis, wspd)
    ok = np.isfinite(raw)
    if int(ok.sum()) < 50_000:
        raise ValueError(f"too few valid canvas cells ({int(ok.sum())})")
    # TV-style display smoothing (razor model-grid edges -> soft gradients;
    # single-cell speckles dissolve, coherent fog persists). Statistics stay
    # on raw values.
    field = smooth_nan(raw)
    subtitle = (f"Fog index 0-100 (derived, HRRR hourly)  |  "
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
         "stats": {"derivation": CONFIG["derivation"],
                   "current_min": float(raw[ok].min()),
                   "current_max": float(raw[ok].max()),
                   "display_smoothing": "NaN-aware 2-pass blur; stats on raw"}},
        source_id, data_time_utc, "n/a (NOMADS)",
        "fog index 0-100 (display; derived, see derivation)",
        "~3 km HRRR CONUS grid fields, mean-binned to the common canvas, "
        "index computed in canvas space, NaN-aware display smoothing",
        "inputs gated to physical ranges pre-binning; index computed only "
        "where TMP+DPT+RH valid; full basin rectangle, no shoreline cut; "
        "missing analysis transparent; never zero-filled; corroborated "
        "visibility density outranks risk, dry-air speckles never paint "
        "active fog.",
        key_ticks=KEY_TICKS)


if __name__ == "__main__":
    sys.exit(main())
