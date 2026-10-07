"""Pipeline V1 — LIVE VISIBILITY (independent).

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
from live_field import (SKIP_NOTE, finish, refresh_kml, should_skip)
from geospatial_utils import (REPO_ROOT, SITE_DIR, load_bounds, now_det_str)

PRODUCT = "visibility"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Atmospheric_Visibility.kml"
OVERLAY_NAME = "🌤️ LIVE VISIBILITY"

M2MI = 1.0 / 1609.344
# FAA flight-category scale (AIM 7-1-7) + NWS Dense Fog Advisory + METAR
# 10SM reporting cap — the field's own professional standard, in the
# source's own units (statute miles). Render stops sit at the exact
# standard boundaries (true linear positions): 0 / 0.25 (NWS dense-fog
# advisory) / 1 (LIFR) / 3 (IFR) / 5 (MVFR) / 10 (VFR at the METAR cap) /
# 30 (display max). Muted professional palette in the existing hue
# family (maroon->red->orange->gold->green->teal->deep blue), never neon.
STOPS = [
    (0.0, (144, 29, 48)),     # dense fog: maroon (NWS <= 1/4 mi)
    (0.25, (205, 33, 33)),    # LIFR boundary: red
    (1.0, (241, 139, 29)),    # IFR boundary (FAA): orange
    (3.0, (248, 224, 47)),    # MVFR boundary (FAA): gold
    (5.0, (92, 182, 99)),     # VFR boundary (FAA): green
    (10.0, (46, 191, 199)),   # VFR at METAR 10SM cap: cyan
    (30.0, (0, 52, 162)),     # VFR, model resolves past 10SM: deep blue
]
LABELS = [(0.0, "0 Dense fog"), (0.25, "0.25 LIFR"), (1.0, "1 IFR"),
          (3.0, "3 MVFR"), (5.0, "5 VFR"), (10.0, "10 VFR 10SM"),
          (30.0, "30 VFR")]
# FAA-chart-style band key (custom legend): one row per source category,
# swatch sampled at the band midpoint so it equals the raster color.
VIS_BANDS = [  # (lo_mi, hi_mi, row label)
    (0.0, 0.25, "0\u20130.25 Dense fog (NWS advisory \u22641/4 mi)"),
    (0.25, 1.0, "0.25\u20131 LIFR (FAA)"),
    (1.0, 3.0, "1\u20133 IFR (FAA)"),
    (3.0, 5.0, "3\u20135 MVFR (FAA)"),
    (5.0, 10.0, "5\u201310 VFR (FAA; 10SM METAR cap)"),
    (10.0, 30.0, "10\u201330 VFR (model resolves past 10SM)"),
]
SCALE_HTML = ("Meteorological visibility in statute miles (HRRR surface "
              "analysis, FAA flight-category scale): <b>0-0.25 Dense "
              "fog</b> maroon (NWS Dense Fog Advisory, 1/4 mi or less) "
              "&rarr; <b>0.25-1 LIFR</b> red &rarr; <b>1-3 IFR</b> orange "
              "&rarr; <b>3-5 MVFR</b> gold &rarr; <b>5-30 VFR</b> green to "
              "deep blue (10SM is the METAR reporting cap; the model "
              "resolves past it). Same visibility always shows the same "
              "color; above 30 mi clamps into deep blue. "
              "Human-observer/mariner/pilot visibility through haze, "
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


STATION_POS = {  # ASOS airports: obs reference only, never the source
    "KORD": (-87.90, 41.97), "KDTW": (-83.35, 42.21),
    "KDLH": (-92.18, 46.84), "KBUF": (-78.73, 42.94),
    "KCLE": (-81.85, 41.41), "KMKE": (-87.90, 42.95),
    "KGRB": (-88.13, 44.48), "KERI": (-80.38, 42.08),
}


def _metar_qc(ddate, dtime, field, bounds, H, W):
    """Compare grid vs airport METAR visibility (Iowa State IEM ASOS
    archive of the NOAA/NWS network — same authoritative feed family as
    the precipitation WMS). Reference only; failures never break the
    pipeline (exit-2 contract stays with the HRRR source)."""
    import math
    import urllib.request
    from datetime import datetime, timedelta, timezone
    from geospatial_utils import USER_AGENT
    out = {}
    try:
        start = datetime(int(ddate[0:4]), int(ddate[4:6]), int(ddate[6:8]),
                         int(dtime[0:2]), tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return {"note": "unparsable analysis timestamp; QC skipped"}
    end = start + timedelta(hours=1)
    s1 = start.strftime("year1=%Y&month1=%-m&day1=%-d&hour1=%-H&minute1=%M")
    s2 = end.strftime("year2=%Y&month2=%-m&day2=%-d&hour2=%-H&minute2=%M")
    want_hh = start.strftime("%Y-%m-%d %H:")
    for sid, (lon, lat) in STATION_POS.items():
        try:
            url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
                   f"?station={sid}&data=vsby&{s1}&{s2}"
                   "&tz=Etc%2FUTC&format=onlycomma"
                   "&latlon=yes&missing=M&trace=T")
            req = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(req, timeout=30) as r:
                lines = r.read().decode(errors="replace").splitlines()
            vals = []
            header = (lines[0].split(",") if lines else [])
            try:
                ci_time = header.index("valid")
                ci_vsby = header.index("vsby")
            except ValueError:
                ci_time, ci_vsby = 1, 2
            for ln in lines[1:]:
                parts = ln.split(",")
                if len(parts) <= max(ci_time, ci_vsby):
                    continue
                if not parts[ci_time].startswith(want_hh):
                    continue
                rawv = parts[ci_vsby].strip().strip("MP")
                try:
                    v = float(rawv)
                    if math.isfinite(v):
                        vals.append(v)
                except (TypeError, ValueError):
                    continue
            if not vals:
                out[sid] = {"note": "no METAR obs this hour"}
                continue
            obs = sorted(vals)[len(vals) // 2]
            col = int((lon - bounds["lon_min"])
                      / (bounds["lon_max"] - bounds["lon_min"]) * W)
            row = int((bounds["lat_max"] - lat)
                      / (bounds["lat_max"] - bounds["lat_min"]) * H)
            cell = None
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < H and 0 <= cc < W \
                            and np.isfinite(field[rr, cc]):
                        cell = float(field[rr, cc])
                        break
                if cell is not None:
                    break
            diff = None if cell is None else round(abs(cell - obs), 2)
            tol = CONFIG.get("metar_qc_tolerance_mi", 3.0)
            if diff is None:
                flag = "NO_GRID_CELL"
            elif obs >= 10.0:
                # METAR reports cap at 10SM ("10 or greater"): the model
                # passes when it also shows clear air, fails only when it
                # claims fog (CHECK) under a 10SM observation.
                flag = "OK" if cell >= obs - tol else "CHECK"
            else:
                flag = ("OK" if diff <= tol else "CHECK")
            print(f"[{PRODUCT}] METAR {sid}: obs {obs}SM grid {cell} "
                  f"diff {diff} -> {flag}")
            out[sid] = {"obs_SM": obs, "grid_SM": cell,
                        "absdiff_SM": diff, "verdict": flag}
        except Exception as e:  # QC must never break a pipeline
            out[sid] = {"note": f"METAR fetch skipped: {str(e)[:120]}"}
    return out


def _build(dd, cc, source_id):
    import hrrr
    from derived_moisture import apply_fog_consistency_gate
    from geospatial_utils import (bin_to_canvas, canvas_indices,
                                  grib_stamp_to_det)
    bounds = load_bounds()
    H, W = bounds["canvas_height"], bounds["canvas_width"]
    base, _dd, _cc = hrrr.latest_cycle(30)
    raw_path = os.path.join(RAW_DIR, "hrrr_vis_current.grib2")
    hrrr.fetch_messages(base, raw_path,
                        [("VIS", "surface"), ("RH", "2 m above ground")])
    got = hrrr.read_messages(raw_path, {"VIS": 1, "RH": 1})
    vv, la, lo, ddate, dtime = got["VIS"]
    rhv = got["RH"][0]
    data_time_utc = grib_stamp_to_det(ddate, dtime)

    def _bin(vals, ok):
        v = np.asarray(vals, dtype=float).ravel()
        la1 = np.asarray(la, dtype=float).ravel()
        lo1 = (((np.asarray(lo, dtype=float).ravel() + 180) % 360) - 180)
        rows, cols, _v = canvas_indices(la1, lo1, bounds)
        inside = (np.isfinite(la1) & np.isfinite(lo1)
                  & (la1 >= bounds["lat_min"]) & (la1 <= bounds["lat_max"])
                  & (lo1 >= bounds["lon_min"]) & (lo1 <= bounds["lon_max"]))
        field, _c = bin_to_canvas(rows, cols, np.where(ok, v, np.nan),
                                  inside & np.isfinite(np.where(ok, v, np.nan)),
                                  (H, W), splat_radius=2)
        return field

    va = np.asarray(vv, dtype=float).ravel()
    ra = np.asarray(rhv, dtype=float).ravel()
    vis_mi = _bin(va * M2MI, np.isfinite(va)
                  & (va >= CONFIG["valid_min_m"])
                  & (va <= CONFIG["valid_max_m"]))
    rh = _bin(ra, np.isfinite(ra) & (ra >= 0.0) & (ra <= 100.0))
    # Model-consistency despike: sub-1-mile claims without saturated air
    # are internally inconsistent (uniform 200–500 m fills over water at
    # 80 % RH). Gated cells are then CONTINUITY-FILLED from surrounding
    # valid values (leaf gap-fill family — valid pixels never altered),
    # so the layer keeps the reference products' full-opaque basin
    # coverage: no interior transparency means Google Earth's bilinear
    # magnification can never blend data with transparent black (the gray
    # rectangles / yellow fringe failure). Corroborated fog cores persist.
    gated = apply_fog_consistency_gate(
        vis_mi, rh, mi_threshold=1.0,
        rh_min=CONFIG.get("corroboration_rh_min", 90.0))
    n_gated = int(np.isfinite(vis_mi).sum() - np.isfinite(gated).sum())
    from live_field import fill_missing_nearest, smooth_nan
    filled, n_filled = fill_missing_nearest(gated)
    # Light single-pass display smoothing: softens razor model-grid edges
    # while keeping the render crisp (a heavier blur read as low quality).
    # Statistics stay on raw values.
    field = smooth_nan(filled, passes=1)
    ok = np.isfinite(vis_mi)
    # ---- METAR QC (airport obs reference only; never alters the grid) ----
    metar_qc = _metar_qc(ddate, dtime, field, bounds, H, W)
    subtitle = (f"Surface visibility (statute miles, HRRR hourly)  |  "
                f"{data_time_utc}")

    def _visibility_legend(legend_path):
        from geospatial_utils import draw_category_legend
        from gradient_scale import color_for
        rows = [(color_for((lo + hi) / 2.0, STOPS), label)
                for lo, hi, label in VIS_BANDS]
        return draw_category_legend(
            legend_path, CONFIG["title"], subtitle, rows,
            "Source: NOAA HRRR surface visibility analysis (FAA AIM 7-1-7 "
            "flight categories; NWS Dense Fog Advisory \u22641/4 mi)",
            note="Missing source data transparent; never zero-filled.")

    return finish(
        PRODUCT, CONFIG, KML_FILE, OVERLAY_NAME, field, STOPS, LABELS, "mi", subtitle,
        f"Source: NOAA HRRR {dd} t{cc}z (analysis)  |  "
        f"Processed {now_det_str()}", SCALE_HTML,
        [CONFIG["what"], CONFIG["field"],
         "Checked hourly; republishes only on a newer HRRR cycle."],
        {"model_cycle": f"{dd} t{cc}z",
         "stats": {"current_min": float(vis_mi[ok].min()),
                   "current_max": float(vis_mi[ok].max()),
                   "n_gated_cells": n_gated,
                   "n_continuity_filled": n_filled,
                   "display_smoothing": "NaN-aware 1-pass blur; stats on raw"},
         "fields": {"metar_qc": metar_qc}},
        source_id, data_time_utc,
        "n/a (NOMADS)",
        "mi, statute (display; source m / 1609.344)",
        "~3 km HRRR CONUS grid, mean-binned to the common canvas, "
        "despike-gated, continuity-filled, NaN-aware display smoothing",
        "only [0,200000] m admitted pre-conversion (ultra-clear Arctic "
        "air often exceeds 60 km; gating lower punched transparent holes "
        "in the clearest regions); sub-1-mile claims require RH>=90% "
        "(model-consistency despike); gated cells continuity-filled from "
        "surrounding valid values (valid pixels never altered; fraction "
        "in metadata); values above 30 mi "
        "clamp into deep blue; full basin rectangle, no shoreline cut; "
        "residual edge no-data transparent with anti-fringe RGB bleed; "
        "never zero-filled.",
        alpha=165, edge_bleed=True, custom_legend=_visibility_legend)


if __name__ == "__main__":
    sys.exit(main())
