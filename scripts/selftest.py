"""Offline self-tests: Beaufort math, direction convention, shoreline mask,
legend/raster consistency. No network. Run: python scripts/selftest.py
Exit 0 = all pass, 1 = failure. Also executed by the publish job.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("ok   " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""))


def main():
    from beaufort import (BEAUFORT, FORCE_COLORS, force_from_kt, force_name,
                          force_range_text, movement_to_canvas_dxdy,
                          uv_to_speed_dir)
    from geospatial_utils import REPO_ROOT, SITE_DIR, load_bounds, load_watermask

    # ---- Beaufort thresholds (exact table from the task) ----
    cases = [(0, 0), (0.99, 0), (1, 1), (3, 1), (4, 2), (6, 2), (7, 3),
             (10, 3), (11, 4), (16, 4), (17, 5), (21, 5), (22, 6), (27, 6),
             (28, 7), (33, 7), (34, 8), (40, 8), (41, 9), (47, 9), (48, 10),
             (55, 10), (56, 11), (63, 11), (64, 12), (100, 12)]
    check("beaufort-13-forces", len(BEAUFORT) == 13, len(BEAUFORT))
    check("beaufort-boundaries",
          all(force_from_kt(kt) == f for kt, f in cases))
    check("beaufort-f12-gte-64", "64" in force_range_text(12), force_range_text(12))
    check("beaufort-f12-hurricane", force_name(12) == "Hurricane-force")
    check("beaufort-colors-13", set(FORCE_COLORS) == set(range(13)))
    check("beaufort-f12-dark-purple", FORCE_COLORS[12] == "#3B0A54")
    check("beaufort-f12-distinct", FORCE_COLORS[12] != FORCE_COLORS[11])

    # ---- U/V direction convention (movement TOWARD, no reversal) ----
    sp, ang = uv_to_speed_dir(5.0, 0.0)
    check("uv-speed", abs(sp - 5.0) < 1e-9, sp)
    check("uv-east-angle", abs(ang - 0.0) < 1e-9, ang)
    sp, ang = uv_to_speed_dir(0.0, 5.0)
    check("uv-north-angle", abs(ang - math.pi / 2) < 1e-9, ang)
    # meteorological "west wind" (FROM 270) blows TOWARD east: U>0 must
    # yield an eastward (screen-right) arrow
    dx, dy = movement_to_canvas_dxdy(10.0, 0.0)
    check("arrow-west-wind-eastward", dx > 0 and dy == 0, (dx, dy))
    dx, dy = movement_to_canvas_dxdy(0.0, 7.0)
    check("arrow-south-wind-northward", dx == 0 and dy < 0, (dx, dy))

    # ---- rendered arrow orientation (synthetic uniform fields) ----
    import numpy as _np
    from build_wind import draw_arrows as _arrows
    _H = _W = 300
    _base = _np.zeros((_H, _W, 4), dtype=_np.uint8)
    _base[:, :, 3] = 205
    _ok = _np.ones((30, 30), bool)
    _rr = _np.tile((_np.arange(30) * 10 + 5)[:, None], (1, 30))
    _cc = _np.tile(_np.arange(30) * 10 + 5, (30, 1))
    _dims = {"canvas_width": _W, "canvas_height": _H}
    _ee = _np.full((30, 30), 10.0)
    _zz = _np.zeros((30, 30))
    _e, _ = _arrows(_base.copy(), _ee, _zz, _ok, _rr, _cc, _dims, step=15)
    check("render-eastward-shaft", _e[155, 151:160, 0].max() > 200)
    _n, _ = _arrows(_base.copy(), _zz, _ee, _ok, _rr, _cc, _dims, step=15)
    check("render-northward-tip", _n[148:152, 155, 0].max() > 200)
    check("render-northward-no-overshoot", _n[140:146, 155, 0].max() < 200)
    check("render-northward-no-reversal", _n[164:172, 155, 0].max() < 200)

    # ---- shared shoreline mask ----
    bounds = load_bounds()
    mp = os.path.join(REPO_ROOT, "assets", "great_lakes_watermask.png")
    check("mask-exists", os.path.exists(mp))
    mask = load_watermask()
    check("mask-dims", mask.shape == (bounds["canvas_height"], bounds["canvas_width"]),
          mask.shape)
    wf = float(((mask * 255) > 127).mean())
    check("mask-water-frac", 0.10 < wf < 0.30, round(wf, 4))

    def mv(lon, lat):
        c = int((lon - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"])
                * bounds["canvas_width"])
        r = int((bounds["lat_max"] - lat) / (bounds["lat_max"] - bounds["lat_min"])
                * bounds["canvas_height"])
        return mask[r, c]

    for name, lon, lat, want in [
            ("mask-superior", -88, 47.5, 1), ("mask-michigan", -87, 44.2, 1),
            ("mask-huron", -82.5, 44.8, 1), ("mask-erie", -81.5, 42.2, 1),
            ("mask-ontario", -77.8, 43.6, 1), ("mask-wisconsin", -90, 45.5, 0),
            ("mask-toronto", -79.4, 43.65, 0), ("mask-isu-royale", -88.9, 48.0, 0)]:
        v = mv(lon, lat)
        check(name, (v > 0.5) == bool(want), round(float(v), 3))

    # ---- KML single-overlay hygiene (GE Web: 1 external image, no Region) ----
    import re as _re
    prods = ["wave_height", "water_temperature", "ice_coverage",
             "ice_thickness", "ice_type", "wind", "leaf_color"]
    for p in prods:
        for kf in (os.path.join(REPO_ROOT, "kml", f"Great_Lakes_Live_{_k(p)}.kml"),):
            t = open(kf, encoding="utf-8").read()
            check(f"{p}-single-overlay", t.count("<GroundOverlay>") == 1,
                  t.count("<GroundOverlay>"))
            check(f"{p}-no-tiles-refs", "/tiles/" not in t)
    import numpy as np
    from PIL import Image
    for p in prods:
        png = os.path.join(SITE_DIR, p, "current.png")
        a = np.array(Image.open(png).convert("RGBA"))
        if p == "leaf_color":
            _bad = ((a[:, :, 3] > 0) & (mask > 250 / 255)).sum() == 0
        else:
            _bad = ((a[:, :, 3] > 0) & (mask <= 0)).sum() == 0
        check(f"{p}-opaque-subset-of-mask", bool(_bad))
        for kf in (os.path.join(REPO_ROOT, "kml", f"Great_Lakes_Live_{_k(p)}.kml"),
                   os.path.join(SITE_DIR, "kml", f"Great_Lakes_Live_{_k(p)}.kml")):
            t = open(kf, encoding="utf-8").read()
            check(f"{p}-no-screenoverlay", "<ScreenOverlay" not in t)
            check(f"{p}-no-placeholders",
                  "REPLACE-GITHUB-USER" not in t and "REPLACE-REPO" not in t)
            check(f"{p}-has-groundoverlay", "<GroundOverlay>" in t)
            check(f"{p}-kml-small", len(t) < 100_000, len(t))

    # ---- leaf phenology engine (synthetic trajectory §43) ----
    from leaf_phenology import (build_leaf_lut, phenology_phase,
                                redness_index)
    _lut = build_leaf_lut()
    check("leaf-lut-256", len(_lut) == 256)
    check("leaf-lut-wrap", _lut[0] == _lut[-1] == (18, 59, 115))
    check("leaf-lut-many-colors", len(set(_lut)) > 200, len(set(_lut)))
    # 15 synthetic states across the annual cycle (deciduous, cls=1)
    _traj = [
        # (ndvi, hist(newest-last), redness, snow, bad) -> expected phase band
        ((0.12, [0.12, 0.13], 0.05, False, False), (0.0, 0.12)),    # 1 deep winter
        ((0.22, [0.13, 0.12], 0.05, False, False), (0.12, 0.30)),   # 2 awakening
        ((0.35, [0.22, 0.15], 0.05, False, False), (0.15, 0.35)),   # 3 budding
        ((0.50, [0.35, 0.28], 0.05, False, False), (0.25, 0.45)),   # 4 first leaves
        ((0.68, [0.55, 0.45], 0.05, False, False), (0.30, 0.50)),   # 5 development
        ((0.82, [0.80, 0.81], 0.05, False, False), (0.45, 0.60)),   # 6 peak green
        ((0.74, [0.82, 0.83], 0.15, False, False), (0.55, 0.70)),   # 7 first change
        ((0.60, [0.74, 0.80], 0.45, False, False), (0.60, 0.78)),   # 8 yellow
        ((0.48, [0.60, 0.70], 0.55, False, False), (0.65, 0.80)),   # 9 gold
        ((0.38, [0.48, 0.58], 0.70, False, False), (0.70, 0.84)),   # 10 orange
        ((0.28, [0.38, 0.48], 0.80, False, False), (0.74, 0.88)),   # 11 red
        ((0.20, [0.28, 0.36], 0.60, False, False), (0.80, 0.95)),   # 12 late autumn
        ((0.14, [0.20, 0.26], 0.30, False, False), (0.85, 1.00)),   # 13 leaf drop
        # 14 dormant (circular: blue edge or purple edge both dormant)
        ((0.11, [0.14, 0.17], 0.05, False, False), None),
    ]
    _mono = True
    _prev = -1.0
    for (_ndvi, _h, _r, _s, _b), _band in _traj:
        _ph, _ = phenology_phase(_ndvi, _h, 1, _r, _s, _b)
        if _band is None:  # wraparound-adjacent dormant
            _ok = _ph is not None and (_ph < 0.12 or _ph > 0.85)
        else:
            _lo, _hi = _band
            _ok = _ph is not None and _lo <= _ph < _hi
        check(f"leaf-phase-{_band}", _ok, _ph)
        _mono = _mono and (_ph is not None and _ph >= _prev - 0.05)
        _prev = _ph if _ph is not None else _prev
    check("leaf-cycle-monotonic", _mono)
    _p0, _ = phenology_phase(0.85, [0.85, 0.85], 1, 0.0, False, False)
    check("leaf-summer-green", 0.40 <= _p0 <= 0.60, _p0)
    _pe, _ = phenology_phase(0.30, [0.45, 0.55], 3, 0.80, False, False)
    check("leaf-evergreen-clamped", 0.38 <= _pe <= 0.58, _pe)
    _ps, _ = phenology_phase(0.30, [0.30, 0.30], 1, 0.0, False, True)
    check("leaf-snow-transparent", _ps is None)
    _ph2, _held = phenology_phase(0.30, [0.30, 0.30], 1, 0.0, False, True,
                                  prev_phase=0.5)
    check("leaf-badobs-hold", _held and _ph2 == 0.5)
    _pc, _ = phenology_phase(0.75, [0.75, 0.75], 6, 0.0, False, False)
    check("leaf-crop-subdued", 0.40 <= _pc <= 0.60, _pc)
    _pu, _ = phenology_phase(0.75, [0.75, 0.75], 7, 0.0, False, False)
    check("leaf-urban-masked", _pu is None)
    check("leaf-redness-summer", redness_index(0.2, 0.4, 0.2) < 0.3)
    check("leaf-redness-autumn", redness_index(0.55, 0.3, 0.15) > 0.5)

    # ---- leaf assets: Michigan mask + land cover ----
    import numpy as _np2
    from PIL import Image as _Im
    _fp = _np2.array(_Im.open(os.path.join(REPO_ROOT, "assets",
                                           "michigan_mask.png")).convert("L"))
    check("leaf-footprint-dims", _fp.shape == (1175, 1800), _fp.shape)
    check("leaf-footprint-frac", 0.05 < (_fp > 0).mean() < 0.30,
          round(float((_fp > 0).mean()), 3))
    _lc = _np2.array(_Im.open(os.path.join(REPO_ROOT, "assets",
                                           "leaf_landcover.png")).convert("L"))
    check("leaf-landcover-dims", _lc.shape == (1175, 1800), _lc.shape)
    _lu, _lcnt = _np2.unique(_lc, return_counts=True)
    check("leaf-landcover-codes", set(_lu.tolist()) <= set(range(10)),
          sorted(_lu.tolist()))
    for _code, _nm in ((1, "deciduous"), (2, "mixed"), (3, "evergreen")):
        check(f"leaf-landcover-{_nm}",
              int((_lc == _code).sum()) > 50_000, int((_lc == _code).sum()))

    # ---- gradient-scale engine (products 8-11) ----
    import numpy as _np3
    from gradient_scale import (anchor_values, build_stops, color_for,
                                family_color, lut_from_stops)
    _rec = {"hist_min": 0.15, "hist_max": 99.9,
            "percentiles": {"p5": 0.44, "p25": 0.63, "p50": 1.1, "p75": 3.6,
                            "p95": 16.0, "p99": 44.0}}
    _st = build_stops(anchor_values(_rec), False)
    check("grad-8-anchors", len(_st) == 8)
    check("grad-ascending",
          all(_st[i][0] < _st[i + 1][0] for i in range(7)))
    check("grad-low-darkblue", _st[0][1] == (16, 52, 140))
    check("grad-high-purple", _st[-1][1] == (70, 15, 100))
    _lut256 = lut_from_stops(_st)
    _dense = lut_from_stops(_st, 4096)
    _jd = _np3.abs(_np3.diff(_np3.array(_dense, dtype=int), axis=0)).sum(axis=1)
    check("grad-continuous", _jd.max() < 25, int(_jd.max()))
    _j = _np3.abs(_np3.diff(_np3.array(_lut256, dtype=int), axis=0)).sum(axis=1)
    check("grad-no-hard-band", _j.max() < 260, int(_j.max()))
    check("grad-clamp-low", color_for(-5, _st) == _st[0][1])
    check("grad-clamp-high", color_for(9999, _st) == _st[-1][1])
    check("grad-mid-varying",
          len({_lut256[i] for i in range(60, 140)}) > 40)
    _neg = build_stops([-22.5, -15.0, -2.0, 18.0, 34.0, 55.0, 68.0, 78.5],
                       True)
    check("grad-neg-min-purple", _neg[0][1] == (40, 10, 70))
    check("grad-neg-zero-blue",
          any(v == 0.0 and c == (16, 52, 140) for v, c in _neg))
    check("grad-neg-continuous",
          _np3.abs(_np3.diff(_np3.array(lut_from_stops(_neg), dtype=int),
                             axis=0)).sum(axis=1).max() < 30)
    _pos = build_stops([0.02, 0.1, 0.15, 0.2, 0.3, 0.8, 1.8, 6.3], False)
    check("grad-nonneg-no-purple-low", _pos[0][1] == (16, 52, 140))
    check("grad-family-endpoints", family_color(0.0) == (16, 52, 140)
          and family_color(1.0) == (70, 15, 100))

    # ---- snow product ----
    import numpy as _np4
    from gradient_scale import SNOW_FAMILY, build_stops as _bs
    _snow_stops = _bs([0.5, 1.0, 3.0, 6.0, 10.0, 16.0, 24.0, 36.0], False,
                      family=SNOW_FAMILY)
    check("snow-8-anchors", len(_snow_stops) == 8)
    check("snow-trace-silver", _snow_stops[0][1] == (192, 200, 208))
    check("snow-extreme-white", _snow_stops[-1][1] == (255, 255, 255))
    from gradient_scale import render_rgba as _rr
    _syn = _np4.full((60, 60), _np4.nan)
    _syn[10:50, 10:50] = _np4.linspace(0.5, 36, 40)[:, None] * _np4.ones(40)
    _rgba = _rr(_syn, _snow_stops, 205)
    check("snow-transparent-outside",
          bool(((_rgba[:, :, 3] == 0).sum()) == 60 * 60 - 40 * 40))
    check("snow-opaque-inside", bool(((_rgba[10:50, 10:50, 3] > 0).all())))
    _uc = len(_np4.unique(_rgba[10:50, 10:50, :3].reshape(-1, 3), axis=0))
    check("snow-many-colors", _uc >= 40, _uc)  # 1:1 value->color, no banding
    # white reserved for the extreme end: only top-decile pixels are ~white
    _white = ((_rgba[:, :, 0] > 245) & (_rgba[:, :, 1] > 245) &
              (_rgba[:, :, 2] > 245) & (_rgba[:, :, 3] > 0)).sum()
    check("snow-white-rare", 0 < _white < 0.15 * 40 * 40, int(_white))

    # ---- leaf Michigan-only + OKLab ----
    from PIL import Image as _Im3
    _lpng = _np4.array(_Im3.open(os.path.join(
        SITE_DIR, "leaf_color", "current.png")).convert("RGBA"))
    _mm2 = _np4.array(_Im3.open(os.path.join(
        REPO_ROOT, "assets", "michigan_mask.png")).convert("L")) > 0
    check("leaf-michigan-only",
          bool(((_lpng[:, :, 3] > 0) & ~_mm2).sum() == 0))
    from gradient_scale import oklab_lut
    from leaf_phenology import PHASE_ANCHORS
    _olut = oklab_lut(PHASE_ANCHORS)
    check("leaf-oklab-19-anchors", len(PHASE_ANCHORS) == 19)
    check("leaf-oklab-wrap", _olut[0] == _olut[-1] == (18, 59, 115))
    check("leaf-oklab-many", len(set(_olut)) > 200, len(set(_olut)))
    _jd = _np4.abs(_np4.diff(_np4.array(
        oklab_lut(PHASE_ANCHORS, 4096), dtype=int), axis=0)).sum(axis=1)
    check("leaf-oklab-continuous", _jd.max() < 25, int(_jd.max()))

    # ---- wind metadata specifics ----
    m = json.load(open(os.path.join(SITE_DIR, "wind", "metadata.json")))
    check("wind-beaufort-13", len(m.get("beaufort_table", [])) == 13)
    check("wind-f12-meta", "64" in m["beaufort_table"][12]["range_kt"])
    check("wind-arrows-meta", m.get("stats", {}).get("arrows_drawn", 0) >= 50)
    check("wind-f12-color-meta", m["beaufort_table"][12]["color"] == "#3B0A54")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


def _k(p):
    return {"wave_height": "Wave_Height", "water_temperature": "Water_Temperature",
            "ice_coverage": "Ice_Coverage", "ice_thickness": "Ice_Thickness",
            "ice_type": "Ice_Type", "wind": "Wind",
            "leaf_color": "Leaf_Color"}[p]


if __name__ == "__main__":
    sys.exit(main())
