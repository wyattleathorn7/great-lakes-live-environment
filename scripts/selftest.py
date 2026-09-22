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
             "ice_thickness", "ice_type", "wind"]
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
        check(f"{p}-opaque-subset-of-mask",
              bool((((a[:, :, 3] > 0) & (mask <= 0)).sum()) == 0))
        for kf in (os.path.join(REPO_ROOT, "kml", f"Great_Lakes_Live_{_k(p)}.kml"),
                   os.path.join(SITE_DIR, "kml", f"Great_Lakes_Live_{_k(p)}.kml")):
            t = open(kf, encoding="utf-8").read()
            check(f"{p}-no-screenoverlay", "<ScreenOverlay" not in t)
            check(f"{p}-no-placeholders",
                  "REPLACE-GITHUB-USER" not in t and "REPLACE-REPO" not in t)
            check(f"{p}-has-groundoverlay", "<GroundOverlay>" in t)
            check(f"{p}-kml-small", len(t) < 100_000, len(t))

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
            "ice_type": "Ice_Type", "wind": "Wind"}[p]


if __name__ == "__main__":
    sys.exit(main())
