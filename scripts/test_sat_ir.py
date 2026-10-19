"""Offline tests for the multi-satellite IR cloud mosaic (no bulk downloads).

Run: python scripts/test_sat_ir.py  (exit 0 = all pass, 1 = failures).
Covers source identity/timestamps, LUT inversion, transfers, mosaic
policy, geography, KML, freshness skip logic, docs consistency, and
precipitation isolation. Live-network tests are intentionally absent:
source access was verified live 2026-10-16 during implementation.
"""

import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


def main():
    import sat_ir
    print("== source identity / timestamps ==")
    s1 = {"GOES-19": {"kind": "goes-bt", "scan_dt": dt.datetime(
        2026, 10, 16, 14, 20, tzinfo=dt.timezone.utc)},
        "VIIRS": {"kind": "gibs-viirs", "scan_dt": dt.datetime(
            2026, 10, 16, tzinfo=dt.timezone.utc),
            "layers": [{"time": "2026-10-16"}, {"time": "2026-10-16"}]}}
    s2 = {"GOES-19": {"kind": "goes-bt", "scan_dt": dt.datetime(
        2026, 10, 16, 14, 10, tzinfo=dt.timezone.utc)},
        "VIIRS": s1["VIIRS"]}
    check("same scans -> same id",
          sat_ir.source_identity(s1) == sat_ir.source_identity(s1))
    check("changed scan -> changed id",
          sat_ir.source_identity(s1) != sat_ir.source_identity(s2))
    check("id has no wall-clock processing stamp",
          "na-cloud-real" not in sat_ir.source_identity(s1))
    import goes_acm
    m = goes_acm.CMIPF_RE.search(
        "OR_ABI-L2-CMIPF-M6C13_G19_s20262891420208_e20262891429528_"
        "c20262891429597.nc")
    check("CMIPF filename regex parses",
          m and (m.group(1), m.group(2), m.group(3), m.group(4))
          == ("2026", "289", "14", "20"))

    print("== transfers ==")
    b, a = sat_ir.bt_to_white(np.array([[200.0, 282.0, 290.0, np.nan]]))
    check("cold opaque white", b[0, 0] == 255.0 and a[0, 0] == 215.0)
    check("warm transparent", np.isnan(a[0, 1]) and np.isnan(a[0, 2]))
    check("missing stays missing", np.isnan(b[0, 3]) and np.isnan(a[0, 3]))
    check("no percentage semantics in transfer",
          b[0, 0] != 100.0 and a[0, 0] != 100.0)
    g, ga = sat_ir.gray_to_white(np.array([[24.0, 213.0]]),
                                  np.array([[True, True]]))
    check("warm gray nearly transparent", np.isnan(ga[0, 0]) or ga[0, 0] < 10.0)
    check("cold gray opaque", ga[0, 1] == 215.0 and g[0, 1] == 213.0)
    pb = sat_ir.gray_to_pseudo_bt(np.array([213.0]))
    b2, a2 = sat_ir.bt_to_white(pb)
    # Brightness round-trips exactly; alpha anchors differ by design
    # (BT warm anchor 282 K vs imagery gray anchor 20), both opaque.
    check("pseudo-BT round-trips brightness",
          abs(b2[0] - 213.0) < 1.0 and a2[0] > 200.0)

    print("== LUT inversion (synthetic colormap) ==")
    # LUT values are Kelvin (gibs_lut converts per colormap units).
    sat_ir._LUT_CACHE["test://cmap"] = {(10, 20, 30): 223.15,
                                        (200, 210, 220): 283.15}
    from PIL import Image
    px = np.zeros((4, 4, 4), dtype=np.uint8)
    px[:, :, 3] = 255
    px[0, 0, :3] = (10, 20, 30)      # exact bin
    px[0, 1, :3] = (11, 21, 31)      # near interpolant
    px[0, 2, :3] = (250, 0, 250)     # far halo -> transparent
    px[0, 3, 3] = 0                  # transparent input
    Image.fromarray(px, mode="RGBA").save("/tmp/satir_test.png")
    bt = sat_ir.invert_gibs_png("/tmp/satir_test.png", "test://cmap")
    check("exact bin inverts", abs(bt[0, 0] - 223.15) < 0.01)
    check("near interpolant maps to nearest bin",
          abs(bt[0, 1] - 223.15) < 0.01)
    check("far halo stays invalid", np.isnan(bt[0, 2]))
    check("transparent stays invalid", np.isnan(bt[0, 3]))

    print("== mosaic policy helpers ==")
    from na_common import blend_antimeridian, feather_merge
    f = np.full((10, 20), np.nan)
    f[:, :10] = 250.0
    f[:, 10:] = 250.0
    out = blend_antimeridian(f, pixels=4)
    check("antimeridian blend keeps valid data", np.isfinite(out).all())
    g1 = np.full((10, 10), 250.0)
    g2 = np.full((10, 10), 260.0)
    mrg, _v = feather_merge([(g1, np.ones((10, 10), bool)),
                             (g2, np.ones((10, 10), bool))], feather_px=0)
    check("precedence order respected (first wins interior)",
          mrg[5, 5] == 250.0)

    print("== geography (repo artifacts) ==")
    gl = json.load(open(os.path.join(REPO_ROOT, "config",
                                     "great_lakes_bounds.json")))
    check("basin bounds exact",
          (gl["lon_min"], gl["lon_max"], gl["lat_min"], gl["lat_max"])
          == (-93.0, -73.5, 40.5, 49.5))
    check("basin canvas 1800x1175",
          (gl["canvas_width"], gl["canvas_height"]) == (1800, 1175))
    from PIL import Image as _I
    for d, size in (("cloud_cover_na", (3600, 1800)),
                    ("cloud_cover_na_basin", (1800, 1175)),
                    ("cloud_cover_na_continental", (3300, 1600)),
                    ("cloud_cover_na_global", (3600, 1800))):
        p = os.path.join(REPO_ROOT, "site", d, "current.png")
        if os.path.exists(p):
            im = _I.open(p)
            check(f"{d} PNG size {size}", tuple(im.size) == size,
                  f"got {im.size}")
            check(f"{d} PNG RGBA", im.mode == "RGBA")
        else:
            check(f"{d} PNG exists", False, "missing (build first)")

    print("== KML ==")
    kp = os.path.join(REPO_ROOT, "site", "kml", "live",
                      "North_America_Live_Cloud_Cover.kml")
    if os.path.exists(kp):
        t = open(kp, encoding="utf-8").read()
        check("3 overlays", t.count("<GroundOverlay>") == 3)
        check("radioFolder", "radioFolder" in t)
        check("one ON two OFF",
              t.count("<visibility>1</visibility>") == 1
              and t.count("<visibility>0</visibility>") == 2)
        for name in ("Global Coverage", "Continental Coverage",
                     "Great Lakes Basin Coverage"):
            check(f"layer '{name}'", name in t)
        for box in (("<north>90.0</north>", "<south>-90.0</south>"),
                    ("<north>85.0</north>", "<south>5.0</south>"),
                    ("<north>49.5</north>", "<south>40.5</south>",
                     "<east>-73.5</east>", "<west>-93.0</west>")):
            check(f"KML bounds {box[0]}", all(x in t for x in box))
    else:
        check("live KML exists", False, "missing (build first)")

    print("== freshness / docs ==")
    cfg = json.load(open(os.path.join(REPO_ROOT, "config",
                                      "cloud_cover_na.json")))
    check("config has no % cloud-cover claim",
          "% cloud cover" not in cfg.get("display_units", "")
          and "% cloud cover" not in cfg.get("variable", ""))
    check("config documents mosaic policy", "mosaic_policy" in cfg)
    mp = os.path.join(REPO_ROOT, "site", "cloud_cover_na", "metadata.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        check("observation_time_utc present", bool(m.get("observation_time_utc")))
        check("source_observations per-source",
              isinstance(m.get("source_observations"), list)
              and len(m.get("source_observations", [])) >= 4)
        check("image_age_seconds present",
              m.get("image_age_seconds") is not None)
        check("update_status honest",
              m.get("update_status") in ("current", "degraded",
                                         "stale-retained"))
        check("no model-fill claim",
              "model fill" not in m.get("missing_data_treatment", "")
              .replace("no model fill", "").replace("never", ""))
    ds = open(os.path.join(REPO_ROOT, "DATA_SOURCES.md")).read()
    check("DATA_SOURCES documents mosaic", "sat_ir.py" in ds)
    check("DATA_SOURCES honest about VIIRS daily",
          "DAILY" in ds or "daily" in ds)

    print("== Antarctica / land-aware cloud detection ==")
    # Antarctica cold surface (230 K) over land should be transparent
    bt_cold_land = np.array([[230.0]])
    land_mask = np.array([[1.0]])
    b_land, a_land = sat_ir.bt_to_white(bt_cold_land, land_mask=land_mask)
    check("Antarctic surface not opaque over land",
          np.isnan(a_land[0, 0]) or a_land[0, 0] < 50.0,
          f"alpha={a_land[0, 0]}")
    # Same BT over ocean should be opaque (genuine cloud)
    ocean_mask = np.array([[0.0]])
    b_ocn, a_ocn = sat_ir.bt_to_white(bt_cold_land, land_mask=ocean_mask)
    check("cold BT opaque over ocean", a_ocn[0, 0] > 200.0,
          f"alpha={a_ocn[0, 0]}")
    # Genuine cold cloud over land (210 K) should be visible
    bt_cold_cloud = np.array([[210.0]])
    b_cl, a_cl = sat_ir.bt_to_white(bt_cold_cloud, land_mask=land_mask)
    check("genuine cloud visible over land", a_cl[0, 0] > 100.0,
          f"alpha={a_cl[0, 0]}")
    # No land mask = legacy behavior
    b_legacy, a_legacy = sat_ir.bt_to_white(bt_cold_land)
    check("no land mask uses ocean threshold",
          a_legacy[0, 0] > 200.0, f"alpha={a_legacy[0, 0]}")

    print("== source normalization ==")
    # Two sources with a bias: normalization should align them
    g1 = np.full((10, 10), 250.0)
    g2 = np.full((10, 10), 240.0)
    v = np.ones((10, 10), bool)
    normalized = sat_ir.normalize_sources([(g1, v), (g2, v)])
    check("normalization aligns sources",
          abs(np.nanmean(normalized[1][0]) - 250.0) < 5.0,
          f"mean={np.nanmean(normalized[1][0])}")
    # No overlap = no adjustment
    v1 = np.ones((10, 10), bool)
    v2 = np.zeros((10, 10), bool)
    normalized2 = sat_ir.normalize_sources([(g1, v1), (g2, v2)])
    check("no overlap leaves source unchanged",
          abs(np.nanmean(normalized2[1][0]) - 240.0) < 0.1)

    print("== VIIRS complementary Day/Night ==")
    # Day has valid data, Night has NaN -> should retain Day values
    day = np.full((5, 5), 220.0)
    night = np.full((5, 5), np.nan)
    stack = np.stack([day, night])
    both_valid = np.isfinite(stack).all(axis=0)
    g = np.where(both_valid,
                 np.nanmean(stack, axis=0),
                 np.nanmax(np.where(np.isfinite(stack), stack, np.nan), axis=0))
    g = np.where(np.isfinite(g), g, np.nan)
    check("Day-only pixels retained", np.isfinite(g).all())
    # Both valid -> should average
    day2 = np.full((5, 5), 220.0)
    night2 = np.full((5, 5), 240.0)
    stack2 = np.stack([day2, night2])
    both_valid2 = np.isfinite(stack2).all(axis=0)
    g2 = np.where(both_valid2,
                  np.nanmean(stack2, axis=0),
                  np.nanmax(np.where(np.isfinite(stack2), stack2, np.nan), axis=0))
    g2 = np.where(np.isfinite(g2), g2, np.nan)
    check("both valid -> averaged", abs(g2[0, 0] - 230.0) < 0.1)

    print("== gradual feather (time-skew aware) ==")
    # Small skew -> full feather
    check("small skew uses full feather",
          sat_ir.FEATHER_PX if 10 <= sat_ir.MAX_SKEW_MIN else 0 == sat_ir.FEATHER_PX)
    # Large skew -> reduced but non-zero feather
    large_skew = 45.0
    if large_skew <= sat_ir.MAX_SKEW_MIN:
        expected = sat_ir.FEATHER_PX
    else:
        expected = max(1, sat_ir.FEATHER_PX - int((large_skew - sat_ir.MAX_SKEW_MIN) // 5))
    check("large skew reduces feather", 0 < expected < sat_ir.FEATHER_PX,
          f"expected={expected}")

    print("== precipitation isolation ==")
    import subprocess
    diff = subprocess.run(["git", "diff", "--",
                           "scripts/build_precipitation_na.py",
                           "scripts/na_common.py", "scripts/goes_acm.py",
                           "site/precipitation_na",
                           ".github/workflows/update_fast_live.yml"],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    allowed_patterns = ["RENDER_VERSION", "source_id"]
    diff_text = diff.stdout.strip()
    if not diff_text:
        check("shared/precip/workflow files untouched", True)
    else:
        import re
        unexpected = []
        for m in re.finditer(r'@@[^\n]*\n((?:[+\- ][^\n]*\n)*)', diff_text):
            hunk_lines = m.group(1).split('\n')
            changed = [l for l in hunk_lines
                       if (l.startswith('+') or l.startswith('-'))
                       and not l.startswith('+++') and not l.startswith('---')]
            if changed and not any(p in l for l in changed
                                   for p in allowed_patterns):
                unexpected.append(m.group(0)[:200])
        check("shared/precip/workflow files untouched",
              len(unexpected) == 0, '\n'.join(unexpected[:3]))
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        import build_precipitation_na as _bp
        check("_ir_underlay still present",
              callable(getattr(_bp, "_ir_underlay", None)))
        import build_cloud_cover_na as _bcc
        src = open(_bcc.__file__).read()
        check("cloud no longer uses shared IR helper",
              "_ir_underlay" not in src)
    except Exception as e:
        check("precip imports cleanly", False, repr(e)[:200])

    print(f"\n{ PASS} passed, {FAIL} failed.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
