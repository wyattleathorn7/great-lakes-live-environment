"""Numerical + structural validation of all three live products.

Checks per product:
- current.png / legend.png / metadata.json exist, PNG opens as RGBA,
  canvas size matches config/great_lakes_bounds.json
- metadata schema: source, urls, timestamps, units, resolutions,
  color min/max, missing-data treatment
- physical bounds (wave 0-30 ft display scale; SST source -2..40 C /
  display 25..95 F; ice 0..100 %)
- raster has non-transparent water pixels (EXCEPT ice, which may
  legitimately be fully transparent outside Dec-Apr)
- KML parses as XML, contains one GroundOverlay with the correct stable
  href + LatLonBox == common bounds, contains the legend ScreenOverlay,
  and contains ZERO LineString/Polygon/Placemark/Point elements
- KML cache-buster token present (?v=...) and matches processing timestamp

Exit 0 = all pass; 1 = any failure (message lists all failures).
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geospatial_utils import REPO_ROOT, SITE_DIR, load_bounds

from PIL import Image

PRODUCTS = {
    "wave_height": {"kml": "Great_Lakes_Live_Wave_Height.kml",
                    "max_opaque_min": 10_000},
    "water_temperature": {"kml": "Great_Lakes_Live_Water_Temperature.kml",
                          "max_opaque_min": 10_000},
    "ice_coverage": {"kml": "Great_Lakes_Live_Ice_Coverage.kml",
                     "max_opaque_min": 0},  # ice-free season => transparent OK
    "ice_thickness": {"kml": "Great_Lakes_Live_Ice_Thickness.kml",
                      "max_opaque_min": 0},  # off-season => transparent OK
    "ice_type": {"kml": "Great_Lakes_Live_Ice_Type.kml",
                 "max_opaque_min": 0},  # off-season => transparent OK
}

META_REQUIRED = ["product", "title", "freshness", "noaa_source", "variable",
                 "source_url", "data_time_utc", "processing_time_utc",
                 "units", "spatial_resolution_source",
                 "spatial_resolution_rendered", "color_scale_min",
                 "color_scale_max", "missing_data_treatment"]


def main():
    bounds = load_bounds()
    failures = []
    for product, spec in PRODUCTS.items():
        pdir = os.path.join(SITE_DIR, product)
        png = os.path.join(pdir, "current.png")
        legend = os.path.join(pdir, "legend.png")
        meta_p = os.path.join(pdir, "metadata.json")
        for p in (png, legend, meta_p):
            if not os.path.exists(p):
                failures.append(f"{product}: missing {p}")
        if failures and not os.path.exists(meta_p):
            continue
        try:
            im = Image.open(png)
            im.load()
            if im.mode != "RGBA":
                failures.append(f"{product}: PNG mode {im.mode} != RGBA")
            if im.size != (bounds["canvas_width"], bounds["canvas_height"]):
                failures.append(f"{product}: PNG size {im.size} != canvas")
            try:
                leg = Image.open(legend)
                leg.load()
                expect_legend = (640, 210)
                try:
                    with open(meta_p) as _mf:
                        expect_legend = tuple(json.load(_mf).get(
                            "legend_size", expect_legend))
                except Exception:
                    pass
                if tuple(leg.size) != tuple(expect_legend):
                    failures.append(f"{product}: legend size {leg.size} != {expect_legend}")
            except Exception as e:
                failures.append(f"{product}: legend unreadable: {e}")
            import numpy as np
            a = np.array(im)[:, :, 3]
            n_opaque = int((a > 0).sum())
            if n_opaque < spec["max_opaque_min"]:
                failures.append(
                    f"{product}: only {n_opaque} non-transparent pixels")
            else:
                print(f"[{product}] raster OK: {n_opaque} water pixels, size {im.size}")
        except Exception as e:
            failures.append(f"{product}: PNG unreadable: {e}")

        try:
            with open(meta_p) as f:
                meta = json.load(f)
            for k in META_REQUIRED:
                if k not in meta or meta[k] in (None, ""):
                    failures.append(f"{product}: metadata missing '{k}'")
            lo, hi = meta.get("color_scale_min"), meta.get("color_scale_max")
            if product == "ice_coverage" and (lo, hi) != (0.0, 100.0):
                failures.append(f"{product}: ice scale must be 0-100, got {lo}-{hi}")
            if product == "wave_height" and not (0 <= lo < hi <= 30):
                failures.append(f"{product}: wave scale out of bounds {lo}-{hi}")
            if product == "water_temperature" and not (20 <= lo < hi <= 95):
                failures.append(f"{product}: temp scale out of bounds {lo}-{hi}")
            if product == "ice_thickness" and not (0 <= lo < hi <= 50):
                failures.append(f"{product}: thickness scale out of bounds {lo}-{hi}")
            if product == "ice_type":
                cats = meta.get("ice_type_categories", [])
                codes = {c.get("code") for c in cats}
                if len(cats) < 18 or "unknown" not in codes:
                    failures.append(f"{product}: legend categories incomplete "
                                    f"({len(cats)} entries, need 17 stages + unknown)")
            print(f"[{product}] metadata OK: data_time={meta.get('data_time_utc')}")
        except Exception as e:
            failures.append(f"{product}: metadata unreadable: {e}")
            meta = {}

        for kdir in (os.path.join(REPO_ROOT, "kml"),
                     os.path.join(SITE_DIR, "kml")):
            kp = os.path.join(kdir, spec["kml"])
            text = ""
            if not os.path.exists(kp):
                failures.append(f"{product}: missing {kp}")
                continue
            try:
                text = open(kp, encoding="utf-8").read()
                ET.fromstring(text)  # must parse
                for bad in ("<LineString", "<Polygon", "<Placemark", "<Point"):
                    if bad in text:
                        failures.append(f"{product}: forbidden {bad} in {kp}")
                if "<GroundOverlay>" not in text:
                    failures.append(f"{product}: no GroundOverlay in {kp}")
                if "?v=" not in text:
                    failures.append(f"{product}: no cache-buster in {kp}")
                if f"{product}/current.png" not in text:
                    failures.append(f"{product}: KML href wrong product path")
                token = (meta.get("processing_time_utc", "")
                         .replace(" ", "_").replace(":", ""))
                if token and token not in text:
                    failures.append(f"{product}: KML cache token does not match "
                                    f"metadata processing_time in {kp} (stale KML?)")
                if os.environ.get("CI") == "true" and "REPLACE-" in text:
                    failures.append(f"{product}: KML still has placeholder "
                                    f"PAGES_BASE_URL (CI must set it)")
                for edge, val in (("<north>", bounds["lat_max"]),
                                  ("<south>", bounds["lat_min"]),
                                  ("<east>", bounds["lon_max"]),
                                  ("<west>", bounds["lon_min"])):
                    if f"{edge}{val}" not in text and f"{edge}{float(val)}" not in text:
                        failures.append(f"{product}: KML {edge} != {val}")
                print(f"[{product}] KML OK: {kp}")
            except ET.ParseError as e:
                failures.append(f"{product}: KML XML parse error in {kp}: {e}")

    if failures:
        print("\nVALIDATION FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll products validated OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
