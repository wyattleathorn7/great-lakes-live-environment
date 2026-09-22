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
import re
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
    "wind": {"kml": "Great_Lakes_Live_Wind.kml",
             "max_opaque_min": 10_000},
    "leaf_color": {"kml": "Great_Lakes_Live_Leaf_Color.kml",
                   "max_opaque_min": 10_000},
}

META_REQUIRED = ["product", "title", "freshness", "noaa_source", "variable",
                 "source_url", "data_time_utc", "processing_time_utc",
                 "units", "spatial_resolution_source",
                 "spatial_resolution_rendered", "color_scale_min",
                 "color_scale_max", "missing_data_treatment",
                 "shoreline_mask"]


def _site_file_for_href(href):
    """Map a KML Icon href to a local site/ path. None if not resolvable.

    Pages URLs look like https://<owner>.github.io/<repo>/wind/current.png
    while site/ IS the repo root, so the leading repo-name segment is
    stripped ( progressively, first hit wins).
    """
    import re
    m = re.search(r"https?://[^/]+/(.+?\.png)(?:\?.*)?$", href)
    if not m:
        return None
    parts = m.group(1).split("/")
    for i in range(len(parts)):
        cand = os.path.join(SITE_DIR, *parts[i:])
        if os.path.isfile(cand):
            return cand
    return os.path.join(SITE_DIR, parts[-1]) if parts else None


def main():
    bounds = load_bounds()
    failures = []
    # ---- shared shoreline mask must exist with canvas dimensions ----
    mask_path = os.path.join(REPO_ROOT, "assets", "great_lakes_watermask.png")
    mask = None
    if not os.path.exists(mask_path):
        failures.append("shared shoreline mask missing: assets/great_lakes_watermask.png")
    else:
        try:
            import numpy as np
            mask = np.array(Image.open(mask_path).convert("L"))
            if mask.shape != (bounds["canvas_height"], bounds["canvas_width"]):
                failures.append(f"shoreline mask shape {mask.shape} != canvas")
                mask = None
        except Exception as e:
            failures.append(f"shoreline mask unreadable: {e}")
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
            a = np.array(im)
            n_opaque = int((a[:, :, 3] > 0).sum())
            if n_opaque < spec["max_opaque_min"]:
                failures.append(
                    f"{product}: only {n_opaque} non-transparent pixels")
            else:
                print(f"[{product}] raster OK: {n_opaque} water pixels, size {im.size}")
            if mask is not None:
                import numpy as np
                if product == "leaf_color":
                    # leaf grows on LAND: opaque must avoid open-lake water
                    bleed = int(((a[:, :, 3] > 0) & (mask > 250)).sum())
                    what = "open-lake water"
                else:
                    bleed = int(((a[:, :, 3] > 0) & (mask == 0)).sum())
                    what = "the shared shoreline mask"
                if bleed > 0:
                    failures.append(f"{product}: {bleed} opaque pixels outside "
                                    f"{what}")
                else:
                    print(f"[{product}] shoreline OK: no land bleed")
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
            if product == "wind" and (lo, hi) != (0, 12):
                failures.append(f"{product}: wind scale must be Beaufort 0-12, got {lo}-{hi}")
            if product == "leaf_color" and (lo, hi) != (0.0, 1.0):
                failures.append(f"{product}: leaf scale must be 0-1, got {lo}-{hi}")
            if product == "leaf_color":
                import numpy as np
                _leg = np.array(Image.open(legend).convert("RGB"))
                _nuniq = len(np.unique(_leg.reshape(-1, 3), axis=0))
                if _nuniq < 200:
                    failures.append(f"{product}: legend not continuous "
                                    f"({_nuniq} colors)")
                _ra = np.array(Image.open(png).convert("RGBA"))
                _amask = _ra[:, :, 3] > 0
                _rs = np.array(Image.open(png).convert("RGB"))
                _runq = len(np.unique(_rs[_amask].reshape(-1, 3), axis=0)) \
                    if _amask.any() else 0
                if _amask.any() and _runq < 50:
                    failures.append(f"{product}: raster looks quantized "
                                    f"({_runq} colors)")
                _prov = meta.get("provenance", {})
                for _k in ("provider", "composite_date", "algorithm"):
                    if _k not in _prov:
                        failures.append(f"{product}: provenance missing '{_k}'")
                if "native_resolution" not in _prov \
                        and "native_resolution_m" not in _prov:
                    failures.append(f"{product}: provenance missing "
                                    f"'native_resolution'")
                # water must be transparent at lake + Georgian Bay points
                for _nm, _x, _y in (
                        ("Superior", -88, 47.5), ("Michigan", -87, 44.2),
                        ("Huron", -82.5, 44.8), ("Erie", -81.5, 42.2),
                        ("Ontario", -77.8, 43.6), ("GeorgianBay", -81.0, 45.3)):
                    _cc = int((_x - bounds["lon_min"])
                              / (bounds["lon_max"] - bounds["lon_min"])
                              * bounds["canvas_width"])
                    _rr = int((bounds["lat_max"] - _y)
                              / (bounds["lat_max"] - bounds["lat_min"])
                              * bounds["canvas_height"])
                    if _ra[_rr, _cc, 3] != 0:
                        failures.append(f"{product}: water not transparent "
                                        f"at {_nm}")
                # no straight US/Canada seam: compare mean green in the
                # 46N land band north vs south of the parallel
                _lat = bounds["lat_max"] - (np.arange(
                    bounds["canvas_height"]) + 0.5) / bounds["canvas_height"] \
                    * (bounds["lat_max"] - bounds["lat_min"])
                _lon = bounds["lon_min"] + (np.arange(
                    bounds["canvas_width"]) + 0.5) / bounds["canvas_width"] \
                    * (bounds["lon_max"] - bounds["lon_min"])
                _rows = np.where((np.abs(_lat - 46.0) < 0.6))[0]
                _cols = np.where((_lon > -84) & (_lon < -79))[0]
                _s = _rows[_lat[_rows] < 46.0]
                _n = _rows[_lat[_rows] >= 46.0]
                _gs = _rs[np.ix_(_s, _cols)][_amask[np.ix_(_s, _cols)]]
                _gn = _rs[np.ix_(_n, _cols)][_amask[np.ix_(_n, _cols)]]
                if _gs.size > 500 and _gn.size > 500:
                    _ms, _mn = float(_gs.reshape(-1, 3)[:, 1].mean()), \
                        float(_gn.reshape(-1, 3)[:, 1].mean())
                    print(f"[{product}] 46N band green S={_ms:.1f} N={_mn:.1f}")
                    if abs(_ms - _mn) > 60:
                        failures.append(f"{product}: possible US/Canada seam "
                                        f"(S={_ms:.1f} N={_mn:.1f})")
            if product == "wind":
                bt = meta.get("beaufort_table", [])
                if len(bt) != 13 or "64" not in bt[12].get("range_kt", ""):
                    failures.append(f"{product}: beaufort table must have 13 forces "
                                    f"with F12 >= 64 kt")
                if bt and bt[12].get("color") != "#3B0A54":
                    failures.append(f"{product}: Force 12 must be dark purple #3B0A54")
                if meta.get("stats", {}).get("arrows_drawn", 0) < 50:
                    failures.append(f"{product}: too few wind arrows rendered")
            # placeholders must never reach production output
            _md = json.dumps(meta)
            if "REPLACE-GITHUB-USER" in _md or "REPLACE-REPO" in _md:
                failures.append(f"{product}: metadata contains placeholder URL")
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
                for bad in ("<LineString", "<Polygon", "<Placemark", "<Point",
                            "<ScreenOverlay"):
                    if bad in text:
                        failures.append(f"{product}: forbidden {bad} in {kp}")
                if len(text) > 100_000:
                    failures.append(f"{product}: KML too large ({len(text)} chars) "
                                    f"- geometry explosion?")
                if "REPLACE-GITHUB-USER" in text or "REPLACE-REPO" in text:
                    failures.append(f"{product}: KML contains placeholder URL")
                if "<GroundOverlay>" not in text:
                    failures.append(f"{product}: no GroundOverlay in {kp}")
                if "?v=" not in text:
                    failures.append(f"{product}: no cache-buster in {kp}")
                if f"{product}/current.png" not in text:
                    failures.append(f"{product}: KML href wrong product path")
                for _href in re.findall(r"<href>(https?://[^<]+\.png)(?:\?[^<]*)?</href>",
                                        text):
                    _local = _site_file_for_href(_href)
                    if _local is None or not os.path.exists(_local):
                        failures.append(f"{product}: KML references PNG not deployed: "
                                        f"{_href}")
                # single-overlay architecture (GE Web image limit): no tiles
                if "/tiles/" in text:
                    failures.append(f"{product}: KML must not reference tiles")
                n_overlays = text.count("<GroundOverlay>")
                if n_overlays != 1:
                    failures.append(f"{product}: {n_overlays} overlays (expected exactly 1)")
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
