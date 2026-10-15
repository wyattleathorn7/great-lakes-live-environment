"""Pipeline F-NA — LIVE WIND, NORTH AMERICA (independent).

Precedence (explicit, one source owns each pixel):
  GLWU v2.1 surface U/V (Great Lakes water) > HRRR 3km 10m U/V (its
  domain, GLWU n/a) > RRFS-NA 10m U/V (elsewhere) > ECCC GDPS 10m winds
  (verified gaps only) > transparent. Missing components are never
  treated as calm. All sources converted m/s -> kt (x1.94384) BEFORE
  Beaufort 0-12; arrows point TOWARD motion; one arrow field rendered
  into the raster at consistent NA-canvas density.

Exit codes: 0 updated; 2 source failure (previous kept); 1 error.
"""

import datetime as dt
import json
import math
import os
import sys
import traceback
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from na_bounds import load_na_bounds
from na_common import (apply_coverage_mask, bin_points_to_canvas,
                       feather_merge, fetch_http, fill_pinholes,
                       gate_field_to_coverage, gdps_wcs_grid,
                       sanitize_model_vals, smooth_field)
from beaufort import FORCE_COLORS, MS_TO_KT, force_from_kt, force_name, force_range_text
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs)
from geospatial_utils import (REPO_ROOT, SITE_DIR, base_metadata, now_det_str,
                              promote_stage, read_state, save_png,
                              source_token, stage_dir, write_metadata,
                              write_state)

PRODUCT = "wind_na"
CONFIG_NA = json.load(open(os.path.join(REPO_ROOT, "config", "wind_na.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "North_America_Live_Wind.kml"
OVERLAY_NAME = "\U0001f4a8 LIVE WIND"
SKIP_NOTE = "Turn on/off independently of all other layers."

# Strategic label anchors (lon, lat): the ORIGINAL Great Lakes fixed-city
# anchors verbatim from build_wind.CITY_ANCHORS (basin values sit exactly
# where the original layer puts them), plus major metros across the wider
# NA footprint in the SAME style/seps/font. No fixed lake-value anchors:
# Canadian lakes get values from the SAME water extras engine as the basin
# water (front-first + shuffled fill), never a separate sparse set.
from build_wind import CITY_ANCHORS as _GL_CITY_ANCHORS
GL_NA_ANCHORS = [(lon, lat) for (_name, lon, lat) in _GL_CITY_ANCHORS]
METRO_ANCHORS = [
    (-74.0, 40.7), (-95.4, 29.8),
    (-99.1, 19.4), (-103.3, 20.7), (-84.1, 9.9), (-82.4, 23.1),
    (-149.9, 61.2), (-113.5, 53.5), (-97.1, 49.9), (-75.7, 45.4),
    (-52.7, 47.6), (-157.8, 21.3), (-61.2, 14.6), (-90.5, 14.6),
]
NA_CITY_ANCHORS = GL_NA_ANCHORS + METRO_ANCHORS


def paint_na_wind_labels(rgba, kt_field, bounds, label_step=32,
                         font_size=7, max_labels=700, extra_thresh_kt=4.0):
    """Exact-copy of the Great Lakes split label engine (build_wind.
    paint_wind_labels), ported to the NA canvas.

    - OVER WATER (Great Lakes + Canadian lakes via lakemask): data-seeded
      jittered step-32 candidates, front cells first (grad >= thresh,
      sep 30), then shuffled spaced fill (sep 48). Same math as the
      original temperature-generator emulation.
    - OVER LAND: fixed strategic anchors (basin cities + NA metros) first
      (sep 30), then high-change extras (sep 30 fronts, sep 48 fill).
    Same white-mono + dark-stroke font/size, same alpha guards. Canadian
    lakes are NOT a separate fixed set: they ride the water extras, so
    the basin and the north look and act identically. Returns
    (rgba, n_strategic, n_fill)."""
    from build_wind import _draw_wind_label, _wind_label_font
    from na_common import load_lakemask
    img = Image.fromarray(rgba)
    d = ImageDraw.Draw(img)
    font = _wind_label_font(font_size)
    H, W = kt_field.shape
    ok = np.isfinite(kt_field)
    if not ok.any():
        del d
        return np.array(img), 0, 0
    try:
        _lake = (load_lakemask() >= 0.5)
        if _lake.shape != (H, W):
            _lake = np.zeros((H, W), dtype=bool)
        is_water = _lake & ok
    except Exception:
        is_water = np.zeros((H, W), dtype=bool)
    fin = kt_field[ok]
    seed = int(abs(float(fin.mean())) * 1000 + float(fin.std()) * 97) % (2 ** 32 - 1)
    rng = np.random.default_rng(seed)
    grad = np.zeros(kt_field.shape)
    for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
        shifted = np.roll(kt_field, shift=(-dr, -dc), axis=(0, 1))
        diff = np.abs(kt_field - shifted)
        both = ok & np.isfinite(shifted)
        grad = np.maximum(grad, np.where(both, diff, 0.0))
    step = label_step  # density-matched per degree (16 on NA ≈ 32 in basin)
    placed = []
    n_strat = n_fill = 0

    def _collides(x0, y0, x1, y1, pad=3):
        for (a0, b0, a1, b1) in placed:
            if not (x1 + pad < a0 or x0 - pad > a1 or y1 + pad < b0 or y0 - pad > b1):
                return True
        return False

    def _put(cx, cy, text, sep):
        tw = d.textlength(text, font=font)
        th = font_size + 4
        x = cx - tw / 2
        y = cy - th / 2
        if not (4 <= x < W - tw - 4 and 4 <= y < H - th - 4):
            return False
        if _collides(x, y, x + tw, y + th):
            return False
        for (a0, b0, a1, b1) in placed:
            acx, acy = (a0 + a1) / 2, (b0 + b1) / 2
            if math.hypot(acx - cx, acy - cy) < sep:
                return False
        for px, py in ((cx, cy), (x + 2, y + 2), (x + tw - 2, y + 2)):
            ix, iy = int(np.clip(round(px), 0, W - 1)), int(np.clip(round(py), 0, H - 1))
            if rgba[iy, ix, 3] == 0:
                return False
        if not np.isfinite(kt_field[int(np.clip(round(cy), 0, H - 1)),
                                    int(np.clip(round(cx), 0, W - 1))]):
            return False
        _draw_wind_label(d, x, y, text, font)
        placed.append((x, y, x + tw, y + th))
        return True

    def _snap_valid(r, c, water_wanted, rad=7):
        for rad_i in range(rad + 1):
            for dr in range(-rad_i, rad_i + 1):
                for dc in (-rad_i, rad_i):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and ok[rr, cc] \
                            and (is_water[rr, cc] if water_wanted else not is_water[rr, cc]) \
                            and rgba[rr, cc, 3] > 0:
                        return rr, cc
            for dc in range(-rad_i + 1, rad_i):
                for dr in (-rad_i, rad_i):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and ok[rr, cc] \
                            and (is_water[rr, cc] if water_wanted else not is_water[rr, cc]) \
                            and rgba[rr, cc, 3] > 0:
                        return rr, cc
        return None

    # ---- PHASE 1: over-water extras (basin + Canadian lakes, same engine)
    wcand = []
    for r0 in range(0, H, step):
        for c0 in range(0, W, step):
            r = r0 + step / 2.0 + (rng.random() - 0.5) * step * 0.9
            c = c0 + step / 2.0 + (rng.random() - 0.5) * step * 0.9
            r, c = int(round(r)), int(round(c))
            if not (8 <= r < H - 8 and 8 <= c < W - 8):
                continue
            if not ok[r, c] or rgba[r, c, 3] == 0:
                continue
            if not is_water[r, c]:
                continue
            wcand.append((grad[r, c], rng.random(), r, c))
    wcand.sort(key=lambda t: (-t[0], t[1]))
    for g, _, r, c in wcand:
        if n_strat + n_fill >= max_labels:
            break
        if g < extra_thresh_kt:
            break
        v = float(kt_field[r, c])
        if not (0.0 <= v <= 120.0):
            continue
        if _put(c, r - 12, f"{v:.0f}kt", sep=30):
            n_strat += 1
    wrest = [t for t in wcand if t[0] < extra_thresh_kt]
    rng.shuffle(wrest)
    for _, _, r, c in wrest:
        if n_strat + n_fill >= max_labels:
            break
        v = float(kt_field[r, c])
        if not (0.0 <= v <= 120.0):
            continue
        if _put(c, r - 12, f"{v:.0f}kt", sep=48):
            n_fill += 1

    # ---- PHASE 2a: fixed strategic anchors over land (basin + metros)
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    for lon, lat in list(GL_NA_ANCHORS) + list(METRO_ANCHORS):
        if n_strat + n_fill >= max_labels:
            break
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        c0 = int((lon - lon_min) / (lon_max - lon_min) * W)
        r0 = int((lat_max - lat) / (lat_max - lat_min) * H)
        if not (8 <= r0 < H - 8 and 8 <= c0 < W - 8):
            continue
        hit = _snap_valid(r0, c0, water_wanted=False)
        if hit is None:
            continue
        r, c = hit
        v = float(kt_field[r, c])
        if not (0.0 <= v <= 120.0):
            continue
        if _put(c, r - 12, f"{v:.0f}kt", sep=30):
            n_strat += 1

    # ---- PHASE 2b: land high-change extras (appear/disappear with wind)
    lcand = []
    for r0 in range(0, H, step):
        for c0 in range(0, W, step):
            r = r0 + step / 2.0 + (rng.random() - 0.5) * step * 0.9
            c = c0 + step / 2.0 + (rng.random() - 0.5) * step * 0.9
            r, c = int(round(r)), int(round(c))
            if not (8 <= r < H - 8 and 8 <= c < W - 8):
                continue
            if not ok[r, c] or rgba[r, c, 3] == 0:
                continue
            if is_water[r, c]:
                continue
            lcand.append((grad[r, c], rng.random(), r, c))
    lcand.sort(key=lambda t: (-t[0], t[1]))
    for g, _, r, c in lcand:
        if n_strat + n_fill >= max_labels:
            break
        if g < extra_thresh_kt:
            break
        v = float(kt_field[r, c])
        if not (0.0 <= v <= 120.0):
            continue
        if _put(c, r - 12, f"{v:.0f}kt", sep=30):
            n_strat += 1
    lrest = [t for t in lcand if t[0] < extra_thresh_kt]
    rng.shuffle(lrest)
    for _, _, r, c in lrest:
        if n_strat + n_fill >= max_labels:
            break
        v = float(kt_field[r, c])
        if not (0.0 <= v <= 120.0):
            continue
        if _put(c, r - 12, f"{v:.0f}kt", sep=48):
            n_fill += 1
    del d
    return np.array(img), n_strat, n_fill


def _read_uv_grib(path):
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    uv = {}
    with open(path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                sn = codes_get(h, "shortName")
                if sn in ("10u", "u") and "U" not in uv:
                    uv["U"] = (codes_get_values(h).astype(float),
                               codes_get_array(h, "latitudes").astype(float),
                               ((codes_get_array(h, "longitudes").astype(float) + 180) % 360) - 180)
                elif sn in ("10v", "v") and "V" not in uv:
                    uv["V"] = (codes_get_values(h).astype(float),
                               codes_get_array(h, "latitudes").astype(float),
                               ((codes_get_array(h, "longitudes").astype(float) + 180) % 360) - 180)
            finally:
                codes_release(h)
    if "U" not in uv or "V" not in uv:
        raise ValueError(f"U/V pair incomplete in {path}")
    return uv["U"], uv["V"]


def hrrr_uv():
    import hrrr
    base, dd, cc = hrrr.latest_cycle()
    dest = os.path.join(RAW_DIR, "na_hrrr_uv.grib2")
    hrrr.fetch_messages(base, dest, [("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")])
    got = hrrr.read_messages(dest, {"UGRD": 1, "VGRD": 1})
    (uv, ulat, ulon, _, _) = got["UGRD"]
    (vv, vlat, vlon, _, _) = got["VGRD"]
    ulat, ulon = np.asarray(ulat, float), ((np.asarray(ulon, float) + 180) % 360) - 180
    return (f"hrrr {dd}/{cc}", np.asarray(uv, float), np.asarray(vv, float), ulat, ulon)


def rrfs_uv_na():
    from na_common import eccodes_read_idx_ranges
    now = dt.datetime.now(dt.timezone.utc)
    for back in range(0, 8):
        t = now - dt.timedelta(hours=back)
        dd = t.strftime("rrfs.%Y%m%d")
        for base in ("https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod",
                     "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/para"):
            fn = "rrfs.t00z.2dfld.13km.f000.na.grib2"
            try:
                idx = fetch_http(f"{base}/{dd}/00/{fn}.idx", timeout=30).decode("utf-8", errors="replace")
            except Exception:
                continue
            try:
                r = eccodes_read_idx_ranges(idx, [("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")])
            except ValueError:
                continue
            dest = os.path.join(RAW_DIR, "na_rrfs_uv.grib2")
            with open(dest + ".part", "wb") as f:
                for short in ("UGRD", "VGRD"):
                    a, b = r[short]
                    req = urllib.request.Request(
                        f"{base}/{dd}/00/{fn}",
                        headers={"User-Agent": "great-lakes-live-environment/1.0",
                                 "Range": f"bytes={a}-{b-1}"})
                    with urllib.request.urlopen(req, timeout=300) as rr:
                        f.write(rr.read())
            os.replace(dest + ".part", dest)
            (uv, ulat, ulon), (vv, vlat, vlon) = _read_uv_grib(dest)
            return f"rrfs-na {dd}/00", uv, vv, ulat, ulon
    raise RuntimeError("no RRFS-NA 10m U/V analysis found")


def glwu_uv():
    """GLWU Great Lakes water U/V (top precedence in its domain).

    Same newest-available-cycle detection as the Great Lakes wind layer
    (UGRD surface :anl: stamp); U/V shortNames are u/v at step 0.
    """
    from build_wave_height import newest_available_cycle
    pick = newest_available_cycle(dt.datetime.now(dt.timezone.utc), "UGRD", label=PRODUCT)
    if pick is None:
        raise RuntimeError("no GLWU analysis cycle available")
    url, dd, cc, stamp = pick
    dest = os.path.join(RAW_DIR, "na_glwu_uv.grib2")
    req0 = urllib.request.Request(url + ".idx", headers={"User-Agent": "great-lakes-live-environment/1.0"})
    with urllib.request.urlopen(req0, timeout=60) as r:
        lines = r.read().decode("utf-8", errors="replace").splitlines()
    # Same match rule as the Great Lakes wind layer: UGRD/VGRD surface
    # :anl: messages (never a forecast lead).
    ranges = {}
    for short in ("UGRD", "VGRD"):
        for l in lines:
            p = l.split(":")
            if len(p) > 5 and p[3] == short and "surface" in l and ":anl:" in l:
                a = int(p[1])
                j = lines.index(l) + 1
                b = int(lines[j].split(":")[1])
                if b > a:
                    ranges[short] = (a, b)
                    break
        if short not in ranges:
            raise ValueError(f"GLWU {short}/surface anl message not in .idx")
    with open(dest + ".part", "wb") as f:
        for short in ("UGRD", "VGRD"):
            a, b = ranges[short]
            req = urllib.request.Request(url, headers={"User-Agent": "great-lakes-live-environment/1.0",
                                                       "Range": f"bytes={a}-{b-1}"})
            with urllib.request.urlopen(req, timeout=300) as rr:
                f.write(rr.read())
    os.replace(dest + ".part", dest)
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    uv = {}
    with open(dest, "rb") as fh:
        while True:
            h = codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                sn = codes_get(h, "shortName")
                if sn in ("10u", "u") and "U" not in uv:
                    uv["U"] = (codes_get_values(h).astype(float),
                               codes_get_array(h, "latitudes").astype(float),
                               ((codes_get_array(h, "longitudes").astype(float) + 180) % 360) - 180)
                elif sn in ("10v", "v") and "V" not in uv:
                    uv["V"] = (codes_get_values(h).astype(float),
                               codes_get_array(h, "latitudes").astype(float),
                               ((codes_get_array(h, "longitudes").astype(float) + 180) % 360) - 180)
            finally:
                codes_release(h)
    if "U" not in uv or "V" not in uv:
        raise ValueError("GLWU U/V pair incomplete")
    return f"glwu {dd}/{cc}", uv["U"][0], uv["V"][0], uv["U"][1], uv["U"][2]


def _hex(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def force_rgb(force):
    return _hex(FORCE_COLORS.get(int(force), "#808080"))


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    bounds = load_na_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    try:
        return _build(bounds, W, H)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] FAILED: {type(e).__name__}: {e}. Keeping previous.")
        return 2


def _build(bounds, W, H):
    from geospatial_utils import draw_category_legend
    # Fresh checkouts (CI runners) lack the gitignored output/raw dir —
    # every GRIB/NetCDF fetch below writes there.
    os.makedirs(RAW_DIR, exist_ok=True)
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    os.makedirs(stage_prod, exist_ok=True)
    # Global canvas (former extended200 variant's replacement): worldwide
    # 3600x1800, 0.10 deg/px — same grid as aurora_global. Basin + NA
    # variants stay on the NA canvas untouched.
    from na_bounds import load_global_bounds
    gbounds = load_global_bounds()
    GW, GH = gbounds["canvas_width"], gbounds["canvas_height"]
    prov = {}
    ulayers, vlayers = [], []  # (grid, joint-valid) in precedence order
    gulayers, gvlayers = [], []  # same precedence on the global canvas
    raw_sources = []  # (label, u, v, la, lo, smooth_coarse) for global re-bin

    def _bin_uv(label, u, v, la, lo, target_bounds, smooth_coarse):
        from na_common import smooth_field
        uu = sanitize_model_vals(u, -150.0, 150.0)
        vv = sanitize_model_vals(v, -150.0, 150.0)
        gu, _ = bin_points_to_canvas(uu, la, lo, target_bounds)
        gv, _ = bin_points_to_canvas(vv, la, lo, target_bounds)
        gu = fill_pinholes(gu, radius=3)
        gv = fill_pinholes(gv, radius=3)
        if smooth_coarse:
            gu = smooth_field(gu, size=5, passes=2)
            gv = smooth_field(gv, size=5, passes=2)
        joint = np.isfinite(gu) & np.isfinite(gv)
        return gu, gv, joint

    def place(label, u, v, la, lo, smooth_coarse=False):
        gu, gv, joint = _bin_uv(label, u, v, la, lo, bounds, smooth_coarse)
        ulayers.append((gu, joint))
        vlayers.append((gv, joint))
        ggu, ggv, gjoint = _bin_uv(label, u, v, la, lo, gbounds, smooth_coarse)
        gulayers.append((ggu, gjoint))
        gvlayers.append((ggv, gjoint))
        raw_sources.append((label, u, v, la, lo, smooth_coarse))
        prov[label] = f"n={int(joint.sum())}/g={int(gjoint.sum())}"

    # Precedence order: GLWU > HRRR > RRFS.
    try:
        label, u, v, la, lo = glwu_uv()
        place(label, u, v, la, lo)
    except Exception as e:
        prov["glwu"] = f"unavailable: {type(e).__name__}"
    try:
        label, u, v, la, lo = hrrr_uv()
        place(label, u, v, la, lo)
    except Exception as e:
        prov["hrrr"] = f"unavailable: {type(e).__name__}"
    try:
        label, u, v, la, lo = rrfs_uv_na()
        place(label, u, v, la, lo, smooth_coarse=True)
    except Exception as e:
        prov["rrfs"] = f"unavailable: {type(e).__name__}"
    # GDPS fallback (ECCC GDPS 15km WindSpeed + WindDir 10m via WCS) ONLY
    # where GLWU/HRRR/RRFS have no verified valid vector. Worldwide model:
    # the same WCS GetCoverage is fetched on BOTH canvases, so GDPS fills
    # the whole world on the global variant, not just NA gaps.
    # WindDir is the meteorological FROM direction, so toward-motion
    # U = -spd*sin(dir), V = -spd*cos(dir).
    def _gdps_uv(target_bounds, tag):
        arr_s, la_s, lo_s, slat, slon = gdps_wcs_grid("GDPS_15km_WindSpeed_10m", target_bounds, RAW_DIR)
        arr_d, _, _, _, _ = gdps_wcs_grid("GDPS_15km_WindDir_10m", target_bounds, RAW_DIR)
        spd = sanitize_model_vals(np.asarray(arr_s[0], dtype=float), 0.0, 120.0)
        wdir = np.asarray(arr_d[0], dtype=float)
        wdir = np.where(np.isfinite(wdir), wdir % 360.0, np.nan)
        rad = np.deg2rad(wdir)
        from na_common import resample_wcs_to_canvas as _bilin
        from na_common import smooth_field as _smooth
        gu = _bilin(np.where(np.isfinite(spd), -spd * np.sin(rad), np.nan), slat, slon, target_bounds)
        gv = _bilin(np.where(np.isfinite(spd), -spd * np.cos(rad), np.nan), slat, slon, target_bounds)
        gu = _smooth(fill_pinholes(gu, radius=3), size=5, passes=2)
        gv = _smooth(fill_pinholes(gv, radius=3), size=5, passes=2)
        joint = np.isfinite(gu) & np.isfinite(gv)
        return gu, gv, joint
    try:
        gu, gv, joint = _gdps_uv(bounds, RAW_DIR)
        ulayers.append((gu, joint))
        vlayers.append((gv, joint))
        ggu, ggv, gjoint = _gdps_uv(gbounds, RAW_DIR)
        gulayers.append((ggu, gjoint))
        gvlayers.append((ggv, gjoint))
        prov["gdps"] = f"GDPS_15km_Wind_10m n={int(joint.sum())}/g={int(gjoint.sum())}"
    except Exception as e:
        prov["gdps"] = f"unavailable: {type(e).__name__}: {e}"
    if not ulayers:
        raise RuntimeError(f"no wind components from any source ({prov})")

    # Exact-copy precedence (same as the original: first-valid wins):
    # GLWU over water, then HRRR, then RRFS-NA, then GDPS gaps, with a
    # narrow cross-fade where domains overlap so rectangular source edges
    # (e.g. the HRRR boundary) never print as visible seams.
    # Coarse 13-15 km layers arrive pre-smoothed above (dots/blocks would
    # otherwise print); GLWU/HRRR render raw like the original.
    U, _ = feather_merge(ulayers, feather_px=12)
    V, _ = feather_merge(vlayers, feather_px=12)
    UG, _ = feather_merge(gulayers, feather_px=6)
    VG, _ = feather_merge(gvlayers, feather_px=6)
    from na_common import gate_field_to_coverage
    from build_kml import (COVERAGE_DIR_SUFFIX, COVERAGE_LABELS,
                           COVERAGE_DISPLAY, coverage_selector_html)
    default_cover = CONFIG_NA.get("coverage_mode", "global")
    if default_cover == "extended200":
        default_cover = "global"
    variant_dirs = {m: f"{PRODUCT}_{COVERAGE_DIR_SUFFIX[m]}"
                    for m in COVERAGE_DISPLAY}
    variant_titles = {m: f"{CONFIG_NA['title']} — {COVERAGE_LABELS[m]}"
                      for m in COVERAGE_DISPLAY}
    # Per-variant canvas: basin stays on the NA canvas; landonly is now
    # Continental Coverage on the worldwide canvas; global renders
    # worldwide (no mask, no lake cut beyond NA lakes).
    VARIANT_BOUNDS = {"basin": bounds, "landonly": gbounds, "global": gbounds,
                      "extended200": gbounds}
    variant_titles = {m: f"{CONFIG_NA['title']} \u2014 {COVERAGE_LABELS[m]}"
                      for m in COVERAGE_DISPLAY}

    # Original Beaufort key rows, verbatim (same colors, same labels).
    key_rows = [(FORCE_COLORS[f], f"F{f} — {force_name(f)} ({force_range_text(f)})")
                for f in range(13)]
    data_time = now_det_str()
    scale_html = ("WIND — BEAUFORT SCALE (colors follow force, dark purple = "
                  "Force 12 only). Speed m/s &rarr; kt (x1.94384) &rarr; WMO Beaufort 0-12. "
                  "Arrows point TOWARD motion; calm (&lt;0.5 m/s) draws no arrow. "
                  "Same force always shows the same color.")
    src_id = f"na-wind-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M}"
    token = source_token(src_id + json.dumps(prov, sort_keys=True))

    from build_wind import WIND_ALPHA_LAND, WIND_ALPHA_WATER, _vivid_for_land
    from build_wind import paint_wind_streamlets
    from na_common import apply_coverage_mask, load_lakemask

    def _render_variant(mode):
        """Gate the merged field to one coverage, paint the original
        finish (exact Beaufort colors, dual opacity, streamlets, kt
        labels). Returns (rgba, kt_gated, stats)."""
        if mode == "global":
            return _render_global()
        if mode == "landonly":
            return _render_continental()
        Ug = gate_field_to_coverage(U, mode)
        Vg = gate_field_to_coverage(V, mode)
        spd_m = np.hypot(Ug, Vg)
        kt_m = spd_m * MS_TO_KT
        m_m = np.isfinite(kt_m)
        lake_m = (load_lakemask() >= 0.5) & m_m
        forces_m = np.array([force_from_kt(x) for x in kt_m[m_m]])
        pal_m = np.array([force_rgb(f) for f in forces_m], dtype=np.uint8)
        vivid_m = np.array([_vivid_for_land(FORCE_COLORS[f]) for f in forces_m],
                           dtype=np.uint8)
        rgba_m = np.zeros((H, W, 4), dtype=np.uint8)
        rgba_m[m_m, :3] = np.where(lake_m[m_m][:, None], pal_m, vivid_m)
        rgba_m[m_m, 3] = np.where(lake_m[m_m], WIND_ALPHA_WATER, WIND_ALPHA_LAND)
        rgba_m, n_stream, n_heads = paint_wind_streamlets(
            rgba_m, np.where(m_m, Ug, np.nan), np.where(m_m, Vg, np.nan),
            np.where(m_m, kt_m, np.nan),
            seed_step=12, ds=3.0, max_steps=40, min_speed=0.5)
        rgba_m, n_strat, n_fill = paint_na_wind_labels(
            rgba_m, np.where(m_m, kt_m, np.nan), bounds,
            label_step=16, font_size=7, max_labels=1500, extra_thresh_kt=4.0)
        rgba_m = apply_coverage_mask(rgba_m, mode)
        return rgba_m, kt_m, {"valid_pixels": int(m_m.sum()),
                              "streamlets": n_stream, "heads": n_heads,
                              "labels": n_strat, "extra_labels": n_fill}

    def _render_global():
        """Worldwide variant on the 3600x1800 global canvas: same
        Beaufort colors + streamlet/label engines, density-matched per
        degree (half the pixel steps at 0.10 deg/px). Dual opacity is
        EXACT: the NA lake mask pooled into the global grid, so Great
        Lakes + Canadian lakes keep the water finish and everything
        else keeps vivid land twins — identical to basin/NA where the
        canvases overlap."""
        from build_wind import (WIND_ALPHA_LAND, WIND_ALPHA_WATER,
                                _vivid_for_land)
        from na_common import load_global_lakemask
        spd_g = np.hypot(UG, VG)
        kt_g = spd_g * MS_TO_KT
        m_g = np.isfinite(kt_g)
        lake_g = load_global_lakemask() & m_g
        forces_g = np.array([force_from_kt(x) for x in kt_g[m_g]])
        pal_g = np.array([force_rgb(f) for f in forces_g], dtype=np.uint8)
        vivid_g = np.array([_vivid_for_land(FORCE_COLORS[f]) for f in forces_g],
                           dtype=np.uint8)
        rgba_g = np.zeros((GH, GW, 4), dtype=np.uint8)
        rgba_g[m_g, :3] = np.where(lake_g[m_g][:, None], pal_g, vivid_g)
        rgba_g[m_g, 3] = np.where(lake_g[m_g], WIND_ALPHA_WATER,
                                  WIND_ALPHA_LAND)
        rgba_g, n_stream, n_heads = paint_wind_streamlets(
            rgba_g, np.where(m_g, UG, np.nan), np.where(m_g, VG, np.nan),
            np.where(m_g, kt_g, np.nan),
            seed_step=6, ds=3.0, max_steps=40, min_speed=0.5)
        rgba_g, n_strat, n_fill = paint_na_wind_labels(
            rgba_g, np.where(m_g, kt_g, np.nan), gbounds,
            label_step=8, font_size=7, max_labels=1500, extra_thresh_kt=4.0)
        rgba_g = apply_coverage_mask(rgba_g, "global")
        return rgba_g, kt_g, {"valid_pixels": int(m_g.sum()),
                              "streamlets": n_stream, "heads": n_heads,
                              "labels": n_strat, "extra_labels": n_fill}

    def _render_continental():
        """Continental Coverage on the worldwide canvas: same Beaufort
        colors/streamlets/labels as global, but gated to crisp worldwide
        land (incl. Antarctica, ocean/seas cut)."""
        from build_wind import (WIND_ALPHA_LAND, WIND_ALPHA_WATER,
                                _vivid_for_land)
        from na_common import (gate_field_to_coverage, load_global_lakemask)
        Ug_c = gate_field_to_coverage(UG, "landonly")
        Vg_c = gate_field_to_coverage(VG, "landonly")
        spd_g = np.hypot(Ug_c, Vg_c)
        kt_g = spd_g * MS_TO_KT
        m_g = np.isfinite(kt_g)
        lake_g = load_global_lakemask() & m_g
        forces_g = np.array([force_from_kt(x) for x in kt_g[m_g]])
        pal_g = np.array([force_rgb(f) for f in forces_g], dtype=np.uint8)
        vivid_g = np.array([_vivid_for_land(FORCE_COLORS[f]) for f in forces_g],
                           dtype=np.uint8)
        rgba_g = np.zeros((GH, GW, 4), dtype=np.uint8)
        rgba_g[m_g, :3] = np.where(lake_g[m_g][:, None], pal_g, vivid_g)
        rgba_g[m_g, 3] = np.where(lake_g[m_g], WIND_ALPHA_WATER,
                                  WIND_ALPHA_LAND)
        rgba_g, n_stream, n_heads = paint_wind_streamlets(
            rgba_g, np.where(m_g, Ug_c, np.nan), np.where(m_g, Vg_c, np.nan),
            np.where(m_g, kt_g, np.nan),
            seed_step=6, ds=3.0, max_steps=40, min_speed=0.5)
        rgba_g, n_strat, n_fill = paint_na_wind_labels(
            rgba_g, np.where(m_g, kt_g, np.nan), gbounds,
            label_step=8, font_size=7, max_labels=1500, extra_thresh_kt=4.0)
        rgba_g = apply_coverage_mask(rgba_g, "landonly")
        return rgba_g, kt_g, {"valid_pixels": int(m_g.sum()),
                              "streamlets": n_stream, "heads": n_heads,
                              "labels": n_strat, "extra_labels": n_fill}

    # Per-coverage rasters: the primary live folder references these
    # PNGs as its three hideable overlays, so every coverage is painted
    # even though only one file is ever added.
    variant_rgba = {}
    variant_kt = {}
    variant_stats = {}
    for mode in COVERAGE_DISPLAY:
        rgba_m, kt_m, st = _render_variant(mode)
        variant_rgba[mode] = rgba_m
        variant_kt[mode] = kt_m
        variant_stats[mode] = st
        vprod, vtitle = variant_dirs[mode], variant_titles[mode]
        vdir = os.path.join(stage, "site", vprod)
        os.makedirs(vdir, exist_ok=True)
        save_png(rgba_m, os.path.join(vdir, "current.png"))
        print(f"[{PRODUCT}] variant {mode}: streamlets={st['streamlets']}/"
              f"{st['heads']} labels={st['labels']}+{st['extra_labels']} "
              f"px={st['valid_pixels']}")
        vmeta = base_metadata(
            vprod, vtitle, CONFIG_NA["freshness_label"],
            CONFIG_NA["source_name"], CONFIG_NA["source_url"],
            CONFIG_NA["variable"],
            data_time_utc=data_time,
            source_last_modified_utc=json.dumps(prov)[:300],
            units="kt (Beaufort 0-12)",
            source_resolution="GLWU ~2.5km; HRRR 3km; RRFS 13km NA",
            color_min=0, color_max=12, color_units="Beaufort force",
            missing_data_treatment="GLWU > HRRR > RRFS > GDPS(gaps) > transparent; missing never calm. "
                "Dual opacity (original finish): lakes at water alpha 205 with exact Beaufort colors; "
                "other land at alpha 140 with saturation-boosted twins (single legend).")
        vmeta["legend_scale_html"] = scale_html
        vmeta["beaufort_table"] = [
            {"force": f, "description": force_name(f),
             "range_kt": force_range_text(f), "color": FORCE_COLORS[f]}
            for f in range(13)]
        vmeta["source_id"] = src_id
        vmeta["source_version"] = token
        vmeta["na_bounds"] = VARIANT_BOUNDS[mode]
        vmeta["coverage_mode"] = mode
        vmeta["stats"] = st
        vmeta["folder_html"] = (
            f"<h2>{vtitle}</h2>"
            f"<p><b>Coverage:</b> {COVERAGE_LABELS[mode]}<br>"
            f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br>"
            f"<b>Provenance:</b> {prov}</p>") + coverage_selector_html(mode)
        write_metadata(vdir, vmeta)

    # Primary (default coverage): legacy paths/URLs. Its live folder
    # holds all three coverages as hideable overlays (default ON, the
    # other two OFF; radioFolder flips them exclusively in Pro).
    # Default is now global (worldwide canvas); basin + NA stay as
    # hideable overlays on the NA canvas.
    rgba = variant_rgba[default_cover]
    kt_def = variant_kt[default_cover]
    st_def = variant_stats[default_cover]
    def_h, def_w = rgba.shape[:2]
    m = np.isfinite(kt_def)
    n_valid = int(m.sum())
    print(f"[{PRODUCT}] valid px={n_valid} ({100*n_valid/(def_w*def_h):.2f}%) prov={prov}")
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    print(f"[{PRODUCT}] streamlets={st_def['streamlets']}/{st_def['heads']} "
          f"labels={st_def['labels']}+{st_def['extra_labels']}")

    lw, lh = draw_category_legend(os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"],
                                  f"{CONFIG_NA['freshness_label']}  |  {data_time}", key_rows,
                                  f"Source: GLWU v2.1 + HRRR 10 m + RRFS-NA + GDPS (see provenance)  |  "
                                  f"Processed {data_time}",
                                   note="Dark purple = Force 12 hurricane-force (≥64 kt) ONLY. "
                                        "Great Lakes + major Canadian lakes ~80% opacity; "
                                        "other land ~55% with intensified colors so the "
                                        "same force reads the same (single key). White streaks trace "
                                        "travel (brightness = knots); white numbers are actual kt. "
                                        "Coverage: Global (worldwide, valid data everywhere).")
    scale_html = ("WIND — BEAUFORT SCALE (colors follow force, dark purple = "
                  "Force 12 only). Speed m/s &rarr; kt (x1.94384) &rarr; WMO Beaufort 0-12. "
                  "Arrows point TOWARD motion; calm (&lt;0.5 m/s) draws no arrow. "
                  "Same force always shows the same color.")
    meta = base_metadata(
        PRODUCT, CONFIG_NA["title"], CONFIG_NA["freshness_label"],
        CONFIG_NA["source_name"], CONFIG_NA["source_url"], CONFIG_NA["variable"],
        data_time_utc=data_time, source_last_modified_utc=json.dumps(prov)[:300],
        units="kt (Beaufort 0-12)", source_resolution="GLWU ~2.5km; HRRR 3km; RRFS 13km NA",
        color_min=0, color_max=12, color_units="Beaufort force",
        missing_data_treatment="GLWU > HRRR > RRFS > GDPS(gaps) > transparent; missing never calm. "
            "Dual opacity (original finish): Great Lakes + major Canadian lakes (Bear, Slave, Winnipeg, "
            "Athabasca, Manitoba, Winnipegosis, Reindeer, Wollaston, Mistassini, Nettilling, Amadjuak, "
            "Dubawnt +) at water alpha 205 with exact Beaufort colors; other land at alpha 140 with "
            "saturation-boosted twins (single legend).")
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["beaufort_table"] = [
        {"force": f, "description": force_name(f), "range_kt": force_range_text(f),
         "color": FORCE_COLORS[f]} for f in range(13)]
    meta["source_id"] = src_id
    meta["source_version"] = token
    meta["na_bounds"] = VARIANT_BOUNDS[default_cover]
    meta["coverage_mode"] = default_cover
    meta["stats"] = {"valid_pixels": n_valid}
    folder_html = (f"<h2>{CONFIG_NA['title']}</h2>"
                   f"<p>Single continental field at NA resolution; same Beaufort "
                   f"colors, dual opacity, streamlet and label engines as the "
                   f"original basin layer, density-matched per degree.</p>"
                   f"<p><b>Coverage:</b> {COVERAGE_LABELS[default_cover]}<br>"
                   f"<b>Precedence:</b> {CONFIG_NA['precedence']}<br><b>Provenance:</b> {prov}</p>")
    meta["folder_html"] = folder_html + coverage_selector_html(default_cover)
    write_metadata(stage_prod, meta)
    outs = live_out_dirs(stage, KML_FILE)
    # All three coverages as hideable overlays in this folder (default
    # ON, others OFF; radioFolder flips exclusively in Pro). Global
    # overlay uses the worldwide LatLonBox; basin + NA use the NA box.
    extras = [(COVERAGE_LABELS[m], f"{variant_dirs[m]}/current.png",
               VARIANT_BOUNDS[m], 11 + i, 0)
              for i, m in enumerate([m for m in COVERAGE_DISPLAY
                                     if m != default_cover])]
    primary_bounds = VARIANT_BOUNDS[default_cover]
    kml_text = build_kml(PRODUCT, KML_FILE, OVERLAY_NAME, f"{PRODUCT}/current.png",
                         f"{PRODUCT}/legend.png",
                         description_html(CONFIG_NA["title"], meta, SKIP_NOTE, "")
                         + coverage_selector_html(default_cover),
                         CONFIG_NA["refresh_interval_seconds"], token,
                         folder=(CONFIG_NA["title"], meta["folder_html"]),
                         out_dirs=outs["live"], meta=meta, bounds=primary_bounds,
                         extra_overlays=extras, folder_radio=True)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(PRODUCT, KML_FILE, OVERLAY_NAME,
                    entry_description_html(CONFIG_NA["title"], meta, SKIP_NOTE),
                    CONFIG_NA["refresh_interval_seconds"], out_dirs=outs["entry"])
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": data_time, "source_id": src_id,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
