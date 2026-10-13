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
from na_common import (apply_coast_mask, bin_points_to_canvas, load_landonly,
                       despeckle_valid, feather_merge, fetch_http,
                       fill_pinholes, gate_field_to_mask, gdps_wcs_grid,
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
    prov = {}
    ulayers, vlayers = [], []  # (grid, joint-valid) in precedence order

    def place(label, u, v, la, lo, smooth_coarse=False):
        from na_common import smooth_field
        u = sanitize_model_vals(u, -150.0, 150.0)
        v = sanitize_model_vals(v, -150.0, 150.0)
        gu, _ = bin_points_to_canvas(u, la, lo, bounds)
        gv, _ = bin_points_to_canvas(v, la, lo, bounds)
        gu = fill_pinholes(gu, radius=3)
        gv = fill_pinholes(gv, radius=3)
        if smooth_coarse:
            # Coarse 13-15 km sources on the 0.05 deg canvas print
            # single-cell dots and block edges without this; the same
            # NaN-aware display smoothing the cloud layer applies to its
            # model fill. GLWU/HRRR stay raw (native resolution carries
            # the original grain).
            gu = smooth_field(gu, size=5, passes=2)
            gv = smooth_field(gv, size=5, passes=2)
        joint = np.isfinite(gu) & np.isfinite(gv)
        ulayers.append((gu, joint))
        vlayers.append((gv, joint))
        prov[label] = f"n={int(joint.sum())}"

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
    # where GLWU/HRRR/RRFS have no verified valid vector (verified RRFS
    # gap: Central America land). WindDir is the meteorological FROM
    # direction, so toward-motion U = -spd*sin(dir), V = -spd*cos(dir).
    try:
        arr_s, la_s, lo_s, slat, slon = gdps_wcs_grid("GDPS_15km_WindSpeed_10m", bounds, RAW_DIR)
        arr_d, _, _, _, _ = gdps_wcs_grid("GDPS_15km_WindDir_10m", bounds, RAW_DIR)
        spd = sanitize_model_vals(np.asarray(arr_s[0], dtype=float), 0.0, 120.0)
        wdir = np.asarray(arr_d[0], dtype=float)
        wdir = np.where(np.isfinite(wdir), wdir % 360.0, np.nan)
        rad = np.deg2rad(wdir)
        from na_common import resample_wcs_to_canvas as _bilin
        from na_common import smooth_field as _smooth
        gu = _bilin(np.where(np.isfinite(spd), -spd * np.sin(rad), np.nan), slat, slon, bounds)
        gv = _bilin(np.where(np.isfinite(spd), -spd * np.cos(rad), np.nan), slat, slon, bounds)
        gu = _smooth(fill_pinholes(gu, radius=3), size=5, passes=2)
        gv = _smooth(fill_pinholes(gv, radius=3), size=5, passes=2)
        joint = np.isfinite(gu) & np.isfinite(gv)
        ulayers.append((gu, joint))
        vlayers.append((gv, joint))
        prov["gdps"] = f"GDPS_15km_Wind_10m n={int(joint.sum())}"
    except Exception as e:
        prov["gdps"] = f"unavailable: {type(e).__name__}: {e}"
    if not ulayers:
        raise RuntimeError(f"no wind components from any source ({prov})")

    # Exact-copy precedence (same as the original: first-valid wins, no
    # blending): GLWU over water, then HRRR, then RRFS-NA, then GDPS gaps.
    # Coarse 13-15 km layers arrive pre-smoothed above (dots/blocks would
    # otherwise print); GLWU/HRRR render raw like the original.
    U = np.full((H, W), np.nan)
    V = np.full((H, W), np.nan)
    owned = np.zeros((H, W), dtype=bool)
    for gu, gv in zip([u for u, _ in ulayers], [v for _, v in vlayers]):
        take = (~owned) & np.isfinite(gu) & np.isfinite(gv)
        U[take] = gu[take]
        V[take] = gv[take]
        owned |= take
    U[~owned] = np.nan
    V[~owned] = np.nan
    Ud, Vd = U, V
    spd = np.hypot(Ud, Vd)  # m/s; NaN where missing (never zero-filled)
    n_valid = int(np.isfinite(spd).sum())
    if n_valid == 0:
        raise RuntimeError(f"no wind components from any source ({prov})")
    kt = spd * MS_TO_KT
    print(f"[{PRODUCT}] valid px={n_valid} ({100*n_valid/(W*H):.2f}%) prov={prov}")

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    m = np.isfinite(kt)
    # Original dual-opacity finish, expanded: the five Great Lakes + the
    # large Canadian lakes get the full water treatment (exact Beaufort
    # table @205); all other land gets the saturation-boosted vivid twins
    # @140 so each force reads the same through the extra transparency.
    # Single legend (water colors), same as the original.
    from build_wind import WIND_ALPHA_LAND, WIND_ALPHA_WATER, _vivid_for_land
    from na_common import load_lakemask
    lake = (load_lakemask() >= 0.5) & m
    forces = np.array([force_from_kt(x) for x in kt[m]])
    pal = np.array([force_rgb(f) for f in forces], dtype=np.uint8)
    vivid = np.array([_vivid_for_land(FORCE_COLORS[f]) for f in forces], dtype=np.uint8)
    idx = np.zeros((H, W), dtype=int)
    idx[m] = np.arange(int(m.sum()))
    rgba[m, :3] = np.where(lake[m][:, None], pal, vivid)
    rgba[m, 3] = np.where(lake[m], WIND_ALPHA_WATER, WIND_ALPHA_LAND)
    # Same engines as the original, density-matched per degree (NA canvas
    # is ~20 px/deg vs basin ~92 px/deg): streamlet seeds 12px and label
    # lattice 16px give the same geographic spacing as the basin's 20px /
    # 32px, so Canada and the south look like the original rectangle —
    # never enlarged, never a sparse 4-5 values. Full rectangle:
    # transparent only where missing, same as the original basin layer.
    from build_wind import paint_wind_streamlets
    from geospatial_utils import bleed_rgb_into_transparent
    rgba, n_stream, n_heads = paint_wind_streamlets(
        rgba, np.where(m, Ud, np.nan), np.where(m, Vd, np.nan), np.where(m, kt, np.nan),
        seed_step=12, ds=3.0, max_steps=40, min_speed=0.5)
    rgba, n_strat, n_fill = paint_na_wind_labels(
        rgba, np.where(m, kt, np.nan), bounds,
        label_step=16, font_size=7, max_labels=1500, extra_thresh_kt=4.0)
    rgba = bleed_rgb_into_transparent(rgba)
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    print(f"[{PRODUCT}] streamlets={n_stream}/{n_heads} labels={n_strat}+{n_fill}")
    # Single continental overlay only (no basin inset): one field, one
    # engine, everywhere — no pasted rectangle, no seams.

    # Original Beaufort key rows, verbatim (same colors, same labels).
    key_rows = [(FORCE_COLORS[f], f"F{f} — {force_name(f)} ({force_range_text(f)})")
                for f in range(13)]
    data_time = now_det_str()
    lw, lh = draw_category_legend(os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"],
                                  f"{CONFIG_NA['freshness_label']}  |  {data_time}", key_rows,
                                  f"Source: GLWU v2.1 + HRRR 10 m + RRFS-NA + GDPS (see provenance)  |  "
                                  f"Processed {data_time}",
                                   note="Dark purple = Force 12 hurricane-force (≥64 kt) ONLY. "
                                        "Great Lakes + major Canadian lakes ~80% opacity; "
                                        "other land ~55% with intensified colors so the "
                                        "same force reads the same (single key). White streaks trace "
                                        "travel (brightness = knots); white numbers are actual kt.")
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
    src_id = f"na-wind-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M}"
    meta["source_id"] = src_id
    meta["source_version"] = source_token(src_id + json.dumps(prov, sort_keys=True))
    meta["na_bounds"] = bounds
    meta["stats"] = {"valid_pixels": n_valid}
    token = meta["source_version"]
    folder_html = (f"<h2>{CONFIG_NA['title']}</h2><p>{scale_html}</p>"
                   f"<p>Single continental field at NA resolution; same Beaufort "
                   f"colors, dual opacity, streamlet and label engines as the "
                   f"original basin layer, density-matched per degree.</p>"
                   f"<p><b>Precedence:</b> {CONFIG_NA['precedence']}<br><b>Provenance:</b> {prov}</p>")
    meta["folder_html"] = folder_html
    write_metadata(stage_prod, meta)
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    outs = live_out_dirs(stage, KML_FILE)
    kml_text = build_kml(PRODUCT, KML_FILE, OVERLAY_NAME, f"{PRODUCT}/current.png",
                         f"{PRODUCT}/legend.png",
                         description_html(CONFIG_NA["title"], meta, SKIP_NOTE, block),
                         CONFIG_NA["refresh_interval_seconds"], token,
                         folder=(CONFIG_NA["title"], folder_html),
                         out_dirs=outs["live"], meta=meta, bounds=bounds)
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
