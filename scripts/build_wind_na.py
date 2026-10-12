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

# Strategic label anchors (lon, lat): major metros + corridor towns across
# the NA footprint, attempted first like the Great Lakes city's anchors.
NA_CITY_ANCHORS = [
    (-74.0, 40.7), (-87.6, 41.9), (-79.4, 43.7), (-95.4, 29.8),
    (-99.1, 19.4), (-103.3, 20.7), (-84.1, 9.9), (-82.4, 23.1),
    (-149.9, 61.2), (-113.5, 53.5), (-97.1, 49.9), (-75.7, 45.4),
    (-52.7, 47.6), (-157.8, 21.3), (-61.2, 14.6), (-90.5, 14.6),
]


def paint_na_wind_labels(rgba, kt_field, bounds, label_step=64,
                         font_size=7, max_labels=1200):
    """Strategic kt values in the Great Lakes label style (white mono +
    dark stroke): fixed city anchors first, then high-change extras on a
    data-seeded jittered lattice with spacing guards. Returns
    (rgba, n_strategic, n_fill)."""
    from build_wind import _draw_wind_label, _wind_label_font
    img = Image.fromarray(rgba)
    d = ImageDraw.Draw(img)
    font = _wind_label_font(font_size)
    H, W = kt_field.shape
    ok = np.isfinite(kt_field)
    if not ok.any():
        del d
        return np.array(img), 0, 0
    fin = kt_field[ok]
    seed = int(abs(float(fin.mean())) * 1000 + float(fin.std()) * 97) % (2 ** 32 - 1)
    rng = np.random.default_rng(seed)
    placed = np.zeros((H, W), dtype=bool)
    rr, cc = np.ogrid[:H, :W]

    def claim(y, x, sep):
        s = int(math.ceil(sep))
        r0, r1 = max(0, y - s), min(H, y + s + 1)
        c0, c1 = max(0, x - s), min(W, x + s + 1)
        dy = rr[r0:r1, 0:1] - y
        dx = cc[0:1, c0:c1] - x
        return not placed[r0:r1, c0:c1][(dy * dy + dx * dx) <= sep * sep].any()

    def mark(y, x, sep):
        s = int(math.ceil(sep))
        r0, r1 = max(0, y - s), min(H, y + s + 1)
        c0, c1 = max(0, x - s), min(W, x + s + 1)
        dy = rr[r0:r1, 0:1] - y
        dx = cc[0:1, c0:c1] - x
        placed[r0:r1, c0:c1][(dy * dy + dx * dx) <= sep * sep] = True

    def cell_value(lon, lat, rad=3):
        x = int((lon - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"]) * W)
        y = int((bounds["lat_max"] - lat) / (bounds["lat_max"] - bounds["lat_min"]) * H)
        best = None
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and ok[yy, xx]:
                    best = float(kt_field[yy, xx])
                    break
            if best is not None:
                break
        return (y, x, best)

    n_strat = n_fill = 0
    for lon, lat in NA_CITY_ANCHORS:
        y, x, v = cell_value(lon, lat)
        if v is None or not (8 <= y < H - 8 and 8 <= x < W - 8):
            continue
        if claim(y, x, 30):
            _draw_wind_label(d, x + 6, y - 12, f"{v:.0f}", font)
            mark(y, x, 30)
            n_strat += 1
    # high-change extras on a jittered lattice
    gy, gx = np.gradient(np.where(ok, kt_field, 0.0))
    grad = np.hypot(gy, gx)
    cand = []
    for yy in range(label_step // 2, H, label_step):
        for xx in range(label_step // 2, W, label_step):
            jx = int(xx + (rng.random() - 0.5) * label_step * 0.5)
            jy = int(yy + (rng.random() - 0.5) * label_step * 0.5)
            if 8 <= jy < H - 8 and 8 <= jx < W - 8 and ok[jy, jx]:
                cand.append((grad[jy, jx], jy, jx, float(kt_field[jy, jx])))
    cand.sort(reverse=True)
    for _, yy, xx, v in cand:
        if n_strat + n_fill >= max_labels:
            break
        if claim(yy, xx, 48):
            _draw_wind_label(d, xx + 6, yy - 12, f"{v:.0f}", font)
            mark(yy, xx, 48)
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
    LANDMASK = load_landonly()  # crisp: shoreline cut, no sea fringe
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    os.makedirs(stage_prod, exist_ok=True)
    prov = {}
    ulayers, vlayers = [], []  # (grid, joint-valid) in precedence order

    def place(label, u, v, la, lo):
        u = sanitize_model_vals(u, -150.0, 150.0)
        v = sanitize_model_vals(v, -150.0, 150.0)
        gu, _ = bin_points_to_canvas(u, la, lo, bounds)
        gv, _ = bin_points_to_canvas(v, la, lo, bounds)
        gu = fill_pinholes(gu, radius=3)
        gv = fill_pinholes(gv, radius=3)
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
        place(label, u, v, la, lo)
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
        gu = _bilin(np.where(np.isfinite(spd), -spd * np.sin(rad), np.nan), slat, slon, bounds)
        gv = _bilin(np.where(np.isfinite(spd), -spd * np.cos(rad), np.nan), slat, slon, bounds)
        gu = fill_pinholes(gu, radius=3)
        gv = fill_pinholes(gv, radius=3)
        joint = np.isfinite(gu) & np.isfinite(gv)
        ulayers.append((gu, joint))
        vlayers.append((gv, joint))
        prov["gdps"] = f"GDPS_15km_Wind_10m n={int(joint.sum())}"
    except Exception as e:
        prov["gdps"] = f"unavailable: {type(e).__name__}: {e}"
    if not ulayers:
        raise RuntimeError(f"no wind components from any source ({prov})")

    # Seamless cross-fade where domains overlap (joint masks keep U/V
    # weights identical), then one shared treatment.
    U, own_u = feather_merge(ulayers, feather_px=12)
    V, own_v = feather_merge(vlayers, feather_px=12)
    own = despeckle_valid(own_u & own_v & np.isfinite(U) & np.isfinite(V), min_size=4)
    U[~own] = np.nan
    V[~own] = np.nan
    Ud = gate_field_to_mask(smooth_field(U, size=5, passes=2), LANDMASK)
    Vd = gate_field_to_mask(smooth_field(V, size=5, passes=2), LANDMASK)
    spd = np.hypot(Ud, Vd)  # m/s; NaN where missing (never zero-filled)
    n_valid = int(np.isfinite(spd).sum())
    if n_valid == 0:
        raise RuntimeError(f"no wind components from any source ({prov})")
    kt = spd * MS_TO_KT
    print(f"[{PRODUCT}] valid px={n_valid} ({100*n_valid/(W*H):.2f}%) prov={prov}")

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    m = np.isfinite(kt)
    forces = np.array([force_from_kt(x) for x in kt[m]])
    pal = np.array([force_rgb(f) for f in forces], dtype=np.uint8)
    rgba[m, :3] = pal
    rgba[m, 3] = 205
    # GL-style direction streamlets (brightness = knots) + strategic kt
    # values, then the shared coastline cut (same rules as every layer).
    from build_wind import paint_wind_streamlets
    rgba, n_stream, n_heads = paint_wind_streamlets(
        rgba, np.where(m, Ud, np.nan), np.where(m, Vd, np.nan), np.where(m, kt, np.nan),
        seed_step=20, ds=3.0, max_steps=40, min_speed=0.5)
    rgba, n_strat, n_fill = paint_na_wind_labels(
        rgba, np.where(m, kt, np.nan), bounds,
        label_step=64, font_size=7, max_labels=1200)
    rgba = apply_coast_mask(rgba, LANDMASK)
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    print(f"[{PRODUCT}] streamlets={n_stream}/{n_heads} labels={n_strat}+{n_fill}")

    # Original Beaufort key rows, verbatim (same colors, same labels).
    key_rows = [(FORCE_COLORS[f], f"F{f} — {force_name(f)} ({force_range_text(f)})")
                for f in range(13)]
    data_time = now_det_str()
    lw, lh = draw_category_legend(os.path.join(stage_prod, "legend.png"), CONFIG_NA["title"],
                                  f"{CONFIG_NA['freshness_label']}  |  {data_time}", key_rows,
                                  f"Source: GLWU v2.1 + HRRR 10 m + RRFS-NA + GDPS (see provenance)  |  "
                                  f"Processed {data_time}",
                                  note="Dark purple = Force 12 hurricane-force (≥64 kt) ONLY. "
                                       "White arrows trace travel direction.")
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
        missing_data_treatment="GLWU > HRRR > RRFS > GDPS(gaps) > transparent; missing never calm.")
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
