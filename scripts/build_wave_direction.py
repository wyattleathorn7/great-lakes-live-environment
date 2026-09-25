"""Pipeline C3 — LIVE WAVE PERIOD & DIRECTION (independent).

NOAA/NCEP GLWU v2.1 (WAVEWATCH III) surface analysis, 6-hourly cycles:
gradient = PERPW peak (primary) wave period in seconds as filed (no
conversion); arrows = WVDIR mean direction filed compass degrees.
Binned from the native 2.5 km Lambert grid onto the common canvas ->
lake-water only via the shared NOAA shoreline mask (land transparent)
-> fixed absolute spectrum 0-12 s (dark-blue short chop to dark-purple
long swell) + toward-travel arrows -> transparent PNG -> key image +
metadata -> Folder live KML + stable entry KML (individual files).

Arrows show the toward-travel vector (filed FROM direction + 180 deg,
standard oceanographic handling). NDBC buoy dominant period (DPD) is QC
reference only, never the rendering source.

Exit codes: 0 updated (or skipped); 2 source/validation failure
(previous kept); 1 unexpected error.
"""

import json
import math
import os
import re
import sys
import traceback
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR,
                              apply_shoreline_mask, base_metadata,
                              bin_to_canvas, canvas_indices, download,
                              grib_stamp_to_det, load_bounds, now_det_str,
                              promote_stage, read_state, save_png,
                              source_token, stage_dir, write_metadata,
                              write_state)
from gradient_scale import draw_scale_legend, render_rgba

PRODUCT = "wave_direction"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")
KML_FILE = "Great_Lakes_Live_Wave_Direction.kml"
OVERLAY_NAME = "🌊 LIVE WAVE DIRECTION"
SKIP_NOTE = "Turn on/off independently of all other layers."
UA = {"User-Agent": "great-lakes-live-environment/1.0"}

# Peak wave period in seconds, fixed absolute scale 0-12 s: short,
# closely-spaced wind chop sits dark-blue; long organized swell runs
# to dark-purple. Same seconds always show the same color.
WPER_STOPS = [
    (0.0, (16, 52, 140)),    # dark blue: flat/calm
    (2.0, (20, 110, 200)),   # blue
    (3.0, (20, 190, 200)),   # cyan
    (4.0, (90, 190, 80)),    # green
    (5.0, (245, 215, 50)),   # yellow
    (6.0, (240, 130, 25)),   # orange
    (7.5, (205, 30, 35)),    # red
    (9.0, (225, 40, 130)),   # magenta/pink
    (10.5, (130, 40, 170)),  # violet
    (12.0, (59, 10, 90)),    # dark purple: long swell
]
WPER_LABELS = [
    (0.0, "LOWEST 0s"),
    (2.0, "2"),
    (4.0, "4"),
    (6.0, "6"),
    (8.0, "8"),
    (10.0, "10"),
    (12.0, "HIGHEST+ 12s"),
]
WPER_MAX = 12.0
BUOY_QC_TOLERANCE_S = 2.5


def candidate_urls(now):
    urls = []
    for d in (now, now - timedelta(days=1)):
        datestr = d.strftime("%Y%m%d")
        for cycle in CONFIG["cycles_try_order"]:
            urls.append((CONFIG["file_pattern"].format(
                date_dir=f"glwu.{datestr}", cycle=cycle), datestr, cycle))
    return urls


def newest_available_cycle(now):
    best = None
    for url, dd, cc in candidate_urls(now):
        try:
            req = urllib.request.Request(url + ".idx", headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                idx = r.read().decode("utf-8", errors="replace").splitlines()
            for line in idx:
                p = line.split(":")
                if (len(p) > 5 and p[3] == "PERPW" and "surface" in line
                        and ":anl:" in line):
                    m = re.search(r"d=(\d{10})", line)
                    if m and (best is None or m.group(1) > best[3]):
                        best = (url, dd, cc, m.group(1))
                    break
        except Exception as e:
            print(f"[{PRODUCT}] idx probe {dd} t{cc}z: {str(e)[:100]}")
    return best


def _idx_range(idx, name):
    for i, line in enumerate(idx):
        p = line.split(":")
        if (len(p) > 4 and p[3] == name and ":surface:" in line
                and ":anl:" in line):
            start = int(p[1])
            end = int(idx[i + 1].split(":")[1]) if i + 1 < len(idx) else None
            return start, end
    return None, None


def fetch_vars(file_url, dest):
    """Byte-range download of the PERPW + WVDIR surface anl messages."""
    req = urllib.request.Request(file_url + ".idx", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        idx = r.read().decode("utf-8", errors="replace").splitlines()
    ranges = []
    for name in ("PERPW", "WVDIR"):
        start, end = _idx_range(idx, name)
        if start is None:
            raise ValueError(f"{name} surface analysis message not in .idx")
        ranges.append((name, start, end))
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        for name, start, end in ranges:
            req = urllib.request.Request(
                file_url,
                headers={**UA, "Range": f"bytes={start}-{end - 1 if end else ''}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                f.write(r.read())
    os.replace(tmp, dest)


def read_vars(path):
    """Return {PERPW: (vals_s, ...), WVDIR: (vals_deg, ...)} step-0 fields."""
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    out = {}
    with open(path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                sn = str(codes_get(h, "shortName")).lower()
                name = {"perpw": "PERPW", "wvdir": "WVDIR",
                        "mwdir": "WVDIR", "dirpw": "WVDIR"}.get(sn)
                if name is not None and str(codes_get(h, "step")) == "0" \
                        and name not in out:
                    vals = codes_get_values(h).astype(float)
                    lats = codes_get_array(h, "latitudes").astype(float)
                    lons = codes_get_array(h, "longitudes").astype(float)
                    lons = ((lons + 180) % 360) - 180
                    out[name] = (vals, lats, lons,
                                 str(codes_get(h, "dataDate")),
                                 str(codes_get(h, "dataTime")).zfill(4), sn)
            finally:
                codes_release(h)
    if "PERPW" not in out:
        raise ValueError("PERPW analysis message (step=0) not found in GRIB2")
    if "WVDIR" not in out:
        raise ValueError("WVDIR analysis message (step=0) not found in GRIB2")
    return out


def paint_arrows(rgba, field_deg, step=16):
    """Toward-travel arrows from filed compass degrees (FROM + 180)."""
    H, W = field_deg.shape
    img = Image.fromarray(rgba, mode="RGBA")
    d = ImageDraw.Draw(img)
    n = 0
    for r in range(step // 2, H, step):
        for c in range(step // 2, W, step):
            if not np.isfinite(field_deg[r, c]):
                continue
            if rgba[r, c, 3] == 0:
                continue
            toward = (float(field_deg[r, c]) + 180.0) % 360.0
            rad = math.radians(toward)
            dx, dy = math.sin(rad), -math.cos(rad)  # east+, north-up
            L = 10.0
            x0, y0 = c - dx * L / 2, r - dy * L / 2
            x1, y1 = c + dx * L / 2, r + dy * L / 2
            if not (8 <= x0 < W - 8 and 8 <= y0 < H - 8
                    and 8 <= x1 < W - 8 and 8 <= y1 < H - 8):
                continue
            ang = math.atan2(dy, dx)
            for ext, w, col in ((2, 3, (20, 20, 20, 235)),
                                (0, 1, (255, 255, 255, 240))):
                d.line([(x0, y0), (x1, y1)], fill=col, width=2 + ext)
                for s in (1, -1):
                    ha = ang + s * (math.pi - 0.5)
                    d.line([(x1, y1),
                            (x1 + math.cos(ha) * 5, y1 + math.sin(ha) * 5)],
                           fill=col, width=2 + ext)
            n += 1
    del d
    return np.array(img), n


def compass(deg):
    pts = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    return pts[int(((float(deg) % 360) + 22.5) // 45)]


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    now = datetime.now(timezone.utc)
    pick = newest_available_cycle(now)
    if pick is None:
        print(f"[{PRODUCT}] NO CYCLE AVAILABLE (keeping previous).")
        return 2
    url, datestr, cycle, stamp = pick
    source_id = f"glwu-{stamp[:8]}-{stamp[8:]}00Z-wvperiod"
    prev = read_state(PRODUCT)
    if (prev.get("source_id") == source_id
            and prev.get("render_version") == RENDER_VERSION
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png"))
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE))):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping raster.")
        try:
            refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                                 CONFIG["title"], SKIP_NOTE,
                                 CONFIG["refresh_interval_seconds"])
        except Exception as e:
            print(f"[{PRODUCT}] WARNING: KML refresh failed: {e}")
        return 0
    try:
        return _build(url, datestr, cycle, stamp, source_id)
    except Exception as e:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED: {type(e).__name__}: {e}. "
              f"Keeping previous.")
        return 2


def _build(url, datestr, cycle, stamp, source_id):
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    bounds = load_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    raw_path = os.path.join(RAW_DIR, "glwu_wvperiod_current.grib2")
    fetch_vars(url, raw_path)
    got = read_vars(raw_path)
    per_s, lats, lons, data_date, data_time, sn_per = got["PERPW"]
    dir_deg, _, _, _, _, sn_dir = got["WVDIR"]
    data_time_utc = grib_stamp_to_det(data_date, data_time)
    per_s = np.asarray(per_s, dtype=float).ravel()
    dir_deg = np.asarray(dir_deg, dtype=float).ravel()
    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()

    ok_src = (np.isfinite(per_s) & (per_s >= 0.0) & (per_s <= 20.0)
              & np.isfinite(dir_deg) & (dir_deg >= 0.0) & (dir_deg <= 360.0))
    if int(ok_src.sum()) < 3_000:
        raise ValueError(f"too few valid source cells ({int(ok_src.sum())})")
    rows, cols, valid = canvas_indices(lats, lons, bounds)
    field, _counts = bin_to_canvas(rows, cols, per_s, valid, (H, W),
                                   splat_radius=2)
    # Direction needs no smoothing: nearest bin keeps filed degrees exact.
    dfield, _ = bin_to_canvas(rows, cols, dir_deg, valid, (H, W),
                              splat_radius=0)
    okv = np.isfinite(field) & (field >= 0.0) & (field <= 20.0)
    if int(okv.sum()) < 5_000:
        raise ValueError(f"too few canvas cells ({int(okv.sum())})")
    if float(field[okv].max()) > 15.0:
        raise ValueError(f"implausible max period {float(field[okv].max()):.1f} s")
    cur_max = float(field[okv].max())
    cur_mean = float(field[okv].mean())
    dokv = np.isfinite(dfield) & (dfield >= 0.0) & (dfield <= 360.0)
    mean_deg = float(np.arctan2(np.sin(np.radians(dfield[dokv])).mean(),
                                np.cos(np.radians(dfield[dokv])).mean())
                     * 180.0 / math.pi % 360.0)

    rgba = render_rgba(field, WPER_STOPS, bounds["overlay_alpha"])
    rgba = apply_shoreline_mask(rgba)  # lake water only
    # Re-mask arrows/field to water for arrow sampling
    from geospatial_utils import load_watermask
    wm = load_watermask()
    water = wm > 0.5
    field_water = np.where(water & np.isfinite(dfield), dfield, np.nan)
    rgba, n_arrows = paint_arrows(rgba, field_water)
    # Arrows near shore can spill 1-2 px onto land: clip alpha back to
    # the water mask (no second bleed — colors already bled once).
    rgba[:, :, 3] = np.round(
        rgba[:, :, 3].astype(np.float32) * wm).astype(np.uint8)
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    print(f"[{PRODUCT}] opaque={n_opaque} arrows={n_arrows} "
          f"period max={cur_max:.1f}s mean={cur_mean:.1f}s "
          f"meandir={mean_deg:.0f} ({compass(mean_deg)})")
    if n_opaque < 5_000:
        raise ValueError("empty raster")
    if n_arrows < 50:
        raise ValueError(f"too few arrows ({n_arrows})")

    # ---- buoy QC: NDBC dominant period vs grid peak period (reference) ----
    from geospatial_utils import REPO_ROOT as _RR, fetch_buoy_obs
    import math as _math
    _BUOY_POS = {
        "45001": (-87.793, 48.061),
        "45002": (-86.411, 45.344),
        "45132": (-81.220, 42.460),
        "45012": (-77.383, 43.619),
        "45005": (-82.398, 41.677),
    }
    _qc = fetch_buoy_obs(CONFIG["buoys"])
    _bq = {}
    for _bid in CONFIG["buoys"]:
        _obs = _qc.get(_bid, {})
        _dpd = None
        try:
            _dpd = float(_obs.get("DPD_s"))
            if not _math.isfinite(_dpd):
                _dpd = None
        except (TypeError, ValueError):
            _dpd = None
        if _dpd is None:
            print(f"[{PRODUCT}] buoy {_bid}: no DPD obs (skipped)")
            _bq[_bid] = {**_obs, "note": "no DPD obs"}
            continue
        _lon, _lat = _BUOY_POS[_bid]
        _cc = int((_lon - bounds["lon_min"])
                  / (bounds["lon_max"] - bounds["lon_min"]) * W)
        _rr = int((bounds["lat_max"] - _lat)
                  / (bounds["lat_max"] - bounds["lat_min"]) * H)
        _cell = None
        for _dr in range(-3, 4):
            for _dc in range(-3, 4):
                _r2, _c2 = _rr + _dr, _cc + _dc
                if 0 <= _r2 < H and 0 <= _c2 < W \
                        and np.isfinite(field[_r2, _c2]):
                    _cell = float(field[_r2, _c2])
                    break
            if _cell is not None:
                break
        _diff = None if _cell is None else round(abs(_cell - _dpd), 2)
        _flag = ("OK" if (_diff is not None
                          and _diff <= BUOY_QC_TOLERANCE_S)
                 else "CHECK" if _diff is not None else "NO_GRID_CELL")
        print(f"[{PRODUCT}] buoy {_bid}: DPD {_dpd:.1f}s grid {_cell} "
              f"diff {_diff} -> {_flag}")
        _bq[_bid] = {"obs_s": round(_dpd, 2), "grid_s": _cell,
                     "absdiff_s": _diff, "verdict": _flag,
                     "obs_time": _obs.get("time_utc")}

    unit = CONFIG["display_units"]
    subtitle = (f"Peak wave period ({unit}) + travel arrows  |  {data_time_utc}  |  "
                f"max {cur_max:.1f}s, mean {cur_mean:.1f}s")
    lw, lh = draw_scale_legend(
        os.path.join(stage_prod, "legend.png"), CONFIG["title"], subtitle,
        "seconds", WPER_STOPS, WPER_LABELS,
        f"Source: NCEP GLWU v2.1 {datestr} t{cycle}z  |  Processed {now_det_str()}",
        note="Arrows show travel direction (filed FROM + 180).")
    scale_html = ("Peak wave period (seconds between crests, as filed by GLWU "
                  "PERPW): <b>LOWEST 0s</b> dark-blue flat &rarr; <b>2</b> blue "
                  "&rarr; <b>3</b> cyan &rarr; <b>4</b> green &rarr; <b>5</b> "
                  "yellow &rarr; <b>6</b> orange &rarr; <b>7.5</b> red &rarr; "
                  "<b>9</b> magenta &rarr; <b>10.5</b> violet &rarr; "
                  "<b>HIGHEST+ 12s</b> dark-purple long swell. Same seconds "
                  "always show the same color; values above 12s stay "
                  "dark-purple. Arrows point where the waves are traveling "
                  "(WVDIR filed direction + 180&deg;). Lake max now: "
                  f"<b>{cur_max:.1f}s</b>, mean <b>{cur_mean:.1f}s</b>, "
                  f"mean travel <b>{mean_deg:.0f}&deg; ({compass(mean_deg)})</b>.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], url, CONFIG["variable"],
        data_time_utc=data_time_utc,
        source_last_modified_utc="n/a (NOMADS)",
        units="s (display = source s)",
        source_resolution="~2.5 km NCEP GLWU Lambert grid (581x361)",
        color_min=0.0, color_max=WPER_MAX, color_units="s",
        missing_data_treatment=("only [0,20] admitted; off-water grid points and "
                                "land outside the NOAA shoreline transparent; "
                                "never zero-filled."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["model_cycle"] = f"{datestr} t{cycle}z"
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["grib_fields"] = {"period": sn_per, "direction": sn_dir}
    meta["buoy_qc"] = _bq
    meta["stats"] = {"valid_cells": int(okv.sum()),
                     "max_period_s": round(cur_max, 2),
                     "mean_period_s": round(cur_mean, 3),
                     "mean_direction_deg": round(mean_deg, 1),
                     "mean_compass": compass(mean_deg),
                     "arrows_drawn": n_arrows}
    token = meta["source_version"]
    folder_html = (
        f"<h2>{CONFIG['title']}</h2>"
        f"<p>{CONFIG['what']}</p>"
        f"<p><img src=\"{legend_src(token)}\" width=\"600\" alt=\"key\"></p>"
        f"<p>{scale_html}</p>"
        f"<p><b>Units:</b> {unit}<br/>"
        f"<b>Source:</b> {CONFIG['source_name']}<br/>"
        f"<b>Update:</b> GLWU 6-hourly cycles (analysis step)<br/>"
        f"<b>Data time:</b> {data_time_utc}<br/>"
        f"<b>Source version:</b> {source_id}<br/>"
        f"<b>Processed:</b> {meta['processing_time_utc']}<br/>"
        f"<b>Provenance:</b> <a href=\"{CONFIG['source_url']}\">NOMADS GLWU</a></p>")
    meta["folder_html"] = folder_html
    write_metadata(stage_prod, meta)
    block = legend_block(f"{PRODUCT}/legend.png", token, scale_html)
    outs = live_out_dirs(stage, KML_FILE)
    kml_text = build_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
        description_html(CONFIG["title"], meta, SKIP_NOTE, block),
        CONFIG["refresh_interval_seconds"], token,
        folder=(CONFIG["title"], folder_html),
        out_dirs=outs["live"])
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])

    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"model_cycle": meta["model_cycle"],
                          "source_id": source_id,
                          "render_version": RENDER_VERSION,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def legend_src(token=None):
    from build_kml import pages_base
    base = pages_base()
    v = f"?v={token}" if token else ""
    return f"{base}/{PRODUCT}/legend.png{v}"


if __name__ == "__main__":
    sys.exit(main())
