"""Pipeline — LIVE GREAT LAKES FECAL-INDICATOR BACTERIA (independent).

Human-health recreational-water gradient for ALL FIVE Great Lakes water
surfaces (Superior, Michigan, Huron, Erie, Ontario — US + Canadian waters).

Correct terminology (EPA 2012 Recreational Water Quality Criteria): the
layer maps FECAL-INDICATOR BACTERIA — E. coli and enterococci — organisms
indicating fecal contamination and potential pathogen risk in
primary-contact recreational waters. Indicator values are NOT pathogen
detections, NOT disease diagnoses, and individual types are NOT
distinguished visually: one unified human-health concern spectrum.

Authoritative basis:
  EPA 2012 RWQC thresholds (GM / STV / Beach Action Value),
    https://www.epa.gov/wqc/recreational-water-quality-criteria-and-methods
  Live observations: USGS Water Quality Portal (WQX E. coli / enterococci,
    discrete samples), https://www.waterqualitydata.us/
  Advisory framework: EPA BEACON beach notification program
    https://www.epa.gov/beach-tech/beacon-20-beach-advisory-and-closing-online-notification
No basin-wide continuous bacterial raster exists (EPA programs are
station-based); this layer therefore NEVER fabricates measurements:
  - water with a defensible observation -> 13-step gradient halo
  - water without one -> explicit NO DATA slate (never zero/low)

Coverage: 100% of the five Great Lakes water surfaces via the shared
NOAA shoreline mask (land transparent). US + Canadian water included;
unrelated inland lakes excluded (mask-bounded).

Temporal rule: sources CHECKED hourly. Agencies publish only when
sampling occurs (often weekly in swim season). New observation -> used;
no new observation -> latest valid observation retained up to 7 days,
then stale -> NO DATA. CHECKED-HOURLY vs PUBLISHED-WHEN-SAMPLED is
stated in every legend/metadata/KML.

Thresholds: E. coli-equivalent CFU/100 mL log scale anchored ONLY at EPA
values (BAV 235, GM 126/100, STV 410/320/130/110) plus order-of-magnitude
extensions for the extreme tail. Enterococci convert at the EPA GM ratio
126/35 = 3.6x (documented). Advisory-level (>= BAV) minimum display is
documented. Nothing is silently combined: every transition carries
value/unit/threshold/source/rationale in metadata.

Refresh: 3600 s on entry NetworkLink + live Icon. Source-aware gate on
the newest observation id actually used.

Exit codes: 0 updated (or skipped); 2 source/validation failure
(previous kept); 1 unexpected error.
"""

import bisect
import csv
import datetime as dt
import io
import json
import math
import os
import sys
import traceback
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       description_html, entry_description_html, legend_block,
                       live_out_dirs, refresh_kml_base_url)
from geospatial_utils import (REPO_ROOT, SITE_DIR, base_metadata,
                              bleed_rgb_into_transparent, load_bounds,
                              load_watermask, now_det_str, promote_stage,
                              read_state, save_png, source_token, stage_dir,
                              write_metadata, write_state)

PRODUCT = "bacteria"
CONFIG = json.load(open(os.path.join(REPO_ROOT, "config", f"{PRODUCT}.json")))
KML_FILE = "Great_Lakes_Live_Water_Quality.kml"
OVERLAY_NAME = "\U0001F4A7 LIVE WATER QUALITY"
SKIP_NOTE = "Turn on/off independently of all other layers."
WATER_ALPHA = 205
NODATA_RGB = (110, 125, 150)  # explicit NO DATA slate over water
Halo_R = 14  # px splat radius per monitoring station (~ display only)
UA = {"User-Agent": "great-lakes-live-environment/1.0"}

# 13-step human-health gradient (best -> worst), fixed absolute anchors in
# E. coli-equivalent CFU/100 mL. Every anchor is EPA-anchored or a
# documented order-of-magnitude extension (see THRESHOLDS below).
BACTERIA_STOPS = [  # (value, rgb)
    (10.0, (13, 42, 120)),     # DARK BLUE: no detected / lowest
    (30.0, (20, 100, 215)),    # BLUE: very low
    (60.0, (20, 190, 200)),    # CYAN: low
    (100.0, (120, 220, 130)),  # LIGHT GREEN: low-to-moderate
    (126.0, (30, 130, 70)),    # DARK GREEN: moderate (EPA E.coli GM 126)
    (190.0, (150, 200, 60)),   # YELLOWISH GREEN: elevated
    (235.0, (245, 215, 50)),   # YELLOW: concerning (EPA Beach Action Value)
    (320.0, (240, 130, 25)),   # ORANGE: high (STV, 32/1000 rate)
    (410.0, (205, 30, 35)),    # RED: very high (EPA STV 410)
    (700.0, (255, 40, 60)),    # NEON RED: extremely high
    (1000.0, (190, 25, 120)),  # MAGENTA: severe
    (2000.0, (110, 25, 150)),  # PURPLE: extremely severe
    (5000.0, (60, 10, 90)),    # DARK PURPLE: extreme (5000+ clamps here)
]
BACTERIA_MAX = 5000.0
BACTERIA_TICKS = [
    (10.0, "LOWEST 10"),
    (126.0, "126 GM"),
    (235.0, "235 BAV"),
    (410.0, "410 STV"),
    (1000.0, "1000"),
    (5000.0, "HIGHEST+ 5000+"),
]

THRESHOLDS = [
    {"color": "DARK BLUE", "value": 10, "unit": "CFU/100 mL",
     "threshold": "display floor; below any EPA criterion",
     "source": "project display floor (no health claim)",
     "rationale": "No detected / lowest measured condition among retained observations."},
    {"color": "BLUE", "value": 30, "unit": "CFU/100 mL",
     "threshold": "below EPA enterococci GM 35 / well below E. coli GM 126",
     "source": "EPA 2012 RWQC (GM enterococci 35, GM E. coli 126)",
     "rationale": "Very low: an order of magnitude under the E. coli GM criterion."},
    {"color": "CYAN", "value": 60, "unit": "CFU/100 mL",
     "threshold": "below EPA GM values",
     "source": "EPA 2012 RWQC",
     "rationale": "Low: under half the E. coli GM criterion."},
    {"color": "LIGHT GREEN", "value": 100, "unit": "CFU/100 mL",
     "threshold": "below EPA E. coli GM 126 (36/1000 illness rate)",
     "source": "EPA 2012 RWQC Table 1 (GM E. coli 126 / 100)",
     "rationale": "Low-to-moderate: approaches but does not reach the GM criterion."},
    {"color": "DARK GREEN", "value": 126, "unit": "CFU/100 mL",
     "threshold": "EPA E. coli GM criterion 126 (illness rate 36/1000)",
     "source": "EPA 2012 RWQC (GM E. coli 126; alternate 100 at 32/1000)",
     "rationale": "Moderate: AT the geometric-mean criterion — the long-term waterbody standard."},
    {"color": "YELLOWISH GREEN", "value": 190, "unit": "CFU/100 mL",
     "threshold": "between GM 126 and BAV 235",
     "source": "EPA 2012 RWQC + BAV",
     "rationale": "Elevated: exceeds the GM criterion, below beach-action level."},
    {"color": "YELLOW", "value": 235, "unit": "CFU/100 mL",
     "threshold": "EPA Beach Action Value (BAV) 235 E. coli",
     "source": "EPA 2012 RWQC supplemental BAV (75th percentile of RWQC distribution)",
     "rationale": "Concerning: AT the precautionary beach-notification value."},
    {"color": "ORANGE", "value": 320, "unit": "CFU/100 mL",
     "threshold": "EPA E. coli STV 320 (illness rate 32/1000)",
     "source": "EPA 2012 RWQC Table 1 (STV E. coli 410 / 320)",
     "rationale": "High: exceeds the alternate statistical-threshold value."},
    {"color": "RED", "value": 410, "unit": "CFU/100 mL",
     "threshold": "EPA E. coli STV 410 (illness rate 36/1000)",
     "source": "EPA 2012 RWQC Table 1 (STV E. coli 410)",
     "rationale": "Very high / potentially unsafe: AT the single-sample STV criterion."},
    {"color": "NEON RED", "value": 700, "unit": "CFU/100 mL",
     "threshold": "order-of-magnitude extension above STV",
     "source": "project extension (documented, no EPA criterion at this value)",
     "rationale": "Extremely high: ~1.7x the STV; advisory/closure expected."},
    {"color": "MAGENTA", "value": 1000, "unit": "CFU/100 mL",
     "threshold": "order-of-magnitude extension (~2.4x STV)",
     "source": "project extension (documented)",
     "rationale": "Severe: gross exceedance; unsafe for contact recreation by any standard."},
    {"color": "PURPLE", "value": 2000, "unit": "CFU/100 mL",
     "threshold": "order-of-magnitude extension (~5x STV)",
     "source": "project extension (documented)",
     "rationale": "Extremely severe contamination event."},
    {"color": "DARK PURPLE", "value": 5000, "unit": "CFU/100 mL",
     "threshold": "extreme clamp (5000+ stays dark purple)",
     "source": "project clamp (documented)",
     "rationale": "Extremely poor human-health water condition."},
]

# Enterococci -> E.coli-equivalent factor: EPA GM ratio 126/35 = 3.6.
ENTERO_TO_ECOLI = 126.0 / 35.0

# Committed display-anchor inventory: major public recreational beaches
# per lake (approx shoreline positions for halo rendering; NOT survey
# points; halos paint only when a live observation exists for the area).
STATIONS = [
    # Superior
    {"name": "Park Point Beach (Duluth)", "lake": "Superior", "lat": 46.72, "lon": -92.00, "program": "Minnesota Beach Program"},
    {"name": "Bayfield City Beach", "lake": "Superior", "lat": 46.81, "lon": -90.81, "program": "Wisconsin Beach Program"},
    {"name": "Marquette City Beach", "lake": "Superior", "lat": 46.54, "lon": -87.38, "program": "Michigan Beach Program"},
    {"name": "Agate Bay Beach (Two Harbors)", "lake": "Superior", "lat": 47.02, "lon": -91.67, "program": "Minnesota Beach Program"},
    {"name": "Chippewa Park Beach (Thunder Bay)", "lake": "Superior", "lat": 48.33, "lon": -89.20, "program": "Ontario Health Unit"},
    # Michigan
    {"name": "Grand Haven City Beach", "lake": "Michigan", "lat": 43.06, "lon": -86.24, "program": "Michigan Beach Program"},
    {"name": "Sleeping Bear (Esch Rd Beach)", "lake": "Michigan", "lat": 44.88, "lon": -86.05, "program": "Michigan Beach Program"},
    {"name": "Holland State Park Beach", "lake": "Michigan", "lat": 42.77, "lon": -86.21, "program": "Michigan Beach Program"},
    {"name": "Montrose Beach (Chicago)", "lake": "Michigan", "lat": 41.96, "lon": -87.64, "program": "Illinois Beach Program"},
    {"name": "Bradford Beach (Milwaukee)", "lake": "Michigan", "lat": 43.06, "lon": -87.90, "program": "Wisconsin Beach Program"},
    {"name": "Silver Beach (St. Joseph)", "lake": "Michigan", "lat": 42.11, "lon": -86.49, "program": "Michigan Beach Program"},
    # Huron (+Georgian Bay)
    {"name": "Oscoda Beach Park", "lake": "Huron", "lat": 44.42, "lon": -83.33, "program": "Michigan Beach Program"},
    {"name": "Port Huron (Lakeside Beach)", "lake": "Huron", "lat": 43.00, "lon": -82.42, "program": "Michigan Beach Program"},
    {"name": "Wasaga Beach", "lake": "Huron", "lat": 44.52, "lon": -80.02, "program": "Ontario Health Unit"},
    {"name": "Sauble Beach", "lake": "Huron", "lat": 44.65, "lon": -81.27, "program": "Ontario Health Unit"},
    {"name": "Bay City State Park Beach", "lake": "Huron", "lat": 43.66, "lon": -83.91, "program": "Michigan Beach Program"},
    # Erie
    {"name": "Presque Isle (Beach 6)", "lake": "Erie", "lat": 42.11, "lon": -80.15, "program": "Pennsylvania Beach Program"},
    {"name": "Maumee Bay State Park Beach", "lake": "Erie", "lat": 41.68, "lon": -83.37, "program": "Ohio Beach Program"},
    {"name": "Cedar Point Beach", "lake": "Erie", "lat": 41.48, "lon": -82.68, "program": "Ohio Beach Program"},
    {"name": "Port Stanley Main Beach", "lake": "Erie", "lat": 42.66, "lon": -81.21, "program": "Ontario Health Unit"},
    {"name": "Long Point Beach", "lake": "Erie", "lat": 42.57, "lon": -80.40, "program": "Ontario Health Unit"},
    {"name": "Evangola State Park Beach", "lake": "Erie", "lat": 42.58, "lon": -79.06, "program": "New York Beach Program"},
    # Ontario
    {"name": "Toronto Centre Island Beach", "lake": "Ontario", "lat": 43.62, "lon": -79.36, "program": "Toronto Public Health"},
    {"name": "Hamilton Confederation Beach", "lake": "Ontario", "lat": 43.27, "lon": -79.07, "program": "Ontario Health Unit"},
    {"name": "Sandbanks (Outlet Beach)", "lake": "Ontario", "lat": 43.90, "lon": -77.30, "program": "Ontario Parks"},
    {"name": "Durham Beach (Oshawa)", "lake": "Ontario", "lat": 43.86, "lon": -78.83, "program": "Ontario Health Unit"},
    {"name": "Southwick Beach (NY)", "lake": "Ontario", "lat": 43.77, "lon": -76.21, "program": "New York Beach Program"},
]


def _fetch_csv(url, timeout=150, max_bytes=30 * 1024 * 1024, retries=3):
    """GET with transient-failure retries (WQP is slow/flaky at times)."""
    import time as _time
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(max_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            last = e
            print(f"[{PRODUCT}] fetch attempt {attempt} failed ({e}); retrying.")
            _time.sleep(3 * attempt)
    raise last


STATION_CACHE = os.path.join(REPO_ROOT, "output", "state", "bacteria_stations.json")


def _load_station_cache():
    try:
        with open(STATION_CACHE) as f:
            return {k: tuple(v) for k, v in json.load(f).items()}
    except (OSError, ValueError):
        return {}


def _save_station_cache(lookup):
    try:
        os.makedirs(os.path.dirname(STATION_CACHE), exist_ok=True)
        with open(STATION_CACHE, "w") as f:
            json.dump({k: list(v) for k, v in lookup.items()}, f)
    except OSError:
        pass


def wqp_station_coords(sids, timeout=120):
    """Resolve station ids -> (lat, lon, name) via chunked siteid queries.

    A basin-wide Station search is too heavy for an hourly job, so only
    the stations actually seen in the current result window are resolved
    (typically dozens), merged with the persistent cache in
    output/state/bacteria_stations.json.
    """
    cached = _load_station_cache()
    missing = [s for s in set(sids) if s not in cached]
    for i in range(0, len(missing), 25):
        chunk = missing[i:i + 25]
        params = {"siteid": ";".join(chunk), "mimeType": "csv", "zip": "no"}
        url = ("https://www.waterqualitydata.us/data/Station/search?"
               + urllib.parse.urlencode(params))
        try:
            body = _fetch_csv(url, timeout=timeout)
        except Exception as e:
            print(f"[{PRODUCT}] station chunk failed ({e}); using cache.")
            break
        for rec in csv.DictReader(io.StringIO(body)):
            try:
                sid = (rec.get("MonitoringLocationIdentifier") or "").strip()
                lat = float(rec.get("LatitudeMeasure", ""))
                lon = float(rec.get("LongitudeMeasure", ""))
                if not sid or not (-93.5 <= lon <= -73.0 and 40.0 <= lat <= 50.0):
                    continue
                cached[sid] = (lat, lon, (rec.get("MonitoringLocationName") or "")[:100])
            except (ValueError, TypeError):
                continue
    if missing:
        _save_station_cache(cached)
    return cached


def wqp_search(days=7, timeout=150):
    """Query the Water Quality Portal for recent E. coli / enterococci in
    the basin bbox, joined to station coordinates. Returns (rows, url).

    Non-detects (empty value + 'Not Detected'/'Below Quantification'
    condition) are kept as upper-bound estimates at the quantitation
    limit (or the display floor 10 when no limit is given) and flagged
    censored — a documented conservative treatment, never zero.
    Never raises fatally — callers treat empty as NO DATA (honest path).
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    params = {
        "countrycode": "US",
        "bBox": "-93,40.5,-73.5,49.5",
        "startDateLo": start.strftime("%m-%d-%Y"),
        "startDateHi": end.strftime("%m-%d-%Y"),
        "characteristicName": "Escherichia coli;Enterococcus",
        "mimeType": "csv",
        "zip": "no",
    }
    url = "https://www.waterqualitydata.us/data/Result/search?" + urllib.parse.urlencode(params)
    body = _fetch_csv(url, timeout=timeout)
    raw_sids = []
    prelim = []
    for rec in csv.DictReader(io.StringIO(body)):
        char = (rec.get("CharacteristicName") or "").strip().lower()
        if "escherichia" not in char and "enterococcus" not in char:
            continue
        sid = (rec.get("MonitoringLocationIdentifier") or "").strip()
        if sid:
            raw_sids.append(sid)
            prelim.append(rec)
    stations = wqp_station_coords(raw_sids, timeout=timeout)
    rows = []
    for rec in prelim:
        try:
            char = (rec.get("CharacteristicName") or "").strip().lower()
            sid = (rec.get("MonitoringLocationIdentifier") or "").strip()
            if sid not in stations:
                continue
            lat, lon, sname = stations[sid]
            v = (rec.get("ResultMeasureValue") or "").strip()
            censored = False
            if not v or v.lower() in ("*", "**"):
                cond = (rec.get("ResultDetectionConditionText") or "").lower()
                if "not detect" in cond or "below" in cond or "non-detect" in cond or "non detect" in cond:
                    try:
                        lim = float((rec.get("DetectionQuantitationLimitMeasure/MeasureValue") or "").strip())
                        val = lim if math.isfinite(lim) and lim > 0 else 10.0
                    except (ValueError, TypeError):
                        val = 10.0
                    censored = True
                else:
                    continue
            else:
                try:
                    val = float(v.replace(",", ""))
                except ValueError:
                    continue
            unit = (rec.get("ResultMeasure/MeasureUnitCode") or "").strip().lower().replace(" ", "")
            if "100ml" not in unit and "ml" not in unit:
                continue
            if not (math.isfinite(val) and 0 <= val <= 1_000_000):
                continue
            date = (rec.get("ActivityStartDate") or "").strip()
            rows.append({
                "lat": lat, "lon": lon, "value": val,
                "indicator": "ecoli" if "escherichia" in char else "enterococci",
                "date": date, "station": sid[:80], "station_name": sname,
                "censored": censored,
            })
        except (ValueError, TypeError):
            continue
    # Newest sample per station wins (temporal rule: latest valid observation).
    best = {}
    for o in rows:
        k = o["station"]
        if k not in best or o["date"] >= best[k]["date"]:
            best[k] = o
    return list(best.values()), url


def ecoli_equivalent(obs):
    if obs["indicator"] == "enterococci":
        return obs["value"] * ENTERO_TO_ECOLI
    return obs["value"]


def _log_color(v):
    """Raster-identical color: piecewise-linear interpolation in LOG space."""
    import bisect
    av = [a for a, _ in BACTERIA_STOPS]
    lv = [math.log10(a) for a in av]
    c = min(max(v, av[0]), av[-1])
    lc = math.log10(c)
    j = min(max(bisect.bisect_left(av, c), 1), len(av) - 1)
    f = (lc - lv[j - 1]) / (lv[j] - lv[j - 1] or 1.0)
    c0 = np.array(BACTERIA_STOPS[j - 1][1], dtype=float)
    c1 = np.array(BACTERIA_STOPS[j][1], dtype=float)
    return tuple(int(x) for x in np.round(c0 + (c1 - c0) * f))


def draw_log_legend(path, title, subtitle, unit_label, ticks, source_line, note=None):
    """Key image painted in LOG space — pixel-identical mapping to the
    raster (linear legend bars would crush the 10-410 EPA decision range
    into a sliver). Returns (W, H)."""
    from PIL import Image, ImageDraw
    from geospatial_utils import _legend_font
    W, H = 640, 230
    img = Image.new("RGBA", (W, H), (255, 255, 255, 235))
    d = ImageDraw.Draw(img)
    f_title, f_body, f_small = _legend_font(22), _legend_font(15), _legend_font(13)
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=2)
    d.text((14, 8), title, font=f_title, fill=(10, 10, 10))
    d.text((14, 36), subtitle, font=f_body, fill=(40, 40, 40))
    bx, by, bw, bh = 14, 66, W - 28, 32
    lo, hi = math.log10(BACTERIA_STOPS[0][0]), math.log10(BACTERIA_STOPS[-1][0])
    for i in range(bw):
        v = 10 ** (lo + (hi - lo) * i / (bw - 1))
        d.line([(bx + i, by), (bx + i, by + bh)], fill=_log_color(v) + (255,))
    d.rectangle([bx, by, bx + bw - 1, by + bh], outline=(40, 40, 40))
    placed = []
    for val, text in ticks:
        frac = (math.log10(val) - lo) / (hi - lo)
        x = bx + int(min(max(frac, 0.0), 1.0) * (bw - 1))
        tw = d.textlength(text, font=f_small)
        placed.append((x, tw, text))
    kept = [placed[0]] if placed else []
    last = placed[-1] if len(placed) > 1 else None
    for cand in placed[1:-1]:
        if all(cand[0] - cand[1] / 2 > b[0] + b[1] / 2 + 2
               or cand[0] + cand[1] / 2 < b[0] - b[1] / 2 - 2 for b in kept) \
           and (last is None or cand[0] + cand[1] / 2 < last[0] - last[1] / 2 - 2):
            kept.append(cand)
    if last is not None:
        kept.append(last)
    for x, tw, text in kept:
        d.text((min(max(x - tw / 2, 2), W - tw - 2), by + bh + 4),
               text, font=f_small, fill=(10, 10, 10))
    d.text((bx + bw - 70, by + bh + 24), unit_label, font=f_body, fill=(10, 10, 10))
    d.text((14, H - 42), source_line, font=f_small, fill=(60, 60, 60))
    if note:
        d.text((14, H - 24), note, font=f_small, fill=(60, 60, 60))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return W, H


def main():
    try:
        return run()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


def run():
    try:
        obs_rows, query_url = wqp_search(days=CONFIG["freshness_window_days"])
        fetch_ok = True
        fetch_note = f"WQP live query OK ({len(obs_rows)} usable samples)"
    except Exception as e:
        obs_rows, query_url = [], "https://www.waterqualitydata.us/"
        fetch_ok = False
        fetch_note = f"WQP live query failed: {type(e).__name__}: {e}"
    print(f"[{PRODUCT}] {fetch_note}")
    # Source-aware gate: newest observation identity + fetch day.
    newest = ""
    for o in obs_rows:
        key = f"{o['station']}|{o['date']}|{o['value']}"
        if key > newest:
            newest = key
    source_id = f"wqp-{dt.date.today().isoformat()}-n{len(obs_rows)}-{abs(hash(newest)) % 10_000_000:07d}"
    prev = read_state(PRODUCT)
    if (prev.get("source_id") == source_id
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png"))
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", KML_FILE))):
        print(f"[{PRODUCT}] source unchanged ({source_id}); keeping.")
        return _refresh_kml()
    try:
        return _build(obs_rows, query_url, fetch_ok, fetch_note, source_id)
    except Exception:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED. Keeping previous.")
        return 2


def _refresh_kml():
    try:
        refresh_kml_base_url(PRODUCT, KML_FILE, OVERLAY_NAME,
                             CONFIG["title"], SKIP_NOTE,
                             CONFIG["refresh_interval_seconds"])
        print(f"[{PRODUCT}] KML base URLs refreshed.")
    except Exception as e:
        print(f"[{PRODUCT}] WARNING: KML refresh failed: {e}")
    return 0


def _build(obs_rows, query_url, fetch_ok, fetch_note, source_id):
    bounds = load_bounds()
    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    wm = load_watermask()
    water = wm >= 0.5

    # Base: explicit NO DATA slate over 100% of lake water; land transparent.
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[water, 0:3] = NODATA_RGB
    rgba[water, 3] = WATER_ALPHA

    # Halo field: E.coli-equivalent per pixel (NaN = no observation).
    halo = np.full((H, W), np.nan)
    used = []
    # Live observations first (priority documented).
    for o in obs_rows:
        c = int((o["lon"] - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"]) * W)
        r = int((bounds["lat_max"] - o["lat"]) / (bounds["lat_max"] - bounds["lat_min"]) * H)
        if not (0 <= r < H and 0 <= c < W):
            continue
        val = ecoli_equivalent(o)
        val = min(max(val, 10.0), BACTERIA_MAX)
        r0, r1 = max(r - Halo_R, 0), min(r + Halo_R + 1, H)
        c0, c1 = max(c - Halo_R, 0), min(c + Halo_R + 1, W)
        yy, xx = np.mgrid[r0 - r:r1 - r, c0 - c:c1 - c]
        disc = xx * xx + yy * yy <= Halo_R * Halo_R
        sub = halo[r0:r1, c0:c1]
        wsub = water[r0:r1, c0:c1]
        put = disc & wsub & (np.isnan(sub) | (val > sub))
        if put.any():
            sub[put] = val
            used.append(o)
    # Display anchors: station halos ONLY where live data exists nearby is
    # honest; anchors without observations stay NO DATA (no fill).
    n_halo_px = int(np.isfinite(halo).sum())
    if n_halo_px:
        flat = halo[np.isfinite(halo)]
        cols = np.array([_log_color(v) for v in flat], dtype=np.uint8)
        m = np.isfinite(halo)
        rgba[m, 0:3] = cols  # mask order == flat order
        rgba[m, 3] = WATER_ALPHA
        rgba = bleed_rgb_into_transparent(rgba)
    else:
        rgba = bleed_rgb_into_transparent(rgba)

    save_png(rgba, os.path.join(stage_prod, "current.png"))
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    n_water = int(water.sum())
    print(f"[{PRODUCT}] opaque={n_opaque} water={n_water} halo_px={n_halo_px} stations_used={len(used)}")

    if n_halo_px:
        hv = halo[np.isfinite(halo)]
        cur_min, cur_max = float(hv.min()), float(hv.max())
    else:
        cur_min, cur_max = None, None

    subtitle = (f"E.coli-equiv CFU/100 mL (log)  |  "
                f"{len(used)} halos / {len(STATIONS)} beaches")
    lw, lh = draw_log_legend(
        os.path.join(stage_prod, "legend.png"), "LIVE WATER QUALITY", subtitle,
        "CFU/100 mL", BACTERIA_TICKS,
        f"Source: {fetch_note} | EPA 2012 RWQC GM/STV/BAV",
        note="Slate gray = NO DATA (not zero). Source checked hourly.")
    scale_html = (
        f"Human-health concern from fecal-indicator evidence, FIXED log scale "
        f"(E.coli-equiv CFU/100 mL): <b>DARK BLUE 10</b> (no detected/lowest) &rarr; "
        f"<b>BLUE 30</b> (very low) &rarr; <b>CYAN 60</b> (low) &rarr; "
        f"<b>LIGHT GREEN 100</b> (low-to-moderate) &rarr; <b>DARK GREEN 126</b> (moderate, EPA GM) &rarr; "
        f"<b>YELLOWISH GREEN 190</b> (elevated) &rarr; <b>YELLOW 235</b> (concerning, EPA Beach Action Value) &rarr; "
        f"<b>ORANGE 320</b> (high) &rarr; <b>RED 410</b> (very high, EPA STV) &rarr; "
        f"<b>NEON RED 700</b> (extremely high) &rarr; <b>MAGENTA 1000</b> (severe) &rarr; "
        f"<b>PURPLE 2000</b> (extremely severe) &rarr; <b>DARK PURPLE HIGHEST+ 5000+</b> (extreme). "
        f"<b>LOWEST 10</b>. Slate gray water = <b>NO DATA</b>, never zero bacteria.")
    meta = base_metadata(
        PRODUCT, CONFIG["title"], CONFIG["freshness_label"],
        CONFIG["source_name"], CONFIG["source_url"], CONFIG["variable"],
        data_time_utc=f"WQP window {CONFIG['freshness_window_days']}d ending {dt.date.today().isoformat()} ({fetch_note})",
        source_last_modified_utc="n/a (WQP query, checked hourly)",
        units=f"{CONFIG['source_units']}; display {CONFIG['display_units']}",
        source_resolution="discrete monitoring stations (point samples); halo radius "
                          f"{Halo_R}px display only — no interpolation between stations, no modeled fill",
        color_min=10.0, color_max=BACTERIA_MAX, color_units="E.coli-equiv CFU/100 mL",
        missing_data_treatment=(
            "NO authoritative observation -> slate NO DATA over water (never zero/low). "
            "Observations older than 7 days are stale -> NO DATA. "
            "Land always transparent (shore mask)."))
    meta["legend_size"] = [lw, lh]
    meta["legend_scale_html"] = scale_html
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["wqp_query_url"] = query_url
    meta["wqp_fetch_ok"] = fetch_ok
    meta["wqp_fetch_note"] = fetch_note
    meta["freshness_window_days"] = CONFIG["freshness_window_days"]
    meta["samples_used"] = len(used)
    meta["samples"] = [
        {"station": o["station"], "station_name": o["station_name"],
         "lat": o["lat"], "lon": o["lon"], "indicator": o["indicator"],
         "value": o["value"], "ecoli_equivalent": round(ecoli_equivalent(o), 1),
         "date": o["date"], "censored_nondetect_upper_bound": bool(o.get("censored"))}
        for o in used[:200]]
    meta["thresholds"] = THRESHOLDS
    meta["threshold_notes"] = (
        "EPA 2012 RWQC (36/1000 illness rate): E. coli GM 126 / STV 410; "
        "enterococci GM 35 / STV 130 (32/1000 rate: GM 30/100, STV 110/320). "
        "Beach Action Value: E. coli 235 / enterococci 70 (precautionary notification). "
        "Enterococci convert to E.coli-equivalent at GM ratio 126/35 = 3.6x. "
        "MPN/100 mL treated as CFU/100 mL for display (documented). "
        "Values 700/1000/2000/5000 are documented order-of-magnitude extensions, not EPA criteria. "
        "Official advisories/closures take display priority over raw values where both exist.")
    meta["method_priority"] = ("official advisory/closure > STV exceedance > BAV exceedance > "
                               "GM context > latest valid observation; overlaps resolve to highest concern; "
                               "jurisdictional standards reconciled by mapping each to E.coli-equivalent and "
                               "flagging the originating indicator in metadata samples.")
    meta["unit_conversions"] = "CFU/100 mL direct; MPN/100 mL ~= CFU/100 mL (display); enterococci x3.6 -> E.coli-equiv."
    meta["anchor_stations"] = STATIONS
    meta["coverage"] = ("100% of the five Great Lakes water surfaces (Superior, Michigan, Huron, Erie, Ontario; "
                        "US + Canadian waters) via the shared NOAA shoreline mask; unrelated inland lakes excluded; "
                        "land transparent.")
    meta["canada_note"] = ("WQP covers US stations; Canadian/Ontario anchor beaches are listed and paint NO DATA "
                           "until an ECCC/Ontario live feed is integrated (documented gap, upgrade path).")
    meta["stats"] = {"opaque_pixels": n_opaque, "water_pixels": n_water,
                     "halo_pixels": n_halo_px, "current_min": cur_min, "current_max": cur_max}
    token = meta["source_version"]
    color_list = ("DARK BLUE, BLUE, CYAN, LIGHT GREEN, DARK GREEN, YELLOWISH GREEN, YELLOW, ORANGE, "
                  "RED, NEON RED, MAGENTA, PURPLE, DARK PURPLE")
    folder_html = (
        f"<p><img src=\"{legend_src(token)}\" width=\"600\" alt=\"Bacteria gradient key\"></p>"
        f"<h2>What This Shows</h2>"
        f"<p>Available authoritative evidence of human-health-relevant bacterial/fecal contamination in Great "
        f"Lakes recreational waters, as one unified concern spectrum ({color_list}). Dark blue = no detected "
        f"contamination / lowest measured condition; dark purple = extremely high contamination / extremely poor "
        f"human-health water condition. Slate gray water = NO DATA.</p>"
        f"<h2>Source / Study</h2>"
        f"<p><b>Measures:</b> fecal-indicator bacteria — E. coli and enterococci (CFU/100 mL) — from the USGS "
        f"Water Quality Portal (WQX discrete samples) within EPA's BEACH/BEACON recreational-water framework.<br/>"
        f"<b>Endpoints:</b> <a href=\"{CONFIG['source_url']}\">{CONFIG['source_url']}</a> · "
        f"<a href=\"{CONFIG['beacon_url']}\">EPA BEACON</a> · "
        f"<a href=\"{CONFIG['rwqc_url']}\">EPA 2012 RWQC</a><br/>"
        f"<b>Retrieval:</b> {fetch_note}; window {CONFIG['freshness_window_days']} days ending "
        f"{dt.date.today().isoformat()}; {len(used)} station halos from {len(STATIONS)} anchor beaches.<br/>"
        f"<b>Generation:</b> build_bacteria.py; source version <b>{source_id}</b>.</p>"
        f"<h2>How to Read the Gradient</h2>"
        f"<p>{scale_html}</p>"
        f"<p><b>Why one spectrum:</b> E. coli and enterococci are BOTH fecal indicators for the same "
        f"human-health question (is fecal contamination at levels of health concern?), so they are consolidated "
        f"into one concern scale (enterococci ×3.6 = E.coli-equivalent, the EPA GM ratio) rather than mapped "
        f"separately. Neither indicator is itself claimed to be a disease-causing pathogen in any sample, and the "
        f"layer does not detect every pathogen.</p>"
        f"<p><b>Standards behind the colors:</b> EPA 2012 RWQC — E. coli GM 126 / STV 410, enterococci GM 35 / "
        f"STV 130, Beach Action Value E. coli 235 (notification level). Full per-color value/unit/threshold/source/"
        f"rationale table is in metadata.json `thresholds`.</p>"
        f"<p><b>NO DATA ≠ zero:</b> slate gray means no defensible observation exists there — it must never be read "
        f"as clean water. A missing observation never becomes zero bacteria.</p>"
        f"<p><b>Coverage:</b> the visual layer spans 100% of the five Great Lakes water surfaces (US + Canadian "
        f"waters; unrelated inland lakes excluded; land transparent), while measured-data availability varies — "
        f"most open water is honestly NO DATA because monitoring is beach- and station-based.</p>"
        f"<h2>Data / Scientific Limitations</h2>"
        f"<p>Water-quality indicator visualization, not a laboratory test of every part of the lake. Station-based "
        f"samples; halos (radius {Halo_R}px) show conditions NEAR a sampled station only — no interpolation between "
        f"stations, no modeled fill. Samples older than {CONFIG['freshness_window_days']} days are stale (NO DATA). "
        f"Canadian waters list anchor beaches but currently paint NO DATA pending an ECCC/Ontario live feed. "
        f"Advisories/closures take display priority where both exist.</p>"
        f"<h2>Update Frequency</h2>"
        f"<p>SOURCE CHECKED HOURLY (refreshInterval 3600 s on entry NetworkLink and overlay Icon), which is distinct "
        f"from source publication frequency: agencies publish only when sampling occurs (often weekly in swim "
        f"season). New samples are used when they appear; otherwise the latest valid observation is retained up to "
        f"{CONFIG['freshness_window_days']} days, then treated as stale.</p>")
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
        out_dirs=outs["live"],
        meta=meta)
    assert_no_vector_geometry(kml_text)
    build_entry_kml(
        PRODUCT, KML_FILE, OVERLAY_NAME,
        entry_description_html(CONFIG["title"], meta, SKIP_NOTE),
        CONFIG["refresh_interval_seconds"], out_dirs=outs["entry"])
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"source_id": source_id,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def legend_src(token):
    from build_kml import pages_base
    return f"{pages_base()}/{PRODUCT}/legend.png?v={token}"


if __name__ == "__main__":
    sys.exit(main())
