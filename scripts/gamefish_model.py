"""Game-fish distribution model: six separate components + weighted combination.

Live modeled distribution / habitat likelihood for nine Great Lakes game-fish
species + Lake Sturgeon. NOT a census, NOT fish counts, NOT confirmed presence.

Components (each 0..1, preserved separately by the builder):
  T  telemetry evidence  (behavioral evidence only; spatial+temporal decay;
                          unresolved tags never attributed; dismantled/invalid
                          excluded; historical GLATOS deployments = downweighted
                          behavioral prior, never current counts)
  M  seasonal migration  (date-aware windows from per-species config)
  TH thermal suitability (Gaussian around preferred_c on live GLSEA SST;
                          surface-only LIMITATION)
  D  diel behavior       (solar-elevation day/night factor modulating
                          nearshore/offshore patterns)
  H  habitat suitability (shore-proximity proxy; no bathymetry LIMITATION)
  C  movement corridors  (named documented systems incl. connecting waters)
Final = weighted arithmetic mean (documented per-species weights; NEVER blind
multiplication). Alpha scaled by confidence/support (agreement + telemetry bonus).

Research basis: GLFC Special Pub 87-3 (thermal backbone); USGS Edsall & Cleland
2000 + MI DNR RR1871 + Vandergoot et al. (lake trout); USFWS profile + Auer et
al. (sturgeon); Raby et al. 2018 + WI DNR (walleye); GLATOS project timing.
Chinook/Coho excluded: no tag evidence in the telemetry corpus
(insufficient-data). Decay constants are MODEL_ASSUMPTIONs (see configs).
"""

import math
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geospatial_utils import (apply_shoreline_mask, load_bounds, load_watermask)

MODEL_VERSION = "1.1.0"

# Single unified spectrum for ALL nine species (dark blue = lowest likelihood,
# dark red = highest). One shared meaning and progression everywhere.
UNIFIED_STOPS = [
    (13, 42, 120),    # dark blue: none/lowest
    (20, 110, 220),   # blue
    (25, 180, 220),   # cyan
    (60, 190, 120),   # green
    (240, 220, 60),   # yellow
    (245, 150, 30),   # orange
    (210, 40, 30),    # red
    (140, 10, 15),    # dark red: highest
]

# Named movement-corridor geography: documented biological movement systems
# (spawning rivers, connecting channels, migration axes), NOT lines between
# receivers. Rectangles in lon/lat. Connecting waters are wide boxes so narrow
# channels survive rasterization.
CORRIDORS = {
    "maumee_bay_reefs": (-83.85, 41.35, -82.90, 42.00),
    "sandusky_river": (-83.15, 41.25, -82.70, 41.55),
    "erie_west_to_central": (-83.60, 41.30, -81.00, 42.40),
    "detroit_river": (-83.25, 41.75, -82.95, 42.35),
    "lake_st_clair": (-82.95, 42.25, -82.40, 42.70),
    "st_clair_river": (-82.65, 42.70, -82.35, 43.25),
    "st_marys_river": (-84.75, 45.95, -83.90, 46.55),
    "saginaw_bay": (-84.30, 43.40, -83.40, 44.10),
    "saginaw_river": (-84.05, 43.35, -83.80, 43.70),
    "green_bay": (-88.30, 44.40, -87.60, 45.40),
    "superior_shoals": (-92.20, 46.60, -84.50, 48.00),
    "huron_reefs": (-83.50, 43.50, -81.50, 46.00),
    "huron_shoals": (-83.00, 43.80, -81.80, 45.50),
    "michigan_reefs": (-87.50, 43.50, -85.80, 46.00),
    "michigan_tribs": (-87.80, 41.80, -85.80, 46.00),
    "huron_tribs": (-83.80, 43.00, -81.50, 46.20),
    "ontario_tribs": (-80.50, 43.10, -76.20, 44.10),
    "ontario_east": (-78.50, 43.20, -76.20, 44.10),
    "superior_tribs": (-92.30, 46.50, -84.40, 48.10),
    "niagara_shore_erie": (-79.20, 42.60, -78.80, 43.00),
}

# (Retired in model 1.1.0: all species share UNIFIED_STOPS. The table is
# kept for provenance so past renders remain interpretable.)
SPECIES_HUES_RETIRED_V1_0 = {
    "walleye": [(16, 42, 28), (27, 94, 52), (64, 150, 80), (150, 200, 110), (235, 225, 130)],
    "yellow_perch": [(46, 32, 8), (128, 84, 16), (205, 140, 30), (240, 190, 80), (250, 235, 160)],
    "lake_trout": [(8, 24, 60), (20, 70, 140), (30, 130, 200), (120, 200, 235), (225, 245, 255)],
    "steelhead": [(40, 16, 56), (110, 40, 140), (170, 80, 190), (220, 150, 220), (245, 220, 250)],
    "brown_trout": [(46, 26, 10), (120, 70, 25), (185, 120, 45), (225, 175, 100), (250, 230, 180)],
    "smallmouth_bass": [(20, 40, 12), (60, 110, 40), (110, 165, 70), (180, 210, 110), (240, 245, 180)],
    "northern_pike": [(10, 34, 30), (20, 100, 85), (60, 170, 120), (150, 220, 150), (230, 250, 200)],
    "muskellunge": [(30, 28, 30), (90, 85, 95), (150, 145, 155), (205, 200, 210), (245, 243, 245)],
    "lake_sturgeon": [(24, 20, 40), (70, 60, 120), (120, 110, 180), (185, 175, 225), (235, 230, 250)],
}

DISCLAIMER = ("Live modeled distribution estimate based on current environmental "
              "conditions, known species behavior, movement knowledge, and available "
              "telemetry evidence. This layer does not represent a census or confirmed "
              "presence of individual fish.")

_WATER = None
_SHORE = None
_OFF = None


def _water():
    global _WATER
    if _WATER is None:
        _WATER = load_watermask() > 0.5
    return _WATER


def _blur(a, sigma):
    m = float(a.max()) or 1.0
    img = Image.fromarray((np.clip(a, 0, None) / m * 255).astype(np.uint8))
    b = np.array(img.filter(ImageFilter.GaussianBlur(float(sigma))),
                 dtype=np.float32) / 255.0
    return (b * m).astype(np.float32)


def shore_proximity():
    """Nearshore proxy (high at coast) from blurred land mask."""
    global _SHORE
    if _SHORE is None:
        w = _water().astype(np.float32)
        s = _blur(1.0 - w, 12.0) * w
        _SHORE = (s / (s.max() or 1.0)).astype(np.float32)
    return _SHORE


def offshore_proximity():
    """Offshore proxy (high in lake centers) from wide-blurred water mask."""
    global _OFF
    if _OFF is None:
        w = _water().astype(np.float32)
        o = _blur(w, 40.0) * w
        _OFF = (o / (o.max() or 1.0)).astype(np.float32)
    return _OFF


def lonlat_to_rowcol(lons, lats):
    b = load_bounds()
    W, H = b["canvas_width"], b["canvas_height"]
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    cols = ((lons - b["lon_min"]) / (b["lon_max"] - b["lon_min"]) * W).astype(int)
    rows = ((b["lat_max"] - lats) / (b["lat_max"] - b["lat_min"]) * H).astype(int)
    valid = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    return rows, cols, valid


def gaussian_splat(rows, cols, weights, shape, sigma_px=8.0):
    """Log-compressed amplitude + fixed blur + normalization (no giant hotspots)."""
    H, W = shape
    grid = np.zeros((H, W), dtype=np.float64)
    for r, c, w in zip(rows, cols, weights):
        if 0 <= r < H and 0 <= c < W:
            grid[r, c] += w
    grid = np.log1p(grid)
    if grid.max() > 0:
        grid = grid / grid.max()
    grid = _blur(grid.astype(np.float32), sigma_px)
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid.astype(np.float32)


# ---------------- T: telemetry evidence ----------------
def build_telemetry(snapshot_live, history_resolved, audit, registry_name,
                    species_projects, tau_hours, sigma_deg, now_utc):
    b = load_bounds()
    H, W = b["canvas_height"], b["canvas_width"]
    rows, cols, weights = [], [], []
    evidence_events = 0
    newest = None
    deg_per_px = (b["lon_max"] - b["lon_min"]) / b["canvas_width"]
    sig_px = max(4.0, sigma_deg / deg_per_px)
    for rec in (snapshot_live or []):
        ev = 0
        for k, v in (rec.get("species_counts_24h", {}) or {}).items():
            if registry_name.lower() in str(k).lower() or str(k).lower() in registry_name.lower():
                ev += v
        if ev <= 0:
            continue
        age_h = 12.0
        try:
            t = datetime.strptime(rec.get("last_detection", ""),
                                  "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            age_h = max(0.0, (now_utc - t).total_seconds() / 3600.0)
        except Exception:
            pass
        r, c, valid = lonlat_to_rowcol([rec["longitude"]], [rec["latitude"]])
        if valid[0]:
            rows.append(int(r[0]))
            cols.append(int(c[0]))
            weights.append(math.log1p(ev) * math.exp(-age_h / tau_hours))
            evidence_events += ev
            newest = rec.get("last_detection")
    for e in (history_resolved or []):
        # caller pre-resolves tags to this species; weight by recency decay
        try:
            t = datetime.strptime(e["t"], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc)
            age_h = max(0.0, (now_utc - t).total_seconds() / 3600.0)
        except Exception:
            age_h = 72.0
        r, c, valid = lonlat_to_rowcol([e["_lon"]], [e["_lat"]])
        if valid[0]:
            rows.append(int(r[0]))
            cols.append(int(c[0]))
            weights.append(e.get("events", 1) * math.exp(-age_h / tau_hours))
            evidence_events += e.get("events", 1)
    prior_n = 0
    for rec in (audit or []):
        if str(rec.get("current_status", "")).startswith("DISMANTLED"):
            continue
        if rec.get("latitude") is None or rec.get("longitude") is None:
            continue
        projs = " ".join(rec.get("projects", [])).lower()
        if any(p.lower() in projs for p in species_projects):
            r, c, valid = lonlat_to_rowcol([rec["longitude"]], [rec["latitude"]])
            if valid[0]:
                rows.append(int(r[0]))
                cols.append(int(c[0]))
                weights.append(0.15)  # historical/behavioral prior, downweighted
                prior_n += 1
    field = (gaussian_splat(np.array(rows), np.array(cols), np.array(weights),
                            (H, W), sigma_px=sig_px) if rows
             else np.zeros((H, W), np.float32))
    return field, {"evidence_events": evidence_events, "prior_receivers": prior_n,
                   "newest_evidence": newest, "n_foci": len(rows)}


# ---------------- M: seasonal migration ----------------
def _in_window(today, start, end):
    s = date(2000, int(start[:2]), int(start[3:]))
    e = date(2000, int(end[:2]), int(end[3:]))
    t = date(2000, today.month, today.day)
    if s <= e:
        return s <= t <= e
    return t >= s or t <= e


def build_seasonal(today, windows, corridor_field, shore, off):
    H, W = np.asarray(corridor_field).shape
    best = np.zeros((H, W), np.float32)
    active = []
    for w in windows:
        if _in_window(today, w["start"], w["end"]):
            active.append(w["name"])
            f, s = w["focus"], float(w["strength"])
            if f == "river_mouths":
                pat = np.clip(corridor_field * 0.7 + shore * 0.5, 0, 1)
            elif f == "nearshore":
                pat = shore
            elif f == "offshore":
                pat = off
            elif f == "central_east":
                pat = np.clip(corridor_field * 0.8 + off * 0.4, 0, 1)
            elif f == "reefs_shoals":
                pat = np.clip(off * 0.8 + corridor_field * 0.4, 0, 1)
            else:
                pat = np.full((H, W), 0.5, np.float32)
            pat = 0.35 + 0.65 * pat
            cand = np.clip(s * pat, 0, 1)
            best = np.maximum(best, cand)
    if not active:
        best = np.full((H, W), 0.45, np.float32)
    else:
        best[best == 0] = 0.35
    return best.astype(np.float32), {"active_windows": active}


# ---------------- TH / D / H / C ----------------
def build_thermal(sst_c, preferred_c, sigma_c):
    out = np.full(np.asarray(sst_c).shape, 0.35, np.float32)
    ok = np.isfinite(sst_c)
    out[ok] = np.exp(-0.5 * ((sst_c[ok] - preferred_c) / sigma_c) ** 2).astype(np.float32)
    return out


def solar_elevation_deg(when_utc, lon=-84.0, lat=44.5):
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    n = when_utc.timetuple().tm_yday
    frac = (when_utc.hour + when_utc.minute / 60.0 + when_utc.second / 3600.0) / 24.0
    gamma = 2 * math.pi / 365 * (n - 1 + (frac - 0.5))
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    ha = math.radians((frac * 1440 + eqtime + 4 * lon) / 4 - 180)
    latr = math.radians(lat)
    cosz = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, cosz))))


def diel_phase(when_utc):
    el = solar_elevation_deg(when_utc)
    if el > 10:
        return "day", el, 0.0
    if el < -12:
        return "night", el, 1.0
    return "crepuscular", el, float(max(0.0, min(1.0, (10 - el) / 22.0)))


def build_diel(when_utc, cfg, shore, off):
    phase, el, night_f = diel_phase(when_utc)
    day_f = 1.0 - night_f
    field = (0.5
             + cfg["night_nearshore_boost"] * night_f * (shore - 0.5)
             + cfg["day_offshore_bias"] * day_f * (off - 0.5))
    if cfg.get("crepuscular_peak") and phase == "crepuscular":
        field = field + 0.08 * shore
    return np.clip(field, 0, 1).astype(np.float32), {
        "phase": phase, "solar_elev": round(el, 1), "night_factor": round(night_f, 3)}


def build_habitat(cfg):
    shore, off = shore_proximity(), offshore_proximity()
    h = np.clip(0.25 + cfg["nearshore_affinity"] * shore
                + cfg["offshore_affinity"] * off, 0, 1).astype(np.float32)
    h[~_water()] = 0
    return shore, off, h


def build_corridors(names_weights, smooth_px=6):
    b = load_bounds()
    H, W = b["canvas_height"], b["canvas_width"]
    field = np.zeros((H, W), np.float32)
    lons = np.linspace(b["lon_min"], b["lon_max"], W)
    lats = np.linspace(b["lat_max"], b["lat_min"], H)
    for name, w in names_weights:
        box = CORRIDORS.get(name)
        if not box:
            continue
        x0, y0, x1, y1 = box
        m = ((lons[None, :] >= x0) & (lons[None, :] <= x1)
             & (lats[:, None] >= y0) & (lats[:, None] <= y1)).astype(np.float32)
        m = _blur(m, smooth_px)
        if m.max() > 0:
            m /= m.max()
        field = np.maximum(field, m * float(w))
    field[~_water()] = 0
    return field.astype(np.float32)


# ---------------- combination + confidence ----------------
def combine(comps, weights):
    keys = ["T", "M", "TH", "D", "H", "C"]
    tot = sum(weights[k] for k in keys)
    return np.clip(sum(comps[k] * weights[k] for k in keys) / tot, 0, 1).astype(np.float32)


def confidence_value(comps, water, telemetry_bonus=0.25):
    keys = ["M", "TH", "D", "H", "C"]
    agree = sum((comps[k] > 0.5).astype(float) for k in keys) / len(keys)
    tele = (comps["T"] > 0.3).astype(float) if "T" in comps else 0.0
    a = float(agree[water].mean()) if water.any() else 0.0
    t = float(tele[water].mean()) if water.any() else 0.0
    return float(max(0.0, min(1.0, 0.35 + 0.65 * a + telemetry_bonus * t)))


# ---------------- rendering ----------------
def _interp(stops, t):
    n = len(stops) - 1
    x = min(max(t, 0.0), 1.0) * n
    i = min(int(x), n - 1)
    f = x - i
    return tuple(int(round(a + (b - a) * f)) for a, b in zip(stops[i], stops[i + 1]))


def render_species(final, conf, species_key, alpha=205):
    stops = UNIFIED_STOPS  # one shared spectrum for all species (v1.1.0)
    water = _water()
    rgba = np.zeros(final.shape + (4,), np.uint8)
    f = np.clip(final, 0, 1)[water]
    lut = np.array([_interp(stops, i / 255) for i in range(256)], np.uint8)
    rgba[water, :3] = lut[(f * 255).astype(int)]
    a = (alpha * (0.35 + 0.65 * conf) * np.ones_like(f)).astype(np.uint8)
    a[f < 0.03] = 0  # no fake coverage: near-zero stays transparent
    rgba[water, 3] = a
    return apply_shoreline_mask(rgba)


LEGEND_SIZE = (640, 300)


def _font(size):
    from PIL import ImageFont
    for p in ("DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_fish_legend(path, species_key, label, lines_top, lines_bottom, model_version):
    W, H = LEGEND_SIZE
    img = Image.new("RGBA", (W, H), (255, 255, 255, 235))
    d = ImageDraw.Draw(img)
    ft, fb = _font(20), _font(13)
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=2)
    d.text((12, 6), f"\U0001F41F {label} \u2014 LIVE MODELED DISTRIBUTION",
           font=ft, fill=(10, 10, 10))
    y = 36
    for s in lines_top:
        d.text((12, y), s, font=fb, fill=(40, 40, 40))
        y += 17
    stops = UNIFIED_STOPS  # shared spectrum; legend matches the raster
    bx, by, bw, bh = 12, y + 4, W - 24, 26
    for i in range(bw):
        d.line([(bx + i, by), (bx + i, by + bh)],
               fill=_interp(stops, i / (bw - 1)) + (255,))
    d.rectangle([bx, by, bx + bw - 1, by + bh], outline=(40, 40, 40))
    d.text((bx, by + bh + 3), "LOW", font=fb, fill=(10, 10, 10))
    d.text((bx + 60, by + bh + 3),
           "MODELED DISTRIBUTION / HABITAT INDEX (not fish count)",
           font=fb, fill=(10, 10, 10))
    d.text((W - 60, by + bh + 3), "HIGH", font=fb, fill=(10, 10, 10))
    y2 = by + bh + 24
    for s in lines_bottom:
        d.text((12, y2), s, font=fb, fill=(60, 60, 60))
        y2 += 16
    d.text((12, H - 18),
           f"Model {model_version}. Transparency = low confidence / outside modeled water.",
           font=fb, fill=(60, 60, 60))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path
