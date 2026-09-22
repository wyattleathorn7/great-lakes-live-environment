"""Shared continuous-gradient scale engine for the post-wind products.

Each product keeps a HISTORICAL RECORD in its state dir:
  {hist_min, hist_max, percentiles, reservoir.npy (<=20k validated values)}
The color domain is [hist_min, hist_max] from ACTUAL recorded values
(fill/invalid/impossible excluded by each builder before contributing).
New validated records extend the endpoints automatically (labeled HIGHEST+).

Color mapping (continuous, no bins): percentile anchors
  [min, p5, p25, p50, p75, p95, p99, max]
sit at fixed normalized positions [0, .10, .28, .48, .66, .84, .93, 1.0],
so the common-value region owns most of the color resolution while rare
extremes compress into red-violet/deep-purple. Piecewise-linear RGB
interpolation => a small value change always yields a slightly different
color; no hard boundaries exist by construction.

Master family: dark blue -> blue -> cyan -> green -> yellow -> orange ->
red -> red-violet -> deep purple. Where negative values are physically
meaningful (air temperature), sub-zero anchors use a purple ramp
(deep purple -> blue-violet -> violet -> blue at zero) with no break at
freezing; products that cannot be negative never get purple lows.
"""

import json
import math
import os
from datetime import datetime, timezone

import numpy as np

from geospatial_utils import REPO_ROOT

MASTER = [  # (family position, rgb) in display order
    (0.00, (16, 52, 140)),    # dark blue
    (0.14, (20, 110, 200)),   # blue
    (0.28, (20, 190, 200)),   # cyan
    (0.42, (90, 190, 80)),    # green
    (0.56, (245, 215, 50)),   # yellow
    (0.68, (240, 130, 25)),   # orange
    (0.80, (205, 30, 35)),    # red
    (0.90, (150, 25, 110)),   # red-violet
    (1.00, (70, 15, 100)),    # deep purple
]

NEG_RAMP = [  # (fraction from min to 0, rgb) for sub-zero anchors
    (0.00, (40, 10, 70)),     # deep purple at the record low
    (0.40, (90, 40, 160)),    # blue-violet
    (0.75, (130, 40, 170)),   # violet
    (1.00, (16, 52, 140)),    # blue at zero
]

ANCHOR_POS = [0.0, 0.10, 0.28, 0.48, 0.66, 0.84, 0.93, 1.0]
RESERVOIR_N = 20000
RNG = np.random.default_rng(7)


def _interp_table(table, t):
    if t <= table[0][0]:
        return table[0][1]
    for (p0, c0), (p1, c1) in zip(table, table[1:]):
        if t <= p1:
            f = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            return tuple(int(round(a + (b - a) * f)) for a, b in zip(c0, c1))
    return table[-1][1]


def family_color(t):
    """Master-family RGB at normalized position t in [0,1]."""
    return _interp_table(MASTER, min(max(t, 0.0), 1.0))


def negative_color(frac):
    """Purple-ramp RGB; frac=0 at record low, 1 at zero."""
    return _interp_table(NEG_RAMP, min(max(frac, 0.0), 1.0))


def build_stops(values, allow_negative):
    """Anchor (value, rgb) stops from percentile values.

    values = [min, p5, p25, p50, p75, p95, p99, max] (ascending, finite).
    Sub-zero anchors get the purple ramp; the rest follow the master
    family by normalized position. Anchor values stay strictly increasing
    so the gradient is continuous and invertible.
    """
    stops = []
    vmin, vmax = values[0], values[-1]
    span = vmax - vmin if vmax > vmin else 1.0
    for v, pos in zip(values, ANCHOR_POS):
        if allow_negative and v < 0 and vmin < 0:
            frac = (v - vmin) / (min(0.0, vmax) - vmin) if min(0.0, vmax) > vmin else 1.0
            stops.append((v, negative_color(frac)))
        else:
            stops.append((v, family_color(pos)))
    # enforce strictly increasing values for the renderer (it interpolates
    # by value between anchors)
    if allow_negative and values[0] < 0 and values[-1] > 0:
        # explicit blue anchor at freezing/zero so the purple cold ramp
        # always resolves into ordinary cold blue with no break
        for i, v in enumerate(values):
            if v >= 0:
                stops.insert(i, (0.0, (16, 52, 140)))
                break
    return stops


def color_for(value, stops):
    """Interpolated RGB for a value given anchor stops; clamps out of range
    into the endpoint colors (values above max stay deep purple, never break)."""
    if not math.isfinite(value):
        return None
    if value <= stops[0][0]:
        return stops[0][1]
    if value >= stops[-1][0]:
        return stops[-1][1]
    for (v0, c0), (v1, c1) in zip(stops, stops[1:]):
        if value <= v1:
            f = 0.0 if v1 == v0 else (value - v0) / (v1 - v0)
            return tuple(int(round(a + (b - a) * f)) for a, b in zip(c0, c1))
    return stops[-1][1]


def lut_from_stops(stops, n=256):
    """256-entry RGB LUT sampled uniformly across [min, max]."""
    vmin, vmax = stops[0][0], stops[-1][0]
    span = vmax - vmin if vmax > vmin else 1.0
    return [color_for(vmin + span * i / (n - 1), stops) for i in range(n)]


STATE_KEYS = ("hist_min", "hist_max", "percentiles", "n_obs")


def state_dir(product):
    d = os.path.join(REPO_ROOT, "output", "state", product)
    os.makedirs(d, exist_ok=True)
    return d


def load_record(product):
    """Return (record dict, reservoir array). Missing -> (None, empty)."""
    d = os.path.join(REPO_ROOT, "output", "state", product)
    try:
        with open(os.path.join(d, "record.json")) as f:
            rec = json.load(f)
        res = np.load(os.path.join(d, "reservoir.npy"))
        return rec, np.asarray(res, dtype=float).ravel()
    except Exception:
        return None, np.array([], dtype=float)


def save_record(product, rec, reservoir):
    d = state_dir(product)
    with open(os.path.join(d, "record.json"), "w") as f:
        json.dump(rec, f, indent=2)
    np.save(os.path.join(d, "reservoir.npy"),
            np.asarray(reservoir, dtype=np.float32))


def update_record(rec, reservoir, new_values):
    """Merge validated values: extend extrema on records, refresh reservoir
    (bounded random replacement => distribution tracks history), recompute
    percentiles. Returns (rec, reservoir)."""
    vals = np.asarray(new_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return rec, reservoir
    if rec is None:
        rec = {"hist_min": float(vals.min()), "hist_max": float(vals.max()),
               "n_obs": 0, "extends": []}
    extended = []
    if float(vals.min()) < rec["hist_min"]:
        rec["hist_min"] = float(vals.min())
        extended.append("low")
    if float(vals.max()) > rec["hist_max"]:
        rec["hist_max"] = float(vals.max())
        extended.append("high")
    if extended:
        rec["extends"].append({
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "kind": "+".join(extended)})
    res = np.asarray(reservoir, dtype=float).ravel()
    res = res[np.isfinite(res)]
    pool = np.concatenate([res, vals]) if res.size else vals
    if pool.size > RESERVOIR_N:
        idx = RNG.choice(pool.size, RESERVOIR_N, replace=False)
        pool = pool[idx]
    rec["n_obs"] = int(rec.get("n_obs", 0)) + int(vals.size)
    qs = np.percentile(pool, [5, 25, 50, 75, 95, 99]).tolist()
    rec["percentiles"] = {"p5": qs[0], "p25": qs[1], "p50": qs[2],
                          "p75": qs[3], "p95": qs[4], "p99": qs[5]}
    return rec, pool


def anchor_values(rec):
    """[min, p5, p25, p50, p75, p95, p99, max] from a record."""
    p = rec["percentiles"]
    return [rec["hist_min"], p["p5"], p["p25"], p["p50"], p["p75"],
            p["p95"], p["p99"], rec["hist_max"]]


def render_rgba(field, stops, alpha, missing_value=None):
    """Map a 2D float field through anchor stops to RGBA uint8.

    NaN (and optional missing sentinel) -> transparent; out-of-range
    values clamp into endpoint colors (HIGHEST+ stays deep purple).
    """
    H, W = field.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    ok = np.isfinite(field)
    if missing_value is not None:
        ok = ok & (field != missing_value)
    if not np.any(ok):
        return rgba
    lut = np.array(lut_from_stops(stops), dtype=np.uint8)
    vmin, vmax = stops[0][0], stops[-1][0]
    span = vmax - vmin if vmax > vmin else 1.0
    idx = np.clip(((field[ok] - vmin) / span * 255), 0, 255).astype(int)
    rgba[ok, 0:3] = lut[idx]
    rgba[ok, 3] = alpha
    return rgba


def draw_scale_legend(path, title, subtitle, unit_label, stops, labels,
                      source_line, note=None):
    """Continuous key image painted with the SAME stops as the raster.

    labels = [(value, text)] drawn at true scale positions (e.g. LOWEST,
    common percentiles, HIGHEST+). Returns (W, H).
    """
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
    lut = lut_from_stops(stops, bw)
    for i, c in enumerate(lut):
        d.line([(bx + i, by), (bx + i, by + bh)], fill=tuple(c) + (255,))
    d.rectangle([bx, by, bx + bw - 1, by + bh], outline=(40, 40, 40))
    vmin, vmax = stops[0][0], stops[-1][0]
    span = vmax - vmin if vmax > vmin else 1.0
    for val, text in labels:
        frac = min(max((val - vmin) / span, 0.0), 1.0)
        x = bx + int(frac * (bw - 1))
        tw = d.textlength(text, font=f_small)
        d.text((min(max(x - tw / 2, 2), W - tw - 2), by + bh + 4),
               text, font=f_small, fill=(10, 10, 10))
    d.text((bx + bw - 70, by + bh + 24), unit_label, font=f_body, fill=(10, 10, 10))
    d.text((14, H - 42), source_line, font=f_small, fill=(60, 60, 60))
    if note:
        d.text((14, H - 24), note, font=f_small, fill=(60, 60, 60))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return W, H


def fmt_val(v, unit_decimals=2):
    """Compact number formatting for legend labels."""
    a = abs(v)
    if a != 0 and (a >= 1000 or a < 0.01):
        return f"{v:.2e}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3f}".rstrip("0").rstrip(".")
