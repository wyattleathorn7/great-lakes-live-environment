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

SNOW_FAMILY = [  # trace silver -> blue -> purple -> magenta -> pink -> white
    (0.00, (192, 200, 208)),  # silver-gray (trace)
    (0.14, (184, 212, 232)),  # pale blue
    (0.28, (127, 212, 232)),  # light cyan
    (0.42, (46, 127, 208)),   # medium blue
    (0.56, (30, 72, 200)),    # royal blue
    (0.68, (90, 46, 200)),    # blue-violet
    (0.80, (122, 30, 168)),   # purple
    (0.88, (160, 24, 144)),   # deep magenta
    (0.94, (232, 72, 160)),   # pink
    (1.00, (255, 255, 255)),  # white (extreme only)
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


def family_color(t, table=None):
    """Master-family RGB at normalized position t in [0,1]."""
    return _interp_table(table or MASTER, min(max(t, 0.0), 1.0))


def negative_color(frac):
    """Purple-ramp RGB; frac=0 at record low, 1 at zero."""
    return _interp_table(NEG_RAMP, min(max(frac, 0.0), 1.0))


def build_stops(values, allow_negative, family=None):
    """Anchor (value, rgb) stops from percentile values.

    values = [min, p5, p25, p50, p75, p95, p99, max] (ascending, finite).
    Sub-zero anchors get the purple ramp; the rest follow the given family
    (default master) by normalized position. Anchor values stay strictly
    increasing so the gradient is continuous and invertible.
    """
    fam = family or MASTER
    stops = []
    vmin, vmax = values[0], values[-1]
    span = vmax - vmin if vmax > vmin else 1.0
    for v, pos in zip(values, ANCHOR_POS):
        if allow_negative and v < 0 and vmin < 0:
            frac = (v - vmin) / (min(0.0, vmax) - vmin) if min(0.0, vmax) > vmin else 1.0
            stops.append((v, negative_color(frac)))
        else:
            stops.append((v, _interp_table(fam, pos)))
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


# ---- OKLab perceptually-uniform interpolation (Björn Ottosson) ----
def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * (np.clip(c, 0, None) ** (1 / 2.4)) - 0.055)


def rgb_to_oklab(rgb):
    """sRGB 0-255 tuple/array -> OKLab (L, a, b)."""
    import numpy as _np
    r, g, b = [_srgb_to_linear(_np.asarray(rgb[..., i], dtype=float))
               for i in range(3)]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                     1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                     0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_],
                    axis=-1)


def oklab_to_rgb(lab):
    """OKLab -> sRGB 0-255 uint8 (gamut-clipped)."""
    import numpy as _np
    L, a, b_ = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b_
    m_ = L - 0.1055613458 * a - 0.0638541728 * b_
    s_ = L - 0.0894841775 * a - 1.2914855480 * b_
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    rgb = np.stack([_linear_to_srgb(r), _linear_to_srgb(g),
                    _linear_to_srgb(b)], axis=-1)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def oklab_lut(hex_anchors, n=256):
    """Continuous LUT from [(position 0..1, '#hex')] via OKLab.

    Positions need not be uniform; first and last SHOULD match for a
    wraparound scale (checked by tests). Returns list of (r, g, b).
    """
    import numpy as _np
    pts = sorted(hex_anchors)
    labs = _np.array([rgb_to_oklab(
        _np.array([int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)]))
        for _, h in pts])
    poss = _np.array([p for p, _ in pts])
    t = _np.linspace(0, 1, n)
    out = _np.empty((n, 3))
    for k in range(3):
        out[:, k] = _np.interp(t, poss, labs[:, k])
    return [tuple(int(v) for v in px) for px in oklab_to_rgb(out)]


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
