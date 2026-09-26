"""Annual leaf-color phenology engine (shared, tested by selftest.py).

Pipeline per valid land pixel:
  NDVI observations (current + history) -> smoothed signal + seasonal
  baseline (rolling max) + direction of change -> normalized circular
  phenology phase 0..1 (0/1 = dormant winter) -> land-class modulation ->
  spectral autumn/snow gating -> continuous circular color LUT.

The LUT anchors below are the spec's exact colors; _interp positions give
a mathematically continuous, wraparound-identical scale.
"""

import math

# Spec §7 anchors (phase position, hex). Beginning and end are the
# identical deep blue so the annual cycle wraps with no seam.
PHASE_ANCHORS = [
    (0.00, "#123B73"), (0.08, "#1679A8"), (0.16, "#19C6D1"),
    (0.24, "#4FD5C4"), (0.31, "#82DC9A"), (0.37, "#A9DF6B"),
    (0.43, "#42B84A"), (0.49, "#218C3A"), (0.54, "#A9C84A"),
    (0.59, "#E2DD62"), (0.64, "#F2C84B"), (0.69, "#E9A83A"),
    (0.74, "#E77A2E"), (0.78, "#D94B35"), (0.82, "#C6283C"),
    (0.86, "#9F243B"), (0.90, "#671F46"), (0.95, "#432050"),
    (1.00, "#123B73"),
]

# class code -> (name, amplitude weight, phase cap or None)
# (class grid built by scripts/build_leaf_landcover.py)
CLASS_INFO = {
    0: ("nodata", 0.0, None),
    1: ("deciduous", 1.0, None),
    2: ("mixed", 0.55, None),
    3: ("evergreen", 0.0, None),  # special-cased: clamped green band
    4: ("shrub", 0.45, 0.66),
    5: ("grass", 0.45, 0.66),
    6: ("crop", 0.35, 0.62),
    7: ("urban", 0.0, None),  # masked (transparent)
    8: ("barren", 0.0, None),  # masked (transparent)
    9: ("water", 0.0, None),  # masked (transparent)
}

LEAF_LUT_N = 256


def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def build_leaf_lut(n=LEAF_LUT_N):
    """256-entry circular RGB LUT interpolated in OKLab (perceptually
    smooth, no muddy intermediates); index 0 == index 255 (wraparound)."""
    from gradient_scale import oklab_lut
    lut = oklab_lut(PHASE_ANCHORS, n)
    lut[0] = lut[-1] = tuple(int(v) for v in lut[0])
    return [tuple(int(v) for v in c) for c in lut]


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def phenology_phase(ndvi_now, history, cls, redness, snow, bad_obs,
                    prev_phase=None):
    """Return (phase 0..1, hold_forward_bool).

    ndvi_now: current composite NDVI (-1..1). history: list of older valid
    NDVI (any order; median is the reference). cls: land class code.
    redness: 0..1 autumn spectral proxy. snow/bad_obs: bools. prev_phase:
    last published phase (hold-forward for rejected observations).

    Composites are already best-pixel products, so the current observation
    is trusted as the signal; history supplies only the baseline maximum
    (always floored by a class prior so spring cannot saturate) and the
    direction reference (median, robust to one bad composite).
    """
    if bad_obs:
        return (prev_phase, True) if prev_phase is not None else (None, True)
    if snow:
        return None, False  # transparent; snow is not foliage
    if cls in (0, 7, 8, 9):
        return None, False  # masked classes
    if ndvi_now is None or not math.isfinite(ndvi_now):
        return (prev_phase, True) if prev_phase is not None else (None, True)
    hist = [v for v in history if v is not None and math.isfinite(v)]
    n_ref = _median(hist) if hist else ndvi_now
    d = ndvi_now - n_ref
    prior = 0.72 if cls in (1, 2, 3) else 0.55
    n_max = max([ndvi_now] + hist + [prior])
    span = max(n_max - 0.10, 0.20)
    ngr = min(max((ndvi_now - 0.10) / span, 0.0), 1.0)

    if cls == 3:  # evergreen: clamped green/blue-green band, never autumn red
        return 0.40 + 0.16 * ngr, False
    if ngr <= 0.12 and abs(d) <= 0.03 and redness < 0.3:
        phase = 0.02 + 0.06 * (ngr / 0.12)  # dormant winter blues
    elif d > 0.02:
        phase = 0.15 + 0.30 * ngr  # spring rise
    elif d < -0.02:
        t = 1.0 - ngr
        phase = 0.55 + 0.30 * t + 0.08 * redness * t  # autumn decline
        if ngr < 0.22:  # leaf drop -> dormancy edge
            phase = max(phase, 0.85 + 0.15 * (0.22 - ngr) / 0.22)
    else:
        phase = 0.45 + 0.10 * min(max((ngr - 0.7) / 0.3, 0.0), 1.0) \
            if ngr > 0.6 else 0.10 + 0.25 * (ngr / 0.6)
    _name, w, cap = CLASS_INFO.get(cls, ("nodata", 0.0, None))
    phase = 0.5 + (phase - 0.5) * w
    if cap is not None:
        phase = min(phase, cap)
    return min(max(phase, 0.0), 0.999), False


def redness_index(red, green, blue):
    """Autumn spectral proxy 0..1 (high red chromaticity + green decline).

    Deliberately NOT a foliage classifier by itself: the engine gates it by
    trajectory (declining NDVI) and class, so summer stress cannot paint red.
    """
    tot = red + green + blue + 1e-6
    cr = red / tot
    return min(max((cr - 0.38) / 0.12, 0.0), 1.0)


def brightness(red, green, blue):
    return (red + green + blue) / 3.0


LEAF_LEGEND_PHASES = [
    (0.04, "Dormant"), (0.20, "Spring awakening"), (0.30, "Bud break"),
    (0.38, "Leaf emergence"), (0.50, "Growing season"), (0.60, "First color"),
    (0.72, "Autumn coloring"), (0.88, "Leaf drop"), (0.97, "Dormant"),
]


def draw_leaf_legend(path, title, subtitle, source_line, lut):
    """Large legible phenology key (bar + phase labels, two clear rows).

    Sized for the Google Earth info panel: big bar, staggered labels with
    room to breathe so nothing overlaps.
    """
    from PIL import Image, ImageDraw
    from geospatial_utils import _legend_font
    W, H = 760, 300
    img = Image.new("RGBA", (W, H), (255, 255, 255, 235))
    d = ImageDraw.Draw(img)
    f_title, f_body, f_small = _legend_font(26), _legend_font(17), _legend_font(15)
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=2)
    d.text((16, 10), title, font=f_title, fill=(10, 10, 10))
    d.text((16, 44), subtitle, font=f_body, fill=(40, 40, 40))
    bx, by, bw, bh = 16, 80, W - 32, 44
    for i in range(bw):
        c = lut[int(i / (bw - 1) * 255)]
        d.line([(bx + i, by), (bx + i, by + bh)], fill=tuple(c) + (255,))
    d.rectangle([bx, by, bx + bw - 1, by + bh], outline=(40, 40, 40))
    for row, items in (0, LEAF_LEGEND_PHASES[::2]), (1, LEAF_LEGEND_PHASES[1::2]):
        for phase, label in items:
            x = bx + int(phase * (bw - 1))
            tw = d.textlength(label, font=f_small)
            d.text((min(max(x - tw / 2, 2), W - tw - 2), by + bh + 8 + row * 24),
                   label, font=f_small, fill=(10, 10, 10))
    d.text((16, H - 26), source_line, font=f_small, fill=(60, 60, 60))
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return W, H
