"""Beaufort wind-force scale (authoritative, unmodified thresholds).

Standard scale (WMO): force from wind speed in knots.
"""

# (force, description, kt_min_inclusive, kt_max_inclusive or None)
BEAUFORT = [
    (0, "Calm", 0.0, 0.99),
    (1, "Light air", 1, 3),
    (2, "Light breeze", 4, 6),
    (3, "Gentle breeze", 7, 10),
    (4, "Moderate breeze", 11, 16),
    (5, "Fresh breeze", 17, 21),
    (6, "Strong breeze", 22, 27),
    (7, "High wind", 28, 33),
    (8, "Gale", 34, 40),
    (9, "Strong gale", 41, 47),
    (10, "Storm", 48, 55),
    (11, "Violent storm", 56, 63),
    (12, "Hurricane-force", 64, None),
]

MS_TO_KT = 1.94384

# Force -> raster color. Blue (calm) through red (storm) to dark purple,
# which is reserved EXCLUSIVELY for Beaufort Force 12 (>=64 kt).
FORCE_COLORS = {
    0: "#1B4FD8", 1: "#1E78E0", 2: "#14A8DC", 3: "#14C8C8",
    4: "#3CBE50", 5: "#96D232", 6: "#FAC828", 7: "#FAA01E",
    8: "#F07814", 9: "#DC281E", 10: "#C81428", 11: "#A50A3C",
    12: "#3B0A54",
}


def force_from_kt(kt):
    """Beaufort force (0-12) for wind speed in knots. Raises on bad input."""
    if kt is None or not kt >= 0:
        raise ValueError(f"invalid wind speed: {kt!r}")
    force = 0
    for f, _name, kmin, _kmax in BEAUFORT:
        if kt >= kmin:
            force = f
    return force


def force_range_text(f):
    for force, _name, kmin, kmax in BEAUFORT:
        if force == f:
            return f"{kmin:g}\u2013{kmax:g} kt" if kmax is not None else f"\u2265{kmin:g} kt"
    raise ValueError(f"unknown force {f}")


def force_name(f):
    for force, name, _kmin, _kmax in BEAUFORT:
        if force == f:
            return name
    raise ValueError(f"unknown force {f}")


def uv_to_speed_dir(u, v):
    """U (eastward), V (northward) m/s movement components -> (speed, math angle).

    GRIB UGRD/VGRD point in the direction the air MOVES TOWARD (unlike the
    meteorological 'FROM' convention), so no reversal is applied. Returns
    speed (same units as input) and the math angle in radians measured
    counter-clockwise from east.
    """
    import math
    return math.hypot(u, v), math.atan2(v, u)


def movement_to_canvas_dxdy(u, v):
    """Movement vector -> canvas pixel delta (x right, y DOWN)."""
    return u, -v
