"""Shared derived moisture-hazard metrics for the fog + condensation layers.

Both indices are DERIVED from authoritative NOAA/NCEP HRRR 2 m analysis
fields (TMP/DPT/RH) plus, for fog only, HRRR 10 m wind (UGRD/VGRD) and
HRRR surface visibility (VIS) as an active-fog confirmation gate.
Neither index duplicates relative humidity, dew point, or visibility:
each combines the inputs with different, documented weightings toward a
different physical question.

All inputs/outputs are numpy arrays in SI-ish display units:
  tmp_c   : 2 m air temperature, degrees Celsius
  dpt_c   : 2 m dew point temperature, degrees Celsius
  rh      : 2 m relative humidity, percent 0..100
  vis_m   : HRRR surface visibility, metres (fog only)
  wspd    : 10 m wind speed, m/s (fog only)

Formulas (documented identically in DATA_SOURCES.md):
  spread S = tmp_c - dpt_c (dew-point depression, C deg)

  CONDENSATION INDEX (0..100, no wind, no visibility):
    spread_term = clip((2.5 - S) / 2.5, 0, 1) ** 0.7
    rh_term     = clip((RH - 60) / 40, 0, 1)
    condensation = 100 * (0.65 * spread_term + 0.35 * rh_term)
  Question answered: "how close is near-surface air to saturation right
  now" (favorable conditions for dew/frost/film condensation on exposed
  surfaces). Independent of whether fog is actually present.

  FOG RISK INDEX (0..100, adds wind + visibility confirmation):
    rh_factor     = clip((RH - 70) / 30, 0, 1)
    spread_factor = clip((3.0 - S) / 3.0, 0, 1)
    calm_factor   = clip((6.0 - wspd) / 6.0, 0.35, 1.0)
    base = 100 * rh_factor * spread_factor * calm_factor
    active-fog gate: where VIS < 1000 m AND RH >= 95 -> risk >= 85
                     where 1000 <= VIS < 5000 m      -> risk >= 60
    (visibility can only RAISE the index: an observed low-visibility
    field confirms fog the thermodynamic terms may understate.)
  Question answered: "is fog likely forming / already present" for
  navigation, shoreline, aviation, and nighttime monitoring.
"""

import numpy as np


def _clip01(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def condensation_index(tmp_c, rh, dpt_c):
    """0..100 condensation-favorability index. NaN in -> NaN out."""
    tmp_c = np.asarray(tmp_c, dtype=float)
    rh = np.asarray(rh, dtype=float)
    dpt_c = np.asarray(dpt_c, dtype=float)
    with np.errstate(invalid="ignore"):
        spread = tmp_c - dpt_c
        spread_term = _clip01((2.5 - spread) / 2.5) ** 0.7
        rh_term = _clip01((rh - 60.0) / 40.0)
        out = 100.0 * (0.65 * spread_term + 0.35 * rh_term)
    bad = ~(np.isfinite(tmp_c) & np.isfinite(rh) & np.isfinite(dpt_c))
    return np.where(bad, np.nan, out)


def fog_risk_index(tmp_c, rh, dpt_c, vis_m, wspd_ms):
    """0..100 fog-risk index with active-fog visibility gate."""
    tmp_c = np.asarray(tmp_c, dtype=float)
    rh = np.asarray(rh, dtype=float)
    dpt_c = np.asarray(dpt_c, dtype=float)
    vis_m = np.asarray(vis_m, dtype=float)
    wspd = np.asarray(wspd_ms, dtype=float)
    with np.errstate(invalid="ignore"):
        spread = tmp_c - dpt_c
        rh_factor = _clip01((rh - 70.0) / 30.0)
        spread_factor = _clip01((3.0 - spread) / 3.0)
        calm_factor = np.clip((6.0 - wspd) / 6.0, 0.35, 1.0)
        base = 100.0 * rh_factor * spread_factor * calm_factor
        active = np.isfinite(vis_m) & np.isfinite(rh) & (vis_m < 1000.0) & (rh >= 95.0)
        base = np.where(active, np.maximum(base, 85.0), base)
        reduced = np.isfinite(vis_m) & (vis_m >= 1000.0) & (vis_m < 5000.0)
        base = np.where(reduced & ~active, np.maximum(base, 60.0), base)
    bad = ~(np.isfinite(tmp_c) & np.isfinite(rh) & np.isfinite(dpt_c))
    return np.where(bad, np.nan, base)
