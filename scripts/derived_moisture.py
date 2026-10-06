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

  FOG RISK INDEX v2 (0..100: risk owns 0..35, active fog owns 35..100):
    r = rh_factor * spread_factor * calm_factor  (each in [0,1], as above)
    risk_zone = 35 * r                            (no observed fog)
    corroborated density d (only where VIS < 5000 m AND RH >= 90):
        d = clip((5000 - VIS) / 5000, 0, 1) ** 0.8
    fog_zone = 35 + 65 * d                        (active fog by density)
    index = max(risk_zone, fog_zone)
  The scale is continuous at the boundary (VIS = 5000 m gives d = 0, so
  fog_zone = 35 >= risk_zone). Observed reduced visibility IS fog even
  where the thermodynamic terms understate it (e.g. advected fog), so
  corroborated density outranks risk — but corroboration requires high
  humidity, so dry-air model speckles can never paint active fog.
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
    """0..100 fog index v2: 0..35 thermodynamic risk, 35..100 active fog
    by observed density. NaN in -> NaN out."""
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
        risk = 35.0 * rh_factor * spread_factor * calm_factor
        corroborated = (np.isfinite(vis_m) & np.isfinite(rh)
                        & (vis_m < 5000.0) & (rh >= 90.0))
        density = np.where(corroborated,
                           _clip01((5000.0 - vis_m) / 5000.0) ** 0.8, 0.0)
        fogged = 35.0 + 65.0 * density
        base = np.where(corroborated, np.maximum(risk, fogged), risk)
    bad = ~(np.isfinite(tmp_c) & np.isfinite(rh) & np.isfinite(dpt_c))
    return np.where(bad, np.nan, base)
