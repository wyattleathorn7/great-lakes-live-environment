"""Shared USNIC NAIS Great Lakes SIGRID-3 shapefile helpers.

Source: https://usicecenter.gov/File/DownloadCurrent?pId=35
(daily Great Lakes ice analysis; off-season the current file is the last
spring analysis, e.g. a single ice-free CT=00 polygon).

Stage codes follow WMO SIGRID-3 Table A-3 (via NSIDC G10013 user guide,
based on WMO 2010). Thicknesses are DOCUMENTED DERIVED ESTIMATES:
range midpoints of the WMO stage thickness ranges, concentration-weighted
(see STAGE_TABLE). They are not direct measurements.
"""

import os
import re
import zipfile

NIC_SHP_URL = "https://usicecenter.gov/File/DownloadCurrent?pId=35"

# code -> {name, range text, midpoint cm (None = no authoritative thickness)}
STAGE_TABLE = {
    55: {"name": "Ice Free", "range": "open water", "mid_cm": None},
    70: {"name": "Brash Ice", "range": "thickness not coded in SIGRID-3", "mid_cm": None},
    80: {"name": "No Stage of Development", "range": "unstaged", "mid_cm": None},
    81: {"name": "New Ice", "range": "<10 cm", "mid_cm": 5.0},
    82: {"name": "Nilas, Ice Rind", "range": "<10 cm", "mid_cm": 5.0},
    83: {"name": "Young Ice", "range": "10-<30 cm", "mid_cm": 20.0},
    84: {"name": "Grey Ice", "range": "10-<15 cm", "mid_cm": 12.5},
    85: {"name": "Grey-White Ice", "range": "15-<30 cm", "mid_cm": 22.5},
    86: {"name": "First-Year Ice", "range": ">=30-200 cm", "mid_cm": 115.0},
    87: {"name": "Thin First-Year Ice", "range": "30-<70 cm", "mid_cm": 50.0},
    88: {"name": "Thin First-Year Ice Stage 1", "range": "30-<50 cm", "mid_cm": 40.0},
    89: {"name": "Thin First-Year Ice Stage 2", "range": "50-<70 cm", "mid_cm": 60.0},
    91: {"name": "Medium First-Year Ice", "range": "70-<120 cm", "mid_cm": 95.0},
    93: {"name": "Thick First-Year Ice", "range": ">=120 cm (open-ended; 150 cm assumed)", "mid_cm": 150.0},
    95: {"name": "Old Ice", "range": "WMO gives no thickness; no estimate made", "mid_cm": None},
    96: {"name": "Second-Year Ice", "range": "WMO gives no thickness; no estimate made", "mid_cm": None},
    97: {"name": "Multi-Year Ice", "range": "WMO gives no thickness; no estimate made", "mid_cm": None},
    98: {"name": "Glacier Ice", "range": "not applicable (icebergs)", "mid_cm": None},
}

# Every ice-bearing WMO stage, in legend order, plus the unknown bucket.
TYPE_ORDER = [70, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 93, 95, 96, 97, 98]

TYPE_COLORS = {
    70: "#C9A227", 80: "#6C757D", 81: "#DFF6FF", 82: "#BDE0FE",
    83: "#48CAE4", 84: "#7D8597", 85: "#E0EAF5", 86: "#1D3557",
    87: "#3A0CA3", 88: "#7209B7", 89: "#B5179E", 91: "#F72585",
    93: "#D00000", 95: "#370617", 96: "#9D0208", 97: "#FFBA08",
    98: "#FFFFFF", "unknown": "#2B2D42",
}

UNKNOWN_CODES = {99, -9}

CM_PER_IN = 2.54


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def download_nic_shapefile(raw_dir, download_fn):
    """Download current NIC shapefile zip. Returns (zip_path, info)."""
    dest = os.path.join(raw_dir, "nic_great_lakes_shp.zip")
    info = download_fn(NIC_SHP_URL, dest, timeout=180)
    return dest, info


def shapefile_base(zip_path, dest_dir):
    """Extract zip, return the .shp base path (member name varies by date)."""
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        shp = [n for n in names if n.lower().endswith(".shp")]
        if not shp:
            raise ValueError(f"no .shp member in {zip_path}: {names[:5]}")
        z.extractall(dest_dir)
    base = os.path.join(dest_dir, os.path.splitext(os.path.basename(shp[0]))[0])
    # extractall preserves subdirs; locate the actual .shp if nested
    if not os.path.exists(base + ".shp"):
        for dp, _dn, fn in os.walk(dest_dir):
            for f in fn:
                if f.lower().endswith(".shp"):
                    return os.path.join(dp, os.path.splitext(f)[0])
        raise ValueError("extracted .shp not found on disk")
    return base


def analysis_date_from_name(name):
    """NIC names like GL260516_lam / 20250221 -> 'YYYY-MM-DD'. None if unknown."""
    m = re.search(r"GL(\d{2})(\d{2})(\d{2})", name)
    if m:
        return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(19|20)(\d{2})(\d{2})(\d{2})", name)
    if m:
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}-{m.group(4)}"
    return None


def _stage_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _conc_tenths(v):
    """Concentration in tenths (0-10). None when missing/invalid."""
    try:
        c = int(str(v).strip())
    except (TypeError, ValueError):
        return None
    return c if 0 <= c <= 10 else None


def load_polygons(shp_base):
    """Read SIGRID-3 polygons, reproject to WGS84.

    Returns list of dicts: {ct, stages:[(code, conc_tenths)...],
    poly_type, geom}. Stages sorted thickest-first per SIGRID convention
    (SA/SB/SC); SO/SD rare extras are ignored and documented.
    """
    import shapefile
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import transform as shp_transform
    import pyproj

    r = shapefile.Reader(shp_base)
    fields = [f[0] for f in r.fields[1:]]

    prj_path = shp_base + ".prj"
    if os.path.exists(prj_path):
        with open(prj_path) as f:
            src = pyproj.CRS.from_wkt(f.read())
    else:
        src = None
    transformer = (pyproj.Transformer.from_crs(src, pyproj.CRS.from_epsg(4326),
                                               always_xy=True)
                   if src is not None else None)

    polys = []
    shapes = list(r.iterShapes())
    records = list(r.iterRecords())
    for shape, rec in zip(shapes, records):
        vals = list(rec)
        def get(*names):
            for n in names:
                if n in fields:
                    return vals[fields.index(n)]
            return None
        ct = _conc_tenths(get("CT"))
        stages = []
        for sname, cname in (("SA", "CA"), ("SB", "CB"), ("SC", "CC")):
            code = _stage_int(get(sname))
            conc = _conc_tenths(get(cname))
            if code is not None and conc is not None:
                stages.append((code, conc))
        ptype = str(get("POLY_TYPE") or "").strip().upper()
        # pyshp parts -> shapely (Multi)Polygon
        pts = shape.points
        parts = list(shape.parts) + [len(pts)]
        geoms = []
        for i in range(len(parts) - 1):
            ring = pts[parts[i]:parts[i + 1]]
            if len(ring) < 4:
                continue
            try:
                g = Polygon(ring)
                if not g.is_valid:
                    g = g.buffer(0)
                if transformer is not None:
                    g = shp_transform(transformer.transform, g)
                if not g.is_empty:
                    geoms.append(g)
            except Exception:
                continue
        if not geoms:
            continue
        geom = geoms[0] if len(geoms) == 1 else MultiPolygon(geoms)
        polys.append({"ct": ct, "stages": stages, "poly_type": ptype,
                      "geom": geom})
    return polys


def polygon_thickness(poly):
    """Return (thickness_in, ct) or (None, ct).

    Concentration-weighted midpoint of WMO stage ranges over SA/SB/SC.
    Stages without an authoritative thickness (70/80/95/96/97/98/unknown)
    are excluded; a polygon with no valid stage yields None (unknown).
    Only ice polygons (POLY_TYPE 'I') with CT>=1 are eligible.
    """
    ct = poly["ct"]
    if poly["poly_type"] not in ("I", "") or ct is None or ct < 1:
        return None, ct
    num, den = 0.0, 0
    for code, conc in poly["stages"]:
        entry = STAGE_TABLE.get(code)
        if entry is None or entry["mid_cm"] is None:
            continue
        if conc <= 0:
            continue
        num += conc * entry["mid_cm"]
        den += conc
    if den == 0:
        return None, ct
    return (num / den) / CM_PER_IN, ct


def polygon_type(poly):
    """Return (stage_code|55|'unknown', ct).

    Predominant stage = highest partial concentration (ties -> thickest,
    i.e. first in SIGRID SA/SB/SC order). CT<=0 or missing -> (55, ct)
    treated as open water by callers; CT>=1 with no known stage -> unknown.
    """
    ct = poly["ct"]
    if poly["poly_type"] == "W" or ct is None or ct < 1:
        return 55, ct
    if poly["poly_type"] not in ("I", ""):
        return 55, ct
    known = [(c, k) for c, k in poly["stages"]
             if c not in UNKNOWN_CODES and k and k > 0]
    if not known:
        return "unknown", ct
    best_conc = max(k for _, k in known)
    cands = [c for c, k in known if k == best_conc]
    # tie-break: thickest stage = lowest index in SA/SB/SC order
    order = [c for c, _ in poly["stages"]]
    for c in order:
        if c in cands:
            return c, ct
    return cands[0], ct


def concentration_alpha(ct, full=205, floor=50):
    """Alpha scaled by total concentration (partial-pixel honesty)."""
    if ct is None or ct < 1:
        return 0
    return max(floor, round(full * min(ct, 10) / 10))
