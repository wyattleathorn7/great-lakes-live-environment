"""NCEP HRRR CONUS analysis fetcher (shared by solar + air temperature).

Downloads ONLY the needed GRIB2 messages via .idx byte ranges (a full
HRRR 2D file is ~150 MB; TMP2m is ~8 MB, DSWRF ~1 MB). Decodes with
eccodes (pip wheel). Each product downloads independently so jobs stay
independent; each keeps its own state.
"""

import datetime as dt
import os
import urllib.error
import urllib.request

import numpy as np

UA = {"User-Agent": "great-lakes-live-environment/1.0"}
NOMADS = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"


def latest_cycle(max_back_hours=30):
    """Newest available hrrr.YYYYMMDD/conus/tCCz analysis. Returns
    (file_url, datestr, cycle). Raises if none found."""
    now = dt.datetime.now(dt.timezone.utc)
    last = None
    for back in range(max_back_hours + 1):
        t = now - dt.timedelta(hours=back)
        dd, cc = t.strftime("hrrr.%Y%m%d"), t.strftime("%H")
        base = f"{NOMADS}/{dd}/conus/hrrr.t{cc}z.wrfsfcf00.grib2"
        try:
            req = urllib.request.Request(base + ".idx", headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                idx = r.read().decode("utf-8", errors="replace").splitlines()
            if any(":anl:" in line for line in idx):
                return base, dd, cc
        except Exception as e:
            last = e
    raise last or RuntimeError("no HRRR analysis cycle found")


def fetch_messages(file_url, dest, specs, timeout=300):
    """specs = [(shortName, level_substring)]. Downloads the matching
    analysis (':anl:') messages by byte range into one concatenated file.
    Returns {shortName: (start_byte, end_byte)}."""
    from geospatial_utils import download  # noqa (ensures dirs; not used)
    req = urllib.request.Request(file_url + ".idx", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        idx = r.read().decode("utf-8", errors="replace").splitlines()
    lines = [(i, l) for i, l in enumerate(idx) if ":anl:" in l]
    found = {}
    for short, level in specs:
        for i, l in lines:
            p = l.split(":")
            if len(p) > 4 and p[3] == short and level in l:
                a = int(p[1])
                j = idx.index(l) + 1
                if j >= len(idx):
                    continue
                b = int(idx[j].split(":")[1])
                if b <= a:
                    continue
                found[short] = (a, b)
                break
        if short not in found:
            raise ValueError(f"{short}/{level} analysis message not in .idx")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        for short, (a, b) in found.items():
            req = urllib.request.Request(
                file_url, headers={**UA, "Range": f"bytes={a}-{b - 1}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                f.write(r.read())
    os.replace(tmp, dest)
    return found


def read_messages(path, wanted):
    """wanted = {idxName: count} where idxName is the NOMADS .idx abbreviation
    (TMP, DSWRF). Returns {idxName: (values, lats, lons, dataDate, dataTime)}.
    First `count` step-0 messages per name. eccodes reports GRIB shortNames
    (2t, sdswrf), mapped here. Raises if any wanted message is missing."""
    from eccodes import (codes_get, codes_get_array, codes_get_values,
                         codes_grib_new_from_file, codes_release)
    alias = {"TMP": "2t", "DSWRF": "sdswrf", "TCDC": "tcc",
             "SNOD": "sde", "SNOWC": "snowc", "WEASD": "wased"}
    out = {}
    with open(path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                sn = codes_get(h, "shortName")
                name = next((k for k, v in alias.items() if v == sn),
                            sn if sn in wanted else None)
                if name is not None and str(codes_get(h, "step")) == "0" \
                        and len(out.get(name, [])) < wanted[name]:
                    vals = codes_get_values(h).astype(float)
                    lats = codes_get_array(h, "latitudes").astype(float)
                    lons = codes_get_array(h, "longitudes").astype(float)
                    lons = ((lons + 180) % 360) - 180
                    out.setdefault(name, []).append(
                        (vals, lats, lons, str(codes_get(h, "dataDate")),
                         str(codes_get(h, "dataTime")).zfill(4)))
            finally:
                codes_release(h)
    for k, c in wanted.items():
        if len(out.get(k, [])) < c:
            raise ValueError(f"only {len(out.get(k, []))}/{c} {k} messages decoded")
    return {k: v[0] for k, v in out.items()}
