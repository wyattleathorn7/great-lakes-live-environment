# NOAA Data Sources — research findings (verified live 2026-09-21)

All endpoints below were probed live with HTTP requests before any download
code was written. Only authoritative NOAA / NCEP / USNIC sources are used.

## Common geography

| Item | Value |
|---|---|
| CRS | WGS84 (EPSG:4326), equirectangular canvas |
| Render bounds (all 3 layers, identical) | lon −93.0 … −73.5, lat 40.5 … 49.5 |
| Canvas | 1800 × 1175 px PNG (RGBA) |
| Land/missing handling | alpha = 0 (fully transparent) outside valid water data |
| Lake mask | NOAA-GLERL `1024_lake_ids.txt` (0=land, 1=Superior, 2=Michigan, 3=Huron, 4=Erie, 5=Ontario, 6=St Clair) from `coords.zip` (`https://www.glerl.noaa.gov/data/ice/glicd/grids/coords.zip`), plus per-cell WGS84 LUTs `1024_latgrid.txt` / `1024_longrid.txt` |

Because every layer is nearest-neighbour binned onto the same canvas from
its own WGS84-mapped source grid, the three overlays line up exactly in
Google Earth while remaining technically independent.

---

## 1. Water temperature — NOAA/GLERL CoastWatch GLSEA (daily)

- **Product:** Great Lakes Surface Environmental Analysis (GLSEA),
  produced daily at NOAA GLERL (Ann Arbor, MI) via the NOAA CoastWatch program.
- **Type:** Satellite-derived (NOAA AVHRR + VIIRS S-NPP + VIIRS NOAA-20),
  cloud-free previous-day imagery composited over a ±10-day window with
  smoothing fallback. Operational/daily (not experimental).
- **Download (verified 200, ~8 MB, Last-Modified daily ~02:50 UTC):**
  `https://apps.glerl.noaa.gov/coastwatch/webdata/glsea/cur/glsea_cur.asc`
  (directory listing also exposes `glsea_cur.dat`, `glsea_cur.png`,
  and ACSPO-era `glsea_cur_3.*` variants)
- **Format:** ESRI ASCII grid, 1024 cols × 1024 rows, cellsize 1800 m,
  `xllcorner -10288921.955`, `yllcorner 4676874.158`, `NODATA_value -9999.0`.
- **Encoding (verified by sampling):** lake-cell values are **°C directly**
  (e.g. 14–22 °C across the lakes on 2026-09-21, physically correct for
  September); `255.0` = land mask; `-9999` = no-data.
- **Update interval:** daily. Freshness wording: **"CURRENT DAILY"**.
- **Data timestamp used:** HTTP `Last-Modified` of the `.asc` file.
- **Units:** source °C; legend displayed in **°F** (conversion `F = C×9/5+32`).
- **Resolution documented:** source ~1.8 km; rendered 1800×1175 canvas over
  bounds above (no fake precision claimed).
- **Docs:** `https://coastwatch.glerl.noaa.gov/satellite-data-products/great-lakes-surface-environmental-analysis-glsea`
- **Restrictions:** none for public download; attribution required
  (no NOAA endorsement claimed).

## 2. Ice coverage — U.S. National Ice Center (NAIS) daily Great Lakes analysis

- **Product:** NAIS (USNIC + Canadian Ice Service) **daily Great Lakes ice
  analysis**, total ice concentration grids. Charts are daily during the ice
  season; off-season the current grid is still published with 0 % over water.
- **Type:** Analysis of satellite imagery (operational). GLERL adds the same
  NIC ice field to GLSEA.
- **Download (verified 200, ~3 MB):**
  `https://usicecenter.gov/File/DownloadCurrent?pId=38` — "GRID 1800"
  (sister resolutions: `pId=37` = 1275 m, `pId=39` = 2550 m;
  `pId=125` = GRIB, `pId=35` = shapefile, `pId=45` = KMZ).
- **Format:** ESRI ASCII grid, 1024×1024, cellsize 1800 m
  (`xllcorner -10288021.9553`, `yllcorner 4675974.1582`, `NODATA_value -9999`).
- **Encoding (verified + per NOAA-GLERL `icegridresampling` sample reader
  `read_ice.py`: "ice cover 1024x1024 grid (unit: %)"):** **percent 0–100**;
  `-1` = land/off-water; values `< -1` (e.g. `-9999`) = no-data and are
  **never** interpreted as 0 % ice. September file contains only `-1`/`0`
  (correct: ice-free), which is *valid* data, not corruption.
- **Update interval:** daily during ice season; file `Last-Modified` used as
  data timestamp. Freshness wording: **"LATEST AVAILABLE (daily analysis,
  seasonal product)"** — never "real-time".
- **Resolution documented:** source 1.8 km; rendered on common canvas.
- **Docs:** `https://usicecenter.gov/Products/GreatLakesData`,
  `https://www.glerl.noaa.gov/data/ice`, concentration "in tenths" on charts
  ≡ percent/10 (grids themselves are percent).

## 3. Wave height — NCEP operational Great Lakes Wave Unstructured v2.1 (GLWU)

- **Product:** NOAA/NCEP operational Great Lakes Wave (WAVEWATCH III,
  unstructured mesh GLWUv2.1, Abdolali et al. 2024, GMD). **Operational**
  (preferred over the experimental GLERL WW3 page, which runs 2×/day and is
  explicitly research-grade).
- **Type:** Numerical wave model, 6-hourly cycles, analysis + forecast hours.
- **Access (verified live):** NOMADS grib-filter index
  `https://nomads.ncep.noaa.gov/cgi-bin/filter_glwu.pl`
  → daily dirs (`glwu.YYYYMMDD`) → full-domain file
  `glwu.grlc_2p5km.tCCz.grib2` (CC = 01/07/13 + hourly `_sr`/`500m` variants).
  Direct HTTPS download works:
  `https://nomads.ncep.noaa.gov/pub/data/nccf/com/glwu/prod/glwu.YYYYMMDD/glwu.grlc_2p5km.tCCz.grib2`
  (directory listing is 403 but file + `.idx` fetches are 200).
- **Grid (decoded live with ecCodes):** Lambert conformal 581×361
  (~2.5 km), lat 40.52…49.20, lon −92.64…−74.25 — covers all five lakes.
  (`_lc` suffix files are Lake Champlain sub-grids — **not** used.)
- **Variable:** `HTSGW` (significant height of combined wind waves and
  swell), **analysis step (`step=0`)**, units **metres**; missing = 9999.
  Verified 2026-09-21 t13z analysis range 0–2.66 m (0–8.7 ft), physical.
- **Update interval:** 6-hourly cycles. Freshness wording:
  **"LIVE / CURRENT MODEL (analysis)"** with exact `dataDate/dataTime` stamp.
- **Decoding:** `eccodes` (pip wheel, no system deps) reading only the
  `HTSGW/step=0` message; per-cell WGS84 via GRIB lat/lon arrays.
- **Resolution documented:** source ~2.5 km; rendered on common canvas.
- **Validation buoys (NDBC `realtime2`, verified live):**
  `45001` (Superior), `45007` (S. Michigan), `45132` (Erie), `45012` (Ontario)
  — `WVHT`/`WTMP` used as QC reference only, never as the rendering source.
- **Docs:** `https://polar.ncep.noaa.gov/waves/download2.shtml`,
  `https://www.glerl.noaa.gov/emf/waves/WW3` (experimental cousin, not used).

## Cache / refresh design (Google Earth)

GitHub Pages cannot set custom cache headers, so cache-busting is done with
**query-string versioning**: every successful product run regenerates its KML
(same stable filename/URL) with `current.png?v=<processing-timestamp>` and
`legend.png?v=<…>`. Each KML additionally contains a self-`NetworkLink`
(`refreshMode=onInterval`) so Google Earth re-fetches the KML on a schedule
and discovers the new image URL without any user action.

## Failure semantics

Per-product independence: a failed download/validation aborts **only that
product's** update (exit status recorded, previous `site/` assets untouched);
the other two products and the Pages publish proceed normally.
