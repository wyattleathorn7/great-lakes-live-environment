# NOAA Data Sources — research findings (verified live 2026-09-21)

All endpoints below were probed live with HTTP requests before any download
code was written. Only authoritative NOAA / NCEP / USNIC sources are used.

## Common geography

| Item | Value |
|---|---|
| CRS | WGS84 (EPSG:4326), equirectangular canvas |
| Render bounds (all layers, identical) | lon −93.0 … −73.5, lat 40.5 … 49.5 |
| Canvas | 1800 × 1175 px PNG (RGBA) |
| Shoreline | ONE shared mask pair (`assets/great_lakes_watermask.png` 1800×1175 + `assets/great_lakes_watermask_4x.png` 7200×4700, 8-bit alpha), built from the **NOAA Medium-Resolution Digital Vector Shoreline** (NOAA/NOS, compiled from nautical charts, mean-high-water datum; the "Great Lakes Medium-resolution Shoreline" linked from NOAA GLERL's data page, NOAA Tech. Memo ERL GLERL-104; NAD83 degrees kept at native precision). Pipeline (`python scripts/build_watermask.py --shoreline <us_medium_shoreline base>`): snap open arc endpoints ≤0.02° (chart digitizing gaps only; interiors untouched) → node → polygonize → water = faces containing the 5 lake + St. Clair seeds (connecting channels kept where charted) minus originally-closed island rings (Isle Royale, Beaver, Manitoulin verified as holes) plus inland closed rings. Validated: 6/6 seeds water, 3/3 islands holes, ~194k shoreline vertices, total water ≈ 251,000 km². Assessed and rejected: CUSP/ENC/ESI (line data behind viewer-gated services, no bulk polygons), GLERL-104 FTP (offline). Every product multiplies overlay alpha by this mask and bleeds edge RGB into transparent pixels (anti-fringe for bilinear clients), so all layers share one shoreline. |
| Land/missing handling | alpha = 0 (fully transparent) outside valid water data |
| Lake mask (legacy cross-check) | NOAA-GLERL `1024_lake_ids.txt` (0=land, 1=Superior, 2=Michigan, 3=Huron, 4=Erie, 5=Ontario, 6=St Clair) from `coords.zip` (`https://www.glerl.noaa.gov/data/ice/glicd/grids/coords.zip`), plus per-cell WGS84 LUTs `1024_latgrid.txt` / `1024_longrid.txt` |

Because every layer is nearest-neighbour binned onto the same canvas from
its own WGS84-mapped source grid AND cut by the same shoreline mask, the
overlays line up exactly in Google Earth while remaining technically
independent.

## KML compatibility note (ScreenOverlay removed)

The Google Earth client used for testing reports `Unsupported element:
"ScreenOverlay"`, so no KML in this repository contains ScreenOverlay.
Legends are delivered inside each KML Document description as HTML (the
legend PNG via absolute URL plus an explicit scale/category text block);
standalone `legend.png` files remain published for the web index and for
automated pixel-exact validation. KMLs contain one GroundOverlay + one
self-refresh NetworkLink and zero vector geometry.

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

## 2b. Ice thickness & ice type — USNIC NAIS daily SIGRID-3 shapefile

- **Product:** NAIS daily Great Lakes ice analysis, GIS shapefile
  (`POLY_TYPE IN('W','I')`, WGS84 Lambert Conformal Conic, metres).
- **Download (verified 200):**
  `https://usicecenter.gov/File/DownloadCurrent?pId=35`
  (sister files: `pId=45` KMZ, `pId=125` GRIB).
- **Why this source:** it is the only machine-readable *authoritative*
  Great Lakes product carrying per-polygon ice stage (type) from which
  thickness can be honestly derived. Alternatives rejected: USCG District 9
  thickness charts (raster PNGs, twice-weekly in season only — not
  machine-readable); GLCFS modelled ice thickness (experimental, THREDDS
  endpoints bot-walled); NIC ASCII grids (concentration only).
- **Attributes used:** `CT` (total concentration, tenths), `CA/CB/CC`
  (partial concentrations of 1st/2nd/3rd thickest ice), `SA/SB/SC`
  (their WMO stages of development), `POLY_TYPE` (`W` water / `I` ice).
  `SO/SD` extras ignored (documented). No `Last-Modified` header is sent;
  the analysis date is parsed from the member filename (`GLYYMMDD`).
- **Stage semantics:** WMO SIGRID-3 Table A-3 (via NSIDC G10013 user guide,
  based on WMO 2010): 55 ice-free, 70 brash, 80 unstaged, 81 new <10 cm,
  82 nilas <10 cm, 83 young 10–<30 cm, 84 grey 10–<15 cm, 85 grey-white
  15–<30 cm, 86 first-year ≥30–200 cm, 87 thin FY 30–<70 cm, 88 S1 30–<50 cm,
  89 S2 50–<70 cm, 91 medium FY 70–<120 cm, 93 thick FY ≥120 cm (open-ended),
  95/96/97 old/second-year/multi-year (no WMO thickness), 98 glacier,
  99/−9 unknown. All 17 ice-bearing stages appear in the Ice Type key.
- **Ice Type product:** predominant stage = highest partial concentration
  (ties → thickest-listed); 17 fixed key colors + Unknown (#2B2D42);
  open water (CT<1) transparent; pixel alpha scaled by CT/10.
- **Ice Thickness product (DERIVED, labelled as such):** per polygon,
  `Σ(conc_i × stage_midpoint_i)/Σ(conc_i)` over stages with authoritative
  ranges, cm→inches; stages 70/80/95/96/97/98/unknown excluded (no
  authoritative thickness — never guessed); open water transparent; alpha
  scaled by CT/10; continuous light-blue→purple gradient, adaptive max
  `ceil(p99.5)` clamped to [6, 40] in.
- **Off-season:** the current file is the last spring analysis (a single
  ice-free `CT=00` polygon); transparent rasters with full legends and
  explicit metadata notes are VALID output, not failure.
- **Update interval:** daily in season. Freshness: **"LATEST AVAILABLE"**.

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
- **Visualization:** fixed scientific scale 0–30 ft as ONE continuous
  piecewise-linear anchor gradient (anchors at 0/1/2/3/5/6/9/10/12/13/15/16/
  20/21/23/24/26/27/30 ft: dark blue → blue/cyan → green → yellow →
  yellow-orange → orange → red → red-violet → violet → purple → dark purple,
  compressed toward the top; legend ticks 0/2/5/9/12/15/20/23/26/30+ ft;
  values above 30 ft clamp into dark purple, never transparent).
- **Resolution documented:** source ~2.5 km; rendered on common canvas.
- **Validation buoys (NDBC `realtime2`, verified live):**
  `45001` (Superior), `45002` (N. Michigan), `45132` (Erie), `45012` (Ontario),
  `45005` (W. Erie) — `WVHT`/`WTMP`/`WSPD` used as QC reference only, never
  as the rendering source.
- **Docs:** `https://polar.ncep.noaa.gov/waves/download2.shtml`,
  `https://www.glerl.noaa.gov/emf/waves/WW3` (experimental cousin, not used).

## 4. Wind — NCEP GLWU UGRD/VGRD (same operational files as wave height)

- **Product:** same `glwu.grlc_2p5km.tCCz.grib2` NOMADS files (§3), variables
  `UGRD` + `VGRD`, surface, analysis step. Operational, 6-hourly.
- **Method:** speed = √(U²+V²) m/s ×1.94384 → knots → unmodified WMO Beaufort
  Force 0–12 (0:<1, 1:1–3, 2:4–6, 3:7–10, 4:11–16, 5:17–21, 6:22–27, 7:28–33,
  8:34–40, 9:41–47, 10:48–55, 11:56–63, 12:≥64 kt). Colors blue→…→red→dark
  purple, dark purple reserved for Force 12. Arrows follow the (U,V)
  movement vector (GRIB components point where air moves TOWARD, so no
  meteorological FROM→TOWARD reversal); calm (<0.5 m/s) gets no arrow;
  arrows are rasterized into the PNG (no KML placemarks).
- Freshness wording: **"LIVE / CURRENT MODEL (analysis)"**.

## LOD tile pyramid (crisp shoreline at all zooms)

Each product publishes `site/<product>/tiles/`: 2×2 tiles at 2× density
(z1) plus 4×4 at 4× (z2), every tile 1800×1175 with the product's own color
table, masked from `assets/great_lakes_watermask_4x.png` (7200×4700),
binned with a source halo so seams are discontinuity-free (verified:
seam color jump ≤ interior jump). The KML references tiles with
Region/Lod hints over the shared-color overview; clients without Region
support simply overdraw the same colors. Tiles are Pages-deployed, never
committed; a KML references tiles only when generated in that run.

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
the other products and the Pages publish proceed normally.
