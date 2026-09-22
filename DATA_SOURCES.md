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

## Single-overlay delivery (Google Earth Web limits)

Each product publishes exactly one overview PNG referenced by its KML.
Google Earth Web enforces a max-external-image limit per document and does
not support Region, so multi-image tile pyramids are deliberately avoided:
they produce fetch failures, not sharper shores. Shoreline crispness comes
from the full-precision NOAA vector mask plus edge RGB bleed, in one image.

## 5. Leaf color — NASA MODIS Aqua via Planetary Computer (Michigan-only)

- **Products:** MYD13A1.061 (NDVI + pixel reliability, 16-day) and
  MYD09A1.061 (surface reflectance red/green/blue/SWIR + state QA, 8-day),
  queried per MODIS tile (h11/h12/h13v04) through the Planetary Computer
  STAC API with anonymous SAS (`planetary-computer` SDK `sign()` flow —
  verified working; hand-rolled token URLs 403, so the SDK flow is mandatory).
  COGs are range-read (WarpedVRT to EPSG:4326, canvas window, nearest —
  no bulk downloads).

- **Products:** MYD13A1.061 (NDVI + pixel reliability, 16-day) and
  MYD09A1.061 (surface reflectance red/green/blue/SWIR + state QA, 8-day),
  queried per MODIS tile (h11/h12/h13v04) through the Planetary Computer
  STAC API with anonymous SAS (`planetary-computer` SDK `sign()` flow —
  verified working; hand-rolled token URLs 403, so the SDK flow is mandatory).
  COGs are range-read (WarpedVRT to EPSG:4326, canvas window, nearest —
  no bulk downloads).
- **Why not VNP13A4N:** the specified VIIRS NRT product requires NASA
  Earthdata credentials, which cannot exist for unattended automation
  without operator-supplied secrets. The substitute is the same
  vegetation-index family (MODIS Aqua = same afternoon orbit class, same
  500 m, same NDVI/EVI/QA science, global incl. Canada, QA'd composites)
  with honest costs: 16-day cadence and PC processing lag (~4–6 weeks;
  exposed as composite date + age in every legend/metadata/KML, with a
  75-day STALE flag). The *current* spectral/autumn signal comes from the
  8-day reflectance. Upgrade path: add Earthdata-auth VNP13A4N/VNP09
  readers behind repo secrets without touching the phenology engine.
  Rejected: GIBS (renders colormapped pictures, not data values —
  verified), PC MODIS lag is documented not hidden, STAR VHP (4 km,
  coarser than the 500 m target), USGS eVIIRS (CONUS-only).
- **Land cover (static ancillary):** US side USGS NLCD 2021 + Canada side
  NRCan 2020 Land Cover of Canada (the two national inputs NALCMS
  harmonizes), mosaicked once to `assets/leaf_landcover.png`
  (deciduous/mixed/evergreen/shrub/grass/crop/urban/barren/water;
  wetlands→shrub behavior). `scripts/build_leaf_landcover.py` reproduces it.
- **Footprint (Michigan-only):** `assets/michigan_mask.png` = authoritative
  Michigan state boundary (Natural Earth 50m admin-1), both peninsulas,
  hard clip; Great Lakes water cut by the shared shoreline mask. (The old
  50-mi buffer asset is retired.)
- **Phenology (leaf_phenology v2):** per-pixel NDVI trajectory (current +
  rolling quarter-res history for baseline max with class priors +
  median direction reference) → circular phase 0..1 → class modulation
  (deciduous full, mixed 0.55, evergreen clamped green, shrub/grass 0.45
  capped, crop 0.35 capped, urban/barren/nodata/water transparent) →
  spectral gating (redness proxy only counts while declining; NDSI snow →
  transparent; cloud/bad-QA → hold previous phase or transparent).
  Continuous 19-anchor circular LUT (wraparound-identical deep blue).
- Freshness wording: **"LATEST AVAILABLE COMPOSITE"** (+ age, + STALE flag).

## 6. Chlorophyll — NOAA CoastWatch S-NPP VIIRS (Science Quality, daily)

- **Product:** `nesdisVHNSQchlaDaily` on CoastWatch West ERDDAP (pfeg, no
  bot-wall): Chlorophyll-a, NOAA S-NPP VIIRS, Science Quality, Global 4 km,
  Daily. Variable `chlor_a` (OC3 algorithm), mg m⁻³, valid 0.001–1000.
  The NRT aggregation (`nesdisVHNchlaDaily_Lon0360`) is server-broken
  ("underlying dataset not found"), so Science Quality is used and the
  ~10-day latency is exposed, not hidden. Rendered as a newest-valid
  mosaic of the latest 7 dailies (daily ocean color is cloud-sparse: most
  water pixels are empty on any single day, so each pixel shows its newest
  valid observation in the window; ERDDAP stride-2 fetch, canvas upscales).
  Balanced LINEAR color scale (equal color share per value interval).
- Freshness wording: **"LATEST AVAILABLE (daily composite)"**.

## 7. Water clarity — NOAA CoastWatch S-NPP VIIRS Kd(PAR) (NRT, daily)

- **Product:** `nesdisVHNkdparDaily`: KdPAR, NOAA S-NPP VIIRS, Near
  Real-Time, Global 4 km, Daily. Variable `kd_par` = Diffuse Attenuation
  Coefficient for PAR (NOAA MECB algorithm; product status Experimental —
  stated in metadata), m⁻¹, valid 0.016–32. Larger values mean MORE turbid
  water (no reversal of source values; clear water naturally sits blue).
  The GLERL "Water Clarity-Turbidity Index" ERDDAP was assessed but is
  bot-walled (HTTP 200 serving a bot check even for REST CSV), so KdPAR is
  the operational alternative; the description names the exact variable.
- Freshness wording: **"LATEST AVAILABLE (daily composite)"**.

## 8. Solar + 9. Air temperature — NOAA/NCEP HRRR 3 km analyses (hourly)

- **Product:** HRRR CONUS `wrfsfcf00` analysis via NOMADS (direct HTTPS +
  `.idx` byte ranges: TMP2m ~8 MB, DSWRF ~1 MB — never the ~150 MB file).
  Variables: `TMP` 2 m above ground (K → °F) and `DSWRF` surface
  (hourly-averaged downward shortwave flux, W/m²; nighttime zero is VALID
  data). Lambert 1799×1059, per-cell WGS84 via ecCodes. Both are clipped
  to lake water with the shared NOAA shoreline mask (land transparent).
  Solar uses a balanced LINEAR scale over its historical range (broadband
  flux shown exactly as observed — this is NOT the UV index). Air
  temperature uses a FIXED Apple-style absolute spectrum (-40..130 °F);
  the historical record still tracks LOWEST/HIGHEST+ as ticks on it.
- Freshness wording: **"LIVE / CURRENT MODEL (analysis)"** with cycle stamp.

## 10. Gradient scales — balanced LINEAR + LOWEST / HIGHEST+ (products 8–11)

`scripts/gradient_scale.py`: each product keeps `{hist_min, hist_max,
percentiles, reservoir≤20k}` in `output/state/<product>/`, seeded from
real observed distributions and extended only by validated in-bounds
records. Color mapping is LINEAR and balanced: anchor colors are spread
evenly across the value range so every part of the scale owns an equal
share of color resolution (no compression of any value region). Record
percentiles are drawn as tick labels at true linear positions. Air
temperature instead uses a FIXED Apple-style absolute spectrum
(-40..130 °F); its record ticks ride the fixed axis. One mapping
function paints raster + legend identically; crowded middle ticks are
de-collided (endpoints always kept).

## 11. Snow — HRRR SNOD/SNOWC analysis + VIIRS/MODIS access findings

- **Snow product source:** NOAA/NCEP HRRR 3 km `SNOD` snow depth (m) gated
  by `SNOWC` snow cover % (analysis step, hourly cycles, NOMADS ranged
  GRIB2). Gate: SNOD > 0.002 m AND SNOWC > 0 AND within Michigan AND not
  lake water; everything else transparent. Depth displayed in inches
  (direct conversion). Assessed and rejected for operability: SNODAS
  (all endpoints dead/retired: NOMADS paths 403/404, NSIDC mirror stale
  at 2023, NOHRSC reorganized to overview pages); MOD10A1/MYD10A1 on
  Planetary Computer (updates end June 2025); VIIRS VNP09GA/VJ109GA +
  LANCE NRT (require Earthdata Bearer tokens — verified 401/404 patterns,
  no anonymous bulk path; GIBS serves pictures, not data; PC has no
  daily-reflectance collection); GlobSnow/CMC/ERA5 (coarse or account
  walls); ECCC datamart CaLDAS path unverifiable. Because no
  depth-in-inches satellite source is operable without credentials, the
  product states this explicitly and uses HRRR analysis state (never
  precipitation/forecast variables) with an upgrade path recorded.
- **Leaf rebuild source decision:** same VIIRS finding → MODIS Aqua
  8-day/16-day via Planetary Computer retained as the operational
  daily-checked stream (architecture polls STAC daily; observation
  cadence follows the 8-day composite cycle; VNP13A4N upgrade path
  documented for a credentialed future).
- Freshness wording: snow **"LIVE / CURRENT MODEL (analysis)"**.

`scripts/gradient_scale.py`: snow uses the same balanced LINEAR mapping
over its historical record (family: silver→blue→purple→magenta→pink→
white at the extreme end only); the no-snow provisional legend is linear
0..24 in. Leaf color is unaffected (uniform circular phenology phase —
never value-compressed). One mapping function paints raster + legend
identically.

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
