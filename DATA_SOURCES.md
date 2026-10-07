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
  `glwu.grlc_2p5km.tCCz.grib2` (CC = 01/07/13/19 — t19z verified live
  2026-09-23 as the 18Z-run file; hourly `_sr`/`500m` variants ignored).
  Direct HTTPS download works:
  `https://nomads.ncep.noaa.gov/pub/data/nccf/com/glwu/prod/glwu.YYYYMMDD/glwu.grlc_2p5km.tCCz.grib2`
  (directory listing is 403 but file + `.idx` fetches are 200).
  Cycle detection (verified 2026-09-23): the `.idx` carries the analysis
  stamp (`d=YYYYMMDDHH`) on `:anl:` lines, so the newest AVAILABLE cycle
  is found with KB probes — no 66 MB download to discover staleness, no
  assumption that the schedule equals a posted cycle.
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

## 5. Leaf color — NASA MODIS Aqua via Planetary Computer (basin-wide land)

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
- **Footprint (basin-wide land):** every basin land pixel paints (whole
  lon −93…−73.5 / lat 40.5…49.5 rectangle minus open lake water);
  Great Lakes water cut by the shared shoreline mask (leaves do not
  grow on open water, so water stays transparent). Full-coverage
  gap-fill (hold-forward history, nearest-valid propagation, circular-
  median fallback) leaves no land holes.
- **Phenology (leaf_phenology v2):** per-pixel NDVI trajectory (current +
  rolling quarter-res history for baseline max with class priors +
  median direction reference) → circular phase 0..1 → class modulation
  (deciduous full, mixed 0.55, evergreen clamped green, shrub/grass 0.45
  capped, crop 0.35 capped, urban/barren/nodata/water transparent) →
  spectral gating (redness proxy only counts while declining; NDSI snow →
  transparent; cloud/bad-QA → filled, see below).
  Continuous 19-anchor circular LUT (wraparound-identical deep blue).
- **Mosaic + rendering (v7, observed-only):** tiles combined
  first-valid-wins (single-tile pixels keep exact source values); the
  narrow inter-tile overlap bands get a 3-px horizontal blend so zipper
  steps and resampling edge lines vanish. ONLY currently-observed
  vegetated land paints — clouds, snow, bad QA, masked classes and water
  stay transparent (100% the source's current information; never carried
  views, never modeled fills). History (observations only,
  duplicate-suppressed) feeds the trajectory baseline alone and never
  renders.
- Freshness wording: **"LATEST AVAILABLE COMPOSITE"** (+ age, + STALE flag).

## 6. Chlorophyll — NOAA CoastWatch S-NPP+NOAA-20 VIIRS (NRT gapfilled, daily)

- **Product (verified live 2026-09-23):**
  `nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily` on CoastWatch ERDDAP
  (newest 2026-09-20): Chlorophyll-a, S-NPP + NOAA-20 VIIRS, Near
  Real-Time, gapfilled, Daily. Variable `chlor_a` (OC3), mg m⁻³, valid
  0.001–1000. Fallback: `nesdisVHNSQchlaDaily` (Science Quality, newest
  2026-09-13, ~10 d production latency). Dead ends confirmed live:
  `erdVHNchla1day` (stale at 2026-06-14), `nesdisVHNchlaDaily` (stale at
  2026-08-25), `nesdisVHNnoaa20chlaDaily` (stale at 2026-09-02). NOAA-21
  is not yet in CoastWatch ERDDAP, so no NOAA-21 chlorophyll is
  fabricated; the S-NPP+NOAA-20 NRT stream is the operational coverage.
  Rendered as a newest-valid mosaic of the latest 7 dailies (daily ocean
   color is cloud-sparse; ERDDAP stride-2 fetch, canvas upscales).
   Balanced LINEAR color scale. The gapfilled NRT stream also removes the
   scan-line gaps of the old SQ-only mosaic.
- **Display scale (v6):** Carlson Trophic State Index 0–100
  (TSI = 9.81·ln(chl-a)+30.6, clamped 0–100), even 5-unit steps with the
  trophic color table (oligotrophic blues → mesotrophic yellows →
  eutrophic oranges → hypereutrophic red-violet/deep-purple). Legend
  labels show TSI (chl-a µg/L equivalents). Same TSI always shows the
  same color; high TSI marks biomass/activity, never toxins.
- Freshness wording: **"LATEST AVAILABLE (daily composite)"**.

## 7. Water clarity — NOAA CoastWatch S-NPP VIIRS Kd(PAR) (SQ, daily)

- **Product (verified live 2026-09-28):** KdPAR, S-NPP VIIRS, Global 4 km,
  Daily, newest-wins across `nesdisVHNkdparDaily` (NRT, newest 2026-09-21)
  and `nesdisVHNSQkdparDaily` (Science Quality, newest 2026-09-18, ~10 d
  production latency), with per-day fallback across both ids: every
  candidate time axis is probed and the freshest end date owns the mosaic
  (a fixed priority order previously stranded the mosaic on SQ 09-18 while
  NRT had 09-21; the NRT id 404'd on 2026-09-23 but is back online).
  Retired 2026-10-03: a MODIS Aqua Kd490 NRT per-day fallback mixed a
  DIFFERENT variable (attenuation at 490 nm, numerically lower than PAR
  broadband for the same water) into the KdPAR scale — whole lakes pinned
  to the floor navy with square seams at the variable boundaries. One
  variable, one scale since: missing days stay transparent. Variable `kd_par` = Diffuse
  Attenuation Coefficient for PAR (NOAA MECB algorithm; product status
  Experimental — stated in metadata), m⁻¹, valid 0.016–32. Larger values
  mean MORE turbid water (clear water naturally sits blue). The legacy
  `nesdisVHNkdparDaily` id is kept as fallback but returned 404 on its
  time axis on 2026-09-23 (intermittent/gone). No NRT KdPAR dataset
  exists on ERDDAP (re-searched 2026-09-25: SQ newest 2026-09-15,
  ~10 d production latency; no NRT/gapfilled KdPAR id exists). The GLERL
  clarity-turbidity ERDDAP was assessed but is bot-walled, so KdPAR stays
  the operational source; the description names the exact variable.
- Freshness wording: **"LATEST AVAILABLE (daily composite)"**.
- **Display scale (v6):** FIXED absolute 0.016–2 m⁻¹, built like the
  other working products (UV/pressure style): hand-placed anchors tuned
  to where lake water lives, raw values rendered directly with no
  transforms. The 0.016–2 domain keeps the legend labels evenly spread
  (0/24/50/75/100% of the bar). The floor equals valid_min, so no valid
  observation can clamp into the floor color (retires the old scale's
  dark-navy "holes" in ultra-clear water). Values shown exactly as
  observed; 2+ clamps into deep purple as honest extremes.
- **Cadence:** source probed hourly (workflow) with a 7-day newest-valid
  mosaic, so any new ERDDAP publication rebuilds within the hour.

## 8. Solar — NOAA/NCEP GFS DSWRF f000 analysis (6-hourly); Air temperature — HRRR (hourly)

- **Solar product (RAP, verified live 2026-09-25):** NOAA/NCEP Rapid
  Refresh (RAP) hourly surface DSWRF via NOMADS
  (`.../rap/prod/rap.YYYYMMDD/rap.tCCz.awp252bgrbfHH.grib2`, 13 km native
  grid, DSWRF message byte-ranged via `.idx`). The newest cycle whose
  DSWRF field is valid for the current hour is used (analysis step when
  present, else the shortest forecast lead; cycle, valid time, units, and
  lead are all recorded in metadata). Variable `sdswrf` = Surface
  downward short-wave radiation flux, W m⁻² — modeled incoming sunlight
  energy at the surface (broadband: ultraviolet + visible +
  near-infrared). This is not a UV Index and not a visible-light meter
  reading; it is an hourly weather-model estimate, not a ground sensor
  measurement at every location.
  native 13 km Lambert grid is mean-binned onto the canvas (same approach
  as the HRRR products) plus NaN-aware blur smoothing for a steady
  gradient (resampling pinholes between the coarse cells filled; true
  domain edge left missing; stats stay on raw values); a flood-fill
  interior-hole guard over watermask water fails the run instead of
  shipping artefacts. Rendered at alpha 160 (softer than the shared
  205) so the base map reads through. Nighttime zero is
  VALID data (near-black), never NoData.
  FIXED absolute sequential scale 0–1000+ W/m² (near-black night →
  navy → blue → cyan → green → yellow → orange → red → near-white
  extreme; same flux always shows the same color; numeric W/m² ticks
  only, no UV-level categories).
- **Air temperature product:** HRRR CONUS `wrfsfcf00` analysis via NOMADS
  (direct HTTPS + `.idx` byte ranges, TMP2m only). Variable `TMP` 2 m
  above ground (K → °F). Lambert 1799×1059, per-cell WGS84 via ecCodes.
  FIXED brightened banded 5 °F key (-60..150 °F): blue-white extreme
  cold → gray-blue subfreezing → royal-blue freezing wall (30–40) →
  frigid turquoise/blues (40–55) → lime/teal transition (55–70) →
  mellow yellows (70–85) → orange/gold warming (85–100) → scorching
  pink/reds past 100 (dark maroons lifted so no band renders dark);
  the historical record still tracks LOWEST/HIGHEST+ as values in the
  description.
- Both paint the FULL basin rectangle (no shoreline cut; only missing
  data transparent). Freshness wording: **"LIVE / CURRENT MODEL
  (analysis)"** with cycle stamp.

## 12. UV Index — NOAA/NCEP CPC Global UV (operational GRIB2, 12Z run)

- **Product (verified live 2026-09-23):** daily dirs
  `https://nomads.ncep.noaa.gov/pub/data/nccf/com/uvi/prod/uvi.YYYYMMDD/`
  (note the `uvi.` prefix), files `uv.t12z.grbfHH.grib2` (HH = 01..120,
  all present for 20260922). GRIB2 parameter verified by decoding:
  `shortName=uvi`, `name=UV index`, grid `regular_ll` 1440×721
  (~0.25°), values 0–0.366 = erythemally weighted flux in W/m², so
  **UV Index = filed value × 40, applied exactly once** (guard: values
  already > 2 are NOT scaled again). The displayed raster is the
  forecast hour nearest the current time from the newest available 12Z
  run (a FORECAST field, not hourly satellite observations); the hour
  advances as time passes and a new raster publishes per (run, hour).
  Fixed absolute EPA-style 0–11+ scale. Metadata records source run,
  forecast hour, valid time, and processing time separately.
- Freshness wording: **"LIVE / CURRENT FORECAST (12Z run)"**.

## 10. Gradient scales — balanced LINEAR + LOWEST / HIGHEST+ (products 8–11)

`scripts/gradient_scale.py`: each product keeps `{hist_min, hist_max,
percentiles, reservoir≤20k}` in `output/state/<product>/`, seeded from
real observed distributions and extended only by validated in-bounds
records. Color mapping is LINEAR and balanced: anchor colors are spread
evenly across the value range so every part of the scale owns an equal
share of color resolution (no compression of any value region). Record
percentiles are drawn as tick labels at true linear positions. Air
temperature instead uses a FIXED brightened banded 5 °F key
(-60..150 °F, flat bands); its record LOWEST/HIGHEST+ values ride in
the description text. One mapping
function paints raster + legend identically; crowded middle ticks are
de-collided (endpoints always kept).

## 11. Snow — HRRR SNOD/SNOWC analysis + VIIRS/MODIS access findings

- **Snow product source:** NOAA/NCEP HRRR 3 km `SNOD` snow depth (m) gated
  by `SNOWC` snow cover % (analysis step, hourly cycles, NOMADS ranged
  GRIB2). Gate: SNOD > 0.002 m AND SNOWC > 0 anywhere in the basin
  (open lake water stays transparent — no ground-snow signal exists on
  water); everything else transparent. Depth displayed in inches
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

## Cache / refresh design (Google Earth) — source-aware entry/live split

GitHub Pages cannot set custom cache headers, so cache-busting is done
with **deterministic source-version query strings**: the live overlay
file `site/kml/live/<Name>.kml` points at
`<product>/current.png?v=<SOURCE_VERSION>`, where the version changes
if and only if the underlying source observation/cycle changes (model
cycle, observation date, or content hash — never a processing timestamp
or random value). Users add the stable entry file `kml/<Name>.kml`
once: it contains exactly one `NetworkLink` (no imagery) with
`refreshMode=onInterval` at the product's discovery interval (fastest
practical: wind, wave height, solar at 900 s on the hourly workflow;
other hourly sources at 3600 s; daily and slower sources at 21600 s), so Google Earth
re-fetches the live KML on schedule and the new `?v=` forces the new
PNG past CDN and client caches. The live file holds exactly one
`GroundOverlay` (Icon `onInterval`), zero `NetworkLink`s (no
self-reference loops), zero vector geometry, zero `ScreenOverlay`.
Every builder detects its newest available source id first and skips
the rebuild (exit 0, deterministic KML rewrite) when the source is
unchanged — including cheap pre-checks (GLWU `.idx` stamp probes,
ERDDAP time-axis probes) that avoid bulk downloads.

Verified live 2026-09-23: Planetary Computer hosts NO VIIRS collections
(MODIS only), so VJ209GA_NRT requires NASA Earthdata credentials that
anonymous CI cannot provide — MODIS Aqua stays the documented
operational stream. PC's `modis-13A1-061` newest tile composite was
A2026217 (2026-08-05) on 2026-09-23 (upstream lag, correctly held with
a STALE flag past 75 d, not fabricated).

## Failure semantics

Per-product independence: a failed download/validation aborts **only that
product's** update (exit status recorded, previous `site/` assets untouched);
the other products and the Pages publish proceed normally.

## 14. Cloud cover — GOES-East ABI Clear Sky Mask (live observation)

- **Product:** NOAA GOES-East (GOES-19) ABI L2 Clear Sky Mask, CONUS
  2 km, new scan every 5 minutes, public S3
  (`s3://noaa-goes19/ABI-L2-ACMC/YYYY/DDD/HH/`, anonymous HTTPS, no
  credentials). Replaced HRRR TCDC in v2: the model's cloud analysis
  carries rectangular assimilation footprints that read as unphysical
  blocks — the satellite retrieval has none.
- **Variables:** `BCM` binary mask (0 = clear_or_probably_clear,
  1 = cloudy_or_probably_cloudy) gated by `DQF==0` (good quality);
  `ACM` 4-level mask retained for reference. Fixed-grid → WGS84 via
  pyproj with the file's own subpoint/height (hand-rolled geostationary
  math was cross-checked against it and dropped after a 3° latitude
  error was found in the hand derivation).
- **Rendering:** % = observed fraction of cloudy 2 km pixels per canvas
  neighborhood (true area density, no invented precision); oblique view
  stretches ground spacing, so splat radius 2 closes the diamond gaps.
  Clear (<1%) and missing stay transparent.
- **Scale:** FIXED 0–100 % spectrum (same stops as before).
- **Update:** 5-minute scans; the dedicated fast workflow
  (`update_fast_live.yml`) polls on its own 5-minute cadence with a lean
  publish (no commit), so scans ship in minutes.
  Freshness: **"LIVE / CURRENT OBSERVATION (GOES-East 5-min scans)"**.
- **Retired (v1):** HRRR `TCDC` entire-atmosphere analysis. It verified
  against METARs but its analysis increments carry rectangular
  assimilation footprints (confirmed present in the raw GRIB2), which
  read as unphysical blocks — replaced by the satellite retrieval.

## 15. Surface pressure — NOAA/NCEP HRRR MSLMA (hourly analysis)

- **Product:** same HRRR `wrfsfcf00` files, `MSLMA` / `mean sea level`
  message (shortName `mslma`, step 0).
- **Variable:** filed Pa ÷ 100 = **hPa** (the only conversion);
  admitted [900, 1100]. Displayed on a FIXED 980–1040 hPa spectrum with
  the standard atmosphere **1013.25 hPa in the middle (yellow)** — lows
  run dark-blue → green left, highs run orange → dark-purple right
  (labels LOWEST 980 / 1000 / AVERAGE 1013.25 / 1025 / HIGHEST+ 1040).
- **Coverage:** full basin rectangle (no shoreline cut); only missing
  data transparent.
- **Update:** hourly cycles. Freshness: **"LIVE / CURRENT MODEL (hourly
  HRRR cycle)"**.

## 16. Wave period & direction — NCEP GLWU PERPW + WVDIR (6-hourly analysis)

- **Product:** same operational `glwu.grlc_2p5km.tCCz.grib2` NOMADS files
  as wave height/wind (§3–4); `.idx` byte-ranges fetch the `PERPW` and
  `WVDIR` `surface` / `:anl:` messages (both verified live in the same
  files — no new source needed).
- **Gradient:** `PERPW` peak (primary) wave period, filed **seconds**,
  used as-is (no conversion). Admitted [0, 20], implausible >15 s
  rejected. FIXED absolute 0–12 s spectrum (dark-blue flat → blue →
  cyan → green → yellow → orange → red → magenta → violet →
  dark-purple long swell; labels LOWEST 0s / 2 / 4 / 6 / 8 / 10 /
  HIGHEST+ 12s; above 12 s clamps dark-purple).
- **Arrows:** `WVDIR` filed compass degrees drive white-shaft
  toward-travel arrows only (filed FROM + 180°, standard oceanographic
  handling) — direction never colors the map.
- **Coverage:** lake water only via the shared NOAA shoreline mask.
- **Buoy QC (reference only):** NDBC dominant period (DPD) vs nearest
  grid cell, tolerance 2.5 s (verified 4/4 reporting buoys OK on
  2026-09-25).
- **Update:** 6-hourly cycles. Freshness: **"LIVE / CURRENT MODEL
  (analysis), 6-hourly cycles"**.

## 17. Precipitation — IEM NEXRAD N0Q base reflectivity (live radar, 5-minute mosaic)

- **Product (verified live 2026-09-27):** Iowa Environmental Mesonet
  (Iowa State University) CONUS NEXRAD N0Q Base Reflectivity mosaic,
  built from the NOAA NWS WSR-88D Level-III network and rebuilt every
  5 minutes. WMS endpoint
  `https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q.cgi?`
  (layer `nexrad-n0q-900913`, verified via GetCapabilities + GetMap).
  Professional + credible: Iowa State archives the operational NWS feed
  (same data that drives NWS warnings) and serves it as an OGC WMS.
  Docs: `https://mesonet.agron.iastate.edu/current/radar.phtml`,
  `https://mesonet.agron.iastate.edu/docs/nexrad_composites/`.
- **Access:** WMS GetMap for the exact common canvas
  (SRS EPSG:4326, BBOX lon −93,lat 40.5,lon −73.5,lat 49.5,
  WIDTH 1800 HEIGHT 1175, FORMAT image/png, TRANSPARENT=TRUE).
  The WMS renders the N0Q palette itself, so the live gradient in
  Google Earth IS the source's real-time gradient — never a recolor.
  Verified live 2026-09-27: 1800×1175 RGBA, ~5% opaque (light rain over
  the basin), remainder alpha 0.
- **Variable:** N0Q base reflectivity, dBZ (decibels of reflectivity):
  raw radar echo strength. FIXED absolute 5–75 dBZ key matching the
  reference image: Very Light cyan/blue (~5–20: drizzle, mist) →
  Light greens (~20–30: light rain/snow) → Moderate dark green
  (~30–40: steady rain/snow) → Heavy yellow/orange (~40–50) →
  Severe red (~50–60: hail possible) → Intense magenta/purple (~60–70:
  large hail, flash flooding) → Extreme white (70+: destructive).
- **All seasons / all types:** base reflectivity is echo strength from
  any hydrometeor — spring/summer/fall rain, winter snow and
  lake-effect bands, thunderstorms + hail whenever convection fires.
  Reflectivity does not classify type; it shows intensity, so one scale
  covers rain, snow, and thunderstorms all year.
- **Transparency:** no echo (no precipitation) is fully transparent
  (alpha 0) — kept exactly as the source returns it, never zero-filled.
  Full basin rectangle (no shoreline cut: precipitation falls on land
  and water alike). A dry-over-the-basin run is a VALID transparent
  raster, not a failure (same semantics as a clear-sky cloud run).
- **Refresh (fastest possible, GitHub-only — no Mac execution):** source
  rebuilds every 5 minutes; the dedicated fast workflow
  (`update_fast_live.yml`) runs on its own 5-minute cron (`*/5 * * * *`,
  GitHub's minimum schedule interval) with a lean publish (validate +
  deploy, no commit), so scans ship in minutes; both KML
  refreshIntervals (entry NetworkLink + live Icon) are 300 s, matching
  the source cadence and the workflow schedule — each client poll can
  pick up a newly published scan with no sub-minute polling and no
  battery drain on any local machine. No launchd/cron heartbeat is
  needed or wanted: GitHub's cron is the sole driver.
  Overlay only (GroundOverlay) — Google Earth has no live-video
  primitive. Freshness wording: **"LIVE / CURRENT OBSERVATION
  (NEXRAD 5-min mosaic)"**.

## 18. Ovation Aurora Forecast — NOAA/SWPC OVATION Prime (live forecast, hourly check)

- **Product (verified live 2026-10-04):** OVATION Prime real-time auroral
  grid `https://services.swpc.noaa.gov/json/ovation_aurora_latest.json`
  (`[Longitude, Latitude, Aurora]` 1-degree global grid + Observation /
  Forecast Time; the machine-readable source behind the
  `aurora-viewline-tonight-and-tomorrow-night-experimental` Viewline
  product — the webpage image is never scraped). Run context from
  `https://services.swpc.noaa.gov/text/ovation_latest_aurora_n.txt`
  (hemispheric power in GW, forecast Kp); Kp history/context from
  `https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json`.
- **What it is:** OVATION converts upstream solar-wind energy into maps of
  auroral particle precipitation (Newell et al.; real-time implementation
  Machol/Redmon NCEI, Viereck SWPC). Intensity = model output in OVATION
  units (observed range 0–64+), not a naked-eye brightness meter.
- **Footprint:** the exact LIVE LEAF COLOR rectangle
  (lon −93…−73.5, lat 40.5…49.5, 1800×1175, WGS84), full basin rectangle
  with no shoreline cut (atmospheric layer, like solar/air-temperature).
  LEAF files untouched. Below the detection floor (2.0) the canvas stays
  transparent — a quiet-oval interval is VALID output (same semantics as
  a clear-sky cloud run), and color appears automatically on the next
  hourly rebuild when a storm arrives.
- **Scale:** FIXED absolute 0–30+ (faint green 3 → oval green 6 → yellow
  10 → orange 14 → red 18 → magenta 23 → violet 30+ clamp).
- **Viewline:** southernmost northern-hemisphere latitude in the basin
  longitude band reaching intensity 5+, computed from the GLOBAL grid
  (not canvas-limited) — the product's true visibility boundary even
  when the oval sits poleward of the rectangle. Reported in
  legend/metadata/folder text; no KML line is drawn (raster-only rule).
  Meaning: southernmost region from which aurora may be visible low on
  the northern horizon under dark/clear conditions — an opportunity
  boundary, never a sighting guarantee.
- **Refresh:** source checked hourly (entry NetworkLink + live Icon
  `refreshInterval` 3600 s); rebuilds only when Observation/Forecast
  time changes (source-versioned `?v=`).
- Freshness wording: **"LIVE / CURRENT FORECAST (OVATION Prime, hourly check)"**.

## 19. Water quality — fecal-indicator bacteria (EPA RWQC + USGS WQP + BEACON, hourly check)

- **Terminology:** FECAL-INDICATOR BACTERIA (E. coli + enterococci + fecal
  coliform) for primary-contact recreational water — the complete EPA
  swim-water bacterial panel (EPA ECHO Water Quality Indicators program:
  2012 RWQC for E. coli/enterococci, 1976 Red Book for fecal coliform).
  Not pathogen detections; types not distinguished visually: one
  human-health concern spectrum. Total coliform deliberately excluded
  (drinking-water-oriented group, no recreational criterion).
- **Thresholds (EPA 2012 RWQC, 36/1000 illness rate):** E. coli GM 126 /
  STV 410; enterococci GM 35 / STV 130 (32/1000 rate: 100/320, 30/110);
  Beach Action Value E. coli 235 / enterococci 70. The 13-step log scale
  (10 dark-blue floor → 126 GM → 235 BAV → 320/410 STVs → 5000+ dark
  purple clamp) anchors every transition at these values; 700/1000/2000/
  5000 are documented order-of-magnitude extensions, not EPA criteria.
  Enterococci ×3.6 (= 126/35 GM ratio) → E.coli-equivalent; MPN ≈ CFU
  for display. Full per-color value/unit/threshold/source/rationale table
  in `site/bacteria/metadata.json` (`thresholds`).
- **Live data:** USGS Water Quality Portal Result search (7-day window,
  `Escherichia coli;Enterococcus;Fecal Coliform`, basin bbox) joined to Station-search
  coordinates via chunked siteid queries + persistent cache
  (`output/state/bacteria_stations.json`); newest sample per station
  wins; non-detects kept as censored upper bounds (quantitation limit or
  floor 10, flagged); halo radius 14 px on lake water only (no
  interpolation, no modeled fill). Conversions: enterococci ×3.6 (EPA GM
  ratio), fecal coliform ×0.63 (criterion-anchored: Red Book 200 → E. coli
  GM 126), MPN/CCE ≈ CFU (as EPA WQI does). State beach programs (MI/OH/WI/
  IL/IN/PA/NY BeachGuard networks) submit to WQX and are covered through
  this query. Verified live 2026-10-04: 97 stations over 45 days (range 1–3690 E.coli-equiv); the 7-day October window is
  legitimately empty (post-season) → transparent NO DATA.
- **Coverage:** 100% of all five Great Lakes water surfaces (US +
   Canadian) via the shared shoreline mask; land transparent; unobserved
   water = transparent NO DATA (slate swatch in key; never zero). Canadian anchor beaches listed,
  NO DATA pending an ECCC/Ontario live feed (documented gap).
- **Temporal rule:** SOURCE CHECKED HOURLY (3600 s) ≠ publication
  frequency (agencies sample ~weekly in season); latest valid retained ≤
  7 days, then stale → NO DATA. No measurements invented between samples.
- Freshness wording: **"SOURCE CHECKED HOURLY / LATEST VALID OBSERVATION"**.

## 20. Surface currents — operational NOAA/NOS GLOFS nowcast + GLERL experimental corridor fill

- **Primary (operational, preferred):** NOAA/NOS Great Lakes Operational
  Forecast System (GLOFS) — LSOFS (Superior), LMHOFS (Michigan-Huron),
  LEOFS (Erie), LOOFS (Ontario). FVCOM-based, 6-hourly cycles, hourly
  nowcast hours. Machine-readable NetCDF on the CO-OPS THREDDS server
  (`https://opendap.co-ops.nos.noaa.gov/thredds/catalog/NOAA/<MODEL>/MODELS/...`),
  files `<model>.t<CC>z.<YYYYMMDD>.regulargrid.n<HHH>.nc` (n000–n006 =
  nowcast). Verified live 2026-10-04/05: all four models post current
  cycles; LEOFS grid 313×933 (~0.5 km), LMHOFS 478×837 (~1.1 km), LOOFS
  217×757, LSOFS 529×1555 (~0.55 km).
- **Variables:** `u_eastward` (CF `eastward_sea_water_velocity`) +
  `v_northward` (CF `northward_sea_water_velocity`), filed **m/s**,
  direction of water motion **TOWARD** (no FROM→TOWARD reversal);
  surface layer = Depth index 0 (0.0 m); `_FillValue -99999`, `mask==1`
  = water. Speed = √(U²+V²)×100 → **cm/s** display.
- **Access:** OPeNDAP hyperslab (surface layer of U/V + lat/lon + mask +
  Times only, ~13 MB total per rebuild — never the 75 MB full files;
  THREDDS NCSS returns `Dimension ny does not exist` for these grids, so
  NCSS is not used). Newest source = newest catalog (date, cycle, nowcast
  hour); the file's own `Times` is the data valid time (never fetch time).
- **Corridor supplement (experimental, documented as such):** GLERL
  next-gen GLCFS FVCOM nowcast (`https://apps.glerl.noaa.gov/thredds/...
  /glcfs/<lake>/nowcast/MMDDHH_0001.nc`, 12-hourly, 12 hourly steps,
  current through 2026-10-04): HEC (Huron-Erie Corridor, 36k elements —
  St. Clair River / Lake St. Clair / Detroit River, median channel flow
  40–90 cm/s southward verified) and Michigan-Huron (lower St. Marys
  reach only, bbox-restricted). Surface = siglay 0, last time step,
  `wet_cells==1`. Precedence: operational wins everywhere valid;
  experimental fills ONLY cells with no operational vector (no blending).
- **Source-coverage audit (verified 2026-10-04 from live masks):**
  Superior ✓ LSOFS; Michigan ✓ LMHOFS; Huron ✓ LMHOFS; Erie ✓ LEOFS;
  Ontario ✓ LOOFS; Mackinac ✓ LMHOFS (736 cells); upper St. Marys ✓
  LSOFS (46.42–46.60); lower St. Marys ✓ mih-exp (1214 cells);
  St. Clair River ✓ HEC (operational has ~10 cells — unresolved);
  Lake St. Clair ✓ HEC (operational: 0 cells); Detroit River ✓ HEC
  (+LEOFS mouth); upper Niagara mouth ✓ LEOFS (to 42.905); lower
  Niagara ✓ LOOFS (from 43.23). Gaps with no authoritative vectors:
  St. Marys rapids/locks (~46.35–46.42), Niagara Falls/gorge
  (42.905–43.23), Welland/Trent-Severn/minor canals.
- **Rendering:** FIXED absolute 0–100 cm/s spectrum (anchors at
  0/3/6/10/15/20/30/50/75/100: dark-blue stagnant → blue → cyan →
  teal → yellow-green → yellow → orange → red → red-violet →
  dark-purple jets; above 100 clamps). NO preset arrow lattice:
  deterministic jittered seeds advected downstream along RK2-midpoint
  streamlines (3 px steps, ≤40 steps ≈ 120 px) integrated through the
  filed U/V field with bilinear sampling, drawn as thousands of 2 px
  flow streaks (soft white: full strength would bleach the gradient); streak BRIGHTNESS encodes speed (dim drift, bright
  jets) while the gradient carries the absolute scale; every second
  streak carries a tiny downstream chevron. Separation-aware seeding
  (Jobard-Lefer style) keeps converging flow from piling into blobs;
  traces stop at land/no-data, stagnant water (<1 cm/s), or hairpins.
  Stagnant skipped; ≥1500 streaks + ≥300 chevrons required (typically
  several thousand streaks). Land/no-data transparent under a HARD
  shoreline clip (mask majority or strict channel water: opaque;
  everything else alpha exactly 0 — no feathered fringe on land at
  any zoom). Narrow rivers (St. Clair, Detroit, St. Marys,
  Niagara) read land in the shared open-lake shoreline mask, so river
  alpha is restored product-locally ONLY where a source vector strictly
  lands inside documented RIVER_BOXES (shared mask asset untouched;
  validators permit opaque-outside-mask solely there).
- **Refresh:** source checked hourly (entry NetworkLink + live Icon
  `refreshInterval` 3600 s); GLOFS cycles 6-hourly, GLCFS 12-hourly, so
  most hourly checks are catalog-probe skips. All four operational
  models required per rebuild (a partial basin never ships);
  experimental-fill failure is non-fatal (flagged in metadata).
- Freshness wording: **"LIVE / CURRENT MODEL (nowcast)"**.
- Rejected: S-111 (no basin-wide Great Lakes S-111 service found —
  GLOFS regulargrid NetCDF is the machine-readable equivalent);
  third-party current maps (not authoritative); wind-derived or
  buoy-interpolated vectors (fabrication, forbidden).

## 21. Atmospheric moisture/visibility/particle/light gradients (8 products, same footprint as Leaf Color)

- **Coverage inheritance (Leaf Color analyzed, never edited):** the
  existing LIVE LEAF COLOR footprint is the whole
  lon −93…−73.5 / lat 40.5…49.5 rectangle at 1800×1175 (WGS84), with
  open lake water cut by the shared shoreline mask (land-only signal).
  All eight products below read the SAME single authoritative canvas
  (`config/great_lakes_bounds.json` via `load_bounds()`) and write the
  SAME `LatLonBox` into every KML, so toggling any of them against Leaf
  Color aligns exactly. Atmospheric fields are valid over land AND water
  (like air temperature / pressure / solar / aurora), so they paint the
  FULL basin rectangle with no shoreline cut; only missing source data
  is transparent. No Leaf Color raster, KML, KMZ, bounds, source,
  refresh logic, styling, legend, or folder description was modified.
- **🌤️ Visibility — NCEP HRRR `VIS` surface analysis (m → statute miles),
  hourly cycles.** Direct analysis field on the FAA flight-category
  scale (AIM 7-1-7): 0–0.25 Dense fog (NWS advisory ≤¼ mi), 0.25–1
  LIFR, 1–3 IFR, 3–5 MVFR, 5–30 VFR (10SM METAR reporting cap), stops
  at the exact standard boundaries with a per-category band key.
  NaN-aware 2-pass display smoothing (razor model-grid edges
  render as soft TV-style gradients; single-cell speckles dissolve while
  coherent fog/low-vis areas persist; statistics on raw values).
  Meteorological (human/mariner/pilot) visibility, never
  astronomical seeing. Model-consistency QC: sub-1-mile claims require
  RH >= 90% — dense fog inside dry air is an internally inconsistent
  model fill (uniform 200–500 m values, 68% uncorroborated basin-wide),
  rendered missing instead of maroon; corroborated fog passes through.
  Overlay alpha 165 (slightly transparent, like
  solar). Freshness: LIVE / CURRENT MODEL (analysis).
- **♨️ Relative humidity — NCEP HRRR `RH` 2 m analysis (%), hourly
  cycles.** FIXED 0–100 % spectrum. Near-surface saturation percentage;
  never PWAT or dew point. Freshness: LIVE / CURRENT MODEL (analysis).
- **💡 Light pollution — NASA VIIRS Black Marble annual nighttime-lights
  composite (Suomi NPP DNB, Román et al.) via the public NASA GIBS WMTS
  (`VIIRS_Black_Marble`, currently the 2016 annual; 10 level-6 tiles at
  4.5° each mosaicked to the canvas).** Displayed as a RELATIVE brightness index
  0–100 via a documented three-step monotonic mapping (tile-background
  floor subtraction at the 0.5th percentile, a NaN-aware skyglow bloom
  ~9 km sigma so point-like city cores render as the diffuse metro glow
  every published light-pollution map shows, then a square-root
  perceptual map against the 99.99th percentile) — documented as
  imagery-derived, never fabricated radiance units. Continuous
  environmental surface (no preserve points). Hourly check =
  one probe-tile content hash; rebuilds only when NASA publishes a
  newer composite. Freshness: LATEST AVAILABLE COMPOSITE. Folder text
  distinguishes dominant artificial glow from natural night-sky
  brightness (moonlight/airglow). Rejected: EOG annual downloads
  (login-walled 302), commercial astronomy APIs.
- **🌫️ Air quality — ECMWF CAMS PM2.5 near-surface analysis via the
  Open-Meteo Air Quality API (free, no key), sampled on a 48×30 basin
  grid and bilinearly resampled to the canvas.** Underlying variable
  PM2.5 (µg/m³), converted pixel-wise to the US EPA Air Quality Index
  (AirNow) via EPA PM2.5 breakpoints on a FIXED linear 0–500 scale
  (0–50 Good, 51–100 Moderate, 101–150 USG, 151–200 Unhealthy,
  201–300 Very unhealthy, 301–500 Hazardous) in a muted EPA palette;
  labeled MODELED analysis output everywhere (never station observations, never
  aerosol optical depth). Hourly check (single-point probe); rebuilds
  only on a newer CAMS valid hour. Title stays LIVE AIR QUALITY.
- **💦 Condensation — DERIVED index 0–100 from HRRR TMP+DPT+RH (2 m),
  hourly cycles.** Formula (scripts/derived_moisture.py):
  `100×(0.65×clip((2.5−S)/2.5,0,1)^0.7 + 0.35×clip((RH−60)/40,0,1))`,
  S = TMP−DPT (°C). No wind, no visibility inputs — structurally
  distinct from fog risk, RH, and dew point. Labeled DERIVED.
- **🌫️ Fog risk / active fog — DERIVED index 0–100 v2 from HRRR
  TMP+DPT+RH (2 m) + 10 m wind + VIS, hourly cycles.** Risk owns 0–35
  (`35 × rh_factor × spread_factor × calm_factor`, same factors as v1);
  corroborated observed density owns 35–100
  (`35 + 65 × clip((5000−VIS)/5000,0,1)^0.8`, only where VIS<5000 m AND
  RH≥90 — continuous at the 5000 m boundary). Index = max of the two, so
  observed fog outranks risk but dry-air speckles can never paint active
  fog. NaN-aware display smoothing + raw-value statistics, same contract
  as visibility. Key labels step evenly (No observed fog, Fog risk, then
  Active fog with exact source-visibility equivalents). Labeled DERIVED.
- **🧊 Dew point — NCEP HRRR `DPT` 2 m analysis (K → °F), hourly
  cycles.** Actual dew-point temperature, FIXED −20…90 °F spectrum;
  never depression/spread/probability/RH. Freshness: LIVE / CURRENT
  MODEL (analysis).
- **Refresh/skip contract (all 8):** hourly workflow jobs; deterministic
  source-version `?v=` (HRRR cycle, CAMS hour, Black Marble probe hash);
  skip-and-rewrite-KML when the source is unchanged; exit 2 keeps the
  previous raster. Legends are 640×230 continuous keys painted with the
  SAME stops as the raster; KMLs carry one GroundOverlay + entry
  NetworkLink, zero vector geometry, zero ScreenOverlay (client-rejected;
  legends live in descriptions + `legend.png`, per repo rule).

## 13. Game-fish distribution gradients (16 species, `gamefish_<species>`)


- **What:** live modeled distribution / habitat-likelihood rasters (0..1 index,
  NOT fish counts) for walleye, yellow perch, lake trout, steelhead, brown
  trout, smallmouth bass, northern pike, muskellunge, lake sturgeon, lake
  whitefish, chinook salmon, coho salmon, largemouth bass, burbot,
  cisco (lake herring), sauger — one shared blue(low) to red(high) spectrum
  (model v1.1.0).
- **Telemetry evidence tiers:** strong GLATOS programs (lake whitefish 13
  projects, cisco 10, plus the original nine); NO current telemetry —
  suitability-only at low confidence, flagged in metadata `warnings` (chinook,
  coho, largemouth bass, burbot, sauger). Thermal/migration parameters for all species rest on GLFC
  Sp87-3, USGS/USFWS/agency literature (whitefish: USGS 2023 loggers + Ebener;
  burbot: winter 0.6–1.7 °C spawn, feeding 12–14 °C; sauger: spring river spawn
  ~6–10 °C); decay constants remain labeled MODEL_ASSUMPTIONs.
- **Live inputs:** (1) GLSEA SST from §1 (same URL/cadence; a failed SST fetch
  keeps the previous raster, exit 2); (2) acoustic-telemetry evidence read from
  the `great-lakes-live-fish-telemetry` checkout when available (USGS real-time
  detections + tag-species resolutions + GLATOS deployment priors), otherwise
  the run degrades honestly to suitability-only (flagged in metadata
  `warnings`). Telemetry is behavioral evidence only — unresolved tags stay
  unresolved, historical priors are never presented as current counts.
- **Model:** six separate 0..1 components (telemetry with spatial/temporal
  decay; date-aware seasonal windows; Gaussian thermal suitability on live SST;
  solar-elevation diel factor; shore-proximity habitat; named movement-corridor
  boxes incl. St. Marys / St. Clair River / Lake St. Clair / Detroit River)
  combined by per-species weighted mean (`config/gamefish_<species>.json`),
  alpha gated by confidence/support. Thermal backbone GLFC Sp87-3 + USGS/USFWS/
  agency literature (thermal/timing values in configs; decay constants are
  labeled MODEL_ASSUMPTIONs). Known v0.1 limits: surface temperature only, no
  bathymetry/substrate, connecting-channel boxes are coarse at this canvas.
- **Outputs:** `site/gamefish_<species>/{current.png,legend.png,metadata.json}`
  (legend 640×300, `legend_size` recorded), entry `kml/<SPECIES>_LIVE.kml` +
  live `site/kml/live/<SPECIES>_LIVE.kml` (one GroundOverlay, versioned
  `?v=<source_id>`), refreshed on the 4×-daily schedule (source id includes the
  run date so seasonal/diel drift rebuilds). Same stage→promote, exit-2, and
  validation contract as every other product.
