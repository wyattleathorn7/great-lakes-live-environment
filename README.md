# Great Lakes Live Environment

One automated GitHub system publishing **forty-two independent live raster layers**
for Google Earth: twenty-six environmental layers from official NOAA/NASA operational
sources, plus **sixteen live game-fish distribution gradients** (see below).

| Layer | Source | Refresh wording |
|---|---|---|
| 🌊 Live Wave Height | NCEP operational Great Lakes Wave Unstructured v2.1 (WAVEWATCH III), `HTSGW` analysis | LIVE / CURRENT MODEL (analysis), 6-hourly cycles |
| 🌡️ Live Water Temperature | NOAA/GLERL CoastWatch **GLSEA** daily (satellite AVHRR/VIIRS composite) | CURRENT DAILY |
| 🧊 Live Ice Coverage | U.S. National Ice Center NAIS daily Great Lakes analysis (GRID 1800, %) | LATEST AVAILABLE (seasonal) |
| 🧊 Live Ice Thickness | USNIC NAIS daily SIGRID-3 shapefile → WMO stage-range midpoints, concentration-weighted (**derived**, inches) | LATEST AVAILABLE (seasonal) |
| 🧊 Live Ice Type | USNIC NAIS daily SIGRID-3 shapefile → predominant WMO stage (**analyzed**, 17 categories + unknown) | LATEST AVAILABLE (seasonal) |
| 💨 Live Wind | NCEP GLWU `UGRD/VGRD` surface analysis → knots → **Beaufort Force 0–12** + direction arrows | LIVE / CURRENT MODEL (analysis), 6-hourly cycles |
| 🍂 Live Leaf Color | NASA MODIS Aqua MYD13A1/MYD09A1 global 500 m via Planetary Computer (NDVI trajectory + reflectance + NDSI snow) → circular phenology phase, basin-wide land (open water transparent) | LATEST AVAILABLE COMPOSITE |
| 🦠 Live Chlorophyll | NOAA CoastWatch S-NPP VIIRS chlorophyll-a, Science Quality, Global 4 km, Daily (ERDDAP); newest-valid 7-day mosaic; balanced linear scale, water-only | LATEST AVAILABLE (daily composite) |
| 🌫️ Live Water Clarity | NOAA CoastWatch S-NPP VIIRS Kd(PAR) diffuse attenuation, NRT, Global 4 km, Daily (ERDDAP); newest-valid 7-day mosaic; balanced linear scale, water-only | LATEST AVAILABLE (daily composite) |
| ☀️ Live Solar Radiation | NOAA/NCEP RAP hourly DSWRF surface flux (incoming sunlight W/m²; not a UV Index), value valid for the current hour; fixed sequential scale, full basin rectangle | LIVE / CURRENT MODEL (hourly RAP cycle) |
| 🌡️ Live Air Temperature | NOAA/NCEP HRRR 3 km 2 m temperature analysis, hourly cycles; FIXED brightened banded 5°F key (-60..150 °F, blue-white → royal freeze wall → yellows → scorching pink/reds), full basin rectangle | LIVE / CURRENT MODEL (analysis), hourly cycles |
| ❄️ Live Snow Coverage | NOAA/NCEP HRRR 3 km SNOD/SNOWC snow analysis (ground snow gate), hourly cycles, basin-wide (open water transparent) | LIVE / CURRENT MODEL (analysis), hourly cycles |
| ☁️ Live Cloud Cover | NOAA/NCEP HRRR 3 km TCDC entire-atmosphere analysis, hourly cycles; basin rectangle, clear (<1 %) transparent | LIVE / CURRENT MODEL (hourly cycle, 15-min steps) |
| 🧭 Live Surface Pressure | NOAA/NCEP HRRR 3 km MSLMA sea-level pressure analysis, hourly cycles; basin rectangle, 1013.25 hPa mid-scale | LIVE / CURRENT MODEL (analysis), hourly cycles |
| 🧭 Live Wave Direction | NCEP GLWU v2.1 PERPW peak period (seconds) + WVDIR travel arrows, water-only | LIVE / CURRENT MODEL (analysis), 6-hourly cycles |
| 🧭 Live Wave Direction | NCEP GLWU v2.1 PERPW peak period (seconds) + WVDIR travel arrows, water-only | LIVE / CURRENT MODEL (analysis), 6-hourly cycles |
| 🔄 Live Surface Currents | NOAA/NOS GLOFS (LSOFS/LMHOFS/LEOFS/LOOFS) surface U/V nowcast + GLERL experimental FVCOM corridor fill (HEC + lower St. Marys); speed gradient (cm/s) + toward-motion arrows rasterized in | LIVE / CURRENT MODEL (nowcast), 6-hourly cycles |
| 🌧️ Live Precipitation | IEM (Iowa State) CONUS NEXRAD N0Q base reflectivity from NOAA WSR-88D (dBZ, all-season rain/snow/storms), 5-min mosaic; basin rectangle, no-echo transparent | LIVE / CURRENT OBSERVATION (NEXRAD 5-min mosaic) |
| 🌌 Live Ovation Aurora Forecast | NOAA/SWPC OVATION Prime auroral grid (`ovation_aurora_latest.json`) + hemispheric power / forecast Kp context; fixed 0–30+ intensity scale, full basin rectangle (same footprint as Leaf Color) | LIVE / CURRENT FORECAST (OVATION Prime, hourly check) |
| 💧 Live Water Quality | USGS Water Quality Portal E. coli/enterococci/fecal coliform (7-day newest-per-station halos) under EPA BEACON + EPA 2012 RWQC & Red Book thresholds (GM/STV/BAV); transparent NO DATA where unobserved, water-only all five lakes | SOURCE CHECKED HOURLY / LATEST VALID OBSERVATION |
| 🌙 Live Atmospheric Visibility | NCEP HRRR `VIS` surface analysis (m → statute miles), FIXED 0–30 mi scale | LIVE / CURRENT MODEL (analysis), hourly cycles |
| ♨️ Live Relative Humidity | NCEP HRRR `RH` 2 m analysis (%), FIXED 0–100 % scale | LIVE / CURRENT MODEL (analysis), hourly cycles |
| ☁️ Live Precipitable Water Vapor | NCEP HRRR `PWAT` entire-atmosphere analysis (kg m⁻² = mm), FIXED 0–60 mm scale | LIVE / CURRENT MODEL (analysis), hourly cycles |
| 💡 Live Light Pollution | NASA VIIRS Black Marble annual nighttime-lights composite via GIBS WMTS (relative brightness index 0–100) | LATEST AVAILABLE COMPOSITE (annual, hourly check) |
| 🌫️ Live Air Quality | ECMWF CAMS PM2.5 analysis via Open-Meteo (µg/m³, EPA-anchored FIXED 0–150 scale, labeled modeled) | LIVE / CURRENT MODEL (hourly check) |
| 💦 Live Condensation | HRRR TMP+DPT+RH-derived condensation-favorability index 0–100 (**derived**, no wind/visibility inputs) | LIVE / CURRENT MODEL (derived analysis), hourly cycles |
| 🌫️ Live Fog Risk/Active Fog | HRRR TMP+DPT+RH+10 m wind+VIS-derived fog-risk index 0–100 with active-fog visibility gate (**derived**) | LIVE / CURRENT MODEL (derived analysis), hourly cycles |
| 🧊 Live Dew Point | NCEP HRRR `DPT` 2 m analysis (K → °F), FIXED −20…90 °F scale | LIVE / CURRENT MODEL (analysis), hourly cycles |

Sixteen additional **live game-fish distribution gradients** (walleye, yellow perch,
lake trout, steelhead, brown trout, smallmouth bass, northern pike, muskellunge,
lake sturgeon, lake whitefish, chinook salmon, coho salmon, largemouth bass,
burbot, cisco, sauger) are produced by
`scripts/build_gamefish.py` (matrix job, one species per run) from live GLSEA SST
+ acoustic-telemetry evidence via a six-component model (telemetry × seasonal ×
thermal × diel × habitat × corridors, weighted mean, confidence-gated transparency).
Telemetry support varies: whitefish/cisco have strong GLATOS programs;
chinook/coho/largemouth/burbot/sauger have
no current telemetry and run suitability-only at low confidence (flagged in
metadata). Each publishes `site/gamefish_<species>/{current.png,legend.png,
metadata.json}` plus entry/live KMLs (`WALLEYE_LIVE.kml`, etc.). Modeled likelihood
only — never fish counts. See DATA_SOURCES.md §13 and `scripts/gamefish_model.py`.

All water-based layers share one committed NOAA shoreline mask
(`assets/great_lakes_watermask.png`, built from the NOAA Medium-Resolution
Digital Vector Shoreline — nautical-chart compilation, mean-high-water
datum — polygonized once at full vertex precision), so every
overlay cuts out at exactly the same coastline. Each KML is a single
overview GroundOverlay (Google Earth Web caps external images per document
and does not support Region, so multi-image tile pyramids are deliberately
avoided) with a self-refresh NetworkLink. Tiles are never generated.
KMLs use GroundOverlay + self-refresh NetworkLink only — no ScreenOverlay
(rejected by some Google Earth clients); legends live in each KML
description (PNG + scale text) and as standalone `legend.png` files.

See [DATA_SOURCES.md](DATA_SOURCES.md) for the verified endpoints, formats,
resolutions, and update intervals (probed live 2026-09-21).

## Repository layout

The existing `Great Lakes.kml` project file is **never touched** — these are
separate KML/KMZ products:

```text
.
├── .github/workflows/update_environment.yml  # schedule + manual dispatch
├── scripts/   # per-product pipelines (environment + gamefish) + shared raster/KML/validation utils
├── config/    # ONE common bounds/CRS + per-product configs
├── output/    # raw downloads (git-ignored) + per-product state
├── assets/    # shared NOAA shoreline masks (committed, identical for all)
├── kml/       # canonical KMLs (copied to site/kml/ on each build)
└── site/      # GitHub Pages root: stable URLs
    ├── wave_height/{current.png,legend.png,metadata.json,tiles/}
    ├── water_temperature/{...}
    ├── ice_coverage/{...}
    ├── ice_thickness/{...}
    ├── ice_type/{...}
    ├── wind/{...}
    ├── gamefish_walleye/{current.png,legend.png,metadata.json}
    ├── gamefish_<8 more species>/{...}
    ├── kml/{Great_Lakes_Live_*.kml,WALLEYE_LIVE.kml,...}
    └── index.html
```

## Design rules (enforced)

- **Raster only.** Each KML has one overview `GroundOverlay` plus LOD detail
  tiles with Regions (no legend `ScreenOverlay` — rejected by some Google
  Earth clients; legends live in the KML description) + one self-refresh
  `NetworkLink`. Zero `LineString`, `Polygon`, `Placemark`, or per-cell
  features (`validate_outputs.py` asserts this).
- **One common canvas** (`config/great_lakes_bounds.json`, WGS84
  lon −93…−73.5, lat 40.5…49.5, 1800×1175): all layers align exactly but
  share no data.
- **Transparent outside valid water.** Land and missing data are alpha=0, so
  shipwrecks, lighthouses, harbors, and parks stay visible. Open water at 0 %
  ice is likewise transparent.
- **Fault tolerant.** Exit code 2 = source/validation failure: the previous
  valid raster is kept and the other products still update. Each product job
  fails independently in Actions.
- **Cache-busting without URL churn.** Stable filenames; every successful run
  rewrites the KML with `current.png?v=<timestamp>` + a self-`NetworkLink`
  (`onInterval`), so Google Earth picks up new imagery on refresh.
- **Honest timestamps.** Every legend, KML description, and `metadata.json`
  carries the real NOAA data time — never "real-time" for daily products.

## Setup (one time, repo owner)

1. Create a GitHub repo (suggested name `great-lakes-live-environment`) and
   push this directory as its root.
2. Enable **Settings → Pages → Build and deployment → Source: GitHub Actions**
   (the workflow publishes `site/` via `deploy-pages`). The workflow derives
   the public base URL automatically
   (`https://<owner>.github.io/<repo>`); KMLs are regenerated with it.
3. Run the workflow manually once (**Run workflow → product: all**), then open
   `https://<owner>.github.io/<repo>/` and add the six
   `/kml/Great_Lakes_Live_*.kml` links to Google Earth independently.

## Reliability notes (production)

- Runners are pinned to `ubuntu-24.04` (GitHub migrates `ubuntu-latest` to
  26.04 beginning 2026-10-19; validate before adopting). Actions track
  current Node-24 majors (checkout v7, setup-python v7, artifacts v7/v8,
  pages v5/v6, auto-commit v7).
- Downloads retry transient failures (HTTP 5xx/408/429, timeouts, resets)
  and never retry 404s; files are written atomically (`.part` + rename).
- Each product renders into `output/stage/<product>/` and is promoted to
  `site/` + `kml/` only on full success, so artifacts can never contain a
  half-updated layer.
- Any download/parse/validation failure exits 2: the previous valid raster
  is kept, the failure is logged, the job stays green, and the other
  products update normally. Unexpected engine errors exit 1 (red job).
- Every successful run rebuilds its product fully; byte-identical
  outputs simply produce no commit.
- NIC sends no `Last-Modified` header, so ice change-detection uses a
  SHA-256 content hash; GLSEA uses `Last-Modified`; GLWU uses the model
  cycle stamp.
- `python scripts/validate_outputs.py` also runs in CI (`CI=true`), where it
  additionally requires KMLs to carry the real Pages base URL (no
  `REPLACE-` placeholders) and legend PNGs to match their metadata
  `legend_size` (environment legends are 640×210; game-fish legends 640×300).

Local test: `pip install -r requirements.txt`, then
`python scripts/build_water_temperature.py`,
`python scripts/build_ice_coverage.py`,
`python scripts/build_ice_thickness.py`,
`python scripts/build_ice_type.py`,
`python scripts/build_wave_height.py`,
`python scripts/build_wind.py`,
`python scripts/build_leaf_color.py`,
`python scripts/build_chlorophyll.py`,
`python scripts/build_water_clarity.py`,
`python scripts/build_solar_radiation.py`,
`python scripts/build_air_temperature.py`,
`python scripts/build_snow_coverage.py`,
`python scripts/build_cloud_cover.py`,
`python scripts/build_surface_pressure.py`,
`python scripts/build_wave_direction.py`,
`python scripts/build_surface_currents.py`,
`python scripts/build_precipitation.py`,
`python scripts/build_aurora.py`,
`python scripts/build_bacteria.py`,
`python scripts/build_visibility.py`,
`python scripts/build_humidity.py`,
`python scripts/build_precipitable_water.py`,
`python scripts/build_light_pollution.py`,
`python scripts/build_air_quality.py`,
`python scripts/build_condensation.py`,
`python scripts/build_fog.py`,
`python scripts/build_dew_point.py`,
`python scripts/build_gamefish.py --species walleye` (16 species: walleye,
yellow_perch, lake_trout, steelhead, brown_trout, smallmouth_bass,
northern_pike, muskellunge, lake_sturgeon, lake_whitefish, chinook_salmon,
coho_salmon, largemouth_bass, burbot, cisco, sauger),
`python scripts/validate_outputs.py`,
`python scripts/selftest.py`.

## Future AIS layer

Add `scripts/build_ais.py` + `config/ais.json` following the same
download → validate → render → metadata → KML → publish contract; add one job
to the workflow. No changes to the existing products are needed.
