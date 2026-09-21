# Great Lakes Live Environment

One automated GitHub system publishing **six independent live raster layers**
for Google Earth, all from official NOAA sources:

| Layer | Source | Refresh wording |
|---|---|---|
| 🌊 Live Wave Height | NCEP operational Great Lakes Wave Unstructured v2.1 (WAVEWATCH III), `HTSGW` analysis | LIVE / CURRENT MODEL (analysis), 6-hourly cycles |
| 🌡️ Live Water Temperature | NOAA/GLERL CoastWatch **GLSEA** daily (satellite AVHRR/VIIRS composite) | CURRENT DAILY |
| 🧊 Live Ice Coverage | U.S. National Ice Center NAIS daily Great Lakes analysis (GRID 1800, %) | LATEST AVAILABLE (seasonal) |
| 🧊 Live Ice Thickness | USNIC NAIS daily SIGRID-3 shapefile → WMO stage-range midpoints, concentration-weighted (**derived**, inches) | LATEST AVAILABLE (seasonal) |
| 🧊 Live Ice Type | USNIC NAIS daily SIGRID-3 shapefile → predominant WMO stage (**analyzed**, 17 categories + unknown) | LATEST AVAILABLE (seasonal) |
| 💨 Live Wind | NCEP GLWU `UGRD/VGRD` surface analysis → knots → **Beaufort Force 0–12** + direction arrows | LIVE / CURRENT MODEL (analysis), 6-hourly cycles |

All water-based layers share one committed NOAA shoreline mask
(`assets/great_lakes_watermask.png`, built from the NOAA Medium-Resolution
Digital Vector Shoreline — nautical-chart compilation, mean-high-water
datum — polygonized once at full vertex precision), so every
overlay cuts out at exactly the same coastline. For crisp shorelines at
all zooms, each product additionally publishes an LOD tile pyramid
(`site/<product>/tiles/`: 2×2 tiles at 2× plus 4×4 at 4× density, masked
from `assets/great_lakes_watermask_4x.png`), referenced from its KML with
Region/Lod hints over a shared-color overview. Tiles are Pages-deployed,
never committed; a KML references tiles only when they were generated in
that run, so failed runs stay overview-only and always resolve.
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
├── scripts/   # 6 independent pipelines + shared raster/KML/validation utils
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
    ├── kml/{Great_Lakes_Live_*.kml}
    └── index.html
```

## Design rules (enforced)

- **Raster only.** Each KML has one overview `GroundOverlay` plus LOD detail
  tiles with Regions (no legend `ScreenOverlay` — rejected by some Google
  Earth clients; legends live in the KML description) + one self-refresh
  `NetworkLink`. Zero `LineString`, `Polygon`, `Placemark`, or per-cell
  features (`validate_outputs.py` asserts this).
- **One common canvas** (`config/great_lakes_bounds.json`, WGS84
  lon −93…−73.5, lat 40.5…49.5, 1800×1175): all six layers align exactly but
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
- Every successful run rebuilds its product fully (tiles included), so the
  deployed site always has matching KML tile references; byte-identical
  outputs simply produce no commit.
- NIC sends no `Last-Modified` header, so ice change-detection uses a
  SHA-256 content hash; GLSEA uses `Last-Modified`; GLWU uses the model
  cycle stamp.
- `python scripts/validate_outputs.py` also runs in CI (`CI=true`), where it
  additionally requires KMLs to carry the real Pages base URL (no
  `REPLACE-` placeholders) and legend PNGs to be 640×210.

Local test: `pip install -r requirements.txt`, then
`python scripts/build_water_temperature.py`,
`python scripts/build_ice_coverage.py`,
`python scripts/build_ice_thickness.py`,
`python scripts/build_ice_type.py`,
`python scripts/build_wave_height.py`,
`python scripts/build_wind.py`,
`python scripts/validate_outputs.py`,
`python scripts/selftest.py`.

## Future AIS layer

Add `scripts/build_ais.py` + `config/ais.json` following the same
download → validate → render → metadata → KML → publish contract; add one job
to the workflow. No changes to the six existing products are needed.
