# Great Lakes Live Environment

One automated GitHub system publishing **three independent live raster layers**
for Google Earth, all from official NOAA sources:

| Layer | Source | Refresh wording |
|---|---|---|
| 🌊 Live Wave Height | NCEP operational Great Lakes Wave Unstructured v2.1 (WAVEWATCH III), `HTSGW` analysis | LIVE / CURRENT MODEL (analysis), 6-hourly cycles |
| 🌡️ Live Water Temperature | NOAA/GLERL CoastWatch **GLSEA** daily (satellite AVHRR/VIIRS composite) | CURRENT DAILY |
| 🧊 Live Ice Coverage | U.S. National Ice Center NAIS daily Great Lakes analysis (GRID 1800) | LATEST AVAILABLE (seasonal) |

See [DATA_SOURCES.md](DATA_SOURCES.md) for the verified endpoints, formats,
resolutions, and update intervals (probed live 2026-09-21).

## Repository layout

The existing `Great Lakes.kml` project file is **never touched** — these are
separate KML/KMZ products:

```text
.
├── .github/workflows/update_environment.yml  # schedule + manual dispatch
├── scripts/   # 3 independent pipelines + shared raster/KML/validation utils
├── config/    # ONE common bounds/CRS + per-product configs
├── output/    # raw downloads (git-ignored) + per-product state
├── kml/       # canonical KMLs (copied to site/kml/ on each build)
└── site/      # GitHub Pages root: stable URLs
    ├── wave_height/{current.png,legend.png,metadata.json}
    ├── water_temperature/{...}
    ├── ice_coverage/{...}
    ├── kml/{Great_Lakes_Live_*.kml}
    └── index.html
```

## Design rules (enforced)

- **Raster only.** Each KML has one `GroundOverlay` + one legend
  `ScreenOverlay` + one self-refresh `NetworkLink`. Zero `LineString`,
  `Polygon`, `Placemark`, or per-cell features (`validate_outputs.py` asserts this).
- **One common canvas** (`config/great_lakes_bounds.json`, WGS84
  lon −93…−73.5, lat 40.5…49.5, 1800×1175): the three layers align exactly but
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
2. Enable **GitHub Pages → Deploy from GitHub Actions** (the workflow
   publishes `site/` via `deploy-pages`). The workflow derives the public
   base URL automatically
   (`https://<owner>.github.io/<repo>`); KMLs are regenerated with it.
3. Run the workflow manually once (**Run workflow → product: all**), then open
   `https://<owner>.github.io/<repo>/` and add the three
   `/kml/Great_Lakes_Live_*.kml` links to Google Earth independently.

Local test: `pip install -r requirements.txt`, then
`python scripts/build_water_temperature.py`,
`python scripts/build_ice_coverage.py`,
`python scripts/build_wave_height.py`,
`python scripts/validate_outputs.py`.

## Future AIS layer

Add `scripts/build_ais.py` + `config/ais.json` following the same
download → validate → render → metadata → KML → publish contract; add one job
to the workflow. No changes to the three existing products are needed.
