# North American Expansion — four live gradients

Supersedes the Great Lakes basin versions of Live Precipitation, Live Air
Temperature, Live Cloud Cover, and Live Wind with full North American
coverage on ONE shared boundary. All other Great Lakes environmental
layers are untouched.

## Shared boundary (authoritative)

- `config/north_america_bounds.json` — lon −180…−15, lat 5…85,
  3300×1600 px (0.05°/px), EPSG:4326. Read only via `scripts/na_bounds.py`.
- `config/north_america_coverage.geojson` — bbox enclosing the
  100-statute-mile (160.934 km) geodesic coastline buffer + included land
  (Alaska, Canadian mainland, Arctic islands, CONUS, Mexico, Central
  America, Caribbean, Greenland). The bbox encloses the buffer; it does not
  claim the rectangle itself hugs the coast, and it does not imply data
  everywhere — gaps stay transparent.

## Archived Great Lakes layers (disabled, recoverable)

- `archive/great_lakes_original/` — exact config + builder copies,
  original rectangle (−93…−73.5 / 40.5…49.5), `MANIFEST.json`, restore steps.
- Disabled, not deleted: `output/state/<product>.disabled`,
  `site/<product>/DISABLED.json`; original site outputs and
  `Great_Lakes_Live_*.kml` files remain on disk with original bounds.
- Automation: old `wind` / `air_temperature` / `cloud_cover` /
  `precipitation` workflow jobs are `if: false`; fast-live runs the NA jobs.
- Restore: see `archive/great_lakes_original/README.md`.

## Active North American layers

| Layer | Output | KML | Refresh |
|---|---|---|---|
| Live Precipitation — North America | `site/precipitation_na/` | `North_America_Live_Precipitation.kml` | 300 s |
| Live Air Temperature — North America | `site/air_temperature_na/` | `North_America_Live_Air_Temperature.kml` | 3600 s |
| Live Cloud Cover — North America | `site/cloud_cover_na/` | `North_America_Live_Cloud_Cover.kml` | 300 s |
| Live Wind — North America | `site/wind_na/` | `North_America_Live_Wind.kml` | 900 s |

Builders: `scripts/build_*_na.py` + shared `scripts/na_common.py`
(WMS fetch, canvas binning with valid/missing separation, GRIB-missing
sanitization, interior pinhole fill, GDPS WCS fallback) and
`scripts/na_bounds.py`. Configs: `config/*_na.json`.
Validation: `scripts/validate_outputs.py` covers the four NA products on
the NA canvas (size 3300×1600, NA LatLonBox edges, same scale contracts).

## Source precedence (no overlaps, no blending)

- **Precipitation:** IEM N0Q inside lon −130…−65 / lat 22…50 (dry = valid,
  suppresses ECCC) → ECCC RRAI/RSNO outside it where `RADAR_COVERAGE_*.INV`
  is transparent (verified authoritative coverage) → transparent.
  ECCC colors are qualitative rank-maps, not dBZ.
- **Air temperature:** HRRR → RRFS-NA (para feed until prod HTTPS opens) →
  GDPS WCS (verified Central America gap) → transparent. Same 5°F bands.
- **Cloud cover:** GOES-19 + GOES-18 full-disk BCM/DQF → RRFS TCDC →
  GDPS reserve → transparent (never 0%). Same 0–100% scale.
- **Wind:** GLWU (lakes water) → HRRR → RRFS-NA → GDPS speed/dir
  (verified Central America gap) → transparent. Same kt/Beaufort/TOWARD
  arrows. Missing never calm.

## Observations vs model data

- Observed: IEM/ECCC radar, GOES cloud masks. Model: HRRR, RRFS, GDPS,
  GLWU. Every NA KML description + `metadata.json` records per-source
  provenance, timestamps, and which pixels are model fill. Stale products
  are never presented as current (exit 2 keeps the previous raster).

## Known gaps / limitations (2026-10-11)

- RRFS `prod/` HTTPS 403 → `para/` feed used (preferred-first code).
- ECCC radar: no coverage over oceans, far Arctic, Greenland interior.
- GOES geostationary quality degrades at highest Arctic latitudes.
- Far-western Aleutians past −180 and a sliver of NE Greenland sit at the
  canvas edge (documented in the coverage GeoJSON).
- GDPS is 15 km — coarser than HRRR/RRFS — and used only as fallback.
