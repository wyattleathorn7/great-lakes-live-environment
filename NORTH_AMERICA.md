# Continental Expansion — four live gradients

Supersedes the Great Lakes basin versions of Live Precipitation, Live Air
Temperature, Live Cloud Cover, and Live Wind with Continental
coverage on selectable boundaries. All other Great Lakes environmental
layers are untouched.

## Coverage switch (all four NA layers)

Every NA builder reads `coverage_mode` from its `config/*_na.json`:
`basin` (original Great Lakes rectangle −93…−73.5 / 40.5…49.5),
`landonly` (Continental Coverage: crisp worldwide land incl. Antarctica,
ocean/seas cut — 3600×1800 global canvas), or `global` (worldwide
3600×1800 canvas, lon −180…180 / lat −90…90 at 0.10°/px — every valid
pixel paints, gaps stay transparent, no mask cuts at all). Current
defaults: wind, precipitation, and cloud cover use `global`; air
temperature uses `landonly`. To change a layer's coverage, edit its
config value and rebuild — no code changes needed.

## Coverage switch (Google Earth)

Each gradient's live folder holds all three coverages as hideable
overlays — Great Lakes Basin, Continental, Global —
with only the default switched on. The folder is marked radioFolder, so
Google Earth Pro flips them exclusively; Web/app users check exactly
one box. No links, no extra files: switching happens inside the folder
you already added. Basin is painted on the NA canvas; Continental +
Global are painted worldwide on the 3600×1800 canvas from the same
sources (high-res nests where they exist, GDPS worldwide fill
elsewhere), with the identical legend, colors, opacity, and per-degree
label/streamlet density — per-coverage rasters under
`site/<product>_{basin,continental,global}/`.

## Shared boundary (authoritative)

- `config/north_america_bounds.json` — lon −180…−15, lat 5…85,
  3300×1600 px (0.05°/px), EPSG:4326. Read only via `scripts/na_bounds.py`.
- `config/global_bounds.json` — lon −180…180, lat −90…90,
  3600×1800 px (0.10°/px, same grid as `aurora_global`), EPSG:4326. The
  Global + Continental variants; Global has no mask, Continental uses
  `assets/global_landonly.npy` (crisp worldwide land incl. Antarctica).
  Basin keeps the NA canvas above.
- `config/north_america_coverage.geojson` — bbox enclosing the
  100-statute-mile (160.934 km) geodesic coastline buffer + included land
  (Alaska, Canadian mainland, Arctic islands, CONUS, Mexico, Central
  America, Caribbean, Greenland). The bbox encloses the buffer; it does not
  claim the rectangle itself hugs the coast, and it does not imply data
  everywhere — gaps stay transparent.
- Coverage helpers: `scripts/na_common.load_coverage_mask` /
  `apply_coverage_mask` / `gate_field_to_coverage` (`global` is a no-op:
  every valid pixel paints). The retired `extended200` offshore mask
  (`scripts/build_na_extmask.py` + `assets/north_america_extended.*`)
  was deleted when Extended became Global.

## Archived Great Lakes layers (disabled, recoverable)

- `archive/great_lakes_original/` — exact config + builder copies,
  original rectangle (−93…−73.5 / 40.5…49.5), `MANIFEST.json`, restore steps.
- Disabled, not deleted: `output/state/<product>.disabled`,
  `site/<product>/DISABLED.json`; original site outputs and
  `Great_Lakes_Live_*.kml` files remain on disk with original bounds.
- Automation: old `wind` / `air_temperature` / `cloud_cover` /
  `precipitation` workflow jobs are `if: false`; fast-live runs the NA jobs.
- Restore: see `archive/great_lakes_original/README.md`.

## Active Continental layers

| Layer | Output | KML | Refresh |
|---|---|---|---|
| Live Precipitation — Continental | `site/precipitation_na/` | `North_America_Live_Precipitation.kml` | 300 s |
| Live Air Temperature — Continental | `site/air_temperature_na/` | `North_America_Live_Air_Temperature.kml` | 3600 s |
| Live Cloud Cover — Continental | `site/cloud_cover_na/` | `North_America_Live_Cloud_Cover.kml` | 300 s |
| Live Wind — Continental | `site/wind_na/` | `North_America_Live_Wind.kml` | 900 s |

Builders: `scripts/build_*_na.py` + shared `scripts/na_common.py`
(WMS fetch, canvas binning with valid/missing separation, GRIB-missing
sanitization, interior pinhole fill, GDPS WCS fallback) and
`scripts/na_bounds.py`. Configs: `config/*_na.json`.
Validation: `scripts/validate_outputs.py` covers the four NA products:
wind/precipitation/cloud primaries on the global canvas (3600×1800,
world LatLonBox), air temperature on the NA canvas (3300×1600), same
scale contracts, 3 hideable overlays + radioFolder + 1-ON/2-OFF.

## Source precedence (no overlaps, no blending)

- **Precipitation:** MRMS SeamlessHSR inside lon −130…−65 / lat 22…50
  (dry = valid, suppresses ECCC; replaced the old IEM N0Q WMS source
  within the same footprint) → ECCC RRAI/RSNO outside it where
  `RADAR_COVERAGE_*.INV` is transparent (verified authoritative coverage)
  → regional MRMS SeamlessHSR (ALASKA / HAWAII / CARIB domains) in the
  remaining gaps — this restores the old IEM regional footprint that the
  ECCC INV layer reads as falsely "covered" → OPERA EU DBZH where neither
  covers (e.g. Iceland) → JMA nowcast
  intensity (mm/h classes 1–80 via Z-R equivalents, undifferentiated) in
  the Japan box (lon 122–148, lat 24–46) → GFS estimates
  (CSNOW/CRAIN, same colors as radar, ≥0.25 mm/h,
  organized systems only) in verified radar gaps → transparent.
  The IEM box is intersected with land+~200km offshore (WSR-88D range),
  so open ocean never suppresses model estimates — previously the raw
  box hid GFS-wet open-ocean pixels and cut systems at lon −130.
  GFS longitude is periodic: the −180 grid column is duplicated at +180
  and flag lookup wraps, so no seam at the dateline.
  All echo pixels share one uniform alpha (160/255) and one color
  scale — estimates are disclosed as model output in the metadata
  and registry, not by separate colors.
  Rain and snow both render on the original N0Q key: there is no
  separate snow ramp or snow legend. ECCC colors are Z-R estimated dBZ
  equivalents (saturate ~57 dBZ), not measured dBZ; GFS estimates use
  the same colors (model output).
  Fine ~1 km MRMS is max-composited onto the canvas cell, so no
  echo is dropped and the −99 no-echo sentinel cannot paint a halo.
- **Air temperature:** HRRR → RRFS-NA (para feed until prod HTTPS opens) →
  GDPS WCS worldwide → transparent. Same 5°F bands.
- **Cloud cover:** GOES-19 + GOES-18 full-disk BCM/DQF → RRFS TCDC →
  GDPS worldwide → transparent (never 0%). Same 0–100% scale.
- **Wind:** GLWU (lakes water) → HRRR → RRFS-NA → GDPS speed/dir
  worldwide → transparent. Same kt/Beaufort/TOWARD
  arrows. Missing never calm.

## Observations vs model data

- Observed: MRMS/ECCC radar, GOES cloud masks. Model: HRRR, RRFS, GDPS,
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
