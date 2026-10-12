# Archived Great Lakes originals — precipitation / wind / cloud cover / air temperature

These are **recoverable, byte-identical copies** of the four Great Lakes
basin layers replaced by the North American expansion. They are DISABLED
in the active project (not rebuilt by any scheduled job, not listed as
active in Google Earth output) and preserved here for audit and restore.

## Contents

- `configs/` — exact copies of `config/precipitation.json`,
  `config/wind.json`, `config/cloud_cover.json`,
  `config/air_temperature.json` at time of archival (2026-10-11).
- `scripts/` — exact copies of `scripts/build_precipitation.py`,
  `scripts/build_wind.py`, `scripts/build_cloud_cover.py`,
  `scripts/build_air_temperature.py` at time of archival.
- `great_lakes_bounds.json` — copy of the original shared rectangle
  (lon −93…−73.5, lat 40.5…49.5, canvas 1800×1175, EPSG:4326).
- `MANIFEST.json` — per-layer record: configuration, source, rendering,
  refresh interval, output path, automation dependencies.

## Original settings (unchanged)

| Layer | Source | Refresh | Output |
|---|---|---|---|
| precipitation | IEM CONUS NEXRAD N0Q `nexrad-n0q-900913` (NOAA WSR-88D), dBZ 5–75 | 300 s | `site/precipitation/`, `Great_Lakes_Live_Precipitation.kml` |
| wind | NCEP GLWU v2.1 UGRD/VGRD over water + HRRR 10 m fill, kt / Beaufort 0–12 | 900 s | `site/wind/`, `Great_Lakes_Live_Wind.kml` |
| cloud_cover | GOES-East (GOES-19) ABI L2 Clear Sky Mask CONUS 2 km, BCM/DQF, 0–100 % | 300 s | `site/cloud_cover/`, `Great_Lakes_Live_Cloud_Cover.kml` |
| air_temperature | NCEP HRRR 3 km TMP 2 m, K→°F, fixed 5 °F bands −60…150 °F | 3600 s | `site/air_temperature/`, `Great_Lakes_Live_Air_Temperature.kml` |

Original automation: `.github/workflows/update_environment.yml`
(wind hourly-group, air_temperature halfhourly-group, cloud_cover +
precipitation dedicated groups) and
`.github/workflows/update_fast_live.yml`
(precipitation + cloud_cover, 5-min). The archived builders are no
longer referenced by either workflow.

Original docs: `DATA_SOURCES.md` §§ 4, 8, 14, 17 (retained in place).

## How to restore

1. Copy `configs/<name>.json` back to `config/<name>.json`.
2. Copy `scripts/build_<name>.py` back to `scripts/build_<name>.py`.
3. Re-add the product job to `.github/workflows/update_environment.yml`
   (and fast-live for precipitation/cloud_cover) and to `publish.needs`.
4. Remove the corresponding `output/state/<product>.disabled` flag and
   re-list the KML in `site/index.html`.
5. Run `python scripts/build_<name>.py && python scripts/validate_outputs.py`.
