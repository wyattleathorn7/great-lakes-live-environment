"""Shared North American bounds loader for the four NA live layers.

The Great Lakes builders use config/great_lakes_bounds.json via
geospatial_utils.load_bounds(). The NA builders use THIS module so the
two coverage families can never be confused: every NA layer references
the same config/north_america_bounds.json (lon -180..-15, lat 5..85,
3300x1600, EPSG:4326), which encloses the 100-statute-mile buffered
landmass (see config/north_america_coverage.geojson).
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA_BOUNDS_PATH = os.path.join(REPO_ROOT, "config", "north_america_bounds.json")

_NA_BOUNDS = None


def load_na_bounds():
    global _NA_BOUNDS
    if _NA_BOUNDS is None:
        with open(NA_BOUNDS_PATH) as f:
            _NA_BOUNDS = json.load(f)
    return _NA_BOUNDS


def na_canvas_indices(bounds=None):
    """(lon_grid, lat_grid) cell-center coordinates for the NA canvas."""
    import numpy as np
    b = bounds or load_na_bounds()
    w, h = b["canvas_width"], b["canvas_height"]
    lon = b["lon_min"] + (np.arange(w) + 0.5) * (b["lon_max"] - b["lon_min"]) / w
    lat = b["lat_max"] - (np.arange(h) + 0.5) * (b["lat_max"] - b["lat_min"]) / h
    return np.meshgrid(lon, lat)
