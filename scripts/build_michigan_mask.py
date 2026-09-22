"""One-time Michigan mask asset (committed; CI never rebuilds it).

Michigan (both peninsulas) from Natural Earth 50m admin-1 (public domain),
rasterized at canvas resolution. Both Michigan-only products load this
identical file and hard-clip to it; Great Lakes water is additionally cut
by the shared shoreline mask (state polygons extend over lake water).

Regeneration (one-time, local):
  python scripts/build_michigan_mask.py --states <ne_50m... base>
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_leaf_footprint import rasterize, read_michigan  # noqa: E402
from geospatial_utils import REPO_ROOT, load_bounds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    args = ap.parse_args()
    bounds = load_bounds()
    W, H = bounds["canvas_width"], bounds["canvas_height"]
    mich = read_michigan(args.states)
    grid = rasterize(mich, bounds, W, H)
    print(f"michigan frac={grid.mean():.4f}")
    from shapely.geometry import Point
    for name, (x, y), want in [
            ("Detroit", (-83.05, 42.33), True),
            ("Ironwood", (-90.17, 46.45), True),
            ("CopperHarbor", (-87.89, 47.47), True),
            ("Mackinaw", (-84.73, 45.78), True),
            ("Chicago", (-87.63, 41.88), False),
            ("Toronto", (-79.38, 43.65), False),
            ("Superior-water", (-88.0, 47.5), True),
            ("Wisconsin", (-89.5, 45.0), False)]:
        got = bool(mich.contains(Point(x, y)))
        print(f"  {name}: {got} (want {want})")
        assert got == want, name
    out = os.path.join(REPO_ROOT, "assets", "michigan_mask.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray((grid * 255).astype(np.uint8), mode="L").save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
