"""Vendor the acoustic-telemetry inputs the gamefish model needs.

Reads a great-lakes-live-fish-telemetry checkout (TELEMETRY_REPO env, or the
sibling directory of this repo) and writes the minimal snapshot the model
consumes into assets/gamefish_telemetry/ (same data/ + source/ layout, so
build_gamefish.load_telemetry works unchanged):

  data/live_detections.json, data/live_receivers.json,
  data/detection_history.json, data/tag_species_cache.json,
  source/species_registry.json, source/provenance.json,
  data/receiver_audit.json -- SLIMMED to [{receiver_id, latitude,
  longitude}] (the model only resolves receiver coordinates + counts from
  the 9 MB audit; the full records stay in the telemetry repo).

Run: python scripts/vendor_telemetry.py
Exit 0 = snapshot written; 1 = no telemetry source found.
Re-run before deleting/refreshing the upstream telemetry checkout; the
snapshot's provenance + vendor_manifest.json record its vintage, and the
model's tau_hours decay keeps old evidence honest.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geospatial_utils import REPO_ROOT

SRC_CANDS = [os.environ.get("TELEMETRY_REPO", ""),
             os.path.join(os.path.dirname(REPO_ROOT),
                          "great-lakes-live-fish-telemetry")]
DEST = os.path.join(REPO_ROOT, "assets", "gamefish_telemetry")
COPY = ["data/live_detections.json", "data/live_receivers.json",
        "data/detection_history.json", "data/tag_species_cache.json",
        "source/species_registry.json", "source/provenance.json"]


def main():
    src = next((c for c in SRC_CANDS
                if c and os.path.isdir(os.path.join(c, "data"))), None)
    if src is None:
        print("[vendor_telemetry] no telemetry checkout found "
              f"(tried {SRC_CANDS}).")
        return 1
    made = []
    for rel in COPY:
        s, d = os.path.join(src, rel), os.path.join(DEST, rel)
        with open(s) as f:
            doc = json.load(f)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        with open(d, "w") as f:
            json.dump(doc, f)
        made.append(rel)
    # Slim the audit: the model needs coordinates + counts only.
    with open(os.path.join(src, "data", "receiver_audit.json")) as f:
        audit = json.load(f)
    slim = [{"receiver_id": r.get("receiver_id"),
             "latitude": r.get("latitude"),
             "longitude": r.get("longitude")} for r in audit]
    with open(os.path.join(DEST, "data", "receiver_audit.json"), "w") as f:
        json.dump(slim, f)
    n_coord = sum(1 for r in slim
                  if r["latitude"] is not None and r["longitude"] is not None)
    manifest = {
        "vendored_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"),
        "source": src,
        "files": made + ["data/receiver_audit.json (slimmed)"],
        "audit_records": len(slim),
        "audit_with_coords": n_coord,
    }
    with open(os.path.join(DEST, "vendor_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[vendor_telemetry] snapshot from {src}: "
          f"{len(slim)} audit records ({n_coord} with coords).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
