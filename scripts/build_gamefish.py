"""Pipeline G — LIVE GAME-FISH DISTRIBUTION GRADIENTS (one species per run).

Six-component model (telemetry x seasonal x thermal x diel x habitat x
corridors, weighted mean + confidence-gated transparency) rendered onto the
shared canvas. Follows the standard product contract:

  --species <key> (required: walleye, yellow_perch, lake_trout, steelhead,
                   brown_trout, smallmouth_bass, northern_pike, muskellunge,
                   lake_sturgeon). Chinook/Coho have no configs (insufficient
                   telemetry evidence) and are NOT built.
  Exit 0 = updated (or skipped, source unchanged); 2 = source/validation
  failure (previous valid raster untouched); 1 = unexpected error.
  Stage-first rendering + promote_stage() on full success only.

Live inputs: NOAA GLSEA SST (same URL as the water-temperature product;
failure -> exit 2) + acoustic-telemetry evidence (optional: TELEMETRY_REPO
env, ./telemetry_src CI checkout, or local sibling checkout; absent ->
suitability-only run with warning, still valid).
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gamefish_model as G
from build_kml import (assert_no_vector_geometry, build_entry_kml, build_kml,
                       legend_block, live_out_dirs)
from geospatial_utils import (RENDER_VERSION, REPO_ROOT, SITE_DIR, base_metadata,
                              bin_to_canvas, canvas_indices, download,
                              ensure_coords, http_date_to_det, now_det_str,
                              promote_stage, read_state, save_png, source_token,
                              stage_dir, write_metadata, write_state)

GLSEA_URL = json.load(open(os.path.join(
    REPO_ROOT, "config", "water_temperature.json")))["source_url"]
RAW_DIR = os.path.join(REPO_ROOT, "output", "raw")

SPECIES = ["walleye", "yellow_perch", "lake_trout", "steelhead", "brown_trout",
           "smallmouth_bass", "northern_pike", "muskellunge", "lake_sturgeon"]
REGISTRY_NAMES = {"walleye": "Walleye", "yellow_perch": "Yellow Perch",
                  "lake_trout": "Lake Trout", "steelhead": "Steelhead",
                  "brown_trout": "Brown Trout",
                  "smallmouth_bass": "Smallmouth Bass",
                  "northern_pike": "Northern Pike", "muskellunge": "Muskellunge",
                  "lake_sturgeon": "Lake Sturgeon"}

PRODUCT_OF = {s: f"gamefish_{s}" for s in SPECIES}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, choices=SPECIES)
    return ap.parse_args()


def main():
    try:
        return run(parse_args().species)
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        return 1


# ---------------------------------------------------------- telemetry input
def find_telemetry_repo():
    cands = [os.environ.get("TELEMETRY_REPO", ""),
             os.path.join(REPO_ROOT, "telemetry_src"),
             os.path.join(os.path.dirname(REPO_ROOT),
                          "great-lakes-live-fish-telemetry")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "data")):
            return c
    return None


def load_telemetry(troot):
    def _load(rel, default):
        p = os.path.join(troot, rel)
        if not os.path.exists(p):
            return default
        with open(p) as f:
            return json.load(f)
    live_det = _load("data/live_detections.json", [])
    live_rx = _load("data/live_receivers.json", [])
    history = _load("data/detection_history.json", {})
    audit = _load("data/receiver_audit.json", [])
    registry = _load("source/species_registry.json", {"entries": {}})
    tag_cache = _load("data/tag_species_cache.json", {})
    coords = {r["receiver_id"]: (r.get("latitude"), r.get("longitude"))
              for r in live_rx}
    snap = []
    for d in live_det:
        c = coords.get(d["receiver_id"])
        if not c or c[0] is None or c[1] is None:
            continue
        snap.append({"receiver_id": d["receiver_id"], "latitude": c[0],
                     "longitude": c[1],
                     "species_counts_24h": d.get("species_counts_24h", {}),
                     "last_detection": d.get("last_detection")})
    proj_map = {}
    for sp, ent in registry.get("entries", {}).items():
        codes = []
        for e in ent.get("evidence", []):
            if e.startswith("GLATOS project"):
                codes.append(e.split(":")[0].split()[-1].strip())
        proj_map[sp] = codes
    rx_by_id = {r["receiver_id"]: r for r in (audit or [])}
    tele_id = _load("source/provenance.json", {}).get("live_version", "none")
    tele_id = f"{tele_id}|audit={len(audit)}"
    return {"snapshot": snap, "history": history, "audit": audit,
            "proj_map": proj_map, "rx_by_id": rx_by_id,
            "tag_cache": tag_cache, "tele_id": tele_id}


def resolve_history(tele, reg_name):
    """History events whose tags authoritatively resolve to this species."""
    out = []
    for rid, events in (tele.get("history") or {}).items():
        r = tele["rx_by_id"].get(rid)
        if not r or r.get("latitude") is None or r.get("longitude") is None:
            continue
        for e in events[-40:]:
            ok = [t for t in e.get("tags", [])
                  if str((tele["tag_cache"].get(t) or {}).get("common", "")
                         ).lower() == reg_name.lower()]
            if ok:
                e2 = dict(e)
                e2["_lon"] = r["longitude"]
                e2["_lat"] = r["latitude"]
                out.append(e2)
    return out


# ------------------------------------------------------------------ pipeline
def run(species):
    from geospatial_utils import load_bounds
    cfg = json.load(open(os.path.join(
        REPO_ROOT, "config", f"gamefish_{species}.json")))
    PRODUCT = PRODUCT_OF[species]
    bounds = load_bounds()
    shape = (bounds["canvas_height"], bounds["canvas_width"])
    now = datetime.now(timezone.utc)

    raw_path = os.path.join(RAW_DIR, "glsea_cur.asc")
    try:
        info = download(GLSEA_URL, raw_path)
    except Exception as e:
        print(f"[{PRODUCT}] DOWNLOAD FAILED (keeping previous): {e}")
        return 2
    if info["size_bytes"] < 1_000_000:
        print(f"[{PRODUCT}] file too small; keeping previous.")
        return 2
    with open(raw_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    glsea_id = f"glsea-{info['http_last_modified'] or f'sha256:{digest[:16]}'}"

    troot = find_telemetry_repo()
    tele = load_telemetry(troot) if troot else None
    tele_id = tele["tele_id"] if tele else "telemetry-none"
    source_id = (f"gfish-{now.strftime('%Y-%m-%d')}-{glsea_id}-"
                 f"{hashlib.sha256(tele_id.encode()).hexdigest()[:12]}"
                 f"-m{G.MODEL_VERSION}-r{RENDER_VERSION}")
    prev = read_state(PRODUCT)
    live_kml_name = cfg["kml_filename"]
    if prev.get("source_id") == source_id \
            and os.path.exists(os.path.join(SITE_DIR, PRODUCT, "current.png")) \
            and os.path.exists(os.path.join(SITE_DIR, "kml", "live", live_kml_name)):
        print(f"[{PRODUCT}] source unchanged; refreshing KMLs only.")
        rewrite_kmls_from_metadata(cfg, PRODUCT)
        return 0
    try:
        return _build(cfg, PRODUCT, species, shape, now, info, raw_path,
                      source_id, tele)
    except Exception:
        traceback.print_exc()
        print(f"[{PRODUCT}] VALIDATION FAILED; keeping previous.")
        return 2


def _build(cfg, PRODUCT, species, shape, now, info, raw_path, source_id, tele):
    from geospatial_utils import load_bounds
    bounds = load_bounds()
    with open(raw_path) as f:
        header = [f.readline() for _ in range(6)]
    try:
        ncols = int(float(header[0].split()[-1]))
        nrows = int(float(header[1].split()[-1]))
    except (ValueError, IndexError):
        print(f"[{PRODUCT}] bad GLSEA header.")
        return 2
    if (ncols, nrows) != (1024, 1024):
        print(f"[{PRODUCT}] bad GLSEA dims.")
        return 2
    data = np.loadtxt(raw_path, skiprows=6)
    if data.shape != (1024, 1024):
        print(f"[{PRODUCT}] bad GLSEA shape.")
        return 2
    cdir = ensure_coords(RAW_DIR)
    lats = np.loadtxt(os.path.join(cdir, "1024_latgrid.txt"))
    lons = np.loadtxt(os.path.join(cdir, "1024_longrid.txt"))
    lake_ids = np.loadtxt(os.path.join(cdir, "1024_lake_ids.txt"))
    ok_src = (lake_ids >= 1) & (lake_ids <= 6) & np.isfinite(data) \
        & (data > -3) & (data < 45)
    n_valid = int(ok_src.sum())
    if n_valid < 80_000:
        print(f"[{PRODUCT}] implausible SST coverage ({n_valid}).")
        return 2
    rows, cols, valid = canvas_indices(lats, lons, bounds)
    vals = np.where(ok_src, data, np.nan)
    sst, _counts = bin_to_canvas(rows, cols, vals.ravel(), valid,
                                 shape, splat_radius=1)
    print(f"[{PRODUCT}] SST C range=[{np.nanmin(sst):.1f},{np.nanmax(sst):.1f}]")

    reg_name = REGISTRY_NAMES[species]
    projs = (tele["proj_map"].get(reg_name, []) if tele else [])
    hist = resolve_history(tele, reg_name) if tele else []
    t_field, t_info = G.build_telemetry(
        tele["snapshot"] if tele else [], hist, tele["audit"] if tele else [],
        reg_name, projs, cfg["telemetry"]["tau_hours"],
        cfg["telemetry"]["sigma_deg"], now)

    shore, off, h_field = G.build_habitat(cfg["habitat"])
    c_field = G.build_corridors([(c["name"], c["weight"]) for c in cfg["corridors"]])
    th_field = G.build_thermal(sst, cfg["thermal"]["preferred_c"],
                               cfg["thermal"]["sigma_c"])
    d_field, d_info = G.build_diel(now, cfg["diel"], shore, off)
    m_field, m_info = G.build_seasonal(now.date(), cfg["seasonal_windows"],
                                       c_field, shore, off)

    water = G._water()
    comps = {}
    for k, f in {"T": t_field, "M": m_field, "TH": th_field, "D": d_field,
                 "H": h_field, "C": c_field}.items():
        f = np.asarray(f, dtype=np.float32)
        f[~water] = 0
        f[~np.isfinite(f)] = 0
        comps[k] = np.clip(f, 0, 1)
    final = G.combine(comps, cfg["combination_weights"])
    final[~water] = 0
    if not np.isfinite(final).all() or final.max() < 0.05:
        print(f"[{PRODUCT}] degenerate final field; keeping previous.")
        return 2
    conf = G.confidence_value(comps, water, cfg["uncertainty"]["telemetry_bonus"])
    rgba = G.render_species(final, conf, species, alpha=bounds["overlay_alpha"])
    n_opaque = int((rgba[:, :, 3] > 0).sum())
    if n_opaque < 10_000:
        print(f"[{PRODUCT}] too few water pixels ({n_opaque}).")
        return 2

    warnings = []
    if tele is None:
        warnings.append("telemetry repo unavailable; suitability-only (no live evidence).")
    if t_info["evidence_events"] == 0:
        warnings.append("no current resolved telemetry for this species; "
                        "behavioral priors + environment carry the layer.")

    data_time = http_date_to_det(info["http_last_modified"]) or "unknown"
    meta = base_metadata(
        PRODUCT, cfg["title"], cfg["freshness_label"],
        "NOAA/GLERL CoastWatch GLSEA SST + USGS/GLATOS acoustic-telemetry evidence",
        GLSEA_URL,
        "live modeled distribution / habitat likelihood index (0..1; NOT fish counts)",
        data_time_utc=data_time,
        source_last_modified_utc=data_time,
        units="modeled index 0..1",
        source_resolution="~1.8 km GLSEA SST binned to shared canvas; telemetry evidence splatted",
        color_min=0.0, color_max=1.0, color_units="modeled distribution index",
        missing_data_treatment=("land and missing SST fully transparent; "
                                "near-zero index suppressed to transparent (no fake coverage)."))
    meta["source_id"] = source_id
    meta["source_version"] = source_token(source_id)
    meta["legend_size"] = list(G.LEGEND_SIZE)
    meta["telemetry"] = {**t_info, "repo_available": tele is not None}
    meta["seasonal"] = m_info
    meta["diel"] = d_info
    meta["confidence"] = round(conf, 3)
    meta["component_means"] = {k: round(float(v[water].mean()), 4) for k, v in comps.items()}
    meta["warnings"] = warnings
    meta["stats"] = {"mean_index": round(float(final[water].mean()), 4),
                     "p95_index": round(float(np.percentile(final[water], 95)), 4),
                     "max_index": round(float(final.max()), 4)}
    scale_html = (f"Modeled distribution / habitat index: LOW <b>0</b> (transparent) "
                  f"&rarr; HIGH <b>1</b>. Current mean: "
                  f"<b>{meta['stats']['mean_index']}</b>, p95: "
                  f"<b>{meta['stats']['p95_index']}</b>, confidence/support: "
                  f"<b>{meta['confidence']}</b>. Telemetry events: "
                  f"<b>{t_info['evidence_events']}</b> (+{t_info['prior_receivers']} "
                  f"behavioral-prior receivers). Transparency marks low confidence "
                  f"or water outside the modeled domain.")
    meta["legend_scale_html"] = scale_html

    stage = stage_dir(PRODUCT)
    stage_prod = os.path.join(stage, "site", PRODUCT)
    save_png(rgba, os.path.join(stage_prod, "current.png"))
    G.draw_fish_legend(os.path.join(stage_prod, "legend.png"), species,
                       cfg["label"],
                       [f"Model time {now.strftime('%Y-%m-%d %H:%M UTC')}  |  "
                        f"SST: {data_time[:60]}",
                        f"Telemetry: {str(t_info['newest_evidence'] or 'behavioral priors only')[:60]}  |  "
                        f"Confidence {round(conf, 3)}"],
                       [f"Events: {t_info['evidence_events']}  Prior receivers: "
                        f"{t_info['prior_receivers']}  p95: {meta['stats']['p95_index']}",
                        f"Windows: {', '.join(m_info['active_windows']) or 'none'}  |  "
                        f"Diel: {d_info['phase']}"],
                       G.MODEL_VERSION)
    write_metadata(stage_prod, meta)
    write_kmls(cfg, PRODUCT, meta, meta["source_version"], stage)
    promoted = promote_stage(PRODUCT)
    write_state(PRODUCT, {"source_id": source_id,
                          "render_version": RENDER_VERSION,
                          "processing_time_utc": meta["processing_time_utc"]})
    print(f"[{PRODUCT}] UPDATED OK ({len(promoted)} files promoted).")
    return 0


def live_description_html(cfg, meta, token, legend_img_html):
    """Static folder description: no retrieval/update timestamps (they would
    go stale inside Google Earth). Timestamps live in metadata.json and the
    versioned legend image instead."""
    base = pages_base_url()
    return (
        f"<h2>{cfg['title']}</h2>"
        f"<p>{G.DISCLAIMER}</p>"
        f"{legend_img_html}"
        f"<p><b>All nine species key:</b><br>"
        f"<img src=\"{base}/gamefish/legend_key.png?v={G.MODEL_VERSION}\" "
        f"width=\"600\" alt=\"combined nine-species gradient key\"><br>"
        f"Each species keeps its own gradient colors and low-to-high meaning; "
        f"this key maps species to colors to intensity.</p>"
        f"<p><b>Method:</b> telemetry evidence &times; seasonal migration &times; "
        f"thermal suitability &times; diel behavior &times; habitat &times; movement "
        f"corridors (weighted mean; confidence-gated transparency). Telemetry is "
        f"behavioral evidence only &mdash; never abundance; it decays spatially and "
        f"temporally, and unresolved tags stay unresolved.</p>"
        f"<p><b>Reading the gradient:</b> LOW (transparent) &rarr; HIGH (saturated) "
        f"modeled distribution / habitat index. Transparency marks low confidence "
        f"or water outside the modeled domain; a strong color with weak support "
        f"fades toward transparent.</p>"
        f"<p><b>Sources:</b> NOAA/GLERL CoastWatch GLSEA lake surface temperature "
        f"(<a href=\"https://apps.glerl.noaa.gov/coastwatch/webdata/glsea/cur/glsea_cur.asc\">"
        f"GLSEA current analysis</a>) and acoustic-telemetry evidence from "
        f"<a href=\"https://github.com/wyattleathorn7/great-lakes-live-fish-telemetry\">"
        f"great-lakes-live-fish-telemetry</a> (USGS real-time receivers and GLATOS "
        f"deployments; read as input, never modified).</p>"
        f"<p><b>Limitations:</b> surface temperature only (no depth resolution); "
        f"shore-proximity habitat proxy with no bathymetry, substrate, or vegetation "
        f"layers; telemetry decay constants are documented modeling assumptions.</p>"
        f"<p>Transparent outside valid water data so existing project layers stay "
        f"visible. Turn on/off independently of other layers.</p>"
    )


def entry_description_html_static(cfg):
    """Static entry description: explains the product and the auto-refresh;
    carries no timestamps or version tokens (entry files stay byte-stable)."""
    base = pages_base_url()
    return (
        f"<h2>{cfg['title']}</h2>"
        f"<p>{G.DISCLAIMER}</p>"
        f"<p>This entry auto-refreshes from the live overlay "
        f"(telemetry &times; seasonal &times; thermal &times; diel &times; habitat "
        f"&times; corridors, weighted mean). "
        f"Add this file once; new model cycles appear automatically.</p>"
        f"<p><b>Sources:</b> NOAA/GLERL CoastWatch GLSEA and acoustic-telemetry "
        f"evidence (USGS real-time + GLATOS deployments).</p>"
        f"<p><b>All nine species key:</b><br>"
        f"<img src=\"{base}/gamefish/legend_key.png\" "
        f"width=\"600\" alt=\"combined nine-species gradient key\"></p>"
    )


def pages_base_url():
    from build_kml import pages_base
    return pages_base()


def write_kmls(cfg, PRODUCT, meta, token, stage=None):
    from geospatial_utils import SITE_DIR, KML_DIR
    block = legend_block(f"{PRODUCT}/legend.png", token, meta["legend_scale_html"])
    desc = live_description_html(cfg, meta, token, block)
    live_name = cfg["kml_filename"]
    if stage is None:
        build_kml(PRODUCT, live_name, cfg["overlay_name"],
                  f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png", desc,
                  cfg["refresh_interval_seconds"], token)
        build_entry_kml(PRODUCT, live_name, cfg["overlay_name"],
                        entry_description_html_static(cfg),
                        cfg["refresh_interval_seconds"])
    else:
        outs = live_out_dirs(stage, live_name)
        kml_text = build_kml(PRODUCT, live_name, cfg["overlay_name"],
                             f"{PRODUCT}/current.png", f"{PRODUCT}/legend.png",
                             desc, cfg["refresh_interval_seconds"], token,
                             out_dirs=outs["live"])
        assert_no_vector_geometry(kml_text)
        build_entry_kml(PRODUCT, live_name, cfg["overlay_name"],
                        entry_description_html_static(cfg),
                        cfg["refresh_interval_seconds"], out_dirs=outs["entry"])


def rewrite_kmls_from_metadata(cfg, PRODUCT):
    with open(os.path.join(SITE_DIR, PRODUCT, "metadata.json")) as f:
        meta = json.load(f)
    write_kmls(cfg, PRODUCT, meta, source_token(meta.get("source_version", "")))


if __name__ == "__main__":
    sys.exit(main())
