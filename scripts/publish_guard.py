#!/usr/bin/env python3
"""Publish staleness guard: block a Pages deploy that would move a product
backwards in time.

Two deploy lanes write the same Pages site -- the every-5-minute "Fast
live refresh" (deploys directly, never commits) and the hourly/half-
hourly main workflow (commits + deploys the full site). Merge-lane
content can therefore be older than what Pages is currently serving;
publishing it unguarded regresses the live layers (observed 2026-11-01:
the fast cycle served cloud rasters ~7 GOES cycles stale while newer
cycles existed).

For each product directory this script:
  1. reads the CANDIDATE metadata (post-merge site tree), and
  2. fetches the LIVE metadata from Pages,
and blocks (exit 1) when the candidate is staler than live by more than
--threshold-min minutes. Identical source_version tokens pass (same
content); directories not deployed live yet pass; unparseable times
pass with a warning (never gate on data we cannot read).

Exit codes: 0 = safe, 1 = stale / could not verify (block), 2 = usage
error. HTTP fetch failures are exit 1: failing closed is the point --
CI retries on the next cycle.
"""

import argparse
import datetime as dt
import json
import re
import sys
import urllib.error
import urllib.request

UA = {"User-Agent": "great-lakes-live-environment publish-guard/1"}

# Dir families the fast-refresh lane owns (cloud + precipitation, all
# coverage variants). The guard compares exactly what the merge lane
# could regress: these eight directories.
FAST_FAMILIES = [f"cloud_cover_na{s}" for s in ("", "_basin", "_continental",
                                               "_global")] + \
                [f"precipitation_na{s}" for s in ("", "_basin", "_continental",
                                                  "_global")]


def _ts(meta):
    """Best-effort freshness extraction from a metadata dict.

    Prefers observation_time_utc ("2026-11-02 00:40 UTC"); falls back to
    timestamps embedded in source_version / source_id tokens (formats
    "YYYYMMDD-HHMM" and "YYYYMMDDTHHMM"). Returns a datetime or None.
    """
    o = meta.get("observation_time_utc")
    if isinstance(o, str):
        m = re.match(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})", o)
        if m:
            return dt.datetime.strptime(m.group(1) + " " + m.group(2),
                                        "%Y-%m-%d %H:%M")
    for k in ("source_version", "source_id"):
        v = meta.get(k)
        if not isinstance(v, str):
            continue
        m = (re.search(r"(\d{8})T(\d{4})", v)
             or re.search(r"(\d{8})-(\d{4})", v))
        if m:
            return dt.datetime.strptime(m.group(1) + " " + m.group(2),
                                        "%Y%m%d %H%M")
    return None


def _fetch_json(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", required=True,
                   help="Pages base URL, e.g. https://owner.github.io/repo")
    p.add_argument("--site", default="site", help="candidate site tree")
    p.add_argument("--threshold-min", type=int, default=20,
                   help="block if candidate is staler than live by more "
                        "than this many minutes (default 20)")
    p.add_argument("--product", action="append", default=None,
                   help="product directory under site/ (repeatable; "
                        "defaults to the fast-refresh families)")
    p.add_argument("--sync-live-fast", action="store_true",
                   help="preserve newer live cloud/precipitation assets before publishing")
    a = p.parse_args()
    products = a.product or FAST_FAMILIES

    base = a.base.rstrip("/") + "/"
    if a.sync_live_fast:
        import os
        import shutil
        # Restore the currently served fast products into this full-site
        # candidate. This prevents daily/hourly publishes from rolling back
        # the independent 5-minute cloud and precipitation lane.
        for product in FAST_FAMILIES:
            local = os.path.join(a.site, product)
            try:
                with open(os.path.join(local, "metadata.json")) as f:
                    candidate = json.load(f)
            except (OSError, ValueError):
                candidate = {}
            live = _fetch_json(f"{base}{product}/metadata.json")
            if not live:
                continue
            live_ts, candidate_ts = _ts(live), _ts(candidate)
            same = live.get("source_version") and live.get("source_version") == candidate.get("source_version")
            if same or (live_ts is not None and candidate_ts is not None and live_ts <= candidate_ts):
                continue
            if live_ts is None:
                raise RuntimeError(f"[{product}] cannot determine live timestamp; refusing unsafe publish")
            os.makedirs(local, exist_ok=True)
            for filename in ("metadata.json", "current.png", "legend.png", "provenance.png"):
                data = _fetch_bytes(f"{base}{product}/{filename}")
                target = os.path.join(local, filename)
                if data is None:
                    if filename in ("metadata.json", "current.png"):
                        raise RuntimeError(f"[{product}] missing required live file {filename}")
                    if os.path.exists(target):
                        os.remove(target)
                else:
                    tmp = target + ".sync-part"
                    with open(tmp, "wb") as f:
                        f.write(data)
                    os.replace(tmp, target)
            print(f"[{product}] restored newer live fast-lane files ({live_ts:%Y-%m-%d %H:%M} UTC)")
    unverifiable = []
    for product in products:
        cand = f"{a.site}/{product}/metadata.json"
        try:
            with open(cand, "r") as f:
                c_meta = json.load(f)
        except FileNotFoundError:
            continue   # product skipped this run: merge kept whatever
        except Exception as e:
            print(f"[{product}] candidate metadata unreadable: {e}")
            unverifiable.append(product)
            continue
        try:
            live_meta = _fetch_json(f"{base}{product}/metadata.json")
        except Exception as e:
            print(f"[{product}] live metadata fetch failed: "
                  f"{type(e).__name__}: {e}")
            unverifiable.append(product)
            continue
        if not live_meta:
            continue   # first deploy of this directory
        if (live_meta.get("source_version")
                and live_meta.get("source_version")
                == c_meta.get("source_version")):
            continue   # identical content
        live_ts, cand_ts = _ts(live_meta), _ts(c_meta)
        if live_ts is None or cand_ts is None:
            print(f"[{product}] WARN: could not parse freshness "
                  f"(candidate={cand_ts}, live={live_ts}); allowing")
            continue
        delta_min = (live_ts - cand_ts).total_seconds() / 60.0
        if delta_min > a.threshold_min:
            stale.append((product, cand_ts, live_ts, delta_min))
        else:
            print(f"[{product}] ok: candidate {cand_ts:%Y-%m-%d %H:%M} "
                  f"vs live {live_ts:%Y-%m-%d %H:%M}")

    for product, cand, live, dmin in stale:
        print(f"[{product}] STALE: candidate {cand:%Y-%m-%d %H:%M} UTC is "
              f"{dmin:.0f} min behind live {live:%Y-%m-%d %H:%M} UTC")
    if unverifiable:
        for p2 in unverifiable:
            print(f"[{p2}] could not verify; blocking (fail closed)")
        return 1
    if stale:
        print(f"BLOCKED publish: {len(stale)} product(s) would regress; "
              f"deploy nothing, retry on the next cycle.")
        return 1
    print("publish guard: safe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
