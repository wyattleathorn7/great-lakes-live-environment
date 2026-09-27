"""Generate the independent Google Earth KML overlays.

Two-file live-update architecture (source-aware, version-busted):

* ENTRY file ``kml/<Name>.kml`` (repo) mirrored at ``site/kml/<Name>.kml``
  (Pages): what users add to Google Earth. Stable URL, stable content.
  Contains exactly ONE ``NetworkLink`` pointing at the live file with
  ``refreshMode=onInterval`` + per-product ``refreshInterval``. No
  GroundOverlay, no geometry. Because the entry never carries a version,
  already-added projects keep polling the current live file forever.

* LIVE file ``site/kml/live/<Name>.kml`` (Pages, regenerated on every run
  from committed metadata): contains exactly ONE ``GroundOverlay`` whose
  ``Icon`` href is ``<product>/current.png?v=<SOURCE_VERSION>`` with
  ``refreshMode=onInterval``. ``SOURCE_VERSION`` changes ONLY when the
  underlying source observation/cycle changes (never on re-runs), so a new
  raster can neither hide behind a CDN-cached old URL nor trigger a
  pointless Google Earth refetch. No NetworkLink inside the live file
  (no self-reference: Google Earth Web loop-protects self-links).

Update chain::

    source changes -> workflow detects new source id -> PNG rebuilt ->
    live KML rewritten with new ?v= -> entry NetworkLink poll discovers
    the new live KML -> Google Earth requests the new versioned PNG.

RASTER ONLY in both files: zero LineString / Polygon / Placemark /
Point / ScreenOverlay elements (asserted). Legends travel inside the
live Document description as HTML (legend PNG <img> + scale text).
"""

import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

from geospatial_utils import (KML_DIR, PROD_BASE_URL, SITE_DIR, load_bounds,
                              source_token)

KML_NS = "http://www.opengis.net/kml/2.2"
ET.register_namespace("", KML_NS)


def pages_base():
    return os.environ.get("PAGES_BASE_URL", PROD_BASE_URL).rstrip("/")


def _q(tag, text=None):
    el = ET.Element(f"{{{KML_NS}}}{tag}")
    if text is not None:
        el.text = text
    return el


def legend_block(legend_path, cache_token, scale_html):
    """HTML legend for the Document description (GE-compatible)."""
    base = pages_base()
    return (
        f"<p><b>Legend</b><br>"
        f"<img src=\"{base}/{legend_path}?v={cache_token}\" width=\"600\" "
        f"alt=\"legend\"><br>{scale_html}</p>"
    )


def fetch_header_html(meta, refresh_interval):
    """Two-line Detroit-time header for the very top of live descriptions.

    ``Time fetched`` = when this live file's data was fetched/processed
    (Detroit wall clock, compact AM/PM). ``Next update`` = fetched +
    this layer's own KML refreshInterval, i.e. the exact time Google
    Earth next re-checks the layer per the refreshMode=onInterval
    contract in this same file. That re-check moment is deterministic
    (hence a trustworthy prediction); whether it carries NEW imagery
    depends on the source having published since (often it hasn't, and
    the current image simply re-confirms).

    Derived from committed metadata (processing_time_utc), never wall-now,
    so skip-path KML rewrites stay byte-identical while the source is
    unchanged (no commit churn, no stale-while-claiming-fresh times).
    """
    from datetime import datetime, timedelta
    raw = (meta or {}).get("processing_time_utc", "")
    dt = None
    try:
        dt = datetime.strptime(raw.strip(), "%Y-%m-%d %I:%M %p %Z")
    except (ValueError, AttributeError):
        dt = None
    if dt is None:
        try:
            from geospatial_utils import detroit_tz
            dt = datetime.now(detroit_tz()).replace(tzinfo=None)
        except Exception:
            from datetime import timezone
            dt = datetime.now(timezone.utc)
    tzname = (raw.strip().rsplit(" ", 1)[-1] if raw else "") or "ET"

    def _compact(d):
        return f"{d.hour % 12 or 12}:{d.minute:02d}{d.strftime('%p')}"

    try:
        interval = int(refresh_interval)
    except (TypeError, ValueError):
        interval = 3600
    nxt = dt + timedelta(seconds=interval)
    return (f"<p><b>Time fetched:</b> {_compact(dt)} {tzname}<br/>"
            f"<b>Next update:</b> {_compact(nxt)} {tzname}</p>")


def _serialize(doc, descriptions):
    """Pretty-print; inject CDATA descriptions in document order."""
    xml = minidom.parseString(ET.tostring(doc)).toprettyxml(
        indent="  ", encoding="utf-8").decode("utf-8")
    for html in descriptions:
        cdata = f"<description><![CDATA[{html}]]></description>"
        xml = xml.replace("<description/>", cdata, 1)
    return xml


def _write(xml, out_dirs):
    for out in out_dirs:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(xml)


def build_kml(product, kml_filename, overlay_name, png_path, legend_path,
              description_html, refresh_interval, cache_token, out_dirs=None,
              folder=None, meta=None):
    """Write the LIVE overlay file (one GroundOverlay, versioned PNG href).

    ``cache_token`` MUST be the deterministic source version
    (``source_token(...)`` of the source observation/cycle id), never a
    processing timestamp or random value. ``out_dirs`` selects where the
    live file goes; the default is the deployed live path. When ``meta``
    is given, the Time-fetched/Next-update header is prepended to the
    main description (the Folder description when present, else the
    Document description).
    """
    bounds = load_bounds()
    base = pages_base()
    png_url = f"{base}/{png_path}?v={cache_token}"
    header = fetch_header_html(meta, refresh_interval) if meta else ""

    doc = _q("kml")
    document = _q("Document")
    doc.append(document)

    document.append(_q("name", overlay_name))
    desc = _q("description")
    desc.text = None
    document.append(desc)

    ground = _q("GroundOverlay")
    ground.append(_q("name", overlay_name))
    ground.append(_q("drawOrder", "10"))
    icon = _q("Icon")
    icon.append(_q("href", png_url))
    icon.append(_q("refreshMode", "onInterval"))
    icon.append(_q("refreshInterval", str(refresh_interval)))
    ground.append(icon)
    box = _q("LatLonBox")
    box.append(_q("north", str(bounds["lat_max"])))
    box.append(_q("south", str(bounds["lat_min"])))
    box.append(_q("east", str(bounds["lon_max"])))
    box.append(_q("west", str(bounds["lon_min"])))
    ground.append(box)
    descriptions = [header + description_html]
    if folder is not None:
        folder_el = _q("Folder")
        folder_el.append(_q("name", folder[0]))
        _fdesc = _q("description")
        _fdesc.text = None
        folder_el.append(_fdesc)
        folder_el.append(ground)
        document.append(folder_el)
        descriptions.append(header + strip_old_stamps(folder[1]))
    else:
        document.append(ground)

    xml = _serialize(doc, descriptions)
    if out_dirs is None:
        out_dirs = [os.path.join(SITE_DIR, "kml", "live", kml_filename)]
    _write(xml, out_dirs)
    assert_no_vector_geometry(xml)
    assert "<NetworkLink" not in xml, "live file must not self-link"
    return xml


def build_entry_kml(product, kml_filename, overlay_name, entry_description_html,
                    refresh_interval, out_dirs=None):
    """Write the stable ENTRY file (one self-refreshing NetworkLink).

    Users add this file to Google Earth once. It never carries a version
    token, so it stays byte-stable across rebuilds (no commit churn) while
    its NetworkLink poll discovers each newly published live file.
    """
    base = pages_base()
    live_url = f"{base}/kml/live/{kml_filename}"

    doc = _q("kml")
    document = _q("Document")
    doc.append(document)
    document.append(_q("name", overlay_name))
    desc = _q("description")
    desc.text = None
    document.append(desc)

    link = _q("NetworkLink")
    link.append(_q("name", overlay_name + " — auto-refresh"))
    url = _q("Link")
    url.append(_q("href", live_url))
    url.append(_q("refreshMode", "onInterval"))
    url.append(_q("refreshInterval", str(refresh_interval)))
    link.append(url)
    document.append(link)

    xml = _serialize(doc, [entry_description_html])
    if out_dirs is None:
        out_dirs = [os.path.join(KML_DIR, kml_filename),
                    os.path.join(SITE_DIR, "kml", kml_filename)]
    _write(xml, out_dirs)
    assert_no_vector_geometry(xml)
    assert "<GroundOverlay" not in xml, "entry file must not carry imagery"
    return xml


def entry_description_html(title, meta, note):
    # Entry files are what users add first, so they carry the gradient key
    # image too (unversioned legend URL: entries must stay byte-stable and
    # version-free, and legends are fixed-scale so they rarely change).
    legend_html = ""
    try:
        product = meta.get("product")
        scale_html = meta.get("legend_scale_html")
        if product and scale_html:
            base = pages_base()
            legend_html = (
                f"<p><b>Gradient key</b><br>"
                f"<img src=\"{base}/{product}/legend.png\" width=\"600\" "
                f"alt=\"gradient key\"><br>{scale_html}</p>"
            )
    except Exception:
        legend_html = ""
    return (
        f"<h2>{title}</h2>"
        f"{legend_html}"
        f"<p>This entry auto-refreshes from the live overlay "
        f"(source: {meta.get('noaa_source')}).</p>"
        f"<p>{note} Add this file once; new source cycles appear "
        f"automatically.</p>"
        f"<p><i>Not endorsed by NOAA.</i></p>"
    )


def description_html(title, meta, kml_self_hint, legend_html=""):
    return (
        f"<h2>{title}</h2>"
        f"<p><b>Status:</b> {meta.get('freshness')}</p>"
        f"{legend_html}"
        f"<p><b>Source:</b> {meta.get('noaa_source')}<br/>"
        f"<a href=\"{meta.get('source_url')}\">{meta.get('source_url')}</a></p>"
        f"<p>Transparent outside valid water data so existing project layers "
        f"(shipwrecks, lighthouses, harbors, parks) stay visible. "
        f"{kml_self_hint}</p>"
        f"<p><i>Not endorsed by NOAA.</i></p>"
    )


import re as _re

_STAMP_RE = _re.compile(
    r"<b>(Data time|Source updated|Source version|Processed):</b>"
    r"[^<]*(?:<br\s*/>)?")


def strip_old_stamps(html):
    """Remove legacy timestamp lines from a stored folder description.

    The Time-fetched/Next-update header (prepended separately) is now the
    only clock in live descriptions; older Data-time/Processed/etc. lines
    would contradict it once cached. Values never contain tags, so the
    label-anchored match is exact. Cleans up dangling breaks.
    """
    if not html:
        return html
    out = _STAMP_RE.sub("", html)
    out = out.replace("<p><br/>", "<p>").replace("<p> <br/>", "<p>")
    return out


def live_out_dirs(stage, kml_filename):
    """Stage-relative outputs for a full build: entry pair + live file."""
    return {
        "entry": [os.path.join(stage, "kml", kml_filename),
                  os.path.join(stage, "site", "kml", kml_filename)],
        "live": [os.path.join(stage, "site", "kml", "live", kml_filename)],
    }


def refresh_kml_base_url(product, kml_filename, overlay_name, title,
                         note, refresh_interval):
    """Rewrite entry + live KMLs from committed metadata (skip path).

    Legend HTML is reproduced from metadata, so the refresh path needs no
    source data. Deterministic: byte-identical when the source is unchanged.
    """
    with open(os.path.join(SITE_DIR, product, "metadata.json")) as f:
        meta = json.load(f)
    if not meta.get("source_version"):
        # One-time backfill: adopt the state-tracked source id so the
        # versioned URL matches the actually-published raster.
        try:
            from geospatial_utils import read_state, write_metadata
            sid = read_state(product).get("source_id")
            if sid:
                meta["source_id"] = sid
                meta["source_version"] = source_token(sid)
                write_metadata(os.path.join(SITE_DIR, product), meta)
        except Exception:
            pass
    version = source_token(meta.get("source_version") or
                           meta.get("processing_time_utc", ""))
    block = legend_block(f"{product}/legend.png", version,
                         meta.get("legend_scale_html", ""))
    folder = meta.get("folder_html")
    build_kml(product, kml_filename, overlay_name,
              f"{product}/current.png", f"{product}/legend.png",
              description_html(title, meta, note, block),
              refresh_interval, version,
              folder=(title, folder) if folder else None, meta=meta)
    build_entry_kml(product, kml_filename, overlay_name,
                    entry_description_html(title, meta, note),
                    refresh_interval)
    return version


def assert_no_vector_geometry(kml_text):
    for bad in ("<LineString", "<Polygon", "<Placemark", "<Point",
                "<ScreenOverlay"):
        assert bad not in kml_text, f"forbidden KML element present: {bad}"
