"""Generate the independent Google Earth KML overlays.

RASTER ONLY: each KML contains exactly one GroundOverlay (the current.png
raster) + one self-refresh NetworkLink. ZERO LineString / Polygon /
Placemark elements are emitted.

Legends: the Google Earth client used for testing rejects ScreenOverlay
("Unsupported element"), so legends are delivered inside the Document
description as HTML (legend PNG <img> + explicit scale text). The
standalone legend.png files remain published for the web index and for
pixel-exact validation. No ScreenOverlay element is ever emitted.
Cache-busting: the PNG hrefs carry ?v=<processing-timestamp-token> which is
regenerated on every successful product run. Filenames/URLs stay stable.
"""

import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

from geospatial_utils import KML_DIR, PROD_BASE_URL, SITE_DIR, load_bounds

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


def build_kml(product, kml_filename, overlay_name, png_path, legend_path,
              description_html, refresh_interval, cache_token, out_dirs=None,
              tiles=None):
    """tiles = [(rel_png_path, tile_bounds_dict, min_lod_pixels)] or None.

    Tiles are LOD GroundOverlays with Regions: Google Earth fetches a tile
    only when its region is in view and large enough on screen, giving a
    crisp shoreline at every zoom while the overview stays the always-on
    base. Builders include tiles ONLY when they were generated in that run
    (skip/fail paths stay overview-only), so KML tile refs always resolve.
    """
    bounds = load_bounds()
    base = pages_base()
    png_url = f"{base}/{png_path}?v={cache_token}"
    self_url = f"{base}/kml/{kml_filename}"

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
    icon.append(_q("refreshMode", "onChange"))
    ground.append(icon)
    box = _q("LatLonBox")
    box.append(_q("north", str(bounds["lat_max"])))
    box.append(_q("south", str(bounds["lat_min"])))
    box.append(_q("east", str(bounds["lon_max"])))
    box.append(_q("west", str(bounds["lon_min"])))
    ground.append(box)
    document.append(ground)

    for rel_path, tb, min_lod in tiles or []:
        tile = _q("GroundOverlay")
        tile.append(_q("name", f"{overlay_name} — detail tile"))
        tile.append(_q("drawOrder", "20"))
        ticon = _q("Icon")
        ticon.append(_q("href", f"{base}/{rel_path}?v={cache_token}"))
        ticon.append(_q("refreshMode", "onChange"))
        tile.append(ticon)
        tbox = _q("LatLonBox")
        tbox.append(_q("north", str(tb["lat_max"])))
        tbox.append(_q("south", str(tb["lat_min"])))
        tbox.append(_q("east", str(tb["lon_max"])))
        tbox.append(_q("west", str(tb["lon_min"])))
        tile.append(tbox)
        region = _q("Region")
        llab = _q("LatLonAltBox")
        llab.append(_q("north", str(tb["lat_max"])))
        llab.append(_q("south", str(tb["lat_min"])))
        llab.append(_q("east", str(tb["lon_max"])))
        llab.append(_q("west", str(tb["lon_min"])))
        region.append(llab)
        lod = _q("Lod")
        lod.append(_q("minLodPixels", str(min_lod)))
        lod.append(_q("maxLodPixels", "-1"))
        region.append(lod)
        tile.append(region)
        document.append(tile)

    link = _q("NetworkLink")
    link.append(_q("name", overlay_name + " — auto-refresh"))
    url = _q("Link")
    url.append(_q("href", self_url))
    url.append(_q("refreshMode", "onInterval"))
    url.append(_q("refreshInterval", str(refresh_interval)))
    link.append(url)
    document.append(link)

    xml = minidom.parseString(ET.tostring(doc)).toprettyxml(
        indent="  ", encoding="utf-8").decode("utf-8")
    # inject CDATA description (ElementTree would escape the HTML)
    cdata = f"<description><![CDATA[{description_html}]]></description>"
    xml = xml.replace("<description/>", cdata, 1)

    if out_dirs is None:
        out_dirs = [os.path.join(KML_DIR, kml_filename),
                    os.path.join(SITE_DIR, "kml", kml_filename)]
    for out in out_dirs:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(xml)
    return xml


def description_html(title, meta, kml_self_hint, legend_html=""):
    return (
        f"<h2>{title}</h2>"
        f"<p><b>Data time:</b> {meta.get('data_time_utc')}<br/>"
        f"<b>Source updated:</b> {meta.get('source_last_modified_utc')}<br/>"
        f"<b>Processed:</b> {meta.get('processing_time_utc')}<br/>"
        f"<b>Status:</b> {meta.get('freshness')}</p>"
        f"{legend_html}"
        f"<p><b>Source:</b> {meta.get('noaa_source')}<br/>"
        f"<a href=\"{meta.get('source_url')}\">{meta.get('source_url')}</a></p>"
        f"<p>Transparent outside valid water data so existing project layers "
        f"(shipwrecks, lighthouses, harbors, parks) stay visible. "
        f"{kml_self_hint}</p>"
        f"<p><i>Not endorsed by NOAA.</i></p>"
    )


def refresh_kml_base_url(product, kml_filename, overlay_name, title,
                         note, refresh_interval):
    """Rewrite existing KMLs with the current PAGES_BASE_URL.

    Legend HTML is reproduced from metadata (legend_scale_html), so the
    refresh path needs no source data.
    """
    with open(os.path.join(SITE_DIR, product, "metadata.json")) as f:
        meta = json.load(f)
    token = meta["processing_time_utc"].replace(" ", "_").replace(":", "")
    block = legend_block(f"{product}/legend.png", token,
                         meta.get("legend_scale_html", ""))
    kml_text = build_kml(product, kml_filename, overlay_name,
                         f"{product}/current.png", f"{product}/legend.png",
                         description_html(title, meta, note, block),
                         refresh_interval, token)
    assert_no_vector_geometry(kml_text)
    return kml_text


def assert_no_vector_geometry(kml_text):
    for bad in ("<LineString", "<Polygon", "<Placemark", "<Point",
                "<ScreenOverlay"):
        assert bad not in kml_text, f"forbidden KML element present: {bad}"
