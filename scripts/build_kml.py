"""Generate the three independent Google Earth KML overlays.

RASTER ONLY: each KML contains exactly one GroundOverlay (the current.png
raster) + one ScreenOverlay (the legend PNG) + one self-refresh NetworkLink.
ZERO LineString / Polygon / Placemark elements are emitted.
Cache-busting: the PNG hrefs carry ?v=<processing-timestamp-token> which is
regenerated on every successful product run. Filenames/URLs stay stable.
"""

import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

from geospatial_utils import KML_DIR, SITE_DIR, load_bounds

KML_NS = "http://www.opengis.net/kml/2.2"
ET.register_namespace("", KML_NS)


def pages_base():
    return os.environ.get(
        "PAGES_BASE_URL", "https://REPLACE-GITHUB-USER.github.io/REPLACE-REPO")


def _q(tag, text=None):
    el = ET.Element(f"{{{KML_NS}}}{tag}")
    if text is not None:
        el.text = text
    return el


def build_kml(product, kml_filename, overlay_name, png_path, legend_path,
              description_html, refresh_interval, cache_token):
    bounds = load_bounds()
    base = pages_base().rstrip("/")
    png_url = f"{base}/{png_path}?v={cache_token}"
    legend_url = f"{base}/{legend_path}?v={cache_token}"
    self_url = f"{base}/kml/{kml_filename}"

    doc = _q("kml")
    document = _q("Document")
    doc.append(document)

    name = _q("name", overlay_name)
    document.append(name)
    desc = _q("description")
    desc.text = None
    document.append(desc)
    # CDATA description (set after serialization to keep markup intact)
    cdata_holder = desc

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

    screen = _q("ScreenOverlay")
    screen.append(_q("name", overlay_name + " — legend"))
    sicon = _q("Icon")
    sicon.append(_q("href", legend_url))
    screen.append(sicon)
    for tag, x, y, xunits, yunits in (
            ("overlayXY", "0", "1", "fraction", "fraction"),
            ("screenXY", "0.01", "0.08", "fraction", "fraction"),
            ("size", "0.32", "0", "fraction", "fraction")):
        el = _q(tag)
        el.set("x", x)
        el.set("y", y)
        el.set("xunits", xunits)
        el.set("yunits", yunits)
        screen.append(el)
    document.append(screen)

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

    for out in (os.path.join(KML_DIR, kml_filename),
                os.path.join(SITE_DIR, "kml", kml_filename)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(xml)
    return xml


def description_html(title, meta, kml_self_hint):
    return (
        f"<h2>{title}</h2>"
        f"<p><b>Data time:</b> {meta.get('data_time_utc')}<br/>"
        f"<b>Source updated:</b> {meta.get('source_last_modified_utc')}<br/>"
        f"<b>Processed:</b> {meta.get('processing_time_utc')}<br/>"
        f"<b>Status:</b> {meta.get('freshness')}</p>"
        f"<p><b>Source:</b> {meta.get('noaa_source')}<br/>"
        f"<a href=\"{meta.get('source_url')}\">{meta.get('source_url')}</a></p>"
        f"<p>Transparent outside valid water data so existing project layers "
        f"(shipwrecks, lighthouses, harbors, parks) stay visible. "
        f"{kml_self_hint}</p>"
        f"<p><i>Not endorsed by NOAA.</i></p>"
    )


def assert_no_vector_geometry(kml_text):
    for bad in ("<LineString", "<Polygon", "<Placemark", "<Point", "<Model"):
        assert bad not in kml_text, f"forbidden KML geometry present: {bad}"
