import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import FluorophoreDatabase
from chisurf.plugins.spectra_downloader.download.threed_optix import (
    scrape_product_detail,
    download_threed_optix_to_db,
    parse_wl,
)

# Sample HTML snippets matching 3DOptix layout
MOCK_INDEX_HTML = """
<html>
<body>
<table>
  <tbody>
    <tr>
      <td><a href="/catalog/optics/filter/optosigma/RSF-25C-325RU">RSF-25C-325RU 25 mm Dia. Mounted Raman Longpass Edge Filter; t=3.5mm 325nm</a></td>
      <td>Optics</td>
      <td>Filter</td>
      <td>Longpass Filter</td>
      <td>OptoSigma</td>
      <td>UVFS</td>
      <td>Circular</td>
    </tr>
  </tbody>
</table>
</body>
</html>
"""

MOCK_DETAIL_HTML = """
<html>
<body>
<div class="Product-module__info">
  <h2 itemprop="name">RSF-25C-325RU 25 mm Dia. Mounted Raman Longpass Edge Filter; t=3.5mm 325nm</h2>
  <dl>
    <div><dt>Type</dt><dd>Filter</dd></div>
    <div><dt>Subtype</dt><dd>Longpass Filter</dd></div>
    <div><dt>Brand</dt><dd itemprop="brand">OptoSigma</dd></div>
    <div><dt>Product webpage</dt><dd><a href="https://www.optosigma.com/us_en/raman.html">OptoSigma</a></dd></div>
  </dl>
</div>
<div class="Product-module__parameters">
  <h2>Properties</h2>
  <dl>
    <div><dt>Shape</dt><dd>Circular</dd></div>
    <div><dt>Diameter</dt><dd>25 mm</dd></div>
    <div><dt>Cut On Frequency</dt><dd>325 nm</dd></div>
  </dl>
</div>
<div class="Product-module__parameters">
  <h2>Material </h2>
  <dl>
    <div><dt>Name</dt><dd>UVFS</dd></div>
  </dl>
</div>
<div class="Product-module__description">
  <p>The filter's coating is another critical aspect. Both the front and back surfaces are coated with Opt RSF-325RU, a coating that enhances performance.</p>
</div>
</body>
</html>
"""

@pytest.fixture
def db():
    return FluorophoreDatabase(":memory:")

def test_parse_wl():
    assert parse_wl("325 nm") == 325.0
    assert parse_wl("325.5nm") == 325.5
    assert parse_wl("1200") == 1200.0
    assert parse_wl("") is None
    assert parse_wl(None) is None




def test_scrape_product_detail():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = MOCK_DETAIL_HTML
        mock_get.return_value = mock_response

        res = scrape_product_detail("https://example.com/item")
        assert res["external_url"] == "https://www.optosigma.com/us_en/raman.html"
        assert res["coating"] == "Opt RSF-325RU"
        assert res["properties"]["Cut On Frequency"] == "325 nm"
        assert res["material"]["Name"] == "UVFS"

def test_download_threed_optix_to_db(db):
    with patch("requests.get") as mock_get:
        # Mock index response first, then detail response
        mock_resp_index = MagicMock()
        mock_resp_index.status_code = 200
        mock_resp_index.text = MOCK_INDEX_HTML

        mock_resp_detail = MagicMock()
        mock_resp_detail.status_code = 200
        mock_resp_detail.text = MOCK_DETAIL_HTML

        mock_get.side_effect = [mock_resp_index, mock_resp_detail]

        counts = download_threed_optix_to_db(db, max_pages=1)
        # Counts are keyed by the canonical component kind (shared across scrapers).
        assert counts.get("longpass") == 1

        # Check DB entry — the canonical ingestion contract sets category + source.
        probes = db.get_standardized_items(include_uncurated=True)
        assert len(probes) == 1
        assert probes[0]["chromophore_name"] == "RSF-25C-325RU"
        assert probes[0]["category"] == "filter"
        assert probes[0]["source"] == "3doptix"

        # Check optical properties
        props = db.get_optical_properties(probes[0]["probe_id"])
        # Every component now carries an Origin + granular component_kind.
        assert props.get("Origin") == "OptoSigma Longpass Filter"
        assert props.get("component_kind") == "longpass"
        assert props.get("Brand") == "OptoSigma"
        assert props.get("Subtype") == "Longpass Filter"
        assert props.get("Cut On Frequency") == "325 nm"
        assert props.get("Material Name") == "UVFS"
        


