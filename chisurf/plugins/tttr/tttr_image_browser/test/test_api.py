from chisurf.plugins.tttr.tttr_image_browser.api.contract import METHOD_LIST_FILES, contract_descriptor
from chisurf.plugins.tttr.tttr_image_browser.api.io import list_files
from chisurf.plugins.tttr.tttr_image_browser.core.metadata import load_meta, save_meta


def test_contract_describes_rpc_methods():
    contract = contract_descriptor()
    assert METHOD_LIST_FILES in contract["methods"]


def test_metadata_roundtrip(tmp_path):
    save_meta(tmp_path, {"a.ptu": {"rating": 2, "annotation": "good"}})
    assert load_meta(tmp_path)["a.ptu"]["rating"] == 2


def test_list_files(tmp_path):
    # Create a dummy file with allowed extension
    img_file = tmp_path / "image.ptu"
    img_file.write_bytes(b"dummy")
    rows = list_files(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["name"] == "image.ptu"
