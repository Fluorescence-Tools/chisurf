from chisurf.plugins.tttr.trace_browser.api.contract import METHOD_LIST_FILES, contract_descriptor
from chisurf.plugins.tttr.trace_browser.api.io import list_files
from chisurf.plugins.tttr.trace_browser.core.metadata import load_meta, save_meta


def test_contract_describes_rpc_methods(tmp_path):
    contract = contract_descriptor()
    assert METHOD_LIST_FILES in contract["methods"]


def test_metadata_roundtrip(tmp_path):
    save_meta(tmp_path, {"a.ptu": {"rating": 2, "annotation": "good"}})
    assert load_meta(tmp_path)["a.ptu"]["rating"] == 2


def test_list_files(tmp_path):
    trace = tmp_path / "demo.ptu"
    trace.write_bytes(b"trace")
    rows = list_files(str(tmp_path))
    assert rows[0]["name"] == "demo.ptu"
