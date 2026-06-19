from pathlib import Path

from chisurf.core.plugin import load_manifest


def test_manifest_loads():
    manifest_path = Path(__file__).parents[1] / "manifest.json"
    manifest = load_manifest(manifest_path)
    assert manifest is not None
    assert manifest.id == "trace_browser"
    assert manifest.version == "2.0.0"
    assert manifest.display_name == "Spectroscopy:Single-Molecule:Trace Browser"
    assert manifest.entrypoints.gui == "chisurf.plugins.tttr.trace_browser.gui.tool:TraceBrowserTool"
    assert manifest.entrypoints.cli == "trace-browser=chisurf.plugins.tttr.trace_browser.cli:cli"
    assert manifest.entrypoints.services == "chisurf.plugins.tttr.trace_browser.backend.services:register_services"
    assert len(manifest.rpc_methods) == 6
    assert manifest.state_namespace == "trace_browser"
