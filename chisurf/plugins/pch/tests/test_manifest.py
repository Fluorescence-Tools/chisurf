from pathlib import Path

from chisurf.core.plugin import load_manifest


def test_manifest_loads():
    manifest_path = Path(__file__).parents[1] / "manifest.json"
    assert manifest_path.exists()
    manifest = load_manifest(manifest_path)
    assert manifest is not None
    assert manifest.id == "pch"
    assert manifest.version == "2.0.0"
    assert manifest.entrypoints.gui is not None
    assert manifest.entrypoints.cli is not None
    assert manifest.entrypoints.services is not None
    assert len(manifest.rpc_methods) == 3


def test_manifest_rpc_methods():
    manifest_path = Path(__file__).parents[1] / "manifest.json"
    manifest = load_manifest(manifest_path)
    names = [m.name for m in manifest.rpc_methods]
    assert "pch.load_tttr" in names
    assert "pch.compute" in names
    assert "pch.fit" in names
