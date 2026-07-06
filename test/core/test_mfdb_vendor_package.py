"""Regression tests for the vendored MFDB package boundary."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mfdb_canonical_import_exports_repository():
    """The vendored package is importable as ``mfdb``."""
    import mfdb
    from mfdb.repository import MFDatabase

    assert mfdb.MFDatabase is MFDatabase


def test_chisurf_core_mfdb_remains_transitional_facade():
    """Existing ChiSurf imports still resolve while callers migrate to ``mfdb``."""
    import chisurf.core.mfdb as legacy_mfdb
    import mfdb

    assert legacy_mfdb.MFDatabase is mfdb.MFDatabase


def test_chisurf_core_mfdb_submodules_alias_vendored_modules():
    """Legacy submodule imports must not load duplicate class objects."""
    import chisurf.core.mfdb.base as legacy_base
    import chisurf.core.mfdb.payload_codec as legacy_payload_codec
    import chisurf.core.mfdb.payload_models as legacy_payload_models
    import chisurf.core.mfdb.repository as legacy_repository
    import mfdb.base
    import mfdb.payload_codec
    import mfdb.payload_models
    import mfdb.repository

    assert legacy_base is mfdb.base
    assert legacy_payload_codec is mfdb.payload_codec
    assert legacy_payload_models is mfdb.payload_models
    assert legacy_repository is mfdb.repository
    assert legacy_repository.MFDatabase is mfdb.repository.MFDatabase
    assert legacy_payload_codec.PayloadSchemaError is mfdb.payload_codec.PayloadSchemaError


def test_mfdb_extraction_boundary_modules_do_not_import_chisurf():
    """Standalone MFDB boundary modules must not import ChiSurf internals."""
    checked = [
        ROOT / "modules" / "mfdb" / "src" / "mfdb" / "config.py",
        ROOT / "modules" / "mfdb" / "src" / "mfdb" / "database_resolver.py",
        ROOT / "modules" / "mfdb" / "src" / "mfdb" / "payload_models.py",
        ROOT / "modules" / "mfdb" / "src" / "mfdb" / "project_archiver.py",
        ROOT / "modules" / "mfdb" / "src" / "mfdb" / "repository.py",
        ROOT / "modules" / "mfdb" / "src" / "mfdb" / "result_registry.py",
        ROOT / "modules" / "mfdb" / "src" / "mfdb" / "seed_data.py",
        ROOT / "modules" / "mfdb" / "src" / "mfdb" / "session.py",
    ]

    offenders = [
        str(path.relative_to(ROOT))
        for path in checked
        if "import chisurf" in path.read_text() or "from chisurf" in path.read_text()
    ]

    assert offenders == []


def test_active_mfdb_plugins_import_vendored_namespace_directly():
    """Active MFDB-facing plugin paths should not regress to the old namespace."""
    checked_roots = [
        ROOT / "chisurf" / "plugins" / "core" / "mfdb_admin",
        ROOT / "chisurf" / "plugins" / "core" / "database_connector",
        ROOT / "chisurf" / "plugins" / "core" / "project_browser",
    ]
    offenders: list[str] = []
    for root in checked_roots:
        for path in root.rglob("*.py"):
            text = path.read_text()
            if (
                "chisurf.core.mfdb" in text
                or "chisurf.plugins.core.mfdb_admin.backend" in text
                or "chisurf.plugins.core.mfdb_admin.gui" in text
            ):
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_mfdb_admin_lives_inside_mfdb_package():
    """MFDB Admin services and GUI load from the canonical ``mfdb.admin`` app."""
    import importlib
    import json

    manifest = json.loads(
        (ROOT / "chisurf" / "plugins" / "core" / "mfdb_admin" / "manifest.json").read_text()
    )
    assert manifest["entrypoints"]["gui"] == "mfdb.admin.gui.tool:MFDBWidget"
    assert manifest["entrypoints"]["services"] == (
        "mfdb.admin.backend.services:register_services"
    )

    gui_tool = importlib.import_module("mfdb.admin.gui.tool")
    legacy_gui_tool = importlib.import_module(
        "chisurf.plugins.core.mfdb_admin.gui.tool"
    )
    services = importlib.import_module("mfdb.admin.backend.services")
    legacy_services = importlib.import_module(
        "chisurf.plugins.core.mfdb_admin.backend.services"
    )

    assert legacy_gui_tool.MFDBWidget is gui_tool.MFDBWidget
    assert legacy_services.register_services is services.register_services
    legacy_gui_source = (
        ROOT / "chisurf" / "plugins" / "core" / "mfdb_admin" / "gui" / "tool.py"
    ).read_text()
    legacy_source = (
        ROOT / "chisurf" / "plugins" / "core" / "mfdb_admin" / "backend" / "services.py"
    ).read_text()
    assert "from mfdb.admin.gui.tool import *" in legacy_gui_source
    assert "from mfdb.admin.backend.services import *" in legacy_source


def test_mfdb_admin_manifest_declares_registered_methods():
    """The MFDB Admin manifest should declare every registered RPC method."""
    import json

    from chisurf.server.dispatcher import ServiceDispatcher
    from chisurf.server.session import SessionState
    from mfdb.admin.backend.services import register_services

    manifest = json.loads(
        (ROOT / "chisurf" / "plugins" / "core" / "mfdb_admin" / "manifest.json").read_text()
    )
    declared = {item["name"] for item in manifest.get("rpc_methods", [])}
    dispatcher = ServiceDispatcher(SessionState())
    register_services(dispatcher)
    registered = set(dispatcher.list_methods())

    assert sorted(registered - declared) == []
    assert sorted(declared - registered) == []
