"""Tests for the integrated burst workflow shell."""

from __future__ import annotations

from pathlib import Path


def test_burst_workflow_panel_order() -> None:
    """Meta burst workflow exposes requested ordered steps."""
    from chisurf.plugins.burst.burst_analysis.gui.tool import BURST_PANELS

    labels = [f"{panel.get('icon', '')} {panel['name']}".strip() for panel in BURST_PANELS]
    assert labels == [
        "📂 1. Data Selection",
        "🔢 2. Channels",
        "🔎 3. Burst Selection",
        "📊 4. BVA",
        "🎯 5. MLE-Lifetime",
        "🔀 6. H2MM",
        "📋 7. Browser",
        "────────",
        "🌙 Background",
    ]
    assert BURST_PANELS[7]["separator"] is True


def test_h2mm_panel_is_flagged_experimental() -> None:
    """The H2MM step carries the experimental flag so the nav shows ⚠ + banner."""
    from chisurf.plugins.burst.burst_analysis.gui.tool import BURST_PANELS

    h2mm = next(p for p in BURST_PANELS if p.get("role") == "h2mm")
    assert h2mm.get("experimental") is True
    assert h2mm.get("experimental_message")


def test_workflow_context_payload_is_json_ready(tmp_path: Path) -> None:
    """Workflow context serializes paths and MFDB artifact handoff data."""
    from chisurf.plugins.burst.burst_analysis.gui.tool import BurstWorkflowContext

    context = BurstWorkflowContext(
        raw_files=[tmp_path / "a.spc"],
        burst_folder=tmp_path / "burstwise",
        bur_files=[tmp_path / "burstwise" / "bi4_bur" / "a.bur"],
        mfdb_artifacts={"sidecar_artifacts": {"output_folder": "artifact-1"}},
        raw_mfdb_artifacts={"imports": {"a.spc": {"object_result": {"ok": True}}}},
    )
    payload = context.to_payload()
    assert payload["raw_files"] == [str(tmp_path / "a.spc")]
    assert payload["burst_folder"] == str(tmp_path / "burstwise")
    assert payload["bur_files"] == [str(tmp_path / "burstwise" / "bi4_bur" / "a.bur")]
    assert payload["mfdb_artifacts"]["sidecar_artifacts"]["output_folder"] == "artifact-1"
    assert payload["raw_mfdb_artifacts"]["imports"]["a.spc"]["object_result"]["ok"] is True


def test_bva_factory_hides_internal_channel_tab(monkeypatch) -> None:
    """Embedded BVA keeps detector_page but hides its channel-definition dock."""
    from chisurf.plugins.burst.burst_analysis.gui import tool as meta_tool

    class FakeDockArea:
        def __init__(self) -> None:
            self._all_widgets = [object(), object()]
            self.hidden: list[int] = []

        def tabText(self, index: int) -> str:
            return ["BVA Settings", "Channel Definitions"][index]

        def hideTab(self, index: int) -> None:
            self.hidden.append(index)

    class FakeBVA:
        def __init__(self, parent=None, *, embedded: bool = False) -> None:
            self.parent = parent
            self.embedded = embedded
            self.detector_page = object()
            self.dock_area = FakeDockArea()

    monkeypatch.setattr(
        "chisurf.plugins.burst.burst_bva.gui.tool.BVATool",
        FakeBVA,
    )

    widget = meta_tool._burst_bva(parent=None)
    assert widget.detector_page is not None
    assert widget.embedded is True
    assert widget.dock_area.hidden == [1]


def test_navigation_embeds_main_window_pages_as_child_widgets() -> None:
    """Embedded legacy main windows stay child widgets inside the navigation shell."""
    from qtpy import QtCore, QtWidgets

    from chisurf.gui.widgets.navigation import NavigationPanelTool

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    loaded: list[QtWidgets.QMainWindow] = []

    def factory(parent):
        widget = QtWidgets.QMainWindow(parent)
        loaded.append(widget)
        return widget

    tool = NavigationPanelTool(
        title="Navigation Test",
        panels=[{"name": "Panel", "factory": factory}],
        minimum_size=(200, 120),
        initial_size=(220, 140),
        navigation_width=100,
    )
    app.processEvents()

    assert loaded
    assert not loaded[0].isWindow()
    assert loaded[0].windowType() == QtCore.Qt.Widget
    assert loaded[0].testAttribute(QtCore.Qt.WA_DontCreateNativeAncestors)
    assert not loaded[0].testAttribute(QtCore.Qt.WA_QuitOnClose)
    tool.close()
    app.processEvents()


def test_bva_embedded_mode_does_not_restore_top_level_geometry(monkeypatch) -> None:
    """Embedded BVA reuses dock layout without restoring standalone window state."""
    import chisurf.plugins.burst.burst_bva.gui.tool as bva_tool

    class Settings:
        def __init__(self, *args) -> None:
            self.values = {
                "dock_layout": '{"tabs": []}',
                "window_geometry": b"geometry",
                "window_state": b"state",
            }

        def value(self, key):
            return self.values.get(key)

    class DockArea:
        def __init__(self) -> None:
            self.applied = None

        def set_layout_state(self, layout_state, emit_change=False) -> None:
            self.applied = (layout_state, emit_change)

    tool = bva_tool.BVATool.__new__(bva_tool.BVATool)
    tool._embedded = True
    tool.dock_area = DockArea()
    restore_calls: list[str] = []
    tool.restoreGeometry = lambda geometry: restore_calls.append("geometry")
    tool.restoreState = lambda state: restore_calls.append("state")
    monkeypatch.setattr(bva_tool, "QSettings", Settings)

    tool._restore_dock_layout()

    assert restore_calls == []
    assert tool.dock_area.applied == ({"tabs": []}, False)


def test_data_selection_imports_local_files_to_mfdb(tmp_path: Path) -> None:
    """Adding local TTTR data imports it through MFDB RPC."""
    from qtpy import QtWidgets

    from chisurf.plugins.burst.burst_analysis.gui.tool import BurstDataSelectionWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    spc = tmp_path / "a.spc"
    spc.write_bytes(b"raw")
    calls: list[tuple[str, dict]] = []

    class Client:
        def call(self, method, params=None):
            calls.append((method, params or {}))
            if method == "mfdb.objects.put":
                return {"ok": True, "object": {"object_uuid": "obj-1"}}
            if method == "raw_data.register":
                return {"ok": True, "raw_data": {"raw_data_id": "raw-1"}}
            return {}

    widget = BurstDataSelectionWidget()
    widget._mfdb_client = Client()
    widget.add_paths([spc])

    assert widget.paths() == [spc.resolve()]
    assert [call[0] for call in calls] == ["mfdb.objects.put", "raw_data.register"]
    payload = widget.mfdb_payload()
    assert payload["imports"][str(spc.resolve())]["object_result"]["object"]["object_uuid"] == "obj-1"
    assert payload["imports"][str(spc.resolve())]["raw_data_result"]["raw_data"]["raw_data_id"] == "raw-1"
    widget.close()
    app.processEvents()


def test_data_selection_resolves_mfdb_dataset_path() -> None:
    """MFDB data selection resolves artifact IDs through mfdb.datasets.open."""
    from qtpy import QtWidgets

    from chisurf.plugins.burst.burst_analysis.gui.tool import BurstDataSelectionWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    calls: list[tuple[str, dict]] = []

    class Client:
        def call(self, method, params=None):
            calls.append((method, params or {}))
            return {"local_path": "/tmp/from-mfdb.spc"}

    widget = BurstDataSelectionWidget()
    widget._mfdb_client = Client()
    assert widget._open_mfdb_dataset("artifact-1") == "/tmp/from-mfdb.spc"
    assert calls == [("mfdb.datasets.open", {"artifact_id": "artifact-1"})]
    widget.close()
    app.processEvents()
