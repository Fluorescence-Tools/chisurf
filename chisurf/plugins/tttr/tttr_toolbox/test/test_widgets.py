"""Tests for the TTTR Tools window (shared NavigationPanelTool shell)."""


def test_tttr_tools_uses_shared_navigation_shell(qapp, qtbot):
    from chisurf.gui.widgets.navigation import NavigationPanelTool
    from chisurf.plugins.tttr.tttr_toolbox.gui.tool import TTTR_PANELS, TttrToolboxTool

    w = TttrToolboxTool()
    qtbot.addWidget(w)

    # same base / look as FCS Tools, Structure Tools, Burst Analysis, ...
    assert isinstance(w, NavigationPanelTool)
    assert "TTTR Tools" in w.windowTitle()

    # one navigation entry per panel (incl. the separator)
    assert w.nav_list.count() == len(TTTR_PANELS)
    names = [w.nav_list.item(i).text() for i in range(w.nav_list.count())]
    assert any("ALEX Creator" in n for n in names)
    assert any("Split / Convert" in n for n in names)
    assert any("Count Rate Analysis" in n for n in names)

    # panels load lazily: only the first + the (possibly persisted) current panel
    # are instantiated; every other panel stays a placeholder.
    kept = {0, w.nav_list.currentRow()}
    assert all(
        p.get("instance") is None
        for i, p in enumerate(w.panels)
        if i not in kept and not p.get("separator")
    )


def test_toolbox_remembers_window_and_selection(qapp, qtbot, tmp_path):
    """The toolbox persists geometry, splitter and the selected panel."""
    from qtpy import QtCore

    from chisurf.plugins.tttr.tttr_toolbox.gui.tool import TttrToolboxTool

    # Isolate to a throwaway ini file so the real user settings are untouched.
    ini = str(tmp_path / "toolbox.ini")

    def _settings():
        return QtCore.QSettings(ini, QtCore.QSettings.IniFormat)

    w = TttrToolboxTool()
    qtbot.addWidget(w)
    assert w._settings_key == "tttr_toolbox"
    w._settings = _settings
    w.resize(1180, 760)
    w.nav_list.setCurrentRow(6)  # Count Rate Analysis
    w._save_window_state()

    w2 = TttrToolboxTool()
    qtbot.addWidget(w2)
    w2._settings = _settings
    w2._restore_window_state()
    assert w2.nav_list.currentRow() == 6


def test_panels_are_data_driven_from_json():
    """The panel list is built from panels.json (entrypoint strings), not Python."""
    import json
    import pathlib

    from chisurf.plugins.tttr.tttr_toolbox.gui import tool as tool_mod

    spec = json.loads((pathlib.Path(tool_mod.__file__).with_name("panels.json")).read_text())
    tool_panels = [p for p in spec["panels"] if not p.get("separator")]
    # every non-separator entry declares a resolvable "module:Class" entrypoint
    for p in tool_panels:
        module_name, _, attr = p["entrypoint"].partition(":")
        assert module_name and attr, p
    # and each becomes a panel with a generated factory
    built = [p for p in tool_mod.TTTR_PANELS if not p.get("separator")]
    assert len(built) == len(tool_panels)
    assert all(callable(p["factory"]) for p in built)


def test_included_plugins_are_menu_hidden():
    import importlib

    for mod in (
        "chisurf.plugins.tttr.ptu_alex_creator",
        "chisurf.plugins.tttr.tttr_microtime_shifter",
        "chisurf.plugins.tttr.ptu_header_edit",
        "chisurf.plugins.tttr.tttr_splitter",
        "chisurf.plugins.tttr.tttr_count_rate_analysis",
        "chisurf.plugins.tttr.audifier",
    ):
        m = importlib.import_module(mod)
        assert getattr(m, "menu_hidden", False) is True, mod
