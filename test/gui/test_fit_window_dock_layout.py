from __future__ import annotations

from qtpy import QtCore, QtWidgets

import chisurf as cs
from chisurf.gui.widgets.dock_area.dock_area import DockArea, DockSplitter, DockTabWidget
from chisurf.gui.widgets.fitting.fit_subwindow import FitSubWindow


class DummyModel:
    """Dummy model used by dock layout persistence tests."""


class DummyFit:
    """Dummy fit used by dock layout persistence tests."""

    model = DummyModel()


def _make_dock_area(host: QtWidgets.QWidget):
    """Create a dock area with two labelled plot widgets."""
    area = DockArea(host)
    first = QtWidgets.QLabel("first")
    second = QtWidgets.QLabel("second")
    for idx, widget, name in ((0, first, "first"), (1, second, "second")):
        widget.setProperty("fit_plot_index", idx)
        widget.setProperty("fit_plot_name", name)
        area.addTab(widget, name)
    return area, first, second


def _split_second_plot(area: DockArea, second: QtWidgets.QWidget) -> None:
    """Move the second plot into a separate right-side panel."""
    main_tab_widget = area.find_main_tab_widget()
    new_tab_widget = DockTabWidget(area)
    new_tab_widget.addTab(second, area.tabText(1))
    area.split_tab_widget(main_tab_widget, new_tab_widget, "right")


def test_dock_area_ignores_empty_layout_state(qapp, qtbot):
    """Empty serialized layouts do not clear the current dock area."""
    host = QtWidgets.QWidget()
    area, first, second = _make_dock_area(host)

    empty_state = {
        "version": 1,
        "root": {"type": "tab", "tabs": []},
        "active_tab_widget": None,
        "current_index": -1,
    }

    assert not area.set_layout_state(
        empty_state,
        key_func=lambda widget: widget.property("fit_plot_name"),
        emit_change=False,
    )
    qtbot.wait(0)

    assert area.count() == 2
    assert area.widget(0) is first
    assert area.widget(1) is second


def test_dock_area_layout_state_roundtrip(qapp, qtbot):
    """DockArea layout state round-trips through split tab widgets."""
    host = QtWidgets.QWidget()
    area, first, second = _make_dock_area(host)
    _split_second_plot(area, second)

    state = area.get_layout_state(key_func=lambda widget: widget.property("fit_plot_name"))

    area.set_layout_state(
        state,
        key_func=lambda widget: widget.property("fit_plot_name"),
        emit_change=False,
    )
    qtbot.wait(0)

    assert isinstance(area._root_widget, DockSplitter)
    assert area._root_widget.orientation() == QtCore.Qt.Horizontal
    assert area.count() == 2
    assert area.widget(0) is first
    assert area.widget(1) is second
    assert area._root_widget.count() == 2


def test_fit_window_dock_layout_persists_per_model_class(qapp, monkeypatch, tmp_path, qtbot):
    """Fit-window dock layouts are stored under a model-class key."""
    monkeypatch.setattr(cs.core.settings, "get_path", lambda path_type: tmp_path)

    host = QtWidgets.QWidget()
    fit_window = FitSubWindow.__new__(FitSubWindow)
    fit_window.fit = DummyFit()
    fit_window.plot_tab_widget = _make_dock_area(host)[0]
    _split_second_plot(fit_window.plot_tab_widget, fit_window.plot_tab_widget.widget(1))

    fit_window.save_fit_dock_layout_state()

    restored_host = QtWidgets.QWidget()
    restored_window = FitSubWindow.__new__(FitSubWindow)
    restored_window.fit = DummyFit()
    restored_window.plot_tab_widget = _make_dock_area(restored_host)[0]

    restored_window.restore_fit_dock_layout_state()
    qtbot.wait(0)

    assert isinstance(restored_window.plot_tab_widget._root_widget, DockSplitter)
    assert restored_window.plot_tab_widget.count() == 2
    assert restored_window.plot_tab_widget._root_widget.count() == 2
    assert (tmp_path / "fit_window_dock_layouts.ini").is_file()
