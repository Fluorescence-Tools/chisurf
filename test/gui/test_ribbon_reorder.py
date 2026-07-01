"""Headless tests for drag-to-reorder of ribbon panel buttons.

These exercise the reordering/persistence logic (``_moveWidget``,
``_saveButtonOrder``, ``_restoreButtonOrder``) without simulating an actual
mouse drag, which is impractical under the offscreen platform.
"""
import pytest
from qtpy import QtCore

from chisurf.gui.widgets.ribbon.panel import RibbonPanel

# Unique title so the QSettings key we touch cannot clobber real ribbon state.
PANEL_TITLE = "PytestReorderPanel"
STORAGE_KEY = f"order/UnknownCategory::{PANEL_TITLE}"


@pytest.fixture
def clean_settings():
    """Remove the panel's persisted order before and after each test."""
    settings = QtCore.QSettings("ChiSurf", "RibbonState")
    settings.remove(STORAGE_KEY)
    settings.sync()
    yield settings
    settings.remove(STORAGE_KEY)
    settings.sync()


def _make_panel(qtbot):
    panel = RibbonPanel(title=PANEL_TITLE)
    qtbot.addWidget(panel)
    buttons = [panel.addButton(text=name) for name in ("A", "B", "C")]
    return panel, buttons


def _order(panel):
    return [w.text() for w in panel.widgets()]


def test_move_widget_reorders_and_persists(qapp, qtbot, clean_settings):
    panel, _ = _make_panel(qtbot)
    assert _order(panel) == ["A", "B", "C"]

    # Move "A" (index 0) to the end (target index == len -> after "C").
    panel._moveWidget(0, len(panel.widgets()))
    assert _order(panel) == ["B", "C", "A"]

    # Order is persisted as a list of stable per-panel sequence ids.
    saved = clean_settings.value(STORAGE_KEY, [])
    assert saved == ["1", "2", "0"]


def test_move_widget_onto_itself_is_noop(qapp, qtbot, clean_settings):
    panel, _ = _make_panel(qtbot)
    panel._moveWidget(1, 1)
    assert _order(panel) == ["A", "B", "C"]
    # No persistence write for a no-op move.
    assert clean_settings.value(STORAGE_KEY, None) is None


def test_restore_applies_saved_order(qapp, qtbot, clean_settings):
    # Persist a custom order: C, A, B (sequence ids 2, 0, 1).
    clean_settings.setValue(STORAGE_KEY, ["2", "0", "1"])
    clean_settings.sync()

    panel, _ = _make_panel(qtbot)
    panel._restoreButtonOrder()
    assert _order(panel) == ["C", "A", "B"]


def test_restore_without_saved_order_keeps_insertion_order(qapp, qtbot, clean_settings):
    panel, _ = _make_panel(qtbot)
    panel._restoreButtonOrder()
    assert _order(panel) == ["A", "B", "C"]
