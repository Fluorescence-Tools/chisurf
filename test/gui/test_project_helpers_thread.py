"""Test that recent-projects menu refresh is safe when called from worker threads."""

import threading

import pytest
from qtpy import QtWidgets

from chisurf.gui.project_helpers import refresh_recent_projects_menu


@pytest.fixture
def main_window(qtbot):
    """Create a minimal main window with a Recent Projects menu."""
    window = QtWidgets.QMainWindow()
    window._recent_projects = ["/tmp/project_a", "/tmp/project_b"]
    window._menu_recent_projects = QtWidgets.QMenu("Recent Projects", window)
    qtbot.add_widget(window)
    return window


def test_refresh_recent_projects_menu_on_main_thread(main_window):
    """When called on the GUI thread, refresh should run synchronously."""
    refresh_recent_projects_menu(main_window)

    actions = main_window._menu_recent_projects.actions()
    texts = [a.text() for a in actions]
    assert any("project_a" in t for t in texts)
    assert any("project_b" in t for t in texts)


def test_refresh_recent_projects_menu_from_worker_thread(main_window, qtbot):
    """refresh_recent_projects_menu must not manipulate Qt widgets from a worker thread.

    When called from a worker thread, the actual refresh must be marshalled to
    the GUI thread via run_on_gui_thread.
    """
    errors = []

    def worker():
        try:
            refresh_recent_projects_menu(main_window)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "Worker thread did not finish in time"
    assert not errors, f"Worker raised: {errors}"

    # Give the queued GUI callback a chance to run.
    def menu_populated():
        actions = main_window._menu_recent_projects.actions()
        return any("project_a" in (a.text() or "") for a in actions)

    qtbot.wait_until(menu_populated, timeout=2000)

    actions = main_window._menu_recent_projects.actions()
    texts = [a.text() for a in actions]
    assert any("project_a" in t for t in texts)
    assert any("project_b" in t for t in texts)
