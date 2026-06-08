from qtpy import QtCore, QtWidgets

from chisurf.gui.widgets.dock_area import DockArea
from chisurf.gui.widgets.dock_area.dock_area import DockSplitter, DockTabWidget


def test_dock_area_basic(qtbot):
    """Test basic tab adding and switching index in DockArea."""
    dock_area = DockArea()
    qtbot.addWidget(dock_area)

    # Create test widgets
    w1 = QtWidgets.QWidget()
    w2 = QtWidgets.QWidget()
    w3 = QtWidgets.QWidget()

    # Add tabs
    dock_area.addTab(w1, "Tab 1")
    dock_area.addTab(w2, "Tab 2")
    dock_area.addTab(w3, "Tab 3")

    # Verify root widget is a DockTabWidget
    assert isinstance(dock_area._root_widget, DockTabWidget)
    assert dock_area._root_widget.count() == 3
    assert dock_area.currentIndex() == 0

    # Switch tab index and check signal
    signals = []

    def on_change(idx):
        signals.append(idx)

    dock_area.currentChanged.connect(on_change)

    dock_area._root_widget.setCurrentIndex(1)
    assert dock_area.currentIndex() == 1
    assert signals == [1]


def test_dock_area_split(qtbot):
    """Test split behavior and tree simplification/cleanup in DockArea."""
    dock_area = DockArea()
    qtbot.addWidget(dock_area)

    w1 = QtWidgets.QWidget()
    w2 = QtWidgets.QWidget()

    dock_area.addTab(w1, "Tab 1")
    dock_area.addTab(w2, "Tab 2")

    root_tw = dock_area._root_widget

    # Split: Remove Tab 2 (w2) and place in split to the right of root_tw
    root_tw.removeTab(1)
    new_tw = DockTabWidget(dock_area)
    new_tw.addTab(w2, "Tab 2")

    dock_area.split_tab_widget(root_tw, new_tw, "right")

    # Check that root widget is now a DockSplitter
    assert isinstance(dock_area._root_widget, DockSplitter)
    assert dock_area._root_widget.orientation() == QtCore.Qt.Horizontal
    assert dock_area._root_widget.count() == 2

    # Left widget should be root_tw, right should be new_tw
    assert dock_area._root_widget.widget(0) == root_tw
    assert dock_area._root_widget.widget(1) == new_tw

    # Remove last tab from root_tw, it should collapse the splitter and make new_tw the root widget
    root_tw.removeTab(0)
    dock_area.cleanup_empty_tab_widget(root_tw)

    assert dock_area._root_widget == new_tw
    assert isinstance(dock_area._root_widget, DockTabWidget)


def test_dock_area_restore(qtbot):
    """Test restoring split panels back to the main/primary tab group."""
    dock_area = DockArea()
    qtbot.addWidget(dock_area)

    w1 = QtWidgets.QWidget()
    w2 = QtWidgets.QWidget()

    dock_area.addTab(w1, "Tab 1")
    dock_area.addTab(w2, "Tab 2")

    root_tw = dock_area._root_widget

    # Split
    root_tw.removeTab(1)
    new_tw = DockTabWidget(dock_area)
    new_tw.addTab(w2, "Tab 2")
    dock_area.split_tab_widget(root_tw, new_tw, "right")

    assert isinstance(dock_area._root_widget, DockSplitter)

    # Restore Tab 2 back to the main tab widget
    dock_area.restore_tab(new_tw, 0)

    # Root splitter should be cleaned up and root widget should be back to DockTabWidget
    assert isinstance(dock_area._root_widget, DockTabWidget)
    assert dock_area._root_widget.count() == 2
    assert dock_area._root_widget.widget(0) == w1
    assert dock_area._root_widget.widget(1) == w2
