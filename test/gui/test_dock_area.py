from qtpy import QtCore, QtWidgets

from chisurf.gui.widgets.dock_area import DockArea
from chisurf.gui.widgets.dock_area.dock_area import DockSplitter, DockTabWidget
from chisurf.gui.widgets.dock_area.dock_stacked_tab_widget import DockStackedTabWidget


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


def test_dock_area_close_hides_by_default_and_restores(qtbot):
    """Default close behavior should hide docks without destroying them."""
    dock_area = DockArea()
    qtbot.addWidget(dock_area)

    w1 = QtWidgets.QWidget()
    w2 = QtWidgets.QWidget()
    dock_area.addTab(w1, "Keep")
    dock_area.addTab(w2, "Hide")

    assert dock_area.count() == 2
    assert dock_area.visibleCount() == 2

    dock_area._request_close_tab(1)

    assert dock_area.count() == 2
    assert dock_area.visibleCount() == 1
    assert dock_area.hiddenIndexes() == [1]
    assert dock_area.widget(1) is w2
    assert w2.parentWidget() is dock_area
    assert dock_area.showTab(1) is True
    assert dock_area.visibleCount() == 2


def test_dock_area_does_not_close_last_visible_dock(qtbot):
    """The last visible dock must stay open."""
    dock_area = DockArea()
    qtbot.addWidget(dock_area)

    dock_area.addTab(QtWidgets.QWidget(), "Only")

    assert dock_area.canCloseTab(0) is False
    dock_area._request_close_tab(0)
    assert dock_area.visibleCount() == 1
    assert dock_area.hiddenIndexes() == []


def test_dock_area_remove_close_mode_removes_dock(qtbot):
    """Docks marked with remove close mode should be removed from registry."""
    dock_area = DockArea()
    qtbot.addWidget(dock_area)

    keep = QtWidgets.QWidget()
    files = QtWidgets.QWidget()
    dock_area.addTab(keep, "Keep")
    dock_area.addTab(files, "Files", close_mode="remove")

    dock_area._request_close_tab(1)

    assert dock_area.count() == 1
    assert dock_area.widget(0) is keep
    assert dock_area.hiddenIndexes() == []


def test_layout_restore_preserves_pages_missing_from_saved_layout(qtbot):
    """Restoring a partial layout must not destroy registered page widgets."""
    dock_area = DockArea()
    qtbot.addWidget(dock_area)

    keep = QtWidgets.QWidget()
    filter_page = QtWidgets.QWidget()
    filter_layout = QtWidgets.QVBoxLayout(filter_page)
    combo = QtWidgets.QComboBox(filter_page)
    combo.addItem("Test")
    filter_layout.addWidget(combo)

    dock_area.addTab(keep, "Keep")
    dock_area.addTab(filter_page, "Filter Settings")
    state = {
        "version": 1,
        "root": {
            "type": "tab",
            "tabs": [
                {
                    "widget_key": "Keep",
                    "tab_name": "Keep",
                    "tab_text": "Keep",
                }
            ],
            "current_index": 0,
        },
        "active_tab_widget": [],
        "current_index": 0,
    }

    assert dock_area.set_layout_state(state)
    QtWidgets.QApplication.processEvents()

    assert combo.currentText() == "Test"
    assert filter_page.parentWidget() is dock_area
    assert dock_area.hiddenIndexes() == [1]
    assert dock_area.showTab(1) is True
    assert combo.currentText() == "Test"
    assert dock_area.isTabVisible(1) is True


def test_dock_area_stacked_option_creates_stacked_tab_widget(qtbot):
    """A DockArea created with stacked_tabs=True uses DockStackedTabWidget."""
    dock_area = DockArea(stacked_tabs=True)
    qtbot.addWidget(dock_area)

    w1 = QtWidgets.QWidget()
    dock_area.addTab(w1, "Tab 1")

    assert isinstance(dock_area._root_widget, DockStackedTabWidget)


def test_dock_area_stacked_basic(qtbot):
    """Basic tab adding and switching works in stacked tab mode."""
    dock_area = DockArea(stacked_tabs=True)
    qtbot.addWidget(dock_area)

    w1 = QtWidgets.QWidget()
    w2 = QtWidgets.QWidget()
    w3 = QtWidgets.QWidget()

    dock_area.addTab(w1, "Tab 1")
    dock_area.addTab(w2, "Tab 2")
    dock_area.addTab(w3, "Tab 3")

    assert dock_area._root_widget.count() == 3
    assert dock_area.currentIndex() == 0

    signals = []

    def on_change(idx):
        signals.append(idx)

    dock_area.currentChanged.connect(on_change)
    dock_area._root_widget.setCurrentIndex(1)

    assert dock_area.currentIndex() == 1
    assert signals == [1]


def test_dock_area_stacked_layout_state_roundtrip(qtbot):
    """Layout state round-trips through split tab widgets in stacked mode."""
    dock_area = DockArea(stacked_tabs=True)
    qtbot.addWidget(dock_area)

    w1 = QtWidgets.QWidget()
    w2 = QtWidgets.QWidget()

    dock_area.addTab(w1, "Tab 1")
    dock_area.addTab(w2, "Tab 2")

    root_tw = dock_area._root_widget
    root_tw.removeTab(1)
    new_tw = DockStackedTabWidget(dock_area)
    new_tw.addTab(w2, "Tab 2")
    dock_area.split_tab_widget(root_tw, new_tw, "right")

    assert isinstance(dock_area._root_widget, DockSplitter)
    assert dock_area._root_widget.orientation() == QtCore.Qt.Horizontal
    assert dock_area._root_widget.count() == 2

    dock_area.restore_tab(new_tw, 0)

    assert isinstance(dock_area._root_widget, DockStackedTabWidget)
    assert dock_area._root_widget.count() == 2
    assert dock_area._root_widget.widget(0) is w1
    assert dock_area._root_widget.widget(1) is w2


def test_dock_area_stacked_close_hides_and_restores(qtbot):
    """Default close behavior hides docks without destroying them in stacked mode."""
    dock_area = DockArea(stacked_tabs=True)
    qtbot.addWidget(dock_area)

    w1 = QtWidgets.QWidget()
    w2 = QtWidgets.QWidget()
    dock_area.addTab(w1, "Keep")
    dock_area.addTab(w2, "Hide")

    assert dock_area.count() == 2
    assert dock_area.visibleCount() == 2

    dock_area._request_close_tab(1)

    assert dock_area.count() == 2
    assert dock_area.visibleCount() == 1
    assert dock_area.hiddenIndexes() == [1]
    assert dock_area.widget(1) is w2
    assert w2.parentWidget() is dock_area
    assert dock_area.showTab(1) is True
    assert dock_area.visibleCount() == 2


def test_dock_area_is_inside_client_content(qtbot):
    """Test is_inside_client_content correctly identifies clicks inside client widgets."""
    dock_area = DockArea()
    qtbot.addWidget(dock_area)

    w1 = QtWidgets.QWidget()
    child_btn = QtWidgets.QPushButton("Click Me", w1)
    layout = QtWidgets.QVBoxLayout(w1)
    layout.addWidget(child_btn)

    dock_area.addTab(w1, "Tab 1")
    dock_area.show()

    # Wait for the widget to be visible/exposed on screen
    qtbot.waitExposed(dock_area)

    # Get the global position of the center of child_btn
    global_pos = child_btn.mapToGlobal(child_btn.rect().center())

    # It should be identified as inside client content
    assert dock_area.is_inside_client_content(global_pos) is True

    # A position outside should be False
    outside_pos = QtCore.QPoint(-1000, -1000)
    assert dock_area.is_inside_client_content(outside_pos) is False

