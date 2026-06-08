from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.dock_area.dock_overlay import DockDropOverlay
from chisurf.gui.widgets.dock_area.dock_tab_bar import DockTabBar


class DockSplitter(QtWidgets.QSplitter):
    """Custom QSplitter with styled splitter handle."""

    def __init__(self, orientation: QtCore.Qt.Orientation, parent: QtWidgets.QWidget = None):
        """Initialize the custom splitter.

        Parameters
        ----------
        orientation : Qt.Orientation
            The splitter orientation (Horizontal or Vertical).
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(orientation, parent)
        self.setChildrenCollapsible(False)
        self.setHandleWidth(4)
        self.setStyleSheet("""
            QSplitter::handle {
                background-color: rgba(128, 128, 128, 60);
            }
            QSplitter::handle:hover {
                background-color: rgba(0, 150, 255, 150);
            }
        """)


class DockTabWidget(QtWidgets.QTabWidget):
    """Custom QTabWidget that supports dragging tabs and drop actions."""

    def __init__(self, dock_area: QtWidgets.QWidget, parent: QtWidgets.QWidget = None):
        """Initialize the custom tab widget.

        Parameters
        ----------
        dock_area : DockArea
            The parent DockArea widget.
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.dock_area = dock_area
        self.tab_bar = DockTabBar(self)
        self.setTabBar(self.tab_bar)
        self.setAcceptDrops(True)

        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(128, 128, 128, 40);
                background-color: transparent;
            }
            QTabBar::tab {
                background-color: rgba(128, 128, 128, 30);
                padding: 1px 10px;
                border: 1px solid rgba(128, 128, 128, 30);
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background-color: rgba(0, 150, 255, 40);
                border-color: rgba(0, 150, 255, 100);
            }
            QTabBar::tab:hover:!selected {
                background-color: rgba(128, 128, 128, 50);
            }
        """)

        self.currentChanged.connect(self._on_current_changed)
        self.tab_bar.doubleClickedTab.connect(self._on_tab_double_clicked)
        self.tab_bar.doubleClickedTabBar.connect(self._on_tab_bar_double_clicked)

    def _on_current_changed(self, index: int) -> None:
        if index >= 0:
            self.dock_area.set_active_tab_widget(self)

    def _on_tab_double_clicked(self, index: int) -> None:
        self.dock_area.restore_tab(self, index)

    def _on_tab_bar_double_clicked(self) -> None:
        self.dock_area.restore_all_tabs(self)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept dragging of dock tabs.

        Parameters
        ----------
        event : QDragEnterEvent
            The drag enter event.
        """
        if event.mimeData().hasFormat("application/x-chisurf-dock-tab"):
            event.acceptProposedAction()
            self.dock_area.update_overlay(self, event.pos())

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        """Update overlay layout when dragging moves.

        Parameters
        ----------
        event : QDragMoveEvent
            The drag move event.
        """
        if event.mimeData().hasFormat("application/x-chisurf-dock-tab"):
            event.acceptProposedAction()
            self.dock_area.update_overlay(self, event.pos())

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent) -> None:
        """Hide overlay on drag leaving widget.

        Parameters
        ----------
        event : QDragLeaveEvent
            The drag leave event.
        """
        self.dock_area.hide_overlay()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Handle dropping of a dock tab.

        Parameters
        ----------
        event : QDropEvent
            The drop event.
        """
        if event.mimeData().hasFormat("application/x-chisurf-dock-tab"):
            event.acceptProposedAction()
            self.dock_area.handle_drop(self, event.pos(), event.mimeData())


class DockArea(QtWidgets.QWidget):
    """A custom layout area for organizing plot widgets in tabbed and split panels.

    This replaces QTabWidget in the FitSubWindow, allowing users to drag tabs
    to split views in left, right, top, or bottom regions.

    Signals
    -------
    currentChanged : QtCore.Signal(int)
        Emitted when the active plot index changes.
    """

    currentChanged = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget = None):
        """Initialize the DockArea.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self._all_widgets = []
        self._root_widget = None
        self._active_tab_widget = None
        self._last_emitted_index = -1

        # Setup main layout
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Add overlay child
        self._overlay = DockDropOverlay(self)

        # Monitor focus changes to track the active plot/tab widget
        QtWidgets.QApplication.instance().focusChanged.connect(self._on_focus_changed)
        self.destroyed.connect(self._cleanup)

    def _cleanup(self) -> None:
        try:
            QtWidgets.QApplication.instance().focusChanged.disconnect(self._on_focus_changed)
        except Exception:
            pass

    def _on_focus_changed(self, old: QtWidgets.QWidget, now: QtWidgets.QWidget) -> None:
        if now is None:
            return
        p = now
        while p is not None:
            if isinstance(p, DockTabWidget) and p.dock_area == self:
                self.set_active_tab_widget(p)
                break
            p = p.parentWidget()

    def set_root_widget(self, widget: QtWidgets.QWidget) -> None:
        """Set the root widget of the dock area.

        Parameters
        ----------
        widget : QWidget
            The new root widget (either a DockTabWidget or DockSplitter).
        """
        if self._root_widget is not None:
            self._layout.removeWidget(self._root_widget)
        self._root_widget = widget
        if widget is not None:
            self._layout.addWidget(widget)

    def set_active_tab_widget(self, tw: DockTabWidget) -> None:
        """Set the active tab widget and emit currentChanged if the active tab index changes.

        Parameters
        ----------
        tw : DockTabWidget
            The tab widget that is currently active/focused.
        """
        self._active_tab_widget = tw
        idx = self.currentIndex()
        if idx != self._last_emitted_index:
            self._last_emitted_index = idx
            self.currentChanged.emit(idx)

    def active_tab_widget(self) -> DockTabWidget:
        """Return the currently active tab widget.

        Returns
        -------
        DockTabWidget or None
        """
        if self._active_tab_widget is not None:
            return self._active_tab_widget
        return self.find_main_tab_widget()

    def find_main_tab_widget(self, exclude_tw: DockTabWidget = None) -> DockTabWidget:
        """Find the primary tab widget (typically the first in the hierarchy).

        Parameters
        ----------
        exclude_tw : DockTabWidget, optional
            A tab widget to exclude from the search.

        Returns
        -------
        DockTabWidget or None
        """
        tws = self.findChildren(DockTabWidget)
        for tw in tws:
            if tw != exclude_tw:
                return tw
        return None

    def addTab(self, widget: QtWidgets.QWidget, name: str) -> None:
        """Add a tab with the given widget and name.

        Parameters
        ----------
        widget : QWidget
            The page widget to add.
        name : str
            The name to display on the tab.
        """
        self._all_widgets.append(widget)
        if self._root_widget is None:
            tab_widget = DockTabWidget(self)
            self.set_root_widget(tab_widget)
            tab_widget.addTab(widget, name)
            self.set_active_tab_widget(tab_widget)
        else:
            main_tw = self.find_main_tab_widget()
            if main_tw is not None:
                main_tw.addTab(widget, name)

    def add_panel(self, widget: QtWidgets.QWidget, name: str) -> None:
        """Alternative name for addTab.

        Parameters
        ----------
        widget : QWidget
            The page widget to add.
        name : str
            The name to display on the tab.
        """
        self.addTab(widget, name)

    def currentIndex(self) -> int:
        """Get the absolute index of the currently active tab/plot.

        Returns
        -------
        int
            The absolute index matching the order tabs were added.
        """
        active_tw = self.active_tab_widget()
        if active_tw is None:
            return -1
        current_widget = active_tw.currentWidget()
        if current_widget in self._all_widgets:
            return self._all_widgets.index(current_widget)
        return -1

    def currentWidget(self) -> QtWidgets.QWidget:
        """Get the currently active page widget.

        Returns
        -------
        QWidget or None
        """
        active_tw = self.active_tab_widget()
        if active_tw is not None:
            return active_tw.currentWidget()
        return None

    def update(self, *args, **kwargs) -> None:
        """Repaint the dock area and trigger updates on all child widgets."""
        super().update(*args, **kwargs)
        for child in self.findChildren(QtWidgets.QWidget):
            child.update(*args, **kwargs)

    def update_overlay(self, target_tw: DockTabWidget, local_pos: QtCore.QPoint) -> None:
        """Calculate the active drop zone and update the visual overlay.

        Parameters
        ----------
        target_tw : DockTabWidget
            The tab widget the cursor is hovering over.
        local_pos : QPoint
            The hover position relative to the target tab widget.
        """
        rect = target_tw.rect()
        zone, zone_rect = self.get_dock_zone(rect, local_pos)

        # Map target_tw's relative zone_rect to DockArea coordinates
        top_left = target_tw.mapTo(self, zone_rect.topLeft())
        global_zone_rect = QtCore.QRect(top_left, zone_rect.size())

        self._overlay.set_highlight(global_zone_rect)

    def hide_overlay(self) -> None:
        """Hide the drop zone visual overlay."""
        self._overlay.hide()

    def get_dock_zone(self, rect: QtCore.QRect, pos: QtCore.QPoint) -> tuple[str, QtCore.QRect]:
        """Calculate the active dock zone based on geometry and position.

        Parameters
        ----------
        rect : QRect
            The bounding rectangle of the target widget.
        pos : QPoint
            The cursor position relative to the target widget.

        Returns
        -------
        tuple (str, QRect)
            The zone name ('left', 'right', 'top', 'bottom', 'center') and
            its relative bounding rectangle.
        """
        w = rect.width()
        h = rect.height()
        x = pos.x()
        y = pos.y()

        margin_w = w * 0.25
        margin_h = h * 0.25

        if x < margin_w:
            return "left", QtCore.QRect(0, 0, int(w * 0.5), h)
        elif x > w - margin_w:
            return "right", QtCore.QRect(int(w * 0.5), 0, int(w * 0.5), h)
        elif y < margin_h:
            return "top", QtCore.QRect(0, 0, w, int(h * 0.5))
        elif y > h - margin_h:
            return "bottom", QtCore.QRect(0, int(h * 0.5), w, int(h * 0.5))
        else:
            return "center", rect

    def handle_drop(
        self, target_tw: DockTabWidget, pos: QtCore.QPoint, mime_data: QtCore.QMimeData
    ) -> None:
        """Process a completed drop action.

        Parameters
        ----------
        target_tw : DockTabWidget
            The tab widget where the tab was dropped.
        pos : QPoint
            The drop position relative to target_tw.
        mime_data : QMimeData
            The drag-and-drop MIME payload containing source information.
        """
        self.hide_overlay()

        source_tw = getattr(mime_data, "source_widget", None)
        source_idx = getattr(mime_data, "source_index", -1)
        if source_tw is None or source_idx < 0:
            return

        zone, _ = self.get_dock_zone(target_tw.rect(), pos)

        widget = source_tw.widget(source_idx)
        title = source_tw.tabText(source_idx)

        if zone == "center":
            if source_tw == target_tw:
                # Reorder tab inside the same widget
                insert_idx = target_tw.tabBar().tabAt(pos)
                source_tw.removeTab(source_idx)
                if insert_idx < 0:
                    target_tw.addTab(widget, title)
                    target_tw.setCurrentWidget(widget)
                else:
                    target_tw.insertTab(insert_idx, widget, title)
                    target_tw.setCurrentIndex(insert_idx)
            else:
                # Move tab to target tab widget
                source_tw.removeTab(source_idx)
                insert_idx = target_tw.tabBar().tabAt(pos)
                if insert_idx < 0:
                    target_tw.addTab(widget, title)
                    target_tw.setCurrentWidget(widget)
                else:
                    target_tw.insertTab(insert_idx, widget, title)
                    target_tw.setCurrentIndex(insert_idx)
                self.cleanup_empty_tab_widget(source_tw)
        else:
            # Split drop
            source_tw.removeTab(source_idx)
            new_tw = DockTabWidget(self)
            new_tw.addTab(widget, title)
            self.split_tab_widget(target_tw, new_tw, zone)
            self.cleanup_empty_tab_widget(source_tw)
            self.set_active_tab_widget(new_tw)

    def split_tab_widget(self, target_tw: DockTabWidget, new_tw: DockTabWidget, zone: str) -> None:
        """Split a tab widget by wrapping them in a splitter.

        Parameters
        ----------
        target_tw : DockTabWidget
            The existing tab widget to be split.
        new_tw : DockTabWidget
            The new tab widget to be placed.
        zone : str
            The split zone ('left', 'right', 'top', 'bottom').
        """
        parent = target_tw.parentWidget()
        orientation = QtCore.Qt.Horizontal if zone in ("left", "right") else QtCore.Qt.Vertical
        splitter = DockSplitter(orientation, parent)

        self.replace_widget(target_tw, splitter)

        if zone in ("left", "top"):
            splitter.addWidget(new_tw)
            splitter.addWidget(target_tw)
        else:
            splitter.addWidget(target_tw)
            splitter.addWidget(new_tw)

        splitter.setSizes([100, 100])

    def replace_widget(self, old_widget: QtWidgets.QWidget, new_widget: QtWidgets.QWidget) -> None:
        """Replace a widget in the dock hierarchy.

        Parameters
        ----------
        old_widget : QWidget
            The widget to be replaced.
        new_widget : QWidget
            The new widget to insert.
        """
        parent = old_widget.parentWidget()
        if isinstance(parent, QtWidgets.QSplitter):
            index = parent.indexOf(old_widget)
            parent.insertWidget(index, new_widget)
        elif parent == self:
            self.set_root_widget(new_widget)

    def cleanup_empty_tab_widget(self, tw: DockTabWidget) -> None:
        """Remove empty tab widgets and simplify splitters.

        Parameters
        ----------
        tw : DockTabWidget
            The tab widget that became empty.
        """
        if tw.count() > 0:
            return

        parent = tw.parentWidget()
        if parent == self:
            return

        if isinstance(parent, QtWidgets.QSplitter):
            sibling = None
            for i in range(parent.count()):
                w = parent.widget(i)
                if w != tw:
                    sibling = w
                    break

            if sibling is not None:
                self.replace_widget(parent, sibling)
                tw.setParent(None)
                tw.deleteLater()
                parent.setParent(None)
                parent.deleteLater()

    def restore_tab(self, tw: DockTabWidget, index: int) -> None:
        """Move a specific tab back to the main/primary tab group.

        Parameters
        ----------
        tw : DockTabWidget
            The source tab widget.
        index : int
            The index of the tab within tw.
        """
        main_tw = self.find_main_tab_widget(exclude_tw=tw)
        if main_tw is None:
            return

        widget = tw.widget(index)
        title = tw.tabText(index)
        tw.removeTab(index)

        main_tw.addTab(widget, title)
        main_tw.setCurrentWidget(widget)

        self.cleanup_empty_tab_widget(tw)
        self.set_active_tab_widget(main_tw)

    def restore_all_tabs(self, tw: DockTabWidget) -> None:
        """Move all tabs in a widget back to the main/primary tab group.

        Parameters
        ----------
        tw : DockTabWidget
            The source tab widget.
        """
        main_tw = self.find_main_tab_widget(exclude_tw=tw)
        if main_tw is None:
            return

        while tw.count() > 0:
            widget = tw.widget(0)
            title = tw.tabText(0)
            tw.removeTab(0)
            main_tw.addTab(widget, title)

        self.cleanup_empty_tab_widget(tw)
        self.set_active_tab_widget(main_tw)
