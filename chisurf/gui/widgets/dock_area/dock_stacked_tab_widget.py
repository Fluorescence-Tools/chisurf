from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.dock_area.dock_stacked_tab_bar import DockStackedTabBar


class DockStackedTabWidget(QtWidgets.QWidget):
    """A tab widget that uses a multi-row stacked tab bar.

    This widget exposes the same public interface as
    :class:`~chisurf.gui.widgets.dock_area.dock_area.DockTabWidget`
    so that :class:`~chisurf.gui.widgets.dock_area.dock_area.DockArea`
    can use it interchangeably.

    Signals
    -------
    currentChanged : QtCore.Signal(int)
        Emitted when the active tab index changes.
    tabCloseRequested : QtCore.Signal(int)
        Emitted when a tab close button is clicked.
    """

    currentChanged = QtCore.Signal(int)
    tabCloseRequested = QtCore.Signal(int)

    def __init__(
        self,
        dock_area: QtWidgets.QWidget,
        parent: QtWidgets.QWidget = None,
    ):
        """Initialize the stacked tab widget.

        Parameters
        ----------
        dock_area : DockArea
            The owning dock area.
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.dock_area = dock_area
        self._corner_widgets: dict[QtCore.Qt.Corner, QtWidgets.QWidget] = {}
        self._new_tab_btn: QtWidgets.QToolButton | None = None

        self.setAcceptDrops(True)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        # Top bar with optional corner widgets and the stacked tab bar.
        self._top_bar = QtWidgets.QWidget(self)
        self._top_bar.setAutoFillBackground(False)
        self._top_bar.setAcceptDrops(False)  # Let parent handle drops
        top_layout = QtWidgets.QHBoxLayout(self._top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        self._left_corner = QtWidgets.QWidget(self._top_bar)
        self._left_corner.setAcceptDrops(False)
        self._left_corner_layout = QtWidgets.QHBoxLayout(self._left_corner)
        self._left_corner_layout.setContentsMargins(0, 0, 0, 0)
        self._left_corner_layout.setSpacing(0)
        self._left_corner_layout.addStretch()

        self._right_corner = QtWidgets.QWidget(self._top_bar)
        self._right_corner.setAcceptDrops(False)
        self._right_corner_layout = QtWidgets.QHBoxLayout(self._right_corner)
        self._right_corner_layout.setContentsMargins(0, 0, 0, 0)
        self._right_corner_layout.setSpacing(0)
        self._right_corner_layout.addStretch()

        self._tab_bar = DockStackedTabBar(self)

        top_layout.addWidget(self._left_corner)
        top_layout.addWidget(self._tab_bar, stretch=1)
        top_layout.addWidget(self._right_corner)

        # Content area.
        self._stacked_widget = QtWidgets.QStackedWidget(self)
        self._stacked_widget.setAutoFillBackground(False)
        self._stacked_widget.setAcceptDrops(False)  # Let parent handle drops
        transparent = QtGui.QColor(0, 0, 0, 0)
        pal = self._stacked_widget.palette()
        pal.setColor(QtGui.QPalette.Window, transparent)
        pal.setColor(QtGui.QPalette.Base, transparent)
        self._stacked_widget.setPalette(pal)

        # Main layout.
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._top_bar)
        layout.addWidget(self._stacked_widget, stretch=1)

        # Connections.
        self._tab_bar.currentChanged.connect(self._on_tab_bar_current_changed)
        self._tab_bar.doubleClickedTab.connect(self._on_tab_double_clicked)
        self._tab_bar.doubleClickedTabBar.connect(self._on_tab_bar_double_clicked)
        self._tab_bar.contextMenuRequested.connect(self._on_tab_context_menu)
        self._tab_bar.tabCloseRequested.connect(self.tabCloseRequested.emit)
        self._stacked_widget.currentChanged.connect(self._on_stacked_current_changed)

    def count(self) -> int:
        """Return the number of tabs."""
        return self._stacked_widget.count()

    def widget(self, index: int) -> QtWidgets.QWidget | None:
        """Return the page widget at ``index``."""
        if 0 <= index < self.count():
            return self._stacked_widget.widget(index)
        return None

    def addTab(self, widget: QtWidgets.QWidget, text: str) -> int:
        """Add ``widget`` as a new tab with ``text``."""
        idx = self._stacked_widget.addWidget(widget)
        self._tab_bar.addTab(text)
        return idx

    def insertTab(
        self, index: int, widget: QtWidgets.QWidget, text: str
    ) -> int:
        """Insert ``widget`` as a tab at ``index`` with ``text``."""
        idx = self._stacked_widget.insertWidget(index, widget)
        self._tab_bar.insertTab(index, text)
        return idx

    def removeTab(self, index: int) -> None:
        """Remove the tab at ``index`` without destroying its page widget."""
        page = self._stacked_widget.widget(index)
        if page is not None:
            self._stacked_widget.removeWidget(page)
        self._tab_bar.removeTab(index)

    def currentIndex(self) -> int:
        """Return the index of the currently active tab."""
        return self._stacked_widget.currentIndex()

    def setCurrentIndex(self, index: int) -> None:
        """Activate the tab at ``index``."""
        self._stacked_widget.setCurrentIndex(index)

    def currentWidget(self) -> QtWidgets.QWidget | None:
        """Return the currently active page widget."""
        return self._stacked_widget.currentWidget()

    def setCurrentWidget(self, widget: QtWidgets.QWidget) -> None:
        """Activate the tab containing ``widget``."""
        self._stacked_widget.setCurrentWidget(widget)

    def tabText(self, index: int) -> str:
        """Return the text of the tab at ``index``."""
        return self._tab_bar.tabText(index)

    def setTabText(self, index: int, text: str) -> None:
        """Set the text of the tab at ``index``."""
        self._tab_bar.setTabText(index, text)

    def indexOf(self, widget: QtWidgets.QWidget) -> int:
        """Return the index of ``widget``."""
        return self._stacked_widget.indexOf(widget)

    def tabBar(self) -> DockStackedTabBar:
        """Return the stacked tab bar."""
        return self._tab_bar

    def setTabsClosable(self, closable: bool) -> None:
        """Show or hide tab close buttons."""
        self._tab_bar.setTabsClosable(closable)

    def setCornerWidget(
        self,
        widget: QtWidgets.QWidget,
        corner: QtCore.Qt.Corner = QtCore.Qt.TopRightCorner,
    ) -> None:
        """Set the widget in the given corner of the tab bar.

        Parameters
        ----------
        widget : QWidget
            The corner widget to place.
        corner : Qt.Corner, optional
            The corner where the widget is placed.
        """
        old = self._corner_widgets.get(corner)
        if old is not None and old is not widget:
            old.setParent(None)
            old.deleteLater()
        self._corner_widgets[corner] = widget
        if widget is None:
            return
        if corner == QtCore.Qt.TopLeftCorner:
            self._left_corner_layout.insertWidget(0, widget)
        else:
            self._right_corner_layout.insertWidget(0, widget)

    def cornerWidget(
        self, corner: QtCore.Qt.Corner = QtCore.Qt.TopRightCorner
    ) -> QtWidgets.QWidget | None:
        """Return the widget in ``corner``."""
        return self._corner_widgets.get(corner)

    def setNewTabButtonVisible(self, visible: bool = True) -> None:
        """Show or hide the local '+' new-tab button in the left corner."""
        if visible:
            if self._new_tab_btn is None:
                self._new_tab_btn = QtWidgets.QToolButton(self)
                self._new_tab_btn.setText("+")
                self._new_tab_btn.setAutoRaise(True)
                if self.dock_area is not None:
                    self._new_tab_btn.clicked.connect(
                        self.dock_area.newTabRequested.emit
                    )
                self.setCornerWidget(self._new_tab_btn, QtCore.Qt.TopLeftCorner)
            self._new_tab_btn.show()
            return
        if self._new_tab_btn is not None:
            self._new_tab_btn.hide()

    def setDocumentMode(self, enabled: bool) -> None:
        """No-op for API compatibility; stacked mode is always document style."""
        pass

    def setTabToolTip(self, index: int, tooltip: str) -> None:
        """Set the tooltip of the tab at ``index``."""
        self._tab_bar.setTabToolTip(index, tooltip)

    def tabToolTip(self, index: int) -> str:
        """Return the tooltip of the tab at ``index``."""
        return self._tab_bar.tabToolTip(index)

    def _on_tab_bar_current_changed(self, index: int) -> None:
        """Synchronize the stacked widget with the tab bar selection."""
        if index != self._stacked_widget.currentIndex():
            self._stacked_widget.setCurrentIndex(index)
        if self.dock_area is not None:
            self.dock_area.set_active_tab_widget(self)

    def _on_stacked_current_changed(self, index: int) -> None:
        """Synchronize the tab bar and notify the dock area on content changes."""
        self._tab_bar.setCurrentIndex(index)
        self.currentChanged.emit(index)
        if self.dock_area is not None:
            self.dock_area.set_active_tab_widget(self)

    def _on_tab_double_clicked(self, index: int) -> None:
        """Restore a single tab to the main tab group."""
        if self.dock_area is not None:
            self.dock_area.restore_tab(self, index)

    def _on_tab_bar_double_clicked(self) -> None:
        """Restore all tabs to the main tab group."""
        if self.dock_area is not None:
            self.dock_area.restore_all_tabs(self)

    def _on_tab_context_menu(
        self, local_index: int, global_pos: QtCore.QPoint
    ) -> None:
        """Forward tab context menu requests to the dock area."""
        if self.dock_area is not None:
            self.dock_area._on_tab_context_menu(self, local_index, global_pos)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept dragging of dock tabs."""
        if event.mimeData().hasFormat("application/x-chisurf-dock-tab"):
            event.acceptProposedAction()
            self.dock_area.update_overlay(self, event.pos())

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        """Update overlay layout when dragging moves."""
        if event.mimeData().hasFormat("application/x-chisurf-dock-tab"):
            event.acceptProposedAction()
            self.dock_area.update_overlay(self, event.pos())

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent) -> None:
        """Hide overlay on drag leaving widget."""
        self.dock_area.hide_overlay()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Handle dropping of a dock tab."""
        if event.mimeData().hasFormat("application/x-chisurf-dock-tab"):
            event.acceptProposedAction()
            self.dock_area.handle_drop(self, event.pos(), event.mimeData())

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Show dock-area context actions when right-clicking the tab pane."""
        if self.dock_area is not None and self.dock_area.is_inside_client_content(event.globalPos()):
            event.accept()
            return

        if self.dock_area._show_area_context_menu(event.globalPos()):
            event.accept()
            return
        super().contextMenuEvent(event)
