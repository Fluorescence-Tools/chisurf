from qtpy import QtCore, QtGui, QtWidgets


class DockTabBar(QtWidgets.QTabBar):
    """Custom QTabBar that supports dragging tabs for splitting and emitting double-click signals.

    Signals
    -------
    doubleClickedTab : QtCore.Signal(int)
        Emitted when a tab at the given index is double clicked.
    doubleClickedTabBar : QtCore.Signal()
        Emitted when empty area of the tab bar is double clicked.
    contextMenuRequested : QtCore.Signal(int, QtCore.QPoint)
        Emitted when a tab is right-clicked; carries the tab index and global position.
    """

    doubleClickedTab = QtCore.Signal(int)
    doubleClickedTabBar = QtCore.Signal()
    contextMenuRequested = QtCore.Signal(int, QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget = None):
        """Initialize the custom dock tab bar.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._drag_start_pos = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Record the initial mouse position when left button is pressed.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event.
        """
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Initiate drag and drop if mouse dragged past the minimum distance threshold.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event.
        """
        if not (event.buttons() & QtCore.Qt.LeftButton) or self._drag_start_pos is None:
            super().mouseMoveEvent(event)
            return

        if (
            event.pos() - self._drag_start_pos
        ).manhattanLength() < QtWidgets.QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return

        index = self.tabAt(self._drag_start_pos)
        if index < 0:
            super().mouseMoveEvent(event)
            return

        # Check if we can drag this tab (must not be the only tab in DockArea)
        p = self.parent()
        from chisurf.gui.widgets.dock_area.dock_area import DockArea, DockTabWidget

        while p is not None and not isinstance(p, DockArea):
            p = p.parent()
        dock_area = p

        if dock_area is not None:
            total_tabs = 0
            for tw in dock_area.findChildren(DockTabWidget):
                total_tabs += tw.count()
            if total_tabs <= 1:
                super().mouseMoveEvent(event)
                return

        drag = QtGui.QDrag(self)
        mime_data = QtCore.QMimeData()

        # Store source widget reference and tab index
        source_tab_widget = self.parent()
        mime_data.setData(
            "application/x-chisurf-dock-tab", QtCore.QByteArray(str(id(source_tab_widget)).encode())
        )
        mime_data.source_widget = source_tab_widget
        mime_data.source_index = index

        drag.setMimeData(mime_data)

        tab_rect = self.tabRect(index)
        pixmap = self.grab(tab_rect)
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos() - tab_rect.topLeft())

        drag.exec_(QtCore.Qt.MoveAction)

        if dock_area is not None:
            dock_area.hide_overlay()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Emit custom signals on double click events.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event.
        """
        index = self.tabAt(event.pos())
        if index >= 0:
            self.doubleClickedTab.emit(index)
        else:
            self.doubleClickedTabBar.emit()
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Emit contextMenuRequested when right-clicking on a tab.

        Parameters
        ----------
        event : QContextMenuEvent
            The context menu event.
        """
        index = self.tabAt(event.pos())
        if index >= 0:
            self.contextMenuRequested.emit(index, event.globalPos())
        super().contextMenuEvent(event)
