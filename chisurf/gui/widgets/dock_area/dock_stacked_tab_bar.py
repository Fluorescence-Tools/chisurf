from qtpy import QtCore, QtGui, QtWidgets


class FlowLayout(QtWidgets.QLayout):
    """A simple flow layout that wraps widgets into multiple rows.

    This is used by :class:`DockStackedTabBar` to stack tab buttons in rows
    when they do not fit horizontally.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget = None,
        margin: int = 0,
        h_spacing: int = 2,
        v_spacing: int = 2,
    ):
        """Initialize the flow layout.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        margin : int, optional
            Content margin on all sides.
        h_spacing : int, optional
            Horizontal spacing between items.
        v_spacing : int, optional
            Vertical spacing between rows.
        """
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self._item_list: list[QtWidgets.QLayoutItem] = []

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        """Add a layout item."""
        self._item_list.append(item)

    def count(self) -> int:
        """Return the number of layout items."""
        return len(self._item_list)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        """Return the layout item at ``index``."""
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        """Remove and return the layout item at ``index``."""
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self):
        """Return expanding directions (none for a flow layout)."""
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        """Return True because height depends on width."""
        return True

    def heightForWidth(self, width: int) -> int:
        """Return the required height for the given width."""
        return self._do_layout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        """Lay out items inside ``rect``."""
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self) -> QtCore.QSize:
        """Return the layout size hint."""
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        """Return the minimum size."""
        size = QtCore.QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QtCore.QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom(),
        )
        return size

    def insertWidget(
        self, index: int, widget: QtWidgets.QWidget
    ) -> int:
        """Insert ``widget`` at ``index``.

        Parameters
        ----------
        index : int
            Insertion index. Values outside the valid range are clamped.
        widget : QWidget
            The widget to insert.

        Returns
        -------
        int
            The index where the widget was inserted.
        """
        if widget is None:
            return -1
        self.addChildWidget(widget)
        item = QtWidgets.QWidgetItem(widget)
        if index < 0 or index > len(self._item_list):
            index = len(self._item_list)
        self._item_list.insert(index, item)
        self.invalidate()
        return index

    def removeWidget(self, widget: QtWidgets.QWidget) -> None:
        """Remove ``widget`` from the layout if present."""
        for i, item in enumerate(self._item_list):
            if item.widget() is widget:
                self._item_list.pop(i)
                self.invalidate()
                return

    def _do_layout(self, rect: QtCore.QRect, test_only: bool) -> int:
        """Compute or apply the flow layout geometry."""
        left, top, right, bottom = self.getContentsMargins()
        effective = rect.adjusted(left, top, -right, -bottom)
        x = effective.x()
        y = effective.y()
        line_height = 0
        for item in self._item_list:
            next_x = x + item.sizeHint().width() + self._h_spacing
            if next_x - self._h_spacing > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + self._v_spacing
                next_x = x + item.sizeHint().width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(
                    QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint())
                )
            x = next_x
            line_height = max(line_height, item.sizeHint().height())
        return y + line_height - rect.y() + bottom


class DockStackedTabItem(QtWidgets.QWidget):
    """A single tab button rendered as a regular tab in a stacked tab bar.

    Signals
    -------
    clicked : QtCore.Signal()
        Emitted when the tab is clicked (but not dragged).
    doubleClicked : QtCore.Signal()
        Emitted when the tab is double clicked.
    closeRequested : QtCore.Signal()
        Emitted when the close button is clicked.
    contextMenuRequested : QtCore.Signal(QtCore.QPoint)
        Emitted when the tab is right-clicked.
    dragStarted : QtCore.Signal(QtCore.QPoint)
        Emitted when a drag gesture starts. Carries the drag start position
        in item-local coordinates.
    """

    clicked = QtCore.Signal()
    doubleClicked = QtCore.Signal()
    closeRequested = QtCore.Signal()
    contextMenuRequested = QtCore.Signal(QtCore.QPoint)
    dragStarted = QtCore.Signal(QtCore.QPoint)

    _BASE_STYLE = """
        DockStackedTabItem {
            background-color: rgba(128, 128, 128, 30);
            border: 1px solid rgba(128, 128, 128, 30);
            border-bottom: none;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
        }
        DockStackedTabItem[selected="true"] {
            background-color: rgba(0, 150, 255, 40);
            border-color: rgba(0, 150, 255, 100);
        }
        DockStackedTabItem:hover[selected="false"] {
            background-color: rgba(128, 128, 128, 50);
        }
        DockStackedTabItem QLabel {
            background: transparent;
            border: none;
            padding: 2px 6px;
            color: palette(window-text);
        }
        DockStackedTabItem QToolButton {
            background: transparent;
            border: none;
            color: palette(window-text);
            font-weight: bold;
            padding: 0px 4px;
        }
        DockStackedTabItem QToolButton:hover {
            background-color: rgba(255, 80, 80, 120);
            color: white;
            border-radius: 2px;
        }
    """

    def __init__(
        self,
        text: str,
        closable: bool = False,
        parent: QtWidgets.QWidget = None,
    ):
        """Initialize a stacked tab item.

        Parameters
        ----------
        text : str
            The tab text.
        closable : bool, optional
            Whether the tab shows a close button.
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setProperty("selected", False)
        self.setStyleSheet(self._BASE_STYLE)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedHeight(22)

        self._text = text
        self._drag_start_pos: QtCore.QPoint | None = None
        self._dragging = False

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(2)

        self._label = QtWidgets.QLabel(text, self)
        layout.addWidget(self._label)

        self._close_btn: QtWidgets.QToolButton | None = None
        if closable:
            self._close_btn = QtWidgets.QToolButton(self)
            self._close_btn.setText("\u00d7")
            self._close_btn.setAutoRaise(True)
            self._close_btn.setFixedSize(16, 16)
            self._close_btn.setToolTip("Close tab")
            self._close_btn.clicked.connect(self.closeRequested.emit)
            layout.addWidget(self._close_btn)

    def text(self) -> str:
        """Return the tab text."""
        return self._text

    def setText(self, text: str) -> None:
        """Set the tab text."""
        self._text = text
        self._label.setText(text)

    def setChecked(self, checked: bool) -> None:
        """Set whether this tab is the selected tab."""
        self.setProperty("selected", checked)
        self.style().unpolish(self)
        self.style().polish(self)

    def isChecked(self) -> bool:
        """Return whether this tab is selected."""
        return bool(self.property("selected"))

    def setClosable(self, closable: bool) -> None:
        """Show or hide the close button."""
        if closable and self._close_btn is None:
            layout = self.layout()
            self._close_btn = QtWidgets.QToolButton(self)
            self._close_btn.setText("\u00d7")
            self._close_btn.setAutoRaise(True)
            self._close_btn.setFixedSize(16, 16)
            self._close_btn.setToolTip("Close tab")
            self._close_btn.clicked.connect(self.closeRequested.emit)
            layout.addWidget(self._close_btn)
        elif not closable and self._close_btn is not None:
            self._close_btn.deleteLater()
            self._close_btn = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Record the drag start position on left press."""
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_start_pos = event.pos()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Start a drag if the mouse is moved past the drag threshold."""
        if not (event.buttons() & QtCore.Qt.LeftButton) or self._drag_start_pos is None:
            super().mouseMoveEvent(event)
            return
        if (
            event.pos() - self._drag_start_pos
        ).manhattanLength() < QtWidgets.QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return
        self._dragging = True
        start_pos = self._drag_start_pos
        self._drag_start_pos = None
        self.dragStarted.emit(start_pos)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Emit ``clicked`` if the gesture was not a drag."""
        if event.button() == QtCore.Qt.LeftButton and not self._dragging:
            self.clicked.emit()
        self._dragging = False
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Emit ``doubleClicked`` on left double click."""
        if event.button() == QtCore.Qt.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Emit ``contextMenuRequested`` on right click."""
        if event.reason() != QtGui.QContextMenuEvent.Mouse:
            super().contextMenuEvent(event)
            return
        self.contextMenuRequested.emit(event.globalPos())
        super().contextMenuEvent(event)


class DockStackedTabBar(QtWidgets.QWidget):
    """A tab bar that arranges tabs in multiple rows when they overflow.

    This mirrors enough of :class:`~chisurf.gui.widgets.dock_area.dock_tab_bar.DockTabBar`
    so that :class:`DockStackedTabWidget` can be used as a drop-in replacement
    for :class:`~chisurf.gui.widgets.dock_area.dock_area.DockTabWidget` inside
    :class:`~chisurf.gui.widgets.dock_area.dock_area.DockArea`.

    Signals
    -------
    currentChanged : QtCore.Signal(int)
        Emitted when the active tab index changes.
    tabCloseRequested : QtCore.Signal(int)
        Emitted when a tab close button is clicked.
    doubleClickedTab : QtCore.Signal(int)
        Emitted when a tab is double clicked.
    doubleClickedTabBar : QtCore.Signal()
        Emitted when the empty area of the bar is double clicked.
    contextMenuRequested : QtCore.Signal(int, QtCore.QPoint)
        Emitted when a tab is right-clicked.
    """

    currentChanged = QtCore.Signal(int)
    tabCloseRequested = QtCore.Signal(int)
    doubleClickedTab = QtCore.Signal(int)
    doubleClickedTabBar = QtCore.Signal()
    contextMenuRequested = QtCore.Signal(int, QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget = None):
        """Initialize the stacked tab bar.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget, expected to be a ``DockStackedTabWidget``.
        """
        super().__init__(parent)
        self._items: list[DockStackedTabItem] = []
        self._current_index = -1
        self._closable = False
        self._drag_start_index = -1

        self._flow_layout = FlowLayout(self, margin=0, h_spacing=2, v_spacing=2)
        self.setLayout(self._flow_layout)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )

    def count(self) -> int:
        """Return the number of tabs."""
        return len(self._items)

    def addTab(self, text: str) -> int:
        """Add a tab with the given text and return its index."""
        return self.insertTab(self.count(), text)

    def insertTab(self, index: int, text: str) -> int:
        """Insert a tab with ``text`` at ``index``."""
        item = DockStackedTabItem(text, self._closable, self)
        self._connect_item(item)
        if index < 0 or index > len(self._items):
            index = len(self._items)
        self._items.insert(index, item)
        self._flow_layout.insertWidget(index, item)
        return index

    def removeTab(self, index: int) -> None:
        """Remove the tab at ``index``."""
        if not (0 <= index < len(self._items)):
            return
        item = self._items.pop(index)
        self._flow_layout.removeWidget(item)
        item.setParent(None)
        item.deleteLater()
        if self._current_index == index:
            self._current_index = -1
            if self._items:
                new_index = max(0, min(index, len(self._items) - 1))
                self.setCurrentIndex(new_index)
        elif self._current_index > index:
            self._current_index -= 1

    def tabText(self, index: int) -> str:
        """Return the text of the tab at ``index``."""
        if 0 <= index < len(self._items):
            return self._items[index].text()
        return ""

    def setTabText(self, index: int, text: str) -> None:
        """Set the text of the tab at ``index``."""
        if 0 <= index < len(self._items):
            self._items[index].setText(text)

    def setTabToolTip(self, index: int, tooltip: str) -> None:
        """Set the tooltip of the tab at ``index``."""
        if 0 <= index < len(self._items):
            self._items[index].setToolTip(tooltip)

    def tabToolTip(self, index: int) -> str:
        """Return the tooltip of the tab at ``index``."""
        if 0 <= index < len(self._items):
            return self._items[index].toolTip()
        return ""

    def currentIndex(self) -> int:
        """Return the currently selected tab index."""
        return self._current_index

    def setCurrentIndex(self, index: int) -> None:
        """Set the currently selected tab index."""
        if index == self._current_index:
            return
        if 0 <= index < len(self._items):
            if self._current_index >= 0:
                self._items[self._current_index].setChecked(False)
            self._current_index = index
            self._items[index].setChecked(True)
            self.currentChanged.emit(index)

    def tabAt(self, pos: QtCore.QPoint) -> int:
        """Return the tab index at position ``pos`` in tab-bar coordinates."""
        mapped = self.mapFromParent(pos)
        for i, item in enumerate(self._items):
            if item.geometry().contains(mapped):
                return i
        return -1

    def tabRect(self, index: int) -> QtCore.QRect:
        """Return the bounding rectangle of the tab at ``index``."""
        if 0 <= index < len(self._items):
            return self._items[index].geometry()
        return QtCore.QRect()

    def setTabsClosable(self, closable: bool) -> None:
        """Show or hide close buttons on all tabs."""
        self._closable = closable
        for item in self._items:
            item.setClosable(closable)

    def setTabBarVisible(self, visible: bool) -> None:
        """Show or hide the tab bar."""
        self.setVisible(visible)

    def _connect_item(self, item: DockStackedTabItem) -> None:
        """Wire an item's signals to the bar's handlers."""
        item.clicked.connect(lambda _checked=False, it=item: self._on_item_clicked(it))
        item.doubleClicked.connect(lambda it=item: self._on_item_double_clicked(it))
        item.closeRequested.connect(lambda it=item: self._on_item_close_requested(it))
        item.contextMenuRequested.connect(
            lambda pos, it=item: self._on_item_context_menu(it, pos)
        )
        item.dragStarted.connect(lambda pos, it=item: self._on_item_drag_started(it, pos))

    def _on_item_clicked(self, item: DockStackedTabItem) -> None:
        """Select the clicked tab."""
        try:
            index = self._items.index(item)
        except ValueError:
            return
        self.setCurrentIndex(index)

    def _on_item_double_clicked(self, item: DockStackedTabItem) -> None:
        """Forward tab double clicks."""
        try:
            index = self._items.index(item)
        except ValueError:
            return
        self.doubleClickedTab.emit(index)

    def _on_item_close_requested(self, item: DockStackedTabItem) -> None:
        """Forward tab close requests."""
        try:
            index = self._items.index(item)
        except ValueError:
            return
        self.tabCloseRequested.emit(index)

    def _on_item_context_menu(
        self, item: DockStackedTabItem, pos: QtCore.QPoint
    ) -> None:
        """Forward tab context menu requests."""
        try:
            index = self._items.index(item)
        except ValueError:
            return
        self.contextMenuRequested.emit(index, pos)

    def _on_item_drag_started(
        self, item: DockStackedTabItem, start_pos: QtCore.QPoint
    ) -> None:
        """Initiate a drag for splitting the dock area."""
        try:
            index = self._items.index(item)
        except ValueError:
            return

        # Walk up the parent chain to find the DockStackedTabWidget.
        # self.parentWidget() returns _top_bar (not DockStackedTabWidget) because
        # top_layout.addWidget() re-parents self to _top_bar when the layout is
        # applied to _top_bar.  We must keep walking until we find the owner.
        from chisurf.gui.widgets.dock_area.dock_stacked_tab_widget import DockStackedTabWidget
        from chisurf.gui.widgets.dock_area.dock_area import DockArea, DockTabWidget
        source_tw = self.parentWidget()
        while source_tw is not None and not isinstance(source_tw, DockStackedTabWidget):
            source_tw = source_tw.parentWidget()
        if source_tw is None:
            return

        # Refuse to drag if only one tab exists across the entire dock area.
        dock_area = getattr(source_tw, "dock_area", None)
        if dock_area is not None:
            total_tabs = sum(tw.count() for tw in dock_area.findChildren(DockTabWidget))
            total_tabs += sum(tw.count() for tw in dock_area.findChildren(DockStackedTabWidget))
            if total_tabs <= 1:
                return

        drag = QtGui.QDrag(self)
        mime_data = QtCore.QMimeData()
        mime_data.setData(
            "application/x-chisurf-dock-tab",
            QtCore.QByteArray(str(id(source_tw)).encode()),
        )
        mime_data.source_widget = source_tw
        mime_data.source_index = index
        drag.setMimeData(mime_data)

        pixmap = item.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(start_pos)

        drag.exec_(QtCore.Qt.MoveAction)
        dock_area = getattr(source_tw, "dock_area", None)
        if dock_area is not None:
            dock_area.hide_overlay()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Emit ``doubleClickedTabBar`` for double clicks on empty space."""
        if event.button() == QtCore.Qt.LeftButton:
            self.doubleClickedTabBar.emit()
        super().mouseDoubleClickEvent(event)

    def minimumSizeHint(self) -> QtCore.QSize:
        """Return a minimum size hint that allows vertical growth."""
        hint = super().minimumSizeHint()
        if self._items:
            hint.setHeight(self._items[0].height())
        return hint
