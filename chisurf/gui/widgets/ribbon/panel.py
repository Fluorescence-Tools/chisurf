from __future__ import annotations

import functools
import re
from typing import Any, Callable, Dict, List, Union, overload

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from .constants import (
    ColumnWise,
    Large,
    Medium,
    RibbonButtonStyle,
    RibbonSpaceFindMode,
    Small,
)
from .gallery import RibbonGallery
from .separator import RibbonSeparator
from .toolbutton import (
    RibbonToolButton,
    RibbonMenuButton,
    RibbonDelayedMenuButton,
    RibbonSplitButton
)

#: MIME type used to identify a ribbon button dragged for in-panel reordering.
_RIBBON_REORDER_MIME = "application/x-chisurf-ribbon-reorder"


class RibbonPanelTitle(QtWidgets.QLabel):
    """Widget to display the title of a panel."""

    pass


class RibbonGridLayoutManager(object):
    """Grid Layout Manager."""

    def __init__(self, rows: int):
        """Create a new grid layout manager.

        :param rows: The number of rows in the grid layout.
        """
        self.rows = rows
        self.cells = np.ones((rows, 1), dtype=bool)

    def request_cells(self, rowSpan: int = 1, colSpan: int = 1, mode: RibbonSpaceFindMode = ColumnWise):
        """Request a number of available cells from the grid.

        :param rowSpan: The number of rows the cell should span.
        :param colSpan: The number of columns the cell should span.
        :param mode: The mode of the grid.
        :return: row, col, the row and column of the requested cell.
        """
        if rowSpan > self.rows:
            raise ValueError("RowSpan is too large")
        if mode == ColumnWise:
            for row in range(self.cells.shape[0] - rowSpan + 1):
                for col in range(self.cells.shape[1] - colSpan + 1):
                    if self.cells[row : row + rowSpan, col : col + colSpan].all():
                        self.cells[row : row + rowSpan, col : col + colSpan] = False
                        return row, col
        else:
            for col in range(self.cells.shape[1]):
                if self.cells[0, col:].all():
                    if self.cells.shape[1] - col < colSpan:
                        self.cells = np.append(
                            self.cells, np.ones((self.rows, colSpan - (self.cells.shape[1] - col)), dtype=bool), axis=1
                        )
                    self.cells[0, col:] = False
                    return 0, col
        cols = self.cells.shape[1]
        colSpan1 = colSpan
        if self.cells[:, -1].all():
            cols -= 1
            colSpan1 -= 1
        self.cells = np.append(self.cells, np.ones((self.rows, colSpan1), dtype=bool), axis=1)
        self.cells[:rowSpan, cols : cols + colSpan] = False
        return 0, cols


class RibbonPanelItemWidget(QtWidgets.QFrame):
    """Widget to display a panel item."""

    def __init__(self, parent=None):
        """Create a new panel item.

        :param parent: The parent widget.
        """
        super().__init__(parent)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout().setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMaximumSize)

    def addWidget(self, widget):
        """Add a widget to the panel item.

        :param widget: The widget to add.
        """
        self.layout().addWidget(widget)


class RibbonPanelOptionButton(QtWidgets.QToolButton):
    """Button to display the options of a panel."""

    pass


class RibbonPanel(QtWidgets.QFrame):
    """Panel in the ribbon category."""

    #: maximal number of rows
    _maxRows: int = 6
    #: rows for large widgets
    _largeRows: int = 6
    #: rows for medium widgets
    _mediumRows: int = 3
    #: rows for small widgets
    _smallRows: int = 2
    #: GridLayout manager to request available cells.
    _gridLayoutManager: RibbonGridLayoutManager
    #: whether to show the panel option button
    _showPanelOptionButton: bool

    #: widgets that are added to the panel
    _widgets: List[QtWidgets.QWidget] = []

    # height of the title widget
    _titleHeight: int = 15

    # Panel options signal
    panelOptionClicked = QtCore.Signal(bool)

    @overload
    def __init__(self, title: str = "", maxRows: int = 6, showPanelOptionButton=True, parent=None):
        pass

    @overload
    def __init__(self, parent=None):
        pass

    def __init__(self, *args, **kwargs):
        """Create a new panel.

        :param title: The title of the panel.
        :param maxRows: The maximal number of rows in the panel.
        :param showPanelOptionButton: Whether to show the panel option button.
        :param parent: The parent widget.
        """
        if (args and not isinstance(args[0], QtWidgets.QWidget)) or ("title" in kwargs or "maxRows" in kwargs):
            title = args[0] if len(args) > 0 else kwargs.get("title", "")
            maxRows = args[1] if len(args) > 1 else kwargs.get("maxRows", 6)
            showPanelOptionButton = args[2] if len(args) > 2 else kwargs.get("showPanelOptionButton", True)
            parent = args[3] if len(args) > 3 else kwargs.get("parent", None)
        else:
            title = ""
            maxRows = 6
            showPanelOptionButton = True
            parent = args[0] if len(args) > 0 else kwargs.get("parent", None)
        super().__init__(parent)
        self._maxRows = maxRows
        self._largeRows = maxRows
        self._mediumRows = max(round(maxRows / 2), 1)
        self._smallRows = max(round(maxRows / 3), 1)
        self._gridLayoutManager = RibbonGridLayoutManager(self._maxRows)
        self._widgets = []
        self._showPanelOptionButton = showPanelOptionButton

        # Drag-and-drop reordering state
        self.setAcceptDrops(True)
        self._reorderStartPos = None
        self._reorderStartObj = None
        self._reorderCandidate = None
        self._addSeq = 0
        self._orderRestoreScheduled = False

        # Main layout
        self._mainLayout = QtWidgets.QVBoxLayout(self)
        self._mainLayout.setContentsMargins(0, 0, 0, 0)
        self._mainLayout.setSpacing(0)

        # Actions layout
        self._actionsLayout = QtWidgets.QGridLayout()
        self._actionsLayout.setContentsMargins(5, 5, 5, 5)
        self._actionsLayout.setSpacing(0)
        self._mainLayout.addLayout(self._actionsLayout, 1)

        # Title layout
        self._titleWidget = QtWidgets.QWidget()
        self._titleWidget.setFixedHeight(self._titleHeight)
        self._titleLayout = QtWidgets.QHBoxLayout(self._titleWidget)
        self._titleLayout.setContentsMargins(0, 0, 0, 0)
        self._titleLayout.setSpacing(0)
        self._titleLabel = RibbonPanelTitle()  # type: ignore
        self._titleLabel.setText(title)
        self._titleLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._titleLayout.addWidget(self._titleLabel, 1)

        # Panel option button
        if showPanelOptionButton:
            self._panelOption = RibbonPanelOptionButton()  # type: ignore
            self._panelOption.setAutoRaise(True)
            # Use Unicode icon instead of PNG
            self._panelOption.setText("...")
            self._panelOption.setIconSize(QtCore.QSize(self._titleHeight, self._titleHeight))
            self._panelOption.setToolTip("Panel options")
            self._panelOption.clicked.connect(self.panelOptionClicked)  # type: ignore
            self._titleLayout.addWidget(self._panelOption, 0)

        self._mainLayout.addWidget(self._titleWidget, 0)

    def maximumRows(self) -> int:
        """Return the maximal number of rows in the panel.

        :return: The maximal number of rows in the panel.
        """
        return self._maxRows

    def largeRows(self) -> int:
        """Return the number of span rows for large widgets.

        :return: The number of span rows for large widgets.
        """
        return self._largeRows

    def mediumRows(self) -> int:
        """Return the number of span rows for medium widgets.

        :return: The number of span rows for medium widgets.
        """
        return self._mediumRows

    def smallRows(self) -> int:
        """Return the number of span rows for small widgets.

        :return: The number of span rows for small widgets.
        """
        return self._smallRows

    def setMaximumRows(self, maxRows: int):
        """Set the maximal number of rows in the panel.

        :param maxRows: The maximal number of rows in the panel.
        """
        self._maxRows = maxRows
        self._largeRows = maxRows
        self._mediumRows = max(round(maxRows / 2), 1)
        self._smallRows = max(round(maxRows / 3), 1)

    def setLargeRows(self, rows: int):
        """Set the number of span rows for large widgets.

        :param rows: The number of span rows for large widgets.
        """
        assert rows <= self._maxRows, "Invalid number of rows"
        self._largeRows = rows

    def setMediumRows(self, rows: int):
        """Set the number of span rows for medium widgets.

        :param rows: The number of span rows for medium widgets.
        """
        assert 0 < rows <= self._maxRows, "Invalid number of rows"
        self._mediumRows = rows

    def setSmallRows(self, rows: int):
        """Set the number of span rows for small widgets.

        :param rows: The number of span rows for small widgets.
        """
        assert 0 < rows <= self._maxRows, "Invalid number of rows"
        self._smallRows = rows

    def defaultRowSpan(self, rowSpan: Union[int, RibbonButtonStyle]) -> int:
        """Return the number of span rows for the given widget type.

        :param rowSpan: row span or type.
        :return: The number of span rows for the given widget type.
        """
        if not isinstance(rowSpan, RibbonButtonStyle):
            return rowSpan
        if rowSpan == Large:
            return self._largeRows
        elif rowSpan == Medium:
            return self._mediumRows
        elif rowSpan == Small:
            return self._smallRows
        else:
            raise ValueError("Invalid row span")

    def panelOptionButton(self) -> RibbonPanelOptionButton:
        """Return the panel option button.

        :return: The panel option button.
        """
        return self._panelOption

    def setPanelOptionToolTip(self, text: str):
        """Set the tooltip of the panel option button.

        :param text: The tooltip text.
        """
        self._panelOption.setToolTip(text)

    def rowHeight(self) -> int:
        """Return the height of a row."""
        return int(
            (
                self.size().height()
                - self._mainLayout.contentsMargins().top()
                - self._mainLayout.contentsMargins().bottom()
                - self._mainLayout.spacing()
                - self._titleWidget.height()
                - self._actionsLayout.contentsMargins().top()
                - self._actionsLayout.contentsMargins().bottom()
                - self._actionsLayout.verticalSpacing() * (self._gridLayoutManager.rows - 1)
            )
            / self._gridLayoutManager.rows
        )

    def setTitle(self, title: str):
        """Set the title of the panel.

        :param title: The title to set.
        """
        self._titleLabel.setText(title)

    def title(self):
        """Get the title of the panel.

        :return: The title.
        """
        return self._titleLabel.text()

    def setTitleHeight(self, height: int):
        """Set the height of the title widget.

        :param height: The height to set.
        """
        self._titleHeight = height
        self._titleWidget.setFixedHeight(height)
        self._panelOption.setIconSize(QtCore.QSize(height, height))

    def titleHeight(self) -> int:
        """Get the height of the title widget.

        :return: The height of the title widget.
        """
        return self._titleHeight

    def addWidgetsBy(self, data: Dict[str, Dict]) -> Dict[str, QtWidgets.QWidget]:
        """Add widgets to the panel.

        :param data: The data to add. The dict is of the form:

            .. code-block:: python

                {
                    "widget-name": {
                        "type": "Button",
                        "args": (),
                        "kwargs": {  # or "arguments" for backward compatibility
                            "key1": "value1",
                            "key2": "value2"
                        }
                    },
                }

            Possible types are: Button, SmallButton, MediumButton, LargeButton,
            ToggleButton, SmallToggleButton, MediumToggleButton, LargeToggleButton, ComboBox, FontComboBox,
            LineEdit, TextEdit, PlainTextEdit, Label, ProgressBar, SpinBox, DoubleSpinBox, DataEdit, TimeEdit,
            DateTimeEdit, TableWidget, TreeWidget, ListWidget, CalendarWidget, Separator, HorizontalSeparator,
            VerticalSeparator, Gallery.
        :return: A dictionary of the added widgets.
        """
        widgets = {}  # type: Dict[str, QtWidgets.QWidget]
        for key, widget_data in data.items():
            type = widget_data.pop("type", "").capitalize()
            method = getattr(self, f"add{type}", None)  # type: Callable
            assert callable(method), f"Method add{type} is not callable or does not exist"
            args = widget_data.get("args", ())
            kwargs = widget_data.get("kwargs", widget_data.get("arguments", {}))
            widgets[key] = method(*args, **kwargs)
        return widgets

    def addWidget(
        self,
        widget: QtWidgets.QWidget,
        *,
        rowSpan: Union[int, RibbonButtonStyle] = Small,
        colSpan: int = 1,
        mode: RibbonSpaceFindMode = ColumnWise,
        alignment: QtCore.Qt.AlignmentFlag = QtCore.Qt.AlignmentFlag.AlignCenter,
        fixedHeight: Union[bool, float] = False,
    ) -> QtWidgets.QWidget | Any:
        """Add a widget to the panel.

        :param widget: The widget to add.
        :param rowSpan: The number of rows the widget should span, 2: small, 3: medium, 6: large.
        :param colSpan: The number of columns the widget should span.
        :param mode: The mode to find spaces.
        :param alignment: The alignment of the widget.
        :param fixedHeight: Whether to fix the height of the widget, it can be a boolean, a percentage or a fixed
                            height, when a boolean is given, the height is fixed to the maximum height allowed if the
                            value is True, when a percentage is given (0 < percentage < 1) the height is calculated
                            from the height of the maximum height allowed, depends on the number of rows to span. The
                            minimum height is 40% of the maximum height allowed.
        :return: The added widget.
        """
        rowSpan = self.defaultRowSpan(rowSpan)
        # Stable per-panel identity used to persist a custom button order
        widget._ribbon_seq = self._addSeq
        self._addSeq += 1
        self._widgets.append(widget)
        self._installReorderFilter(widget)

        # Save layout metadata for reflowing
        widget._ribbon_rowSpan = rowSpan
        widget._ribbon_colSpan = colSpan
        widget._ribbon_mode = mode
        widget._ribbon_alignment = alignment
        widget._ribbon_fixedHeight = fixedHeight
        
        row, col = self._gridLayoutManager.request_cells(rowSpan, colSpan, mode)
        maximumHeight = self.rowHeight() * rowSpan + self._actionsLayout.verticalSpacing() * (rowSpan - 2)
        widget.setMaximumHeight(maximumHeight)
        if fixedHeight is True or fixedHeight > 0:
            fixedHeight = (
                int(fixedHeight * maximumHeight)
                if 0 < fixedHeight <= 1
                else fixedHeight if 1 < fixedHeight < maximumHeight else maximumHeight
            )
            fixedHeight = max(fixedHeight, 0.4 * maximumHeight)  # minimum height is 40% of the maximum height
            widget.setFixedHeight(fixedHeight)
        item = RibbonPanelItemWidget(self)
        item.addWidget(widget)
        self._actionsLayout.addWidget(item, row, col, rowSpan, colSpan, alignment)  # type: ignore
        
        # Register for hiding/QAT tracking with the RibbonBar
        ribbon = self
        while ribbon is not None and ribbon.__class__.__name__ != 'RibbonBar':
            ribbon = ribbon.parent()
        if ribbon is not None and hasattr(ribbon, 'registerTargetButton'):
            ribbon.registerTargetButton(widget)

        # Apply any persisted custom order once all widgets of this panel have
        # been added (a single-shot timer runs after the current call stack).
        if not self._orderRestoreScheduled:
            self._orderRestoreScheduled = True
            QtCore.QTimer.singleShot(0, self._restoreButtonOrder)

        return widget

    addSmallWidget = functools.partialmethod(addWidget, rowSpan=Small)
    addMediumWidget = functools.partialmethod(addWidget, rowSpan=Medium)
    addLargeWidget = functools.partialmethod(addWidget, rowSpan=Large)

    def reflow(self):
        """Reflow the remaining visible widgets to fill any gaps left by hidden widgets."""
        # Clean current layout but preserve widgets
        
        # We must pull the source widgets out of the RibbonPanelItemWidget containers
        for i in reversed(range(self._actionsLayout.count())):
            item = self._actionsLayout.takeAt(i)
            container = item.widget()
            if container is not None and isinstance(container, RibbonPanelItemWidget):
                # The actual button is inside the container
                button_item = container.layout().takeAt(0)
                if button_item:
                    button = button_item.widget()
                    if button:
                        button.setParent(self) # Keep alive
                container.deleteLater()
                
        # Reset grid layout manager
        self._gridLayoutManager = RibbonGridLayoutManager(self._maxRows)
        
        ribbon = self
        while ribbon is not None and ribbon.__class__.__name__ != 'RibbonBar':
            ribbon = ribbon.parent()
            
        hidden_ids = getattr(ribbon, '_hidden_button_ids', []) if ribbon else []
            
        # Re-add all widgets sequentially
        for widget in self._widgets:
            btn_id = getattr(widget, '_ribbon_btn_id', None)
            if btn_id in hidden_ids:
                # Explicitly hide
                widget.hide()
                continue
                
            # If visible, request new cells and re-add
            rowSpan = getattr(widget, '_ribbon_rowSpan', self.defaultRowSpan(Small))
            colSpan = getattr(widget, '_ribbon_colSpan', 1)
            mode = getattr(widget, '_ribbon_mode', ColumnWise)
            alignment = getattr(widget, '_ribbon_alignment', QtCore.Qt.AlignmentFlag.AlignCenter)
            
            row, col = self._gridLayoutManager.request_cells(rowSpan, colSpan, mode)
            
            # Wrap to item again
            item = RibbonPanelItemWidget(self)
            item.addWidget(widget)
            item.show()
            widget.show()
            self._actionsLayout.addWidget(item, row, col, rowSpan, colSpan, alignment)  # type: ignore

    def removeWidget(self, widget: QtWidgets.QWidget):
        """Remove a widget from the panel."""
        self._actionsLayout.removeWidget(widget)

    def widget(self, index: int) -> QtWidgets.QWidget:
        """Get the widget at the given index.

        :param index: The index of the widget, starting from 0.
        :return: The widget at the given index.
        """
        return self._widgets[index]

    def widgets(self) -> List[QtWidgets.QWidget]:
        """Get all the widgets in the panel.

        :return: A list of all the widgets in the panel.
        """
        return self._widgets

    # ------------------------------------------------------------------
    # Drag-and-drop reordering of buttons within the panel
    # ------------------------------------------------------------------

    def _installReorderFilter(self, widget: QtWidgets.QWidget):
        """Enable drag-to-reorder for button widgets.

        Only button widgets participate; input widgets (spin boxes, line
        edits, sliders, ...) keep their normal mouse behaviour.

        :param widget: The widget that was just added to the panel.
        """
        if isinstance(widget, RibbonSplitButton):
            widget.actionButton().installEventFilter(self)
            widget.menuButton().installEventFilter(self)
        elif isinstance(widget, (RibbonToolButton, RibbonMenuButton, RibbonDelayedMenuButton)):
            widget.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Start a drag once the pointer moves far enough with the left button."""
        et = event.type()
        if et == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._reorderStartPos = event.pos()
                self._reorderStartObj = obj
                self._reorderCandidate = self._ownerWidget(obj)
            else:
                self._reorderStartPos = None
        elif et == QtCore.QEvent.Type.MouseMove:
            if (
                self._reorderStartPos is not None
                and obj is self._reorderStartObj
                and (event.buttons() & QtCore.Qt.MouseButton.LeftButton)
            ):
                distance = (event.pos() - self._reorderStartPos).manhattanLength()
                if distance >= QtWidgets.QApplication.startDragDistance():
                    candidate = self._reorderCandidate
                    self._reorderStartPos = None
                    self._reorderStartObj = None
                    self._reorderCandidate = None
                    self._startReorderDrag(candidate)
                    return True
        elif et == QtCore.QEvent.Type.MouseButtonRelease:
            self._reorderStartPos = None
            self._reorderStartObj = None
            self._reorderCandidate = None
        return super().eventFilter(obj, event)

    def _ownerWidget(self, obj: QtCore.QObject) -> QtWidgets.QWidget | None:
        """Return the panel-level widget owning ``obj`` (walks up the parents)."""
        widget = obj
        while widget is not None:
            if widget in self._widgets:
                return widget
            widget = widget.parent()
        return None

    def _startReorderDrag(self, widget: QtWidgets.QWidget):
        """Begin a drag operation for ``widget``."""
        if widget is None or widget not in self._widgets:
            return
        index = self._widgets.index(widget)
        drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        mime.setData(_RIBBON_REORDER_MIME, str(index).encode("ascii"))
        drag.setMimeData(mime)
        pixmap = widget.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(QtCore.QPoint(pixmap.width() // 2, pixmap.height() // 2))
        drag.exec_(QtCore.Qt.DropAction.MoveAction)
        # A finished QDrag can leave the source button visually "pressed".
        if hasattr(widget, "setDown"):
            widget.setDown(False)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """Accept only reorder drags originating from this panel."""
        if event.source() is self and event.mimeData().hasFormat(_RIBBON_REORDER_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        """Keep accepting the drag while it hovers over the panel."""
        if event.source() is self and event.mimeData().hasFormat(_RIBBON_REORDER_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        """Reorder the dragged button to the drop position."""
        if event.source() is not self or not event.mimeData().hasFormat(_RIBBON_REORDER_MIME):
            event.ignore()
            return
        try:
            source_index = int(bytes(event.mimeData().data(_RIBBON_REORDER_MIME)).decode("ascii"))
        except (ValueError, TypeError):
            event.ignore()
            return
        target_index = self._dropIndexAt(event.pos())
        self._moveWidget(source_index, target_index)
        event.acceptProposedAction()

    def _dropIndexAt(self, pos: QtCore.QPoint) -> int:
        """Return the insertion index in ``self._widgets`` for a drop at ``pos``.

        The index is chosen by comparing the drop x-coordinate against the
        horizontal centre of each visible widget (the panel lays out
        column-wise, left to right).
        """
        for widget in self._widgets:
            if not widget.isVisible():
                continue
            container = widget.parentWidget()
            geometry = (
                container.geometry()
                if isinstance(container, RibbonPanelItemWidget)
                else widget.geometry()
            )
            if pos.x() < geometry.x() + geometry.width() / 2:
                return self._widgets.index(widget)
        return len(self._widgets)

    def _moveWidget(self, source_index: int, target_index: int):
        """Move the widget at ``source_index`` to ``target_index`` and persist."""
        if not (0 <= source_index < len(self._widgets)):
            return
        insert_index = target_index - 1 if target_index > source_index else target_index
        insert_index = max(0, min(insert_index, len(self._widgets) - 1))
        if insert_index == source_index:
            return  # dropped onto itself; nothing to do
        widget = self._widgets.pop(source_index)
        self._widgets.insert(insert_index, widget)
        self.reflow()
        self._saveButtonOrder()

    def _panelStorageKey(self) -> str:
        """Return the QSettings key that stores this panel's button order."""
        category = self
        while category is not None and 'Category' not in category.__class__.__name__:
            category = category.parent()
        category_title = (
            category.title() if category is not None and hasattr(category, 'title') else "UnknownCategory"
        )
        return f"order/{category_title}::{self.title()}"

    def _saveButtonOrder(self):
        """Persist the current widget order to QSettings."""
        settings = QtCore.QSettings("ChiSurf", "RibbonState")
        order = [str(getattr(w, '_ribbon_seq', i)) for i, w in enumerate(self._widgets)]
        settings.setValue(self._panelStorageKey(), order)

    def _restoreButtonOrder(self):
        """Reorder widgets to match a persisted custom order, if any."""
        settings = QtCore.QSettings("ChiSurf", "RibbonState")
        value = settings.value(self._panelStorageKey(), [])
        order = value if isinstance(value, list) else [value] if value else []
        if not order:
            return
        rank = {seq: idx for idx, seq in enumerate(order)}
        default = len(order)
        # Stable sort keeps widgets not present in the saved order in place.
        self._widgets.sort(key=lambda w: rank.get(str(getattr(w, '_ribbon_seq', -1)), default))
        self.reflow()

    def addButton(
        self,
        text: str = None,
        icon: QtGui.QIcon = None,
        showText: bool = True,
        slot: Callable = None,
        shortcut: (
            QtCore.Qt.Key | QtGui.QKeySequence | QtCore.QKeyCombination | QtGui.QKeySequence.StandardKey | str | int
        ) = None,
        tooltip: str = None,
        statusTip: str = None,
        checkable: bool = False,
        *,
        rowSpan: RibbonButtonStyle = Large,
        **kwargs,
    ) -> RibbonToolButton:
        """Add a button to the panel.

        :param text: The text of the button.
        :param icon: The icon of the button.
        :param showText: Whether to show the text of the button.
        :param slot: The slot to call when the button is clicked.
        :param shortcut: The shortcut of the button.
        :param tooltip: The tooltip of the button.
        :param statusTip: The status tip of the button.
        :param checkable: Whether the button is checkable.
        :param rowSpan: The type of the button corresponding to the number of rows it should span.
        :param kwargs: keyword arguments to control the properties of the widget on the ribbon bar.

        :return: The button that was added.
        """
        assert isinstance(rowSpan, RibbonButtonStyle), "rowSpan must be an instance of RibbonButtonStyle"
        style = rowSpan
        button = RibbonToolButton(self)
        button.setButtonStyle(style)
        button.setText(text) if text else None
        button.setIcon(icon) if icon else None
        button.clicked.connect(slot) if slot else None  # type: ignore
        button.setShortcut(shortcut) if shortcut else None
        button.setToolTip(tooltip) if tooltip else None
        button.setStatusTip(statusTip) if statusTip else None
        
        button._ribbon_text = text
        button._ribbon_icon = icon
        button._ribbon_slot = slot
        button._ribbon_shortcut = shortcut
        button._ribbon_tooltip = tooltip
        button._ribbon_statusTip = statusTip
        button._ribbon_checkable = checkable
        
        maximumHeight = (
            self.height()
            - self._titleLabel.sizeHint().height()
            - self._mainLayout.spacing()
            - self._mainLayout.contentsMargins().top()
            - self._mainLayout.contentsMargins().bottom()
        )
        button.setMaximumHeight(maximumHeight)
        if style == Large:
            fontSize = max(button.font().pointSize() * 4 / 3, button.font().pixelSize())
            arrowSize = fontSize
            maximumIconSize = max(maximumHeight - fontSize * 2 - arrowSize, 48)
            button.setMaximumIconSize(int(maximumIconSize))
        if not showText:
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setCheckable(checkable)
        kwargs["rowSpan"] = (
            self.defaultRowSpan(Small)
            if style == Small
            else self.defaultRowSpan(Medium) if style == Medium else self.defaultRowSpan(Large)
        )
        self.addWidget(button, **kwargs)  # noqa
        return button

    addSmallButton = functools.partialmethod(addButton, rowSpan=Small)
    addMediumButton = functools.partialmethod(addButton, rowSpan=Medium)
    addLargeButton = functools.partialmethod(addButton, rowSpan=Large)
    addToggleButton = functools.partialmethod(addButton, checkable=True)
    addSmallToggleButton = functools.partialmethod(addToggleButton, rowSpan=Small)
    addMediumToggleButton = functools.partialmethod(addToggleButton, rowSpan=Medium)
    addLargeToggleButton = functools.partialmethod(addToggleButton, rowSpan=Large)
    
    def addMenuButton(
        self,
        text: str = None,
        icon: QtGui.QIcon = None,
        showText: bool = True,
        slot: Callable = None,
        shortcut: (
            QtCore.Qt.Key | QtGui.QKeySequence | QtCore.QKeyCombination | QtGui.QKeySequence.StandardKey | str | int
        ) = None,
        tooltip: str = None,
        statusTip: str = None,
        *,
        rowSpan: RibbonButtonStyle = Large,
        **kwargs,
    ) -> RibbonMenuButton:
        """Add a menu button to the panel.
        
        :param text: The text of the button.
        :param icon: The icon of the button.
        :param showText: Whether to show the text of the button.
        :param slot: The slot to call when the button is clicked.
        :param shortcut: The shortcut of the button.
        :param tooltip: The tooltip of the button.
        :param statusTip: The status tip of the button.
        :param rowSpan: The type of the button corresponding to the number of rows it should span.
        :param kwargs: keyword arguments to control the properties of the widget on the ribbon bar.
        
        :return: The menu button that was added.
        """
        assert isinstance(rowSpan, RibbonButtonStyle), "rowSpan must be an instance of RibbonButtonStyle"
        style = rowSpan
        button = RibbonMenuButton(self)
        button.setButtonStyle(style)
        button.setText(text) if text else None
        button.setIcon(icon) if icon else None
        button.clicked.connect(slot) if slot else None  # type: ignore
        button.setShortcut(shortcut) if shortcut else None
        button.setToolTip(tooltip) if tooltip else None
        button.setStatusTip(statusTip) if statusTip else None
        
        button._ribbon_text = text
        button._ribbon_icon = icon
        button._ribbon_slot = slot
        button._ribbon_shortcut = shortcut
        button._ribbon_tooltip = tooltip
        button._ribbon_statusTip = statusTip
        
        maximumHeight = (
            self.height()
            - self._titleLabel.sizeHint().height()
            - self._mainLayout.spacing()
            - self._mainLayout.contentsMargins().top()
            - self._mainLayout.contentsMargins().bottom()
        )
        button.setMaximumHeight(maximumHeight)
        if style == Large:
            fontSize = max(button.font().pointSize() * 4 / 3, button.font().pixelSize())
            arrowSize = fontSize
            maximumIconSize = max(maximumHeight - fontSize * 2 - arrowSize, 48)
            button.setMaximumIconSize(int(maximumIconSize))
        if not showText:
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
            
        kwargs["rowSpan"] = (
            self.defaultRowSpan(Small)
            if style == Small
            else self.defaultRowSpan(Medium) if style == Medium else self.defaultRowSpan(Large)
        )
        self.addWidget(button, **kwargs)  # noqa
        return button
        
    def addDelayedMenuButton(
        self,
        text: str = None,
        icon: QtGui.QIcon = None,
        showText: bool = True,
        slot: Callable = None,
        shortcut: (
            QtCore.Qt.Key | QtGui.QKeySequence | QtCore.QKeyCombination | QtGui.QKeySequence.StandardKey | str | int
        ) = None,
        tooltip: str = None,
        statusTip: str = None,
        *,
        rowSpan: RibbonButtonStyle = Large,
        **kwargs,
    ) -> RibbonDelayedMenuButton:
        """Add a delayed menu button to the panel.
        
        :param text: The text of the button.
        :param icon: The icon of the button.
        :param showText: Whether to show the text of the button.
        :param slot: The slot to call when the button is clicked.
        :param shortcut: The shortcut of the button.
        :param tooltip: The tooltip of the button.
        :param statusTip: The status tip of the button.
        :param rowSpan: The type of the button corresponding to the number of rows it should span.
        :param kwargs: keyword arguments to control the properties of the widget on the ribbon bar.
        
        :return: The delayed menu button that was added.
        """
        assert isinstance(rowSpan, RibbonButtonStyle), "rowSpan must be an instance of RibbonButtonStyle"
        style = rowSpan
        button = RibbonDelayedMenuButton(self)
        button.setButtonStyle(style)
        button.setText(text) if text else None
        button.setIcon(icon) if icon else None
        button.clicked.connect(slot) if slot else None  # type: ignore
        button.setShortcut(shortcut) if shortcut else None
        button.setToolTip(tooltip) if tooltip else None
        button.setStatusTip(statusTip) if statusTip else None
        
        button._ribbon_text = text
        button._ribbon_icon = icon
        button._ribbon_slot = slot
        button._ribbon_shortcut = shortcut
        button._ribbon_tooltip = tooltip
        button._ribbon_statusTip = statusTip
        
        maximumHeight = (
            self.height()
            - self._titleLabel.sizeHint().height()
            - self._mainLayout.spacing()
            - self._mainLayout.contentsMargins().top()
            - self._mainLayout.contentsMargins().bottom()
        )
        button.setMaximumHeight(maximumHeight)
        if style == Large:
            fontSize = max(button.font().pointSize() * 4 / 3, button.font().pixelSize())
            arrowSize = fontSize
            maximumIconSize = max(maximumHeight - fontSize * 2 - arrowSize, 48)
            button.setMaximumIconSize(int(maximumIconSize))
        if not showText:
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
            
        kwargs["rowSpan"] = (
            self.defaultRowSpan(Small)
            if style == Small
            else self.defaultRowSpan(Medium) if style == Medium else self.defaultRowSpan(Large)
        )
        self.addWidget(button, **kwargs)  # noqa
        return button
        
    def addSplitButton(
        self,
        text: str = None,
        icon: QtGui.QIcon = None,
        showText: bool = True,
        slot: Callable = None,
        shortcut: (
            QtCore.Qt.Key | QtGui.QKeySequence | QtCore.QKeyCombination | QtGui.QKeySequence.StandardKey | str | int
        ) = None,
        tooltip: str = None,
        statusTip: str = None,
        *,
        rowSpan: RibbonButtonStyle = Large,
        **kwargs,
    ) -> RibbonSplitButton:
        """Add a split button to the panel.
        
        :param text: The text of the button.
        :param icon: The icon of the button.
        :param showText: Whether to show the text of the button.
        :param slot: The slot to call when the button is clicked.
        :param shortcut: The shortcut of the button.
        :param tooltip: The tooltip of the button.
        :param statusTip: The status tip of the button.
        :param rowSpan: The type of the button corresponding to the number of rows it should span.
        :param kwargs: keyword arguments to control the properties of the widget on the ribbon bar.
        
        :return: The split button that was added.
        """
        assert isinstance(rowSpan, RibbonButtonStyle), "rowSpan must be an instance of RibbonButtonStyle"
        style = rowSpan
        button = RibbonSplitButton(self)
        button.setText(text) if text else None
        button.setIcon(icon) if icon else None
        button.actionClicked.connect(slot) if slot else None  # type: ignore
        button.setToolTip(tooltip) if tooltip else None
        button.setStatusTip(statusTip) if statusTip else None
        
        button._ribbon_text = text
        button._ribbon_icon = icon
        button._ribbon_slot = slot
        button._ribbon_shortcut = shortcut
        button._ribbon_tooltip = tooltip
        button._ribbon_statusTip = statusTip
        
        # Configure action button
        action_button = button.actionButton()
        action_button.setMaximumHeight(
            self.height()
            - self._titleLabel.sizeHint().height()
            - self._mainLayout.spacing()
            - self._mainLayout.contentsMargins().top()
            - self._mainLayout.contentsMargins().bottom()
        )
        if style == Large:
            fontSize = max(action_button.font().pointSize() * 4 / 3, action_button.font().pixelSize())
            arrowSize = fontSize
            maximumIconSize = max(action_button.maximumHeight() - fontSize * 2 - arrowSize, 48)
            action_button.setMaximumIconSize(int(maximumIconSize))
        if not showText:
            action_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
            
        kwargs["rowSpan"] = (
            self.defaultRowSpan(Small)
            if style == Small
            else self.defaultRowSpan(Medium) if style == Medium else self.defaultRowSpan(Large)
        )
        self.addWidget(button, **kwargs)  # noqa
        return button
        
    # Convenience methods for different sizes
    addSmallMenuButton = functools.partialmethod(addMenuButton, rowSpan=Small)
    addMediumMenuButton = functools.partialmethod(addMenuButton, rowSpan=Medium)
    addLargeMenuButton = functools.partialmethod(addMenuButton, rowSpan=Large)
    addSmallDelayedMenuButton = functools.partialmethod(addDelayedMenuButton, rowSpan=Small)
    addMediumDelayedMenuButton = functools.partialmethod(addDelayedMenuButton, rowSpan=Medium)
    addLargeDelayedMenuButton = functools.partialmethod(addDelayedMenuButton, rowSpan=Large)
    addSmallSplitButton = functools.partialmethod(addSplitButton, rowSpan=Small)
    addMediumSplitButton = functools.partialmethod(addSplitButton, rowSpan=Medium)
    addLargeSplitButton = functools.partialmethod(addSplitButton, rowSpan=Large)

    def _addAnyWidget(
        self,
        *args,
        cls,
        initializer: Callable = None,
        rowSpan: Union[int, RibbonButtonStyle] = Small,
        colSpan: int = 1,
        mode: RibbonSpaceFindMode = ColumnWise,
        alignment: QtCore.Qt.AlignmentFlag = QtCore.Qt.AlignmentFlag.AlignCenter,
        fixedHeight: Union[bool, float] = False,
        **kwargs,
    ) -> QtWidgets.QWidget:
        """Add any widget to the panel.

        :param cls: The class of the widget to add.
        :param initializer: The initializer function of the widget to add.
        :param args: The arguments passed to the initializer.
        :param rowSpan: The number of rows the widget should span, 2: small, 3: medium, 6: large.
        :param colSpan: The number of columns the widget should span.
        :param mode: The mode to find spaces.
        :param alignment: The alignment of the widget.
        :param fixedHeight: Whether to fix the height of the widget, it can be a boolean, a percentage or a fixed
                            height, when a boolean is given, the height is fixed to the maximum height allowed if the
                            value is True, when a percentage is given (0 < percentage < 1) the height is calculated
                            from the height of the maximum height allowed, depends on the number of rows to span. The
                            minimum height is 40% of the maximum height allowed.
        :param kwargs: The keyword arguments are passed to the initializer
        """
        widget = cls(self)
        if callable(initializer):
            initializer(widget, *args, **kwargs)
        elif args or kwargs:
            raise ValueError("Arguments are provided but the initializer is not set")
        return self.addWidget(
            widget, rowSpan=rowSpan, colSpan=colSpan, mode=mode, alignment=alignment, fixedHeight=fixedHeight
        )

    def __getattr__(self, method: str) -> Callable:
        """Get the dynamic method `add[Small|Medium|Large][Widget]`.

        :param method: The name of the method to get.
        :return: The method of the widget to add.
        """
        # Match the method name
        match = re.match(r"add(Small|Medium|Large)(\w+)", method)
        assert match, "Invalid method name"

        # Get the widget class and the size
        size = match.group(1)
        base_method_name = f"add{match.group(2)}"
        assert hasattr(self, base_method_name), f"Invalid method name {base_method_name}"

        # Get the base method
        base_method = getattr(self, base_method_name)
        rowSpan = Small if size == "Small" else Medium if size == "Medium" else Large

        # Create the new method
        return functools.partial(base_method, rowSpan=rowSpan)

    addCheckBox = functools.partialmethod(
        _addAnyWidget, cls=QtWidgets.QCheckBox, initializer=QtWidgets.QCheckBox.setText
    )
    addComboBox = functools.partialmethod(
        _addAnyWidget, cls=QtWidgets.QComboBox, initializer=QtWidgets.QComboBox.addItems
    )
    addFontComboBox = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QFontComboBox)
    addLineEdit = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QLineEdit)
    addTextEdit = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QTextEdit)
    addPlainTextEdit = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QPlainTextEdit)
    addLabel = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QLabel, initializer=QtWidgets.QLabel.setText)
    addProgressBar = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QProgressBar)
    addSlider = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QSlider)
    addSpinBox = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QSpinBox)
    addDoubleSpinBox = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QDoubleSpinBox)
    addDateEdit = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QDateEdit)
    addTimeEdit = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QTimeEdit)
    addDateTimeEdit = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QDateTimeEdit)
    addTableWidget = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QTableWidget, rowSpan=Large)
    addTreeWidget = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QTreeWidget, rowSpan=Large)
    addListWidget = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QListWidget, rowSpan=Large)
    addCalendarWidget = functools.partialmethod(_addAnyWidget, cls=QtWidgets.QCalendarWidget, rowSpan=Large)

    def addSeparator(self, orientation=QtCore.Qt.Orientation.Vertical, width=6, **kwargs) -> RibbonSeparator:
        """Add a separator to the panel.

        :param orientation: The orientation of the separator.
        :param width: The width of the separator.
        :param kwargs: keyword arguments to control the properties of the widget on the ribbon bar.

        :return: The separator.
        """
        kwargs["rowSpan"] = Large if "rowSpan" not in kwargs else kwargs["rowSpan"]
        return self.addWidget(RibbonSeparator(orientation, width), **kwargs)

    addHorizontalSeparator = functools.partialmethod(addSeparator, orientation=QtCore.Qt.Orientation.Horizontal)
    addVerticalSeparator = functools.partialmethod(addSeparator, orientation=QtCore.Qt.Orientation.Vertical)

    def addGallery(self, minimumWidth=800, popupHideOnClick=False, **kwargs) -> RibbonGallery:
        """Add a gallery to the panel.

        :param minimumWidth: The minimum width of the gallery.
        :param popupHideOnClick: Whether the gallery popup should be hidden when a user clicks on it.
        :param kwargs: keyword arguments to control the properties of the widget on the ribbon bar.

        :return: The gallery.
        """
        kwargs["rowSpan"] = Large if "rowSpan" not in kwargs else kwargs["rowSpan"]
        rowSpan = self.defaultRowSpan(kwargs["rowSpan"])
        gallery = RibbonGallery(minimumWidth, popupHideOnClick, self)
        maximumHeight = self.rowHeight() * rowSpan + self._actionsLayout.verticalSpacing() * (rowSpan - 2)
        gallery.setFixedHeight(maximumHeight)
        return self.addWidget(gallery, **kwargs)
