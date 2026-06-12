from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.dock_area.dock_overlay import DockDropOverlay
from chisurf.gui.widgets.dock_area.dock_tab_bar import DockTabBar

_MAX_TAB_TEXT_LEN = 30


def _shorten_path(name: str) -> tuple[str, str]:
    """Return ``(display_text, full_tooltip)`` for a tab name.

    If *name* looks like an absolute file path and exceeds
    ``_MAX_TAB_TEXT_LEN``, the display text is shortened to
    ``…/parent/filename`` while the tooltip retains the full path.
    """
    import os
    if not os.path.isabs(name) or len(name) <= _MAX_TAB_TEXT_LEN:
        return name, name
    parent = os.path.basename(os.path.dirname(name))
    base = os.path.basename(name)
    short = f"…/{parent}/{base}" if parent else f"…/{base}"
    return short, name


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
        self.setDocumentMode(True)
        self.setAcceptDrops(True)
        self._new_tab_btn = None

        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(128, 128, 128, 40);
                background-color: transparent;
            }
            QTabBar::tab {
                background-color: rgba(128, 128, 128, 30);
                padding: 0px 8px;
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
        transparent = QtGui.QColor(0, 0, 0, 0)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, transparent)
        pal.setColor(QtGui.QPalette.Base, transparent)
        self.setPalette(pal)
        stacked = self.findChild(QtWidgets.QStackedWidget)
        if stacked is not None:
            stacked.setAutoFillBackground(False)
            sp = stacked.palette()
            sp.setColor(QtGui.QPalette.Window, transparent)
            sp.setColor(QtGui.QPalette.Base, transparent)
            stacked.setPalette(sp)

        self.currentChanged.connect(self._on_current_changed)
        self.tab_bar.doubleClickedTab.connect(self._on_tab_double_clicked)
        self.tab_bar.doubleClickedTabBar.connect(self._on_tab_bar_double_clicked)
        self.tab_bar.contextMenuRequested.connect(self._on_tab_context_menu)
        self.tabCloseRequested.connect(
            lambda idx, tw=self: self.dock_area._on_tab_close_requested(tw, idx)
        )

    def setNewTabButtonVisible(self, visible: bool = True) -> None:
        """Show or hide the local '+' new-tab button in the left corner of the tab bar."""
        if visible:
            if self._new_tab_btn is None:
                self._new_tab_btn = QtWidgets.QToolButton(self)
                self._new_tab_btn.setText("+")
                self._new_tab_btn.setAutoRaise(True)
                self._new_tab_btn.clicked.connect(self.dock_area.newTabRequested.emit)
                self.setCornerWidget(self._new_tab_btn, QtCore.Qt.TopLeftCorner)
            self._new_tab_btn.show()
            return
        if self._new_tab_btn is not None:
            self._new_tab_btn.hide()

    def _on_current_changed(self, index: int) -> None:
        if index >= 0:
            self.dock_area.set_active_tab_widget(self)

    def _on_tab_double_clicked(self, index: int) -> None:
        self.dock_area.restore_tab(self, index)

    def _on_tab_bar_double_clicked(self) -> None:
        self.dock_area.restore_all_tabs(self)

    def _on_tab_context_menu(self, local_index: int, global_pos: QtCore.QPoint) -> None:
        """Forward context menu requests to the owning DockArea."""
        self.dock_area._on_tab_context_menu(self, local_index, global_pos)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Show dock-area context actions when right-clicking the tab pane."""
        if self.dock_area._show_area_context_menu(event.globalPos()):
            event.accept()
            return
        super().contextMenuEvent(event)

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
    tabCloseRequested = QtCore.Signal(int)
    tabActionRequested = QtCore.Signal(str, int)
    newTabRequested = QtCore.Signal()
    layoutChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget = None):
        """Initialize the DockArea.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self._all_widgets = []
        self._tab_names: dict[QtWidgets.QWidget, str] = {}
        self._hidden_widgets: list[QtWidgets.QWidget] = []
        self._tab_close_modes: dict[QtWidgets.QWidget, str] = {}
        self._root_widget = None
        self._active_tab_widget = None
        self._last_emitted_index = -1
        self._corner_widget = None
        self._corner = QtCore.Qt.TopRightCorner
        self._tabs_closable = False
        self._new_tab_button_visible = False
        self._context_menu_enabled = False
        self._context_menu_callback = None
        self._context_menu_mode = "document"
        self._close_tab_callback = None
        self._tab_bar_visible = True

        # Setup main layout
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        # Add overlay child
        self._overlay = DockDropOverlay(self)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.DefaultContextMenu)

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


    def setCornerWidget(self, widget: QtWidgets.QWidget, corner: QtCore.Qt.Corner = QtCore.Qt.TopRightCorner) -> None:
        """Set the widget in the given corner of the tab bar.

        Parameters
        ----------
        widget : QWidget
            The corner widget to place.
        corner : Qt.Corner, optional
            The corner where the widget is placed.
        """
        self._corner_widget = widget
        self._corner = corner
        main_tw = self.find_main_tab_widget()
        if main_tw is not None:
            main_tw.setCornerWidget(widget, corner)

    def setContextMenuEnabled(self, enabled: bool = True) -> None:
        """Enable or disable the right-click context menu on tabs.

        Parameters
        ----------
        enabled : bool
            Whether context menus are enabled.
        """
        self._context_menu_enabled = enabled

    def setContextMenuCallback(self, callback: Any | None) -> None:
        """Set an optional callback that can extend tab context menus.

        Parameters
        ----------
        callback : callable, optional
            Called as ``callback(menu, absolute_index)`` after the default
            actions have been added and before the menu is shown.
        """
        self._context_menu_callback = callback

    def setContextMenuMode(self, mode: str) -> None:
        """Set which default actions appear in tab context menus.

        Parameters
        ----------
        mode : {"document", "basic"}
            ``"document"`` includes file-oriented actions such as save,
            rename, reload, and copy path. ``"basic"`` only includes close
            actions plus any callback-provided actions.
        """
        normalized = mode.lower()
        if normalized not in {"document", "basic"}:
            raise ValueError("context menu mode must be 'document' or 'basic'")
        self._context_menu_mode = normalized

    def setNewTabButtonVisible(self, visible: bool = True) -> None:
        """Show or hide the ``+`` new-tab button in the left corner of the tab bar.

        Parameters
        ----------
        visible : bool
            Whether the new-tab button should be visible.
        """
        self._new_tab_button_visible = visible
        for tw in self.findChildren(DockTabWidget):
            tw.setNewTabButtonVisible(visible)


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

    def _widget_key(self, widget: QtWidgets.QWidget, key_func=None) -> str:
        """Return a stable key for a docked page widget.

        Parameters
        ----------
        widget : QWidget
            The page widget.
        key_func : callable, optional
            Optional callback returning a caller-defined widget key.

        Returns
        -------
        str or None
            A stable key, or None if no key can be determined.
        """
        key = None
        if callable(key_func):
            try:
                key = key_func(widget)
            except Exception:
                key = None
        if key is None:
            key = self._tab_names.get(widget)
        if key is None:
            return None
        return str(key)

    def _iter_nested_widgets(self, widget: QtWidgets.QWidget | None):
        """Yield a widget and all nested splitter page widgets."""
        if widget is None:
            return
        yield widget
        if isinstance(widget, DockSplitter):
            for index in range(widget.count()):
                yield from self._iter_nested_widgets(widget.widget(index))

    def _find_widget_by_key(
        self,
        key: str,
        key_func=None
    ) -> QtWidgets.QWidget:
        """Find a page widget by its stable key.

        Parameters
        ----------
        key : str
            The stored widget key.
        key_func : callable, optional
            Optional callback returning the current widget key.

        Returns
        -------
        QWidget or None
            The matching page widget.
        """
        for widget in self._all_widgets:
            for nested_widget in self._iter_nested_widgets(widget):
                if self._widget_key(nested_widget, key_func) == key:
                    return nested_widget
        return None

    def _find_widget_by_tab_name(self, tab_name: str) -> QtWidgets.QWidget:
        """Find a page widget by its stored tab name.

        Parameters
        ----------
        tab_name : str
            The stored tab name.

        Returns
        -------
        QWidget or None
            The matching page widget.
        """
        for widget in self._all_widgets:
            for nested_widget in self._iter_nested_widgets(widget):
                if self._tab_names.get(nested_widget) == tab_name:
                    return nested_widget
        return None

    def _widget_path(self, widget: QtWidgets.QWidget) -> list[int] | None:
        """Return the path to a widget inside the dock tree.

        Parameters
        ----------
        widget : QWidget
            The widget to locate.

        Returns
        -------
        list of int or None
            Child indexes from the root to the widget.
        """
        def _walk(node, path):
            if node is widget:
                return list(path)
            if isinstance(node, DockSplitter):
                for idx in range(node.count()):
                    found = _walk(node.widget(idx), path + [idx])
                    if found is not None:
                        return found
            return None

        return _walk(self._root_widget, [])

    def _widget_from_path(
        self,
        root: QtWidgets.QWidget,
        path: list[int] | None
    ) -> QtWidgets.QWidget:
        """Return the widget at a dock-tree path.

        Parameters
        ----------
        root : QWidget
            The root dock widget.
        path : list of int or None
            Child indexes from the root to the widget.

        Returns
        -------
        QWidget or None
            The widget at the path.
        """
        if not isinstance(path, list) or root is None:
            return None
        node = root
        for idx in path:
            if not isinstance(node, DockSplitter):
                return None
            if idx < 0 or idx >= node.count():
                return None
            node = node.widget(idx)
        return node

    def _serialize_widget(
        self,
        widget: QtWidgets.QWidget,
        key_func=None
    ) -> dict | None:
        """Serialize a dock-tree node.

        Parameters
        ----------
        widget : QWidget
            The node to serialize.
        key_func : callable, optional
            Optional callback returning a caller-defined widget key.

        Returns
        -------
        dict or None
            Serialized node.
        """
        if isinstance(widget, DockTabWidget):
            tabs = []
            for idx in range(widget.count()):
                page = widget.widget(idx)
                key = self._widget_key(page, key_func)
                tabs.append(
                    {
                        "widget_key": key,
                        "tab_name": self._tab_names.get(page, ""),
                        "tab_text": widget.tabText(idx),
                    }
                )
            return {
                "type": "tab",
                "tabs": tabs,
                "current_index": int(widget.currentIndex()),
            }
        if isinstance(widget, DockSplitter):
            orientation = "horizontal"
            if widget.orientation() == QtCore.Qt.Vertical:
                orientation = "vertical"
            return {
                "type": "splitter",
                "orientation": orientation,
                "sizes": [int(size) for size in widget.sizes()],
                "children": [
                    self._serialize_widget(widget.widget(idx), key_func)
                    for idx in range(widget.count())
                ],
            }
        return None

    def get_layout_state(self, key_func=None) -> dict:
        """Return a JSON-serializable snapshot of the dock layout.

        Parameters
        ----------
        key_func : callable, optional
            Optional callback returning a stable key for each page widget.

        Returns
        -------
        dict
            Serialized dock layout.
        """
        active_path = None
        if self.active_tab_widget() is not None:
            active_path = self._widget_path(self.active_tab_widget())
        return {
            "version": 1,
            "root": self._serialize_widget(self._root_widget, key_func),
            "active_tab_widget": active_path,
            "current_index": int(self.currentIndex()),
        }

    def _layout_state_has_tabs(self, state: dict) -> bool:
        """Return whether a serialized layout contains at least one tab.

        Parameters
        ----------
        state : dict
            Serialized dock-tree node.

        Returns
        -------
        bool
            True if the layout contains at least one tab.
        """
        if not isinstance(state, dict):
            return False
        if state.get("type") == "tab":
            return bool(state.get("tabs"))
        if state.get("type") == "splitter":
            return any(
                self._layout_state_has_tabs(child_state)
                for child_state in state.get("children", [])
            )
        return False

    def _build_widget_from_state(
        self,
        state: dict,
        key_func=None
    ) -> QtWidgets.QWidget | None:
        """Build a dock-tree node from serialized state.

        Parameters
        ----------
        state : dict
            Serialized node state.
        key_func : callable, optional
            Optional callback returning a stable key for each page widget.

        Returns
        -------
        QWidget or None
            Restored dock node.
        """
        if not isinstance(state, dict):
            return None
        node_type = state.get("type")
        if node_type == "tab":
            tab_widget = DockTabWidget(self)
            tab_widget.setTabsClosable(self._tabs_closable)
            tab_widget.tabBar().setVisible(self._tab_bar_visible)
            tab_widget.setNewTabButtonVisible(self._new_tab_button_visible)
            if self._corner_widget is not None:
                tab_widget.setCornerWidget(self._corner_widget, self._corner)
            for tab_state in state.get("tabs", []):
                if not isinstance(tab_state, dict):
                    continue
                key = tab_state.get("widget_key")
                widget = None
                if key is not None:
                    widget = self._find_widget_by_key(str(key), key_func)
                if widget is None:
                    tab_name = tab_state.get("tab_name")
                    if isinstance(tab_name, str) and tab_name:
                        widget = self._find_widget_by_tab_name(tab_name)
                if widget is None:
                    continue
                display = tab_state.get("tab_text") or self._tab_names.get(widget, key) or ""
                tab_widget.addTab(widget, display)
                self._set_tab_tooltip(tab_widget, tab_widget.count() - 1, self._tab_names.get(widget, display))
            if tab_widget.count() == 0:
                tab_widget.deleteLater()
                return None
            current_index = state.get("current_index")
            if isinstance(current_index, int) and 0 <= current_index < tab_widget.count():
                tab_widget.setCurrentIndex(current_index)
            return tab_widget
        if node_type == "splitter":
            orientation = QtCore.Qt.Horizontal
            if state.get("orientation") == "vertical":
                orientation = QtCore.Qt.Vertical
            splitter = DockSplitter(orientation, self)
            restored_children = []
            for child_state in state.get("children", []):
                child = self._build_widget_from_state(child_state, key_func)
                if child is not None:
                    restored_children.append(child)
            if not restored_children:
                splitter.deleteLater()
                return None
            for child in restored_children:
                splitter.addWidget(child)
            sizes = state.get("sizes")
            if isinstance(sizes, list) and sizes:
                splitter.setSizes([int(size) for size in sizes])
            return splitter
        return None

    @staticmethod
    def _has_parent_in(
        widget: QtWidgets.QWidget | None,
        parents: set[QtWidgets.QWidget]
    ) -> bool:
        """Return whether ``widget`` is one of ``parents`` or below one."""
        node = widget
        while node is not None:
            if node in parents:
                return True
            node = node.parentWidget()
        return False

    def _detach_page_widget(
        self,
        widget: QtWidgets.QWidget | None,
        mark_hidden: bool = False
    ) -> None:
        """Detach a registered page widget without destroying it."""
        if widget is None:
            return
        widget.setParent(self)
        widget.hide()
        if mark_hidden and widget in self._all_widgets and widget not in self._hidden_widgets:
            self._hidden_widgets.append(widget)

    def _clear_root_widgets(self, exclude_widgets=None) -> None:
        """Clear existing dock-tree widgets without deleting page widgets.

        This removes old tab widgets and splitters from the layout while
        preserving the page widgets so they can be reattached during restore.

        Parameters
        ----------
        exclude_widgets : iterable of QWidget, optional
            Newly restored dock widgets that must not be deleted.
        """
        excluded = set(exclude_widgets or [])
        for widget in list(self._all_widgets):
            if self._has_parent_in(widget, excluded):
                continue
            self._detach_page_widget(widget, mark_hidden=True)
        if self._root_widget is not None and self._root_widget not in excluded:
            self._layout.removeWidget(self._root_widget)
            self._root_widget = None
        for tab_widget in list(self.findChildren(DockTabWidget)):
            if self._has_parent_in(tab_widget, excluded):
                continue
            for idx in range(tab_widget.count() - 1, -1, -1):
                widget = tab_widget.widget(idx)
                tab_widget.removeTab(idx)
                if not self._has_parent_in(widget, excluded):
                    self._detach_page_widget(widget, mark_hidden=True)
            tab_widget.deleteLater()
        for splitter in list(self.findChildren(DockSplitter)):
            if self._has_parent_in(splitter, excluded):
                continue
            if splitter in self._all_widgets:
                continue
            splitter.deleteLater()

    def set_layout_state(
        self,
        state: dict,
        key_func=None,
        emit_change: bool = True
    ) -> bool:
        """Restore a dock layout from a state returned by get_layout_state.

        Parameters
        ----------
        state : dict
            Serialized dock layout.
        key_func : callable, optional
            Optional callback returning a stable key for each page widget.
        emit_change : bool, optional
            Whether to emit layoutChanged after restoring.

        Returns
        -------
        bool
            True if the layout was restored, False otherwise.
        """
        if not isinstance(state, dict):
            return False
        root_state = state.get("root")
        if not isinstance(root_state, dict) or not self._layout_state_has_tabs(root_state):
            return False
        restored_root = self._build_widget_from_state(root_state, key_func)
        if restored_root is None:
            return False
        self._hidden_widgets.clear()
        excluded_widgets = set(restored_root.findChildren(DockTabWidget))
        excluded_widgets.update(restored_root.findChildren(DockSplitter))
        if isinstance(restored_root, (DockTabWidget, DockSplitter)):
            excluded_widgets.add(restored_root)
        self._clear_root_widgets(exclude_widgets=excluded_widgets)
        self.set_root_widget(restored_root)
        active_widget = self._widget_from_path(restored_root, state.get("active_tab_widget"))
        if isinstance(active_widget, DockTabWidget):
            self.set_active_tab_widget(active_widget)
        current_index = state.get("current_index")
        if isinstance(current_index, int):
            self.setCurrentIndex(current_index)
        if emit_change:
            self.layoutChanged.emit()
        return True

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

    def addTab(self, widget: QtWidgets.QWidget, name: str, close_mode: str = "hide") -> None:
        """Add a tab with the given widget and name.

        Parameters
        ----------
        widget : QWidget
            The page widget to add.
        name : str
            The name to display on the tab.
        close_mode : {"hide", "remove"}, default="hide"
            Whether close requests hide the dock for later restore or remove it.
        """
        self.setTabCloseMode(widget, close_mode)
        display, tooltip = _shorten_path(name)
        if widget not in self._all_widgets:
            self._all_widgets.append(widget)
        self._tab_names[widget] = name
        if widget in self._hidden_widgets:
            self._hidden_widgets.remove(widget)
        if self._root_widget is None:
            tab_widget = DockTabWidget(self)
            tab_widget.setTabsClosable(self._tabs_closable)
            tab_widget.tabBar().setVisible(self._tab_bar_visible)
            tab_widget.setNewTabButtonVisible(self._new_tab_button_visible)
            if self._corner_widget is not None:
                tab_widget.setCornerWidget(self._corner_widget, self._corner)
            self.set_root_widget(tab_widget)
            tab_widget.addTab(widget, display)
            self._set_tab_tooltip(tab_widget, tab_widget.count() - 1, tooltip)
            self.set_active_tab_widget(tab_widget)
        else:
            main_tw = self.find_main_tab_widget()
            if main_tw is not None:
                main_tw.addTab(widget, display)
                self._set_tab_tooltip(main_tw, main_tw.count() - 1, tooltip)

    def setTabCloseMode(self, widget: QtWidgets.QWidget, mode: str) -> None:
        """Set close behavior for a dock widget.

        Parameters
        ----------
        widget : QWidget
            Dock page widget.
        mode : {"hide", "remove"}
            ``"hide"`` keeps the dock available for restore. ``"remove"``
            removes it from the dock registry.
        """
        normalized = mode.lower()
        if normalized not in {"hide", "remove"}:
            raise ValueError("dock close mode must be 'hide' or 'remove'")
        self._tab_close_modes[widget] = normalized

    def setCloseTabCallback(self, callback) -> None:
        """Set a callable to handle tab close requests instead of the signal.

        Parameters
        ----------
        callback : callable(int)
            Called with the absolute tab index when a close button is clicked.
        """
        self._close_tab_callback = callback

    def _on_tab_close_requested(self, tw: 'DockTabWidget', local_idx: int) -> None:
        """Translate a local DockTabWidget tab-close to an absolute index and forward."""
        w = tw.widget(local_idx)
        if w not in self._all_widgets:
            return
        abs_idx = self._all_widgets.index(w)
        if self._close_tab_callback is not None:
            self._close_tab_callback(abs_idx)
        else:
            self.tabCloseRequested.emit(abs_idx)

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

    def setCurrentWidget(self, widget: QtWidgets.QWidget) -> None:
        """Activate the tab containing the given widget.

        Parameters
        ----------
        widget : QWidget
            The page widget to activate.
        """
        if widget in self._all_widgets:
            self.setCurrentIndex(self._all_widgets.index(widget))

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

    def count(self) -> int:
        """Return the total number of tabs across all DockTabWidgets.

        Returns
        -------
        int
        """
        return len(self._all_widgets)

    def widget(self, index: int) -> QtWidgets.QWidget:
        """Return the page widget at the given absolute index.

        Parameters
        ----------
        index : int
            The absolute tab index.

        Returns
        -------
        QWidget or None
        """
        if 0 <= index < len(self._all_widgets):
            return self._all_widgets[index]
        return None

    def tabText(self, index: int) -> str:
        """Return the tab text at the given absolute index.

        Parameters
        ----------
        index : int
            The absolute tab index.

        Returns
        -------
        str
        """
        w = self.widget(index)
        if w is not None:
            return self._tab_names.get(w, "")
        return ""

    def setTabText(self, index: int, text: str) -> None:
        """Set the tab text at the given absolute index.

        Parameters
        ----------
        index : int
            The absolute tab index.
        text : str
            The tab text to display.
        """
        w = self.widget(index)
        if w is None:
            return
        # If the incoming text is a dirty-marker variant of the current
        # shortened display (e.g. "…/dir/file.py *"), keep the original
        # full path and just re-shorten with the marker appended.
        prev_full = self._tab_names.get(w, "")
        prev_display, _ = _shorten_path(prev_full)
        dirty = text.endswith(" *")
        clean = text[:-2] if dirty else text
        if clean == prev_display and prev_full:
            # Preserve the full path, just add/remove dirty marker
            full_for_shorten = prev_full
        else:
            full_for_shorten = text
            self._tab_names[w] = text
        display, tooltip = _shorten_path(full_for_shorten)
        if dirty and not display.endswith(" *"):
            display += " *"
        for tw in self.findChildren(DockTabWidget):
            for i in range(tw.count()):
                if tw.widget(i) is w:
                    tw.setTabText(i, display)
                    self._set_tab_tooltip(tw, i, tooltip)
                    return

    def removeTab(self, index: int) -> None:
        """Remove the tab at the given absolute index.

        Parameters
        ----------
        index : int
            The absolute tab index to remove.
        """
        w = self.widget(index)
        if w is None:
            return
        self._all_widgets.pop(index)
        self._tab_names.pop(w, None)
        self._tab_close_modes.pop(w, None)
        if w in self._hidden_widgets:
            self._hidden_widgets.remove(w)
        for tw in self.findChildren(DockTabWidget):
            for i in range(tw.count()):
                if tw.widget(i) is w:
                    tw.removeTab(i)
                    self._detach_page_widget(w)
                    self.cleanup_empty_tab_widget(tw)
                    self.layoutChanged.emit()
                    return

    def hideTab(self, index: int) -> bool:
        """Hide the tab at ``index`` without removing its dock widget.

        Parameters
        ----------
        index : int
            Absolute tab index.

        Returns
        -------
        bool
            ``True`` if the dock was hidden.
        """
        if not self.canCloseTab(index):
            return False
        w = self.widget(index)
        if w is None or w in self._hidden_widgets:
            return False
        for tw in self.findChildren(DockTabWidget):
            for local_index in range(tw.count()):
                if tw.widget(local_index) is w:
                    tw.removeTab(local_index)
                    w.setParent(self)
                    w.hide()
                    self._hidden_widgets.append(w)
                    self.cleanup_empty_tab_widget(tw)
                    self.layoutChanged.emit()
                    return True
        return False

    def showTab(self, index: int) -> bool:
        """Restore a hidden tab by absolute index."""
        w = self.widget(index)
        if w is None:
            return False
        if w not in self._hidden_widgets and self._tab_widget_for_page(w) is not None:
            return False
        if w in self._hidden_widgets:
            self._hidden_widgets.remove(w)
        display, tooltip = _shorten_path(self._tab_names.get(w, ""))
        tab_widget = self.find_main_tab_widget()
        if tab_widget is None:
            tab_widget = DockTabWidget(self)
            tab_widget.setTabsClosable(self._tabs_closable)
            tab_widget.tabBar().setVisible(self._tab_bar_visible)
            tab_widget.setNewTabButtonVisible(self._new_tab_button_visible)
            if self._corner_widget is not None:
                tab_widget.setCornerWidget(self._corner_widget, self._corner)
            self.set_root_widget(tab_widget)
        tab_widget.addTab(w, display)
        self._set_tab_tooltip(tab_widget, tab_widget.count() - 1, tooltip)
        tab_widget.setCurrentWidget(w)
        w.show()
        self.set_active_tab_widget(tab_widget)
        self.layoutChanged.emit()
        return True

    def isTabVisible(self, index: int) -> bool:
        """Return whether the tab at ``index`` is currently visible."""
        w = self.widget(index)
        if w is None or w in self._hidden_widgets:
            return False
        return self._tab_widget_for_page(w) is not None

    def visibleCount(self) -> int:
        """Return the number of currently visible dock tabs."""
        return sum(1 for index in range(self.count()) if self.isTabVisible(index))

    def hiddenIndexes(self) -> list[int]:
        """Return absolute indexes of hidden dock tabs."""
        return [
            self._all_widgets.index(widget)
            for widget in self._hidden_widgets
            if widget in self._all_widgets
        ]

    def canCloseTab(self, index: int) -> bool:
        """Return whether a close request may close or hide ``index``."""
        return self.isTabVisible(index) and self.visibleCount() > 1

    def _tab_widget_for_page(self, widget: QtWidgets.QWidget) -> DockTabWidget | None:
        """Return the tab widget containing ``widget``."""
        for tw in self.findChildren(DockTabWidget):
            for local_index in range(tw.count()):
                if tw.widget(local_index) is widget:
                    return tw
        return None


    def setCurrentIndex(self, index: int) -> None:
        """Activate the tab at the given absolute index.

        Parameters
        ----------
        index : int
            The absolute tab index to activate.
        """
        w = self.widget(index)
        for tw in self.findChildren(DockTabWidget):
            for i in range(tw.count()):
                if tw.widget(i) is w:
                    tw.setCurrentIndex(i)
                    self.set_active_tab_widget(tw)
                    return

    def indexOf(self, widget: QtWidgets.QWidget) -> int:
        """Return the absolute index of the given widget.

        Parameters
        ----------
        widget : QWidget
            The page widget to locate.

        Returns
        -------
        int
        """
        if widget in self._all_widgets:
            return self._all_widgets.index(widget)
        return -1

    def setTabsClosable(self, closable: bool) -> None:
        """Set whether tabs are closable on all internal DockTabWidgets.

        Parameters
        ----------
        closable : bool
            Whether tabs are closable.
        """
        self._tabs_closable = closable
        for tw in self.findChildren(DockTabWidget):
            tw.setTabsClosable(closable)

    def setTabBarVisible(self, visible: bool) -> None:
        """Show or hide the tab bar on all internal DockTabWidgets.

        Parameters
        ----------
        visible : bool
            Whether the tab bar should be visible.
        """
        self._tab_bar_visible = visible
        for tw in self.findChildren(DockTabWidget):
            tw.tabBar().setVisible(visible)

    def setDocumentMode(self, enabled: bool) -> None:
        """Set document mode on all internal DockTabWidgets.

        Parameters
        ----------
        enabled : bool
            Whether document mode is enabled.
        """
        for tw in self.findChildren(DockTabWidget):
            tw.setDocumentMode(enabled)

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
        full_name = self._tab_names.get(widget, title)
        display, tooltip = _shorten_path(full_name)

        def _insert(tw, idx):
            if idx < 0:
                tw.addTab(widget, display)
                tw.setCurrentWidget(widget)
            else:
                tw.insertTab(idx, widget, display)
                tw.setCurrentIndex(idx)
            if widget in self._tab_names:
                self._set_tab_tooltip(tw, tw.indexOf(widget), tooltip)

        if zone == "center":
            if source_tw == target_tw:
                # Reorder tab inside the same widget
                insert_idx = target_tw.tabBar().tabAt(pos)
                source_tw.removeTab(source_idx)
                _insert(target_tw, insert_idx)
            else:
                # Move tab to target tab widget
                source_tw.removeTab(source_idx)
                insert_idx = target_tw.tabBar().tabAt(pos)
                _insert(target_tw, insert_idx)
                self.cleanup_empty_tab_widget(source_tw)
        else:
            # Split drop
            source_tw.removeTab(source_idx)
            new_tw = DockTabWidget(self)
            new_tw.setTabsClosable(self._tabs_closable)
            new_tw.tabBar().setVisible(self._tab_bar_visible)
            new_tw.setNewTabButtonVisible(self._new_tab_button_visible)
            if self._corner_widget is not None:
                new_tw.setCornerWidget(self._corner_widget, self._corner)
            new_tw.addTab(widget, display)
            self._set_tab_tooltip(new_tw, new_tw.count() - 1, tooltip)
            self.split_tab_widget(target_tw, new_tw, zone)
            self.cleanup_empty_tab_widget(source_tw)
            self.set_active_tab_widget(new_tw)
        self.layoutChanged.emit()

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
        full_name = self._tab_names.get(widget, tw.tabText(index))
        tw.removeTab(index)

        display, tooltip = _shorten_path(full_name)
        main_tw.addTab(widget, display)
        self._set_tab_tooltip(main_tw, main_tw.count() - 1, tooltip)
        main_tw.setCurrentWidget(widget)

        self.cleanup_empty_tab_widget(tw)
        self.set_active_tab_widget(main_tw)
        self.layoutChanged.emit()

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
            full_name = self._tab_names.get(widget, tw.tabText(0))
            tw.removeTab(0)
            display, tooltip = _shorten_path(full_name)
            main_tw.addTab(widget, display)
            self._set_tab_tooltip(main_tw, main_tw.count() - 1, tooltip)

        self.cleanup_empty_tab_widget(tw)
        self.set_active_tab_widget(main_tw)
        self.layoutChanged.emit()

    def _request_close_tab(self, abs_index: int) -> None:
        """Request a tab close through the configured close policy."""
        if not self.canCloseTab(abs_index):
            return
        if self._close_tab_callback is not None:
            self._close_tab_callback(abs_index)
            return
        widget = self.widget(abs_index)
        close_mode = self._tab_close_modes.get(widget, "hide")
        if close_mode == "remove":
            self.removeTab(abs_index)
        else:
            self.hideTab(abs_index)
        self.tabCloseRequested.emit(abs_index)

    @staticmethod
    def _set_tab_tooltip(tw: DockTabWidget, local_index: int, tooltip: str) -> None:
        """Set a tooltip on a tab inside a DockTabWidget."""
        bar = tw.tabBar()
        if 0 <= local_index < bar.count():
            bar.setTabToolTip(local_index, tooltip)

    def _on_tab_context_menu(
        self, tw: 'DockTabWidget', local_index: int, global_pos: QtCore.QPoint
    ) -> None:
        """Build and show the right-click context menu for a tab.

        Close-related actions are handled directly. Document-oriented
        Save/Copy/Rename/Reload actions are only added in document mode.
        """
        if not self._context_menu_enabled:
            return
        # Resolve absolute index
        w = tw.widget(local_index)
        if w not in self._all_widgets:
            return
        abs_index = self._all_widgets.index(w)

        menu = QtWidgets.QMenu(self)

        # --- Close section ---
        action_close = menu.addAction("Close")
        action_close.setEnabled(self.canCloseTab(abs_index))
        action_close.triggered.connect(lambda idx=abs_index: self._request_close_tab(idx))

        action_close_except = menu.addAction("Close All Except Active Document")
        action_close_except.triggered.connect(lambda idx=abs_index: self._close_all_except(idx))

        action_close_left = menu.addAction("Close All to the Left")
        action_close_left.triggered.connect(lambda idx=abs_index: self._close_all_left(idx))

        action_close_right = menu.addAction("Close All to the Right")
        action_close_right.triggered.connect(lambda idx=abs_index: self._close_all_right(idx))

        self._add_dock_visibility_actions(menu)

        if self._context_menu_mode == "document":
            menu.addSeparator()

            # --- Save / Rename / Reload section ---
            action_save = menu.addAction("Save")
            action_save.triggered.connect(
                lambda: self.tabActionRequested.emit("save", abs_index)
            )

            action_save_as = menu.addAction("Save As...")
            action_save_as.triggered.connect(
                lambda: self.tabActionRequested.emit("save_as", abs_index)
            )

            action_rename = menu.addAction("Rename...")
            action_rename.triggered.connect(
                lambda: self.tabActionRequested.emit("rename", abs_index)
            )

            action_reload = menu.addAction("Reload")
            action_reload.triggered.connect(
                lambda: self.tabActionRequested.emit("reload", abs_index)
            )

            menu.addSeparator()

            # --- Copy section ---
            action_copy_path = menu.addAction("Copy Full Path")
            action_copy_path.triggered.connect(
                lambda: self.tabActionRequested.emit("copy_path", abs_index)
            )

            action_copy_name = menu.addAction("Copy File Name")
            action_copy_name.triggered.connect(
                lambda: self.tabActionRequested.emit("copy_name", abs_index)
            )

            action_copy_dir = menu.addAction("Copy File Directory")
            action_copy_dir.triggered.connect(
                lambda: self.tabActionRequested.emit("copy_dir", abs_index)
            )

        if self._context_menu_callback is not None:
            self._context_menu_callback(menu, abs_index)

        menu.exec_(global_pos)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Show dock-area context menu actions outside the tab bar.

        This path intentionally delegates to the same extension callback used
        by tab context menus so callers can expose actions such as reopening
        closed dock panes from any empty dock area.
        """
        if self._show_area_context_menu(event.globalPos()):
            event.accept()
            return
        super().contextMenuEvent(event)

    def _show_area_context_menu(self, global_pos: QtCore.QPoint) -> bool:
        """Show context-menu callback actions for the dock area.

        Parameters
        ----------
        global_pos : QtCore.QPoint
            Global screen position for the menu.

        Returns
        -------
        bool
            ``True`` when a menu was shown.

        """
        if not self._context_menu_enabled:
            return False
        menu = QtWidgets.QMenu(self)
        self._add_dock_visibility_actions(menu)
        if self._context_menu_callback is not None:
            self._context_menu_callback(menu, self.currentIndex())
        if not menu.actions():
            return False
        menu.exec_(global_pos)
        return True

    def _add_dock_visibility_actions(self, menu: QtWidgets.QMenu) -> None:
        """Add checkable dock visibility actions to ``menu``."""
        if not self._all_widgets:
            return
        if menu.actions():
            menu.addSeparator()
        visible_count = self.visibleCount()
        for index in range(self.count()):
            action = menu.addAction(self.tabText(index))
            action.setCheckable(True)
            visible = self.isTabVisible(index)
            action.setChecked(visible)
            if visible and visible_count <= 1:
                action.setEnabled(False)
            action.triggered.connect(
                lambda checked=False, idx=index: self._set_dock_visible(idx, checked)
            )

    def _set_dock_visible(self, index: int, visible: bool) -> None:
        """Set dock visibility from a checkable context-menu action."""
        if visible:
            self.showTab(index)
        else:
            self.hideTab(index)

    def _close_all_except(self, keep_index: int) -> None:
        """Close every tab except the one at ``keep_index``."""
        for i in range(self.count() - 1, keep_index, -1):
            self._request_close_tab(i)
        for _ in range(keep_index):
            self._request_close_tab(0)

    def _close_all_left(self, anchor: int) -> None:
        """Close all tabs to the left of ``anchor``."""
        for _ in range(anchor):
            self._request_close_tab(0)

    def _close_all_right(self, anchor: int) -> None:
        """Close all tabs to the right of ``anchor``."""
        for i in range(self.count() - 1, anchor, -1):
            self._request_close_tab(i)
