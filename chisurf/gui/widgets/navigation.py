"""Reusable left-navigation stacked tool shell."""

from __future__ import annotations

import importlib
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

from qtpy import QtCore, QtWidgets


class NavigationPanelTool(QtWidgets.QMainWindow):
    """Main-window shell with a left selector and lazy-loaded right panels."""

    def __init__(
        self,
        *,
        title: str,
        panels: Sequence[Mapping[str, Any]],
        parent: QtWidgets.QWidget | None = None,
        minimum_size: tuple[int, int] = (850, 550),
        initial_size: tuple[int, int] = (1020, 680),
        navigation_width: int = 220,
        navigation_min_width: int | None = None,
        panel_margins: tuple[int, int, int, int] = (12, 12, 12, 12),
        searchable: bool = True,
    ) -> None:
        """Create a navigation shell.

        ``searchable`` (default ``True``) adds a search box at the top of the left
        pane that filters the navigation list to matching panels.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*initial_size)
        self.setMinimumSize(*minimum_size)

        self.panels: list[dict[str, Any]] = [dict(panel) for panel in panels]
        self._panel_margins = panel_margins
        self._searchable = searchable
        self._navigation_min_width = navigation_min_width or navigation_width
        self._build_ui(navigation_width)

    def _build_ui(self, navigation_width: int) -> None:
        """Build the navigation and stacked panel area."""
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # Left pane: a search box on top of the navigation list.
        left_pane = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.nav_search: QtWidgets.QLineEdit | None = None
        if self._searchable:
            self.nav_search = QtWidgets.QLineEdit()
            self.nav_search.setPlaceholderText("Search…")
            self.nav_search.setClearButtonEnabled(True)
            self.nav_search.setStyleSheet("QLineEdit { margin: 6px 10px 2px 10px; }")
            self.nav_search.textChanged.connect(self._on_search_changed)
            left_layout.addWidget(self.nav_search)

        self.nav_list = QtWidgets.QListWidget()
        self.nav_list.setMinimumWidth(self._navigation_min_width)
        self.nav_list.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.nav_list.setIconSize(QtCore.QSize(20, 20))
        self.nav_list.setSpacing(4)
        self.nav_list.setStyleSheet(
            """
            QListWidget {
                border: none;
                border-right: 1px solid rgba(128, 128, 128, 0.3);
                padding-top: 5px;
            }
            QListWidget::item {
                height: 32px;
                padding-left: 10px;
                border-radius: 8px;
                margin: 2px 10px;
                font-weight: bold;
                font-size: 14px;
            }
            """
        )

        for panel in self.panels:
            item = QtWidgets.QListWidgetItem(self._panel_label(panel))
            description = panel.get("description")
            if description:
                item.setToolTip(str(description))
            if panel.get("separator"):
                item.setFlags(QtCore.Qt.NoItemFlags)
            self.nav_list.addItem(item)

        left_layout.addWidget(self.nav_list, 1)
        self.splitter.addWidget(left_pane)

        self.stacked_widget = QtWidgets.QStackedWidget()
        for panel in self.panels:
            placeholder = self._placeholder_widget(panel)
            self.stacked_widget.addWidget(placeholder)
            panel["instance"] = None

        self.splitter.addWidget(self.stacked_widget)
        self.splitter.setCollapsible(0, False)
        self.splitter.setSizes(
            [
                max(navigation_width, self._navigation_min_width),
                max(600, self.width() - navigation_width),
            ]
        )
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        assert self.nav_list.minimumWidth() >= self._navigation_min_width

        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        if self.panels:
            self.nav_list.setCurrentRow(0)

    def _on_search_changed(self, text: str) -> None:
        """Filter the nav list to panels whose name matches ``text``.

        Leaf panels are shown when the (case-insensitive) query is a substring of
        their name; a separator group header is shown only while at least one of
        its child panels is still visible. An empty query restores everything.
        """
        query = (text or "").strip().lower()

        # First pass: leaf visibility (separators hidden, decided in pass two).
        for i, panel in enumerate(self.panels):
            item = self.nav_list.item(i)
            if item is None:
                continue
            if panel.get("separator"):
                item.setHidden(bool(query))
            else:
                name = str(panel.get("name") or "").lower()
                item.setHidden(bool(query) and query not in name)

        if not query:
            return

        # Second pass: reveal a group header only if its group has a visible child.
        sep_row: int | None = None
        group_has_visible = False
        for i, panel in enumerate(self.panels):
            if panel.get("separator"):
                if sep_row is not None:
                    self.nav_list.item(sep_row).setHidden(not group_has_visible)
                sep_row = i
                group_has_visible = False
            elif not self.nav_list.item(i).isHidden():
                group_has_visible = True
        if sep_row is not None:
            self.nav_list.item(sep_row).setHidden(not group_has_visible)

    def _panel_label(self, panel: Mapping[str, Any]) -> str:
        """Return the selector label for a panel (flagged when experimental)."""
        icon = str(panel.get("icon") or "").strip()
        name = str(panel.get("name") or "").strip()
        label = f"{icon} {name}".strip()
        if panel.get("experimental"):
            label = f"{label}  ⚠"
        return label

    def _placeholder_widget(self, panel: Mapping[str, Any]) -> QtWidgets.QWidget:
        """Create an unloaded placeholder widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(*self._panel_margins)

        label = QtWidgets.QLabel(f"Select {panel.get('name', 'panel')} to load.")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        return widget

    def _on_nav_changed(self, index: int) -> None:
        """Load and show the selected panel."""
        if index < 0 or index >= len(self.panels):
            return

        was_active = self.window().isActiveWindow()
        panel = self.panels[index]
        if panel.get("separator"):
            return
        if panel["instance"] is None:
            panel["instance"] = self._load_panel(panel, index)

        self.stacked_widget.setCurrentWidget(panel["instance"])
        if was_active:
            QtCore.QTimer.singleShot(0, self._restore_active_window)

    def _load_panel(self, panel: dict[str, Any], index: int) -> QtWidgets.QWidget:
        """Load a panel and replace its placeholder in the stack."""
        try:
            widget = self._create_panel_widget(panel)
            wrapper = self._wrap_panel(widget, panel)
        except Exception as exc:  # pragma: no cover - GUI error path
            traceback.print_exc()
            wrapper = self._error_widget(panel, exc)

        placeholder = self.stacked_widget.widget(index)
        self.stacked_widget.removeWidget(placeholder)
        placeholder.deleteLater()
        self.stacked_widget.insertWidget(index, wrapper)
        return wrapper

    def _create_panel_widget(self, panel: Mapping[str, Any]) -> QtWidgets.QWidget:
        """Instantiate a panel widget from its definition."""
        factory = panel.get("factory")
        if factory is not None:
            widget = factory(self)
        else:
            module = importlib.import_module(str(panel["class_path"]))
            widget_class = getattr(module, str(panel["class_name"]))
            widget = widget_class(parent=self)

        if not isinstance(widget, QtWidgets.QWidget):
            raise TypeError(f"Panel {panel.get('name')!r} did not create a QWidget")
        return widget

    def _wrap_panel(
        self, widget: QtWidgets.QWidget, panel: Mapping[str, Any] | None = None
    ) -> QtWidgets.QWidget:
        """Wrap a panel widget with margins, child-window flags and an experimental banner.

        For panels flagged experimental, a prominent warning banner is prepended.
        """
        wrapper = QtWidgets.QWidget()
        self._prepare_embedded_widget(widget, wrapper)
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(*self._panel_margins)
        if panel is not None and panel.get("experimental"):
            layout.addWidget(self._experimental_banner(panel))
        layout.addWidget(widget)
        return wrapper

    def _experimental_banner(self, panel: Mapping[str, Any]) -> QtWidgets.QLabel:
        """Build the red 'experimental / untested' banner for an experimental panel."""
        msg = panel.get("experimental_message") or (
            f"{panel.get('name', 'This tool')} is EXPERIMENTAL and UNTESTED — "
            "results are not validated"
        )
        banner = QtWidgets.QLabel(f"⚠  {msg}")
        banner.setAlignment(QtCore.Qt.AlignCenter)
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "QLabel { background-color: #b30000; color: white; font-weight: bold; "
            "font-size: 14px; padding: 5px; border-bottom: 2px solid #7d0000; }"
        )
        return banner

    def _prepare_embedded_widget(
        self,
        widget: QtWidgets.QWidget,
        parent: QtWidgets.QWidget,
    ) -> None:
        """Force lazily-loaded tools to behave as child widgets."""
        widget.setAttribute(QtCore.Qt.WA_QuitOnClose, False)
        widget.setAttribute(QtCore.Qt.WA_DontCreateNativeAncestors, True)
        widget.setWindowFlags(QtCore.Qt.Widget)
        widget.setParent(parent)

    def _restore_active_window(self) -> None:
        """Keep the hosting tool active after a lazy page is embedded."""
        window = self.window()
        window.raise_()
        window.activateWindow()
        self.nav_list.setFocus(QtCore.Qt.OtherFocusReason)

    def _error_widget(self, panel: Mapping[str, Any], exc: Exception) -> QtWidgets.QWidget:
        """Create an error panel for failed lazy imports."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(*self._panel_margins)

        label = QtWidgets.QLabel(f"Failed to load {panel.get('name', 'panel')}:\n{exc}")
        label.setWordWrap(True)
        label.setStyleSheet("color: red; font-size: 13px; font-weight: bold;")
        layout.addWidget(label)
        layout.addStretch()
        return widget
