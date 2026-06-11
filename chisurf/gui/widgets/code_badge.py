"""
Code Badge Widget for Dev Mode Source Jumping.

A tiny floating `</>` overlay button that appears on key UI areas
when Dev mode is enabled. Clicking opens the embedded editor at the
relevant source location.
"""
from __future__ import annotations

import inspect
import typing
from typing import Callable, Optional, Tuple

import chisurf.core.settings
from chisurf.gui import QtCore, QtGui, QtWidgets


class CodeBadgeButton(QtWidgets.QToolButton):
    """A small floating code badge button for jumping to source.

    Features:
    - Text: `</>` (monospace, small)
    - Flat: no border, no background, AutoRaise
    - Semi-transparent until hover
    - Only visible in dev mode
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        target_resolver: Optional[Callable[[], Optional[Tuple[str, int]]]] = None,
    ):
        super().__init__(parent)
        self._target_resolver = target_resolver
        self._opacity_normal = 0.4
        self._opacity_hover = 1.0

        self._setup_ui()
        self.set_editor_font_from_settings()
        self._update_visibility()

    def set_editor_font_from_settings(self) -> None:
        """Apply the configured code editor font family to the badge."""
        font = QtGui.QFont()
        font.setFamily(chisurf.core.settings.gui['editor']['font_family'])
        font.setStyleHint(QtGui.QFont.Monospace)
        font.setPointSize(max(7, min(10, int(chisurf.core.settings.gui['editor']['font_size']) - 1)))
        self.setFont(font)

    def _setup_ui(self):
        self.setText("</>")
        self.setFixedSize(20, 16)
        self.setAutoRaise(True)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip("Open source (Dev mode)")

        self.setStyleSheet("""
            QToolButton {
                border: none;
                background: transparent;
                color: rgba(120, 180, 220, 180);
                padding: 0px;
                margin: 0px;
            }
            QToolButton:hover {
                color: rgba(120, 180, 220, 255);
            }
        """)

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)

        self.clicked.connect(self._on_clicked)

    def set_editor_font(self, font: QtGui.QFont) -> None:
        """Apply an editor font to the badge while keeping it compact."""
        badge_font = QtGui.QFont(font)
        badge_font.setStyleHint(QtGui.QFont.Monospace)
        badge_font.setPointSize(max(7, min(10, int(font.pointSize()) - 1)))
        self.setFont(badge_font)

    def _update_visibility(self):
        visible = chisurf.core.settings.is_dev_mode()
        self.setVisible(visible)
        if visible:
            self.setOpacity(self._opacity_normal)

    def setOpacity(self, opacity: float):
        effect = QtWidgets.QGraphicsOpacityEffect(self)
        effect.setOpacity(opacity)
        self.setGraphicsEffect(effect)

    def enterEvent(self, event: QtGui.QEnterEvent):
        super().enterEvent(event)
        if chisurf.core.settings.is_dev_mode():
            self.setOpacity(self._opacity_hover)

    def leaveEvent(self, event: QtCore.QEvent):
        super().leaveEvent(event)
        if chisurf.core.settings.is_dev_mode():
            self.setOpacity(self._opacity_normal)

    def _on_clicked(self):
        if not chisurf.core.settings.is_dev_mode():
            return

        if self._target_resolver is None:
            self._show_no_target_message()
            return

        try:
            result = self._target_resolver()
            if result is None:
                self._show_no_target_message()
                return

            path, line = result if len(result) == 2 else (result[0], None)
            self._open_in_editor(path, line)
        except Exception as e:
            chisurf.logging.error(f"Code badge error: {e}")
            self._show_error_message(str(e))

    def _open_in_editor(self, path: str, line: Optional[int] = None):
        try:
            import chisurf.gui.devtools.source_jump as sj
            main_window = self._find_main_window()
            if main_window:
                sj.open_in_editor(main_window, path, line)
            else:
                self._show_error_message("Could not find main window")
        except ImportError:
            self._show_error_message("Source jump module not available")

    def _find_main_window(self) -> Optional[QtWidgets.QMainWindow]:
        widget = self.parent()
        while widget is not None:
            if isinstance(widget, QtWidgets.QMainWindow):
                return widget
            widget = widget.parent()
        return None

    def _show_no_target_message(self):
        QtWidgets.QMessageBox.information(
            self,
            "No Source Target",
            "Could not resolve a source file for this widget.",
        )

    def _show_error_message(self, message: str):
        QtWidgets.QMessageBox.warning(
            self,
            "Code Badge Error",
            f"Could not open source: {message}",
        )

    def refresh_visibility(self):
        """Public method to refresh visibility based on dev mode."""
        self._update_visibility()


class CodeBadgeManager(QtCore.QObject):
    """Manages code badges for widgets, handling positioning and lifecycle."""

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._badges: typing.Dict[int, CodeBadgeButton] = {}

    def install_badge(
        self,
        widget: QtWidgets.QWidget,
        target_resolver: Optional[Callable[[], Optional[Tuple[str, int]]]] = None,
        corner: str = "top-right",
        margin: int = 4,
    ) -> Optional[CodeBadgeButton]:
        """Install a code badge on a widget.

        Args:
            widget: The widget to install the badge on
            target_resolver: Callable that returns (path, line) or None
            corner: Position corner ("top-right", "top-left", "bottom-right", "bottom-left")
            margin: Margin from edges in pixels

        Returns:
            The created badge or None if dev mode is off
        """
        if not chisurf.core.settings.is_dev_mode():
            return None

        widget_id = id(widget)
        if widget_id in self._badges:
            return self._badges[widget_id]

        badge = CodeBadgeButton(widget, target_resolver)
        badge._corner = corner
        badge._margin = margin
        badge.setToolTip(f"{widget.__class__.__name__}")

        self._badges[widget_id] = badge
        self._position_badge(badge, widget, corner, margin)

        widget.installEventFilter(self)

        return badge

    def remove_badge(self, widget: QtWidgets.QWidget):
        """Remove the code badge from a widget."""
        widget_id = id(widget)
        if widget_id in self._badges:
            badge = self._badges.pop(widget_id)
            badge.deleteLater()
            widget.removeEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
            widget_id = id(obj)
            if widget_id in self._badges:
                badge = self._badges[widget_id]
                self._position_badge(
                    badge,
                    obj,
                    getattr(badge, "_corner", "top-right"),
                    getattr(badge, "_margin", 4),
                )
        return super().eventFilter(obj, event)

    def _position_badge(
        self,
        badge: CodeBadgeButton,
        widget: QtWidgets.QWidget,
        corner: str,
        margin: int,
    ):
        """Position the badge in the specified corner of the widget."""
        if not widget:
            return

        widget_rect = widget.rect()
        badge_size = badge.sizeHint()

        if corner == "top-right":
            x = widget_rect.right() - badge_size.width() - margin
            y = widget_rect.top() + margin
        elif corner == "top-left":
            x = widget_rect.left() + margin
            y = widget_rect.top() + margin
        elif corner == "bottom-right":
            x = widget_rect.right() - badge_size.width() - margin
            y = widget_rect.bottom() - badge_size.height() - margin
        elif corner == "bottom-left":
            x = widget_rect.left() + margin
            y = widget_rect.bottom() - badge_size.height() - margin
        else:
            x = widget_rect.right() - badge_size.width() - margin
            y = widget_rect.top() + margin

        badge.move(x, y)
        badge.raise_()

    def refresh_all(self):
        """Refresh visibility of all managed badges."""
        for badge in self._badges.values():
            badge.refresh_visibility()


_badge_manager: Optional[CodeBadgeManager] = None


def get_badge_manager() -> CodeBadgeManager:
    """Get the global badge manager instance."""
    global _badge_manager
    if _badge_manager is None:
        _badge_manager = CodeBadgeManager()
    return _badge_manager


def install_code_badge(
    widget: QtWidgets.QWidget,
    target_resolver: Optional[Callable[[], Optional[Tuple[str, int]]]] = None,
    corner: str = "top-right",
    margin: int = 4,
) -> Optional[CodeBadgeButton]:
    """Install a code badge on a widget.

    Convenience function that uses the global badge manager.

    Args:
        widget: The widget to install the badge on
        target_resolver: Callable that returns (path, line) or None
        corner: Position corner
        margin: Margin from edges in pixels

    Returns:
        The created badge or None if dev mode is off
    """
    return get_badge_manager().install_badge(widget, target_resolver, corner, margin)


def remove_code_badge(widget: QtWidgets.QWidget):
    """Remove the code badge from a widget."""
    get_badge_manager().remove_badge(widget)


def make_widget_source_resolver(widget: QtWidgets.QWidget) -> Callable[[], Optional[Tuple[str, int]]]:
    """Create a source resolver for a widget.

    Priority:
    1. Widget's class source file via inspect
    2. If widget has _chisurf_ui_path attribute, return (.ui, 1)
    """
    def resolver() -> Optional[Tuple[str, int]]:
        try:
            ui_path = getattr(widget, "_chisurf_ui_path", None)
            if ui_path:
                return (str(ui_path), 1)

            widget_class = widget.__class__
            try:
                file_path = inspect.getsourcefile(widget_class)
                if file_path:
                    lines, start_line = inspect.getsourcelines(widget_class)
                    return (file_path, start_line)
            except (TypeError, OSError):
                pass

            return None
        except Exception:
            return None

    return resolver


def make_object_source_resolver(obj: object) -> Callable[[], Optional[Tuple[str, int]]]:
    """Create a source resolver for any Python object."""
    def resolver() -> Optional[Tuple[str, int]]:
        try:
            obj_class = obj.__class__
            try:
                file_path = inspect.getsourcefile(obj_class)
                if file_path:
                    lines, start_line = inspect.getsourcelines(obj_class)
                    return (file_path, start_line)
            except (TypeError, OSError):
                pass

            return None
        except Exception:
            return None

    return resolver
