from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

from .theme import color as theme_color, metric as theme_metric, text as theme_text


def apply_node_ui_theme(root: QtWidgets.QWidget) -> None:
    """Apply a consistent dark theme to child widgets in a node content area."""

    base_font = root.font()
    family = theme_text("node_font_family", base_font.defaultFamily())
    size_pt = int(theme_metric("node_font_size", 10))
    base_font.setFamily(family)
    base_font.setPointSize(size_pt)
    root.setFont(base_font)

    size_px = size_pt  # simple mapping pt->px for our small UI
    sb_size = int(theme_metric("scrollbar_width", 8))
    sb_handle_min = int(theme_metric("scrollbar_handle_min_length", 18))
    btn_h = int(theme_metric("node_button_height", size_px * 2))
    sb_track = theme_color("scrollbar_track", (40, 44, 52))
    sb_handle = theme_color("scrollbar_handle", (120, 125, 135))
    sb_handle_hover = theme_color("scrollbar_handle_hover", (145, 155, 170))

    qss = f"""
    QLabel {{ color: rgb(235,235,235); background: transparent; padding: 0px; font-size: {size_px}px; }}
    /* Slightly lighter than node body (~55,60,65) */
    QComboBox {{ background: rgb(72, 80, 90); color: rgb(235,235,235); border: 1px solid rgb(40,40,40); border-radius: 4px; padding: 2px 18px 2px 6px; font-size: {size_px}px; min-height: 18px; }}
    QComboBox:hover {{ border: 1px solid rgb(60,120,160); }}
    QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 16px; border: 0px; background: transparent; }}
    QComboBox::down-arrow {{ width: 8px; height: 8px; }}
    QComboBox QAbstractItemView {{ background: rgb(60,64,70); color: rgb(235,235,235); selection-background-color: rgb(60,120,160); selection-color: white; outline: 0; border: 1px solid rgb(40,40,40); padding: 2px; }}
    QComboBox QAbstractItemView::item {{ padding: 3px 6px; min-height: 18px; }}
    QComboBox QAbstractItemView::item:hover {{ background: rgb(60,120,160); }}
    QPushButton {{ color: rgb(235,235,235); background: rgb(70,120,160); border: 0px; border-radius: 4px; padding: 2px 8px; font-size: {size_px}px; min-height: {btn_h}px; max-height: {btn_h}px; }}
    QPushButton:hover {{ background: rgb(80,130,175); }}
    QPushButton:pressed {{ background: rgb(60,110,145); }}
    QCheckBox, QRadioButton {{ color: rgb(235,235,235); background: transparent; font-size: {size_px}px; spacing: 6px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 12px; height: 12px; }}
    QCheckBox::indicator {{ border-radius: 2px; border:1px solid rgb(80,80,80); background: rgb(60,60,60); }}
    QCheckBox::indicator:checked {{ background: rgb(70,120,190); border: 1px solid rgb(90,140,210); }}
    QRadioButton::indicator {{ border-radius: 7px; border:1px solid rgb(80,80,80); background: rgb(60,60,60); }}
    QRadioButton::indicator:checked {{ background: rgb(70,120,190); border: 1px solid rgb(90,140,210); }}
    QScrollBar:vertical {{ background: rgb({sb_track.red()},{sb_track.green()},{sb_track.blue()}); width: {sb_size}px; margin: 0px; border-radius: 3px; }}
    QScrollBar::handle:vertical {{ background: rgb({sb_handle.red()},{sb_handle.green()},{sb_handle.blue()}); border-radius: 3px; min-height: {sb_handle_min}px; }}
    QScrollBar::handle:vertical:hover {{ background: rgb({sb_handle_hover.red()},{sb_handle_hover.green()},{sb_handle_hover.blue()}); }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
    QScrollBar:horizontal {{ background: rgb({sb_track.red()},{sb_track.green()},{sb_track.blue()}); height: {sb_size}px; margin: 0px; border-radius: 3px; }}
    QScrollBar::handle:horizontal {{ background: rgb({sb_handle.red()},{sb_handle.green()},{sb_handle.blue()}); border-radius: 3px; min-width: {sb_handle_min}px; }}
    QScrollBar::handle:horizontal:hover {{ background: rgb({sb_handle_hover.red()},{sb_handle_hover.green()},{sb_handle_hover.blue()}); }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
    """
    root.setStyleSheet(qss)


class StyledComboBox(QtWidgets.QComboBox):
    """ComboBox with custom painting and edit-start/finish signals."""

    editingStarted = QtCore.Signal()
    editingFinished = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setMinimumHeight(18)
        self._before_index = -1

    def showPopup(self) -> None:  # type: ignore[override]
        super().showPopup()
        # The popup is a separate top-level QFrame; force it opaque so the
        # QGraphicsScene background doesn't bleed through on macOS.
        container = self.view().parentWidget() if self.view() else None
        if container is not None:
            container.setAttribute(QtCore.Qt.WA_TranslucentBackground, False)
            container.setAutoFillBackground(True)
        self._before_index = self.currentIndex()
        self.editingStarted.emit()

    def hidePopup(self) -> None:  # type: ignore[override]
        super().hidePopup()
        try:
            if self._before_index != -1 and self._before_index != self.currentIndex():
                self.editingFinished.emit(self.currentText())
        finally:
            self._before_index = -1

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        super().paintEvent(e)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = self.rect()
        size = 8
        cx = r.right() - 10
        cy = r.center().y()
        path = QtGui.QPainterPath()
        path.moveTo(cx - size / 2, cy - size / 4)
        path.lineTo(cx + size / 2, cy - size / 4)
        path.lineTo(cx, cy + size / 4)
        path.closeSubpath()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(235, 235, 235))
        p.drawPath(path)
