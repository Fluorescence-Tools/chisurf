from __future__ import annotations

from typing import Optional, Tuple

from qtpy import QtCore, QtGui, QtWidgets


class InlineLabeledSlider(QtWidgets.QWidget):
    """Compact labeled slider with value text inside and click-to-edit mode.

    This widget is used by the example "Constant" node to provide a small
    inline slider with a numeric value display and optional text editing.
    """

    valueChanged = QtCore.Signal(float)
    editingFinished = QtCore.Signal(float)
    editingStarted = QtCore.Signal()

    def __init__(
        self,
        text: str,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 3,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._min = float(minimum)
        self._max = float(maximum) if maximum != minimum else float(minimum) + 1.0
        self._dec = max(0, min(6, int(decimals)))
        self._value = float(value)
        self._m = 3
        self.setMinimumHeight(18)
        self.setMaximumHeight(18)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self._editor: Optional[QtWidgets.QLineEdit] = None
        self._editing = False

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        return QtCore.QSize(120, 18)

    def setRange(self, minimum: float, maximum: float) -> None:
        self._min = float(minimum)
        self._max = float(maximum) if maximum != minimum else float(minimum) + 1.0
        self.update()

    def setDecimals(self, decs: int) -> None:
        self._dec = max(0, min(6, int(decs)))
        self.update()

    def setValue(self, v: float) -> None:
        v = max(self._min, min(self._max, float(v)))
        step = 10 ** (-(self._dec + 1))
        if abs(v - self._value) > step:
            self._value = v
            self.valueChanged.emit(self._value)
            self.update()

    def value(self) -> float:
        return self._value

    def _ratio(self) -> float:
        if self._max == self._min:
            return 0.0
        return max(0.0, min(1.0, (self._value - self._min) / (self._max - self._min)))

    def _rects(self) -> Tuple[QtCore.QRectF, QtCore.QRectF, QtCore.QRectF]:
        extra_right = 2
        r = QtCore.QRectF(self.rect()).adjusted(self._m, self._m, -(self._m + extra_right), -self._m)
        half_w = r.width() * 0.5
        label_rect = QtCore.QRectF(r.left() + 6, r.top(), half_w * 0.8 - 6, r.height())
        value_rect = QtCore.QRectF(r.left() + half_w, r.top(), half_w * 1.2 - 11, r.height())
        return r, label_rect, value_rect

    def _set_from_pos(self, pos: QtCore.QPointF) -> None:
        r, _, _ = self._rects()
        if r.width() <= 0:
            return
        t = max(r.left(), min(r.right(), pos.x()))
        ratio = (t - r.left()) / r.width()
        v = self._min + ratio * (self._max - self._min)
        step = 10 ** (-self._dec)
        v = round(v / step) * step
        self.setValue(v)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if not self._editing:
            self._editing = True
            self.editingStarted.emit()
        _r, _lr, vr = self._rects()
        if vr.contains(e.pos()):
            self._start_edit()
            return
        self._set_from_pos(e.pos())

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        QtWidgets.QWidget.mouseReleaseEvent(self, e)
        if self._editing:
            self._editing = False
            self.editingFinished.emit(self._value)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if e.buttons() & QtCore.Qt.LeftButton:
            self._set_from_pos(e.pos())

    def wheelEvent(self, e: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        if not self._editing:
            self._editing = True
            self.editingStarted.emit()
        delta = e.angleDelta().y() / 120.0
        step = (self._max - self._min) / 100.0
        self.setValue(self._value + delta * step)
        self._editing = False
        self.editingFinished.emit(self._value)

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(e)
        if self._editor and self._editor.isVisible():
            self._position_editor()

    def _position_editor(self) -> None:
        _r, _lr, vr = self._rects()
        geom = QtCore.QRect(int(vr.left()), int(vr.top()), int(vr.width()), int(vr.height()))
        self._editor.setGeometry(geom)

    def _start_edit(self) -> None:
        if not self._editing:
            self._editing = True
            self.editingStarted.emit()
        if self._editor is None:
            self._editor = QtWidgets.QLineEdit(self)
            self._editor.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self._editor.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self._editor.setStyleSheet("border: 0px; background: transparent; color: rgb(240,240,240);")
            self._editor.returnPressed.connect(self._commit_edit)
            self._editor.editingFinished.connect(self._commit_edit)
        self._editor.setText(f"{self._value:.{self._dec}f}")
        self._position_editor()
        self._editor.show()
        self._editor.setFocus(QtCore.Qt.OtherFocusReason)
        self._editor.selectAll()
        self.update()

    def _commit_edit(self) -> None:
        if not self._editor:
            return
        text = self._editor.text().strip()
        try:
            v = float(text)
            self.setValue(v)
        except Exception:
            pass
        self._editing = False
        self.editingFinished.emit(self._value)
        self._editor.hide()
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r, label_rect, value_rect = self._rects()
        base = QtGui.QColor(120, 120, 120)
        fill = QtGui.QColor(70, 120, 190)
        text_col = QtGui.QColor(240, 240, 240)
        p.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 1))
        p.setBrush(base)
        p.drawRoundedRect(r, 4, 4)
        fr = QtCore.QRectF(r)
        fr.setRight(r.left() + r.width() * self._ratio())
        p.setBrush(fill)
        p.setPen(QtCore.Qt.NoPen)
        path = QtGui.QPainterPath()
        path.addRoundedRect(fr, 4, 4)
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(QtCore.QRectF(r), 4, 4)
        p.setClipPath(clip)
        p.fillPath(path, fill)
        p.setClipping(False)
        p.setPen(text_col)
        p.setFont(self.font())
        p.drawText(label_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, self._text)
        if not (self._editor and self._editor.isVisible()):
            val_text = f"{self._value:.{self._dec}f}"
            p.drawText(value_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter, val_text)
