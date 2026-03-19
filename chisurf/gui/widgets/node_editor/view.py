from __future__ import annotations

from typing import Optional

from qtpy import QtCore, QtGui, QtWidgets

from .theme import color as theme_color


class NodeView(QtWidgets.QGraphicsView):
    """View with pan & zoom for the node scene."""

    def __init__(self, scene: QtWidgets.QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self._prev_drag_mode: Optional[QtWidgets.QGraphicsView.DragMode] = None
        self._timeline_item: Optional[QtWidgets.QGraphicsItem] = None
        self._autofit_item: Optional[QtWidgets.QGraphicsItem] = None

        # Create the auto-fit overlay button ("A" icon) in the lower-left
        # corner of the view. The item ignores transformations so it stays a
        # constant on-screen size; its position is managed by
        # _update_timeline_position to keep it anchored to the viewport.
        try:
            self._autofit_item = _AutoFitButtonItem(self)  # type: ignore[name-defined]
            if scene is not None:
                scene.addItem(self._autofit_item)
        except Exception:
            self._autofit_item = None

    def wheelEvent(self, event: QtGui.QWheelEvent):
        # Zoom
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.12 if delta > 0 else 1 / 1.12
        # AnchorUnderMouse ensures we zoom into the region beneath the cursor.
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        self._update_timeline_position()
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        item = self.itemAt(event.pos())
        if event.button() == QtCore.Qt.LeftButton and item is None:
            # Switch to hand-drag mode for panning
            self._prev_drag_mode = self.dragMode()
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            super().mousePressEvent(event)
            return

        elif event.button() == QtCore.Qt.MiddleButton:
            # Switch to hand-drag mode for panning
            self._prev_drag_mode = self.dragMode()
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            # Convert to left-button drag for panning
            fake_event = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonPress,
                event.localPos(),
                event.screenPos(),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                event.modifiers(),
            )
            super().mousePressEvent(fake_event)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton and self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            super().mouseReleaseEvent(event)
            if self._prev_drag_mode is not None:
                self.setDragMode(self._prev_drag_mode)
            else:
                self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.unsetCursor()
            return

        elif event.button() == QtCore.Qt.MiddleButton:
            fake_event = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonRelease,
                event.localPos(),
                event.screenPos(),
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoButton,
                event.modifiers(),
            )
            super().mouseReleaseEvent(fake_event)
            # Restore previous drag mode and cursor
            if self._prev_drag_mode is not None:
                self.setDragMode(self._prev_drag_mode)
            else:
                self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.unsetCursor()
            return
        super().mouseReleaseEvent(event)

    # ----- Zoom helpers used by menus/toolbars ---------------------------
    def zoom(self, factor: float):
        self.scale(factor, factor)
        self._update_timeline_position()

    def zoom_in(self):
        self.zoom(1.15)

    def zoom_out(self):
        self.zoom(1 / 1.15)

    def fit_all(self):
        scene = self.scene()
        if scene is None:
            return
        rect = scene.itemsBoundingRect()
        if rect.isNull():
            return
        margin = 40
        rect = rect.adjusted(-margin, -margin, margin, margin)
        self.fitInView(rect, QtCore.Qt.KeepAspectRatio)
        self._update_timeline_position()

    def reset_zoom(self):
        self.resetTransform()
        self._update_timeline_position()

    def resizeEvent(self, event: QtGui.QResizeEvent):  # type: ignore[override]
        super().resizeEvent(event)
        self._update_timeline_position()

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # type: ignore[override]
        super().scrollContentsBy(dx, dy)
        self._update_timeline_position()

    # ----- Overlay helpers -----------------------------------------------

    def set_timeline_item(self, item: Optional[QtWidgets.QGraphicsItem]) -> None:
        self._timeline_item = item
        self._update_timeline_position()

    def _update_timeline_position(self) -> None:
        rect = self.viewport().rect()
        if rect.isEmpty():
            return
        margin_bottom = 6
        bottom = rect.bottom() - margin_bottom
        try:
            # For items with ItemIgnoresTransformations set, the view applies
            # only the translation part of the transform (no scaling). To keep
            # overlays visually pinned to the viewport, use the transform's
            # translation directly.
            t = self.transform()
            tx = float(t.m31())
            ty = float(t.m32())

            # Timeline: anchored to bottom-center in viewport coordinates.
            if self._timeline_item is not None:
                br_t = self._timeline_item.boundingRect()
                vx = float(rect.center().x())
                vy = float(bottom)
                x_t = vx - tx - br_t.width() / 2.0
                y_t = vy - ty - br_t.height()
                self._timeline_item.setPos(x_t, y_t)

            # Auto-fit button: anchored to bottom-left with a small margin.
            if self._autofit_item is not None:
                br_a = self._autofit_item.boundingRect()
                left_margin = 10.0
                vx_a = float(rect.left()) + left_margin
                vy_a = float(bottom)
                x_a = vx_a - tx
                y_a = vy_a - ty - br_a.height()
                self._autofit_item.setPos(x_a, y_a)
        except RuntimeError:
            # Underlying C++ item was already deleted; drop the reference.
            self._timeline_item = None
            self._autofit_item = None


class _AutoFitButtonItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, view: NodeView):  # type: ignore[name-defined]
        super().__init__()
        self._view = view
        self.setZValue(1005.0)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)

        rect = QtCore.QRectF(0.5, 0.5, 22.0, 18.0)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 4.0, 4.0)
        self.setPath(path)

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        self.setCursor(QtCore.Qt.PointingHandCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.LeftButton:
            try:
                self._view.fit_all()
            except Exception:
                pass
            event.accept()
            return
        super().mousePressEvent(event)

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:  # type: ignore[override]
        rect = self.path().boundingRect()
        if rect.isEmpty():
            return

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        bg = theme_color("autofit_button_background", (60, 64, 70))
        border = theme_color("autofit_button_border", (30, 30, 34))
        icon_col = theme_color("autofit_button_icon", (235, 235, 235))

        painter.setPen(QtGui.QPen(border, 1))
        painter.setBrush(QtGui.QBrush(bg))
        painter.drawRoundedRect(rect, 4.0, 4.0)

        # Draw a simple "A" using white lines (no text glyph) so it matches
        # the rest of the node editor iconography.
        inner = rect.adjusted(6.0, 4.0, -6.0, -4.0)
        left = QtCore.QPointF(inner.left(), inner.bottom())
        right = QtCore.QPointF(inner.right(), inner.bottom())
        top = QtCore.QPointF((inner.left() + inner.right()) / 2.0, inner.top())
        mid_y = (inner.top() + inner.bottom()) / 2.0
        bar_left = QtCore.QPointF(inner.left() + (inner.width() * 0.2), mid_y)
        bar_right = QtCore.QPointF(inner.right() - (inner.width() * 0.2), mid_y)

        pen = QtGui.QPen(icon_col, 2)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawLine(left, top)
        painter.drawLine(right, top)
        painter.drawLine(bar_left, bar_right)
