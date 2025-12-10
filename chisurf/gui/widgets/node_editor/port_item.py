from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

from .model import PortSpec
from .theme import color as theme_color


class NodePortGraphicsItem(QtWidgets.QGraphicsEllipseItem):
    """Small circle representing an input or output port on a node."""

    RADIUS = 5.0

    def __init__(self, node_item: "NodeGraphicsItem", spec: PortSpec, index: int):
        size = self.RADIUS * 2
        super().__init__(-self.RADIUS, -self.RADIUS, size, size, node_item)
        self.setBrush(QtGui.QBrush(theme_color("port_neutral", (135, 135, 135))))
        pen = QtGui.QPen(QtGui.QColor(10, 10, 10), 1.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsScenePositionChanges)
        self.setAcceptHoverEvents(True)

        self.node_item = node_item
        self.spec = spec
        self.index = index
        self.connected_count = 0

        # Optional text label for certain node types (e.g. PT Transform).
        self._label: QtWidgets.QGraphicsSimpleTextItem | None = None
        node_type = getattr(getattr(node_item, "model", None), "node_type", "")
        if str(node_type) == "pt_transform":
            self._label = QtWidgets.QGraphicsSimpleTextItem(self)
            self._label.setText(self.spec.name)
            font = self._label.font()
            try:
                font.setPointSize(max(7, font.pointSize() - 1))
            except Exception:
                pass
            self._label.setFont(font)
            self._label.setBrush(QtGui.QBrush(theme_color("port_label_text", (235, 235, 235))))
            self._update_label_pos()

        # Optional type label (e.g. sF, sI, vF, vI) shown above the port.
        self._type_label: QtWidgets.QGraphicsSimpleTextItem | None = None
        port_type = getattr(self.spec, "port_type", "") or ""
        if port_type:
            self._type_label = QtWidgets.QGraphicsSimpleTextItem(self)
            self._type_label.setText(str(port_type))
            tfont = self._type_label.font()
            try:
                tfont.setPointSize(max(6, tfont.pointSize() - 2))
            except Exception:
                pass
            self._type_label.setFont(tfont)
            self._type_label.setBrush(QtGui.QBrush(theme_color("port_type_text", (220, 220, 220))))
            self._update_type_label_pos()

        # Ensure visual state (color/outline) reflects fixed/bounded flags.
        self._update_color()

    def set_connected(self, count: int):
        self.connected_count = max(0, int(count))
        self._update_color()
        # Optional UX hook: for certain nodes, hide the embedded editor and
        # show only a label when an *input* port is wired. Nodes opt in via
        # model.config['hide_value_when_connected'] and a content widget that
        # implements ``set_label_only(bool)`` (e.g. TextBoxWidget).
        if self.connected_count is None:  # type: ignore[comparison-overlap]
            return
        if self.spec.is_output:
            return
        try:
            node = self.node_item
            model = getattr(node, "model", None)
            cfg = getattr(model, "config", None)
        except Exception:
            return
        if not isinstance(cfg, dict) or not cfg.get("hide_value_when_connected", False):
            return
        widget = getattr(node, "content_widget", None)
        if widget is None or not hasattr(widget, "set_label_only"):
            return
        try:
            widget.set_label_only(self.connected_count > 0)
        except Exception:
            pass

    def _update_label_pos(self) -> None:
        if self._label is None:
            return
        margin = 4.0
        br = self._label.boundingRect()
        if self.spec.is_output:
            # Outputs: label to the left of the port circle
            x = -br.width() - margin
        else:
            # Inputs: label to the right of the port circle
            x = margin
        y = -br.height() / 2.0
        self._label.setPos(x, y)

    def _update_type_label_pos(self) -> None:
        if self._type_label is None:
            return
        br = self._type_label.boundingRect()
        margin = 1.0
        x = -br.width() / 2.0
        y = -self.RADIUS - br.height() - margin
        self._type_label.setPos(x, y)

    def _update_color(self):
        """Update fill and outline based on connection, fixed and bounds."""
        is_fixed = bool(getattr(self.spec, "fixed", False))
        has_min = getattr(self.spec, "min_value", None) is not None
        has_max = getattr(self.spec, "max_value", None) is not None
        bounded = has_min or has_max

        # Fill color: fixed ports are red; otherwise use existing scheme.
        if is_fixed:
            base = theme_color("port_fixed", (210, 80, 80))
        elif self.connected_count > 0:
            base = (
                theme_color("port_output_connected", (235, 205, 70))
                if self.spec.is_output
                else theme_color("port_input_connected", (110, 170, 255))
            )
        else:
            base = theme_color("port_neutral", (135, 135, 135))
        self.setBrush(QtGui.QBrush(base))

        # Outline thickness: thicker when bounded, thin when unbounded.
        pen = self.pen()
        pen.setWidthF(2.0 if bounded else 1.0)
        pen.setCosmetic(True)
        self.setPen(pen)

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value):
        # When a port moves (because its node moved), update connected edges
        if change == QtWidgets.QGraphicsItem.ItemScenePositionHasChanged:
            scene = self.scene()
            # local import to avoid circulars
            from .scene import NodeScene  # noqa
            if isinstance(scene, NodeScene):
                scene.update_edges_for_port(self)
        return super().itemChange(change, value)

    def scene_pos(self) -> QtCore.QPointF:
        return self.sceneBoundingRect().center()

    def hoverEnterEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent):
        # Slightly enlarge and brighten on hover for better discoverability
        self.setScale(1.25)
        brush = self.brush()
        c = brush.color()
        c = QtGui.QColor(min(255, int(c.red() * 1.1)), min(255, int(c.green() * 1.1)), min(255, int(c.blue() * 1.1)))
        brush.setColor(c)
        self.setBrush(brush)
        return super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent):
        self.setScale(1.0)
        self._update_color()
        return super().hoverLeaveEvent(event)

    # ----- State helpers used by the scene/context menu -------------------
    def set_fixed(self, fixed: bool) -> None:
        try:
            self.spec.fixed = bool(fixed)
        except Exception:
            pass
        self._update_color()

    def update_bounds(self, min_value, max_value) -> None:
        try:
            self.spec.min_value = min_value
            self.spec.max_value = max_value
        except Exception:
            pass
        self._update_color()

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        """Toggle fixed/unfixed on double-click without affecting wiring."""
        current = bool(getattr(self.spec, "fixed", False))
        self.set_fixed(not current)
        event.accept()
