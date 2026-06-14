from __future__ import annotations

import copy
import math
from typing import Optional

from qtpy import QtCore, QtGui, QtWidgets

from .port_item import NodePortGraphicsItem
from .theme import color as theme_color


class EdgeGraphicsItem(QtWidgets.QGraphicsPathItem):
    """Bezier-like connection between two ports."""

    def __init__(self, start_port: NodePortGraphicsItem, end_port: Optional[NodePortGraphicsItem] = None):
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        self.temp_end_pos: Optional[QtCore.QPointF] = None

        self.setZValue(-1)  # under nodes
        self._is_in_cycle = False
        # Optional per-edge style overrides, typically populated from JSON
        self._normal_color_override: Optional[QtGui.QColor] = None
        self._cycle_color_override: Optional[QtGui.QColor] = None
        self._arrow_color_override: Optional[QtGui.QColor] = None
        self._style_config = {}
        self._config = {}

        self.set_cycle(False)

        # Allow edge selection so Delete key and rubber-band selection work
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

        self.update_path()

    def _draw_arrow(self, painter: QtGui.QPainter):
        # Determine logical direction: output -> input
        if self.end_port is not None:
            if self.start_port.spec.is_output and not self.end_port.spec.is_output:
                p_out_item = self.start_port
                p_in_item = self.end_port
            elif self.end_port.spec.is_output and not self.start_port.spec.is_output:
                p_out_item = self.end_port
                p_in_item = self.start_port
            else:
                p_out_item = self.start_port
                p_in_item = self.end_port

            p_out = p_out_item.scene_pos()
            p_in = p_in_item.scene_pos()
        elif self.temp_end_pos is not None:
            # While dragging, show direction from start port to mouse
            p_out = self.start_port.scene_pos()
            p_in = self.temp_end_pos
        else:
            return

        # Direction vector from output towards input
        dx = p_in.x() - p_out.x()
        dy = p_in.y() - p_out.y()
        length = math.hypot(dx, dy)
        if length == 0:
            return

        ux = dx / length
        uy = dy / length
        arrow_size = 16.0

        # Place the tip slightly before the input port so it is not hidden
        tip_offset = 12.0

        angle = math.radians(25.0)
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        left_x = ux * cos_a - uy * sin_a
        left_y = ux * sin_a + uy * cos_a
        right_x = ux * cos_a + uy * sin_a
        right_y = -ux * sin_a + uy * cos_a

        p_tip = QtCore.QPointF(
            p_in.x() - ux * tip_offset,
            p_in.y() - uy * tip_offset,
        )
        p1 = QtCore.QPointF(
            p_tip.x() + left_x * arrow_size,
            p_tip.y() + left_y * arrow_size,
        )
        p2 = QtCore.QPointF(
            p_tip.x() + right_x * arrow_size,
            p_tip.y() + right_y * arrow_size,
        )

        poly = QtGui.QPolygonF([p_tip, p1, p2])
        painter.save()
        base = theme_color(
            "arrow_in_cycle" if self._is_in_cycle else "arrow_normal",
            self.pen().color().getRgb()[:3],
        )
        arrow_col = self._arrow_color_override or base
        painter.setBrush(arrow_col)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawPolygon(poly)
        painter.restore()

    def set_temp_end_pos(self, pos: QtCore.QPointF):
        self.temp_end_pos = pos
        self.update_path()

    def set_end_port(self, port: NodePortGraphicsItem):
        self.end_port = port
        self.temp_end_pos = None
        self.update_path()

    def update_path(self):
        p1 = self.start_port.scene_pos()
        if self.end_port is not None:
            p2 = self.end_port.scene_pos()
        elif self.temp_end_pos is not None:
            p2 = self.temp_end_pos
        else:
            p2 = p1

        path = QtGui.QPainterPath(p1)
        dx = (p2.x() - p1.x()) * 0.5
        c1 = QtCore.QPointF(p1.x() + dx, p1.y())
        c2 = QtCore.QPointF(p2.x() - dx, p2.y())
        path.cubicTo(c1, c2, p2)
        self.setPath(path)

    def paint(self, painter: QtGui.QPainter, option, widget=None):
        super().paint(painter, option, widget)
        self._draw_arrow(painter)

    def set_cycle(self, in_cycle: bool):
        """Update this edge's pen/color based on whether it belongs to a cycle."""
        self._is_in_cycle = in_cycle
        if in_cycle:
            if self._cycle_color_override is not None:
                col = self._cycle_color_override
            else:
                col = theme_color("edge_in_cycle", (200, 60, 60))
        else:
            if self._normal_color_override is not None:
                col = self._normal_color_override
            else:
                col = theme_color("edge_normal", (160, 160, 160))
        pen = QtGui.QPen(col, 2)
        pen.setCosmetic(True)
        self.setPen(pen)

    def apply_config(self, cfg: dict) -> None:
        """Apply per-edge color overrides from a JSON-style config dict.

        Recognized keys (all optional):

        - "color": [r, g, b]          -> normal line color
        - "cycle_color": [r, g, b]    -> line color when in cycle
        - "arrow_color": [r, g, b]    -> arrow-head fill color
        """

        def _to_qcolor(value):
            try:
                if not isinstance(value, (list, tuple)) or len(value) != 3:
                    return None
                r, g, b = int(value[0]), int(value[1]), int(value[2])
                return QtGui.QColor(r, g, b)
            except Exception:
                return None

        if not isinstance(cfg, dict):
            return

        self._config = copy.deepcopy(cfg)
        style = {}

        col = _to_qcolor(cfg.get("color"))
        if col is not None:
            self._normal_color_override = col
            style["color"] = [col.red(), col.green(), col.blue()]

        cyc = _to_qcolor(cfg.get("cycle_color"))
        if cyc is not None:
            self._cycle_color_override = cyc
            style["cycle_color"] = [cyc.red(), cyc.green(), cyc.blue()]

        arr = _to_qcolor(cfg.get("arrow_color"))
        if arr is not None:
            self._arrow_color_override = arr
            style["arrow_color"] = [arr.red(), arr.green(), arr.blue()]

        if style:
            self._style_config = style
            self.set_cycle(self._is_in_cycle)
