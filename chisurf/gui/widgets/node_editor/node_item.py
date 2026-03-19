from __future__ import annotations

from typing import List, Optional

from qtpy import QtCore, QtGui, QtWidgets

from .model import NodeModel
from .port_item import NodePortGraphicsItem
from .theme import color as theme_color, metric as theme_metric


class NodeGraphicsItem(QtWidgets.QGraphicsPathItem):
    """View/controller for a NodeModel.

    Draws the node, embeds the optional widget, positions ports, and
    manages interaction like selection and collapsing.
    """

    def __init__(
        self,
        model: NodeModel,
        *,
        width: float = 190.0,
        title_height: float = 26.0,
        min_body_height: float = 60.0,
        radius: float = 8.0,
        collapsed: bool = False,
    ):
        super().__init__()

        self.model = model
        self.title = model.title
        self.inputs = model.inputs
        self.outputs = model.outputs

        # Instantiate content widget once from factory (if any)
        self.content_widget = model.content_factory() if model.content_factory else None
        self.proxy: Optional[QtWidgets.QGraphicsProxyWidget] = None
        self.port_items: List[NodePortGraphicsItem] = []

        # Geometry constants
        self.width = width
        self.title_height = title_height
        self.min_body_height = min_body_height
        self.radius = radius
        self.collapsed = collapsed

        # Track width before collapse so we can restore it when expanding.
        self._width_before_collapse: Optional[float] = None

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        self.setZValue(1)

        # Soft drop shadow
        effect = QtWidgets.QGraphicsDropShadowEffect()
        effect.setBlurRadius(18)
        effect.setOffset(0, 4)
        effect.setColor(QtGui.QColor(0, 0, 0, 180))
        self.setGraphicsEffect(effect)

        # Resizing state
        self._resizing = False
        self._resize_start_x = 0.0
        self._resize_initial_width = self.width
        self._resize_margin = float(theme_metric("node_resize_margin", 6.0))

        # Separate state for vertical (height) resize via bottom edge
        self._resizing_height = False
        self._resizing_both = False
        self._resize_start_y = 0.0
        self._resize_initial_body_height = self.min_body_height
        # Optional manual body height override, used when the user resizes
        # vertically. When None, the height follows content sizeHint.
        self._manual_body_height: Optional[float] = None

        self._build_path()
        self._create_ports()
        self.add_widget()

        # Ensure geometry is consistent after the content widget is added so
        # the proxy and inner widgets are resized to the final node size.
        self._build_path()
        self._layout_ports()
        self.update()

    # ----- Geometry and painting -----------------------------------------
    def _build_path(self):
        if self.collapsed:
            body_height = 18
        else:
            # Base height and width determined by minimums and content widget
            base_height = self.min_body_height
            min_width = float(theme_metric("node_min_width", 150))
            if self.content_widget is not None:
                hint = self.content_widget.sizeHint()
                base_height = max(base_height, hint.height() + 8)
                # Ensure width is at least large enough to fit content plus margins
                left_margin = float(theme_metric("node_content_margin_left", 8.0))
                right_margin = float(theme_metric("node_content_margin_right", 8.0))
                self.width = max(self.width, hint.width() + left_margin + right_margin)

            if self._manual_body_height is not None:
                body_height = max(base_height, float(self._manual_body_height))
            else:
                body_height = base_height

        rect = QtCore.QRectF(0, 0, self.width, self.title_height + body_height)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, self.radius, self.radius)
        self.setPath(path)

        # Keep embedded content stretched to the node width with fixed margins
        if self.proxy is not None and self.content_widget is not None:
            left_margin = float(theme_metric("node_content_margin_left", 8.0))
            right_margin = float(theme_metric("node_content_margin_right", 8.0))
            top_offset = float(theme_metric("node_content_top_offset", 4.0))
            content_width = max(10.0, rect.width() - (left_margin + right_margin))
            body_height = rect.height() - self.title_height - top_offset
            if body_height < 10.0:
                body_height = 10.0
            # Resize the proxy itself so layouts inside can use the full width
            self.proxy.setPos(left_margin, self.title_height + top_offset)
            self.proxy.resize(content_width, body_height)
            try:
                self.content_widget.setMinimumWidth(content_width)
                self.content_widget.setMaximumWidth(content_width)
            except Exception:
                pass

    def _create_ports(self):
        # Clear existing ports
        for p in self.port_items:
            p.setParentItem(None)
        self.port_items.clear()

        # Create port items (positions are set in _layout_ports)
        for i, spec in enumerate(self.inputs):
            port = NodePortGraphicsItem(self, spec, i)
            self.port_items.append(port)

        for j, spec in enumerate(self.outputs):
            port = NodePortGraphicsItem(self, spec, len(self.inputs) + j)
            self.port_items.append(port)

        self._layout_ports()

    def _layout_ports(self):
        if not self.port_items:
            return
        if self.collapsed:
            y_start_offset = float(theme_metric("node_port_y_start_collapsed", 4.0))
            y_step = float(theme_metric("node_port_y_step_collapsed", 10.0))
        else:
            y_start_offset = float(theme_metric("node_port_y_start", 8.0))
            y_step = float(theme_metric("node_port_y_step", 14.0))
        y_start = self.title_height + y_start_offset
        width = self.path().boundingRect().width()

        # Inputs on left
        for i, _ in enumerate(self.inputs):
            port = self.port_items[i]
            if self.collapsed:
                y = y_start
            else:
                y = y_start + i * y_step
            port.setPos(0, y)

        # Outputs on right
        for j, _ in enumerate(self.outputs):
            port = self.port_items[len(self.inputs) + j]
            if self.collapsed:
                y = y_start
            else:
                y = y_start + j * y_step
            port.setPos(width, y)

    def add_widget(self):
        if self.content_widget is None:
            return
        self.proxy = QtWidgets.QGraphicsProxyWidget(self)
        self.proxy.setWidget(self.content_widget)
        # Place inside body with a small left/top margin; inner layouts take
        # care of padding so widgets don't touch the borders.
        self.proxy.setPos(8, self.title_height + 4)

    # ----- Z-order helpers -----------------------------------------------
    def bring_to_front(self) -> None:
        """Raise this node above other NodeGraphicsItems in the same scene.

        This mirrors the "Bring to front" context menu action so that
        selection or double-clicking can also promote a node visually.
        """

        sc = self.scene()
        if sc is None:
            return
        try:
            from .node_item import NodeGraphicsItem as RealNodeGraphicsItem  # type: ignore[import-self]
        except Exception:  # pragma: no cover - defensive
            RealNodeGraphicsItem = NodeGraphicsItem  # type: ignore[assignment]

        max_z = self.zValue()
        try:
            for it in sc.items():
                if isinstance(it, RealNodeGraphicsItem):
                    if it is self:
                        continue
                    z = float(it.zValue())
                    if z > max_z:
                        max_z = z
        except Exception:
            pass
        self.setZValue(max_z + 1.0)

    def set_collapsed(self, collapsed: bool):
        if self.collapsed == collapsed:
            return

        previously_collapsed = self.collapsed
        self.collapsed = collapsed

        # When collapsing, remember the current width and shrink to a compact
        # default/minimum width. When expanding, restore the remembered width
        # so manual horizontal resizing is preserved.
        if collapsed and not previously_collapsed:
            try:
                self._width_before_collapse = float(self.width)
            except Exception:
                self._width_before_collapse = self.width
            try:
                self.width = float(theme_metric("node_min_width", self.width))
            except Exception:
                pass
        elif not collapsed and previously_collapsed and self._width_before_collapse is not None:
            self.width = float(self._width_before_collapse)

        if self.proxy is not None:
            self.proxy.setVisible(not collapsed)
        # Hide/show ports when collapsed to reduce visual clutter and save
        # space; labels attached to the ports are children and will follow
        # the visibility state of their parent items.
        for port in self.port_items:
            try:
                port.setVisible(not collapsed)
            except Exception:
                continue
        self._build_path()
        self._layout_ports()
        self.update()

    def toggle_collapsed(self):
        self.set_collapsed(not self.collapsed)

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent):
        rect = self.path().boundingRect()
        pos = event.pos()
        at_right = rect.width() - self._resize_margin <= pos.x() <= rect.width() + self._resize_margin
        at_bottom = rect.height() - self._resize_margin <= pos.y() <= rect.height() + self._resize_margin
        # Change cursor when hovering near the right edge in the body area
        if (
            not self.collapsed
            and at_right
            and at_bottom
        ):
            self.setCursor(QtCore.Qt.SizeFDiagCursor)
        elif at_right and pos.y() >= self.title_height:
            self.setCursor(QtCore.Qt.SizeHorCursor)
        # Or near the bottom edge for vertical resize (height), when not collapsed
        elif (
            not self.collapsed
            and at_bottom
            and 0.0 <= pos.x() <= rect.width()
        ):
            self.setCursor(QtCore.Qt.SizeVerCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        rect = self.path().boundingRect()
        pos = event.pos()
        at_right = rect.width() - self._resize_margin <= pos.x() <= rect.width() + self._resize_margin
        at_bottom = rect.height() - self._resize_margin <= pos.y() <= rect.height() + self._resize_margin
        if event.button() == QtCore.Qt.LeftButton:
            # On primary-button press, bring this node to the front so it
            # visually sits above others while being interacted with.
            try:
                self.bring_to_front()
            except Exception:
                pass
        if (
            event.button() == QtCore.Qt.LeftButton
            and not self.collapsed
            and at_right
            and at_bottom
        ):
            self._resizing_both = True
            self._resizing = False
            self._resizing_height = False
            self._resize_start_x = pos.x()
            self._resize_start_y = pos.y()
            self._resize_initial_width = self.width
            self._resize_initial_body_height = max(10.0, rect.height() - self.title_height)
            event.accept()
            return
        if (
            event.button() == QtCore.Qt.LeftButton
            and at_right
            and pos.y() >= self.title_height
        ):
            # Start horizontal resize
            self._resizing = True
            self._resize_start_x = pos.x()
            self._resize_initial_width = self.width
            event.accept()
            return
        if (
            event.button() == QtCore.Qt.LeftButton
            and not self.collapsed
            and at_bottom
            and 0.0 <= pos.x() <= rect.width()
        ):
            # Start vertical resize (adjust body height via bottom edge)
            self._resizing_height = True
            self._resize_start_y = pos.y()
            # Current body height is total minus title bar
            self._resize_initial_body_height = max(10.0, rect.height() - self.title_height)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self._resizing_both:
            dx = event.pos().x() - self._resize_start_x
            dy = event.pos().y() - self._resize_start_y
            min_width = float(theme_metric("node_min_width", 120.0))
            new_width = max(min_width, self._resize_initial_width + dx)
            base_min = self.min_body_height
            new_body = max(base_min, self._resize_initial_body_height + dy)
            changed = False
            if abs(new_width - self.width) > 0.5:
                self.width = new_width
                changed = True
            if self._manual_body_height is None or abs(new_body - self._manual_body_height) > 0.5:
                self._manual_body_height = new_body
                changed = True
            if changed:
                self._build_path()
                self._layout_ports()
                self.update()
            event.accept()
            return
        if self._resizing:
            dx = event.pos().x() - self._resize_start_x
            min_width = float(theme_metric("node_min_width", 120.0))
            new_width = max(min_width, self._resize_initial_width + dx)
            if abs(new_width - self.width) > 0.5:
                self.width = new_width
                self._build_path()
                self._layout_ports()
                self.update()
            event.accept()
            return
        if self._resizing_height:
            dy = event.pos().y() - self._resize_start_y
            # For manual vertical resize, allow shrinking below the content's
            # preferred size, clamping only to the configured minimum body
            # height. Inner widgets (like PT Transform's editor) will scroll
            # as needed.
            base_min = self.min_body_height
            new_body = max(base_min, self._resize_initial_body_height + dy)
            if self._manual_body_height is None or abs(new_body - self._manual_body_height) > 0.5:
                self._manual_body_height = new_body
                self._build_path()
                self._layout_ports()
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            if self._resizing_both:
                self._resizing_both = False
                event.accept()
                return
            if self._resizing:
                self._resizing = False
                event.accept()
                return
            if self._resizing_height:
                self._resizing_height = False
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def paint(self, painter: QtGui.QPainter, option, widget=None):
        rect = self.path().boundingRect()
        title_rect = QtCore.QRectF(rect.x(), rect.y(), rect.width(), self.title_height)
        body_rect = QtCore.QRectF(
            rect.x(),
            rect.y() + self.title_height,
            rect.width(),
            rect.height() - self.title_height,
        )

        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Border color: stronger accent when selected
        border_color = (
            theme_color("node_border_selected", (90, 180, 255))
            if self.isSelected()
            else theme_color("node_border", (40, 40, 40))
        )
        cfg_border = None
        try:
            cfg_border = self.model.config
        except Exception:
            cfg_border = None
        if isinstance(cfg_border, dict):
            rgb_border = cfg_border.get("border_color")
            if rgb_border is not None:
                try:
                    r_b, g_b, b_b = int(rgb_border[0]), int(rgb_border[1]), int(rgb_border[2])
                    base_border = QtGui.QColor(r_b, g_b, b_b)
                    border_color = base_border.lighter(135) if self.isSelected() else base_border
                except Exception:
                    pass
        border_pen = QtGui.QPen(border_color, 1.8 if self.isSelected() else 1.4)
        border_pen.setCosmetic(True)
        painter.setPen(border_pen)

        # Body gradient (slightly lighter at top)
        body_grad = QtGui.QLinearGradient(body_rect.topLeft(), body_rect.bottomLeft())
        body_top = theme_color("node_body_top", (62, 68, 74))
        body_bottom = theme_color("node_body_bottom", (48, 52, 58))

        cfg_body = None
        try:
            cfg_body = self.model.config
        except Exception:
            cfg_body = None
        if isinstance(cfg_body, dict):
            rgb_body = cfg_body.get("body_color")
            if rgb_body is not None:
                try:
                    r_bd, g_bd, b_bd = int(rgb_body[0]), int(rgb_body[1]), int(rgb_body[2])
                    base_body = QtGui.QColor(r_bd, g_bd, b_bd)
                    body_top = base_body.lighter(115)
                    body_bottom = base_body.darker(110)
                except Exception:
                    pass

        body_grad.setColorAt(0.0, body_top)
        body_grad.setColorAt(1.0, body_bottom)
        painter.setBrush(body_grad)
        painter.drawRoundedRect(body_rect, 8, 8)

        # Title bar gradient
        title_grad = QtGui.QLinearGradient(title_rect.topLeft(), title_rect.bottomLeft())
        # Allow per-node-type header colors via theme.json
        t = getattr(self.model, "node_type", None)
        type_key = str(t) if t is not None else ""
        type_key = type_key.replace(" ", "_")

        if self.isSelected():
            top_key = f"node_title_top_selected_{type_key}" if type_key else "node_title_top_selected"
            bot_key = f"node_title_bottom_selected_{type_key}" if type_key else "node_title_bottom_selected"
            top_col = theme_color(top_key, theme_color("node_title_top_selected", (88, 156, 204)).getRgb()[:3])
            bot_col = theme_color(bot_key, theme_color("node_title_bottom_selected", (60, 120, 170)).getRgb()[:3])
        else:
            top_key = f"node_title_top_{type_key}" if type_key else "node_title_top"
            bot_key = f"node_title_bottom_{type_key}" if type_key else "node_title_bottom"
            top_col = theme_color(top_key, theme_color("node_title_top", (70, 130, 175)).getRgb()[:3])
            bot_col = theme_color(bot_key, theme_color("node_title_bottom", (52, 105, 145)).getRgb()[:3])

        # Optional per-node override from model.config: a "title_color" RGB
        # triplet makes this node use its own header color independent of the
        # theme/type defaults.
        cfg = None
        try:
            cfg = self.model.config
        except Exception:
            cfg = None
        if isinstance(cfg, dict):
            rgb = cfg.get("title_color")
            if rgb is not None:
                try:
                    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
                    base = QtGui.QColor(r, g, b)
                    if self.isSelected():
                        top_col = base.lighter(130)
                        bot_col = base.darker(110)
                    else:
                        top_col = base.lighter(120)
                        bot_col = base.darker(105)
                except Exception:
                    pass

        title_grad.setColorAt(0.0, top_col)
        title_grad.setColorAt(1.0, bot_col)
        painter.setBrush(title_grad)
        painter.drawRoundedRect(title_rect, 8, 8)

        # Chevron
        chevron_rect = QtCore.QRectF(title_rect.x() + 8, title_rect.y() + (self.title_height - 10) / 2, 10, 10)
        path = QtGui.QPainterPath()
        if self.collapsed:
            path.moveTo(chevron_rect.left(), chevron_rect.top())
            path.lineTo(chevron_rect.right(), chevron_rect.center().y())
            path.lineTo(chevron_rect.left(), chevron_rect.bottom())
        else:
            path.moveTo(chevron_rect.left(), chevron_rect.top())
            path.lineTo(chevron_rect.right(), chevron_rect.top())
            path.lineTo(chevron_rect.center().x(), chevron_rect.bottom())
        painter.fillPath(path, QtGui.QColor(230, 230, 230))

        # Title text
        left_padding = int((chevron_rect.right() - title_rect.left()) + 8)
        text_rect = title_rect.adjusted(left_padding, 0, -6, 0)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        painter.drawText(text_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, self.title)

        # Optional icon
        icon_cfg = None
        try:
            icon_cfg = self.model.config.get("icon")
        except Exception:
            icon_cfg = None
        if icon_cfg:
            icon_rect = QtCore.QRectF(title_rect.right() - 18, title_rect.y() + 4, 14, 14)
            if isinstance(icon_cfg, QtGui.QIcon):
                icon_cfg.paint(painter, icon_rect.toRect())
            elif isinstance(icon_cfg, QtGui.QPixmap):
                painter.drawPixmap(icon_rect.topLeft(), icon_cfg.scaled(int(icon_rect.width()), int(icon_rect.height()), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            elif isinstance(icon_cfg, str):
                pm = QtGui.QPixmap(icon_cfg)
                if not pm.isNull():
                    painter.drawPixmap(icon_rect.topLeft(), pm.scaled(int(icon_rect.width()), int(icon_rect.height()), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        # Wrap collapse/expand in an undo step when an editor with an
        # undo_stack is available. This keeps UI behaviour the same while
        # making the change undoable.
        scene = self.scene()
        owner = scene.parent() if scene is not None else None
        if owner is not None and hasattr(owner, "begin_undo"):
            try:
                text = "Open node" if self.collapsed else "Close node"
                owner.begin_undo(text)  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            # Promote to front on double-click as well so the focused node is
            # never visually obscured by overlapping neighbours.
            try:
                self.bring_to_front()
            except Exception:
                pass
            self.toggle_collapsed()
        finally:
            if owner is not None and hasattr(owner, "commit_undo"):
                try:
                    owner.commit_undo()  # type: ignore[attr-defined]
                except Exception:
                    pass
        event.accept()
