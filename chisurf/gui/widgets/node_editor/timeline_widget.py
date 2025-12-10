from __future__ import annotations

from typing import Optional

from qtpy import QtCore, QtGui, QtWidgets
import sip

from .theme import color as theme_color


class TimelineWidget(QtWidgets.QGraphicsPathItem):
    """Simple horizontal timeline bound to a QUndoStack, drawn inside the scene.

    Each visible entry corresponds to one undo command. The item renders a
    light strip with command icons and a baseline with vertical bars marking
    states, with the current state highlighted.

    The item is intended to be added to the node scene with a high z-value so
    it floats above nodes and edges.
    """

    def __init__(self, parent: Optional[QtWidgets.QGraphicsItem] = None):
        super().__init__(parent)

        self._stack: Optional[QtWidgets.QUndoStack] = None
        # Filtering flags controlling which kinds of commands appear on the
        # timeline. "Moves" (pure node drags) and "folding" (open/close
        # nodes) are typically less relevant, so they are hidden by default
        # and can be toggled from the context menu.
        self._show_moves: bool = False
        self._show_folds: bool = False

        # List of (undo_stack_index, pixmap) pairs for visible commands.
        self._entries: list[tuple[int, QtGui.QPixmap]] = []
        # Cached icon rectangles (in local coordinates) for hit testing.
        self._icon_rects: list[QtCore.QRectF] = []

        # Optional embedded navigation controls hosted via QGraphicsProxyWidget
        # so they appear as part of the strip: back/forward buttons and the
        # "Moves" checkbox.
        self._back_proxy: Optional[QtWidgets.QGraphicsProxyWidget] = None
        self._forward_proxy: Optional[QtWidgets.QGraphicsProxyWidget] = None
        self._moves_proxy: Optional[QtWidgets.QGraphicsProxyWidget] = None
        self._back_button_width: float = 0.0
        self._forward_button_width: float = 0.0
        self._moves_width: float = 0.0
        self._button_spacing: float = 4.0

        # Geometry and layout
        # Slightly smaller icons so the timeline feels lighter and the
        # playhead/step markers stand out more.
        self._icon_size = QtCore.QSizeF(8.0, 8.0)
        self._icon_spacing = 4.0
        self._pad_left = 6.0
        self._pad_right = 10.0
        self._pad_top = 4.0
        self._height = 26.0
        self._baseline_margin = 5.0

        # Position indicator (scrubbable playhead) state
        self._indicator_x: float = 0.0
        self._indicator_hit_rect: QtCore.QRectF = QtCore.QRectF()
        self._indicator_dragging: bool = False
        self._drag_preview_active: bool = False
        self._drag_preview_index: int = 0
        self._indicator_x0: float = 0.0
        self._indicator_step: float = 0.0
        self._indicator_uniform: bool = False
        self._controls_right_x: float = 0.0

        # Float above all nodes/edges and keep a constant on-screen size
        # regardless of view zoom.
        self.setZValue(1000.0)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)

        # Base visuals similar to the previous QListView strip.
        self.setPen(QtGui.QPen(QtGui.QColor(210, 210, 210)))
        self.setBrush(QtGui.QBrush(QtGui.QColor(245, 245, 245)))

        self._update_path()

    # --- Internal helpers -------------------------------------------------

    def _stack_alive(self) -> Optional[QtWidgets.QUndoStack]:
        """Return the undo stack if its C++ object is still alive.

        When the owning editor is being torn down, the QUndoStack may be
        deleted on the C++ side before this graphics item is destroyed. In
        that case any method call on the stack is undefined behaviour and can
        segfault. This helper uses ``sip.isdeleted`` to detect that situation
        and turn it into a graceful "stack is None" so the timeline quietly
        stops updating instead of crashing.
        """

        stack = self._stack
        if stack is None:
            return None
        try:
            if sip.isdeleted(stack):  # type: ignore[arg-type]
                self._stack = None
                return None
        except Exception:
            # If sip is unavailable or behaves unexpectedly, fall back to the
            # existing reference; any further issues will be caught by
            # RuntimeError guards elsewhere.
            pass
        return stack

    def set_undo_stack(self, stack: Optional[QtWidgets.QUndoStack]) -> None:
        if self._stack is not None:
            try:
                self._stack.indexChanged.disconnect(self._on_stack_index_changed)
                self._stack.cleanChanged.disconnect(self._on_stack_clean_changed)
            except Exception:
                pass
        self._stack = stack
        if self._stack is not None:
            self._stack.indexChanged.connect(self._on_stack_index_changed)
            self._stack.cleanChanged.connect(self._on_stack_clean_changed)
        self._rebuild()

    def set_navigation_buttons(
        self,
        back_button: QtWidgets.QToolButton,
        forward_button: QtWidgets.QToolButton,
    ) -> None:
        """Host back/forward buttons inside the timeline strip.

        The buttons are wrapped in QGraphicsProxyWidget children so they are
        drawn as part of the graphics item while still emitting their usual
        Qt signals (already connected by the editor).
        """

        if back_button is not None:
            self._back_button_width = float(back_button.sizeHint().width())
            if self._back_proxy is None:
                self._back_proxy = QtWidgets.QGraphicsProxyWidget(self)
            self._back_proxy.setWidget(back_button)

        if forward_button is not None:
            self._forward_button_width = float(forward_button.sizeHint().width())
            if self._forward_proxy is None:
                self._forward_proxy = QtWidgets.QGraphicsProxyWidget(self)
            self._forward_proxy.setWidget(forward_button)

        # Apply a compact, rounded style so the arrows feel integrated with
        # the dark timeline strip instead of native OS buttons.
        try:
            fg = theme_color("timeline_button_foreground", (230, 230, 230))
            fg_hover = theme_color("timeline_button_foreground_hover", (255, 255, 255))
            bg = theme_color("timeline_button_background", (70, 76, 84))
            bg_hover = theme_color("timeline_button_background_hover", (90, 96, 104))
            border = theme_color("timeline_button_border", (40, 40, 40))

            style = (
                "QToolButton {"
                f" background-color: rgba({bg.red()},{bg.green()},{bg.blue()},220);"
                f" border: 1px solid rgb({border.red()},{border.green()},{border.blue()});"
                " border-radius: 4px; padding: 0px; margin: 0px;"
                " min-width: 18px; max-width: 18px; min-height: 18px; max-height: 18px;"
                f" color: rgb({fg.red()},{fg.green()},{fg.blue()});"
                " }"
                " QToolButton:hover {"
                f" background-color: rgba({bg_hover.red()},{bg_hover.green()},{bg_hover.blue()},240);"
                f" color: rgb({fg_hover.red()},{fg_hover.green()},{fg_hover.blue()});"
                " }"
                " QToolButton:pressed {"
                f" background-color: rgba({bg_hover.red()},{bg_hover.green()},{bg_hover.blue()},255);"
                " }"
            )
            if back_button is not None:
                back_button.setStyleSheet(style)
            if forward_button is not None:
                forward_button.setStyleSheet(style)
        except Exception:
            pass

        self._layout_icons()

    def set_moves_checkbox(self, checkbox: QtWidgets.QCheckBox) -> None:
        """Host the "Moves" checkbox inside the timeline controls area."""

        if checkbox is None:
            return
        self._moves_width = float(checkbox.sizeHint().width())
        if self._moves_proxy is None:
            self._moves_proxy = QtWidgets.QGraphicsProxyWidget(self)
        self._moves_proxy.setWidget(checkbox)

        # Slightly compact checkbox styling to fit the strip aesthetics and
        # follow the node editor theme. The label text is empty; the tooltip
        # describes the purpose.
        try:
            fg = theme_color("timeline_moves_foreground", (220, 220, 220))
            bg = theme_color("timeline_background", (58, 64, 72))
            border = theme_color("timeline_button_border", (40, 40, 40))
            accent = theme_color("timeline_tick_current", (90, 180, 255))
            style = (
                "QCheckBox {"
                " spacing: 0px; margin: 0px; padding: 0px; background: transparent;"
                f" color: rgb({fg.red()},{fg.green()},{fg.blue()});"
                " }"
                " QCheckBox::indicator {"
                " width: 12px; height: 12px; border-radius: 2px;"
                f" border: 1px solid rgb({border.red()},{border.green()},{border.blue()});"
                f" background-color: rgb({bg.red()},{bg.green()},{bg.blue()});"
                " }"
                " QCheckBox::indicator:checked {"
                f" background-color: rgb({accent.red()},{accent.green()},{accent.blue()});"
                " }"
                " QCheckBox::indicator:checked:hover {"
                f" background-color: rgb({min(accent.red()+10,255)},{min(accent.green()+10,255)},{min(accent.blue()+10,255)});"
                " }"
            )
            checkbox.setStyleSheet(style)
        except Exception:
            pass

        self._layout_icons()

    def set_show_moves(self, show: bool) -> None:
        if self._show_moves == show:
            return
        self._show_moves = bool(show)
        self._rebuild()

    def set_show_folds(self, show: bool) -> None:
        if self._show_folds == show:
            return
        self._show_folds = bool(show)
        self._rebuild()

    # --- Navigation helpers used by external buttons ---------------------

    def previous_visible_index(self, current_index: int) -> Optional[int]:
        """Return the previous visible undo stack index before *current_index*.

        Respects the current filtering (e.g. hiding move commands). Falls
        back to ``0`` when no earlier visible index exists but the stack has
        history.
        """

        if self._stack is None:
            return None
        prev: Optional[int] = None
        for idx, _pix in self._entries:
            if idx < current_index:
                prev = idx
            else:
                break
        if prev is not None:
            return prev
        if current_index > 0:
            return 0
        return None

    def next_visible_index(self, current_index: int) -> Optional[int]:
        """Return the next visible undo stack index after *current_index*.

        If there is no later visible command but the undo stack has a later
        state (``index() < count()``), the method returns ``count()`` so the
        caller can step all the way to the current end-of-stack state.
        """

        stack = self._stack_alive()
        if stack is None:
            return None
        for idx, _pix in self._entries:
            if idx > current_index:
                return idx
        count = stack.count()
        if current_index < count:
            return count
        return None

    def _compute_required_width(self) -> float:
        n = len(self._entries)
        controls_width = 0.0
        if self._back_button_width or self._forward_button_width or self._moves_width:
            controls_width = (
                self._back_button_width
                + self._forward_button_width
                + self._moves_width
            )
            # Spacing between back, forward, and moves
            controls_width += 3.0 * self._button_spacing
        width = self._pad_left + controls_width + self._pad_right
        if n > 0:
            width += n * self._icon_size.width() + max(0, n - 1) * self._icon_spacing
        return width

    def _update_path(self) -> None:
        min_width = 120.0
        width = max(min_width, self._compute_required_width())
        rect = QtCore.QRectF(0.5, 0.5, width, self._height)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 3.0, 3.0)
        self.setPath(path)

        # If we are hosted in a dedicated QGraphicsScene/QGraphicsView (the
        # standalone timeline widget case), keep the scene rect tightly
        # wrapped around the timeline geometry so the view can keep it nicely
        # centered without scrollbars.
        scn = self.scene()
        if scn is not None:
            try:
                scn.setSceneRect(rect)
            except Exception:
                pass

    def _rebuild(self) -> None:
        self._entries.clear()
        self._icon_rects.clear()
        stack = self._stack_alive()
        if stack is None:
            self._update_path()
            self.update()
            return
        count = stack.count()
        for i in range(count):
            cmd = stack.command(i)
            text = cmd.text() if cmd is not None else ""
            t = (text or "").lower()
            # Moves (simple drags) are optional
            if not self._show_moves and "move nodes" in t:
                continue
            # Folding/open-close of nodes is optional
            if not self._show_folds and ("open node" in t or "close node" in t):
                continue
            pix = self._make_icon_for_text(text or "")
            self._entries.append((i, pix))
        self._layout_icons()
        self.update()

    def _layout_icons(self) -> None:
        self._icon_rects.clear()
        self._update_path()
        rect = self.path().boundingRect()
        # Position navigation buttons (if any) centered vertically in the strip
        # and reserve horizontal space for them.
        button_center_y = rect.center().y()
        base_x = rect.left() + self._pad_left
        icons_start_x = base_x
        icons_right_limit = rect.right() - self._pad_right

        # Place back arrow at the far left, then forward arrow directly to its
        # right so both controls form a compact group on the left side.
        if self._back_proxy is not None:
            br = self._back_proxy.boundingRect()
            bx = base_x
            by = button_center_y - br.height() / 2.0
            self._back_proxy.setPos(bx, by)
            icons_start_x = bx + br.width() + self._button_spacing

        if self._forward_proxy is not None:
            fr = self._forward_proxy.boundingRect()
            fx = icons_start_x
            fy = button_center_y - fr.height() / 2.0
            self._forward_proxy.setPos(fx, fy)
            icons_start_x = fx + fr.width() + self._button_spacing

        # Moves checkbox comes last in the controls group, to the right of the
        # arrows.
        if self._moves_proxy is not None:
            mr = self._moves_proxy.boundingRect()
            mx = icons_start_x
            my = button_center_y - mr.height() / 2.0
            self._moves_proxy.setPos(mx, my)
            icons_start_x = mx + mr.width() + self._button_spacing

        # Remember where the controls section ends so we can paint a visual
        # separator between controls and history items.
        self._controls_right_x = icons_start_x

        x = icons_start_x
        y = rect.top() + self._pad_top
        h = self._icon_size.height()
        for _idx, _pix in self._entries:
            if x + self._icon_size.width() > icons_right_limit:
                break
            r = QtCore.QRectF(x, y, self._icon_size.width(), h)
            self._icon_rects.append(r)
            x += self._icon_size.width() + self._icon_spacing

        # Recompute mapping from undo stack index to indicator X positions
        # after icons and buttons have been laid out.
        self._update_indicator_mapping(rect)

    def _update_indicator_mapping(self, rect: QtCore.QRectF) -> None:
        """Compute parameters for positioning the draggable indicator.

        When every undo command is visible in the timeline, we can treat the
        states as a uniform sequence between icons: state 0 before the first
        command, state N after the last of N commands. In that case we store
        ``_indicator_x0`` and ``_indicator_step`` so that

        ``x(index) = x0 - 0.5 * step + index * step``.

        When some commands are filtered out, we fall back to a simple
        left/right interpolation between the first and last icon.
        """

        self._indicator_uniform = False
        self._indicator_x0 = 0.0
        self._indicator_step = 0.0

        stack = self._stack_alive()
        if stack is None:
            return
        count = stack.count()
        if count <= 0 or not self._icon_rects:
            return

        centers = [r.center().x() for r in self._icon_rects]
        if len(centers) >= 2 and len(self._icon_rects) == count:
            step = centers[1] - centers[0]
            if abs(step) > 0.1:
                self._indicator_uniform = True
                self._indicator_step = float(step)
                self._indicator_x0 = float(centers[0])
        else:
            # Non-uniform or filtered: we leave _indicator_uniform=False and
            # let the helper compute positions on demand when needed.
            self._indicator_uniform = False

    def _indicator_position_for_index(self, rect: QtCore.QRectF, index: int) -> float:
        """Return the X coordinate of the position indicator for *index*.

        Index follows the QUndoStack convention (0..count). When commands are
        laid out uniformly and all are visible, the playhead is positioned
        between icons so that index 0 is to the left of the first icon, index
        1 between icon 0 and 1, ..., and index N after the last of N icons.
        """

        stack = self._stack_alive()
        if stack is None:
            return rect.center().x()
        count = max(0, int(stack.count()))
        if count == 0:
            return rect.center().x()
        index = max(0, min(count, int(index)))

        if self._indicator_uniform and self._indicator_step != 0.0:
            # Uniform spacing using first two icon centers as reference.
            return self._indicator_x0 - 0.5 * self._indicator_step + index * self._indicator_step

        # Fallback: distribute states in a small range around the visible
        # icons so the indicator lies *between* icons rather than directly on
        # top of them. This is especially important when there is only a
        # single visible icon.

        centers = [r.center().x() for r in self._icon_rects]
        if not centers:
            return rect.center().x()
        index = max(0, min(count, int(index)))

        if len(centers) == 1:
            center = centers[0]
            # Approximate a reasonable spacing from the icon geometry so the
            # playhead can move to the left/right without ever covering the
            # icon itself.
            step = float(self._icon_size.width() + self._icon_spacing)
            t = float(index) / float(count)
            return center - 0.5 * step + t * step

        left = centers[0]
        right = centers[-1]
        if left == right:
            return left
        t = float(index) / float(count)
        return left + t * (right - left)

    def _make_icon_for_text(self, text: str) -> QtGui.QPixmap:
        """Return a small pixmap icon representing the command text.

        Icons use simple Unicode glyphs so the meaning is readable at a
        glance: e.g. ``+`` for node creation, ``⇄`` for connections,
        ``✖`` for deletions, ``✎`` for edits.
        """

        t = (text or "").lower()
        kind = "other"
        if t.startswith("add ") or t.startswith("paste ") or "duplicate node" in t:
            # Generic node-adding operations
            kind = "add_node"
        if "add connection" in t:
            kind = "add_edge"
        if "delete" in t:
            kind = "delete"
        if "move nodes" in t:
            kind = "move"
        if "change constant" in t or "change operation" in t or "open node" in t or "close node" in t:
            kind = "edit"
        if "align nodes" in t or "auto layout" in t or "layout" in t:
            kind = "layout"
        if "bring node to front" in t:
            kind = "zorder"

        # Unicode glyph and base color per kind
        if kind == "add_node":
            symbol = "+"   # add
            base = QtGui.QColor(80, 160, 100)
        elif kind == "add_edge":
            symbol = "⇄"   # connection
            base = QtGui.QColor(230, 170, 90)
        elif kind == "delete":
            symbol = "✖"   # delete
            base = QtGui.QColor(210, 90, 90)
        elif kind == "move":
            symbol = "↔"   # move horizontally
            base = QtGui.QColor(130, 140, 215)
        elif kind == "edit":
            symbol = "✎"   # edit
            base = QtGui.QColor(200, 180, 80)
        elif kind == "layout":
            symbol = "≡"   # alignment/layout
            base = QtGui.QColor(130, 130, 130)
        elif kind == "zorder":
            symbol = "⬆"   # bring to front
            base = QtGui.QColor(160, 110, 200)
        else:
            symbol = "•"   # generic
            base = QtGui.QColor(100, 120, 150)

        w = int(max(6.0, self._icon_size.width()))
        h = int(max(6.0, self._icon_size.height()))
        pix = QtGui.QPixmap(w, h)
        pix.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pix)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            # Background rounded rect
            painter.setBrush(base)
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(0, 0, w - 1, h - 1, 2, 2)
            # Symbol
            painter.setPen(QtGui.QPen(QtCore.Qt.white))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(pix.rect(), QtCore.Qt.AlignCenter, symbol)
        finally:
            painter.end()

        return pix

    def _on_stack_index_changed(self, index: int) -> None:  # noqa: ARG002
        self._rebuild()

    def _on_stack_clean_changed(self, clean: bool) -> None:  # noqa: ARG002
        # For now just rebuild; could mark the clean index specially.
        self._rebuild()

    # ----- Interaction ----------------------------------------------------

    def _jump_to_index(self, target: int) -> None:
        stack = self._stack_alive()
        if stack is None:
            return
        # Clamp the requested index into the valid 0..count range and then
        # let QUndoStack drive the necessary undo/redo sequence via
        # setIndex(). This avoids manual loops in Python and relies on Qt's
        # well-tested implementation.
        try:
            count = int(stack.count())
            idx = max(0, min(count, int(target)))
        except Exception:
            return
        if idx == stack.index():
            return
        try:
            stack.setIndex(idx)
        except Exception:
            # If something goes wrong (e.g. during teardown), stop driving
            # this stack from the timeline.
            self._stack = None

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        pos = event.pos()
        hover = self._indicator_hit_rect.contains(pos) or any(r.contains(pos) for r in self._icon_rects)
        if hover:
            self.setCursor(QtCore.Qt.PointingHandCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        stack = self._stack_alive()
        if event.button() != QtCore.Qt.LeftButton or stack is None:
            super().mousePressEvent(event)
            return
        pos = event.pos()
        if self._indicator_hit_rect.contains(pos):
            # Start scrubbing the playhead.
            self._indicator_dragging = True
            self._drag_preview_active = True
            try:
                self._drag_preview_index = int(stack.index())
            except Exception:
                self._drag_preview_index = 0
            event.accept()
            return

        target_index: Optional[int] = None
        for (stack_idx, _pix), rect in zip(self._entries, self._icon_rects):
            if rect.contains(pos):
                target_index = int(stack_idx)
                break
        if target_index is None and self._icon_rects:
            # Allow clicking near the vertical bars by snapping to the
            # closest icon center along X if reasonably close.
            centers = [r.center().x() for r in self._icon_rects]
            closest_idx = min(range(len(centers)), key=lambda i: abs(centers[i] - pos.x()))
            if abs(centers[closest_idx] - pos.x()) <= self._icon_size.width():
                target_index = int(self._entries[closest_idx][0])
        if target_index is None:
            super().mousePressEvent(event)
            return
        self._jump_to_index(target_index)
        event.accept()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if self._indicator_dragging:
            stack = self._stack_alive()
            if stack is None:
                return
            rect = self.path().boundingRect()
            x = float(event.pos().x())
            # Map the dragged X position back to the nearest undo stack index,
            # but only preview the target here. The actual jump is applied in
            # mouseReleaseEvent so we don't hammer QUndoStack with rapid
            # state changes while dragging.
            try:
                count = max(0, int(stack.count()))
            except Exception:
                count = 0
            if count > 0:
                best_index = 0
                best_dist = float("inf")
                for i in range(count + 1):
                    px = self._indicator_position_for_index(rect, i)
                    d = abs(px - x)
                    if d < best_dist:
                        best_dist = d
                        best_index = i
                self._drag_preview_active = True
                self._drag_preview_index = best_index
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if self._indicator_dragging and event.button() == QtCore.Qt.LeftButton:
            self._indicator_dragging = False
            if self._drag_preview_active:
                stack = self._stack_alive()
                if stack is not None:
                    try:
                        self._jump_to_index(self._drag_preview_index)
                    except Exception:
                        pass
            self._drag_preview_active = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ----- Painting -------------------------------------------------------

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:  # type: ignore[override]
        rect = self.path().boundingRect()
        if rect.isEmpty():
            return

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Background strip themed to match the node editor style.
        bg = theme_color("timeline_background", (58, 64, 72))
        border = theme_color("timeline_border", (80, 80, 80))
        painter.setPen(QtGui.QPen(border))
        painter.setBrush(QtGui.QBrush(bg))
        painter.drawRoundedRect(rect, 3.0, 3.0)

        stack = self._stack_alive()
        if stack is None:
            return

        base_y = rect.bottom() - self._baseline_margin

        base_pen = QtGui.QPen(theme_color("timeline_baseline", (100, 100, 100)))
        base_pen.setWidth(1)
        painter.setPen(base_pen)
        painter.drawLine(QtCore.QLineF(rect.left() + 4.0, base_y, rect.right() - 4.0, base_y))

        current_index = int(stack.index())
        count = int(stack.count())

        # Make step markers more prominent against the dark background by
        # using brighter colors and slightly thicker lines.
        past_pen = QtGui.QPen(theme_color("timeline_tick_past", (190, 190, 190)))
        past_pen.setWidth(2)
        future_pen = QtGui.QPen(theme_color("timeline_tick_future", (210, 210, 210)))
        future_pen.setWidth(2)
        future_pen.setStyle(QtCore.Qt.DotLine)
        current_pen = QtGui.QPen(theme_color("timeline_tick_current", (90, 180, 255)))
        current_pen.setWidth(2)

        # Visual separator between controls (arrows + moves) and history
        # elements when we have any controls.
        if self._controls_right_x > 0.0:
            sep_x = max(
                rect.left() + self._pad_left + 8.0,
                min(self._controls_right_x - self._button_spacing * 0.5, rect.right() - 12.0),
            )
            sep_pen = QtGui.QPen(theme_color("timeline_separator", (80, 80, 80)))
            sep_pen.setWidth(1)
            painter.setPen(sep_pen)
            painter.drawLine(
                QtCore.QLineF(
                    sep_x,
                    rect.top() + 3.0,
                    sep_x,
                    rect.bottom() - 3.0,
                )
            )

        # Restore pen for baseline and ticks.
        painter.setPen(base_pen)

        # Icons and their vertical bars
        for (stack_index, pix), icon_rect in zip(self._entries, self._icon_rects):
            painter.drawPixmap(icon_rect.topLeft(), pix)
            x = icon_rect.center().x()
            if stack_index == current_index:
                pen = current_pen
            elif stack_index < current_index:
                pen = past_pen
            else:
                pen = future_pen
            painter.setPen(pen)
            painter.drawLine(QtCore.QLineF(x, base_y, x, base_y - 6.0))

        # Draggable position indicator (playhead), thicker and positioned
        # conceptually *between* icons according to the current (or preview)
        # undo index.
        index_for_indicator = (
            self._drag_preview_index if self._drag_preview_active else current_index
        )
        indicator_x = self._indicator_position_for_index(rect, index_for_indicator)
        self._indicator_x = indicator_x
        # Hit rect slightly wider than the visual line for easier dragging.
        self._indicator_hit_rect = QtCore.QRectF(
            indicator_x - 5.0,
            rect.top() + 2.0,
            10.0,
            rect.height() - 4.0,
        )
        # Slightly brighter tint for the playhead itself so it reads clearly
        # against both the baseline and tick markers.
        indicator_pen = QtGui.QPen(theme_color("timeline_indicator", (120, 200, 255)))
        indicator_pen.setWidth(4)
        painter.setPen(indicator_pen)
        painter.drawLine(
            QtCore.QLineF(
                indicator_x,
                base_y + 1.0,
                indicator_x,
                rect.top() + 2.0,
            )
        )
