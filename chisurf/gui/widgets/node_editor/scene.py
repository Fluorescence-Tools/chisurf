from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from qtpy import QtCore, QtGui, QtWidgets

try:  # optional dependency
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from .model import NodeModel, PortSpec
from .port_item import NodePortGraphicsItem
from .theme import color as theme_color
from .theme import flag as theme_flag
from .theme import metric as theme_metric
from .theme import text as theme_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _MoveNodesCommand(QtWidgets.QUndoCommand):
    """Undo command that stores per-node positions before/after a move.

    This is used for mouse-drag moves so we don't need to rebuild the entire
    graph via JSON snapshots, which can cause visual jumps.
    """

    def __init__(self, nodes, before_pos, after_pos, text="Move nodes"):
        super().__init__(text)
        self._nodes = list(nodes)
        # Copy positions so later changes do not affect stored values
        self._before = {n: QtCore.QPointF(p) for n, p in before_pos.items() if n in nodes}
        self._after = {n: QtCore.QPointF(p) for n, p in after_pos.items() if n in nodes}

    def undo(self) -> None:  # type: ignore[override]
        for n, pos in self._before.items():
            try:
                n.setPos(pos)
            except Exception:
                continue

    def redo(self) -> None:  # type: ignore[override]
        for n, pos in self._after.items():
            try:
                n.setPos(pos)
            except Exception:
                continue


class EdgeGraphicsItem(QtWidgets.QGraphicsPathItem):
    ...  # type: ignore  # placeholder for type checkers


class NodeGraphicsItem(QtWidgets.QGraphicsPathItem):
    ...  # type: ignore


class NodeScene(QtWidgets.QGraphicsScene):
    """Scene that manages nodes and edges and mouse interaction for wiring."""

    def __init__(
        self,
        parent=None,
        *,
        background_color: Optional[QtGui.QColor] = None,
        gradient_top: Optional[QtGui.QColor] = None,
        gradient_bottom: Optional[QtGui.QColor] = None,
        grid_step: Optional[int] = None,
        grid_color: Optional[QtGui.QColor] = None,
        draw_grid: Optional[bool] = None,
        background_pattern_enabled: Optional[bool] = None,
        enable_cycle_highlighting: bool = True,
        enforce_acyclic: bool = False,
        read_only: bool = False,
    ):
        super().__init__(parent)
        self.read_only = read_only
        self.meta = {}

        # Visual configuration
        self.background_color = background_color or theme_color("scene_background", (35, 35, 35))
        self.gradient_top = gradient_top or theme_color("scene_gradient_top", (30, 30, 30))
        self.gradient_bottom = gradient_bottom or theme_color("scene_gradient_bottom", (45, 45, 45))
        step_from_theme = int(theme_metric("scene_grid_step", 20.0))
        if grid_step is None:
            self.grid_step = step_from_theme
        else:
            try:
                self.grid_step = int(grid_step)
            except Exception:
                self.grid_step = step_from_theme
        self.grid_color = grid_color or theme_color("scene_grid_color", (60, 60, 60))
        enabled = background_pattern_enabled
        if enabled is None:
            enabled = draw_grid
        if enabled is None:
            enabled = theme_flag("scene_grid_enabled", True)
        self.draw_grid = bool(enabled)

        self.setBackgroundBrush(self.background_color)
        # Large scene rect so the view can pan well beyond the current nodes.
        try:
            self.setSceneRect(-5000.0, -5000.0, 10000.0, 10000.0)
        except Exception:
            pass

        self._current_edge: Optional["EdgeGraphicsItem"] = None
        self.edges: List["EdgeGraphicsItem"] = []
        # When True, the scene is being cleared/reloaded from a snapshot
        # (e.g. during undo/redo). In that window we must not touch edges or
        # ports, because their underlying QGraphicsItems may already be in the
        # process of being destroyed on the C++ side.
        self._reloading: bool = False
        self.enable_cycle_highlighting: bool = enable_cycle_highlighting
        self.enforce_acyclic: bool = enforce_acyclic
        self.node_adder = None
        self._clipboard: List[Tuple[NodeModel, QtCore.QPointF]] = []
        self._drag_start_positions: Dict[QtWidgets.QGraphicsItem, QtCore.QPointF] = {}
        self._overlay_items: List[QtWidgets.QGraphicsItem] = []

    # ----- Painting -------------------------------------------------------
    def drawBackground(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        super().drawBackground(painter, rect)
        # Subtle vertical gradient background
        grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad.setColorAt(0.0, self.gradient_top)
        grad.setColorAt(1.0, self.gradient_bottom)
        painter.fillRect(rect, grad)

        # Dotted grid on top
        if self.draw_grid and self.grid_step > 0:
            painter.setPen(QtGui.QPen(self.grid_color, 0))
            step = self.grid_step
            left = int(rect.left()) - (int(rect.left()) % step)
            top = int(rect.top()) - (int(rect.top()) % step)
            for x in range(left, int(rect.right()), step):
                for y in range(top, int(rect.bottom()), step):
                    painter.drawPoint(x, y)

    def set_background_pattern_enabled(self, enabled: bool) -> None:
        self.draw_grid = bool(enabled)
        self.update()

    def background_pattern_enabled(self) -> bool:
        return bool(self.draw_grid)

    def set_background_pattern_step(self, step: int) -> None:
        try:
            value = int(step)
        except Exception:
            return
        if value <= 0:
            return
        self.grid_step = value
        self.update()

    # ----- Edge bookkeeping ----------------------------------------------
    def register_edge(self, edge: "EdgeGraphicsItem"):
        if edge not in self.edges:
            self.edges.append(edge)
        if self.enable_cycle_highlighting:
            self.update_cycle_highlighting()
        # Update port connected state
        if edge.start_port is not None:
            edge.start_port.set_connected(edge.start_port.connected_count + 1)
        if edge.end_port is not None:
            edge.end_port.set_connected(edge.end_port.connected_count + 1)

    def register_overlay_item(self, item: QtWidgets.QGraphicsItem) -> None:
        if item is None:
            return
        if item not in self._overlay_items:
            self._overlay_items.append(item)
        if item.scene() is not self:
            try:
                self.addItem(item)
            except Exception:
                pass

    def clear(self) -> None:  # type: ignore[override]
        overlays = list(self._overlay_items)
        # Mark the scene as reloading so helpers like update_edges_for_port
        # become no-ops while QGraphicsScene is tearing down items.
        self._reloading = True
        # Drop all references to old EdgeGraphicsItems *before* we let
        # QGraphicsScene clear the scene, so that any itemChanged hooks that
        # fire during clear() won't iterate deleted edge objects.
        self.edges = []
        for item in overlays:
            try:
                if item.scene() is self:
                    self.removeItem(item)
            except Exception:
                continue
        # Let QGraphicsScene delete all non-overlay items (nodes, edges, etc.)
        super().clear()
        survivors: List[QtWidgets.QGraphicsItem] = []
        for item in overlays:
            try:
                if item.scene() is None:
                    self.addItem(item)
                survivors.append(item)
            except Exception:
                continue
        self._overlay_items = survivors
        self._reloading = False

    def update_edges_for_port(self, port: NodePortGraphicsItem):
        # Safely update only edges that still reference valid ports.
        # When the scene is being cleared/reloaded from a snapshot, skip all
        # updates to avoid touching QGraphicsItems that are being destroyed.
        if getattr(self, "_reloading", False):
            return
        for edge in list(self.edges):
            start = getattr(edge, "start_port", None)
            end = getattr(edge, "end_port", None)
            if start is None and end is None:
                continue
            if start is port or end is port:
                # Skip if the scene or the port is already being torn down.
                try:
                    if port.scene() is None or edge.scene() is None:
                        continue
                except RuntimeError:
                    # Underlying C++ item is gone.
                    continue
                edge.update_path()

    def validate_connection(self, src_port: NodePortGraphicsItem, tgt_port: NodePortGraphicsItem) -> bool:
        """Return True if a connection between src_port and tgt_port is allowed.

        Checks:
        1. Direction: One must be output, one must be input.
        2. Port types: Must be compatible (exact match or one is 'any').
        """
        if src_port.spec.is_output == tgt_port.spec.is_output:
            return False

        src_type = getattr(src_port.spec, "port_type", "spectral") or "spectral"
        tgt_type = getattr(tgt_port.spec, "port_type", "spectral") or "spectral"

        # 'any' type connects to anything
        if src_type == "any" or tgt_type == "any":
            return True

        return src_type == tgt_type

    # ----- Mouse interaction for creating edges --------------------------
    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.read_only:
            super().mousePressEvent(event)
            return
        item = self.itemAt(event.scenePos(), QtGui.QTransform())
        if isinstance(item, NodePortGraphicsItem) and event.button() == QtCore.Qt.LeftButton:
            # Start a temporary edge from this port
            from .edge_item import EdgeGraphicsItem  # local import to avoid circulars

            self._current_edge = EdgeGraphicsItem(item)
            self.addItem(self._current_edge)
            event.accept()
            return
        # Track starting positions for node drags so we can undo a move on
        # mouse release without interfering with the drag itself. Use the
        # snapshot-based undo mechanism (SceneStateTracker) instead of a
        # custom QUndoCommand that stores QGraphicsItem references.
        self._drag_start_positions.clear()
        if event.button() == QtCore.Qt.LeftButton and item is not None:
            try:
                from .node_item import NodeGraphicsItem as RealNodeGraphicsItem

                if isinstance(item, RealNodeGraphicsItem):
                    # If the clicked node is part of a selection, move all
                    # selected nodes together; otherwise only this node.
                    selected = [
                        it for it in self.selectedItems() if isinstance(it, RealNodeGraphicsItem)
                    ]
                    if item.isSelected() and selected:
                        nodes = selected
                    else:
                        nodes = [item]
                    for n in nodes:
                        self._drag_start_positions[n] = QtCore.QPointF(n.pos())

                    # Begin an undo snapshot for the move operation. The
                    # tracker will capture the full graph state before the
                    # drag so we don't have to store QGraphicsItem pointers
                    # inside a custom command.
                    owner = self.parent()
                    if owner is not None and hasattr(owner, "begin_undo"):
                        try:
                            owner.begin_undo("Move nodes")  # type: ignore[attr-defined]
                        except Exception:
                            pass
            except Exception:
                self._drag_start_positions.clear()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self._current_edge is not None:
            self._current_edge.set_temp_end_pos(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self._current_edge is not None and event.button() == QtCore.Qt.LeftButton:
            end_item = self.itemAt(event.scenePos(), QtGui.QTransform())
            # If we didn't hit a port exactly, try snapping to the nearest port
            if not isinstance(end_item, NodePortGraphicsItem):
                snapped = self._find_nearest_port(event.scenePos(), max_dist=18.0)
                if snapped is not None:
                    end_item = snapped

            if isinstance(end_item, NodePortGraphicsItem) and end_item is not self._current_edge.start_port:
                # Validate connection (direction and type)
                if not self.validate_connection(self._current_edge.start_port, end_item):
                    # Incompatible or same direction -> invalid, cancel
                    self.removeItem(self._current_edge)
                else:
                    # Finalize connection (wrapped in undo snapshot)
                    owner = self.parent()
                    if owner is not None and hasattr(owner, "begin_undo"):
                        try:
                            owner.begin_undo("Add connection")  # type: ignore[attr-defined]
                        except Exception:
                            pass

                    self._current_edge.set_end_port(end_item)
                    if self.enforce_acyclic:
                        # Temporarily register to check for cycles
                        self.register_edge(self._current_edge)
                        if not self.is_directed_acyclic():
                            # Would create cycle, unregister and cancel
                            if self._current_edge in self.edges:
                                self.edges.remove(self._current_edge)
                            if self._current_edge.start_port is not None:
                                self._current_edge.start_port.set_connected(self._current_edge.start_port.connected_count - 1)
                            if self._current_edge.end_port is not None:
                                self._current_edge.end_port.set_connected(self._current_edge.end_port.connected_count - 1)
                            self.removeItem(self._current_edge)
                            self._current_edge = None
                            if owner is not None and hasattr(owner, "commit_undo"):
                                try:
                                    owner.commit_undo()  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                            event.accept()
                            return
                    else:
                        self.register_edge(self._current_edge)

                    if owner is not None and hasattr(owner, "commit_undo"):
                        try:
                            owner.commit_undo()  # type: ignore[attr-defined]
                        except Exception:
                            pass
            else:
                # Cancel
                self.removeItem(self._current_edge)
            self._current_edge = None
            # Topology may have changed
            if self.enable_cycle_highlighting:
                self.update_cycle_highlighting()
            event.accept()
            return

        # If we started a node drag, record final positions and push a dedicated
        # undo snapshot via the editor's SceneStateTracker when the positions
        # actually changed.
        if event.button() == QtCore.Qt.LeftButton and self._drag_start_positions:
            try:
                from .node_item import NodeGraphicsItem as RealNodeGraphicsItem

                nodes = [n for n in self._drag_start_positions.keys() if isinstance(n, RealNodeGraphicsItem)]
                if nodes:
                    before = self._drag_start_positions
                    after = {n: QtCore.QPointF(n.pos()) for n in nodes}
                    moved = any(before.get(n) != after.get(n) for n in nodes)
                    if moved:
                        owner = self.parent()
                        if owner is not None and hasattr(owner, "commit_undo"):
                            try:
                                owner.commit_undo()  # type: ignore[attr-defined]
                            except Exception:
                                pass
            finally:
                self._drag_start_positions.clear()

        super().mouseReleaseEvent(event)

    def _find_nearest_port(
        self,
        scene_pos: QtCore.QPointF,
        *,
        max_dist: float,
    ) -> Optional[NodePortGraphicsItem]:
        """Return the closest NodePortGraphicsItem within max_dist (scene units).

        Used to gently snap edge endpoints when the user releases near a port
        without needing to click it exactly.
        """

        closest: Optional[NodePortGraphicsItem] = None
        best_sq = max_dist * max_dist

        for item in self.items():
            if not isinstance(item, NodePortGraphicsItem):
                continue
            p = item.scene_pos()
            dx = p.x() - scene_pos.x()
            dy = p.y() - scene_pos.y()
            d2 = dx * dx + dy * dy
            if d2 <= best_sq:
                best_sq = d2
                closest = item

        return closest

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if self.read_only:
            # Only allow copy (Ctrl+C)
            if event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_C:
                self._copy_selection_to_clipboard()
                event.accept()
                return
            super().keyPressEvent(event)
            return
        # Delete selected nodes/edges with the Delete key
        if event.key() == QtCore.Qt.Key_Delete:
            owner = self.parent()
            if owner is not None and hasattr(owner, "begin_undo"):
                try:
                    owner.begin_undo("Delete selection")  # type: ignore[attr-defined]
                except Exception:
                    pass

            from .edge_item import EdgeGraphicsItem
            from .node_item import NodeGraphicsItem

            deleted_nodes = 0
            deleted_edges = 0

            selected = list(self.selectedItems())
            nodes: List[NodeGraphicsItem] = []
            edges_to_delete: set[EdgeGraphicsItem] = set()

            # First, classify selected items.
            for item in selected:
                if isinstance(item, EdgeGraphicsItem):
                    edges_to_delete.add(item)
                elif isinstance(item, NodeGraphicsItem):
                    nodes.append(item)

            # Also remove all edges attached to any selected node.
            for node in nodes:
                for edge in list(self.edges):
                    if (
                        edge.start_port is not None and edge.start_port.node_item is node
                    ) or (
                        edge.end_port is not None and edge.end_port.node_item is node
                    ):
                        edges_to_delete.add(edge)

            # Delete edges once, updating bookkeeping safely.
            for edge in list(edges_to_delete):
                if edge in self.edges:
                    self.edges.remove(edge)
                if edge.start_port is not None:
                    edge.start_port.set_connected(edge.start_port.connected_count - 1)
                if edge.end_port is not None:
                    edge.end_port.set_connected(edge.end_port.connected_count - 1)
                edge.start_port = None
                edge.end_port = None
                if edge.scene() is self:
                    self.removeItem(edge)
                deleted_edges += 1

            # Now delete the nodes themselves.
            for node in nodes:
                if node.scene() is self:
                    self.removeItem(node)
                    deleted_nodes += 1

            logger.debug("Delete key: removed %d nodes, %d edges", deleted_nodes, deleted_edges)
            self.update_cycle_highlighting()
            # Ensure the scene and all views repaint fully after deletions so
            # no visual remnants of removed nodes/edges remain.
            try:
                self.invalidate(QtCore.QRectF(), QtWidgets.QGraphicsScene.AllLayers)
            except Exception:
                pass
            for view in self.views():
                view.viewport().update()
            if owner is not None and hasattr(owner, "commit_undo"):
                try:
                    owner.commit_undo()  # type: ignore[attr-defined]
                except Exception:
                    pass
            event.accept()
            return
        # ... (rest of the code remains the same)
        # Copy selected nodes: Ctrl+C
        if event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_C:
            self._copy_selection_to_clipboard()
            event.accept()
            return
        # Paste copied nodes: Ctrl+V
        if event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_V:
            owner = self.parent()
            if owner is not None and hasattr(owner, "begin_undo"):
                try:
                    owner.begin_undo("Paste nodes")  # type: ignore[attr-defined]
                except Exception:
                    pass
            self._paste_from_clipboard()
            if owner is not None and hasattr(owner, "commit_undo"):
                try:
                    owner.commit_undo()  # type: ignore[attr-defined]
                except Exception:
                    pass
            event.accept()
            return
        if event.modifiers() == QtCore.Qt.ShiftModifier and event.key() == QtCore.Qt.Key_D:
            from .node_item import NodeGraphicsItem

            selected_nodes = [
                it for it in self.selectedItems() if isinstance(it, NodeGraphicsItem)
            ]
            if selected_nodes:
                offset = QtCore.QPointF(40.0, 40.0)
                owner = self.parent()
                if owner is not None and hasattr(owner, "begin_undo"):
                    try:
                        owner.begin_undo("Duplicate nodes")  # type: ignore[attr-defined]
                    except Exception:
                        pass
                for node in selected_nodes:
                    self._duplicate_node(node, node.pos() + offset)
                if owner is not None and hasattr(owner, "commit_undo"):
                    try:
                        owner.commit_undo()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                event.accept()
                return
        if event.modifiers() == QtCore.Qt.NoModifier and event.key() in (
            QtCore.Qt.Key_C,
            QtCore.Qt.Key_B,
            QtCore.Qt.Key_O,
            QtCore.Qt.Key_K,
        ):
            views = self.views()
            if views:
                view = views[0]
                rect = view.viewport().rect()
                scene_pos = view.mapToScene(rect.center())
            else:
                scene_pos = QtCore.QPointF(0.0, 0.0)

            key_map = {
                QtCore.Qt.Key_C: "constant",
                QtCore.Qt.Key_B: "binary_op",
                QtCore.Qt.Key_O: "output",
                QtCore.Qt.Key_K: "controls",
            }
            node_type = key_map.get(event.key())
            if node_type is not None:
                self._create_node_by_type(node_type, scene_pos)
                event.accept()
                return
        # Arrow keys: nudge selected nodes
        if event.key() in (
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
        ):
            from .node_item import NodeGraphicsItem

            nodes = [it for it in self.selectedItems() if isinstance(it, NodeGraphicsItem)]
            if nodes:
                owner = self.parent()
                if owner is not None and hasattr(owner, "begin_undo"):
                    try:
                        owner.begin_undo("Move nodes")  # type: ignore[attr-defined]
                    except Exception:
                        pass
                step = 10.0
                dx = dy = 0.0
                if event.key() == QtCore.Qt.Key_Left:
                    dx = -step
                elif event.key() == QtCore.Qt.Key_Right:
                    dx = step
                elif event.key() == QtCore.Qt.Key_Up:
                    dy = -step
                elif event.key() == QtCore.Qt.Key_Down:
                    dy = step
                delta = QtCore.QPointF(dx, dy)
                for n in nodes:
                    n.setPos(n.pos() + delta)
                if owner is not None and hasattr(owner, "commit_undo"):
                    try:
                        owner.commit_undo()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                event.accept()
                return
        super().keyPressEvent(event)

    # ----- Clipboard helpers ---------------------------------------------
    def _copy_selection_to_clipboard(self) -> None:
        """Clone selected nodes into an in-memory clipboard.

        Edges are not copied for now; only node models and their relative
        positions are preserved.
        """

        from .node_item import NodeGraphicsItem

        nodes = [it for it in self.selectedItems() if isinstance(it, NodeGraphicsItem)]
        if not nodes:
            self._clipboard = []
            return

        # Use top-left of selection as origin for relative offsets
        min_x = min(n.pos().x() for n in nodes)
        min_y = min(n.pos().y() for n in nodes)
        origin = QtCore.QPointF(min_x, min_y)

        clipboard: List[Tuple[NodeModel, QtCore.QPointF]] = []
        for node in nodes:
            model = node.model
            inputs = [
                PortSpec(
                    name=p.name,
                    is_output=p.is_output,
                    port_type=getattr(p, "port_type", ""),
                    fixed=bool(getattr(p, "fixed", False)),
                    min_value=getattr(p, "min_value", None),
                    max_value=getattr(p, "max_value", None),
                )
                for p in model.inputs
            ]
            outputs = [
                PortSpec(
                    name=p.name,
                    is_output=p.is_output,
                    port_type=getattr(p, "port_type", ""),
                    fixed=bool(getattr(p, "fixed", False)),
                    min_value=getattr(p, "min_value", None),
                    max_value=getattr(p, "max_value", None),
                )
                for p in model.outputs
            ]
            config = dict(model.config) if model.config is not None else {}
            new_model = NodeModel(
                title=model.title,
                inputs=inputs,
                outputs=outputs,
                node_type=model.node_type,
                config=config,
                content_factory=model.content_factory,
            )
            offset = node.pos() - origin
            clipboard.append((new_model, offset))

        self._clipboard = clipboard

    def _paste_from_clipboard(self) -> None:
        """Paste nodes from the internal clipboard at the view center.

        If there is no active view, nodes are pasted near the origin.
        """

        if not self._clipboard:
            return

        views = self.views()
        if views:
            view = views[0]
            rect = view.viewport().rect()
            base_pos = view.mapToScene(rect.center())
        else:
            base_pos = QtCore.QPointF(0.0, 0.0)

        # Slight offset so repeated pastes do not overlap exactly
        base_pos += QtCore.QPointF(40.0, 40.0)

        for model, offset in self._clipboard:
            self._create_node_item_from_model(model, base_pos + offset)

    def _group_node_types_for_menu(self) -> Dict[str, List[str]]:
        """Return mapping of category -> list of node type IDs for menus.

        This helper consults the owning editor's ``available_node_types`` list
        when present; otherwise it falls back to all types known to the
        registry. Categories are taken from the ``NodeType.category`` field,
        defaulting to "General".
        """

        from .registry import registry

        owner = self.parent()
        node_type_ids: Optional[List[str]] = None
        if owner is not None and hasattr(owner, "available_node_types"):
            try:
                node_type_ids = list(owner.available_node_types)  # type: ignore[attr-defined]
            except Exception:
                node_type_ids = None

        if not node_type_ids:
            node_type_ids = registry.available_ids()

        groups: Dict[str, List[str]] = {}
        for type_id in node_type_ids:
            info = None
            try:
                info = registry.get(type_id)
            except Exception:
                info = None
            category = getattr(info, "category", None) or "General"
            groups.setdefault(category, []).append(type_id)

        # Sort types within each category to keep menu ordering stable
        for cat, ids in groups.items():
            groups[cat] = sorted(ids)

        return groups

    # ----- Context menu ---------------------------------------------------
    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent):
        item = self.itemAt(event.scenePos(), QtGui.QTransform())
        scene_pos = event.scenePos()

        # If right-clicked on embedded widget, treat as node
        if isinstance(item, QtWidgets.QGraphicsProxyWidget):
            item = item.parentItem()

        menu = QtWidgets.QMenu()
        # Apply a dark, node-editor-themed style to the context menu so it
        # feels integrated with the scene instead of using the platform
        # default menu colors. Colors, spacing and font are driven from theme.json.
        try:
            bg_top = theme_color("context_menu_bg_top", (46, 50, 56))
            bg_bottom = theme_color("context_menu_bg_bottom", (36, 40, 46))
            border = theme_color("context_menu_border", (20, 20, 20))
            text_col = theme_color("context_menu_text", (230, 230, 230))
            hi_top = theme_color("context_menu_highlight_top", (70, 120, 170))
            hi_bottom = theme_color("context_menu_highlight_bottom", (60, 105, 155))
            sep = theme_color("context_menu_separator", (70, 70, 70))

            radius = int(theme_metric("context_menu_radius", 4))
            padding = int(theme_metric("context_menu_padding", 2))
            item_pad_h = int(theme_metric("context_menu_item_padding_h", 14))
            item_pad_v = int(theme_metric("context_menu_item_padding_v", 2))
            sep_margin_h = int(theme_metric("context_menu_separator_margin_h", 6))
            sep_margin_v = int(theme_metric("context_menu_separator_margin_v", 3))
            item_radius = max(0, radius - 1)

            family = theme_text("node_font_family", "Segoe UI")
            font_px = int(theme_metric("node_font_size", 9))
            bold = theme_flag("context_menu_font_bold", False)
            weight = "bold" if bold else "normal"

            style = (
                "QMenu {"
                " background-color: "
                f"  qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                f"    stop:0 rgba({bg_top.red()},{bg_top.green()},{bg_top.blue()},255),"
                f"    stop:1 rgba({bg_bottom.red()},{bg_bottom.green()},{bg_bottom.blue()},255));"
                f" color: rgb({text_col.red()},{text_col.green()},{text_col.blue()});"
                f" border: 1px solid rgb({border.red()},{border.green()},{border.blue()});"
                f" border-radius: {radius}px;"
                f" padding: {padding}px;"
                f" font-family: '{family}';"
                f" font-size: {font_px}px;"
                f" font-weight: {weight};"
                " }"
                " QMenu::item {"
                f" padding: {item_pad_v}px {item_pad_h}px;"
                f" border-radius: {item_radius}px;"
                " }"
                " QMenu::item:selected {"
                " background-color: "
                f"  qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                f"    stop:0 rgba({hi_top.red()},{hi_top.green()},{hi_top.blue()},255),"
                f"    stop:1 rgba({hi_bottom.red()},{hi_bottom.green()},{hi_bottom.blue()},255));"
                " }"
                " QMenu::separator {"
                f" height: 1px; background: rgb({sep.red()},{sep.green()},{sep.blue()});"
                f" margin: {sep_margin_v}px {sep_margin_h}px {sep_margin_v}px {sep_margin_h}px;"
                " }"
            )
            menu.setStyleSheet(style)
        except Exception:
            pass
        chosen_node = None
        chosen_edge = None
        add_menu = None
        align_menu = None
        act_undo = None
        act_redo = None
        act_raise_node = None
        act_timeline_moves = None
        act_timeline_folds = None

        from .edge_item import EdgeGraphicsItem
        from .node_item import NodeGraphicsItem

        if not self.read_only:
            if isinstance(item, EdgeGraphicsItem):
                chosen_edge = item
                act_del_edge = menu.addAction("Delete connection")
                menu.addSeparator()
                act_toggle = act_dup_node = act_del_node = None  # type: ignore
            elif isinstance(item, NodeGraphicsItem):
                chosen_node = item
                act_toggle = menu.addAction("Open node" if item.collapsed else "Close node")
                act_dup_node = menu.addAction("Duplicate node")
                act_del_node = menu.addAction("Delete node")
                act_raise_node = menu.addAction("Bring to front")
                menu.addSeparator()
                act_del_edge = None  # type: ignore
            else:
                act_toggle = act_dup_node = act_del_node = act_del_edge = None  # type: ignore
        else:
            if isinstance(item, NodeGraphicsItem):
                chosen_node = item
                act_toggle = menu.addAction("Open node" if item.collapsed else "Close node")
                menu.addSeparator()
            else:
                act_toggle = None
            act_del_edge = act_dup_node = act_del_node = act_raise_node = None

        # Node creation submenu: only on scene background (no node/edge under cursor)
        if not self.read_only and chosen_node is None and chosen_edge is None:
            groups = self._group_node_types_for_menu()
            if groups:
                add_menu = menu.addMenu("Add node")

                if len(groups) == 1:
                    # Flat list when there is only a single category
                    (_, type_ids) = next(iter(groups.items()))
                    for t in type_ids:
                        text = str(t).replace("_", " ").title()
                        act = add_menu.addAction(text)
                        act.setData(str(t))
                else:
                    # Grouped by category
                    for category in sorted(groups.keys()):
                        sub = add_menu.addMenu(str(category))
                        for t in groups[category]:
                            text = str(t).replace("_", " ").title()
                            act = sub.addAction(text)
                            act.setData(str(t))

        # Alignment submenu (only meaningful when multiple nodes are selected)
        selected_nodes = [
            it
            for it in self.selectedItems()
            if isinstance(it, NodeGraphicsItem)
        ]
        if len(selected_nodes) >= 2:
            align_menu = menu.addMenu("Align")
            act_align_left = align_menu.addAction("Left")
            act_align_right = align_menu.addAction("Right")
            act_align_top = align_menu.addAction("Top")
            act_align_bottom = align_menu.addAction("Bottom")
            act_align_vcenter = align_menu.addAction("Vertical center")
            act_align_hcenter = align_menu.addAction("Horizontal center")
        else:
            act_align_left = act_align_right = act_align_top = act_align_bottom = None  # type: ignore
            act_align_vcenter = act_align_hcenter = None  # type: ignore

        # Undo/redo actions (if owner provides them)
        owner = self.parent()
        if owner is not None and hasattr(owner, "undo"):
            act_undo = menu.addAction("Undo")
        if owner is not None and hasattr(owner, "redo"):
            act_redo = menu.addAction("Redo")
        if act_undo is not None or act_redo is not None:
            menu.addSeparator()

        # Timeline options (if the owning editor exposes a timeline widget)
        timeline = getattr(owner, "timeline", None) if owner is not None else None
        if timeline is not None and hasattr(timeline, "set_show_moves"):
            act_timeline_moves = menu.addAction("Show moves in timeline")
            act_timeline_moves.setCheckable(True)
            try:
                current_moves = bool(getattr(timeline, "_show_moves", False))
            except Exception:
                current_moves = False
            act_timeline_moves.setChecked(current_moves)

        if timeline is not None and hasattr(timeline, "set_show_folds"):
            act_timeline_folds = menu.addAction("Show folding in timeline")
            act_timeline_folds.setCheckable(True)
            try:
                current_folds = bool(getattr(timeline, "_show_folds", False))
            except Exception:
                current_folds = False
            act_timeline_folds.setChecked(current_folds)

        if act_timeline_moves is not None or act_timeline_folds is not None:
            menu.addSeparator()

        # Global actions
        act_layout = menu.addAction("Auto layout (spring)")
        act_zoom_in = menu.addAction("Zoom in")
        act_zoom_out = menu.addAction("Zoom out")
        act_fit = menu.addAction("Fit to view")
        act_reset = menu.addAction("Reset zoom")
        menu.addSeparator()
        act_help = menu.addAction("Help...")

        action = menu.exec_(event.screenPos())
        if action is None:
            return

        # Handle add-node actions (flat or nested). All node-creation actions
        # store the node type string in QAction.data(). Other actions leave it
        # unset, so we can reliably distinguish them.
        if add_menu is not None:
            data = action.data()
            if isinstance(data, str) and data:
                node_type = data
                logger.debug("context menu add node type=%s at pos=(%.2f, %.2f)", node_type, scene_pos.x(), scene_pos.y())
                self._create_node_by_type(node_type, scene_pos)
                return

        # Undo/redo actions
        if act_undo is not None and action is act_undo:
            if owner is not None and hasattr(owner, "undo"):
                try:
                    owner.undo()  # type: ignore[attr-defined]
                except Exception:
                    logger.error("Error invoking editor.undo() from context menu", exc_info=True)
            return
        if act_redo is not None and action is act_redo:
            if owner is not None and hasattr(owner, "redo"):
                try:
                    owner.redo()  # type: ignore[attr-defined]
                except Exception:
                    logger.error("Error invoking editor.redo() from context menu", exc_info=True)
            return

        # Timeline filter visibility
        if act_timeline_moves is not None and action is act_timeline_moves:
            if timeline is not None and hasattr(timeline, "set_show_moves"):
                try:
                    timeline.set_show_moves(bool(act_timeline_moves.isChecked()))
                except Exception:
                    logger.error("Error toggling timeline moves visibility from context menu", exc_info=True)
            return

        if act_timeline_folds is not None and action is act_timeline_folds:
            if timeline is not None and hasattr(timeline, "set_show_folds"):
                try:
                    timeline.set_show_folds(bool(act_timeline_folds.isChecked()))
                except Exception:
                    logger.error("Error toggling timeline folding visibility from context menu", exc_info=True)
            return

        # Help action
        if action is act_help:
            owner = self.parent()
            if owner is not None and hasattr(owner, "show_help_dialog"):
                try:
                    owner.show_help_dialog()  # type: ignore[attr-defined]
                except Exception:
                    pass
            return

        # Handle alignment actions
        if align_menu is not None and action in align_menu.actions():
            owner = self.parent()
            if owner is not None and hasattr(owner, "begin_undo"):
                try:
                    owner.begin_undo("Align nodes")  # type: ignore[attr-defined]
                except Exception:
                    pass
            if action is act_align_left:
                self._align_nodes(selected_nodes, mode="left")
            elif action is act_align_right:
                self._align_nodes(selected_nodes, mode="right")
            elif action is act_align_top:
                self._align_nodes(selected_nodes, mode="top")
            elif action is act_align_bottom:
                self._align_nodes(selected_nodes, mode="bottom")
            elif action is act_align_vcenter:
                self._align_nodes(selected_nodes, mode="vcenter")
            elif action is act_align_hcenter:
                self._align_nodes(selected_nodes, mode="hcenter")
            if owner is not None and hasattr(owner, "commit_undo"):
                try:
                    owner.commit_undo()  # type: ignore[attr-defined]
                except Exception:
                    pass
            return

        # Handle edge deletion
        if chosen_edge is not None and act_del_edge is not None and action is act_del_edge:
            owner = self.parent()
            if owner is not None and hasattr(owner, "begin_undo"):
                try:
                    owner.begin_undo("Delete connection")  # type: ignore[attr-defined]
                except Exception:
                    pass
            if chosen_edge in self.edges:
                self.edges.remove(chosen_edge)
                # decrement port counts
                if chosen_edge.start_port is not None:
                    chosen_edge.start_port.set_connected(chosen_edge.start_port.connected_count - 1)
                if chosen_edge.end_port is not None:
                    chosen_edge.end_port.set_connected(chosen_edge.end_port.connected_count - 1)
            # Clear references to ports before removing the edge item
            chosen_edge.start_port = None
            chosen_edge.end_port = None
            self.removeItem(chosen_edge)
            if self.enable_cycle_highlighting:
                self.update_cycle_highlighting()
            # Force a full repaint so no ghost edge remains until zoom.
            try:
                self.invalidate(QtCore.QRectF(), QtWidgets.QGraphicsScene.AllLayers)
            except Exception:
                pass
            for view in self.views():
                view.viewport().update()
            if owner is not None and hasattr(owner, "commit_undo"):
                try:
                    owner.commit_undo()  # type: ignore[attr-defined]
                except Exception:
                    pass
            return

        # Handle node actions
        if chosen_node is not None:
            if act_toggle is not None and action is act_toggle:
                owner = self.parent()
                if owner is not None and hasattr(owner, "begin_undo"):
                    try:
                        text = "Open node" if chosen_node.collapsed else "Close node"
                        owner.begin_undo(text)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                try:
                    chosen_node.toggle_collapsed()
                finally:
                    if owner is not None and hasattr(owner, "commit_undo"):
                        try:
                            owner.commit_undo()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                return
            if act_del_node is not None and action is act_del_node:
                # Reuse the Delete-key path to keep deletion logic in one
                # place and avoid subtle differences between code paths.
                chosen_node.setSelected(True)
                delete_event = QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress,
                    QtCore.Qt.Key_Delete,
                    QtCore.Qt.NoModifier,
                )
                self.keyPressEvent(delete_event)
                return
            if act_dup_node is not None and action is act_dup_node:
                self._duplicate_node(chosen_node, scene_pos + QtCore.QPointF(40.0, 40.0))
                return
            if act_raise_node is not None and action is act_raise_node:
                owner = self.parent()
                if owner is not None and hasattr(owner, "begin_undo"):
                    try:
                        owner.begin_undo("Bring node to front")  # type: ignore[attr-defined]
                    except Exception:
                        pass
                try:
                    # Increase this node's z-value above all other nodes.
                    nodes = [
                        it
                        for it in self.items()
                        if isinstance(it, NodeGraphicsItem)
                    ]
                    max_z = max((n.zValue() for n in nodes), default=0.0)
                    chosen_node.setZValue(max_z + 1.0)
                finally:
                    if owner is not None and hasattr(owner, "commit_undo"):
                        try:
                            owner.commit_undo()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                return

        # Global view-related actions
        views = self.views()
        view = views[0] if views else None

        if action is act_layout:
            owner = self.parent()
            if owner is not None and hasattr(owner, "begin_undo"):
                try:
                    owner.begin_undo("Auto layout")  # type: ignore[attr-defined]
                except Exception:
                    pass
            self.auto_layout()
            if owner is not None and hasattr(owner, "commit_undo"):
                try:
                    owner.commit_undo()  # type: ignore[attr-defined]
                except Exception:
                    pass
        elif action is act_zoom_in and view is not None:
            view.zoom_in()
        elif action is act_zoom_out and view is not None:
            view.zoom_out()
        elif action is act_fit and view is not None:
            view.fit_all()
        elif action is act_reset and view is not None:
            view.reset_zoom()

    def _align_nodes(self, nodes, *, mode: str):
        """Align a list of NodeGraphicsItems along the given axis.

        mode can be: "left", "right", "top", "bottom", "vcenter", "hcenter".
        """

        if not nodes:
            return

        # Work in scene coordinates using each node's bounding rect
        rects = [n.sceneBoundingRect() for n in nodes]

        if mode == "left":
            target_x = min(r.left() for r in rects)
            for n, r in zip(nodes, rects):
                dx = target_x - r.left()
                n.setPos(n.pos() + QtCore.QPointF(dx, 0.0))
        elif mode == "right":
            target_x = max(r.right() for r in rects)
            for n, r in zip(nodes, rects):
                dx = target_x - r.right()
                n.setPos(n.pos() + QtCore.QPointF(dx, 0.0))
        elif mode == "top":
            target_y = min(r.top() for r in rects)
            for n, r in zip(nodes, rects):
                dy = target_y - r.top()
                n.setPos(n.pos() + QtCore.QPointF(0.0, dy))
        elif mode == "bottom":
            target_y = max(r.bottom() for r in rects)
            for n, r in zip(nodes, rects):
                dy = target_y - r.bottom()
                n.setPos(n.pos() + QtCore.QPointF(0.0, dy))
        elif mode == "vcenter":
            target_x = sum(r.center().x() for r in rects) / float(len(rects))
            for n, r in zip(nodes, rects):
                dx = target_x - r.center().x()
                n.setPos(n.pos() + QtCore.QPointF(dx, 0.0))
        elif mode == "hcenter":
            target_y = sum(r.center().y() for r in rects) / float(len(rects))
            for n, r in zip(nodes, rects):
                dy = target_y - r.center().y()
                n.setPos(n.pos() + QtCore.QPointF(0.0, dy))

    def _create_node_item_from_model(self, model: NodeModel, scene_pos: QtCore.QPointF):
        owner = self.parent()
        item = None
        if owner is not None and hasattr(owner, "create_node_item"):
            try:
                item = owner.create_node_item(model)  # type: ignore[attr-defined]
            except Exception:
                item = None
        if item is None:
            from .node_item import NodeGraphicsItem

            item = NodeGraphicsItem(model)
        self.addItem(item)
        item.setPos(scene_pos)
        return item

    def _duplicate_node(self, node_item: NodeGraphicsItem, scene_pos: QtCore.QPointF):
        model = node_item.model
        inputs = [PortSpec(name=p.name, is_output=p.is_output) for p in model.inputs]
        outputs = [PortSpec(name=p.name, is_output=p.is_output) for p in model.outputs]
        config = dict(model.config) if model.config is not None else {}
        new_model = NodeModel(
            title=model.title,
            inputs=inputs,
            outputs=outputs,
            node_type=model.node_type,
            config=config,
            content_factory=model.content_factory,
        )
        self._create_node_item_from_model(new_model, scene_pos)

    def _create_node_by_type(self, node_type: str, scene_pos: QtCore.QPointF):
        if self.node_adder is not None:
            try:
                self.node_adder(node_type, scene_pos)
            except Exception:
                return

    # ----- Layout using networkx ------------------------------------------
    def auto_layout(self):
        """Automatically arrange nodes using a hierarchical spring approach.

        This algorithm:
        1. Assigns nodes to horizontal levels (columns) based on graph flow.
        2. Uses a spring layout guess for vertical ordering.
        3. Stacks nodes in columns while ensuring no vertical overlaps.
        4. Adjusts column spacing to account for node widths.
        """
        if nx is None:
            logger.warning("networkx is not available; auto layout disabled")
            return

        graphs = self._build_nx_graphs()
        if graphs is None:
            return

        G_undirected, G_directed, index_by_node = graphs
        if not index_by_node:
            return

        # node_by_idx[int] -> NodeGraphicsItem
        node_by_idx = {i: n for n, i in index_by_node.items()}

        # 1. Assign levels (X-axis)
        # Using simple distance from roots for depth
        levels = {}
        roots = [n for n, d in G_directed.in_degree() if d == 0]
        if not roots and len(G_directed.nodes) > 0:
            roots = [list(G_directed.nodes)[0]] # Handle cycles by picking an arbitrary root

        queue = [(r, 0) for r in roots]
        processed = set()
        while queue:
            idx, lvl = queue.pop(0)
            levels[idx] = max(levels.get(idx, 0), lvl)
            if idx not in processed:
                processed.add(idx)
                for succ in G_directed.successors(idx):
                    queue.append((succ, lvl+1))

        # Ensure every node has a level (e.g. disconnected nodes)
        for idx in G_directed.nodes:
            if idx not in levels:
                levels[idx] = 0

        # 2. Get relative vertical order from spring layout
        # This keeps the general "tangle" of the graph preserved
        pos = nx.spring_layout(G_undirected, k=1.0, iterations=50)

        # 3. Group by level and calculate positions
        nodes_by_level = {}
        for idx, lvl in levels.items():
            nodes_by_level.setdefault(lvl, []).append(idx)

        h_gap = 100.0 # horizontal gap between columns
        v_gap = 40.0  # vertical gap between nodes

        current_x = 0.0

        # To avoid visual jump, we'll store moves in the parent's undo if available
        owner = self.parent()
        if owner and hasattr(owner, "begin_undo"):
             pass # Already called in contextMenuEvent

        for lvl in sorted(nodes_by_level.keys()):
            level_nodes = nodes_by_level[lvl]
            # Order nodes in this level by their spring layout Y position
            level_nodes.sort(key=lambda idx: pos[idx][1])

            max_w = 0.0
            # Calculate total height of this column to center it
            col_height = sum(node_by_idx[idx].boundingRect().height() for idx in level_nodes)
            col_height += v_gap * (len(level_nodes) - 1)

            current_y = -col_height / 2.0

            for idx in level_nodes:
                node = node_by_idx[idx]
                br = node.boundingRect()

                # Snap to horizontal center of its intended column?
                # Better: Left align in column and track max width.
                node.setPos(current_x, current_y)

                max_w = max(max_w, br.width())
                current_y += br.height() + v_gap

            current_x += max_w + h_gap

        self.update()

    def _build_nx_graphs(self):
        if nx is None:
            return None

        from .node_item import NodeGraphicsItem

        nodes: List[NodeGraphicsItem] = [
            item for item in self.items() if isinstance(item, NodeGraphicsItem)
        ]
        if not nodes:
            return None

        index_by_node = {n: i for i, n in enumerate(nodes)}

        G_undirected = nx.Graph()
        G_directed = nx.DiGraph()
        for i in index_by_node.values():
            G_undirected.add_node(i)
            G_directed.add_node(i)

        for edge in self.edges:
            if edge.start_port is None or edge.end_port is None:
                continue
            p1 = edge.start_port
            p2 = edge.end_port
            if p1 is None or p2 is None:
                continue

            if p1.spec.is_output and not p2.spec.is_output:
                src_node_item = p1.node_item
                dst_node_item = p2.node_item
            elif p2.spec.is_output and not p1.spec.is_output:
                src_node_item = p2.node_item
                dst_node_item = p1.node_item
            else:
                src_node_item = p1.node_item
                dst_node_item = p2.node_item

            if src_node_item is dst_node_item:
                continue

            i1 = index_by_node.get(src_node_item)
            i2 = index_by_node.get(dst_node_item)
            if i1 is None or i2 is None:
                continue

            G_undirected.add_edge(i1, i2)
            G_directed.add_edge(i1, i2)

        return G_undirected, G_directed, index_by_node

    def _get_directed_graph(self):
        """Return (G_directed, index_by_node) or None if graph not available.

        This is the single internal place where the directed graph used for
        DAG checks is built, so that helpers like ``is_directed_acyclic``,
        ``has_cycles`` and ``update_cycle_highlighting`` cannot diverge.
        """

        graphs = self._build_nx_graphs()
        if graphs is None:
            return None
        _, G_directed, index_by_node = graphs
        if len(G_directed.edges) == 0:
            return None
        return G_directed, index_by_node

    def is_directed_acyclic(self) -> Optional[bool]:
        result = self._get_directed_graph()
        if result is None:
            return None
        G_directed, _ = result
        return nx.is_directed_acyclic_graph(G_directed)

    def has_cycles(self) -> bool:
        """Check if the scene contains cycles (convenience method)."""
        acyclic = self.is_directed_acyclic()
        return acyclic is False

    def find_cycles(self):
        """Return a list of cycles as lists of node indices.

        The indices correspond to the integer node IDs used in the internal
        NetworkX graphs and ``NodeScene.to_dict``. If NetworkX is not
        available or the scene has no cycles, an empty list is returned.
        """

        if nx is None:
            return []

        result = self._get_directed_graph()
        if result is None:
            return []
        G_directed, _ = result

        try:
            return [list(cyc) for cyc in nx.simple_cycles(G_directed)]
        except Exception:
            return []

    def update_cycle_highlighting(self):
        if not self.enable_cycle_highlighting:
            return
        # Rebuild edge list from current scene items to avoid keeping references
        # to deleted QGraphicsItems, which can cause crashes when accessed.
        from .edge_item import EdgeGraphicsItem

        self.edges = [
            e for e in self.items() if isinstance(e, EdgeGraphicsItem)
        ]

        if nx is None or not self.edges:
            for e in self.edges:
                e.set_cycle(False)
            return

        result = self._get_directed_graph()
        if result is None:
            for e in self.edges:
                e.set_cycle(False)
            return

        G_directed, index_by_node = result

        cycle_edge_pairs = set()
        try:
            cycles = list(nx.simple_cycles(G_directed))
        except Exception:
            cycles = []

        for cyc in cycles:
            if len(cyc) < 2:
                continue
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                cycle_edge_pairs.add((u, v))

        for edge in self.edges:
            if edge.start_port is None or edge.end_port is None:
                edge.set_cycle(False)
                continue

            p1 = edge.start_port
            p2 = edge.end_port
            if p1.spec.is_output and not p2.spec.is_output:
                src_node = p1.node_item
                dst_node = p2.node_item
            elif p2.spec.is_output and not p1.spec.is_output:
                src_node = p2.node_item
                dst_node = p1.node_item
            else:
                src_node = p1.node_item
                dst_node = p2.node_item

            i1 = index_by_node.get(src_node)
            i2 = index_by_node.get(dst_node)
            if i1 is None or i2 is None:
                edge.set_cycle(False)
                continue

            edge.set_cycle((i1, i2) in cycle_edge_pairs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the current scene to a dictionary (JSON schema v1).

        Ports are written using the compact string form when only a name is
        present, and as objects when additional metadata such as type, fixed
        state or bounds are set. This keeps existing JSON concise while
        allowing richer port descriptions.
        """
        from .node_item import NodeGraphicsItem

        def _port_to_entry(p: PortSpec):
            # When no extra metadata is present, store the port as a plain
            # string for backwards compatibility. Otherwise, emit a dict.
            has_type = bool(getattr(p, "port_type", ""))
            is_fixed = bool(getattr(p, "fixed", False))
            has_min = getattr(p, "min_value", None) is not None
            has_max = getattr(p, "max_value", None) is not None
            if not (has_type or is_fixed or has_min or has_max):
                return p.name
            entry = {"name": p.name}
            if has_type:
                entry["type"] = str(getattr(p, "port_type", ""))
            if is_fixed:
                entry["fixed"] = True
            if has_min:
                try:
                    entry["min"] = float(getattr(p, "min_value", 0.0))
                except Exception:
                    pass
            if has_max:
                try:
                    entry["max"] = float(getattr(p, "max_value", 0.0))
                except Exception:
                    pass
            return entry

        nodes = []
        id_by_item = {}

        # Collect nodes
        for idx, item in enumerate(self.items()):
            if isinstance(item, NodeGraphicsItem):
                # Update model.config from widget values
                w = item.content_widget
                if w is not None:
                    if item.model.node_type == "constant":
                        spin = w.findChild(QtWidgets.QDoubleSpinBox)
                        if spin is not None:
                            item.model.config["value"] = float(spin.value())
                    elif item.model.node_type == "binary_op":
                        combo = w.findChild(QtWidgets.QComboBox)
                        if combo is not None:
                            item.model.config["op"] = str(combo.currentText())
                    elif item.model.node_type == "output":
                        labels = w.findChildren(QtWidgets.QLabel)
                        if labels:
                            item.model.config["text"] = str(labels[-1].text())

                node_id = str(getattr(item.model, "id", f"n{idx}"))
                id_by_item[item] = node_id
                model = item.model
                nodes.append({
                    "id": node_id,
                    "title": model.title,
                    "inputs": [_port_to_entry(p) for p in model.inputs],
                    "outputs": [_port_to_entry(p) for p in model.outputs],
                    "type": model.node_type,
                    "config": model.config or {},
                    "pos": [float(item.pos().x()), float(item.pos().y())],
                    "collapsed": bool(getattr(item, "collapsed", False)),
                    "z": float(item.zValue()),
                })

        # Collect edges
        edges = []
        for edge in self.edges:
            if edge.start_port is None or edge.end_port is None:
                continue
            src_item = edge.start_port.node_item
            dst_item = edge.end_port.node_item
            if src_item not in id_by_item or dst_item not in id_by_item:
                continue
            src_id = id_by_item[src_item]
            dst_id = id_by_item[dst_item]
            src_idx = edge.start_port.index
            dst_idx = edge.end_port.index

            edge_entry = {
                "source": src_id,
                "source_port": int(src_idx),
                "target": dst_id,
                "target_port": int(dst_idx),
            }

            cfg = getattr(edge, "_config", None)
            if cfg is None:
                cfg = getattr(edge, "_style_config", None)
            if isinstance(cfg, dict) and cfg:
                edge_entry["config"] = copy.deepcopy(cfg)

            edges.append(edge_entry)

        res = {"nodes": nodes, "edges": edges, "version": 1}
        if getattr(self, "meta", None):
            res["meta"] = self.meta
        return res

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load scene from a dictionary (JSON schema v1), validating first."""
        self.meta = data.get("meta", {})
        from .edge_item import EdgeGraphicsItem
        from .node_item import NodeGraphicsItem
        from .registry import registry
        from .validation import validate_graph_dict

        validate_graph_dict(data)

        # Clear scene
        self.clear()

        nodes_data = data.get("nodes", [])
        edges_data = data.get("edges", [])

        items_by_id: Dict[str, NodeGraphicsItem] = {}

        # Get owner (editor) for factories
        owner = self.parent()

        # Create nodes
        for node_desc in nodes_data:
            node_id = str(node_desc.get("id", ""))
            if not node_id:
                continue
            title = str(node_desc.get("title", node_id))

            raw_inputs = node_desc.get("inputs", [])
            raw_outputs = node_desc.get("outputs", [])

            def _parse_port(entry, is_output: bool) -> Optional[PortSpec]:
                # Support both compact string form ("name") and extended
                # object form ({"name": ..., "type": ..., "fixed": ...}).
                if isinstance(entry, str):
                    return PortSpec(name=entry, is_output=is_output)
                if not isinstance(entry, dict):
                    return None
                name = str(entry.get("name", "")).strip()
                if not name:
                    return None
                port_type = str(entry.get("type", ""))
                fixed = bool(entry.get("fixed", False))
                min_raw = entry.get("min", None)
                max_raw = entry.get("max", None)
                min_value = None
                max_value = None
                try:
                    if min_raw is not None:
                        min_value = float(min_raw)
                except Exception:
                    min_value = None
                try:
                    if max_raw is not None:
                        max_value = float(max_raw)
                except Exception:
                    max_value = None
                return PortSpec(
                    name=name,
                    is_output=is_output,
                    port_type=port_type,
                    fixed=fixed,
                    min_value=min_value,
                    max_value=max_value,
                )

            inputs: List[PortSpec] = []
            outputs: List[PortSpec] = []
            for entry in raw_inputs:
                p = _parse_port(entry, False)
                if p is not None:
                    inputs.append(p)
            for entry in raw_outputs:
                p = _parse_port(entry, True)
                if p is not None:
                    outputs.append(p)
            node_type = str(node_desc.get("type", "generic"))
            cfg = node_desc.get("config") or {}
            if not isinstance(cfg, dict):
                cfg = {}

            # Get content factory from registry (preferred) or owner (legacy)
            factory = None

            # Prefer an editor-specific helper when available so that widgets
            # are bound to the correct owning NodeEditorWidget instance.
            if owner is not None and hasattr(owner, "_create_content_factory_for_json_node"):
                try:
                    factory = owner._create_content_factory_for_json_node(node_type, cfg)  # type: ignore[attr-defined]
                except Exception:
                    factory = None

            if factory is None:
                # 1) Try node type registry
                try:
                    node_type_info = registry.get(node_type)
                except Exception:
                    node_type_info = None

                if node_type_info is not None and getattr(node_type_info, "factory", None):
                    try:
                        # NodeType.factory returns a zero-arg factory when called
                        factory = node_type_info.factory(cfg)  # type: ignore[call-arg]
                    except Exception:
                        factory = None

            model = NodeModel(
                title=title,
                inputs=inputs,
                outputs=outputs,
                node_type=node_type,
                config=cfg,
                content_factory=factory,
                id=node_id,
            )

            # Create item using owner's method if available
            if owner and hasattr(owner, 'create_node_item'):
                try:
                    item = owner.create_node_item(model)
                except Exception:
                    item = NodeGraphicsItem(model)
            else:
                item = NodeGraphicsItem(model)
            if self.read_only:
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
            self.addItem(item)

            pos = node_desc.get("pos")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                try:
                    x = float(pos[0])
                    y = float(pos[1])
                    item.setPos(x, y)
                except Exception:
                    pass

            # Optional z-order for drawing order; higher z draws on top.
            try:
                z_val = node_desc.get("z", None)
                if isinstance(z_val, (int, float)):
                    item.setZValue(float(z_val))
            except Exception:
                pass

            try:
                collapsed_flag = bool(node_desc.get("collapsed", False))
                if collapsed_flag:
                    item.set_collapsed(True)
            except Exception:
                pass

            items_by_id[node_id] = item

        # Create edges
        for edge_desc in edges_data:
            src_id = str(edge_desc.get("source", ""))
            dst_id = str(edge_desc.get("target", ""))
            if src_id not in items_by_id or dst_id not in items_by_id:
                continue
            try:
                src_port_idx = int(edge_desc.get("source_port", 0))
                dst_port_idx = int(edge_desc.get("target_port", 0))
            except Exception:
                continue

            src_item = items_by_id[src_id]
            dst_item = items_by_id[dst_id]
            if not (0 <= src_port_idx < len(src_item.port_items)):
                continue
            if not (0 <= dst_port_idx < len(dst_item.port_items)):
                continue

            src_port = src_item.port_items[src_port_idx]
            dst_port = dst_item.port_items[dst_port_idx]
            edge = EdgeGraphicsItem(src_port, dst_port)
            cfg = edge_desc.get("config")
            if isinstance(cfg, dict):
                edge._config = copy.deepcopy(cfg)
                try:
                    edge.apply_config(cfg)
                except Exception:
                    pass
            self.addItem(edge)
            self.register_edge(edge)

        # Update cycle highlighting
        self.update_cycle_highlighting()
