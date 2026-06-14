"""Read-only graph viewer widget for node-editor graphs."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from qtpy import QtCore, QtWidgets

from .model import NodeModel, PortSpec
from .scene import NodeScene
from .view import NodeView

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NodeViewerWidget(QtWidgets.QWidget):
    """Embeddable read-only widget for displaying node-editor graphs."""

    _node_viewer_layout_owner = True

    graphChanged = QtCore.Signal()
    nodeSelected = QtCore.Signal(dict)
    edgeSelected = QtCore.Signal(dict)
    graphLoaded = QtCore.Signal(dict)
    graphLoadFailed = QtCore.Signal(dict)

    def __init__(
        self,
        parent=None,
        *,
        read_only: bool = True,
        scene_kwargs: Optional[Dict[str, Any]] = None,
        show_toolbar: bool = False,
        graph_purpose: str = "provenance_view",
        client: Any | None = None,
    ):
        """Create a node graph viewer.

        Parameters
        ----------
        parent : QWidget or None
            Optional Qt parent widget.
        read_only : bool, default True
            Whether graph mutation interactions are disabled.
        scene_kwargs : dict or None
            Additional keyword arguments passed to ``NodeScene``.
        show_toolbar : bool, default False
            Whether to create a small toolbar for viewer actions.
        graph_purpose : str, default "provenance_view"
            Metadata purpose stored with the viewer.
        client : object or None
            Optional graph-provider client used by higher-level callers.
        """
        super().__init__(parent)
        self.graph_purpose = graph_purpose
        self.client = client
        self._selected_node: dict[str, Any] = {}
        self._selected_edge: dict[str, Any] = {}
        self._available_node_types: list[str] = []
        self._setup_viewer_scene(read_only=read_only, scene_kwargs=scene_kwargs)
        self._setup_viewer_toolbar(show_toolbar=show_toolbar)
        if self._node_viewer_layout_owner:
            self._install_viewer_layout()
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)

    def _setup_viewer_scene(
        self,
        *,
        read_only: bool,
        scene_kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if scene_kwargs is None:
            scene_kwargs = {}
        scene_kwargs = dict(scene_kwargs)
        scene_kwargs["read_only"] = bool(read_only)
        self.scene = NodeScene(self, **scene_kwargs)
        self.view = NodeView(self.scene, self)
        self.viewer_panel = QtWidgets.QWidget(self)
        viewer_layout = QtWidgets.QVBoxLayout(self.viewer_panel)
        self.viewer_layout = viewer_layout
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)
        self.toolbar = None
        self.viewer_layout.addWidget(self.view, stretch=1)

    def _setup_viewer_toolbar(self, show_toolbar: bool) -> None:
        self.toolbar_visible = bool(show_toolbar)
        if not show_toolbar:
            return
        self.toolbar = QtWidgets.QToolBar(self.viewer_panel)
        self.toolbar.setObjectName("nodeViewerToolbar")
        self.toolbar.addAction("Fit", self.fit_graph)
        self.viewer_layout.addWidget(self.toolbar)

    def _install_viewer_layout(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewer_panel, stretch=1)

    def load_graph_dict(self, data: dict[str, Any]) -> None:
        """Load a graph dictionary and fit it into the view."""
        try:
            self.scene.from_dict(data)
            self.fit_graph()
            self.graphLoaded.emit(dict(data))
        except Exception as exc:
            self.graphLoadFailed.emit(
                {
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "phase": "load_graph_dict",
                }
            )

    def load_graph_from_json(self, json_str: str) -> None:
        """Load a graph from a JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            error = {
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "phase": "parse_graph_json",
            }
            logger.error("Error parsing graph JSON: %s", exc)
            self.graphLoadFailed.emit(error)
            return
        self.load_graph_dict(data)

    def graph_dict(self) -> dict[str, Any]:
        """Return the current graph as a dictionary."""
        return self.scene.to_dict()

    def to_json(self) -> str:
        """Serialize the current graph to a JSON string."""
        return json.dumps(self.graph_dict(), indent=2)

    def clear_graph(self) -> None:
        """Clear all nodes and edges from the viewer."""
        self.scene.clear()

    def fit_graph(self) -> None:
        """Fit the graph into the view."""
        self.view.fit_all()

    def selected_node(self) -> dict[str, Any]:
        """Return the last selected node payload."""
        return dict(self._selected_node)

    def selected_edge(self) -> dict[str, Any]:
        """Return the last selected edge payload."""
        return dict(self._selected_edge)

    def set_toolbar_visible(self, visible: bool) -> None:
        """Show or hide the optional viewer toolbar."""
        if self.toolbar is None and visible:
            self._setup_viewer_toolbar(True)
        if self.toolbar is not None:
            self.toolbar.setVisible(bool(visible))
        self.toolbar_visible = bool(visible)

    def set_read_only(self, read_only: bool) -> None:
        """Enable or disable read-only mode."""
        self.scene.read_only = bool(read_only)

    def _port_to_entry(self, port: PortSpec) -> str | dict[str, Any]:
        has_type = bool(getattr(port, "port_type", ""))
        is_fixed = bool(getattr(port, "fixed", False))
        has_min = getattr(port, "min_value", None) is not None
        has_max = getattr(port, "max_value", None) is not None
        if not (has_type or is_fixed or has_min or has_max):
            return port.name
        entry: dict[str, Any] = {"name": port.name}
        if has_type:
            entry["type"] = str(getattr(port, "port_type", ""))
        if is_fixed:
            entry["fixed"] = True
        if has_min:
            try:
                entry["min"] = float(getattr(port, "min_value", 0.0))
            except Exception:
                pass
        if has_max:
            try:
                entry["max"] = float(getattr(port, "max_value", 0.0))
            except Exception:
                pass
        return entry

    def _on_scene_selection_changed(self) -> None:
        from .edge_item import EdgeGraphicsItem
        from .node_item import NodeGraphicsItem

        selected_items = self.scene.selectedItems()
        selected_nodes = [item for item in selected_items if isinstance(item, NodeGraphicsItem)]
        selected_edges = [item for item in selected_items if isinstance(item, EdgeGraphicsItem)]

        if selected_nodes:
            item = selected_nodes[-1]
            node_id = str(getattr(item.model, "id", ""))
            model: NodeModel = item.model
            node_dict = {
                "id": node_id,
                "title": model.title,
                "inputs": [self._port_to_entry(port) for port in model.inputs],
                "outputs": [self._port_to_entry(port) for port in model.outputs],
                "type": model.node_type,
                "config": model.config or {},
                "pos": [float(item.pos().x()), float(item.pos().y())],
                "collapsed": bool(getattr(item, "collapsed", False)),
                "z": float(item.zValue()),
            }
            self._selected_node = node_dict
            self.nodeSelected.emit(dict(node_dict))

        if selected_edges:
            edge = selected_edges[-1]
            if edge.start_port is not None and edge.end_port is not None:
                src_item = edge.start_port.node_item
                dst_item = edge.end_port.node_item
                src_id = str(getattr(src_item.model, "id", ""))
                dst_id = str(getattr(dst_item.model, "id", ""))
                edge_dict: dict[str, Any] = {
                    "source": src_id,
                    "source_port": int(edge.start_port.index),
                    "target": dst_id,
                    "target_port": int(edge.end_port.index),
                }
                cfg = getattr(edge, "_config", None)
                if cfg is None:
                    cfg = getattr(edge, "_style_config", None)
                if isinstance(cfg, dict) and cfg:
                    edge_dict["config"] = dict(cfg)
                self._selected_edge = edge_dict
                self.edgeSelected.emit(dict(edge_dict))


__all__ = ["NodeViewerWidget"]
