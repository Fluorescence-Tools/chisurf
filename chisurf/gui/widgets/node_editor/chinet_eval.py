from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging

import chinet as cn
from qtpy import QtCore, QtWidgets

from .node_item import NodeGraphicsItem
from .edge_item import EdgeGraphicsItem

logger = logging.getLogger(__name__)


def _build_chinet_eval_order(scene) -> Optional[List[NodeGraphicsItem]]:
    """Return a topological order of PT nodes for evaluation, or None on error.

    The order is computed over all NodeGraphicsItem instances in the scene
    using Kahn's algorithm, with edge direction inferred from port roles
    (output -> input).
    """

    nodes: List[NodeGraphicsItem] = [
        it for it in scene.items() if isinstance(it, NodeGraphicsItem)
    ]
    if not nodes:
        return []

    indegree: Dict[NodeGraphicsItem, int] = {n: 0 for n in nodes}
    succ: Dict[NodeGraphicsItem, List[NodeGraphicsItem]] = {n: [] for n in nodes}

    node_set = set(nodes)

    for edge in list(getattr(scene, "edges", [])):
        if not isinstance(edge, EdgeGraphicsItem):
            continue
        start = getattr(edge, "start_port", None)
        end = getattr(edge, "end_port", None)
        if start is None or end is None:
            continue
        src_item = start.node_item
        dst_item = end.node_item
        if src_item not in node_set or dst_item not in node_set:
            continue
        # Direction: output -> input when possible
        if start.spec.is_output and not end.spec.is_output:
            src = src_item
            dst = dst_item
        elif end.spec.is_output and not start.spec.is_output:
            src = dst_item
            dst = src_item
        else:
            continue
        succ[src].append(dst)
        indegree[dst] += 1

    # Kahn's algorithm
    order: List[NodeGraphicsItem] = []
    queue: List[NodeGraphicsItem] = [n for n, d in indegree.items() if d == 0]

    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in succ.get(n, []):
            indegree[m] -= 1
            if indegree[m] == 0:
                queue.append(m)

    if len(order) != len(nodes):
        logger.error("chinet evaluation: graph contains a cycle or disconnected component")
        return None

    return order


def evaluate_pt_graph(scene) -> Optional[Dict[NodeGraphicsItem, Any]]:
    """Evaluate the current graph using chinet for PT nodes only.

    This operates on nodes of types ``pt_constant``, ``pt_transform`` and
    ``pt_output`` without affecting the existing example nodes.
    """

    try:
        acyclic = scene.is_directed_acyclic()
    except Exception:
        acyclic = None

    if acyclic is False:
        logger.error("Cannot evaluate chinet graph: scene contains cycles")
        return None

    order = _build_chinet_eval_order(scene)
    if order is None:
        return None

    # Map node items to their evaluated output value (or dict of values)
    values: Dict[NodeGraphicsItem, Any] = {}

    for item in order:
        model = item.model
        node_type = getattr(model, "node_type", "")

        if node_type == "pt_constant":
            try:
                val = float(model.config.get("value", 0.0))
            except Exception:
                val = 0.0
            try:
                port = cn.Port(value=val, is_reactive=True)
                _ = port  # keep local to show integration; value is used below
            except Exception:
                pass
            values[item] = val
            continue

        if node_type == "pt_transform":
            code = str(
                model.config.get(
                    "code",
                    "def f(A=0.0, B=0.0):\n    return {'out_00': A + B, 'out_01': A - B, 'out_02': A * B}",
                )
            )
            try:
                import types as _types
                import inspect as _inspect  # noqa: F401  # kept for similarity

                code_obj = compile(code, "<node_editor_pt>", "exec")
                func_obj = None
                for obj in code_obj.co_consts:
                    if isinstance(obj, _types.CodeType):
                        func_obj = _types.FunctionType(obj, globals())
                        break
                if func_obj is None:
                    raise ValueError("No function object found in code")
            except Exception as exc:
                logger.error("Failed to compile PT Transform '%s': %s", model.title, exc)
                continue

            # Construct chinet.Node the same way chisurf's function_to_model_decorator does
            try:
                node = cn.Node()
                try:
                    node.set_python_callback_function(func_obj)
                except Exception as exc:
                    logger.error("Failed to set python callback for '%s': %s", model.title, exc)
                    continue

                # Mark ports as reactive when possible
                try:
                    for p in getattr(node, "inputs", {}).values():
                        try:
                            p.is_reactive = True  # type: ignore[assignment]
                        except Exception:
                            pass
                    for p in getattr(node, "outputs", {}).values():
                        try:
                            p.is_reactive = True  # type: ignore[assignment]
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception as exc:
                logger.error("Failed to create chinet Node for '%s': %s", model.title, exc)
                continue

            # Gather scalar inputs from incoming edges by port name.
            # When a source node produces multiple outputs (e.g. pt_transform),
            # use the specific output selected by the connected port name
            # (e.g. "out_00", "out_01").
            input_values: Dict[str, Any] = {}
            for edge in list(getattr(scene, "edges", [])):
                if not isinstance(edge, EdgeGraphicsItem):
                    continue
                end = getattr(edge, "end_port", None)
                start = getattr(edge, "start_port", None)
                if end is None or start is None:
                    continue
                if end.node_item is not item or end.spec.is_output:
                    continue
                src_item = start.node_item
                if src_item not in values:
                    continue
                port_name = str(end.spec.name)
                upstream_val = values[src_item]
                if isinstance(upstream_val, dict):
                    src_port_name = str(start.spec.name)
                    upstream_val = upstream_val.get(src_port_name)
                input_values[port_name] = upstream_val

            try:
                for pname, v in input_values.items():
                    try:
                        p = node.inputs[pname]
                    except Exception:
                        continue
                    try:
                        p.value = v
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                node.evaluate()
            except Exception as exc:
                logger.error("Error during chinet Node.evaluate for '%s': %s", model.title, exc)
                continue

            # Collect all available outputs into a dict keyed by port name.
            out_map: Dict[str, Any] = {}
            try:
                for oname, oport in getattr(node, "outputs", {}).items():
                    try:
                        out_map[str(oname)] = oport.value
                    except Exception:
                        out_map[str(oname)] = None
            except Exception:
                out_map = {}
            values[item] = out_map
            continue

        if node_type == "pt_plot":
            # Plot node: consume upstream X/Y arrays and update PtPlotWidget
            x_val: Any = None
            y_val: Any = None
            for edge in list(getattr(scene, "edges", [])):
                if not isinstance(edge, EdgeGraphicsItem):
                    continue
                end = getattr(edge, "end_port", None)
                start = getattr(edge, "start_port", None)
                if end is None or start is None:
                    continue
                if end.node_item is not item or end.spec.is_output:
                    continue
                src_item = start.node_item
                if src_item not in values:
                    continue
                upstream_val = values[src_item]
                if isinstance(upstream_val, dict):
                    src_port_name = str(start.spec.name)
                    upstream_val = upstream_val.get(src_port_name)
                if end.spec.name == "X":
                    x_val = upstream_val
                elif end.spec.name == "Y":
                    y_val = upstream_val

            values[item] = {"X": x_val, "Y": y_val}

            # Update embedded PtPlotWidget if present
            try:
                from .widgets.pt_plot_widget import PtPlotWidget  # type: ignore

                w = item.content_widget
                if w is not None:
                    plots = w.findChildren(PtPlotWidget)
                    if plots:
                        plots[0].set_data(x_val, y_val)
            except Exception:
                pass

            continue

        if node_type == "pt_output":
            # Pass-through of the latest connected upstream value
            final_val: Any = None
            for edge in list(getattr(scene, "edges", [])):
                if not isinstance(edge, EdgeGraphicsItem):
                    continue
                end = getattr(edge, "end_port", None)
                start = getattr(edge, "start_port", None)
                if end is None or start is None:
                    continue
                if end.node_item is not item or end.spec.is_output:
                    continue
                src_item = start.node_item
                if src_item in values:
                    upstream_val = values[src_item]
                    if isinstance(upstream_val, dict):
                        src_port_name = str(start.spec.name)
                        final_val = upstream_val.get(src_port_name)
                    else:
                        final_val = upstream_val

            values[item] = final_val

            # Update the embedded label if present
            try:
                w = item.content_widget
                if w is not None:
                    labels = w.findChildren(QtWidgets.QLabel)
                    if labels:
                        labels[-1].setText(str(final_val))
            except Exception:
                pass

            continue

    return values
