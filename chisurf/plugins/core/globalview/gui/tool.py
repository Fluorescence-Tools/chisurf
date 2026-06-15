from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx

import pyqtgraph as pg

from qtpy import QtWidgets, QtCore, QtGui

import chisurf as cs
import chisurf.gui.widgets
import chisurf.core.fitting.fit
import chisurf.core.models
import chisurf.core.parameter
from chisurf.core.parameter import Parameter
from chisurf import logging
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client

from chisurf.plugins.core.globalview.api.graph import build_graph as api_build_graph
from chisurf.plugins.core.globalview.api.graph import GraphResult
from chisurf.plugins.core.globalview.gui.adapter import (
    NODE_COLORS,
    compute_node_types,
    graph_result_to_networkx,
    compute_layout,
)
from chisurf.plugins.core.globalview.gui.graphplotwidget import GraphPlotWidget
from chisurf.plugins.core.globalview.gui.parameter_table_view import ParameterTableView

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c


GRAPH_LAYOUTS = [
    "kamada_kawai",
    "spring",
    "shell",
    "arf",
    "spectral",
]


@persist_plugin_state("globalview")
class GraphWizard(QtWidgets.QWidget):

    graph_layouts = GRAPH_LAYOUTS

    node_colors = dict(NODE_COLORS)

    def recompute_graph(self):
        node_data = self.make_graph_plot(
            update_callback=self.callback_selection,
            fit_list=self.fit_list,
            connect_fits=self.connect_fits,
            include_fixed=self.include_fixed,
            node_size=self.node_size,
        )
        self.node_data = node_data

    @property
    def clear_all(self):
        return self.checkBox_3.isChecked()

    @property
    def selected_nodes(self):
        return [self.node_data["objects"][x] for x in self.graph_widget.g.selected_nodes_idx]

    @property
    def graph_layout(self):
        return self.comboBox_layout.currentText()

    @property
    def include_fixed(self):
        return self.checkBox_include_fixed.isChecked()

    @property
    def connect_fits(self):
        return self.checkBox_connect_fits.isChecked()

    @property
    def graph_scale(self):
        return self.doubleSpinBox_2.value()

    @property
    def node_size(self):
        return self.doubleSpinBox.value()

    def read_graph(self, evt, *args, **kwargs):
        path = kwargs.get(
            "path",
            cs.gui.widgets.get_filename(
                description="ChiSurf-GraphML",
                file_type="CS-GraphML (*.gml)",
            ),
        )
        G = nx.read_graphml(path)
        self.link(G, **kwargs)

    def write_graph(self, evt, G: nx.Graph = None):
        if G is None:
            G = self.G
        path = cs.gui.widgets.save_file(
            description="ChiSurf-GraphML",
            file_type="CS-GraphML (*.gml)",
        )
        nx.write_graphml(
            G,
            path,
            encoding="utf-8",
            prettyprint=True,
            infer_numeric_types=False,
            named_key_ids=False,
            edge_id_from_attribute=None,
        )

    def callback_selection(self):
        cs.gui.widgets.general.clear_layout(self.parameter_layout)
        for node in self.selected_nodes:
            w = cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(node)
            self.parameter_layout.addWidget(w)

    def _fit_idx_for_node(self, obj: Any) -> Optional[int]:
        try:
            idx = self.node_data.get("objects", []).index(obj)
            return self.node_data.get("fit_indices", [])[idx]
        except (ValueError, IndexError):
            return None

    def _validate_and_link(self, source, target) -> bool:
        if Parameter.check_recursive_link(target, source):
            cs.logging.log(0, f"Cycle detected: cannot link {source.name} -> {target.name}")
            return False
        fc = get_fitting_client()
        if fc is not None:
            src_fit_idx = self._fit_idx_for_node(source)
            tgt_fit_idx = self._fit_idx_for_node(target)
            fc.link_parameters(
                parameter_name=str(source.name),
                target_parameter_name=str(target.name),
                fit_index=src_fit_idx,
                target_fit_index=tgt_fit_idx,
            )
        return True

    def on_link_requested(self, source_idx: int, target_idx: int):
        source = self.node_data["objects"][source_idx]
        target = self.node_data["objects"][target_idx]
        if self._validate_and_link(source, target):
            self.recompute_graph()

    def on_link_removal_requested(self, source_idx: int):
        param = self.node_data["objects"][source_idx]
        fit_idx = self.node_data.get("fit_indices", [None])[source_idx]
        fc = get_fitting_client()
        if fc is not None:
            fc.unlink_parameter(
                parameter_name=str(param.name),
                fit_index=fit_idx,
            )
        self.recompute_graph()

    def link_selection(self):
        logging.log(0, "link_selection(self)")
        target, source = self.selected_nodes[:2]
        if self._validate_and_link(source, target):
            self.recompute_graph()

    def link_clear(self):
        logging.log(0, "link_clear(self)")
        fc = get_fitting_client()
        if self.clear_all:
            for fit_idx, fit in enumerate(self.fit_list):
                for p in getattr(fit.model, "parameters_all", []):
                    if fc is not None:
                        fc.unlink_parameter(
                            parameter_name=str(p.name),
                            fit_index=fit_idx,
                        )
        else:
            for n in self.selected_nodes:
                fit_idx = self._fit_idx_for_node(n)
                if fc is not None:
                    fc.unlink_parameter(
                        parameter_name=str(n.name),
                        fit_index=fit_idx,
                    )
        self.recompute_graph()

    def get_fit(self, G, node):
        return self.fit_list[G.nodes[node]["fit.idx"]]

    def get_parameters(self, G, node):
        if G.nodes[node]["node.type"] == "parameter":
            fit = self.get_fit(G, node)
            return fit.model.parameters_all_dict[G.nodes[node]["node.name"]]
        return None

    def link(
        self,
        G: nx.Graph,
        clear_fist: bool = False,
        **kwargs,
    ):
        cs.logging.log(0, "link")
        self.G = G
        fc = get_fitting_client()

        if clear_fist:
            self.link_clear()

        for node in G.nodes:
            p = self.get_parameters(G, node)
            if p is not None:
                if fc is not None:
                    fit_idx = G.nodes[node].get("fit.idx")
                    fc.set_parameter_value(
                        parameter_name=str(p.name),
                        value=float(G.nodes[node]["value"]),
                        fit_index=fit_idx,
                    )
                    fc.set_parameter_fixed(
                        parameter_name=str(p.name),
                        fixed=bool(G.nodes[node]["fixed"]),
                        fit_index=fit_idx,
                    )

        for edge in G.edges:
            n1, n2 = edge
            p1 = self.get_parameters(G, n1)
            p2 = self.get_parameters(G, n2)
            if p1 is not None and p2 is not None:
                if fc is not None:
                    fc.link_parameters(
                        parameter_name=str(p2.name),
                        target_parameter_name=str(p1.name),
                        fit_index=G.nodes[n2].get("fit.idx"),
                        target_fit_index=G.nodes[n1].get("fit.idx"),
                    )

        self.recompute_graph()

    @staticmethod
    def skip_fit(fit, omitted_models: list[cs.core.models.Model] = None):
        if omitted_models is None:
            omitted_models = [cs.core.models.global_model.GlobalFitModel]
        for c in omitted_models:
            if isinstance(fit.model, c):
                print("Omit:", fit.name)
                return True
        return False

    @staticmethod
    def build_graph(
        include_fixed: bool = True,
        fit_list: List[Any] = None,
        connect_fits: bool = False,
        **kwargs,
    ):
        if fit_list is None:
            fc = get_fitting_client()
            fit_list = fc.get_fit_objects() if fc is not None else []
        api_result = api_build_graph(fit_list, include_fixed, connect_fits)
        G = graph_result_to_networkx(api_result)
        node_objects = {}
        for n in api_result.nodes:
            if n.node_type == "fit":
                node_objects[n.node_idx] = fit_list[n.fit_idx] if n.fit_idx < len(fit_list) else None
            else:
                try:
                    fit = fit_list[n.fit_idx]
                    p = getattr(fit.model, "parameters_all_dict", {}).get(n.name)
                    node_objects[n.node_idx] = p
                except Exception:
                    node_objects[n.node_idx] = None
        return G, node_objects

    def make_graph(
        self,
        connect_fits: bool = False,
        include_fixed: bool = False,
        fit_list: Optional[List[Any]] = None,
    ):
        if fit_list is None:
            fc = get_fitting_client()
            fit_list = fc.get_fit_objects() if fc is not None else []
        G, node_objects = self.build_graph(include_fixed, fit_list, connect_fits)

        connections: List[List[int]] = []
        for edge in G.edges:
            n1, n2 = edge
            connections.append(
                [G.nodes[n1]["node.idx"], G.nodes[n2]["node.idx"]]
            )

        node_names = []
        node_ids = []
        node_types = []
        for node in G.nodes:
            o = node_objects[node]
            if o is not None and isinstance(o, cs.core.parameter.Parameter):
                if o.fixed:
                    if not include_fixed:
                        continue
                    node_types.append(1)
                else:
                    if o.is_linked:
                        node_types.append(2)
                    else:
                        node_types.append(3)
            else:
                node_types.append(0)
            node_names.append(G.nodes[node]["node.name"])
            node_ids.append(node)

        fit_indices = []
        for k in node_ids:
            n_data = G.nodes.get(k, {})
            fit_indices.append(n_data.get("fit.idx"))

        node_data = {
            "ids": node_ids,
            "types": node_types,
            "names": node_names,
            "objects": [node_objects[k] for k in node_ids],
            "fit_indices": fit_indices,
        }

        self.G = G
        return connections, node_data

    def get_node_positions(
        self,
        G: nx.Graph = None,
        graph_scale: float = None,
        graph_layout: str = None,
    ):
        if G is None:
            G = self.G
        if graph_scale is None:
            graph_scale = self.graph_scale
        if graph_layout is None:
            graph_layout = self.graph_layout
        pos = compute_layout(G, graph_layout, graph_scale)
        return pos

    def make_graph_plot(
        self,
        fit_list: List[Any],
        update_callback=None,
        node_size: float = 0.02,
        connect_fits: bool = False,
        include_fixed: bool = True,
    ) -> Dict[str, Any]:
        w = QtWidgets.QWidget(parent=self)
        w.setParent(self)
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)

        s = pg.GraphicsLayoutWidget(show=True)
        v = s.addViewBox()
        v.setAspectLocked()
        l.addWidget(s)

        g = GraphPlotWidget(update_callback=update_callback)
        g.linkRequested.connect(self.on_link_requested)
        g.linkRemovalRequested.connect(self.on_link_removal_requested)
        w.g = g
        v.addItem(g)

        connections, node_data = self.make_graph(
            connect_fits,
            fit_list=fit_list,
            include_fixed=include_fixed,
        )

        pos = self.get_node_positions()
        symbolBrush = np.array([self.node_colors[i] for i in node_data["types"]])
        pos = np.array([pos[i] for i in node_data["ids"]], dtype=np.float64)
        adj = np.array(connections)
        if len(pos) > 0:
            g.setData(
                pos=pos,
                adj=adj,
                size=node_size,
                pxMode=False,
                text=node_data["names"],
                symbolBrush=symbolBrush,
                node_types=node_data["types"],
            )

        old = getattr(self, "graph_widget", None)
        try:
            if old is not None:
                old.setParent(None)
                old.deleteLater()
        except Exception:
            pass
        if hasattr(self, "graphTabLayout") and self.graphTabLayout is not None:
            self.graphTabLayout.addWidget(w)
        self.graph_widget = w
        return node_data

    @cs.gui.decorators.init_with_ui(
        "globalview.ui",
        path=cs.core.settings.plugin_path / "core" / "globalview",
    )
    def __init__(
        self,
        fit_list: Optional[List[Any]] = None,
        parent=None,
        connect_fits: bool = False,
        include_fixed: bool = False,
        *args,
        **kwargs,
    ):
        if fit_list is None:
            fc = get_fitting_client()
            fit_list = fc.get_fit_objects() if fc is not None else []
        self.fit_list = fit_list
        self.parent = parent

        self.G: nx.Graph = None
        self.graph_widget = None
        self.node_objects: Dict[Any, Any] = {}
        self.node_data: Dict[str, Any] = {}
        self.connections: List[List[int]] = []

        self.node_data = self.make_graph_plot(
            connect_fits=connect_fits,
            update_callback=self.callback_selection,
            include_fixed=include_fixed,
            fit_list=fit_list,
        )

        self.parameter_table_view = ParameterTableView(self)
        self.parameter_table_view.refresh_from_fits(fit_list)
        if hasattr(self, "paramsTabLayout") and self.paramsTabLayout is not None:
            self.paramsTabLayout.addWidget(self.parameter_table_view)
        self.parameter_table_view.paramChanged.connect(self.recompute_graph)

        self.toolButton.clicked.connect(self.link_selection)
        self.toolButton_2.clicked.connect(self.link_clear)
        self.toolButton_3.clicked.connect(self.recompute_graph)
        self.toolButton_4.clicked.connect(self.write_graph)
        self.toolButton_5.clicked.connect(self.read_graph)

        self.checkBox_connect_fits.stateChanged.connect(self.recompute_graph)
        self.checkBox_include_fixed.stateChanged.connect(self.recompute_graph)
        self.comboBox_layout.currentIndexChanged.connect(self.recompute_graph)
        self.comboBox_layout.addItems(self.graph_layouts)


if __name__ == "plugin":
    graph_wiz = GraphWizard()
    graph_wiz.show()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    graph_wiz = GraphWizard()
    graph_wiz.setWindowTitle("ChiSurf Parameter Network")
    graph_wiz.show()
    sys.exit(app.exec_())
