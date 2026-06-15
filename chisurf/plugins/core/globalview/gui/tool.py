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


COMPACT_STYLE = """
QGroupBox {
    margin-top: 10px;
    padding: 1px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 4px;
    padding: 0 2px;
}
QToolButton {
    margin: 0;
    padding: 1px 6px;
}
QCheckBox {
    margin: 0;
    padding: 0;
    spacing: 2px;
}
QLabel {
    margin: 0;
    padding: 0 2px 0 0;
}
QLineEdit,
QDoubleSpinBox {
    margin: 0;
    padding: 1px 3px;
}
QComboBox {
    margin: 0;
    padding: 1px 8px;
}
QTabWidget::pane {
    margin: 0;
    padding: 0;
}
QTabBar::tab {
    margin: 0;
    padding: 2px 6px;
}
QTableView {
    margin: 0;
    padding: 0;
}
"""


@persist_plugin_state("globalview")
class GraphWizard(QtWidgets.QMainWindow):

    graph_layouts = GRAPH_LAYOUTS

    node_colors = dict(NODE_COLORS)

    @staticmethod
    def _compact_layouts(widget: QtWidgets.QWidget) -> None:
        """Tighten layout margins and spacing for this plugin.

        Parameters
        ----------
        widget : QWidget
            Root widget whose descendant layouts should be compacted.
        """
        for layout in widget.findChildren(QtWidgets.QLayout):
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(1)

    def _setup_ui(self):
        """Set up the main window UI with toolbar, tabs, and statusbar."""
        self.setWindowTitle("🕸️ ChiSurf Network Visualization")
        
        self.parameter_layout = QtWidgets.QVBoxLayout()
        self.parameter_layout.setContentsMargins(0, 0, 0, 0)
        self.parameter_layout.setSpacing(0)
        
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        
        self._setup_toolbar()
        self._setup_tabs(central_widget, main_layout)
        self._setup_statusbar()

    def _setup_toolbar(self):
        """Create the toolbar with save, load, and other actions."""
        toolbar = self.addToolBar("GlobalView Toolbar")
        toolbar.setObjectName("globalview_toolbar")
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        
        self._btn_save = QtWidgets.QToolButton()
        self._btn_save.setText("💾 Save")
        self._btn_save.setToolTip("💾 Save the current graph to a GraphML file")
        toolbar.addWidget(self._btn_save)
        
        self._btn_load = QtWidgets.QToolButton()
        self._btn_load.setText("📂 Load")
        self._btn_load.setToolTip("📂 Load a graph from a GraphML file")
        toolbar.addWidget(self._btn_load)
        
        toolbar.addSeparator()
        
        self._btn_redraw = QtWidgets.QToolButton()
        self._btn_redraw.setText("🔄 Redraw")
        self._btn_redraw.setToolTip("🔄 Redraw the graph with current settings")
        toolbar.addWidget(self._btn_redraw)

    def _setup_tabs(self, central_widget, main_layout):
        """Create the tab widget with Graph and Parameters tabs."""
        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        self._setup_graph_tab()
        self._setup_params_tab()

    def _setup_graph_tab(self):
        """Set up the Graph tab with visualization and link controls."""
        graph_tab = QtWidgets.QWidget()
        graph_tab_layout = QtWidgets.QVBoxLayout(graph_tab)
        graph_tab_layout.setContentsMargins(2, 2, 2, 2)
        graph_tab_layout.setSpacing(2)
        
        self.graphTabLayout = graph_tab_layout
        
        self.tab_widget.addTab(graph_tab, "🕸️ Graph")
        
        self._setup_visualization_controls(graph_tab_layout)
        self._setup_link_controls(graph_tab_layout)

    def _setup_visualization_controls(self, parent_layout):
        """Create the visualization controls group."""
        viz_group = QtWidgets.QGroupBox("🕸️ Visualization")
        viz_group.setToolTip("🕸️ Graph visualization settings")
        viz_layout = QtWidgets.QGridLayout(viz_group)
        viz_layout.setContentsMargins(2, 2, 2, 2)
        viz_layout.setSpacing(2)
        
        row = 0
        
        self._label_node_size = QtWidgets.QLabel("⚪ Node size:")
        self._label_node_size.setToolTip("⚪ Size of nodes in the graph visualization")
        viz_layout.addWidget(self._label_node_size, row, 0)
        
        self._spin_node_size = QtWidgets.QDoubleSpinBox()
        self._spin_node_size.setMinimum(0.01)
        self._spin_node_size.setSingleStep(0.05)
        self._spin_node_size.setValue(0.02)
        self._spin_node_size.setToolTip("⚪ Adjust the size of parameter nodes in the graph")
        viz_layout.addWidget(self._spin_node_size, row, 1)
        
        row += 1
        
        self._label_graph_scale = QtWidgets.QLabel("📏 Graph scale:")
        self._label_graph_scale.setToolTip("📏 Scale factor for the graph layout")
        viz_layout.addWidget(self._label_graph_scale, row, 0)
        
        self._spin_graph_scale = QtWidgets.QDoubleSpinBox()
        self._spin_graph_scale.setMinimum(0.01)
        self._spin_graph_scale.setSingleStep(0.01)
        self._spin_graph_scale.setValue(1.00)
        self._spin_graph_scale.setToolTip("📏 Scale the entire graph layout")
        viz_layout.addWidget(self._spin_graph_scale, row, 1)
        
        row += 1
        
        self._label_layout = QtWidgets.QLabel("🎨 Layout:")
        self._label_layout.setToolTip("🎨 Choose a graph layout algorithm")
        viz_layout.addWidget(self._label_layout, row, 0)
        
        self._combo_layout = QtWidgets.QComboBox()
        self._combo_layout.setToolTip("🎨 Select graph layout algorithm (kamada_kawai, spring, shell, arf, spectral)")
        viz_layout.addWidget(self._combo_layout, row, 1, 1, 2)
        
        row += 1
        
        self._btn_redraw_tab = QtWidgets.QToolButton()
        self._btn_redraw_tab.setText("🔄 Redraw")
        self._btn_redraw_tab.setToolTip("🔄 Redraw the graph with current settings")
        viz_layout.addWidget(self._btn_redraw_tab, row, 2)
        
        row += 1
        
        self._check_connect_fits = QtWidgets.QCheckBox("🔗 Connect fits")
        self._check_connect_fits.setToolTip("🔗 Draw connections between fits (visual only, does not link parameters)")
        viz_layout.addWidget(self._check_connect_fits, row, 0, 1, 2)
        
        row += 1
        
        self._check_include_fixed = QtWidgets.QCheckBox("📌 Include fixed")
        self._check_include_fixed.setToolTip("📌 Include fixed parameters in the graph visualization")
        viz_layout.addWidget(self._check_include_fixed, row, 0, 1, 2)
        
        parent_layout.addWidget(viz_group)

    def _setup_link_controls(self, parent_layout):
        """Create the link controls group."""
        link_group = QtWidgets.QGroupBox("🔗 Link")
        link_group.setToolTip("🔗 Parameter linking controls")
        link_layout = QtWidgets.QGridLayout(link_group)
        link_layout.setContentsMargins(2, 2, 2, 2)
        link_layout.setSpacing(2)
        
        self._param_widget = QtWidgets.QWidget()
        self._param_widget.setLayout(self.parameter_layout)
        link_layout.addWidget(self._param_widget, 0, 0, 4, 1)
        
        self._btn_link = QtWidgets.QToolButton()
        self._btn_link.setText("🔗 Link")
        self._btn_link.setToolTip("🔗 Link selected parameters (first selection = master)")
        link_layout.addWidget(self._btn_link, 0, 1)
        
        self._btn_clear_links = QtWidgets.QToolButton()
        self._btn_clear_links.setText("🧹 Clear")
        self._btn_clear_links.setToolTip("🧹 Clear links from selected parameters")
        link_layout.addWidget(self._btn_clear_links, 1, 1)
        
        self._check_clear_all = QtWidgets.QCheckBox("☑️ all")
        self._check_clear_all.setToolTip("☑️ Clear links from ALL parameters when clearing")
        link_layout.addWidget(self._check_clear_all, 2, 1)
        
        parent_layout.addWidget(link_group)

    def _setup_params_tab(self):
        """Set up the Parameters tab."""
        params_tab = QtWidgets.QWidget()
        params_tab_layout = QtWidgets.QVBoxLayout(params_tab)
        params_tab_layout.setContentsMargins(0, 0, 0, 0)
        params_tab_layout.setSpacing(0)
        
        self.paramsTabLayout = params_tab_layout
        self.tab_widget.addTab(params_tab, "🎛️ Parameters")

    def _setup_statusbar(self):
        """Create the status bar."""
        self.status_bar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready", 3000)

    def _setup_connections(self):
        """Set up widget connections after UI is created."""
        self.parameter_table_view = ParameterTableView(self)
        self.parameter_table_view.refresh_from_fits(self.fit_list)
        if self.paramsTabLayout is not None:
            self.paramsTabLayout.addWidget(self.parameter_table_view)
        self.parameter_table_view.paramChanged.connect(self.recompute_graph)
        
        self._combo_layout.addItems(self.graph_layouts)
        
        self._btn_link.clicked.connect(self.link_selection)
        self._btn_clear_links.clicked.connect(self.link_clear)
        self._btn_redraw.clicked.connect(self.recompute_graph)
        self._btn_redraw_tab.clicked.connect(self.recompute_graph)
        self._btn_save.clicked.connect(self.write_graph)
        self._btn_load.clicked.connect(self.read_graph)

        self._check_connect_fits.stateChanged.connect(self.recompute_graph)
        self._check_include_fixed.stateChanged.connect(self.recompute_graph)
        self._combo_layout.currentIndexChanged.connect(self.recompute_graph)
        self._spin_node_size.valueChanged.connect(self.recompute_graph)
        self._spin_graph_scale.valueChanged.connect(self.recompute_graph)

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
        return self._check_clear_all.isChecked()

    @property
    def selected_nodes(self):
        return [self.node_data["objects"][x] for x in self.graph_widget.g.selected_nodes_idx]

    @property
    def graph_layout(self):
        return self._combo_layout.currentText()

    @property
    def include_fixed(self):
        return self._check_include_fixed.isChecked()

    @property
    def connect_fits(self):
        return self._check_connect_fits.isChecked()

    @property
    def graph_scale(self):
        return self._spin_graph_scale.value()

    @property
    def node_size(self):
        return self._spin_node_size.value()

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
        self._compact_layouts(w)
        return node_data

    def __init__(
        self,
        fit_list: Optional[List[Any]] = None,
        parent=None,
        connect_fits: bool = False,
        include_fixed: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(parent)
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

        self._setup_ui()
        self._setup_connections()
        
        self.node_data = self.make_graph_plot(
            connect_fits=connect_fits,
            update_callback=self.callback_selection,
            include_fixed=include_fixed,
            fit_list=fit_list,
        )
        
        self.setStyleSheet(COMPACT_STYLE)
        self._compact_layouts(self)
        self._connect_events()

    def _connect_events(self) -> None:
        """Subscribe to server events that require a graph rebuild."""
        from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client
        fc = get_fitting_client()
        if fc is None:
            return
        self._subscription_tokens = []
        cb = lambda p: QtCore.QTimer.singleShot(0, self.recompute_graph)
        fc.subscribe("fit.", cb)
        self._subscription_tokens.append(("fit.", cb))
        cb2 = lambda p: QtCore.QTimer.singleShot(0, self.recompute_graph)
        fc.subscribe("parameter.", cb2)
        self._subscription_tokens.append(("parameter.", cb2))

    def closeEvent(self, event):
        fc = get_fitting_client()
        if fc is not None:
            for topic, cb in getattr(self, '_subscription_tokens', []):
                try:
                    fc.unsubscribe(topic, cb)
                except Exception:
                    pass
        super().closeEvent(event)


if __name__ == "plugin":
    graph_wiz = GraphWizard()
    graph_wiz.show()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    graph_wiz = GraphWizard()
    graph_wiz.setWindowTitle("🕸️ ChiSurf Parameter Network")
    graph_wiz.show()
    sys.exit(app.exec_())
