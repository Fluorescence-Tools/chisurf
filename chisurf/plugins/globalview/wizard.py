import numpy as np
import networkx as nx

import pyqtgraph as pg

import chisurf
import chisurf.gui.widgets
import chisurf.fitting.fit
import chisurf.models
import chisurf.parameter

from chisurf.gui import QtWidgets, QtCore, QtGui
from chisurf import logging
from chisurf.plugins.globalview.graphplotwidget import GraphPlotWidget


class GraphWizard(QtWidgets.QWidget):

    graph_layouts = [
        'kamada_kawai',
        'spring',
        'shell',
        'arf',
        'spectral'
    ]

    node_colors = {
        0: [0, 0, 128, 255],  # fit
        1: [0, 128, 0, 128],  # parameter fixed
        2: [0, 128, 0, 255],  # parameter linked
        3: [128, 0, 128, 255]  # parameter free
    }

    def recompute_graph(self):
        self.graph_widget.close()
        node_data = self.make_graph_plot(
            update_callback=self.callback_selection,
            fit_list=self.fit_list,
            connect_fits=self.connect_fits,
            include_fixed=self.include_fixed,
            node_size=self.node_size
        )
        self.node_data = node_data

    @property
    def clear_all(self):
        return self.checkBox_3.isChecked()

    @property
    def selected_nodes(self):
        return [self.node_data['objects'][x] for x in self.graph_widget.g.selected_nodes_idx]

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
            'path',
            chisurf.gui.widgets.get_filename(
                description='ChiSurf-GraphML',
                file_type='CS-GraphML (*.gml)'
            )
        )
        G = nx.read_graphml(path)
        self.link(G, **kwargs)

    def write_graph(self, evt, G: nx.Graph = None):
        if G is None:
            G = self.G
        path = chisurf.gui.widgets.save_file(
            description='ChiSurf-GraphML',
            file_type='CS-GraphML (*.gml)'
        )
        nx.write_graphml(
            G,
            path,
            encoding='utf-8',
            prettyprint=True,
            infer_numeric_types=False,
            named_key_ids=False,
            edge_id_from_attribute=None
        )

    def callback_selection(self):
        chisurf.gui.widgets.general.clear_layout(self.parameter_layout)
        for node in self.selected_nodes:
            w = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(node)
            self.parameter_layout.addWidget(w)

    def link_selection(self):
        logging.log(0,"link_selection(self)")
        target, source = self.selected_nodes[:2]
        source.link = target
        self.recompute_graph()

    def link_clear(self):
        logging.log(0, "link_clear(self)")
        if self.clear_all:
            for fit in self.fit_list:
                for p in fit.model.parameters_all:
                    p.link = None
        else:
            for n in self.selected_nodes:
                n.link = None
        self.recompute_graph()

    def get_fit(self, G, node):
        return self.fit_list[G.nodes[node]['fit.idx']]

    def get_parameters(self, G, node):
        if G.nodes[node]["node.type"] == "parameter":
            fit = self.get_fit(G, node)
            return fit.model.parameters_all_dict[G.nodes[node]['node.name']]
        return None

    def link(
            self,
            G: nx.Graph,
            clear_fist: bool = False,
            **kwargs
    ):
        chisurf.logging.log(0, "link")
        self.G = G
        for node in G:
            chisurf.logging.log(0, "node:" + node)

        # Clear links
        if clear_fist:
            self.link_clear()

        # Set values / fixed
        for node in G.nodes:
            p = self.get_parameters(G, node)
            if p is not None:
                p.value = G.nodes[node]['value']
                p.fixed = G.nodes[node]['fixed']

        # Add links
        for edge in G.edges:
            n1, n2 = edge
            p1 = self.get_parameters(G, n1)
            p2 = self.get_parameters(G, n2)
            if p1 is not None and p2 is not None:
                p2._link = p1
                p2._port.link = p1._port

        self.recompute_graph()

    @staticmethod
    def skip_fit(fit, omitted_models: list[chisurf.models.Model] = None):
        # Skip fits with global models
        if omitted_models is None:
            omitted_models = [chisurf.models.global_model.GlobalFitModel]
        for c in omitted_models:
            if isinstance(fit.model, c):
                print("Omit:", fit.name)
                return True
        return False

    @staticmethod
    def build_graph(
            include_fixed: bool = True,
            fit_list: list[chisurf.fitting.FitGroup] = None,
            connect_fits: bool = False,
            **kwargs
    ):
        if fit_list is None:
            fit_list = chisurf.fits

        node_objects = dict()
        G = nx.Graph()

        # Add fits as nodes
        node_idx = 0
        for i, fit in enumerate(fit_list):
            if GraphWizard.skip_fit(fit):
                continue

            node_id = id(fit)
            G.add_node(node_id)
            G.nodes[node_id]['node.idx'] = node_idx
            G.nodes[node_id]['node.id'] = node_id
            G.nodes[node_id]['node.parent'] = 'None'
            G.nodes[node_id]['node.name'] = f"ID:{i}:{fit.name}"
            G.nodes[node_id]['node.type'] = 'fit'
            G.nodes[node_id]['fit.idx'] = i
            G.nodes[node_id]['name'] = fit.name
            G.nodes[node_id]['fit.data.filename'] = fit.data.filename
            G.nodes[node_id]['fit.model'] = ".".join(
                [
                    fit.model.__class__.__module__,
                    fit.model.__class__.__name__
                ]
            )
            node_objects[node_id] = fit
            node_idx += 1

        # Add parameters of fit
        for i, fit in enumerate(fit_list):
            if GraphWizard.skip_fit(fit):
                continue

            for j, parameter in enumerate(fit.model.parameters_all):
                if parameter.fixed:
                    if not include_fixed:
                        continue
                node_id = id(parameter)
                node_objects[node_id] = parameter
                parent_id = id(fit)
                G.add_node(node_id)
                G.nodes[node_id]['fit.idx'] = i
                G.nodes[node_id]['node.idx'] = node_idx
                G.nodes[node_id]['node.id'] = node_id
                G.nodes[node_id]['node.parent'] = parent_id
                G.nodes[node_id]['node.type'] = 'parameter'
                G.nodes[node_id]['node.name'] = parameter.name
                G.nodes[node_id]['value'] = parameter.value
                G.nodes[node_id]['fixed'] = parameter.fixed
                G.add_edge(node_id, parent_id)
                node_idx += 1

        # Add connections between parameters across fits
        for node in G.nodes:
            fit_idx = G.nodes[node]['fit.idx']
            fit = fit_list[fit_idx]
            if G.nodes[node]['node.type'] == "parameter":
                parameter_name = G.nodes[node]['node.name']
                n = fit.model.parameters_all_dict[parameter_name]
            else:
                n = fit
            if G.nodes[node]['node.parent'] != 'None':
                G.add_edge(G.nodes[node]['node.id'], G.nodes[node]['node.parent'])
            if G.nodes[node]['node.type'] == "parameter":
                if n.is_linked:
                    G.add_edge(id(n), id(n.link))
        if connect_fits:
            for n1 in G.nodes:
                if G.nodes[n1]['node.type'] != 'fit':
                    continue
                for n2 in G.nodes:
                    if G.nodes[n2]['node.type'] != 'fit':
                        continue
                    if n1 == n2:
                        continue
                    else:
                        G.add_edge(n1, n2)

        return G, node_objects

    def make_graph(
            self,
            connect_fits: bool = False,
            include_fixed: bool = False,
            fit_list: list[chisurf.fitting.FitGroup] = None
    ):
        if fit_list is None:
            fit_list = chisurf.fits
        G, node_objects = self.build_graph(include_fixed, fit_list, connect_fits)

        # indices of connections
        connections: list[(int, int)] = list()

        for edge in G.edges:
            n1, n2 = edge
            logging.log(0,"edge:", edge)
            logging.log(0,"G.nodes[n1]:", G.nodes[n1])
            logging.log(0,"G.nodes[n2]:", G.nodes[n2])
            t = [G.nodes[n1]['node.idx'], G.nodes[n2]['node.idx']]
            connections.append(t)

        node_names = list()
        node_ids = list()
        node_types = list()
        for node in G.nodes:
            o = node_objects[node]
            if isinstance(o, chisurf.parameter.Parameter):
                if o.fixed:
                    if not include_fixed:
                        continue
                    else:
                        node_types.append(1)
                else:
                    if o.is_linked:
                        node_types.append(2)
                    else:
                        node_types.append(3)
            else:
                node_types.append(0)
            node_names.append(G.nodes[node]['node.name'])
            node_ids.append(node)

        node_data = {
            'ids': node_ids,
            'types': node_types,
            'names': node_names,
            'objects': [node_objects[k] for k in node_ids]
        }

        self.G = G
        return connections, node_data

    def get_node_positions(
            self,
            G: nx.Graph = None,
            graph_scale: float = None,
            graph_layout: str = None
    ):
        if G is None:
            G = self.G
        if graph_scale is None:
            graph_scale = self.graph_scale
        if graph_layout is None:
            graph_layout = self.graph_layout
        # Compute layout
        if graph_layout == "shell":
            pos = nx.shell_layout(G, scale=graph_scale)  # , nlist=shells)
        elif graph_layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G, scale=graph_scale)
        elif graph_layout == "planar":
            pos = nx.planar_layout(G, scale=graph_scale)
        elif graph_layout == "arf":
            pos = nx.arf_layout(G, etol=1e-9, dt=0.01)
        elif graph_layout == "spectral":
            pos = nx.spectral_layout(G, scale=graph_scale)
        else:
            pos = nx.spring_layout(G, iterations=500, scale=graph_scale)
        return pos

    def make_graph_plot(
            self,
            fit_list: list[chisurf.fitting.FitGroup],
            update_callback=None,
            node_size: float = 0.02,
            connect_fits: bool = False,
            include_fixed: bool = True
    ) -> (QtWidgets.QWidget, dict):

        w = QtWidgets.QWidget(parent=self)
        w.setParent(self)
        layout = self.layout()
        layout.addWidget(w, 2, 0, 1, 2)
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)

        s = pg.GraphicsLayoutWidget(show=True)
        v = s.addViewBox()
        v.setAspectLocked()

        l.addWidget(s)

        g = GraphPlotWidget(update_callback=update_callback)
        w.g = g
        v.addItem(g)

        connections, node_data = self.make_graph(
            connect_fits,
            fit_list=fit_list,
            include_fixed=include_fixed
        )

        pos = self.get_node_positions()

        symbolBrush = np.array([self.node_colors[i] for i in node_data['types']])

        # Update the graph
        pos = np.array([pos[i] for i in node_data['ids']], dtype=np.float64)
        adj = np.array(connections)
        if len(pos) > 0:
            g.setData(
                pos=pos,
                adj=adj,
                size=node_size,
                pxMode=False,
                text=node_data['names'],
                symbolBrush=symbolBrush
            )
        self.graph_widget = w
        return node_data

    @chisurf.gui.decorators.init_with_ui("globalview.ui", path='chisurf/plugins/globalview/')
    def __init__(
            self,
            fit_list: list[chisurf.fitting.FitGroup] = None,
            parent=None,
            connect_fits: bool = False,
            include_fixed: bool = False,
            *args,
            **kwargs
    ):
        #super(GraphWizard, self).__init__(*args, **kwargs)
        if fit_list is None:
            fit_list = chisurf.fits
        self.fit_list = fit_list

        self.parent = parent

        self.G: nx.Graph = None
        self.graph_widget = None

        # python objects of nodes
        self.node_objects: dict = dict()

        self.node_data: dict = dict()
        self.connections: list[(int, int)] = list()

        self.node_data = self.make_graph_plot(
            connect_fits=connect_fits,
            update_callback=self.callback_selection,
            include_fixed=include_fixed,
            fit_list=fit_list
        )

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


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    graph_wiz = GraphWizard()
    graph_wiz.setWindowTitle('ChiSurf Parameter Network')
    graph_wiz.show()
    sys.exit(app.exec_())

