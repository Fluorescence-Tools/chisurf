from qtpy import QtWidgets, QtCore, QtGui
import json
import numpy as np

from chisurf.gui.widgets.node_editor.scene import NodeScene
from chisurf.gui.widgets.node_editor.view import NodeView
from chisurf.gui.widgets.node_editor.widgets.widget_palette import WidgetPalette
from chisurf.gui.widgets.node_editor.state_tracker import SceneStateTracker
from chisurf.plugins._dev.lightpath_simulator.node_types import optical_registry, build_optical_registry
from chisurf.plugins._dev.lightpath_simulator.simulator import OpticalPathSimulator

class LightPathSimulatorWidget(QtWidgets.QMainWindow):
    """Main window for defining and simulating an optical light path."""
    
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Light Path Simulator")
        self.resize(1000, 700)
        self.db = db
        self.sim = OpticalPathSimulator(db)
        self._is_propagating = False
        
        # Ensure our node registry is populated
        build_optical_registry()
        
        # --- UI Setup ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, central)
        layout.addWidget(splitter)
        
        # --- Left Panel: Node Editor ---
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scene = NodeScene(self)
        self.scene.node_adder = self._on_add_node_requested
        
        self.view = NodeView(self.scene, self)
        left_layout.addWidget(self.view)
        
        self.undo_stack = QtWidgets.QUndoStack(self)
        self.state_tracker = SceneStateTracker(self, self.scene, self.undo_stack)
        self.undo_stack.indexChanged.connect(lambda idx, s=self: s.propagate_graph())
        
        # --- Right Panel: Tools & Results ---
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        right_layout.addWidget(QtWidgets.QLabel("Optical Components:"))
        self.palette = WidgetPalette(self)
        self._load_optical_palette()
        self.palette.nodeTypeActivated.connect(self._on_palette_node_type_activated)
        right_layout.addWidget(self.palette, stretch=1)
        
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_calculate = QtWidgets.QPushButton("Calculate Emission Intensity")
        self.btn_calculate.clicked.connect(self.calculate_crosstalk)
        btn_layout.addWidget(self.btn_calculate)
        right_layout.addLayout(btn_layout)
        
        self.results_table = QtWidgets.QTableWidget()
        right_layout.addWidget(self.results_table, stretch=2)
        
        self.btn_reset = QtWidgets.QPushButton("Reset to Default")
        self.btn_reset.clicked.connect(self._setup_default_path)
        right_layout.addWidget(self.btn_reset)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        self._setup_menus()

        # Start with a default setup
        QtCore.QTimer.singleShot(100, self._setup_default_path)

    def _setup_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        export_act = QtWidgets.QAction("Export Instrument Setting (JSON)...", self)
        export_act.triggered.connect(self._on_export_json)
        file_menu.addAction(export_act)
        
        file_menu.addSeparator()
        
        save_act = QtWidgets.QAction("Save Graph...", self)
        save_act.triggered.connect(self._on_save_graph)
        file_menu.addAction(save_act)
        
        load_act = QtWidgets.QAction("Load Graph...", self)
        load_act.triggered.connect(self._on_load_graph)
        file_menu.addAction(load_act)

    def _on_export_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Instrument Setting", "", "JSON (*.json)")
        if path:
            self.propagate_graph()
            setting = self.sim.to_instrument_setting()
            try:
                with open(path, "w") as f:
                    f.write(setting.to_json())
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Failed", f"Could not save file:\n{e}")

    def _on_save_graph(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Graph", "", "JSON (*.json)")
        if path:
            try:
                state = self.scene.to_dict()
                with open(path, "w") as f:
                    json.dump(state, f, indent=2)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Failed", f"Could not save graph:\n{e}")

    def _on_load_graph(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Graph", "", "JSON (*.json)")
        if path:
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                self.scene.from_dict(state)
                self.propagate_graph()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load Failed", f"Could not load graph:\n{e}")

    def _load_optical_palette(self):
        """Build a custom hierarchy for the palette widget using our optical registry."""
        self.palette.clear()
        group_item = QtWidgets.QTreeWidgetItem(["Optical Path"])
        group_item.setFlags(group_item.flags() & ~QtCore.Qt.ItemIsSelectable)
        self.palette.addTopLevelItem(group_item)
        
        for n_id, n_type in optical_registry.all_types().items():
            child = QtWidgets.QTreeWidgetItem([n_type.title])
            child.setData(0, QtCore.Qt.UserRole, n_id)
            group_item.addChild(child)
            
        group_item.setExpanded(True)

    def _on_palette_node_type_activated(self, node_type_id: str):
        """Drops a node into the center of the current view."""
        self.state_tracker.begin_action(f"Add {node_type_id}")
        view_center = self.view.viewport().rect().center()
        scene_pos = self.view.mapToScene(view_center)
        item = self._on_add_node_requested(node_type_id, scene_pos)
        if item is not None:
            self.scene.addItem(item)
        self.state_tracker.commit_action()

    def _on_add_node_requested(self, node_type_id: str, pos: QtCore.QPointF):
        """Hook called by the scene when a node needs to be created, routing to our registry."""
        node_type = optical_registry.get(node_type_id)
        if node_type is None:
            return None
        
        from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
        from chisurf.gui.widgets.node_editor.model import NodeModel
        from chisurf.plugins._dev.lightpath_simulator.node_types import (
            get_optical_node_factory, get_light_source_factory, 
            get_sample_node_factory, get_forster_node_factory
        )
        
        factory_generator = None
        if node_type.id == "light_source":
            factory_generator = get_light_source_factory(self.db)
        elif node_type.id == "sample":
            factory_generator = get_sample_node_factory(self.db)
        elif node_type.id == "filter":
            factory_generator = get_optical_node_factory(self.db, "transmission", "Filter:")
        elif node_type.id == "splitter":
            factory_generator = get_optical_node_factory(self.db, "transmission", "Splitter:")
        elif node_type.id == "detector":
            factory_generator = get_optical_node_factory(self.db, "quantum_efficiency", "Detector:")
        elif node_type.id == "combiner":
            factory_generator = lambda config: QtWidgets.QWidget() # Stub
        elif node_type.id == "forster_radius":
            factory_generator = get_forster_node_factory(self.db)

        model_config = node_type.default_config.copy()
        
        content_factory = None
        if factory_generator is not None:
             content_factory = lambda config=model_config, f=factory_generator: f(config)
             
        model = NodeModel(
            node_type=node_type.id,
            title=node_type.title,
            inputs=node_type.inputs,
            outputs=node_type.outputs,
            config=model_config,
            content_factory=content_factory
        )
        
        item = NodeGraphicsItem(model, width=node_type.width)
        item.setPos(pos.x(), pos.y())
        self.propagate_graph()
        return item

    def propagate_graph(self, state=None):
        """Propagate spectral signals from sources through the graph."""
        if self._is_propagating:
            return
            
        self._is_propagating = True
        try:
            if not state:
                try:
                    state = self.scene.to_dict()
                except RuntimeError:
                    return

            # Delegate to headless simulator
            self.sim.load_from_dict(state)
            self.sim.propagate()
            
            # Sync back to live nodes
            from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
            id_to_item = {it.model.id: it for it in self.scene.items() if isinstance(it, NodeGraphicsItem)}
            
            for n_id, ns in self.sim._states.items():
                if n_id not in id_to_item: continue
                it = id_to_item[n_id]
                # Update live config for plots
                it.model.config.update({
                    "_input_spectra": ns.input_spectra,
                    "_output_spectra": ns.output_spectra,
                    "_node_char": ns.node_char,
                    "_last_signals": ns.config.get("_last_signals", {}),
                    "_last_results": ns.config.get("_last_results", [])
                })
                if "_update_plot" in it.model.config:
                    it.model.config["_update_plot"]()
        finally:
            self._is_propagating = False

    def calculate_crosstalk(self):
        """Builds result table based on propagated signals."""
        self.propagate_graph()
        row_data = self.sim.get_detector_signals()
        
        headers = ["Laser Source", "Detector Name", "Dye", "Detected Intensity"]
        self.results_table.setRowCount(len(row_data))
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        
        for r_idx, row in enumerate(row_data):
            self.results_table.setItem(r_idx, 0, QtWidgets.QTableWidgetItem(row["laser"]))
            self.results_table.setItem(r_idx, 1, QtWidgets.QTableWidgetItem(row["detector"]))
            self.results_table.setItem(r_idx, 2, QtWidgets.QTableWidgetItem(row["dye"]))
            
            val_item = QtWidgets.QTableWidgetItem(f"{row['intensity']:.4e}")
            val_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.results_table.setItem(r_idx, 3, val_item)

    # Required interface for state_tracker
    def begin_undo(self, text: str) -> None:
        self.state_tracker.begin_action(text)

    def commit_undo(self) -> None:
        self.state_tracker.commit_action()
        self.propagate_graph()

    def _setup_default_path(self):
        """Programmatically builds a standard Laser -> Sample -> Dichroic -> 2 Detectors path."""
        self.scene.clear()
        
        # Try to find some reasonable defaults from the database
        green_dye_id = None
        red_dye_id = None
        dichroic_id = None
        green_filter_id = None
        red_filter_id = None
        
        try:
            with self.db:
                items = self.db.get_probes()
                for it in items:
                    i_id = it['probe_id']
                    i_name = it['chromophore_name']
                    name_lower = i_name.lower().replace("-", " ")
                    if "atto 488" in name_lower and not green_dye_id and self.db.get_spectrum(i_id, "emission"): green_dye_id = i_id
                    if "atto 647n" in name_lower and not red_dye_id and self.db.get_spectrum(i_id, "emission"): red_dye_id = i_id
                    if "561lp" in name_lower and not dichroic_id and self.db.get_spectrum(i_id, "transmission"): dichroic_id = i_id
                    if "bp" in name_lower and "500" in name_lower and not green_filter_id and self.db.get_spectrum(i_id, "transmission"): green_filter_id = i_id
                    if "bp" in name_lower and "650" in name_lower and not red_filter_id and self.db.get_spectrum(i_id, "transmission"): red_filter_id = i_id
        except: pass

        # 1. Light Source
        ls = self._on_add_node_requested("light_source", QtCore.QPointF(50, 200))
        ls_cfg = ls.model.config
        ls_cfg["source_mode"] = "manual"
        ls_cfg["manual_lines"] = "488:1.0, 640:1.0"
        self.scene.addItem(ls)
        
        # 2. Sample
        sample = self._on_add_node_requested("sample", QtCore.QPointF(300, 200))
        sample_cfg = sample.model.config
        dyes = []
        if green_dye_id: dyes.append(green_dye_id)
        if red_dye_id: dyes.append(red_dye_id)
        if dyes: sample_cfg["spectrum_ids"] = dyes
        self.scene.addItem(sample)
        
        # 3. Splitter
        dichroic = self._on_add_node_requested("splitter", QtCore.QPointF(550, 200))
        if dichroic_id: dichroic.model.config["spectrum_id"] = dichroic_id
        self.scene.addItem(dichroic)
        
        # 4. Red Path (Transmitted if > 561nm)
        red_f = self._on_add_node_requested("filter", QtCore.QPointF(800, 100))
        if red_filter_id: red_f.model.config["spectrum_id"] = red_filter_id
        self.scene.addItem(red_f)
        
        red_det = self._on_add_node_requested("detector", QtCore.QPointF(1050, 100))
        red_det.model.config["detector_name"] = "Red Channel"
        self.scene.addItem(red_det)
        
        # 5. Green Path (Reflected if < 561nm)
        green_f = self._on_add_node_requested("filter", QtCore.QPointF(800, 300))
        if green_filter_id: green_f.model.config["spectrum_id"] = green_filter_id
        self.scene.addItem(green_f)
        
        green_det = self._on_add_node_requested("detector", QtCore.QPointF(1050, 300))
        green_det.model.config["detector_name"] = "Green Channel"
        self.scene.addItem(green_det)
        
        # Helper to create edges using port indices
        # Registry: light_source: Out=0, sample: In=0, Out=0
        # splitter: In=0, Trans=0, Refl=1, filter: In=0, Out=0, detector: In=0
        
        from chisurf.gui.widgets.node_editor.edge_item import EdgeGraphicsItem
        
        def connect(src_item, src_port_idx, dst_item, dst_port_idx):
            edge = EdgeGraphicsItem(src_item.port_items[src_port_idx])
            edge.set_end_port(dst_item.port_items[dst_port_idx])
            self.scene.addItem(edge)
            self.scene.register_edge(edge)

        # Wire it up
        connect(ls, 0, sample, 0)
        connect(sample, 1, dichroic, 0)
        
        # Forster Radius connection
        forster = self._on_add_node_requested("forster_radius", QtCore.QPointF(300, 400))
        self.scene.addItem(forster)
        connect(sample, 2, forster, 0) # Sample Dye Data -> Forster Dye Data
        
        # A 561LP dummy transmits red (long pass) and reflects green (short)
        connect(dichroic, 1, red_f, 0) # Splitter Trans -> Red Filter
        connect(red_f, 1, red_det, 0)
        
        connect(dichroic, 2, green_f, 0) # Splitter Refl -> Green Filter
        connect(green_f, 1, green_det, 0)
        
        self.view.centerOn(550, 200)
        self.propagate_graph()
