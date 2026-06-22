import logging
from typing import Dict, Any, Callable, List, Tuple, Union, TYPE_CHECKING
import numpy as np

from chisurf.gui.widgets.node_editor.registry import NodeType, NodeRegistry
from chisurf.gui.widgets.node_editor.model import PortSpec

if TYPE_CHECKING:
    from qtpy import QtWidgets, QtCore, QtGui
    import pyqtgraph as pg

logger = logging.getLogger(__name__)

# Create a local registry for optical nodes to avoid polluting the global math registry
optical_registry = NodeRegistry()

def trigger_simulator_update(widget: 'QtWidgets.QWidget', full_crosstalk: bool = False):
    """Safely triggers the main simulator update by traversing the QGraphicsProxyWidget boundary."""
    from qtpy import QtWidgets
    try:
        proxy = widget.graphicsProxyWidget()
        if proxy and proxy.scene() and proxy.scene().views():
            view = proxy.scene().views()[0]
            p = view.parentWidget()
            while p and not hasattr(p, "propagate_graph"):
                p = p.parentWidget()
            if p:
                if full_crosstalk and hasattr(p, "calculate_crosstalk"):
                    p.calculate_crosstalk()
                else:
                    p.propagate_graph()
    except Exception as e:
        logger.error(f"Failed to trigger simulator update: {e}")

def create_node_container() -> Tuple['QtWidgets.QWidget', 'QtWidgets.QVBoxLayout']:
    """Helper to create a standard styled node content container."""
    from qtpy import QtWidgets, QtCore
    w = QtWidgets.QWidget()
    w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
    w.setStyleSheet("background: transparent; font-size: 10px; color: white;")
    lay = QtWidgets.QVBoxLayout(w)
    m_h, m_v, spacing = 4, 1, 1
    lay.setContentsMargins(m_h, m_v, m_h, m_v)
    lay.setSpacing(spacing)
    return w, lay

def finish_node_container(w: 'QtWidgets.QWidget'):
    """Applies width constraints to prevent proxy overflow, then skins it."""
    from qtpy import QtWidgets
    from chisurf.gui.widgets.node_editor.ui import apply_node_ui_theme
    for child in w.findChildren(QtWidgets.QWidget):
        if isinstance(child, (QtWidgets.QComboBox, QtWidgets.QPushButton, QtWidgets.QLineEdit, QtWidgets.QListWidget, QtWidgets.QTableWidget)):
            child.setSizePolicy(QtWidgets.QSizePolicy.Ignored, child.sizePolicy().verticalPolicy())
    apply_node_ui_theme(w)

def add_spectral_plot(container, layout, config, db):
    """Adds the 3-layer spectral plot section to a node."""
    from qtpy import QtWidgets, QtCore
    import pyqtgraph as pg
    
    plot_w = pg.PlotWidget(container)
    plot_w.setMinimumWidth(50)
    plot_w.setFixedHeight(100)
    plot_w.setBackground(None)
    plot_w.getPlotItem().setMenuEnabled(False)
    plot_w.getPlotItem().setMouseEnabled(x=False, y=False)
    plot_w.hideAxis('left')
    ax = plot_w.getAxis('bottom')
    ax.setPen((200, 200, 200))
    ax.setHeight(20)
    ax.setStyle(tickTextOffset=2)
    
    toggle_btn = QtWidgets.QPushButton("Show / Hide Plot", container)
    toggle_btn.setCheckable(True)
    toggle_btn.setChecked(config.get("show_plot", True))
    toggle_btn.setFixedHeight(16)
    toggle_btn.setStyleSheet("font-size: 8px; color: #888; background: #333; border: 1px solid #444;")
    
    plot_container = QtWidgets.QWidget(container)
    plot_lay = QtWidgets.QVBoxLayout(plot_container)
    plot_lay.setContentsMargins(0, 0, 0, 0)
    plot_lay.addWidget(plot_w)
    
    def update_plot():
        plot_w.clear()
        from chisurf.plugins._dev.lightpath_simulator.crosstalk import WAVELENGTHS
        
        in_spectra_hier = config.get("_input_spectra", {})
        in_spec = None
        
        # Traverse hierarchical input spectra: { port_name: { source_id: data } }
        for port_data in in_spectra_hier.values():
            if not isinstance(port_data, dict): continue
            for data in port_data.values():
                if isinstance(data, np.ndarray):
                    if in_spec is None:
                        in_spec = data.copy()
                    else:
                        in_spec += data
        
        if in_spec is not None and isinstance(in_spec, np.ndarray) and np.any(in_spec > 0):
            norm = np.max(in_spec)
            if norm > 0: in_spec = in_spec / norm
            plot_w.plot(WAVELENGTHS, in_spec, pen=pg.mkPen((100, 100, 100), style=QtCore.Qt.DotLine))
            
        out_spec_dict = config.get("_output_spectra", {})
        out_spec = None
        for port_dict in out_spec_dict.values():
            if not isinstance(port_dict, dict): continue
            for s in port_dict.values():
                if isinstance(s, np.ndarray):
                    if out_spec is None:
                        out_spec = s.copy()
                    else:
                        out_spec += s
                     
        if out_spec is not None and isinstance(out_spec, np.ndarray) and np.any(out_spec > 0):
            norm = np.max(out_spec)
            if norm > 0: out_spec = out_spec / norm
            plot_w.plot(WAVELENGTHS, out_spec, pen=pg.mkPen((255, 255, 255), width=2.0))
            
        # Draw node characteristic last so it overlays the white spectra
        node_char = config.get("_node_char")
        if node_char is not None:
            if isinstance(node_char, tuple): # Sample: (Abs, Em)
                # Normalize sample characteristics independently
                c_abs = node_char[0] / max(np.max(node_char[0]), 1e-12)
                c_em = node_char[1] / max(np.max(node_char[1]), 1e-12)
                plot_w.plot(WAVELENGTHS, c_abs, pen=pg.mkPen((0, 200, 255), width=1.5))
                plot_w.plot(WAVELENGTHS, c_em, pen=pg.mkPen((255, 180, 0), width=1.5))
            else:
                c_y = node_char / max(np.max(node_char), 1e-12)
                plot_w.plot(WAVELENGTHS, c_y, pen=pg.mkPen((0, 255, 0), width=1.2))

    def on_toggle():
        visible = toggle_btn.isChecked()
        plot_container.setVisible(visible)
        config["show_plot"] = visible
        # Hack to trigger node resize
        try:
            proxy = container.graphicsProxyWidget()
            if proxy:
                node = proxy.parentItem()
                if hasattr(node, "_build_path"):
                    node._build_path()
                    node._layout_ports()
                    node.update()
        except: pass

    toggle_btn.toggled.connect(on_toggle)
    plot_container.setVisible(toggle_btn.isChecked())
    config["_update_plot"] = update_plot
    
    layout.addWidget(toggle_btn)
    layout.addWidget(plot_container)
    return plot_w

def get_optical_node_factory(db, spectra_type: str, fallback_label: str) -> Callable:
    """Generic factory for Filter, Splitter, Detector."""
    def factory(config: Dict[str, Any]) -> 'QtWidgets.QWidget':
        from qtpy import QtWidgets, QtCore
        w, lay = create_node_container()
        lbl = QtWidgets.QLabel(fallback_label, w)
        
        # Detector naming functionality
        if "Detector" in fallback_label:
            name_edit = QtWidgets.QLineEdit(w)
            name_edit.setPlaceholderText("Detector Name")
            name_edit.setText(config.get("detector_name", "Detector"))
            
            def on_name_change(txt):
                config["detector_name"] = txt
                try:
                    p = w.parentWidget()
                    while p and not hasattr(p, "calculate_crosstalk"): p = p.parentWidget()
                    if p: p.calculate_crosstalk()
                except: pass
                
            name_edit.textChanged.connect(on_name_change)
            lay.addWidget(name_edit)
            
        combo = QtWidgets.QComboBox(w)
        combo.setEditable(True)
        combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        if combo.completer():
            combo.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            combo.completer().setFilterMode(QtCore.Qt.MatchContains)
        combo.addItem("None", None)
        try:
            with db:
                items = db.get_probes(curated_only=False)
                for it in items:
                    item_id = it['probe_id']
                    name = it['chromophore_name']
                    if db.get_spectrum(item_id, spectra_type) is not None:
                        combo.addItem(name, item_id)
        except: pass

        curr = config.get("spectrum_id")
        if curr:
            idx = combo.findData(curr)
            if idx >= 0: combo.setCurrentIndex(idx)

        def on_change(idx):
            config["spectrum_id"] = combo.itemData(idx)
            trigger_simulator_update(w, full_crosstalk=True)

        combo.currentIndexChanged.connect(on_change)
        lay.addWidget(lbl)
        lay.addWidget(combo)
        add_spectral_plot(w, lay, config, db)
        finish_node_container(w)
        return w
    return factory

def get_sample_node_factory(db) -> Callable:
    """Specialized factory for Samples supporting multiple fluorophores."""
    def factory(config: Dict[str, Any]) -> 'QtWidgets.QWidget':
        from qtpy import QtWidgets, QtCore
        w, lay = create_node_container()
        lay.addWidget(QtWidgets.QLabel("Select Fluorophores:", w))

        filter_edit = QtWidgets.QLineEdit(w)
        filter_edit.setPlaceholderText("Filter name...")
        lay.addWidget(filter_edit)
        
        table_widget = QtWidgets.QTableWidget(w)
        table_widget.setFixedHeight(120)
        table_widget.setColumnCount(3)
        table_widget.setHorizontalHeaderLabels(["Dye", "QY", "EC"])
        
        # Space-optimized styling
        table_widget.setStyleSheet("""
            QTableWidget { background: #1a1a1a; border: 1px solid #444; color: #eee; gridline-color: #333; font-size: 10px; }
            QHeaderView::section { background: #2a2a2a; padding: 2px; border: 1px solid #444; font-size: 8px; color: #999; }
            QTableWidget::item { padding: 1px; }
        """)
        
        hh = table_widget.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        table_widget.setColumnWidth(1, 35)
        table_widget.setColumnWidth(2, 50)
        hh.setDefaultAlignment(QtCore.Qt.AlignCenter)
        
        vh = table_widget.verticalHeader()
        vh.setVisible(False)
        vh.setDefaultSectionSize(18)
        
        table_widget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        
        selected_ids = config.get("spectrum_ids", [])
        if not selected_ids and config.get("spectrum_id"):
            selected_ids = [config["spectrum_id"]]
            
        dye_props = config.get("dye_properties", {})
        
        try:
            with db:
                items = db.get_probes(curated_only=False)
                valid_items = []
                for it in items:
                    item_id = it['probe_id']
                    name = it['chromophore_name']
                    if db.get_spectrum(item_id, "absorption") and db.get_spectrum(item_id, "emission"):
                        valid_items.append((item_id, name))
                        
                table_widget.setRowCount(len(valid_items))
                for row, (item_id, name) in enumerate(valid_items):
                    # Checkbox item
                    name_item = QtWidgets.QTableWidgetItem(name)
                    name_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
                    name_item.setCheckState(QtCore.Qt.Checked if item_id in selected_ids else QtCore.Qt.Unchecked)
                    name_item.setData(QtCore.Qt.UserRole, item_id)
                    table_widget.setItem(row, 0, name_item)
                    
                    # QY and EC Defaults from DB or overrides from config
                    props = db.get_standardized_optical_properties(item_id)
                    qy_str = str(props.get("qy") if props.get("qy") is not None else "1.0")
                    ec_str = str(props.get("ext_coeff") if props.get("ext_coeff") is not None else "1.0")
                    
                    item_str_id = str(item_id)
                    if item_str_id in dye_props:
                        qy_val = dye_props[item_str_id].get("qy", qy_str)
                        ec_val = dye_props[item_str_id].get("ec", ec_str)
                    else:
                        qy_val = qy_str
                        ec_val = ec_str
                        
                    try:
                        qy = float(qy_val)
                        ec = float(ec_val)
                        ec_display = f"{ec:.1e}" if ec >= 10000 else f"{ec:g}"
                        qy_display = f"{qy:g}"
                    except (ValueError, TypeError):
                        qy_display = str(qy_val)
                        ec_display = str(ec_val)
                        
                    table_widget.setItem(row, 1, QtWidgets.QTableWidgetItem(qy_display))
                    table_widget.setItem(row, 2, QtWidgets.QTableWidgetItem(ec_display))
                        
                def filter_table():
                    flt_txt = filter_edit.text().lower()
                    for row in range(table_widget.rowCount()):
                        item = table_widget.item(row, 0)
                        if item:
                            table_widget.setRowHidden(row, flt_txt not in item.text().lower())
                
                filter_edit.textChanged.connect(filter_table)
        except: pass

        def on_change():
            ids = []
            props_dict = {}
            for row in range(table_widget.rowCount()):
                name_item = table_widget.item(row, 0)
                if name_item and name_item.checkState() == QtCore.Qt.Checked:
                    item_id = name_item.data(QtCore.Qt.UserRole)
                    ids.append(item_id)
                    
                if name_item:
                    item_id = name_item.data(QtCore.Qt.UserRole)
                    qy_item = table_widget.item(row, 1)
                    ec_item = table_widget.item(row, 2)
                    try:
                        qy = float(qy_item.text()) if qy_item else 1.0
                        ec = float(ec_item.text()) if ec_item else 1.0
                        props_dict[str(item_id)] = {"qy": qy, "ec": ec}
                    except ValueError:
                        pass
                        
            config["spectrum_ids"] = ids
            config["spectrum_id"] = ids[0] if ids else None
            config["dye_properties"] = props_dict
            trigger_simulator_update(w, full_crosstalk=True)

        table_widget.itemChanged.connect(on_change)
        lay.addWidget(table_widget)
        add_spectral_plot(w, lay, config, db)
        finish_node_container(w)
        return w
    return factory

def get_forster_node_factory(db) -> Callable:
    """Factory for Forster Radius node with results table and parameters."""
    def factory(config: Dict[str, Any]) -> 'QtWidgets.QWidget':
        from qtpy import QtWidgets, QtCore
        w, lay = create_node_container()
        
        # Matrix table for R0 values
        table = QtWidgets.QTableWidget(w)
        table.setFixedHeight(140)
        table.setColumnCount(0)
        table.setRowCount(0)
        
        # High density matrix styling
        table.setStyleSheet("""
            QTableWidget { background: #151515; border: 1px solid #444; color: #fff; gridline-color: #333; font-size: 10px; }
            QHeaderView::section { background: #222; padding: 1px; border: 1px solid #333; font-size: 8px; color: #888; }
            QTableWidget::item { padding: 0px; }
        """)
        
        hh = table.horizontalHeader()
        hh.setVisible(True)
        hh.setMinimumSectionSize(20)
        hh.setDefaultSectionSize(50)
        
        vh = table.verticalHeader()
        vh.setVisible(True)
        vh.setDefaultSectionSize(16)
        vh.setMinimumSectionSize(12)
        
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        
        def update_table():
            results = config.get("_last_results", [])
            if not results:
                table.setRowCount(0)
                table.setColumnCount(0)
                return
            
            donors = sorted(list(set(r["donor"] for r in results)))
            acceptors = sorted(list(set(r["acceptor"] for r in results)))
            
            table.setRowCount(len(donors))
            table.setColumnCount(len(acceptors))
            table.setHorizontalHeaderLabels(acceptors)
            table.setVerticalHeaderLabels(donors)
            
            res_map = {(r["donor"], r["acceptor"]): r["r0"] for r in results}
            
            for r, d in enumerate(donors):
                for c, a in enumerate(acceptors):
                    val = res_map.get((d, a), 0.0)
                    item = QtWidgets.QTableWidgetItem(f"{val:.1f}")
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    table.setItem(r, c, item)
                    
        config["_update_plot"] = update_table # Re-use the plot update hook for table
        
        # Local parameter overrides for kappa2 and n
        grid = QtWidgets.QGridLayout()
        
        k2_spin = QtWidgets.QDoubleSpinBox(w)
        k2_spin.setRange(0, 4)
        k2_spin.setSingleStep(0.1)
        k2_spin.setValue(config.get("kappa2", 0.6667))
        k2_spin.setToolTip("Orientation Factor (kappa^2)")
        
        n_spin = QtWidgets.QDoubleSpinBox(w)
        n_spin.setRange(1.0, 2.0)
        n_spin.setSingleStep(0.01)
        n_spin.setValue(config.get("n", 1.33))
        n_spin.setToolTip("Refractive Index (n)")
        
        def on_param_change():
            config["kappa2"] = k2_spin.value()
            config["n"] = n_spin.value()
            trigger_simulator_update(w)
            
        k2_spin.valueChanged.connect(on_param_change)
        n_spin.valueChanged.connect(on_param_change)
        
        grid.addWidget(QtWidgets.QLabel("kappa²:"), 0, 0)
        grid.addWidget(k2_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("n:"), 1, 0)
        grid.addWidget(n_spin, 1, 1)
        
        lay.addWidget(QtWidgets.QLabel("R₀ [Å] Matrix:"))
        lay.addWidget(table)
        lay.addLayout(grid)
        
        finish_node_container(w)
        return w
    return factory
        
def get_light_source_factory(db) -> Callable:
    """Creates a factory for a light source that can be broad-band or discrete lines."""
    def factory(config: Dict[str, Any]) -> 'QtWidgets.QWidget':
        from qtpy import QtWidgets, QtCore
        w, lay = create_node_container()
        
        mode_combo = QtWidgets.QComboBox(w)
        mode_combo.addItem("Database Spectrum", "database")
        mode_combo.addItem("Laser Lines (Manual)", "manual")
        mode_combo.setMinimumWidth(50)
        
        curr_mode = config.get("source_mode", "database")
        idx = mode_combo.findData(curr_mode)
        if idx >= 0: mode_combo.setCurrentIndex(idx)
        
        db_widget = QtWidgets.QComboBox(w)
        db_widget.setEditable(True)
        db_widget.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        if db_widget.completer():
            db_widget.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            db_widget.completer().setFilterMode(QtCore.Qt.MatchContains)
        db_widget.addItem("None", None)
        try:
            with db:
                items = db.get_probes(curated_only=False)
                for it in items:
                    item_id = it['probe_id']
                    name = it['chromophore_name']
                    if db.get_spectrum(item_id, "emission"):
                        db_widget.addItem(name, item_id)
        except: pass
        
        sid = config.get("spectrum_id")
        if sid:
            idx = db_widget.findData(sid)
            if idx >= 0: db_widget.setCurrentIndex(idx)
            
        manual_edit = QtWidgets.QLineEdit(w)
        manual_edit.setPlaceholderText("e.g. 488:1.0, 561:0.5")
        manual_edit.setText(config.get("manual_lines", "488:1.0, 561:0.5"))
        
        def update_ui():
            mode = mode_combo.currentData()
            db_widget.setVisible(mode == "database")
            manual_edit.setVisible(mode == "manual")
            config["source_mode"] = mode
            trigger_simulator_update(w, full_crosstalk=True)

        mode_combo.currentIndexChanged.connect(update_ui)
        db_widget.currentIndexChanged.connect(lambda i: (config.update({"spectrum_id": db_widget.itemData(i)}), update_ui()))
        manual_edit.textChanged.connect(lambda t: (config.update({"manual_lines": t}), update_ui()))
        
        lay.addWidget(QtWidgets.QLabel("Source Mode:", w))
        lay.addWidget(mode_combo)
        lay.addWidget(db_widget)
        lay.addWidget(manual_edit)
        
        add_spectral_plot(w, lay, config, db)
        update_ui()
        finish_node_container(w)
        return w
    return factory

def build_optical_registry():
    """Populates the optical registry if it's empty."""
    if len(optical_registry.all_types()) > 0:
        return
    
    optical_registry.register(NodeType(
        id="light_source",
        title="Light Source",
        inputs=[],
        outputs=[PortSpec("Light", True)],
        category="Optical",
        default_config={
            "source_mode": "manual",
            "spectrum_id": None,
            "manual_lines": "488:1.0, 640:1.0"
        },
        width=240
    ))
    
    optical_registry.register(NodeType(
        id="sample",
        title="Sample / Fluorophore",
        inputs=[PortSpec("In", False, port_type="spectral")],
        outputs=[
            PortSpec("Out", True, port_type="spectral"),
            PortSpec("Dye Data", True, port_type="dye_data")
        ],
        category="Optical",
        default_config={"spectrum_ids": [], "spectrum_id": None},
        width=240
    ))
    
    optical_registry.register(NodeType(
        id="filter",
        title="Filter",
        inputs=[PortSpec("In", False, port_type="spectral")],
        outputs=[PortSpec("Out", True, port_type="spectral")],
        category="Optical",
        default_config={"spectrum_id": None},
        width=240
    ))
    
    optical_registry.register(NodeType(
        id="splitter",
        title="Splitter (Dichroic/Pol)",
        inputs=[PortSpec("In", False, port_type="spectral")],
        outputs=[
            PortSpec("Transmission", True, port_type="spectral"), 
            PortSpec("Reflection", True, port_type="spectral")
        ],
        category="Optical",
        default_config={"spectrum_id": None},
        width=240
    ))
    
    optical_registry.register(NodeType(
        id="detector",
        title="Detector",
        inputs=[PortSpec("In", False, port_type="spectral")],
        outputs=[],
        category="Optical",
        default_config={"spectrum_id": None},
        width=240
    ))
 
    optical_registry.register(NodeType(
        id="combiner",
        title="Combiner",
        inputs=[
            PortSpec("Path 1", False, port_type="spectral"), 
            PortSpec("Path 2", False, port_type="spectral")
        ],
        outputs=[PortSpec("Out", True, port_type="spectral")],
        category="Optical",
        default_config={},
        width=240
    ))

    optical_registry.register(NodeType(
        id="forster_radius",
        title="Förster Radius",
        inputs=[
            PortSpec("Dye Data", False, port_type="dye_data"),
            PortSpec("kappa2", False, port_type="number"),
            PortSpec("n", False, port_type="number")
        ],
        outputs=[],
        category="Analysis",
        default_config={
            "kappa2": 0.6667,
            "n": 1.33,
            "_last_results": []
        },
        width=240
    ))
