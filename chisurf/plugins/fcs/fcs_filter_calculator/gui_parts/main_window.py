from __future__ import annotations

import pathlib
from typing import List, Dict

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import chisurf as cs
from ..api import compute_filters, FilterResult
from .widgets import SpeciesListWidget
from .data_loading import load_vector

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c

@persist_plugin_state("fcs_filter_calculator")
class FcsFilterCalculatorWidget(QtWidgets.QWidget):
    """A modular implementation of the Filtered FCS: Lifetime Filter Calculator."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Filtered FCS: Lifetime Filter Calculator")
        self.resize(1000, 600)
        self.setMaximumHeight(600)  # Prevent weird vertical scaling

        self._total_paths: List[pathlib.Path] = []
        self._result: FilterResult | None = None
        self._result_anisotropy = None  # For single-detector Anisotropy results
        self._result_multi_anisotropy = None  # For multi-detector Anisotropy results (list)
        self._result_multi_detector = None  # For multi-detector stacked results
        
        # Detector Wizard Support
        from .data_loading import HAS_DETECTOR_WIZARD, load_detector_setups
        self._has_detector_wizard = HAS_DETECTOR_WIZARD
        self._load_detector_setups = load_detector_setups
        self._detector_settings = None
        
        # Cache for loaded decay data to avoid redundant file loading
        self._decay_cache = {}  # Key: (tuple(paths), tuple(chs)), Value: loaded_vectors
        self._routing_cache = {}  # Key: file_path, Value: dict of {routing_ch: histogram}
        self._cache_state = None  # Track state for cache invalidation

        self._setup_ui()
        self.tabs.setCurrentIndex(1)  # Start on Filter Computation tab
        self._refresh_detector_checkboxes()
        self._update_status()

    def _setup_ui(self) -> None:
        main_vbox = QtWidgets.QVBoxLayout(self)
        main_vbox.setContentsMargins(2, 2, 2, 2)
        main_vbox.setSpacing(2)

        self.tabs = QtWidgets.QTabWidget()
        main_vbox.addWidget(self.tabs)

        # Tab 1: Detector Setup (Swapped to first)
        self.setup_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.setup_tab, "Detector Setup")
        setup_layout = QtWidgets.QVBoxLayout(self.setup_tab)
        
        if self._has_detector_wizard:
            from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage
            self.detector_wizard_page = DetectorWizardPage()
            setup_layout.addWidget(self.detector_wizard_page)
            
            # Connect wizard signals to update checkboxes in Filter tab
            self._connect_setup_signals()
        else:
            setup_layout.addWidget(QtWidgets.QLabel("Detector wizard not available."))

        # Tab 2: Filters (Swapped to second)
        self.filter_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.filter_tab, "Filter Computation")
        
        filter_layout = QtWidgets.QHBoxLayout(self.filter_tab)

        # Left Sidebar (in Filter Tab)
        sidebar = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(2, 2, 2, 2)
        sidebar_layout.setSpacing(2)
        sidebar.setFixedWidth(280)

        # Data loading group
        data_group = QtWidgets.QGroupBox("Data Sources")
        data_layout = QtWidgets.QVBoxLayout(data_group)
        data_layout.setContentsMargins(3, 3, 3, 3)
        data_layout.setSpacing(2)
        
        # Anisotropy Mode checkbox
        self.anisotropy_mode_cb = QtWidgets.QCheckBox("Anisotropy Mode (stack parallel/perpendicular)")
        self.anisotropy_mode_cb.setToolTip("When checked, uses detector routing channels alternating as par/perp (e.g., ch 8,0 = par,perp)")
        self.anisotropy_mode_cb.stateChanged.connect(self._on_anisotropy_mode_changed)
        data_layout.addWidget(self.anisotropy_mode_cb)
        
        # Detector Selection (Always visible in Filter Tab)
        from .widgets import DetectorSelectionWidget
        self.detector_selection = DetectorSelectionWidget()
        self.detector_selection.selectionChanged.connect(self._on_detector_selection_changed)
        data_layout.addWidget(self.detector_selection)

        # Total Decay Group
        total_group = QtWidgets.QGroupBox("Total Decay")
        total_vbox = QtWidgets.QVBoxLayout(total_group)
        total_vbox.setContentsMargins(3, 3, 3, 3)
        total_vbox.setSpacing(2)
        self.le_total = QtWidgets.QLineEdit()
        self.le_total.setReadOnly(True)
        self.le_total.setPlaceholderText("Drop file here...")
        self.le_total.setToolTip("Path to the total decay histogram")
        total_vbox.addWidget(self.le_total)
        data_layout.addWidget(total_group)
        sidebar_layout.addWidget(data_group)

        # Species Patterns Group
        species_group = QtWidgets.QGroupBox("Species Decay Patterns")
        species_vbox = QtWidgets.QVBoxLayout(species_group)
        species_vbox.setContentsMargins(3, 3, 3, 3)
        species_vbox.setSpacing(2)
        self.lw_species = SpeciesListWidget()
        self.lw_species.filesChanged.connect(self._on_files_changed)  # Files added/removed
        self.lw_species.checkStateChanged.connect(self._on_data_changed)  # Checkboxes toggled
        species_vbox.addWidget(self.lw_species)
        
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(2)
        self.btn_add_species = QtWidgets.QPushButton("Add...")
        self.btn_add_species.clicked.connect(self._add_species_dialog)
        self.btn_remove_species = QtWidgets.QPushButton("Remove")
        self.btn_remove_species.clicked.connect(self._remove_selected_species)
        btn_layout.addWidget(self.btn_add_species)
        btn_layout.addWidget(self.btn_remove_species)
        species_vbox.addLayout(btn_layout)
        sidebar_layout.addWidget(species_group)

        # Actions
        sidebar_layout.addStretch()

        self.btn_save_project = QtWidgets.QPushButton("Save Project...")
        self.btn_save_project.clicked.connect(self._on_save_project)
        sidebar_layout.addWidget(self.btn_save_project)

        self.btn_load_project = QtWidgets.QPushButton("Load Project...")
        self.btn_load_project.clicked.connect(self._on_load_project)
        sidebar_layout.addWidget(self.btn_load_project)

        self.btn_export = QtWidgets.QPushButton("Export Results...")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(False)
        sidebar_layout.addWidget(self.btn_export)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        sidebar_layout.addWidget(self.status_label)

        filter_layout.addWidget(sidebar)

        # Right Plot Area
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        plot_layout.setSpacing(2)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.setHandleWidth(3)
        self.plot_filters = pg.PlotWidget(title="Lifetime Filters")
        self.plot_filters.setLabel("bottom", "TAC bin")
        self.plot_filters.setLabel("left", "Filter value")
        self.plot_filters.addLegend()
        splitter.addWidget(self.plot_filters)
        self.plot_recon = pg.PlotWidget(title="Reconstruction Quality")
        self.plot_recon.setLabel("bottom", "TAC bin")
        self.plot_recon.setLabel("left", "Counts")
        self.plot_recon.setLogMode(y=True)
        self.plot_recon.addLegend()
        splitter.addWidget(self.plot_recon)
        self.plot_residuals = pg.PlotWidget(title="Weighted Residuals")
        self.plot_residuals.setLabel("bottom", "TAC bin")
        self.plot_residuals.setLabel("left", "Residuals (σ)")
        splitter.addWidget(self.plot_residuals)
        plot_layout.addWidget(splitter)
        filter_layout.addWidget(plot_container, 1)

        # Set up drag and drop for the whole widget
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        
        paths = [pathlib.Path(url.toLocalFile()) for url in urls]
        
        # If dropping on total decay group area
        if self.le_total.geometry().translated(self.le_total.parentWidget().mapTo(self, QtCore.QPoint(0,0))).contains(event.pos()):
            self._set_total_paths(paths)
        else:
            # Default to adding as species
            self.lw_species.add_pattern(paths)

    def _set_total_paths(self, paths: List[pathlib.Path]) -> None:
        self._total_paths = paths
        if not paths:
            self.le_total.setText("")
            self.le_total.setToolTip("")
        else:
            name = paths[0].name
            if len(paths) > 1:
                name += f" (+{len(paths)-1} files)"
            self.le_total.setText(name)
            self.le_total.setToolTip("\n".join([str(p.absolute()) for p in paths]))
        # Invalidate cache when total paths change
        self._invalidate_cache()
        self._on_data_changed()

    def _add_species_dialog(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select species decay histograms", "", 
            "All Decay Files (*.txt *.dat *.csv *.bst *.tttr);;Text Files (*.txt *.dat *.csv);;Burst Files (*.bst);;TTTR Files (*.tttr);;All files (*)"
        )
        if paths:
            self.lw_species.add_pattern([pathlib.Path(p) for p in paths])

    def _remove_selected_species(self) -> None:
        for item in self.lw_species.selectedItems():
            self.lw_species.takeItem(self.lw_species.row(item))
        # Removing species files requires cache invalidation
        self._invalidate_cache()
        self._on_data_changed()

    def _on_files_changed(self) -> None:
        """Called when species files are added/removed (not just checkbox toggled)."""
        # Invalidate cache when files change
        self._invalidate_cache()
        self._on_data_changed()

    def _on_anisotropy_mode_changed(self) -> None:
        """Called when Anisotropy mode checkbox is toggled."""
        # Clear all results when switching modes
        self._result = None
        self._result_anisotropy = None
        self._result_multi_anisotropy = None
        self._result_multi_detector = None
        # Don't invalidate cache - just recompute filters from cached decays
        self._on_data_changed()

    def _on_detector_selection_changed(self) -> None:
        """Called when detector selection checkboxes are toggled."""
        # Don't invalidate cache - just recompute filters from cached decays
        self._on_data_changed()

    def _on_data_changed(self) -> None:
        self._result = None
        self.btn_export.setEnabled(False)
        self._update_status()
        # Note: Do NOT invalidate cache here - species checkbox changes don't require reloading files
        # Cache is only invalidated when files or detector selection actually change
        # Trigger auto-compute
        try:
            self._compute_filters()
        except Exception as e:
            import traceback
            cs.logging.error(f"Error in auto-compute: {e}\n{traceback.format_exc()}")
            self._update_status(f"Error: {e}")

    def _update_status(self, msg: str | None = None) -> None:
        if msg is None:
            n_species = self.lw_species.count()
            if not self._total_paths:
                msg = "Missing total decay histogram."
            elif n_species == 0:
                msg = "Add species decay patterns."
            else:
                msg = f"Ready to compute ({n_species} species)."
        self.status_label.setText(msg)

    def _compute_filters(self) -> None:
        if not self._total_paths or self.lw_species.count() == 0:
            return

        try:
            # Detector filtering logic
            chs = None
            if self.detector_selection.checkboxes:
                chs = self.detector_selection.get_selected()
            
            # Check if Anisotropy mode is enabled (takes priority over multi-detector)
            anisotropy_mode = self.anisotropy_mode_cb.isChecked()
            if anisotropy_mode and chs and len(chs) >= 1:
                self._compute_filters_anisotropy(chs)
                return
            
            # Check if multiple detectors are selected (multi-detector stacking mode)
            # Only if NOT in anisotropy mode
            if chs and len(chs) > 1:
                self._compute_filters_multi_detector(chs)
                return
            
            # Standard single-channel mode
            # Load and sum total decay files (with caching)
            total_data = self._load_and_sum_vectors(self._total_paths, chs)

            species_data = []
            species_names = []
            species_patterns = []
            for i in range(self.lw_species.count()):
                item = self.lw_species.item(i)
                if item.checkState() != QtCore.Qt.Checked:
                    continue
                
                paths = item.data(QtCore.Qt.UserRole)
                species_patterns.append([str(p.absolute()) for p in paths])
                
                # Load and sum pattern (with caching)
                pattern_sum = self._load_and_sum_vectors(paths, chs)
                
                species_data.append(pattern_sum)
                species_names.append(item.text())

            # Validation against total - use the total size as the master size
            total_size = total_data.size
            for i in range(len(species_data)):
                sd = species_data[i]
                if sd.size != total_size:
                    # Adjust species pattern to match total size (truncate or pad)
                    if sd.size > total_size:
                        species_data[i] = sd[:total_size]
                    else:
                        padded = np.zeros(total_size, dtype=np.float64)
                        padded[:sd.size] = sd
                        species_data[i] = padded

            self._result = compute_filters(
                total_data, 
                species_data, 
                total_path=[str(p.absolute()) for p in self._total_paths],
                species_patterns=species_patterns
            )
            self._result_anisotropy = None  # Clear Anisotropy result
            self._result_multi_detector = None  # Clear multi-detector result
            self._update_plots()
            self.btn_export.setEnabled(True)
            self._update_status("Filters computed successfully.")

        except Exception as e:
            import traceback
            cs.logging.error(f"Computation error: {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Computation Error", str(e))
            self._update_status(f"Error: {e}")

    def _invalidate_cache(self) -> None:
        """Invalidate decay cache when inputs change."""
        self._decay_cache.clear()
        self._routing_cache.clear()
        self._cache_state = None

    def _get_cache_key(self, paths: List[pathlib.Path], chs: List[str] | None) -> tuple:
        """Generate cache key from file paths and detector channels."""
        path_tuple = tuple(str(p.absolute()) for p in paths)
        ch_tuple = tuple(sorted(chs)) if chs else ()
        return (path_tuple, ch_tuple)

    def _load_routing_channels(self, path: pathlib.Path) -> Dict[int, np.ndarray]:
        """Load and cache all routing channel histograms for a TTTR file.
        
        Returns dict mapping routing_channel_number -> histogram.
        """
        path_str = str(path.absolute())
        
        if path_str in self._routing_cache:
            return self._routing_cache[path_str]
        
        # Load file and extract all routing channels
        ext = path.suffix.lower()
        routing_histograms = {}
        
        if ext in ('.spc', '.ptu', '.ht3', '.tttr'):
            import tttrlib
            try:
                # Load TTTR data once
                if ext == '.spc':
                    try:
                        data = tttrlib.TTTR(str(path), 'SPC-130')
                    except:
                        data = tttrlib.TTTR(str(path))
                else:
                    data = tttrlib.TTTR(str(path))
                
                header = data.get_header()
                try:
                    n_tac = header.number_of_micro_time_channels
                except AttributeError:
                    try:
                        n_tac = header['number_of_micro_time_channels']
                    except (KeyError, TypeError):
                        n_tac = 4096
                
                microtimes = data.micro_times
                routing = data.routing_channels
                
                # Extract histogram for each routing channel
                unique_routing = np.unique(routing)
                for rch in unique_routing:
                    mask = (routing == rch) & (microtimes >= 0) & (microtimes < n_tac)
                    hist = np.zeros(n_tac, dtype=np.float64)
                    np.add.at(hist, microtimes[mask], 1)
                    routing_histograms[int(rch)] = hist
                
            except Exception as e:
                cs.logging.warning(f"Error loading routing channels from {path.name}: {e}")
        
        elif ext == '.bst':
            # For BST files, load the underlying TTTR and extract bursts
            from .data_loading import parse_bst_file
            tttr_path, ranges = parse_bst_file(path)
            
            if tttr_path and ranges:
                import tttrlib
                try:
                    data = tttrlib.TTTR(str(tttr_path))
                    header = data.get_header()
                    try:
                        n_tac = header.number_of_micro_time_channels
                    except AttributeError:
                        try:
                            n_tac = header['number_of_micro_time_channels']
                        except (KeyError, TypeError):
                            n_tac = 4096
                    
                    microtimes = data.micro_times
                    routing = data.routing_channels
                    
                    # Extract histograms for each routing channel from burst regions
                    unique_routing = np.unique(routing)
                    for rch in unique_routing:
                        hist = np.zeros(n_tac, dtype=np.float64)
                        for start, end in ranges:
                            if start < len(microtimes) and end <= len(microtimes):
                                burst_mt = microtimes[start:end]
                                burst_rt = routing[start:end]
                                mask = (burst_rt == rch) & (burst_mt >= 0) & (burst_mt < n_tac)
                                np.add.at(hist, burst_mt[mask], 1)
                        routing_histograms[int(rch)] = hist
                        
                except Exception as e:
                    cs.logging.warning(f"Error loading BST routing channels from {path.name}: {e}")
        
        # Cache the routing histograms
        self._routing_cache[path_str] = routing_histograms
        return routing_histograms

    def _load_and_sum_vectors(self, paths: List[pathlib.Path], chs: List[str] | None) -> np.ndarray:
        """Load and sum vectors with routing channel caching."""
        cache_key = self._get_cache_key(paths, chs)
        
        if cache_key in self._decay_cache:
            return self._decay_cache[cache_key].copy()
        
        # Determine which routing channels to use
        routing_channels = set()
        if chs:
            for ch_name in chs:
                if ch_name.startswith("routing_"):
                    try:
                        routing_channels.add(int(ch_name.split("_")[1]))
                    except:
                        pass
                elif self._detector_settings:
                    # Map detector name to routing channels
                    det_config = self._detector_settings.get("detectors", {}).get(ch_name, {})
                    det_chs = det_config.get("chs", [])
                    routing_channels.update(det_chs)
        
        # Load and combine from routing channel cache
        max_size = 0
        all_histograms = []
        
        for path in paths:
            ext = path.suffix.lower()
            
            if ext in ('.spc', '.ptu', '.ht3', '.tttr', '.bst'):
                # Use routing channel cache
                routing_hists = self._load_routing_channels(path)
                
                if routing_channels:
                    # Combine specified routing channels
                    combined = None
                    for rch in routing_channels:
                        if rch in routing_hists:
                            if combined is None:
                                combined = routing_hists[rch].copy()
                            else:
                                combined += routing_hists[rch]
                    if combined is not None:
                        all_histograms.append(combined)
                        max_size = max(max_size, combined.size)
                else:
                    # Use all routing channels
                    combined = None
                    for hist in routing_hists.values():
                        if combined is None:
                            combined = hist.copy()
                        else:
                            combined += hist
                    if combined is not None:
                        all_histograms.append(combined)
                        max_size = max(max_size, combined.size)
            else:
                # For text files, use old method
                vec = load_vector(path, chs=chs, detector_settings=self._detector_settings)
                all_histograms.append(vec)
                max_size = max(max_size, vec.size)
        
        # Sum all histograms
        if max_size == 0 or not all_histograms:
            # No valid data - return empty array
            cs.logging.warning(f"No valid histogram data loaded for paths: {[p.name for p in paths]}")
            return np.zeros(4096, dtype=np.float64)
        
        summed = np.zeros(max_size, dtype=np.float64)
        for hist in all_histograms:
            if hist.size < max_size:
                padded = np.zeros(max_size, dtype=np.float64)
                padded[:hist.size] = hist
                summed += padded
            else:
                summed += hist
        
        # Cache the result
        self._decay_cache[cache_key] = summed.copy()
        return summed

    def _compute_filters_anisotropy(self, chs: List[str]) -> None:
        """Compute Anisotropy filters for parallel and perpendicular channels.
        
        Uses detector wizard routing channel logic: channels alternate as par, perp, par, perp.
        For a detector with routing channels [8, 0], ch 8 = parallel, ch 0 = perpendicular.
        With multiple detectors - computes separate par/perp for each detector and stacks them.
        """
        from ..api import compute_filters_mfd
        
        try:
            # Check if multiple detectors selected - compute separately for each
            if len(chs) > 1:
                self._compute_filters_multi_anisotropy(chs)
                return
            
            # Single detector anisotropy mode
            # Extract routing channels from the selected detector
            routing_par = []
            routing_perp = []
            
            if self._detector_settings and chs:
                det_name = chs[0]
                det_config = self._detector_settings.get("detectors", {}).get(det_name, {})
                routing_chs = det_config.get("chs", [])
                
                if len(routing_chs) >= 2:
                    # Alternating pattern: index 0, 2, 4... = parallel; index 1, 3, 5... = perpendicular
                    for i, ch in enumerate(routing_chs):
                        if i % 2 == 0:
                            routing_par.append(ch)
                        else:
                            routing_perp.append(ch)
                else:
                    raise ValueError(f"Detector '{det_name}' must have at least 2 routing channels for Anisotropy mode")
            else:
                raise ValueError("Anisotropy mode requires detector settings with routing channels")
            
            # Create routing channel names for load_vector
            ch_par = [f"routing_{ch}" for ch in routing_par]
            ch_perp = [f"routing_{ch}" for ch in routing_perp]
            
            # Load total decay for each channel separately (with caching)
            total_par = self._load_and_sum_vectors(self._total_paths, ch_par)
            total_perp = self._load_and_sum_vectors(self._total_paths, ch_perp)
            
            # Ensure both have same size
            max_size = max(total_par.size, total_perp.size)
            if total_par.size < max_size:
                padded = np.zeros(max_size, dtype=np.float64)
                padded[:total_par.size] = total_par
                total_par = padded
            if total_perp.size < max_size:
                padded = np.zeros(max_size, dtype=np.float64)
                padded[:total_perp.size] = total_perp
                total_perp = padded
            
            # Load species patterns for each channel
            species_par = []
            species_perp = []
            species_patterns = []
            
            for i in range(self.lw_species.count()):
                item = self.lw_species.item(i)
                if item.checkState() != QtCore.Qt.Checked:
                    continue
                
                paths = item.data(QtCore.Qt.UserRole)
                species_patterns.append([str(p.absolute()) for p in paths])
                
                # Load for parallel channel (with caching)
                pattern_par = self._load_and_sum_vectors(paths, ch_par)
                if pattern_par.size < max_size:
                    padded = np.zeros(max_size, dtype=np.float64)
                    padded[:pattern_par.size] = pattern_par
                    pattern_par = padded
                species_par.append(pattern_par)
                
                # Load for perpendicular channel (with caching)
                pattern_perp = self._load_and_sum_vectors(paths, ch_perp)
                if pattern_perp.size < max_size:
                    padded = np.zeros(max_size, dtype=np.float64)
                    padded[:pattern_perp.size] = pattern_perp
                    pattern_perp = padded
                species_perp.append(pattern_perp)
            
            # Compute Anisotropy filters
            metadata = {
                "detector": chs[0],
                "routing_par": routing_par,
                "routing_perp": routing_perp,
            }
            self._result_anisotropy = compute_filters_mfd(
                total_par, total_perp,
                species_par, species_perp,
                metadata=metadata
            )
            self._result = None  # Clear single-channel result
            self._result_multi_detector = None  # Clear multi-detector result
            self._result_multi_anisotropy = None  # Clear multi-anisotropy result
            self._update_plots()
            self.btn_export.setEnabled(True)
            self._update_status(f"Anisotropy filters computed successfully ({chs[0]}: ch {routing_par} || ch {routing_perp}).")
            
        except Exception as e:
            import traceback
            cs.logging.error(f"Anisotropy computation error: {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Anisotropy Computation Error", str(e))
            self._update_status(f"Anisotropy Error: {e}")

    def _compute_filters_multi_anisotropy(self, chs: List[str]) -> None:
        """Compute Anisotropy filters separately for each detector and stack them.
        
        Each detector gets its own par/perp computation.
        """
        from ..api import compute_filters_mfd
        
        try:
            anisotropy_results = []
            
            for det_name in chs:
                # Extract routing channels for this detector
                if not self._detector_settings:
                    raise ValueError("Anisotropy mode requires detector settings")
                
                det_config = self._detector_settings.get("detectors", {}).get(det_name, {})
                routing_chs = det_config.get("chs", [])
                
                if len(routing_chs) < 2:
                    cs.logging.warning(f"Detector '{det_name}' has <2 routing channels, skipping")
                    continue
                
                # Split into par/perp
                routing_par = []
                routing_perp = []
                for i, ch in enumerate(routing_chs):
                    if i % 2 == 0:
                        routing_par.append(ch)
                    else:
                        routing_perp.append(ch)
                
                ch_par = [f"routing_{ch}" for ch in routing_par]
                ch_perp = [f"routing_{ch}" for ch in routing_perp]
                
                # Load total decay for this detector's par/perp
                total_par = self._load_and_sum_vectors(self._total_paths, ch_par)
                total_perp = self._load_and_sum_vectors(self._total_paths, ch_perp)
                
                # Ensure same size
                max_size = max(total_par.size, total_perp.size)
                if total_par.size < max_size:
                    padded = np.zeros(max_size, dtype=np.float64)
                    padded[:total_par.size] = total_par
                    total_par = padded
                if total_perp.size < max_size:
                    padded = np.zeros(max_size, dtype=np.float64)
                    padded[:total_perp.size] = total_perp
                    total_perp = padded
                
                # Load species patterns
                species_par = []
                species_perp = []
                species_patterns = []
                
                for i in range(self.lw_species.count()):
                    item = self.lw_species.item(i)
                    if item.checkState() != QtCore.Qt.Checked:
                        continue
                    
                    paths = item.data(QtCore.Qt.UserRole)
                    species_patterns.append([str(p.absolute()) for p in paths])
                    
                    pattern_par = self._load_and_sum_vectors(paths, ch_par)
                    if pattern_par.size < max_size:
                        padded = np.zeros(max_size, dtype=np.float64)
                        padded[:pattern_par.size] = pattern_par
                        pattern_par = padded
                    species_par.append(pattern_par)
                    
                    pattern_perp = self._load_and_sum_vectors(paths, ch_perp)
                    if pattern_perp.size < max_size:
                        padded = np.zeros(max_size, dtype=np.float64)
                        padded[:pattern_perp.size] = pattern_perp
                        pattern_perp = padded
                    species_perp.append(pattern_perp)
                
                # Compute anisotropy for this detector
                metadata = {
                    "detector": det_name,
                    "routing_par": routing_par,
                    "routing_perp": routing_perp,
                }
                result = compute_filters_mfd(
                    total_par, total_perp,
                    species_par, species_perp,
                    metadata=metadata
                )
                
                anisotropy_results.append({
                    'detector': det_name,
                    'result': result
                })
            
            if not anisotropy_results:
                raise ValueError("No valid detectors for Anisotropy mode")
            
            # Store multi-anisotropy results
            self._result_multi_anisotropy = anisotropy_results
            self._result = None
            self._result_anisotropy = None
            self._result_multi_detector = None
            self._update_plots()
            self.btn_export.setEnabled(True)
            detector_names = ", ".join([ar['detector'] for ar in anisotropy_results])
            self._update_status(f"Multi-detector Anisotropy filters computed successfully ({detector_names}).")
            
        except Exception as e:
            import traceback
            cs.logging.error(f"Multi-Anisotropy computation error: {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Multi-Anisotropy Computation Error", str(e))
            self._update_status(f"Multi-Anisotropy Error: {e}")

    def _compute_filters_multi_detector(self, chs: List[str]) -> None:
        """Compute filters for multiple detectors separately and stack them.
        
        Each detector's photons are isolated - red photons don't contribute to green decay.
        """
        try:
            # Store results for each detector
            detector_results = []
            
            for det_name in chs:
                # Load total decay for this detector only (with caching)
                ch_list = [det_name]
                total_data = self._load_and_sum_vectors(self._total_paths, ch_list)
                
                # Load species patterns for this detector only
                species_data = []
                species_names = []
                species_patterns = []
                for i in range(self.lw_species.count()):
                    item = self.lw_species.item(i)
                    if item.checkState() != QtCore.Qt.Checked:
                        continue
                    
                    paths = item.data(QtCore.Qt.UserRole)
                    species_patterns.append([str(p.absolute()) for p in paths])
                    
                    # Load and sum pattern (with caching)
                    pattern_sum = self._load_and_sum_vectors(paths, ch_list)
                    
                    species_data.append(pattern_sum)
                    species_names.append(item.text())
                
                # Validation against total
                total_size = total_data.size
                for i in range(len(species_data)):
                    sd = species_data[i]
                    if sd.size != total_size:
                        if sd.size > total_size:
                            species_data[i] = sd[:total_size]
                        else:
                            padded = np.zeros(total_size, dtype=np.float64)
                            padded[:sd.size] = sd
                            species_data[i] = padded
                
                # Compute filters for this detector
                result = compute_filters(
                    total_data,
                    species_data,
                    total_path=[str(p.absolute()) for p in self._total_paths],
                    species_patterns=species_patterns
                )
                detector_results.append({
                    'detector': det_name,
                    'result': result
                })
            
            # Store multi-detector results
            self._result_multi_detector = detector_results
            self._result = None  # Clear single-channel result
            self._result_anisotropy = None  # Clear anisotropy result
            self._update_plots()
            self.btn_export.setEnabled(True)
            detector_names = ", ".join(chs)
            self._update_status(f"Multi-detector filters computed successfully ({detector_names}).")
            
        except Exception as e:
            import traceback
            cs.logging.error(f"Multi-detector computation error: {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Multi-Detector Computation Error", str(e))
            self._update_status(f"Multi-Detector Error: {e}")

    def _connect_setup_signals(self):
        if not hasattr(self, 'detector_wizard_page') or not self.detector_wizard_page: return
        for widget in self.detector_wizard_page.findChildren(QtWidgets.QWidget):
            for signal_name in ['textChanged', 'currentIndexChanged', 'stateChanged', 'valueChanged', 'editingFinished']:
                if hasattr(widget, signal_name):
                    try:
                        getattr(widget, signal_name).connect(self._schedule_setup_refresh)
                    except Exception: pass

    def _schedule_setup_refresh(self):
        # Refresh checkboxes when wizard settings change
        self._detector_settings = self.detector_wizard_page.get_settings()
        self._refresh_detector_checkboxes()

    def _refresh_detector_checkboxes(self):
        detector_names = []
        if self._detector_settings:
            detector_names = list(self._detector_settings.get("detectors", {}).keys())
        elif self._has_detector_wizard:
            try:
                setups_data = self._load_detector_setups()
                if setups_data:
                    last_used = setups_data.get('last_used')
                    if last_used and last_used in setups_data.get('setups', {}):
                        self._detector_settings = setups_data['setups'][last_used]
                        detector_names = list(self._detector_settings.get("detectors", {}).keys())
            except Exception: pass
        
        self.detector_selection.refresh(detector_names)

    def _on_save_project(self) -> None:
        if self._result is None:
            # We can still save the UI state if paths are present
            if not self._total_paths and self.lw_species.count() == 0:
                QtWidgets.QMessageBox.warning(self, "Empty Project", "No data loaded to save.")
                return
            
            # Filters are automatically computed, so we can proceed with saving
            # The computation will happen automatically when data is loaded

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save fFCS Project", "fcs_filter.json", "JSON (*.json)"
        )
        if path:
            import json
            
            if self._result is not None:
                # Save result with additional UI state
                self._result.to_json(path, indent=2)
                
                # Also save detector selection state and anisotropy mode
                with open(path, 'r') as f:
                    project_data = json.load(f)
                
                # Add UI state
                project_data['ui_state'] = {
                    'selected_detectors': self.detector_selection.get_selected(),
                    'anisotropy_mode': self.anisotropy_mode_cb.isChecked()
                }
                
                with open(path, 'w') as f:
                    json.dump(project_data, f, indent=2)
            else:
                # Save only UI state when no computation results exist
                project_data = {
                    'total_path': [str(p.absolute()) for p in self._total_paths],
                    'species_patterns': [],
                    'ui_state': {
                        'selected_detectors': self.detector_selection.get_selected(),
                        'anisotropy_mode': self.anisotropy_mode_cb.isChecked()
                    }
                }
                
                # Add species patterns
                for i in range(self.lw_species.count()):
                    item = self.lw_species.item(i)
                    if item.checkState() == QtCore.Qt.Checked:
                        paths = item.data(QtCore.Qt.UserRole)
                        project_data['species_patterns'].append({
                            'name': item.text(),
                            'paths': [str(p.absolute()) for p in paths]
                        })
                
                with open(path, 'w') as f:
                    json.dump(project_data, f, indent=2)
            
            self._update_status(f"Project saved to {pathlib.Path(path).name}")

    def _on_load_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load fFCS Project", "", "JSON (*.json)"
        )
        if not path:
            return

        try:
            res = FilterResult.from_json(path)
            self._result = res
            
            # Restore UI state
            if res.total_path:
                # If it's an old single-string path, wrap in list
                t_paths = res.total_path if isinstance(res.total_path, list) else [res.total_path]
                path_objs = [pathlib.Path(p) for p in t_paths]
                
                # Check for existence
                valid_paths = []
                for p in path_objs:
                    if p.exists():
                        valid_paths.append(p)
                    else:
                        cs.logging.warning(f"Total decay file not found: {p}")
                
                self._set_total_paths(valid_paths)
            
            self.lw_species.clear()
            if res.species_patterns:
                for pattern_files in res.species_patterns:
                    paths = [pathlib.Path(p) for p in pattern_files]
                    # Check if all files in pattern exist
                    missing = [p for p in paths if not p.exists()]
                    if missing:
                        cs.logging.warning(f"Some files missing for pattern: {missing}")
                    self.lw_species.add_pattern(paths)
            
            # Restore UI state if available
            import json
            with open(path, 'r') as f:
                project_data = json.load(f)
            
            ui_state = project_data.get('ui_state', {})
            
            # Restore detector selection
            selected_detectors = ui_state.get('selected_detectors', [])
            if selected_detectors and self.detector_selection.checkboxes:
                # First uncheck all
                for cb in self.detector_selection.checkboxes.values():
                    cb.setChecked(False)
                # Then check the saved ones
                for det_name in selected_detectors:
                    if det_name in self.detector_selection.checkboxes:
                        self.detector_selection.checkboxes[det_name].setChecked(True)
            
            # Restore anisotropy mode
            anisotropy_mode = ui_state.get('anisotropy_mode', False)
            self.anisotropy_mode_cb.setChecked(anisotropy_mode)
            
            self._update_plots()
            self.btn_export.setEnabled(True)
            self._update_status("Project loaded successfully.")
            
        except Exception as e:
            import traceback
            cs.logging.error(f"Error loading project: {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))

    def _update_plots(self) -> None:
        # Handle multi-anisotropy mode - stack each detector's par/perp horizontally
        if self._result_multi_anisotropy:
            anisotropy_results = self._result_multi_anisotropy
            n_detectors = len(anisotropy_results)
            
            # Find maximum bin size
            max_bins = max(ar['result'].n_bins for ar in anisotropy_results)
            offset = max_bins * 1.1  # 10% gap between par/perp pairs
            
            # 1. Filters - stack each detector's par/perp horizontally
            self.plot_filters.clear()
            for det_idx, ar in enumerate(anisotropy_results):
                res = ar['result']
                det_name = ar['detector']
                base_x = det_idx * 2 * offset  # Each detector gets 2 slots (par + perp)
                x = np.arange(res.n_bins)
                
                for i in range(res.n_species):
                    # Parallel filters
                    self.plot_filters.plot(x + base_x, res.filters_par[i], pen=pg.intColor(i, res.n_species), 
                                          name=f"{det_name}: Species {i+1} (||)")
                    # Perpendicular filters
                    self.plot_filters.plot(x + base_x + offset, res.filters_perp[i], pen=pg.intColor(i, res.n_species), 
                                          name=f"{det_name}: Species {i+1} (⊥)")
                # Zero lines
                self.plot_filters.plot(x + base_x, np.zeros(res.n_bins), pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
                self.plot_filters.plot(x + base_x + offset, np.zeros(res.n_bins), pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
            
            # 2. Reconstruction - stack each detector's par/perp
            self.plot_recon.clear()
            for det_idx, ar in enumerate(anisotropy_results):
                res = ar['result']
                det_name = ar['detector']
                base_x = det_idx * 2 * offset
                x = np.arange(res.n_bins)
                
                # Parallel
                self.plot_recon.plot(x + base_x, res.total_decay_par, pen=pg.intColor(det_idx, n_detectors), 
                                    name=f"{det_name}: Total (||)")
                self.plot_recon.plot(x + base_x, res.reconstruction_par, pen=pg.intColor(det_idx, n_detectors), 
                                    name=f"{det_name}: Recon (||)", style=QtCore.Qt.DashLine)
                # Perpendicular
                self.plot_recon.plot(x + base_x + offset, res.total_decay_perp, pen=pg.intColor(det_idx, n_detectors), 
                                    name=f"{det_name}: Total (⊥)")
                self.plot_recon.plot(x + base_x + offset, res.reconstruction_perp, pen=pg.intColor(det_idx, n_detectors), 
                                    name=f"{det_name}: Recon (⊥)", style=QtCore.Qt.DashLine)
            
            # 3. Residuals - stack each detector's par/perp
            self.plot_residuals.clear()
            for det_idx, ar in enumerate(anisotropy_results):
                res = ar['result']
                det_name = ar['detector']
                base_x = det_idx * 2 * offset
                x = np.arange(res.n_bins)
                
                # Parallel
                self.plot_residuals.plot(x + base_x, res.weighted_residuals_par, pen=pg.intColor(det_idx, n_detectors),
                                        name=f"{det_name} (||)")
                # Perpendicular
                self.plot_residuals.plot(x + base_x + offset, res.weighted_residuals_perp, pen=pg.intColor(det_idx, n_detectors),
                                        name=f"{det_name} (⊥)")
                # Reference lines
                for val in [-3, 0, 3]:
                    pen = pg.mkPen('r' if val != 0 else 'w', style=QtCore.Qt.DashLine)
                    self.plot_residuals.plot(x + base_x, np.full(res.n_bins, val), pen=pen)
                    self.plot_residuals.plot(x + base_x + offset, np.full(res.n_bins, val), pen=pen)
            return
        
        # Handle multi-detector mode - stack detectors horizontally
        if self._result_multi_detector:
            detector_results = self._result_multi_detector
            n_detectors = len(detector_results)
            
            # Find maximum bin size across all detectors
            max_bins = max(dr['result'].n_bins for dr in detector_results)
            offset = max_bins * 1.1  # 10% gap between detectors
            
            # 1. Filters - stack each detector horizontally
            self.plot_filters.clear()
            for det_idx, dr in enumerate(detector_results):
                res = dr['result']
                det_name = dr['detector']
                x = np.arange(res.n_bins) + (det_idx * offset)
                
                for i in range(res.n_species):
                    self.plot_filters.plot(x, res.filters[i], pen=pg.intColor(i, res.n_species), 
                                          name=f"{det_name}: Species {i+1}")
                # Zero line for this detector
                self.plot_filters.plot(x, np.zeros(res.n_bins), pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
            
            # 2. Reconstruction - stack each detector horizontally
            self.plot_recon.clear()
            for det_idx, dr in enumerate(detector_results):
                res = dr['result']
                det_name = dr['detector']
                x = np.arange(res.n_bins) + (det_idx * offset)
                
                self.plot_recon.plot(x, res.total_decay, pen=pg.intColor(det_idx, n_detectors), 
                                    name=f"{det_name}: Total")
                self.plot_recon.plot(x, res.reconstruction, pen=pg.intColor(det_idx, n_detectors), 
                                    name=f"{det_name}: Recon", style=QtCore.Qt.DashLine)
            
            # 3. Residuals - stack each detector horizontally
            self.plot_residuals.clear()
            for det_idx, dr in enumerate(detector_results):
                res = dr['result']
                det_name = dr['detector']
                x = np.arange(res.n_bins) + (det_idx * offset)
                
                self.plot_residuals.plot(x, res.weighted_residuals, pen=pg.intColor(det_idx, n_detectors),
                                        name=f"{det_name}")
                # Reference lines for this detector
                for val in [-3, 0, 3]:
                    pen = pg.mkPen('r' if val != 0 else 'w', style=QtCore.Qt.DashLine)
                    self.plot_residuals.plot(x, np.full(res.n_bins, val), pen=pen)
            return
        
        # Handle Anisotropy mode - stack decays horizontally
        if self._result_anisotropy:
            res = self._result_anisotropy
            x = np.arange(res.n_bins)
            
            # Offset for horizontal stacking
            offset = res.n_bins * 1.1  # 10% gap between channels

            # 1. Filters - stack parallel and perpendicular horizontally
            self.plot_filters.clear()
            for i in range(res.n_species):
                # Parallel filters (left side)
                self.plot_filters.plot(x, res.filters_par[i], pen=pg.intColor(i, res.n_species), 
                                      name=f"Species {i+1} (||)")
                # Perpendicular filters (right side, offset)
                self.plot_filters.plot(x + offset, res.filters_perp[i], pen=pg.intColor(i, res.n_species), 
                                      name=f"Species {i+1} (⊥)")
            # Zero lines for both channels
            self.plot_filters.plot(x, np.zeros_like(x), pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
            self.plot_filters.plot(x + offset, np.zeros_like(x), pen=pg.mkPen('w', style=QtCore.Qt.DashLine))

            # 2. Reconstruction - stack horizontally
            self.plot_recon.clear()
            # Parallel (left)
            self.plot_recon.plot(x, res.total_decay_par, pen='w', name="Total (||)")
            self.plot_recon.plot(x, res.reconstruction_par, pen='r', name="Recon (||)")
            # Perpendicular (right)
            self.plot_recon.plot(x + offset, res.total_decay_perp, pen='w', name="Total (⊥)")
            self.plot_recon.plot(x + offset, res.reconstruction_perp, pen='r', name="Recon (⊥)")

            # 3. Residuals - stack horizontally
            self.plot_residuals.clear()
            # Parallel (left)
            self.plot_residuals.plot(x, res.weighted_residuals_par, pen='g', name="Residuals (||)")
            # Perpendicular (right)
            self.plot_residuals.plot(x + offset, res.weighted_residuals_perp, pen='y', name="Residuals (⊥)")
            # Reference lines for both channels
            for val in [-3, 0, 3]:
                pen = pg.mkPen('r' if val != 0 else 'w', style=QtCore.Qt.DashLine)
                self.plot_residuals.plot(x, np.full_like(x, val), pen=pen)
                self.plot_residuals.plot(x + offset, np.full_like(x, val), pen=pen)
            return
        
        # Standard single-channel mode
        if not self._result:
            return

        res = self._result
        x = np.arange(res.n_bins)

        # 1. Filters
        self.plot_filters.clear()
        for i in range(res.n_species):
            self.plot_filters.plot(x, res.filters[i], pen=pg.intColor(i, res.n_species), name=f"Species {i+1}")
        self.plot_filters.plot(x, np.zeros_like(x), pen=pg.mkPen('w', style=QtCore.Qt.DashLine))

        # 2. Reconstruction
        self.plot_recon.clear()
        self.plot_recon.plot(x, res.total_decay, pen='w', name="Total")
        self.plot_recon.plot(x, res.reconstruction, pen='r', name="Recon", style=QtCore.Qt.DashLine)

        # 3. Residuals
        self.plot_residuals.clear()
        self.plot_residuals.plot(x, res.weighted_residuals, pen='g')
        for val in [-3, 0, 3]:
            self.plot_residuals.plot(x, np.full_like(x, val), pen=pg.mkPen('r' if val != 0 else 'w', style=QtCore.Qt.DashLine))

    def _on_export(self) -> None:
        if not self._result and not self._result_anisotropy and not self._result_multi_detector and not self._result_multi_anisotropy:
            return
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Filters", "fcs_filters.json", "JSON (*.json)")
        if path:
            if self._result_multi_anisotropy:
                # Export multi-anisotropy results
                import json
                export_data = {
                    'mode': 'multi_anisotropy',
                    'detectors': [
                        {
                            'detector': ar['detector'],
                            'result': ar['result'].to_dict()
                        }
                        for ar in self._result_multi_anisotropy
                    ]
                }
                with open(path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif self._result_anisotropy:
                self._result_anisotropy.to_json(path)
            elif self._result_multi_detector:
                # Export multi-detector results as a list
                import json
                export_data = {
                    'mode': 'multi_detector',
                    'detectors': [
                        {
                            'detector': dr['detector'],
                            'result': dr['result'].to_dict()
                        }
                        for dr in self._result_multi_detector
                    ]
                }
                with open(path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:
                self._result.to_json(path)
            self._update_status(f"Exported to {pathlib.Path(path).name}")
