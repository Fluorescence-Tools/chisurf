"""
Main GUI wizard for 2D-FLCS analysis.

This module provides the user interface for 2D-FLCS analysis, integrating
the core functionality into a user-friendly ChiSurf plugin.
"""

import sys
from pathlib import Path
import numpy as np
import traceback
from typing import Optional, Dict, Tuple

# ChiSurf logging
from chisurf import logging

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTabWidget, 
    QProgressBar, QMessageBox, QFileDialog
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFont

import pyqtgraph as pg
import tttrlib

# Local imports
try:
    from chisurf.plugins.fcs.fcs_2d.core import TwoDFDCreator
    from chisurf.plugins.fcs.fcs_2d.fit import TwoDMEMFitter
    from chisurf.plugins.fcs.fcs_2d.gui.worker import TwoDFCSWorker
    from chisurf.plugins.fcs.fcs_2d.gui.tabs.data_tab import DataTab
    from chisurf.plugins.fcs.fcs_2d.gui.tabs.fdc_tab import FDCTab
    from chisurf.plugins.fcs.fcs_2d.gui.tabs.fitting_tab import FittingTab
    from chisurf.plugins.fcs.fcs_2d.gui.tabs.results_tab import ResultsTab
    from chisurf.plugins.fcs.fcs_2d.helpers import (
        get_tttrlib_container_type, get_filetype_from_path, set_plot_image,
        microseconds_to_ticks, nanoseconds_to_ticks
    )
except ImportError:
    # Fallback for direct testing
    try:
        from ..core import TwoDFDCreator
        from ..fit import TwoDMEMFitter
        from .worker import TwoDFCSWorker
        from .tabs.data_tab import DataTab
        from .tabs.fdc_tab import FDCTab
        from .tabs.fitting_tab import FittingTab
        from .tabs.results_tab import ResultsTab
        from ..helpers import (
            get_tttrlib_container_type, get_filetype_from_path, set_plot_image,
            microseconds_to_ticks, nanoseconds_to_ticks
        )
    except ImportError:
        TwoDFDCreator = None
        TwoDMEMFitter = None
        TwoDFCSWorker = None
        DataTab = None
        FDCTab = None
        FittingTab = None
        ResultsTab = None


class TwoDFCSWizard(QWidget):
    """
    Main wizard for 2D-FLCS analysis using tabbed interface.
    
    Provides detector setup, data input, 2D-FDC creation, MEM fitting, and results visualization.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("2D-FLCS: TwoDFCSWizard.__init__ called")
        
        if TwoDFDCreator is None or TwoDMEMFitter is None:
            self.logger.error("2D-FLCS: Required 2D-FCS modules not available")
            raise ImportError("Required 2D-FCS modules not available")
        
        # Initialize components
        self.fdc_creator = TwoDFDCreator()
        self.mem_fitter = TwoDMEMFitter()
        self.worker = TwoDFCSWorker()
        
        # Data storage
        self.tttr_data = None
        self.tttr_metadata = {}
        self._detector_settings = None
        self.fdc_data = {}
        self.fit_results = None
        
        self.macro_time_resolution = None
        self.micro_time_resolution = None
        self.micro_binning_factor = 1
        
        # Detector setup page
        self.detector_wizard_page = None
        self._init_detector_page()
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
        
        self.logger.info("2D-FLCS: TwoDFCSWizard initialization completed")
    
    def _init_detector_page(self):
        """Initialize detector wizard page."""
        try:
            from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage
            self.detector_wizard_page = DetectorWizardPage()
            self.detector_wizard_page.setParent(self)
            
            if hasattr(self.detector_wizard_page, 'setup_combo'):
                self.detector_wizard_page.setup_combo.currentTextChanged.connect(self._refresh_detector_combo)
            if hasattr(self.detector_wizard_page, 'micro_binning_combo'):
                self.detector_wizard_page.micro_binning_combo.currentTextChanged.connect(self._on_micro_binning_changed)
                self._sync_micro_binning_from_detector()
                
        except (ImportError, Exception) as e:
            self.logger.warning(f"DetectorWizardPage not available or error: {e}")
            self.detector_wizard_page = None
    
    def _setup_ui(self):
        """Setup the tabbed UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("2D-FLCS Analysis")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self._setup_tabs()
        
        self.tabs.setCurrentIndex(1)  # Default to Data Input
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self._refresh_detector_combo()
    
    def _setup_tabs(self):
        """Instantiate and add tabs."""
        # Detector Setup Tab
        if self.detector_wizard_page is not None:
            self.tabs.addTab(self.detector_wizard_page, "Detector Setup")
        else:
            tab = QWidget()
            QVBoxLayout(tab).addWidget(QLabel("Detector setup not available"))
            self.tabs.addTab(tab, "Detector Setup")
            
        # Data Input Tab
        self.data_tab = DataTab(self)
        self.tabs.addTab(self.data_tab, "Data Input")
        
        # 2D-FCS Creation Tab
        self.fdc_tab = FDCTab(self)
        self.tabs.addTab(self.fdc_tab, "2D-FCS Creation")
        
        # 2D-MEM Fitting Tab
        self.fitting_tab = FittingTab(self)
        self.tabs.addTab(self.fitting_tab, "2D-MEM Fitting")
        
        # Results Tab
        self.results_tab = ResultsTab(self)
        self.tabs.addTab(self.results_tab, "Results")

    def _connect_signals(self):
        """Connect all signals between tabs, worker, and wizard logic."""
        # Tab signals
        self.data_tab.file_changed.connect(self._on_file_path_changed)
        self.fdc_tab.create_clicked.connect(self._create_2d_fdc)
        self.fdc_tab.log_intensity_changed.connect(self._update_fdc_preview)
        self.fitting_tab.start_fitting_clicked.connect(self._start_fitting)
        self.results_tab.export_data_clicked.connect(self._export_data)
        self.results_tab.export_plots_clicked.connect(self._export_plots)
        
        # Worker signals
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.fdc_created.connect(self._on_fdc_created)
        self.worker.fitting_complete.connect(self._on_fitting_complete)
        self.worker.error_occurred.connect(self._on_error_occurred)

    # --- Coordination Logic ---

    def _refresh_detector_combo(self):
        """Delegated to data_tab but called from wizard init."""
        if hasattr(self, 'data_tab'):
            self.data_tab.refresh_detector_combo()

    def _on_file_path_changed(self, file_path):
        """Handle file path changes from data tab."""
        if file_path:
            # Short timer to allow UI to update before long load
            QTimer.singleShot(100, self._load_data)

    def _sync_micro_binning_from_detector(self):
        """Sync microtime binning factor from detector wizard."""
        factor = 1
        try:
            if self.detector_wizard_page and hasattr(self.detector_wizard_page, 'micro_binning_combo'):
                factor_text = self.detector_wizard_page.micro_binning_combo.currentText()
                factor = int(factor_text)
            
            settings = self._get_detector_settings()
            if settings and 'tttr_reading' in settings:
                reading_factor = int(settings['tttr_reading'].get('micro_time_binning', factor))
                factor = reading_factor
        except (ValueError, TypeError):
            factor = 1
        
        self.micro_binning_factor = max(1, factor)

    def _on_micro_binning_changed(self, value):
        try:
            self.micro_binning_factor = max(1, int(value))
        except (ValueError, TypeError):
            pass

    def _get_detector_settings(self):
        if self.detector_wizard_page:
            try:
                return self.detector_wizard_page.get_settings()
            except Exception:
                pass
        return {}

    def _load_data(self):
        """Load TTTR data using tttrlib."""
        file_path = self.data_tab.file_path_edit.text()
        if not file_path:
            return
        
        selected_detector = self.data_tab.get_selected_detector()
        if not selected_detector or "not available" in selected_detector:
            QMessageBox.warning(self, "Warning", "Please configure detector setup first.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0) # Indeterminate
            
            self.tttr_data = tttrlib.TTTR(file_path)
            header = self.tttr_data.get_header()
            
            self.macro_time_resolution = getattr(header, "macro_time_resolution", None)
            self.micro_time_resolution = getattr(header, "micro_time_resolution", None)
            
            self.tttr_metadata = {
                'n_photons': len(self.tttr_data),
                'duration': getattr(self.tttr_data, 'get_duration', lambda: 'Unknown')(),
                'macro_time_resolution': self.macro_time_resolution,
                'micro_time_resolution': self.micro_time_resolution,
                'files': [file_path]
            }
            
            n_photons = len(self.tttr_data)
            info_text = f"Loaded {n_photons:,} photons, Duration: {self.tttr_metadata['duration']}, Detector: {selected_detector}"
            self.data_tab.data_info_label.setText(info_text)
            
            # Use raw tttr_data object for previews in data_tab
            self.data_tab.update_data_preview(self.tttr_data, self.tttr_data, 1e-3, self.micro_binning_factor)
            
            self.progress_bar.setVisible(False)
            if n_photons == 0:
                QMessageBox.warning(self, "Warning", "File loaded but contains 0 photons.")
            else:
                QMessageBox.information(self, "Success", f"Data loaded successfully!\nPhotons: {n_photons:,}")

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")

    def _create_2d_fdc(self, params):
        """Start 2D-FDC creation in worker thread."""
        if self.tttr_data is None:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
            
        try:
            macro_res = self.macro_time_resolution
            micro_res = self.micro_time_resolution
            
            if macro_res is None or micro_res is None:
                raise ValueError("Macro/Micro time resolution not available in TTTR header.")
            
            # Convert UI parameters to ticks
            dT_ticks = microseconds_to_ticks(params['dT'], macro_res)
            ddT_ticks = microseconds_to_ticks(params['ddT'], macro_res)
            tMin_ticks = nanoseconds_to_ticks(params['tMin'], micro_res)
            tMax_ticks = nanoseconds_to_ticks(params['tMax'], micro_res)
            
            # Store ticks for later axis conversion
            self.tmin_ticks = tMin_ticks
            self.tmax_ticks = tMax_ticks
            self.tmin = params['tMin']
            self.tmax = params['tMax']
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.fdc_tab.create_fdc_button.setEnabled(False)
            
            self.worker.set_create_fdc_task(
                macro_times=self.tttr_data.macro_times,
                micro_times=self.tttr_data.micro_times,
                dT=dT_ticks,
                ddT=ddT_ticks,
                tMin=tMin_ticks,
                tMax=tMax_ticks,
                logt_imax=params['logt_imax']
            )
            self.worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start creation: {e}")

    def _start_fitting(self, params):
        """Start MEM fitting in worker thread."""
        if not self.fdc_data:
            QMessageBox.warning(self, "Warning", "Please create 2D-FDC first.")
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.fitting_tab.fit_button.setEnabled(False)
        
        # Prepare data for fitting class
        fit_params = params.copy()
        fit_params.update({
            'mat_2dfdc': self.fdc_data['mat_lin'],
            'mat_2dfdc_cor': self.fdc_data['mat_lin'], # Simple version
            'mat_2dfdc_it': self.fdc_data['mat_lin_t']
        })
        
        self.worker.set_fit_mem_task(**fit_params)
        self.worker.start()

    def _on_progress_updated(self, value):
        self.progress_bar.setValue(int(value * 100))

    def _on_fdc_created(self, result):
        self.fdc_data = result
        self.progress_bar.setVisible(False)
        self.fdc_tab.create_fdc_button.setEnabled(True)
        
        # Convert ticks to nanoseconds for display
        if hasattr(self, 'tmin_ticks') and self.tmax_ticks > self.tmin_ticks:
            ticks = self.fdc_data['mat_lin_t']
            physical_time = self.tmin + (ticks - self.tmin_ticks) / (self.tmax_ticks - self.tmin_ticks) * (self.tmax - self.tmin)
            self.fdc_data['mat_lin_t'] = physical_time
            if 'mat_log_t' in self.fdc_data:
                ticks_log = self.fdc_data['mat_log_t']
                physical_time_log = self.tmin + (ticks_log - self.tmin_ticks) / (self.tmax_ticks - self.tmin_ticks) * (self.tmax - self.tmin)
                self.fdc_data['mat_log_t'] = physical_time_log

        # Update results tab with info
        n_photons = len(self.tttr_data) if self.tttr_data else 0
        lin_shape = result['mat_lin'].shape if 'mat_lin' in result else 'N/A'
        results_text = f"2D-FCS Created:\n- Photons: {n_photons:,}\n- Matrix shape: {lin_shape}"
        self.results_tab.set_results_text(results_text)
        
        self._update_fdc_preview()
        self.tabs.setCurrentIndex(2) # Switch to FDC preview (already there) or Results? Standard flow.

    def _on_fitting_complete(self, result):
        self.fit_results = result
        self.progress_bar.setVisible(False)
        self.fitting_tab.fit_button.setEnabled(True)
        
        results_text = f"2D-MEM Fitting Completed:\n- Success: {result.get('success', False)}\n- Q-value: {result.get('q_value', 'N/A'):.4f}"
        self.results_tab.set_results_text(results_text)
        self.results_tab.update_results_plots(result, self.fdc_data, self.micro_binning_factor)
        self.tabs.setCurrentIndex(4) # Switch to results tab

    def _on_error_occurred(self, error_msg):
        self.logger.error(f"Worker error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
        self.progress_bar.setVisible(False)
        self.fdc_tab.create_fdc_button.setEnabled(True)
        self.fitting_tab.fit_button.setEnabled(True)

    def _update_fdc_preview(self):
        self.fdc_tab.update_previews(self.fdc_data, self.tttr_metadata, self.micro_binning_factor)

    def _export_data(self):
        if not self.fdc_data:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "NPZ Files (*.npz);;All Files (*)")
        if file_path:
            try:
                data_to_save = {**self.fdc_data}
                if self.fit_results:
                    for k, v in self.fit_results.items():
                        if isinstance(v, np.ndarray):
                            data_to_save[f"fit_{k}"] = v
                np.savez(file_path, **data_to_save)
                QMessageBox.information(self, "Success", "Data exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {e}")

    def _export_plots(self):
        QMessageBox.information(self, "Info", "Export plots function - use pyqtgraph context menu (right-click on plots) for exporting.")


# ChiSurf plugin entry point
if __name__ == "plugin":
    try:
        plugin_window = TwoDFCSWizard()
        plugin_window.show()
        plugin_window.resize(800, 600)
    except Exception as e:
        print(f"Error in wizard.py plugin entry: {e}")
        traceback.print_exc()

elif __name__ == "__main__":
    from qtpy.QtWidgets import QApplication
    app = QApplication(sys.argv)
    try:
        test_window = TwoDFCSWizard()
        test_window.show()
        test_window.resize(1000, 800)
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Failed to create test window: {e}")
        sys.exit(1)
