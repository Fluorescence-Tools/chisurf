"""
BrickMicWizard for burst selection analysis

This module provides the main wizard interface for analyzing single-molecule
fluorescence bursts in TTTR data.
"""

from pathlib import Path
from qtpy import QtWidgets, QtCore, QtGui
import zipfile
import tempfile
import io

import pyqtgraph as pg
from guidata.widgets.dataframeeditor import DataFrameEditor

import pandas as pd
import numpy as np

import chisurf.gui.decorators
import chisurf.gui.widgets
import chisurf.gui.widgets.wizard

from chisurf import logging
from chisurf import settings

from sklearn.mixture import GaussianMixture

from .gmm_settings_dialog import GMMSettingsDialog
from chisurf.gui.widgets.progress import EnhancedProgressDialog

# Module-level logger for this file
logger = logging.getLogger(__name__)


class DirectoryDropListWidget(QtWidgets.QListWidget):
    """QListWidget that accepts folder drops and emits a list of dropped paths."""
    pathsDropped = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        has_urls = event.mimeData().hasUrls()
        logger.debug("DirectoryDropListWidget.dragEnterEvent: has_urls=%s", has_urls)
        if has_urls:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        logger.debug("DirectoryDropListWidget.dragMoveEvent")
        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        urls = event.mimeData().urls() or []
        logger.debug("DirectoryDropListWidget.dropEvent: urls=%s", [u.toString() for u in urls])
        paths = []
        for url in urls:
            local = url.toLocalFile()
            if local:
                p = Path(local)
                if p.exists():
                    paths.append(p)
                else:
                    logger.warning("Dropped path does not exist: %s", local)
            else:
                logger.debug("URL without local file ignored: %s", url.toString())
        if paths:
            logger.info("Emitting pathsDropped with %d path(s)", len(paths))
            self.pathsDropped.emit(paths)
        else:
            logger.info("No valid paths to emit from dropEvent")
        event.acceptProposedAction()

    def supportedDropActions(self):
        return QtCore.Qt.CopyAction


class BatchProcessingDialog(QtWidgets.QDialog):
    """
    Dialog for batch analysis: accept dropped folders, recursively find folders
    containing TTTR files, list them, and process sequentially using the
    provided BrickMicWizard instance.
    """
    def __init__(self, parent_wizard: 'BurstSelectionTool'):
        super().__init__(parent_wizard)
        self.wizard = parent_wizard
        self.setWindowTitle("Batch Burst Analysis")
        self.resize(700, 500)

        # Build UI
        main_layout = QtWidgets.QVBoxLayout(self)

        info_label = QtWidgets.QLabel(
            "Drop folders here. Folders containing TTTR files will be added.\n"
            "If a folder does not contain TTTR files, subfolders will be scanned."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        # Custom list widget that accepts directory drops
        self.list_widget = DirectoryDropListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_widget.pathsDropped.connect(self._add_folders_from_paths)
        main_layout.addWidget(self.list_widget, 1)

        # Buttons: delete selected, clear, process
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_delete = QtWidgets.QPushButton("Delete Selected")
        self.btn_clear = QtWidgets.QPushButton("Clear All")
        self.btn_process = QtWidgets.QPushButton("Process")
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_delete)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_process)
        main_layout.addLayout(btn_row)

        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_clear.clicked.connect(self.list_widget.clear)
        self.btn_process.clicked.connect(self._process)

        self.allowed_extensions = {
            '.ht3', '.ptu', '.spc', '.hdf', '.h5'
        }
        logger.debug("BatchProcessingDialog allowed_extensions=%s", sorted(self.allowed_extensions))

    # --- Helpers ---
    def _folder_has_tttr_files(self, folder: Path) -> list[str]:
        files = []
        try:
            for child in folder.iterdir():
                if child.is_file() and child.suffix.lower() in self.allowed_extensions:
                    files.append(str(child.resolve()))
        except Exception as e:
            logger.warning("Failed to scan folder '%s': %s", folder, e)
        logger.debug("Scanned folder '%s' -> %d tttr file(s)", folder, len(files))
        return files

    def _add_folder_unique(self, folder: Path):
        folder_str = str(folder.resolve())
        # avoid duplicates
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).text() == folder_str:
                logger.debug("Folder already in list, skipping: %s", folder_str)
                return
        logger.info("Adding folder to batch list: %s", folder_str)
        self.list_widget.addItem(folder_str)

    def _add_folders_from_paths(self, paths: list[Path]):
        """
        For each dropped path:
        - If it's a directory and contains TTTR files, add the directory.
        - If it doesn't contain TTTR files, recursively scan subdirectories
          and add those that do contain TTTR files.
        - Ignore files.
        """
        logger.info("Received %d path(s) from drop", len(paths) if paths else 0)
        for p in paths:
            try:
                if p.is_dir():
                    logger.debug("Scanning dropped directory: %s", p)
                    tttr_files = self._folder_has_tttr_files(p)
                    if tttr_files:
                        logger.info("Directory has %d TTTR file(s): %s", len(tttr_files), p)
                        self._add_folder_unique(p)
                    else:
                        # recurse into subfolders
                        found_any = False
                        for sub in p.rglob('*'):
                            if sub.is_dir():
                                sub_files = self._folder_has_tttr_files(sub)
                                if sub_files:
                                    found_any = True
                                    self._add_folder_unique(sub)
                        if not found_any:
                            logger.warning("No TTTR files found (even in subfolders) for: %s", p)
                else:
                    logger.debug("Ignoring non-directory drop: %s", p)
            except Exception as e:
                logger.exception("Error while processing dropped path '%s': %s", p, e)

    def _delete_selected(self):
        for item in self.list_widget.selectedItems():
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)

    def _process(self):
        n = self.list_widget.count()
        logger.info("BatchProcessingDialog: starting process for %d folder(s)", n)
        if n == 0:
            QtWidgets.QMessageBox.information(self, "No items", "No folders to process.")
            return

        progress = EnhancedProgressDialog(
            title="Batch Processing",
            label_text="Starting batch...",
            min_value=0,
            max_value=n,
            parent=self
        )
        progress.show()

        # Ensure wizard UI is enabled during processing; wizard.process_all_files manages its own state
        for i in range(n):
            if progress.wasCanceled():
                logger.warning("BatchProcessingDialog: processing canceled by user at index %d", i)
                break
            item = self.list_widget.item(i)
            folder_str = item.text()
            folder = Path(folder_str)

            # Highlight current item and ensure it is visible
            try:
                self.list_widget.setCurrentRow(i)
                self.list_widget.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
                # Light yellow while processing
                item.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 200)))
            except Exception:
                pass

            logger.info("Processing folder %d/%d: %s", i+1, n, folder_str)

            # Collect TTTR files in this folder (non-recursive)
            files = []
            for child in folder.iterdir():
                if child.is_file() and child.suffix.lower() in self.allowed_extensions:
                    files.append(str(child.resolve()))

            logger.debug("Found %d file(s) in folder '%s': %s", len(files), folder_str, files)
            if not files:
                logger.warning("No TTTR files found in folder: %s", folder_str)

            # Use the standard drop-based approach to populate the TTTR photon filter
            # Start from a clean state for each folder
            try:
                self.wizard.burst_finder.onClearFiles()
            except Exception:
                pass

            # Simulate a drop of the folder path into the lineEdit-driven injector
            # The injector expands directories to files and runs the canonical loading flow
            self.wizard.burst_finder.settings['tttr_filenames'] = files

            # For user feedback show the folder in the line edit
            try:
                if files:
                    self.wizard.burst_finder.lineEdit.setText(files[0])
                else:
                    self.wizard.burst_finder.lineEdit.setText(folder_str)
            except Exception:
                pass

            # Call the exposed drop handler if available; otherwise fall back to direct load
            try:
                self.wizard.burst_finder._after_file_drop()
            except Exception:
                # Fallback in case drop handler isn't available
                try:
                    self.wizard.burst_finder.read_tttr()
                except Exception:
                    pass

            logger.debug("Batch: populated via drop handler; current files=%s", self.wizard.burst_finder.settings.get('tttr_filenames'))
            progress.update_progress(i, text=f"Processing {folder.name} ({i+1}/{n})")
            QtWidgets.QApplication.processEvents()
            try:
                self.wizard.process_all_files()
                # Mark as done (light green)
                try:
                    item.setBackground(QtGui.QBrush(QtGui.QColor(200, 255, 200)))
                except Exception:
                    pass
            except Exception as e:
                # Mark as failed (light red)
                try:
                    item.setBackground(QtGui.QBrush(QtGui.QColor(255, 200, 200)))
                except Exception:
                    pass
                logger.exception("Error processing folder '%s' with files=%s: %s", folder_str, files, e)
                QtWidgets.QMessageBox.warning(self, "Error", f"Error processing folder:\n{folder_str}\n\n{e}")

            progress.update_progress(i + 1)

        progress.finish(final_text="Batch completed")
        logger.info("BatchProcessingDialog: finished batch processing")
        self.accept()


class BurstSelectionTool(QtWidgets.QMainWindow):

    def open_batch_dialog(self):
        dlg = BatchProcessingDialog(self)
        dlg.exec_()

    def get_optimal_components(self, data, max_components=None):
        """
        Determines the optimal number of Gaussian components using BIC.

        Args:
            data: The data to fit (must be reshaped for GMM)
            max_components: Maximum number of components to try (uses gmm_settings if None)

        Returns:
            optimal_k: The optimal number of components
            bic_scores: List of BIC scores for each number of components
        """
        if len(data) < 2:
            return 1, [0]

        # Use max_components from gmm_settings if not specified
        if max_components is None:
            max_components = self.gmm_settings['max_components']

        n_components_range = range(1, min(max_components + 1, len(data)))
        bic_scores = []

        for n_components in n_components_range:
            # Fit GMM for this number of components
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=self.gmm_settings['covariance_type'],
                random_state=self.gmm_settings['random_state'],
                max_iter=self.gmm_settings['max_iter'],
                n_init=self.gmm_settings['n_init'],
                tol=self.gmm_settings['tol'],
                reg_covar=self.gmm_settings['reg_covar']
            )
            gmm.fit(data)
            bic_scores.append(gmm.bic(data))

        # Find the number of components with the lowest BIC score
        optimal_k = n_components_range[np.argmin(bic_scores)]

        return optimal_k, bic_scores

    def show_dataframe_editor(self):
        """
        Pop up a spreadsheet-style editor for the full current_df.
        If the user accepts, replace current_df and refresh the UI.
        """
        if self.current_df is None:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "No burst data loaded—nothing to show."
            )
            return

        dlg = DataFrameEditor(self)
        # set up the editor on the current DataFrame
        if not dlg.setup_and_check(self.current_df, title="Burst Results"):
            return

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # user hit OK: grab the possibly-modified DataFrame back
            self.current_df = dlg.get_value()
            # refresh the preview and histogram
            self.populate_table(self.current_df)
            self.update_histogram()


    @chisurf.gui.decorators.init_with_ui("gui.ui", path=chisurf.settings.plugin_path / "burst" / "burst_selection")
    def __init__(self, *args, 
                 show_channel_selection=True,
                 show_clear_button=False, 
                 show_decay_button=False, 
                 show_filter_button=False, 
                 **kwargs):
        # ---------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------
        # base class init is called by decorator
        # super().__init__(*args, **kwargs)

        # Store the initial visibility settings
        self.show_channel_selection = show_channel_selection
        self.show_clear_button = show_clear_button
        self.show_decay_button = show_decay_button
        self.show_filter_button = show_filter_button

        # Initialize GMM settings with default values
        self.gmm_settings = {
            'covariance_type': 'full',
            'random_state': 42,
            'max_iter': 300,  # Increased from 100 to allow more iterations for convergence
            'n_init': 10,     # Increased from 5 to try more initializations
            'tol': 1e-3,
            'max_components': 10,
            'reg_covar': 1e-6  # Add regularization to prevent singular covariance matrices
        }

        # Create the channel settings dialog
        self.channel_settings_dialog = QtWidgets.QDialog(self)
        self.channel_settings_dialog.setWindowTitle("Channel Settings")
        self.channel_settings_dialog.resize(800, 600)  # Set an appropriate size
        dialog_layout = QtWidgets.QVBoxLayout(self.channel_settings_dialog)

        self.channel_definer = chisurf.gui.widgets.wizard.DetectorWizardPage(
            parent=self.channel_settings_dialog,
            json_file=None
        )
        dialog_layout.addWidget(self.channel_definer)

        # Add OK button to close the dialog
        ok_button = QtWidgets.QPushButton("OK", self.channel_settings_dialog)
        ok_button.clicked.connect(self.channel_settings_dialog.accept)
        dialog_layout.addWidget(ok_button)

        # Hide the channel settings by default (will be shown via menu)
        self.channel_settings_dialog.hide()

        self.burst_finder = chisurf.gui.widgets.wizard.WizardTTTRPhotonFilter(
            windows=self.channel_definer.windows,
            detectors=self.channel_definer.detectors,
            show_dT=True,
            show_burst=False,
            show_mcs=True,
            show_decay=False,
            show_filter=False
        )
        self.verticalLayout_2.addWidget(self.burst_finder)

        # Create a pyqtgraph PlotWidget for the histogram.
        self.plotWidget = pg.PlotWidget()
        self.plotWidget.setLabel('bottom', 'Value')
        self.plotWidget.setLabel('left', 'Frequency')
        while self.verticalLayout_4.count():
            child = self.verticalLayout_4.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.verticalLayout_4.addWidget(self.plotWidget)

        # Store references to data and UI elements.
        self.node_objects: dict = {}
        self.node_data: dict = {}
        self.connections: list[(int, int)] = []
        self.current_df = None  # We'll keep the final, concatenated DataFrame here

        # Add a checkbox for auto-determining optimal components
        self.checkBox_auto_components = QtWidgets.QCheckBox("Auto-determine optimal components")
        self.checkBox_auto_components.setToolTip("Automatically determine the optimal number of Gaussian components using BIC")
        self.verticalLayout_3.addWidget(self.checkBox_auto_components)

        # Connect signals to update the histogram
        self.comboBox.currentTextChanged.connect(self.update_histogram)
        self.spinBox_3.valueChanged.connect(self.update_histogram)
        self.doubleSpinBox_3.valueChanged.connect(self.update_histogram)
        self.doubleSpinBox_4.valueChanged.connect(self.update_histogram)
        self.spinBox.valueChanged.connect(self.update_histogram)
        self.checkBox_auto_components.stateChanged.connect(self.update_histogram)

        # Connect buttons
        self.pushButton.clicked.connect(self.process_all_files)
        self.pushButton_2.clicked.connect(self.clear_data)
        self.pushButton_show_df.clicked.connect(self.show_dataframe_editor)
        # Batch processing button
        try:
            self.pushButton_batch.clicked.connect(self.open_batch_dialog)
        except Exception:
            pass
        
        # Connect checkBox_FileCSV and checkBox_FileMFDHDF to their respective handlers
        self.checkBox_FileCSV.stateChanged.connect(self.on_file_format_toggled)
        self.checkBox_FileMFDHDF.stateChanged.connect(self.on_mfd_hdf_toggled)

        # Setup menubar
        self.setup_menubar()

        # Set visibility of UI elements based on parameters
        if not self.show_channel_selection:
            self.burst_finder.groupBox_3.hide()
        if not self.show_clear_button:
            self.burst_finder.toolButton_6.hide()
        if not self.show_decay_button:
            self.burst_finder.toolButton_3.hide()
        if not self.show_filter_button:
            self.burst_finder.toolButton_4.hide()

    # --------------------------------------------------------------------------
    # Drag & Drop Events
    # --------------------------------------------------------------------------
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept only file URLs being dragged into the main window."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """
        Extract file paths from the dropped URLs and process them as needed.
        For example, you might want to store these paths in burst_finder.settings
        and then call process_all_files, or simply parse them here.
        """
        file_paths = []
        urls = event.mimeData().urls() or []
        for url in urls:
            # Convert to local file path (handles local files, not necessarily remote)
            local_path = url.toLocalFile()
            if local_path:
                file_paths.append(local_path)
        event.acceptProposedAction()

        logger.info("Main window drop: %d file(s) -> %s", len(file_paths), file_paths)

        # Store in burst_finder.settings and auto-process
        self.burst_finder.settings['tttr_filenames'] = list(file_paths)
        self.process_all_files()

    # --------------------------------------------------------------------------
    # Processing
    # --------------------------------------------------------------------------
    def process_all_files(self) -> None:
        """
        Calls save_selection first to store burst selections,
        then loads and processes the saved burst files for display.
        """
        tttr_files = self.burst_finder.settings.get('tttr_filenames', [])
        logger.debug("process_all_files: tttr_filenames count=%d types=%s", len(tttr_files) if tttr_files else 0, list({type(f).__name__ for f in (tttr_files or [])}))
        if not tttr_files:
            logger.info("No TTTR files to process.")
            return

        # First, save the selection (ensures all bursts are processed)
        # Determine which output types to use based on the state of both checkboxes
        output_types = set()
        if self.checkBox_FileMFDHDF.isChecked():
            output_types.add("hdf5")
        if self.checkBox_FileCSV.isChecked():
            output_types.add("bur")
            
        # Check if zip output is requested
        zip_output = self.checkBox_ZipOutput.isChecked()
        
        # Check if folder removal is requested
        remove_folder = self.checkBox_RemoveFolder.isChecked()

        self.burst_finder.save_selection(output_types=output_types, zip_output=zip_output, remove_folder=remove_folder)
        logging.info("Photon selection saved. Now loading burst files.")

        accumulated_results = []
        ui_columns = [
            "First Photon",
            "Last Photon",
            "Duration (ms)",
            "Number of Photons (red)",
            "Number of Photons (green)"
        ]

        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(len(tttr_files))
        self.progressBar.setValue(0)
        self.centralwidget.setEnabled(False)

        for index, fn in enumerate(tttr_files):
            file_path = Path(fn)
            analysis_folder_name = self.burst_finder.target_path
            bur_file_path = file_path.parent / analysis_folder_name / 'bi4_bur' / f"{file_path.stem}.bur"

            # Check if the .bur file exists directly
            try:
                # Log the exact path we're checking to help with debugging
                logging.info(f"Checking for .bur file at: {bur_file_path}")
                
                if bur_file_path.exists():
                    # Load the pre-saved burst file
                    logging.info(f"Found .bur file directly: {bur_file_path}")
                    df = pd.read_csv(bur_file_path, sep="\t")
                else:
                    # Try multiple possible zip file locations
                    zip_found = False
                    
                    # Standard zip location: analysis_folder_name/analysis_folder_name.zip
                    zip_file_path = file_path.parent / analysis_folder_name / f"{analysis_folder_name}.zip"
                    
                    # Alternative zip locations to try if the standard one doesn't exist
                    alt_zip_paths = [
                        file_path.parent / f"{analysis_folder_name}.zip",  # analysis_folder_name.zip in parent directory
                        file_path.parent / analysis_folder_name / "output.zip",  # output.zip in analysis folder
                        file_path.parent / "output.zip"  # output.zip in parent directory
                    ]
                    
                    # Try the standard zip location first
                    if zip_file_path.exists():
                        zip_found = True
                    else:
                        # Try alternative zip locations
                        for alt_path in alt_zip_paths:
                            if alt_path.exists():
                                zip_file_path = alt_path
                                zip_found = True
                                break
                    
                    if zip_found:
                        logging.info(f"Trying to load .bur file from zip: {zip_file_path}")
                        try:
                            with zipfile.ZipFile(str(zip_file_path), 'r') as zip_file:
                                # Get all files in the zip to help with debugging
                                all_files = zip_file.namelist()
                                
                                # Try multiple possible paths for the .bur file in the zip
                                bur_filenames = [
                                    f"bi4_bur/{file_path.stem}.bur",
                                    f"bur/{file_path.stem}.bur",
                                    f"{file_path.stem}.bur",
                                    f"{analysis_folder_name}/bi4_bur/{file_path.stem}.bur",
                                    f"{analysis_folder_name}/bur/{file_path.stem}.bur"
                                ]
                                
                                # Try each possible path
                                bur_found = False
                                for bur_filename in bur_filenames:
                                    try:
                                        # Try to extract and read the file
                                        with zip_file.open(bur_filename) as bur_file:
                                            df = pd.read_csv(io.TextIOWrapper(bur_file), sep="\t")
                                        logging.info(f"Successfully loaded .bur file from zip: {bur_filename}")
                                        bur_found = True
                                        break
                                    except KeyError:
                                        # This path doesn't exist, try the next one
                                        continue
                                
                                # If none of the predefined paths worked, try to find a matching .bur file
                                if not bur_found:
                                    stem = file_path.stem
                                    matching_files = [f for f in all_files if f.endswith(f"{stem}.bur")]
                                    if matching_files:
                                        bur_filename = matching_files[0]
                                        with zip_file.open(bur_filename) as bur_file:
                                            df = pd.read_csv(io.TextIOWrapper(bur_file), sep="\t")
                                        logging.info(f"Successfully loaded .bur file from zip using filename search: {bur_filename}")
                                        bur_found = True
                                
                                if not bur_found:
                                    logging.info(f"Warning: .bur file not found in zip. Available files: {all_files}")
                                    continue
                                    
                        except Exception as e:
                            logging.info(f"Error loading from zip: {str(e)}")
                            continue
                    else:
                        logging.info(f"Warning: Neither .bur file nor zip found: {bur_file_path}")
                        continue
            except Exception as e:
                logging.info(f"Unexpected error processing file {file_path}: {str(e)}")
                continue

            # Ensure all expected columns exist
            missing_cols = [col for col in ui_columns if col not in df.columns]
            for col in missing_cols:
                df[col] = 0

            # Create a limited subset DataFrame for UI
            df_ui = df[ui_columns].copy()

            # Compute Proximity Ratio for UI
            df_ui["Proximity Ratio"] = df_ui.apply(
                lambda row: row["Number of Photons (red)"] /
                            (row["Number of Photons (red)"] + row["Number of Photons (green)"])
                if (row["Number of Photons (red)"] + row["Number of Photons (green)"]) > 0 else 0,
                axis=1
            ).round(3)

            accumulated_results.append(df_ui)
            self.progressBar.setValue(index + 1)
            QtWidgets.QApplication.processEvents()

        # Combine all loaded data and update UI
        if accumulated_results:
            final_df = pd.concat(accumulated_results, ignore_index=True)
            self.current_df = final_df
            # self.populate_table(final_df)
            new_columns = final_df.columns.tolist()
            # only refresh if it's different
            existing = [self.comboBox.itemText(i) for i in range(self.comboBox.count())]
            if existing != new_columns:
                self.comboBox.clear()
                self.comboBox.addItems(new_columns)
            # default to "Proximity Ratio" if present
            idx = self.comboBox.findText("Proximity Ratio")
            if idx != -1:
                self.comboBox.setCurrentIndex(idx)
            # now draw the histogram
            self.update_histogram()

        logging.info("All burst files loaded successfully.")
        self.centralwidget.setEnabled(True)

    def update_histogram(self):
        """
        Updates the histogram display and fits a Gaussian Mixture Model (GMM) to the data.

        This method uses scikit-learn's GaussianMixture for robust statistical modeling of
        the data distribution. GMM provides several advantages over curve fitting:
        1. Better handling of multi-modal distributions
        2. More robust parameter estimation
        3. Proper statistical modeling of the underlying data
        4. Automatic handling of component weights

        The number of Gaussian components can be determined in two ways:
        1. Manual: User specifies the number using the spinBox
        2. Automatic: When "Auto-determine optimal components" is checked, the method uses
           Bayesian Information Criterion (BIC) to find the optimal number of components
        """
        if self.current_df is None:
            return

        selected_feature = self.comboBox.currentText()
        if not selected_feature:
            return

        data = self.current_df[selected_feature]
        try:
            data = pd.to_numeric(data)
        except Exception as e:
            logging.info(f"Could not convert data in column {selected_feature} to numeric: {e}")
            return

        num_bins = int(self.spinBox_3.value())
        min_val = float(self.doubleSpinBox_3.value())
        max_val = float(self.doubleSpinBox_4.value())

        # Clear the plot
        self.plotWidget.clear()

        # Set labels
        self.plotWidget.setTitle(f"Histogram of {selected_feature}")
        self.plotWidget.setLabel('bottom', selected_feature)
        self.plotWidget.setLabel('left', 'Frequency')

        # Calculate histogram
        filtered_data = data.dropna()[1::2].values

        # Ensure min_val is less than max_val
        if min_val >= max_val:
            # If values are invalid, use data range or default values
            if len(filtered_data) > 0:
                data_min = np.min(filtered_data)
                data_max = np.max(filtered_data)
                if data_min < data_max:
                    min_val, max_val = data_min, data_max
                else:
                    # If data has no range, use default values
                    min_val, max_val = 0, 1
            else:
                # No data, use default values
                min_val, max_val = 0, 1

            # Update the UI spinboxes without triggering update_histogram again
            self.doubleSpinBox_3.blockSignals(True)
            self.doubleSpinBox_4.blockSignals(True)
            self.doubleSpinBox_3.setValue(min_val)
            self.doubleSpinBox_4.setValue(max_val)
            self.doubleSpinBox_3.blockSignals(False)
            self.doubleSpinBox_4.blockSignals(False)

            logging.info(f"Adjusted histogram range to [{min_val}, {max_val}] because min was >= max")

        y, x = np.histogram(filtered_data, bins=num_bins, range=(min_val, max_val))

        # Create bar graph for histogram
        width = (x[1] - x[0])
        x_centers = (x[:-1] + x[1:]) / 2

        # Create histogram using BarGraphItem
        bargraph = pg.BarGraphItem(x=x_centers, height=y, width=width, brush='b', pen='k', alpha=0.7)
        self.plotWidget.addItem(bargraph)

        # Prepare data for GMM - reshape to 2D array required by sklearn
        data_for_gmm = filtered_data.reshape(-1, 1)

        # Determine number of Gaussians to fit
        if self.checkBox_auto_components.isChecked():
            # Auto-determine optimal number of components using settings from gmm_settings
            optimal_k, bic_scores = self.get_optimal_components(data_for_gmm)

            # Update the spinBox with the optimal value (without triggering update_histogram again)
            self.spinBox.blockSignals(True)
            self.spinBox.setValue(optimal_k)
            self.spinBox.blockSignals(False)

            # Log the BIC scores for debugging
            logging.info(f"BIC scores: {bic_scores}")
            logging.info(f"Optimal number of components: {optimal_k}")

            k = optimal_k
        else:
            # Use user-specified number of components
            k = int(self.spinBox.value())

        if k > 0:
            try:
                # Initialize and fit the GMM model using settings from gmm_settings
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=self.gmm_settings['covariance_type'],
                    random_state=self.gmm_settings['random_state'],
                    max_iter=self.gmm_settings['max_iter'],
                    n_init=self.gmm_settings['n_init'],
                    tol=self.gmm_settings['tol'],
                    reg_covar=self.gmm_settings['reg_covar']
                )
                gmm.fit(data_for_gmm)

                # Generate x values for the fitted curve
                x_fit = np.linspace(min_val, max_val, 200).reshape(-1, 1)

                # Get the probability density for each point
                y_fit_probs = np.exp(gmm.score_samples(x_fit))

                # Get the weighted components for individual Gaussians
                component_probs = []
                for i in range(k):
                    # Calculate the probability density for this component
                    mean = gmm.means_[i, 0]
                    var = gmm.covariances_[i, 0, 0]
                    weight = gmm.weights_[i]

                    # Calculate the Gaussian PDF for this component
                    y_gauss_i = weight * np.exp(-0.5 * ((x_fit.ravel() - mean) ** 2) / var) / np.sqrt(2 * np.pi * var)
                    component_probs.append(y_gauss_i)

                # Scale the GMM probabilities to match the histogram height
                scale_factor = max(y) / max(y_fit_probs) if max(y_fit_probs) > 0 else 1
                y_fit = y_fit_probs * scale_factor

                # Plot sum of all Gaussians
                self.plotWidget.plot(x_fit.ravel(), y_fit, pen=pg.mkPen('r', width=2), name='GMM Fit')

                # Plot individual Gaussian components
                for i in range(k):
                    # Scale the component to match the histogram
                    y_gauss_i = component_probs[i] * scale_factor

                    # Plot individual Gaussian
                    color = pg.intColor(i, hues=k)
                    self.plotWidget.plot(
                        x_fit.ravel(),
                        y_gauss_i,
                        pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine),
                        name=f'Gaussian {i + 1}'
                    )

                # Add legend
                self.plotWidget.addLegend()

                # Build a results table
                header = ["Gaussian #", "Weight", "Mean", "Std. Dev."]
                rows = []
                col_widths = [len(h) for h in header]

                for i in range(k):
                    weight = gmm.weights_[i]
                    mean = gmm.means_[i, 0]
                    std_dev = np.sqrt(gmm.covariances_[i, 0, 0])

                    weight_str = f"{weight:.3f}"
                    mean_str = f"{mean:.3f}"
                    std_dev_str = f"{std_dev:.3f}"

                    row = [f"{i + 1}", weight_str, mean_str, std_dev_str]
                    rows.append(row)
                    for j, item in enumerate(row):
                        col_widths[j] = max(col_widths[j], len(item))

                def build_format_string(widths):
                    return "  ".join(f"{{:<{w}}}" for w in widths)

                fmt = build_format_string(col_widths)
                table_lines = []
                table_lines.append(fmt.format(*header))
                total_width = sum(col_widths) + 2 * (len(col_widths) - 1)
                table_lines.append("-" * total_width)
                for row in rows:
                    table_lines.append(fmt.format(*row))

                table_str = "\n".join(table_lines)

                # Display table
                self.plainTextEdit.setPlainText(table_str)

            except Exception as e:
                err_msg = "GMM fitting failed: " + str(e)
                logging.info(err_msg)
                self.plainTextEdit.setPlainText(err_msg)

                # Additional debug info
                if hasattr(e, '__module__') and 'sklearn' in e.__module__:
                    logging.info(f"This appears to be a scikit-learn error. Check data format and GMM parameters.")
                    if len(filtered_data) < k:
                        logging.info(f"Not enough data points ({len(filtered_data)}) for {k} components. Try reducing the number of components.")
        else:
            self.plainTextEdit.clear()

    def setup_menubar(self):
        """Set up the menubar with settings options."""
        # Create menubar
        menubar = self.menuBar()

        # Create Settings menu
        settings_menu = menubar.addMenu('Settings')

        # Create View menu
        view_menu = menubar.addMenu('View')

        # Create UI Elements submenu
        ui_elements_menu = view_menu.addMenu('UI Elements')

        # Add actions for showing/hiding UI elements
        channel_selection_action = QtWidgets.QAction('Show Channel Selection', self)
        channel_selection_action.setCheckable(True)
        channel_selection_action.setChecked(self.show_channel_selection)  # Set based on initial parameter
        channel_selection_action.triggered.connect(self.toggle_channel_selection)
        ui_elements_menu.addAction(channel_selection_action)
        self.channel_selection_action = channel_selection_action

        clear_button_action = QtWidgets.QAction('Show Clear Button', self)
        clear_button_action.setCheckable(True)
        clear_button_action.setChecked(self.show_clear_button)  # Set based on initial parameter
        clear_button_action.triggered.connect(self.toggle_clear_button)
        ui_elements_menu.addAction(clear_button_action)
        self.clear_button_action = clear_button_action

        decay_button_action = QtWidgets.QAction('Show Decay Button', self)
        decay_button_action.setCheckable(True)
        decay_button_action.setChecked(self.show_decay_button)  # Set based on initial parameter
        decay_button_action.triggered.connect(self.toggle_decay_button)
        ui_elements_menu.addAction(decay_button_action)
        self.decay_button_action = decay_button_action

        filter_button_action = QtWidgets.QAction('Show Filter Button', self)
        filter_button_action.setCheckable(True)
        filter_button_action.setChecked(self.show_filter_button)  # Set based on initial parameter
        filter_button_action.triggered.connect(self.toggle_filter_button)
        ui_elements_menu.addAction(filter_button_action)
        self.filter_button_action = filter_button_action

        # Add action for showing channel settings
        channel_settings_action = QtWidgets.QAction('Channels', self)
        channel_settings_action.triggered.connect(self.show_channel_settings)
        settings_menu.addAction(channel_settings_action)

        # Add action for showing GMM settings
        gmm_settings_action = QtWidgets.QAction('GMM', self)
        gmm_settings_action.triggered.connect(self.show_gmm_settings)
        settings_menu.addAction(gmm_settings_action)

    def toggle_channel_selection(self, checked):
        """Toggle visibility of the channel selection group box."""
        if checked:
            self.burst_finder.groupBox_3.show()
        else:
            self.burst_finder.groupBox_3.hide()

    def toggle_clear_button(self, checked):
        """Toggle visibility of the clear button."""
        if checked:
            self.burst_finder.toolButton_6.show()
        else:
            self.burst_finder.toolButton_6.hide()

    def toggle_decay_button(self, checked):
        """Toggle visibility of the decay button."""
        if checked:
            self.burst_finder.toolButton_3.show()
        else:
            self.burst_finder.toolButton_3.hide()

    def toggle_filter_button(self, checked):
        """Toggle visibility of the filter button."""
        if checked:
            self.burst_finder.toolButton_4.show()
        else:
            self.burst_finder.toolButton_4.hide()

    def show_channel_settings(self):
        """Show the channel settings dialog and update burst_finder when closed."""
        result = self.channel_settings_dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            # Update burst_finder with the new settings
            self.burst_finder.windows = self.channel_definer.windows
            self.burst_finder.detectors = self.channel_definer.detectors
            # Update UI elements that depend on windows and detectors
            self.burst_finder.fill_detectors(self.channel_definer.detectors)
            self.burst_finder.fill_pie_windows(self.channel_definer.windows)

    def show_gmm_settings(self):
        """Show the GMM settings dialog and update GMM settings when closed."""
        dialog = GMMSettingsDialog(self, self.gmm_settings)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            # Update GMM settings with the new values
            self.gmm_settings = dialog.get_settings()
            # Update the histogram with the new settings
            self.update_histogram()

    def clear_data(self):
        """
        Clears all current data from the table, histogram, text fields,
        and resets the data frame.
        """
        self.current_df = None
        # clear data reader
        self.burst_finder.toolButton_6.click()

        # Clear the plot
        self.plotWidget.clear()

        # Clear the plain text area
        self.plainTextEdit.clear()

        # Optionally reset the progress bar if desired
        self.progressBar.setValue(0)

        logging.info("Data cleared.")
        
    def on_file_format_toggled(self, state):
        """
        Handle state changes of the checkBox_FileCSV checkbox.
        This method is kept for backward compatibility but no longer enforces mutual exclusivity.
        """
        # No longer enforcing mutual exclusivity
        pass
            
    def on_mfd_hdf_toggled(self, state):
        """
        Handle state changes of the checkBox_FileMFDHDF checkbox.
        This method is kept for backward compatibility but no longer enforces mutual exclusivity.
        """
        # No longer enforcing mutual exclusivity
        pass
