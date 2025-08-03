import typing
from pathlib import Path
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np

import tttrlib
import chisurf.gui.decorators
import chisurf.settings
from chisurf.gui.widgets.wizard.tttr_channel_definition import DetectorWizardPage

VERBOSE = False

SPECIAL_FILETYPES = {'.spc'}

class FileListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None, file_added_callback=None):
        super().__init__(parent)
        self.parent = parent
        self.setAcceptDrops(True)
        self.file_added_callback = file_added_callback  # Store callback function

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        # Clear the list when new files are dropped
        self.clear()

        file_type = self.parent.tttr_filetype
        if event.mimeData().hasUrls():
            file_paths = []
            special_files = []
            first_file_path = None

            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                path_obj = Path(file_path)

                # Handle directories - look for .bur and .bst files if this is the BID list widget
                if path_obj.is_dir() and hasattr(self.parent, 'listWidget_BID') and self == self.parent.listWidget_BID:
                    # Search for .bur files in the directory
                    bur_files = list(path_obj.glob("**/*.bur"))
                    # Search for .bst files in the directory
                    bst_files = list(path_obj.glob("**/*.bst"))
                    
                    # Combine both file types
                    burst_files = bur_files + bst_files
                    
                    if burst_files:
                        for burst_file in burst_files:
                            file_paths.append(str(burst_file))
                    else:
                        chisurf.logging.info(f"No .bur or .bst files found in directory: {file_path}")
                elif path_obj.is_file():
                    if first_file_path is None:
                        first_file_path = file_path

                    if file_path.endswith(".bst"):
                        suffix = path_obj.stem.rsplit('.', 1)[0]
                    else:
                        suffix = path_obj.suffix.lower()
                    if suffix.lower() in SPECIAL_FILETYPES and file_type == "Auto":
                        special_files.append(file_path)
                    else:
                        file_paths.append(file_path)

            # Do not infer file type from the first file
            # Container type must be set by user

            # Sort files lexically before adding them
            file_paths.sort()
            # Add sorted files
            for file_path in file_paths:
                self.add_file(file_path)
            if special_files:
                QtWidgets.QMessageBox.warning(
                    self, "File Type Requires Selection",
                    "The following files require manual file type selection:\n" + "\n".join(special_files)
                )
            # Call the callback function after dropping files
            if self.file_added_callback:
                self.file_added_callback()
            event.acceptProposedAction()
        else:
            event.ignore()

    def add_file(self, file_path: str):
        item = QtWidgets.QListWidgetItem(file_path, self)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        self.addItem(item)

        # Trigger callback if available
        if self.file_added_callback:
            self.file_added_callback()

    def get_selected_files(self) -> list[Path]:
        return [Path(self.item(i).text()) for i in range(self.count()) if self.item(i).checkState() == QtCore.Qt.Checked]


class MicrotimeHistogram(QtWidgets.QWidget):
    # Class variable to store the singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of MicrotimeHistogram.
        If no instance exists, create one.
        
        Returns:
            MicrotimeHistogram: The singleton instance
        """
        if cls._instance is None or not cls._instance.isVisible():
            cls._instance = MicrotimeHistogram()
        return cls._instance
    
    @property
    def binning_factor(self) -> int:
        return int(self.comboBox_2.currentText())

    @property
    def tttr_filetype(self) -> str | None:
        txt = self.comboBox.currentText()
        if txt == 'Auto':
            # Container type must be set by user, but we still need to handle 'Auto' option
            # Return None to let tttrlib try auto-detection
            return None
        return txt

    def _get_interleaved_channels(self) -> list[int]:
        """
        Get the interleaved channel list from detector wizard page or UI input.
        The channels are expected to be in interleaved format: [par1, perp1, par2, perp2, ...]
        
        Returns:
            list[int]: List of interleaved channel numbers
        """
        # First try to get channels from detector wizard page
        if hasattr(self, 'detector_wizard_page') and self.detector_wizard_page:
            # Try to get detector channels from the current detector
            detectors = self.detector_wizard_page.detectors
            if detectors:
                detector_name = self.current_detector_name or None
                if detector_name and detector_name in detectors:
                    detector = detectors[detector_name]
                    if isinstance(detector, dict) and 'chs' in detector:
                        return detector['chs']
            
            # If we couldn't get channels from the current detector, try the channels method
            channels = self.detector_wizard_page.channels()
            if channels:
                # Combine parallel and perpendicular channels in interleaved format
                if 'parallel' in channels and 'perpendicular' in channels:
                    par_chs = []
                    perp_chs = []
                    
                    for channel_info in channels['parallel']:
                        par_chs.extend(channel_info.get('detector_chs', []))
                    
                    for channel_info in channels['perpendicular']:
                        perp_chs.extend(channel_info.get('detector_chs', []))
                    
                    # Interleave the channels
                    interleaved = []
                    for i in range(max(len(par_chs), len(perp_chs))):
                        if i < len(par_chs):
                            interleaved.append(par_chs[i])
                        if i < len(perp_chs):
                            interleaved.append(perp_chs[i])
                    
                    if interleaved:
                        return interleaved
        
        # Fall back to the original method if detector wizard page doesn't provide channels
        try:
            par_str = self.lineEdit_2.text()
            perp_str = self.lineEdit.text()
            
            par_chs = [int(i) for i in par_str.split(",") if i.strip().isdigit()]
            perp_chs = [int(i) for i in perp_str.split(",") if i.strip().isdigit()]
            
            # Interleave the channels
            interleaved = []
            for i in range(max(len(par_chs), len(perp_chs))):
                if i < len(par_chs):
                    interleaved.append(par_chs[i])
                if i < len(perp_chs):
                    interleaved.append(perp_chs[i])
            
            return interleaved
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Channel input is not valid.")
            return []
    
    @property
    def parallel_channels(self) -> list[int]:
        """
        Get the parallel channels (even indices) from the interleaved channel list.
        
        Returns:
            list[int]: List of parallel channel numbers
        """
        interleaved = self._get_interleaved_channels()
        return interleaved[::2]  # Even indices (0, 2, 4, ...)

    @property
    def perpendicular_channels(self) -> list[int]:
        """
        Get the perpendicular channels (odd indices) from the interleaved channel list.
        
        Returns:
            list[int]: List of perpendicular channel numbers
        """
        interleaved = self._get_interleaved_channels()
        return interleaved[1::2]  # Odd indices (1, 3, 5, ...)
        
    @property
    def current_detector_name(self) -> str:
        """
        Get the current detector name from the detector selection combobox.
        
        Returns:
            str: The current detector name or empty string if not available
        """
        if hasattr(self, 'detector_selection_combobox') and self.detector_selection_combobox:
            return self.detector_selection_combobox.currentText()
        return ""
            
    def on_detector_selection_changed(self, index):
        """
        Handle changes in the detector selection combobox.
        Updates parallel/perpendicular channels and g-factor based on the selected detector.
        Clears the histogram as it's no longer valid for the new detector.
        
        Parameters:
        -----------
        index : int
            The index of the selected item in the combobox
        """
        if index < 0 or not self.detector_wizard_page:
            return
            
        # Clear the histogram plot as it's no longer valid for the new detector
        if hasattr(self, 'plotWidget'):
            self.plotWidget.clear()
            if hasattr(self, 'plotWidget') and hasattr(self.plotWidget, 'addLegend'):
                self.plotWidget.addLegend()  # Re-add legend after clearing
            chisurf.logging.info(f"Cleared histogram plot due to detector change to: {self.detector_selection_combobox.currentText()}")
        
        # Reset histogram data
        if hasattr(self, 'cumulative_ps'):
            self.cumulative_ps = None
        if hasattr(self, 'original_histograms'):
            self.original_histograms = {}
            
        # Get the selected detector name
        detector_name = self.current_detector_name
        if not detector_name:
            return
            
        # Get all detectors from the DetectorWizardPage
        detectors = self.detector_wizard_page.detectors
        if not detectors:
            return
            
        # Find the selected detector
        selected_detector = None
        
        # Check if detectors is a dictionary (expected) or something else
        if isinstance(detectors, dict):
            # If it's a dictionary with detector names as keys, access directly
            if detector_name in detectors:
                selected_detector = detectors[detector_name]
        else:
            # If it's a list, search through it
            for detector in detectors:
                if isinstance(detector, dict) and 'name' in detector and detector['name'] == detector_name:
                    selected_detector = detector
                    break
                elif hasattr(detector, 'get') and callable(detector.get):
                    try:
                        if detector.get('name') == detector_name:
                            selected_detector = detector
                            break
                    except (TypeError, AttributeError):
                        pass
                
        if not selected_detector:
            return
            
        # Safely get detector_chs based on object type
        detector_chs = []
        if isinstance(selected_detector, dict):
            detector_chs = selected_detector.get('chs', [])  # Use 'chs' key as shown in the example data
        elif hasattr(selected_detector, 'get') and callable(selected_detector.get):
            try:
                detector_chs = selected_detector.get('chs', [])  # Try 'chs' first
                if not detector_chs:
                    detector_chs = selected_detector.get('detector_chs', [])  # Fall back to 'detector_chs'
            except (TypeError, AttributeError):
                pass
        
        if detector_chs:
            # Extract parallel and perpendicular channels from interleaved format
            parallel_chs = detector_chs[::2]  # Even indices (0, 2, 4, ...)
            perpendicular_chs = detector_chs[1::2]  # Odd indices (1, 3, 5, ...)
            
            # Update UI with the extracted channels
            self.lineEdit_2.setText(", ".join(map(str, parallel_chs)))  # Update parallel channels
            self.lineEdit.setText(", ".join(map(str, perpendicular_chs)))  # Update perpendicular channels
            
            chisurf.logging.info(f"Updated channels from interleaved format: parallel={parallel_chs}, perpendicular={perpendicular_chs}")
                
        # Update g-factor if available - safely get g_factor based on object type
        g_factor = None
        if isinstance(selected_detector, dict):
            g_factor = selected_detector.get('g_factor')  # Use 'g_factor' key as shown in the example data
        elif hasattr(selected_detector, 'get') and callable(selected_detector.get):
            try:
                g_factor = selected_detector.get('g_factor')
            except (TypeError, AttributeError):
                pass
            
        if g_factor is not None:
            # Update both the attribute and the UI
            self.g_factor = g_factor
            # No need to update lineEdit_gfactor here as the setter does it

        # Log the detector selection for debugging
        chisurf.logging.info(f"Selected detector: {detector_name}, channels: {detector_chs}, g-factor: {g_factor}")
    
    def on_detectors_changed(self):
        """
        Handle changes in detector setup from the DetectorWizardPage.
        This method is called when the detectorsChanged signal is emitted.
        """
        # Get the current setup name
        setup_name = self.detector_wizard_page.setup_combo.currentText()
        if setup_name:
            chisurf.logging.info(f"Selected setup: {setup_name}")
            
            # Update the setup selection combobox to match
            if hasattr(self, 'setup_selection_combobox'):
                # Block signals to prevent triggering the change handler
                self.setup_selection_combobox.blockSignals(True)
                
                # Find and select the matching setup in the combobox
                index = self.setup_selection_combobox.findText(setup_name)
                if index >= 0:
                    self.setup_selection_combobox.setCurrentIndex(index)
                
                # Unblock signals
                self.setup_selection_combobox.blockSignals(False)
        
        # Update the filetype combobox based on the selected setup
        filetype = self.detector_wizard_page.filetype
        if filetype:
            # Find the index of the filetype in the combobox
            index = self.comboBox.findText(filetype)
            if index >= 0:
                self.comboBox.setCurrentIndex(index)
                chisurf.logging.info(f"Updated filetype to: {filetype}")
        
        # Update the binning combobox based on the selected setup
        binning = self.detector_wizard_page.micro_binning_combo.currentText()
        if binning:
            # Find the index of the binning in the comboBox_2
            index = self.comboBox_2.findText(binning)
            if index >= 0:
                self.comboBox_2.setCurrentIndex(index)
                chisurf.logging.info(f"Updated binning to: {binning}")
        
        # Update the dt field based on the selected setup
        effective_resolution = self.detector_wizard_page.effective_micro_time_resolution
        if effective_resolution:
            self.lineEdit_4.setText(f"{effective_resolution:.6f}")
            self.time_resolution = effective_resolution  # Update the time_resolution property
            chisurf.logging.info(f"Updated dt to: {effective_resolution:.6f} ns")
        
        # Populate the detector selection combobox
        self.detector_selection_combobox.clear()
        detectors = self.detector_wizard_page.detectors
        if detectors:
            detector_names = list(detectors.keys())
            if detector_names:
                self.detector_selection_combobox.addItems(detector_names)
                chisurf.logging.info(f"Populated detector selection combobox with: {detector_names}")
            else:
                chisurf.logging.warning("No valid detector names found")
        
        # Update the parallel and perpendicular channel inputs in the UI
        channels = self.detector_wizard_page.channels()
        
        if channels:
            # Extract parallel and perpendicular channels
            parallel_chs = []
            perpendicular_chs = []

            # Create interleaved channels list
            interleaved_chs = []
            for i in range(max(len(parallel_chs), len(perpendicular_chs))):
                if i < len(parallel_chs):
                    interleaved_chs.append(parallel_chs[i])
                if i < len(perpendicular_chs):
                    interleaved_chs.append(perpendicular_chs[i])
            
            # Update UI with the extracted channels
            if parallel_chs:
                self.lineEdit_2.setText(", ".join(map(str, parallel_chs)))
                chisurf.logging.info(f"Updated parallel channels to: {parallel_chs}")
            
            if perpendicular_chs:
                self.lineEdit.setText(", ".join(map(str, perpendicular_chs)))
                chisurf.logging.info(f"Updated perpendicular channels to: {perpendicular_chs}")
            
            chisurf.logging.info(f"Created interleaved channels: {interleaved_chs}")

    @property
    def timeshift_vv(self) -> int:
        return self.spinBox_timeshift_vv.value()

    @property
    def timeshift_vh(self) -> int:
        return self.spinBox_timeshift_vh.value()

    @property
    def g_factor(self) -> float:
        """Get the current g-factor value."""
        return self._g_factor
        
    @g_factor.setter
    def g_factor(self, value: float):
        """Set the g-factor value with validation."""
        # Convert to float and validate
        float_value = float(value)
        self._g_factor = float_value
        # Update the UI if needed
        if hasattr(self, 'lineEdit_gfactor'):
            self.lineEdit_gfactor.setText(f"{float_value:.6f}")

    @property
    def time_step(self) -> float:
        """Get the current time step (resolution) value in nanoseconds."""
        return self._time_step
        
    @time_step.setter
    def time_step(self, value: float):
        """Set the time step value with validation."""
        # Convert to float and validate
        float_value = float(value)
        if float_value <= 0:
            raise ValueError("Time step must be positive")
        self._time_step = float_value
        # Update the UI if needed
        if hasattr(self, 'lineEdit_4'):
            self.lineEdit_4.setText(f"{float_value:.6f}")
        # Update time_resolution for backward compatibility
        self.time_resolution = float_value

    @property
    def selected_files(self):
        return self.listWidget.get_selected_files()

    @chisurf.gui.decorators.init_with_ui("microtime_histogram/wizard.ui", path=chisurf.settings.plugin_path)
    def __init__(self, *args, **kwargs):
        self._tttr = None
        self.tttr_folder = None  # Store the folder where TTTR files are found
        self._g_factor = 1.000000  # Default g-factor value
        self._time_step = 1.0  # Default time step value in nanoseconds

        # Create the detector setup tab
        self.detector_setup_tab = QtWidgets.QWidget()
        self.detector_setup_layout = QtWidgets.QVBoxLayout(self.detector_setup_tab)
        
        # Add DetectorWizardPage to the detector setup tab
        self.detector_wizard_page = DetectorWizardPage(
            show_edit_json=True,
            show_save=True,
            show_setups_file=True,
            show_setup_selection=True,
            show_help=True,
            show_tttr_reading=True,
            show_tables=True,
            show_add_inputs=True
        )
        self.detector_setup_layout.addWidget(self.detector_wizard_page)
        
        # Add the detector setup tab to the tab widget
        self.tabWidget.addTab(self.detector_setup_tab, "Detector Setup")
        
        # Find the UI elements that are now defined in the UI file
        self.setup_info_label = self.findChild(QtWidgets.QLabel, "setup_info_label")
        self.detector_selection_combobox = self.findChild(QtWidgets.QComboBox, "detector_selection_combobox")
        
        # Find the setup selection combobox in the UI
        self.setup_selection_combobox = self.findChild(QtWidgets.QComboBox, "setup_selection_combobox")
        
        # Initialize the original UI elements
        self.comboBox.clear()
        self.comboBox.addItems(['Auto'] + list(tttrlib.TTTR.get_supported_container_names()))
        self.comboBox.setEnabled(False)  # Disable the filetype combobox as requested

        # Initialize storage for original histograms
        self.original_histograms = {}
        self.time_resolution = 1.0

        self.listWidget = FileListWidget(parent=self, file_added_callback=self.update_micro_time_resolution)
        self.verticalLayout_3.addWidget(self.listWidget)

        self.listWidget_BID = FileListWidget(parent=self, file_added_callback=self.load_corresponding_tttr_files)
        self.verticalLayout_5.addWidget(self.listWidget_BID)

        self.plotWidget = pg.PlotWidget()
        self.verticalLayout.addWidget(self.plotWidget)
        self.plotWidget.setLabel('bottom', 'Micro Time (ns)')
        self.plotWidget.setLabel('left', 'Counts')
        self.plotWidget.setLogMode(y=True)
        self.plotWidget.addLegend()

        self.populate_supported_types()
        self.toolButton.clicked.connect(self.browse_and_open_input_files)
        self.toolButton_2.clicked.connect(self.clear_files)
        self.pushButton.clicked.connect(self.compute_microtime_histogram)
        self.comboBox_2.currentIndexChanged.connect(self.update_micro_time_resolution)
        self.pushButton_2.clicked.connect(self.open_save_dialog)
        self.transferButton.clicked.connect(self.on_transfer_clicked)

        self.lineEdit_2.textChanged.connect(self.update_output_filename)  # Parallel channels
        self.lineEdit.textChanged.connect(self.update_output_filename)  # Perpendicular channels

        # Connect timeshift input fields to update method
        self.spinBox_timeshift_vv.valueChanged.connect(self.update_timeshifts)
        self.spinBox_timeshift_vh.valueChanged.connect(self.update_timeshifts)

        # Connect G-factor input field to update g_factor attribute and update method
        self.lineEdit_gfactor.textChanged.connect(self.on_gfactor_changed)
        
        # Connect time step input field to update time_step attribute
        self.lineEdit_4.textChanged.connect(self.on_time_step_changed)
        
        # Connect detector wizard page signals
        self.detector_wizard_page.detectorsChanged.connect(self.on_detectors_changed)
        
        # Connect detector selection combobox
        self.detector_selection_combobox.currentIndexChanged.connect(self.on_detector_selection_changed)
        
        # Populate and connect setup selection combobox
        self.populate_setup_selection_combobox()
        self.setup_selection_combobox.currentIndexChanged.connect(self.on_setup_selection_changed)
        
        # Manually populate detector_selection_combobox after initialization
        self.on_detectors_changed()

    def populate_setup_selection_combobox(self):
        """
        Populate the setup selection combobox with available setups from the detector wizard page.
        """
        if not hasattr(self, 'setup_selection_combobox') or not self.detector_wizard_page:
            return
            
        # Block signals to prevent triggering the change handler while populating
        self.setup_selection_combobox.blockSignals(True)
        self.setup_selection_combobox.clear()
        
        # Get the current setups file from the detector wizard page
        setups_file = self.detector_wizard_page.current_setups_file
        
        # Load setups from the file
        from chisurf.gui.widgets.wizard.tttr_channel_definition import load_detector_setups
        setups = load_detector_setups(setups_file)
        
        # Add all setup names to the combobox
        for setup_name in setups.get("setups", {}).keys():
            self.setup_selection_combobox.addItem(setup_name)
            
        # If there's a current setup in the detector wizard page, select it
        current_setup = self.detector_wizard_page.current_setup_name
        if current_setup:
            index = self.setup_selection_combobox.findText(current_setup)
            if index >= 0:
                self.setup_selection_combobox.setCurrentIndex(index)
                
        # Unblock signals
        self.setup_selection_combobox.blockSignals(False)
        
        chisurf.logging.info(f"Populated setup selection combobox with {self.setup_selection_combobox.count()} setups")
    
    def on_gfactor_changed(self, text):
        """
        Handle changes in the g-factor input field.
        Updates the g_factor attribute and calls update_timeshifts.
        
        Parameters:
        -----------
        text : str
            The new text in the g-factor input field
        """
        # Try to convert the text to a float
        value = float(text)
        # Update the g_factor attribute (using the setter)
        self._g_factor = value
        # Update timeshifts with the new g-factor
        self.update_timeshifts()

    def on_time_step_changed(self, text):
        """
        Handle changes in the time step input field.
        Updates the time_step attribute.
        
        Parameters:
        -----------
        text : str
            The new text in the time step input field
        """
        # Try to convert the text to a float
        value = float(text)
        # Update the time_step attribute (using the setter)
        self.time_step = value

    def on_setup_selection_changed(self, index):
        """
        Handle changes in the setup selection combobox.
        Updates the detector wizard page with the selected setup.
        
        Parameters:
        -----------
        index : int
            The index of the selected item in the combobox
        """
        if index < 0 or not self.detector_wizard_page:
            return
            
        # Get the selected setup name
        setup_name = self.setup_selection_combobox.currentText()
        if not setup_name:
            return
            
        # Update the setup in the detector wizard page
        # Find the index of the setup in the detector wizard page's combobox
        wizard_index = self.detector_wizard_page.setup_combo.findText(setup_name)
        if wizard_index >= 0:
            # Set the current index in the detector wizard page's combobox
            # This will trigger the _on_setup_changed method in the detector wizard page
            self.detector_wizard_page.setup_combo.setCurrentIndex(wizard_index)
            
            chisurf.logging.info(f"Selected setup: {setup_name}")
    
    def load_corresponding_tttr_files(self, n_parent=4):
        """
        Search for TTTR files in the folder structure above the selected BID/BUR files up to n_parent levels.
        This method handles both individual BID files and .bur files from burstwise folders.
        
        Optimization: Since all TTTR files are typically in the same folder, we only search for the
        folder location once (for the first file) and then use that folder for all subsequent files.
        The folder is stored as a class attribute for use in saving output files.
        """
        bid_files = self.listWidget_BID.get_selected_files()
        tttr_files = set()
        local_tttr_folder = None  # Local variable for processing

        for i, bid_file in enumerate(bid_files):
            # Handle different file extensions
            if bid_file.suffix.lower() in ['.bur']:
                # For .bur files, extract the base name (removing _X suffix if present)
                base_name = bid_file.stem
                # If the filename has a pattern like 'name_X', extract just 'name'
                if '_' in base_name:
                    parts = base_name.split('_')
                    if len(parts) > 1 and parts[-1].isdigit():
                        base_name = '_'.join(parts[:-1])
            else:
                # For other files (like .bst), use the stem directly
                base_name = bid_file.stem

            # For the first file or if we haven't found a TTTR folder yet, search for it
            if i == 0 or local_tttr_folder is None:
                # Start searching from the parent folder
                parent_folder = bid_file.parent
                level = 0
                
                # Search up to n_parent levels up in the directory structure
                while parent_folder != parent_folder.root and level < n_parent:
                    # Look for files that match the base name
                    matching_tttr_files = list(parent_folder.glob(f"{base_name}*"))
                    # Filter out .bur files and other non-TTTR files
                    matching_tttr_files = [f for f in matching_tttr_files if f.suffix.lower() not in ['.bur', '.bst']]

                    if matching_tttr_files:
                        # Store the folder where TTTR files were found
                        local_tttr_folder = parent_folder
                        self.tttr_folder = parent_folder  # Store as class attribute for saving output
                        # Add all matching files to the set
                        for tttr_file in matching_tttr_files:
                            tttr_files.add(tttr_file)
                        break  # Stop searching once matches are found

                    # Move up to the parent directory
                    parent_folder = parent_folder.parent
                    level += 1
            else:
                # For subsequent files, use the already found TTTR folder
                if local_tttr_folder is not None:
                    # Look for files that match the base name in the known TTTR folder
                    matching_tttr_files = list(local_tttr_folder.glob(f"{base_name}*"))
                    # Filter out .bur files and other non-TTTR files
                    matching_tttr_files = [f for f in matching_tttr_files if f.suffix.lower() not in ['.bur', '.bst']]
                    
                    # Add all matching files to the set
                    for tttr_file in matching_tttr_files:
                        tttr_files.add(tttr_file)

        # Clear existing TTTR list and add found files
        self.listWidget.clear()
        for tttr_file in sorted(tttr_files):
            self.listWidget.add_file(tttr_file.as_posix())

        self.update_output_filename()

        chisurf.logging.info(f"Loaded TTTR files: {[f.as_posix() for f in tttr_files]}")

    def save_cumulative_histogram(self, file_path):
        """Save the cumulative histogram to a text file as a single-column integer list.
        If no histogram exists, compute it first.
        """
        # If cumulative_ps is not available but we have original histograms, update timeshifts
        if (not hasattr(self, "cumulative_ps") or self.cumulative_ps is None) and self.original_histograms:
            self.update_timeshifts()  # Update timeshifts to generate cumulative_ps
        # If we still don't have cumulative_ps, compute the full histogram
        elif not hasattr(self, "cumulative_ps") or self.cumulative_ps is None:
            self.compute_microtime_histogram()  # Compute histogram first

        # Check again after computation
        if self.cumulative_ps is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Failed to compute cumulative histogram before saving.")
            return

        try:
            # Convert file_path to Path object if it's a string
            path_obj = Path(file_path) if isinstance(file_path, str) else file_path
            
            # Ensure the directory exists
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if the path is just a directory or has no parent
            if path_obj.name == '.' or path_obj.name == '' or str(path_obj) == '.':
                # Invalid path, use a default filename in the current directory
                path_obj = Path.cwd() / "microtime_histogram.dat"
                chisurf.logging.warning(f"Invalid save path. Using default: {path_obj}")
            
            # Ensure data is saved as integers
            np.savetxt(str(path_obj), self.cumulative_ps.astype(int), fmt="%d")

            # Log success instead of showing a message box
            chisurf.logging.info(f"Histogram successfully saved to: {path_obj}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"An error occurred while saving:\n{str(e)}")
            chisurf.logging.error(f"Failed to save histogram: {str(e)}")

    def on_transfer_clicked(self):
        """
        Handle the "Transfer to ChiSurf" button click event.
        Transfers the computed decay to ChiSurf when clicked.
        """
        self.add_to_chisurf()

    def add_to_chisurf(self):
        """
        Add the computed microtime histogram to chisurf as a dataset.
        This method uses the standard loading approach by loading the saved histogram file.
        """
        chisurf.logging.info("MicrotimeHistogram::adding histogram to chisurf")

        # If cumulative_ps is not available but we have original histograms, update timeshifts
        if (not hasattr(self, "cumulative_ps") or self.cumulative_ps is None) and self.original_histograms:
            self.update_timeshifts()  # Update timeshifts to generate cumulative_ps
        # If we still don't have cumulative_ps, compute the full histogram
        elif not hasattr(self, "cumulative_ps") or self.cumulative_ps is None:
            self.compute_microtime_histogram()  # Compute histogram first

        # Check again after computation
        if self.cumulative_ps is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Failed to compute microtime histogram.")
            return

        # Ensure we have selected files
        if not self.selected_files:
            QtWidgets.QMessageBox.warning(self, "No Files", "Please select a file before adding to ChiSurf.")
            return

        # Get the full save path directly from lineEdit_5
        save_path = Path(self.lineEdit_5.text())

        # Ensure the histogram file exists
        if not save_path.exists():
            # Save the histogram file if it doesn't exist
            self.save_cumulative_histogram(str(save_path))

        if not save_path.exists():
            # Display a message box to the user if file still doesn't exist
            QtWidgets.QMessageBox.warning(
                self, 
                "No Histogram File", 
                "No histogram file available. Please save histogram data before adding to ChiSurf."
            )
            return

        from chisurf import cs

        # Get polarization from UI and g-factor from attribute
        polarization = self.comboBox_polarization.currentText()
        # Use the g_factor attribute directly
        g_factor = self.g_factor

        # Set the current experiment to TCSPC
        cs.current_experiment = 'TCSPC'
        cs.current_setup.is_jordi = True
        cs.current_setup.use_header = False
        cs.current_setup.matrix_columns = []
        cs.current_setup.g_factor = g_factor
        cs.current_setup.polarization = polarization
        cs.current_setup.rep_rate = 10.0
        cs.current_setup.rebin = (1, 1)
        cs.current_setup.dt = self.time_step

        # Add dataset to chisurf using the standard approach
        chisurf.macros.add_dataset(filename=str(save_path))

        # Show success message
        chisurf.logging.info(f"Added microtime histogram to ChiSurf: {save_path.name}")

        # Show a success message to the user
        QtWidgets.QMessageBox.information(self, "Success", "Microtime histogram added to ChiSurf successfully.")

    def open_save_dialog(self):
        """Open a save dialog using the full path from lineEdit_5 and save the cumulative histogram."""
        if not self.selected_files:
            QtWidgets.QMessageBox.warning(self, "No Files", "Please select a file before saving.")
            return

        # Get the full path from lineEdit_5
        default_path = self.lineEdit_5.text()
        
        # Open the save file dialog with the full path
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", default_path, "Data Files (*.dat);;All Files (*)"
        )

        if save_path:
            self.save_cumulative_histogram(save_path)

    def open_load_dialog(self):
        """Open a load dialog to select a saved jordi data file and load it into ChiSurf."""
        # Open the load file dialog
        load_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Jordi Data", "", "Data Files (*.dat);;All Files (*)"
        )

        if load_path:
            self.load_jordi_data(load_path)

    def populate_supported_types(self):
        self.comboBox.clear()
        self.comboBox.insertItem(0, "Auto")
        self.comboBox.insertItems(1, list(tttrlib.TTTR.get_supported_container_names()))

    def clear_files(self):
        """
        Clear all files and plots.
        """
        chisurf.logging.info("Clearing files and plot")
        self.listWidget.clear()  # Clear file list
        self.plotWidget.clear()  # Clear plot
        self.plotWidget.addLegend()  # Re-add legend after clearing
        self.listWidget_BID.clear() # Clear bid files
        # Removed: self.comboBox.setCurrentIndex(0)  # Reset combobox to "Auto"
        
    def load_bid_folder(self, folder_path, setup_name=None, auto_transfer=False):
        """
        Load a BID folder, clear existing data, and process the new folder.
        
        Args:
            folder_path (str): Path to the folder containing BID/BUR/BST files
            setup_name (str, optional): Name of the setup to use
            auto_transfer (bool, optional): Whether to automatically transfer to ChiSurf
        """
        chisurf.logging.info(f"Loading BID folder: {folder_path}")
        
        # Clear existing files and plots
        self.clear_files()
        
        # Convert to Path object
        folder_path = Path(folder_path)
        
        # If setup name is provided, select it
        if setup_name:
            # Find the index of the setup in the combobox
            index = self.setup_selection_combobox.findText(setup_name)
            if index >= 0:
                self.setup_selection_combobox.setCurrentIndex(index)
                chisurf.logging.info(f"Selected setup: {setup_name}")
            else:
                chisurf.logging.warning(f"Setup '{setup_name}' not found in available setups")
        
        # Find all .bst files in the BID folder
        bst_files = list(folder_path.glob("*.bst"))
        if not bst_files:
            chisurf.logging.warning(f"No .bst files found in {folder_path}")
        
        # Add the files to the widget
        for bst_file in bst_files:
            self.listWidget_BID.add_file(str(bst_file))
        
        # Compute the microtime histogram
        self.compute_microtime_histogram()
        
        # If auto-transfer is enabled, transfer the histogram to ChiSurf
        if auto_transfer:
            self.add_to_chisurf()

    @staticmethod
    def optimize_filename(filename):
        """
        Optimize the filename by stripping numbered suffixes like '_000'.
        
        Args:
            filename (str): The original filename (without extension)
            
        Returns:
            str: The optimized filename
        """
        import re
        
        # Pattern to match numbered suffixes with exactly 2, 3, or 4 digits
        pattern = r'_\d{2,4}$'
        
        # Check if the filename has a numbered suffix
        if re.search(pattern, filename):
            # Remove the numbered suffix
            optimized = re.sub(pattern, '', filename)
            return optimized
        
        return filename
        
    def update_output_filename(self):
        """Update the output filename with full path based on selected files, channel numbers, and detector name."""
        try:
            # Get the first file name without extension
            original_stem = Path(self.selected_files[0]).stem
            # Optimize the filename by stripping numbered suffixes
            out = self.optimize_filename(original_stem)
            chisurf.logging.info(f"Optimized filename: {original_stem} -> {out}")
            save_directory = Path(self.selected_files[0]).parent
        except IndexError:
            out = "output"
            save_directory = Path.cwd()

        # Get the parallel (p) and perpendicular (s) channel numbers
        p_channels = ",".join(map(str, self.parallel_channels)) if self.parallel_channels else "all"
        s_channels = ",".join(map(str, self.perpendicular_channels)) if self.perpendicular_channels else "all"
        
        detector_name = self.current_detector_name

        # Construct filename in format: filename_detector_(p)-(s).dat
        if detector_name:
            output_filename = f"{out}_{detector_name}_({p_channels})-({s_channels}).dat"
        else:
            output_filename = f"{out}_({p_channels})-({s_channels}).dat"

        # Check if BID/BUR files are being used
        bid_files = self.listWidget_BID.get_selected_files()
        if len(bid_files) > 0:
            # Use the directory of the first BID/BUR file
            bid_directory = Path(bid_files[0]).parent
            save_directory = bid_directory
            chisurf.logging.info(f"Using BID/BUR file folder for saving: {save_directory}")
        # If no BID/BUR files, use TTTR folder if available
        elif self.tttr_folder is not None:
            save_directory = self.tttr_folder
        else:
            # Get the directory of the first selected file as fallback
            chisurf.logging.warning("Neither BID/BUR files nor TTTR folder found, using selected file directory for output path.")

        # Create full path and set it in lineEdit_5
        try:
            # Ensure the directory exists
            save_directory.mkdir(parents=True, exist_ok=True)
            
            # Create full path
            full_path = save_directory / output_filename
            
            # Validate the path
            if not save_directory.is_dir():
                # If save_directory is not a valid directory, use current directory
                full_path = Path.cwd() / output_filename
                chisurf.logging.warning(f"Invalid save directory. Using current directory: {full_path}")
                
            self.lineEdit_5.setText(str(full_path))
            chisurf.logging.info(f"Output filename set to: {full_path}")
        except Exception as e:
            # If there's any error, use a default path in the current directory
            default_path = Path.cwd() / output_filename
            self.lineEdit_5.setText(str(default_path))
            chisurf.logging.warning(f"Error setting output path: {str(e)}. Using default: {default_path}")

    def update_micro_time_resolution(self):
        """Update micro time resolution and output filename when files or binning change."""
        bf = int(self.comboBox_2.currentText())
        self.update_output_filename()
        if self.selected_files:
            # If files are loaded, get micro_time_resolution from the first file
            path = Path(self.selected_files[0])  # Use the first file
            if path.is_file():
                # For SPC files, use resolution from DetectorWizard
                if path.suffix.lower() in SPECIAL_FILETYPES:
                    if hasattr(self, 'detector_wizard_page') and self.detector_wizard_page:
                        try:
                            # Get the micro_time_resolution in picoseconds from the detector_wizard_page
                            micro_time_resolution = float(self.detector_wizard_page.micro_time_le.text())
                            # Calculate the effective resolution with our current binning
                            binned_micro_time_resolution = micro_time_resolution * bf  # ns
                            self.time_step = binned_micro_time_resolution  # Update time_step attribute
                            chisurf.logging.info(f"Updated dt to: {binned_micro_time_resolution:.6f} ns for SPC file based on DetectorWizard")
                        except (ValueError, TypeError, AttributeError) as e:
                            chisurf.logging.warning(f"Failed to update dt for SPC file: {str(e)}")
                    else:
                        chisurf.logging.warning(f"Cannot update micro_time_resolution for SPC file: {path} - DetectorWizard not available")
                    return
                
                # For non-SPC files, use tttr.header
                t = tttrlib.TTTR(path.as_posix(), self.tttr_filetype)
                micro_time_resolution = t.header.micro_time_resolution
                binned_micro_time_resolution = micro_time_resolution * bf * 1e9
                self.time_step = binned_micro_time_resolution  # Update time_step attribute
        elif hasattr(self, 'detector_wizard_page') and self.detector_wizard_page:
            # If no files are loaded but we have a detector_wizard_page, calculate dt based on current binning
            # Get the base micro_time_resolution from detector_wizard_page
            try:
                # Get the micro_time_resolution in picoseconds from the detector_wizard_page
                micro_time_resolution = float(self.detector_wizard_page.micro_time_le.text())
                # Calculate the effective resolution with our current binning (not the one from detector_wizard_page)
                binned_micro_time_resolution = micro_time_resolution * bf # ns
                self.time_step = binned_micro_time_resolution  # Update time_step attribute
                chisurf.logging.info(f"Updated dt to: {binned_micro_time_resolution:.6f} ns based on binning change")
            except (ValueError, TypeError, AttributeError) as e:
                chisurf.logging.warning(f"Failed to update dt: {str(e)}")

    @staticmethod
    def load_bid_ranges(bid_file):
        """
        Load photon start-stop pairs from BID/BUR files.
        BID files are simple tab-separated files with start-stop pairs.
        BUR files have a header row and additional columns.
        """
        bid_ranges = []
        with open(bid_file, 'r') as f:
            # Check if this is a BUR file (has header row)
            first_line = f.readline().strip()
            if first_line.startswith('First File') or first_line.startswith('First Photon'):
                # This is a BUR file with headers
                # Skip the second line (units/description)
                f.readline()
                # Process the rest of the file
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        # BUR files have "First Photon" and "Last Photon" columns
                        # The exact column indices may vary, so we'll try to find them
                        try:
                            # Try to find columns by parsing all numeric values
                            numeric_values = [int(p) for p in parts if p.strip().isdigit()]
                            if len(numeric_values) >= 2:
                                start, stop = numeric_values[0], numeric_values[1]
                                bid_ranges.append((start, stop))
                        except (ValueError, IndexError):
                            # Skip lines that can't be parsed
                            continue
            else:
                # This is a simple BID file
                # Process the first line (which we've already read)
                parts = first_line.split('\t')
                if len(parts) == 2:
                    try:
                        start, stop = map(int, parts)
                        bid_ranges.append((start, stop))
                    except ValueError:
                        # Skip if not integers
                        pass

                # Process the rest of the file
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        try:
                            start, stop = map(int, parts)
                            bid_ranges.append((start, stop))
                        except ValueError:
                            # Skip if not integers
                            continue

        return bid_ranges

    def get_burst_photon_mask(self, tttr, burst_ranges):
        """
        Create a boolean mask of all photons that are part of any burst.
        This is similar to the get_burst_indices_for_current_file method in the MLE plugin.

        Parameters:
        -----------
        tttr : tttrlib.TTTR
            The TTTR object containing all photons
        burst_ranges : list of tuples
            List of (start, stop) tuples representing burst ranges

        Returns:
        --------
        numpy.ndarray
            Boolean mask of photons that are part of any burst
        """
        if not burst_ranges:
            return None

        # Extract start and stop indices
        starts = np.array([start for start, _ in burst_ranges], dtype=np.int32)
        stops = np.array([stop for _, stop in burst_ranges], dtype=np.int32)

        # Build a single "difference" event array with bincount
        # - at each start index we +1, at each (stop+1) we -1
        idxs = np.concatenate([starts, stops + 1])
        weights = np.concatenate([
            np.ones_like(starts, dtype=np.int32),
            -np.ones_like(stops + 1, dtype=np.int32),
        ])

        # Find the maximum index to ensure our bincount covers all photons
        max_len = idxs.max() + 1

        # Create events array using bincount
        events = np.bincount(idxs, weights, minlength=max_len)

        # Cumulative sum >0 gives a boolean mask of covered photons
        # This identifies all photons that are part of any burst
        coverage = np.cumsum(events)[:-1] > 0

        return coverage

    def calculate_fwhm(self, histogram):
        """
        Calculate the Full Width at Half Maximum (FWHM) of a histogram.

        Parameters:
        -----------
        histogram : numpy.ndarray
            The histogram data

        Returns:
        --------
        float
            The FWHM value in channel units
        """
        if histogram is None or len(histogram) == 0:
            return 0.0

        # Find the maximum value and its position
        max_value = np.max(histogram)
        max_pos = np.argmax(histogram)

        # Find the half maximum value
        half_max = max_value / 2.0

        # Find the left and right positions where the histogram crosses half_max
        left_idx = np.where(histogram[:max_pos] <= half_max)[0]
        left_pos = left_idx[-1] if len(left_idx) > 0 else 0

        right_idx = np.where(histogram[max_pos:] <= half_max)[0]
        right_pos = max_pos + right_idx[0] if len(right_idx) > 0 else len(histogram) - 1

        # Calculate FWHM
        fwhm = right_pos - left_pos

        return fwhm

    def update_timeshifts(self):
        """
        Update the histograms with the current timeshift values without recomputing the entire histogram.
        This method is called when the timeshift values are changed.
        """
        # Skip if no original histograms are available
        if not self.original_histograms:
            return

        chisurf.logging.info("Updating timeshifts...")
        self.plotWidget.clear()  # Clear plot before drawing new data
        self.plotWidget.addLegend()  # Re-add legend after clearing

        # Get the current timeshift values
        vv_shift = self.timeshift_vv
        vh_shift = self.timeshift_vh

        # Apply timeshifts to the original histograms
        shifted_histograms = {}
        for key, histograms in self.original_histograms.items():
            y_parallel_orig = histograms['parallel']
            y_perpendicular_orig = histograms['perpendicular']

            # Apply timeshift to parallel channel (VV) if needed
            if vv_shift != 0:
                if vv_shift > 0:
                    # Shift right (positive timeshift)
                    y_parallel = np.pad(y_parallel_orig, (vv_shift, 0), 'constant')[:-vv_shift]
                else:
                    # Shift left (negative timeshift)
                    y_parallel = np.pad(y_parallel_orig, (0, abs(vv_shift)), 'constant')[abs(vv_shift):]
            else:
                y_parallel = y_parallel_orig.copy()

            # Apply timeshift to perpendicular channel (VH) if needed
            if vh_shift != 0:
                if vh_shift > 0:
                    # Shift right (positive timeshift)
                    y_perpendicular = np.pad(y_perpendicular_orig, (vh_shift, 0), 'constant')[:-vh_shift]
                else:
                    # Shift left (negative timeshift)
                    y_perpendicular = np.pad(y_perpendicular_orig, (0, abs(vh_shift)), 'constant')[abs(vh_shift):]
            else:
                y_perpendicular = y_perpendicular_orig.copy()

            shifted_histograms[key] = {
                'parallel': y_parallel,
                'perpendicular': y_perpendicular
            }

            # We no longer plot individual files, only store them for cumulative plotting
            # Create x-axis arrays for both channels (for debugging purposes only)
            x_parallel = np.arange(len(y_parallel)) * self.time_resolution
            x_perpendicular = np.arange(len(y_perpendicular)) * self.time_resolution

        # Compute cumulative histograms for parallel and perpendicular channels separately
        cumulative_parallel = None
        cumulative_perpendicular = None
        
        for key, histograms in shifted_histograms.items():
            y_parallel = histograms['parallel']
            y_perpendicular = histograms['perpendicular']

            # Add to cumulative parallel histogram
            if cumulative_parallel is None:
                cumulative_parallel = np.array(y_parallel)
            else:
                try:
                    # Ensure arrays have the same shape
                    if len(cumulative_parallel) != len(y_parallel):
                        # Resize the smaller array to match the larger one
                        if len(cumulative_parallel) < len(y_parallel):
                            cumulative_parallel = np.pad(cumulative_parallel, (0, len(y_parallel) - len(cumulative_parallel)), 'constant')
                        else:
                            y_parallel = np.pad(y_parallel, (0, len(cumulative_parallel) - len(y_parallel)), 'constant')
                    cumulative_parallel += np.array(y_parallel)
                except ValueError as e:
                    chisurf.logging.error(f"Failed to add cumulative parallel histogram: {str(e)}")
            
            # Add to cumulative perpendicular histogram
            if cumulative_perpendicular is None:
                cumulative_perpendicular = np.array(y_perpendicular)
            else:
                try:
                    # Ensure arrays have the same shape
                    if len(cumulative_perpendicular) != len(y_perpendicular):
                        # Resize the smaller array to match the larger one
                        if len(cumulative_perpendicular) < len(y_perpendicular):
                            cumulative_perpendicular = np.pad(cumulative_perpendicular, (0, len(y_perpendicular) - len(cumulative_perpendicular)), 'constant')
                        else:
                            y_perpendicular = np.pad(y_perpendicular, (0, len(cumulative_perpendicular) - len(y_perpendicular)), 'constant')
                    cumulative_perpendicular += np.array(y_perpendicular)
                except ValueError as e:
                    chisurf.logging.error(f"Failed to add cumulative perpendicular histogram: {str(e)}")

        # Create combined cumulative histogram for backward compatibility
        if cumulative_parallel is not None and cumulative_perpendicular is not None:
            # Make sure both arrays have the same length for hstack
            max_len = max(len(cumulative_parallel), len(cumulative_perpendicular))
            parallel_padded = np.pad(cumulative_parallel, (0, max(0, max_len - len(cumulative_parallel))), 'constant')
            perpendicular_padded = np.pad(cumulative_perpendicular, (0, max(0, max_len - len(cumulative_perpendicular))), 'constant')
            self.cumulative_ps = np.hstack((parallel_padded, perpendicular_padded))
        else:
            self.cumulative_ps = None

        # Plot cumulative data if available
        if cumulative_parallel is not None and cumulative_perpendicular is not None:
            # Create x-axis with proper time units
            x_parallel = np.arange(len(cumulative_parallel)) * self.time_resolution
            x_perpendicular = np.arange(len(cumulative_perpendicular)) * self.time_resolution
            
            # Plot cumulative parallel and perpendicular data
            self.plotWidget.plot(x_parallel, cumulative_parallel, pen='r', name="Cumulative Parallel (VV)")
            self.plotWidget.plot(x_perpendicular, cumulative_perpendicular, pen='g', name="Cumulative Perpendicular (VH)")

            # Calculate and display FWHM of VV + 2G*VH
            try:
                # Use the g_factor attribute
                g_factor = self.g_factor

                # Create combined histogram VV + 2G*VH using cumulative data
                # Make sure both arrays have the same length
                max_len = max(len(cumulative_parallel), len(cumulative_perpendicular))
                parallel_padded = np.pad(cumulative_parallel, (0, max(0, max_len - len(cumulative_parallel))), 'constant')
                perpendicular_padded = np.pad(cumulative_perpendicular, (0, max(0, max_len - len(cumulative_perpendicular))), 'constant')

                # Apply formula VV + 2G*VH
                combined_histogram = parallel_padded + 2 * g_factor * perpendicular_padded

                # Calculate FWHM
                fwhm_channels = self.calculate_fwhm(combined_histogram)
                fwhm_ns = fwhm_channels * self.time_resolution

                # Display FWHM in the UI
                self.lineEdit_fwhm.setText(f"{fwhm_ns:.2f} ns ({fwhm_channels:.1f} channels)")

                # Plot the combined histogram
                x_combined = np.arange(len(combined_histogram)) * self.time_resolution
                self.plotWidget.plot(x_combined, combined_histogram, pen='y', name="Combined (VV + 2G*VH)")

            except Exception as e:
                chisurf.logging.error(f"Failed to calculate FWHM: {str(e)}")
                self.lineEdit_fwhm.setText("Error")

    def compute_microtime_histogram(self):
        chisurf.logging.info("Computing microtime histogram...")
        self.plotWidget.clear()  # Clear plot before drawing new data
        self.plotWidget.addLegend()  # Re-add legend after clearing
        self.cumulative_ps = None  # Reset cumulative_ps before computation
        self.original_histograms = {}  # Reset original histograms

        # Get channels from detector wizard page if available
        use_detector_wizard = False
        micro_time_ranges = {}
        
        if hasattr(self, 'detector_wizard_page') and self.detector_wizard_page:
            channels = self.detector_wizard_page.channels()
            if channels:
                use_detector_wizard = True
                # Extract microtime ranges for parallel and perpendicular channels
                if 'parallel' in channels:
                    micro_time_ranges['parallel'] = [
                        channel_info.get('micro_time_range', None) 
                        for channel_info in channels['parallel']
                    ]
                if 'perpendicular' in channels:
                    micro_time_ranges['perpendicular'] = [
                        channel_info.get('micro_time_range', None) 
                        for channel_info in channels['perpendicular']
                    ]
        
        chisurf.logging.info(f"channels parallel: {self.parallel_channels}")
        chisurf.logging.info(f"channels perpendicular: {self.perpendicular_channels}")
        if use_detector_wizard:
            chisurf.logging.info(f"Using detector wizard page for channel configuration")
            chisurf.logging.info(f"Microtime ranges: {micro_time_ranges}")

        # Check for the presence of BID/BUR files
        bid_files = self.listWidget_BID.get_selected_files()
        if len(bid_files) > 0:
            # Create a dictionary to store bid ranges with more flexible matching
            bid_ranges = {}

            # Process each BID/BUR file
            for f in bid_files:
                # Load the ranges from the file
                ranges = self.load_bid_ranges(f)

                # For .bur files, handle special naming convention
                if f.suffix.lower() == '.bur':
                    # Extract base name (removing _X suffix if present)
                    base_name = f.stem
                    if '_' in base_name:
                        parts = base_name.split('_')
                        if len(parts) > 1 and parts[-1].isdigit():
                            base_name = '_'.join(parts[:-1])

                    # Add to dictionary with base name as key
                    bid_ranges[base_name] = ranges
                else:
                    # For regular BID files, use stem as key
                    bid_ranges[f.stem] = ranges
        else:
            bid_ranges = None

        for filename in self.selected_files:
            path = Path(filename)
            if path.is_file():
                d = tttrlib.TTTR(path.as_posix(), self.tttr_filetype)

                # Try to find matching BID/BUR ranges for this TTTR file
                tttr_stem = path.stem
                tttr_base_name = tttr_stem

                # For TTTR files that might have suffixes, try to extract the base name
                if '_' in tttr_stem:
                    parts = tttr_stem.split('_')
                    if len(parts) > 1 and parts[-1].isdigit():
                        tttr_base_name = '_'.join(parts[:-1])

                # Check if we have ranges for this file (try different name variations)
                matching_key = None
                if bid_ranges:
                    # Try exact stem match first
                    if tttr_stem in bid_ranges:
                        matching_key = tttr_stem
                    # Try base name match
                    elif tttr_base_name in bid_ranges:
                        matching_key = tttr_base_name
                    # Try filename match (without path)
                    elif path.name in bid_ranges:
                        matching_key = path.name

                if matching_key and bid_ranges:
                    burst_ranges = bid_ranges[matching_key]
                    if len(burst_ranges) < 1:
                        continue

                    # Create a mask of photons that are part of any burst
                    photon_mask = self.get_burst_photon_mask(d, burst_ranges)

                    if photon_mask is None or len(photon_mask) == 0:
                        chisurf.logging.warning(f"No valid photon mask created for {path.name}")
                        continue

                    # Get the indices of photons that are part of bursts
                    burst_indices = np.nonzero(photon_mask)[0]

                    if len(burst_indices) == 0:
                        chisurf.logging.warning(f"No burst photons found for {path.name}")
                        continue

                    # Create a new TTTR object with only the burst photons
                    t = d[burst_indices]

                    chisurf.logging.info(f"Using {len(burst_indices)} burst photons from {path.name} (matched with {matching_key})")
                elif bid_ranges:
                    chisurf.logging.info(f"Skipping {path.name} because BID/BUR range not found.")
                    continue  # Skip file if BID range exists but no match found
                else:
                    t = d

                # Get the original histograms without any timeshift
                # Get interleaved channels and then extract parallel and perpendicular
                interleaved_channels = self._get_interleaved_channels()
                parallel_channels = interleaved_channels[::2]  # Even indices (0, 2, 4, ...)
                perpendicular_channels = interleaved_channels[1::2]  # Odd indices (1, 3, 5, ...)
                
                chisurf.logging.info(f"Using interleaved channels: {interleaved_channels}")
                chisurf.logging.info(f"Parallel channels (even indices): {parallel_channels}")
                chisurf.logging.info(f"Perpendicular channels (odd indices): {perpendicular_channels}")
                
                y_parallel, _ = t.get_microtime_histogram(self.binning_factor, parallel_channels)
                y_perpendicular, _ = t.get_microtime_histogram(self.binning_factor, perpendicular_channels)

                # Get time resolution in nanoseconds
                self.time_resolution = self.time_step

                # Store the original histograms for later use
                self.original_histograms[path.name] = {
                    'parallel': y_parallel.copy(),
                    'perpendicular': y_perpendicular.copy()
                }

        # Apply timeshifts to the original histograms
        self.update_timeshifts()

        # Auto-save the histogram
        if self.cumulative_ps is not None and self.selected_files:
            # Get the full save path directly from lineEdit_5
            save_path = Path(self.lineEdit_5.text())

            # Save the histogram
            self.save_cumulative_histogram(str(save_path))

    def browse_and_open_input_files(self):
        dialog = QtWidgets.QFileDialog(self, "Select TTTR Files")
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if dialog.exec_():
            selected_files = dialog.selectedFiles()
            for file in selected_files:
                self.listWidget.add_file(file)


if __name__ == "plugin":
    microtime_hist = MicrotimeHistogram()
    microtime_hist.show()

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    microtime_hist = MicrotimeHistogram()
    microtime_hist.setWindowTitle('Microtime Histogram')
    microtime_hist.show()
    sys.exit(app.exec_())
