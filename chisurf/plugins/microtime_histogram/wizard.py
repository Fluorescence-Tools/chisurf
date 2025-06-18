import typing
from pathlib import Path
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np

import tttrlib
import chisurf.gui.decorators
import chisurf.settings

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

                # Handle directories - look for .bur files if this is the BID list widget
                if path_obj.is_dir() and hasattr(self.parent, 'listWidget_BID') and self == self.parent.listWidget_BID:
                    # Search for .bur files in the directory
                    bur_files = list(path_obj.glob("**/*.bur"))
                    if bur_files:
                        for bur_file in bur_files:
                            file_paths.append(str(bur_file))
                    else:
                        chisurf.logging.info(f"No .bur files found in directory: {file_path}")
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

            # Try to infer file type from the first file if set to Auto
            if file_type == "Auto" and first_file_path:
                file_type_int = tttrlib.inferTTTRFileType(first_file_path)

                # Update comboBox if a file type is recognized
                if file_type_int is not None and file_type_int >= 0:
                    container_names = tttrlib.TTTR.get_supported_container_names()
                    if 0 <= file_type_int < len(container_names):
                        # Find the index in the comboBox
                        file_type_name = container_names[file_type_int]
                        index = self.parent.comboBox.findText(file_type_name)
                        if index >= 0:
                            self.parent.comboBox.setCurrentIndex(index)

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

    @property
    def binning_factor(self) -> int:
        return int(self.comboBox_2.currentText())

    @property
    def tttr_filetype(self) -> str | None:
        txt = self.comboBox.currentText()
        if txt == 'Auto':
            # Try to infer file type from the current file
            if self.selected_files:
                current_file = self.selected_files[0]
                if current_file and Path(current_file).exists():
                    file_type_int = tttrlib.inferTTTRFileType(current_file.as_posix())
                    # Update comboBox if a file type is recognized
                    if file_type_int is not None and file_type_int >= 0:
                        container_names = tttrlib.TTTR.get_supported_container_names()
                        if 0 <= file_type_int < len(container_names):
                            # Return the inferred file type name
                            return container_names[file_type_int]
            # If we can't infer, return None to let tttrlib try auto-detection
            return None
        return txt

    @property
    def parallel_channels(self) -> list[int]:
        try:
            s = self.lineEdit_2.text()
            return [int(i) for i in s.split(",") if i.strip().isdigit()]
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Parallel channels input is not valid.")
            return []

    @property
    def perpendicular_channels(self) -> list[int]:
        try:
            s = self.lineEdit.text()
            return [int(i) for i in s.split(",") if i.strip().isdigit()]
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Perpendicular channels input is not valid.")
            return []

    @property
    def selected_files(self):
        return self.listWidget.get_selected_files()

    @chisurf.gui.decorators.init_with_ui("microtime_histogram/wizard.ui", path=chisurf.settings.plugin_path)
    def __init__(self, *args, **kwargs):
        self._tttr = None
        self.comboBox.clear()
        self.comboBox.addItems(['Auto'] + list(tttrlib.TTTR.get_supported_container_names()))

        self.listWidget = FileListWidget(parent=self, file_added_callback=self.update_micro_time_resolution)
        self.verticalLayout_3.addWidget(self.listWidget)

        self.listWidget_BID = FileListWidget(parent=self, file_added_callback=self.load_corresponding_tttr_files)
        self.verticalLayout_5.addWidget(self.listWidget_BID)

        self.plotWidget = pg.PlotWidget()
        self.verticalLayout.addWidget(self.plotWidget)
        self.plotWidget.setLabel('bottom', 'Counts')
        self.plotWidget.setLabel('left', 'Micro Time (ns)')
        self.plotWidget.setLogMode(y=True)

        self.populate_supported_types()
        self.toolButton.clicked.connect(self.browse_and_open_input_files)
        self.toolButton_2.clicked.connect(self.clear_files)
        self.pushButton.clicked.connect(self.compute_microtime_histogram)
        self.comboBox_2.currentIndexChanged.connect(self.update_micro_time_resolution)
        self.pushButton_2.clicked.connect(self.open_save_dialog)
        self.checkBox.clicked.connect(self.on_transfer_clicked)
        self.checkBox.setChecked(True)  # Turn on transfer to chisurf by default

        self.lineEdit_2.textChanged.connect(self.update_output_filename)  # Parallel channels
        self.lineEdit.textChanged.connect(self.update_output_filename)  # Perpendicular channels

    def load_corresponding_tttr_files(self, n_parent=3):
        """
        Search for TTTR files in the folder structure above the selected BID/BUR files up to n_parent levels.
        This method handles both individual BID files and .bur files from burstwise folders.
        """
        bid_files = self.listWidget_BID.get_selected_files()
        tttr_files = set()

        for bid_file in bid_files:
            # Handle different file extensions
            if bid_file.suffix.lower() == '.bur':
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
                    # Add all matching files to the set
                    for tttr_file in matching_tttr_files:
                        tttr_files.add(tttr_file)
                    break  # Stop searching once matches are found

                # Move up to the parent directory
                parent_folder = parent_folder.parent
                level += 1

        # Clear existing TTTR list and add found files
        self.listWidget.clear()
        for tttr_file in sorted(tttr_files):
            self.listWidget.add_file(tttr_file.as_posix())

        chisurf.logging.info(f"Loaded TTTR files: {[f.as_posix() for f in tttr_files]}")

    def save_cumulative_histogram(self, file_path):
        """Save the cumulative histogram to a text file as a single-column integer list.
        If no histogram exists, compute it first.
        """
        # If cumulative_ps is not available, compute histogram
        if not hasattr(self, "cumulative_ps") or self.cumulative_ps is None:
            self.compute_microtime_histogram()  # Compute histogram first

        # Check again after computation
        if self.cumulative_ps is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Failed to compute cumulative histogram before saving.")
            return

        try:
            # Ensure data is saved as integers
            np.savetxt(file_path, self.cumulative_ps.astype(int), fmt="%d")

            # Log success instead of showing a message box
            chisurf.logging.info(f"Histogram successfully saved to: {file_path}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"An error occurred while saving:\n{str(e)}")
            chisurf.logging.error(f"Failed to save histogram: {str(e)}")

    def on_transfer_clicked(self):
        """
        Handle the "Transfer to ChiSurf" checkbox click event.
        If checked, add the histogram to ChiSurf.
        """
        if self.checkBox.isChecked():
            self.add_to_chisurf()

    def add_to_chisurf(self):
        """
        Add the computed microtime histogram to chisurf as a dataset.
        This method uses the standard loading approach by loading the saved histogram file.
        """
        chisurf.logging.info("MicrotimeHistogram::adding histogram to chisurf")

        # If cumulative_ps is not available, compute histogram
        if not hasattr(self, "cumulative_ps") or self.cumulative_ps is None:
            self.compute_microtime_histogram()  # Compute histogram first

        # Check again after computation
        if self.cumulative_ps is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Failed to compute microtime histogram.")
            return

        # Ensure we have selected files to determine the save path
        if not self.selected_files:
            QtWidgets.QMessageBox.warning(self, "No Files", "Please select a file before adding to ChiSurf.")
            return

        # Get the directory of the first selected file
        first_file_path = Path(self.selected_files[0])
        save_directory = first_file_path.parent

        # Get the output filename from lineEdit_5
        filename = self.lineEdit_5.text()

        # Create full save path
        save_path = save_directory / filename

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

        # Get polarization and g-factor from UI
        polarization = self.comboBox_polarization.currentText()
        try:
            g_factor = float(self.lineEdit_gfactor.text())
        except ValueError:
            # If g-factor is not a valid float, use default value
            g_factor = 1.000000
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid G-Factor",
                "G-Factor must be a valid number. Using default value of 1.0."
            )
            # Update the UI with the default value
            self.lineEdit_gfactor.setText("1.000000")

        # Set the current experiment to TCSPC
        cs.current_experiment = 'TCSPC'
        cs.current_setup.is_jordi = True
        cs.current_setup.use_header = False
        cs.current_setup.matrix_columns = []
        cs.current_setup.g_factor = g_factor
        cs.current_setup.polarization = polarization
        cs.current_setup.rep_rate = 10.0
        cs.current_setup.rebin = (1, 1)
        cs.current_setup.dt = float(self.lineEdit_4.text())

        # Add dataset to chisurf using the standard approach
        chisurf.macros.add_dataset(filename=str(save_path))

        # Show success message
        chisurf.logging.info(f"Added microtime histogram to ChiSurf: {save_path.name}")

        # Show a success message to the user
        QtWidgets.QMessageBox.information(self, "Success", "Microtime histogram added to ChiSurf successfully.")

    def open_save_dialog(self):
        """Open a save dialog in the directory of the first opened file and save the cumulative histogram."""
        if not self.selected_files:
            QtWidgets.QMessageBox.warning(self, "No Files", "Please select a file before saving.")
            return

        # Get the directory of the first selected file
        first_file_path = Path(self.selected_files[0])
        save_directory = first_file_path.parent

        # Get the output filename from lineEdit_5
        default_filename = self.lineEdit_5.text()

        # Open the save file dialog
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", str(save_directory / default_filename), "Data Files (*.dat);;All Files (*)"
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
        chisurf.logging.info("Clearing files and plot")
        self.listWidget.clear()  # Clear file list
        self.plotWidget.clear()  # Clear plot
        self.listWidget_BID.clear() # Clear bid files
        # Removed: self.comboBox.setCurrentIndex(0)  # Reset combobox to "Auto"

    def update_output_filename(self):
        """Update the output filename in based on selected files and channel numbers."""
        if not self.selected_files:
            return

        # Get the first file name without extension
        first_file = Path(self.selected_files[0]).stem

        # Get the parallel (p) and perpendicular (s) channel numbers
        p_channels = ",".join(map(str, self.parallel_channels)) if self.parallel_channels else "all"
        s_channels = ",".join(map(str, self.perpendicular_channels)) if self.perpendicular_channels else "all"

        # Construct filename in format: filename_(p)-(s).dat
        output_filename = f"{first_file}_({p_channels})-({s_channels}).dat"

        # Set the filename in lineEdit_5
        self.lineEdit_5.setText(output_filename)

    def update_micro_time_resolution(self):
        """Update micro time resolution and output filename when files or binning change."""
        if self.selected_files:
            path = Path(self.selected_files[0])  # Use the first file
            if path.is_file():
                t = tttrlib.TTTR(path.as_posix(), self.tttr_filetype)
                micro_time_resolution = t.header.micro_time_resolution
                binned_micro_time_resolution = micro_time_resolution * self.binning_factor * 1e9
                self.lineEdit_4.setText(f"{binned_micro_time_resolution:.6f}")  # Update UI

        # Also update the output filename
        self.update_output_filename()

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

    def compute_microtime_histogram(self):
        chisurf.logging.info("Computing microtime histogram...")
        self.plotWidget.clear()  # Clear plot before drawing new data
        self.cumulative_ps = None  # Reset cumulative_ps before computation

        chisurf.logging.info(f"channels parallel: {self.parallel_channels}")
        chisurf.logging.info(f"channels perpendicular: {self.perpendicular_channels}")

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

                y_parallel, _ = t.get_microtime_histogram(self.binning_factor, self.parallel_channels)
                y_perpendicular, _ = t.get_microtime_histogram(self.binning_factor, self.perpendicular_channels)
                ps = np.hstack((y_parallel, y_perpendicular))

                x = np.arange(len(ps))
                self.plotWidget.plot(x, ps, pen='r', name="Parallel, Perpendicular")

                if self.cumulative_ps is None:
                    self.cumulative_ps = np.array(ps)
                else:
                    try:
                        self.cumulative_ps += np.array(ps)
                    except ValueError:
                        chisurf.logging.error(f"Failed to add cumulative histogram for {file_stem}")

        if self.cumulative_ps is not None:
            x = np.arange(len(self.cumulative_ps))
            self.plotWidget.plot(x, self.cumulative_ps, pen='b', name="Cumulative PS")

            # First auto-save the histogram
            if self.selected_files:
                # Get the directory of the first selected file
                first_file_path = Path(self.selected_files[0])
                save_directory = first_file_path.parent

                # Get the output filename from lineEdit_5
                filename = self.lineEdit_5.text()

                # Create full save path
                save_path = save_directory / filename

                # Save the histogram
                self.save_cumulative_histogram(str(save_path))

            # Then, if the "Transfer to ChiSurf" checkbox is checked, add the histogram to ChiSurf
            if self.checkBox.isChecked():
                self.add_to_chisurf()

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
