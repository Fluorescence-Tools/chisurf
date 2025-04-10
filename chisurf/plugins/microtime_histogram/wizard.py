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
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if Path(file_path).is_file():
                    if file_path.endswith(".bst"):
                        suffix = Path(file_path).stem.rsplit('.', 1)[0]
                    else:
                        suffix = Path(file_path).suffix.lower()
                    if suffix.lower() in SPECIAL_FILETYPES and file_type == "Auto":
                        special_files.append(file_path)
                    else:
                        file_paths.append(file_path)
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
    def tttr_filetype(self) -> str:
        return str(self.comboBox.currentText())

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

        self.lineEdit_2.textChanged.connect(self.update_output_filename)  # Parallel channels
        self.lineEdit.textChanged.connect(self.update_output_filename)  # Perpendicular channels

    def load_corresponding_tttr_files(self, n_parent=3):
        """Search for TTTR files in the folder structure above the selected BID files up to n_parent levels."""
        bid_files = self.listWidget_BID.get_selected_files()
        tttr_files = set()

        for bid_file in bid_files:
            bid_stem = bid_file.stem  # Extract filename without .bst
            parent_folder = bid_file.parent
            level = 0

            while parent_folder != parent_folder.root and level < n_parent:
                matching_tttr_files = list(parent_folder.glob(f"{bid_stem}"))
                if matching_tttr_files:
                    tttr_files.add(matching_tttr_files[0])
                    break  # Stop searching once a match is found
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

    def populate_supported_types(self):
        self.comboBox.clear()
        self.comboBox.insertItem(0, "Auto")
        self.comboBox.insertItems(1, list(tttrlib.TTTR.get_supported_container_names()))

    def clear_files(self):
        chisurf.logging.info("Clearing files and plot")
        self.listWidget.clear()  # Clear file list
        self.plotWidget.clear()  # Clear plot
        self.listWidget_BID.clear() # Clear bid files

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
        """Load photon start-stop pairs from BID files."""
        bid_ranges = []
        with open(bid_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')  # Expect tab-separated values
                if len(parts) == 2:
                    start, stop = map(int, parts)
                    bid_ranges.append((start, stop))
        return bid_ranges

    def compute_microtime_histogram(self):
        chisurf.logging.info("Computing microtime histogram...")
        self.plotWidget.clear()  # Clear plot before drawing new data
        self.cumulative_ps = None  # Reset cumulative_ps before computation

        chisurf.logging.info(f"channels parallel: {self.parallel_channels}")
        chisurf.logging.info(f"channels perpendicular: {self.perpendicular_channels}")

        # Check for the presence of BID files
        bid_files = self.listWidget_BID.get_selected_files()
        if len(bid_files) > 0:
            bid_ranges = dict([(f.stem, self.load_bid_ranges(f)) for f in bid_files])
        else:
            bid_ranges = None

        for filename in self.selected_files:
            path = Path(filename)
            file_stem = path.name
            if path.is_file():
                d = tttrlib.TTTR(path.as_posix(), self.tttr_filetype)
                if bid_ranges and file_stem in bid_ranges:
                    bid_range = bid_ranges[file_stem]
                    if len(bid_range) < 1:
                        continue
                    start, stop = bid_range[0]
                    t = d[start:stop]
                    for start, stop in bid_range[1:]:
                        t += d[start:stop]
                elif bid_ranges:
                    chisurf.logging.info(f"Skipping {file_stem} because BID range not found.")
                    continue  # Skip file if BID range exists but file stem not found
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
