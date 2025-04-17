from pathlib import Path
from qtpy import QtWidgets, QtCore, QtGui

import tttrlib

import chisurf.gui.decorators
import chisurf.settings

VERBOSE = False

def enable_file_drop_for_open(line_edit: QtWidgets.QLineEdit, open_func):
    """
    Enable dropping a *single file* onto a QLineEdit.
    Calls `open_func(path_str)` after dropping the file.
    """
    line_edit.setAcceptDrops(True)

    def dragEnterEvent(event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            # Accept only if exactly one URL and it is an existing file
            if len(urls) == 1:
                if Path(urls[0].toLocalFile()).is_file():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(event: QtGui.QDropEvent):
        if event.mimeData().hasUrls():
            path_str = event.mimeData().urls()[0].toLocalFile()
            if Path(path_str).is_file():
                event.acceptProposedAction()
                open_func(path_str)  # Call the function to actually open the file
                return
        event.ignore()

    # Monkey-patch the lineEdit's events:
    line_edit.dragEnterEvent = dragEnterEvent
    line_edit.dropEvent = dropEvent


def enable_folder_drop(line_edit: QtWidgets.QLineEdit):
    """
    Enable dropping a *single folder* onto a QLineEdit.
    Sets the lineEdit text to the dropped folder path.
    """
    line_edit.setAcceptDrops(True)

    def dragEnterEvent(event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            # Accept only if exactly one URL and it is an existing directory
            if len(urls) == 1:
                if Path(urls[0].toLocalFile()).is_dir():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(event: QtGui.QDropEvent):
        if event.mimeData().hasUrls():
            folder_str = event.mimeData().urls()[0].toLocalFile()
            if Path(folder_str).is_dir():
                event.acceptProposedAction()
                line_edit.setText(folder_str)
                return
        event.ignore()

    # Monkey-patch the lineEdit's events:
    line_edit.dragEnterEvent = dragEnterEvent
    line_edit.dropEvent = dropEvent



class PTUSplitter(QtWidgets.QWidget):

    @chisurf.gui.decorators.init_with_ui("ptu_splitter/wizard.ui",
                                         path=chisurf.settings.plugin_path)
    def __init__(self, *args, **kwargs):
        # NO super() call here (the decorator handles it).
        self._tttr = None

        # Set up drag & drop on lineedits
        enable_file_drop_for_open(self.lineEdit, self._open_input_file)
        enable_folder_drop(self.lineEdit_2)

        # Fill combo box with supported types + "Auto"
        self.populate_supported_types()

        # Connect your UI elements (change names if different in .ui)
        self.toolButton.clicked.connect(self.browse_and_open_input_file)
        self.toolButton_2.clicked.connect(self.browse_output_folder)
        self.pushButton.clicked.connect(self.split_file)

        # Initialize progress bar to 0
        self.progressBar.setValue(0)

    # --------------------------------------------------------------------------
    # Private helper to open a file (browse or drag & drop)
    # --------------------------------------------------------------------------
    def _open_input_file(self, file_path: str):
        """
        Loads the specified file into the TTTR object.
        Also updates lineEdit (input path) and lineEdit_2 (default output folder).
        """
        # Reset progress bar on file load
        self.progressBar.setValue(0)

        p = Path(file_path)
        if not p.is_file():
            QtWidgets.QMessageBox.warning(self, "Invalid File",
                                          f"'{file_path}' is not a valid file.")
            return

        # Update the lineEdit to reflect the chosen file
        self.lineEdit.setText(str(p))

        # Default output folder is file's parent
        self.lineEdit_2.setText(str(p.parent))

        # Create the TTTR object
        if self.tttr_type is None:
            self._tttr = tttrlib.TTTR(str(p))
        else:
            self._tttr = tttrlib.TTTR(str(p), self.tttr_type)

        if VERBOSE:
            QtWidgets.QMessageBox.information(
                self, "File Loaded",
                f"Successfully opened {p.name}."
            )

    def populate_supported_types(self):
        """Populates the comboBox with supported container types plus an 'Auto' option."""
        self.comboBox.clear()
        self.comboBox.insertItem(0, "Auto")
        self.comboBox.insertItems(1, list(tttrlib.TTTR.get_supported_container_names()))

    def browse_and_open_input_file(self):
        """File dialog for selecting a PTU file, then open it."""
        dialog = QtWidgets.QFileDialog(self, "Select PTU File")
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        # Optionally: dialog.setNameFilter("PTU Files (*.ptu)")

        if dialog.exec_():
            selected_files = dialog.selectedFiles()
            if selected_files:
                self._open_input_file(selected_files[0])


    def browse_output_folder(self):
        """Open a File Dialog to select an output folder."""
        dialog = QtWidgets.QFileDialog(self, "Select Output Folder")
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        if dialog.exec_():
            selected_dirs = dialog.selectedFiles()
            if selected_dirs:
                self.lineEdit_2.setText(selected_dirs[0])

    def split_file(self):
        """
        Split the loaded TTTR file into multiple .ptu files.
        Each file will contain `photons_per_file` photons.
        Updates self.progressBar and disables user input during splitting.
        """
        # Ensure we have TTTR data:
        if self._tttr is None:
            QtWidgets.QMessageBox.warning(
                self, "No Data Loaded",
                "Please choose and load a PTU file before splitting."
            )
            return

        # Ensure valid output folder
        out_folder = self.output_folder
        if out_folder is None:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Output Folder",
                "Please specify a valid output folder."
            )
            return

        # Disable UI elements while splitting
        self._set_user_input_enabled(False)

        t = self._tttr
        total_photons = len(t)
        chunk_size = self.photons_per_file

        if total_photons == 0:
            QtWidgets.QMessageBox.warning(self, "Empty File", "No photons to split!")
            self._set_user_input_enabled(True)
            return

        n_full_chunks = total_photons // chunk_size
        remainder = total_photons % chunk_size
        total_files = n_full_chunks + (1 if remainder else 0)

        # Create sub-folder "filename_stem_chunkSize"
        input_file = self.tttr_input_filename
        output_subfolder = out_folder / f"{input_file.stem}"
        output_subfolder.mkdir(parents=True, exist_ok=True)

        # Retrieve header
        header = t.header

        # Loop over each chunk (including remainder if present)
        rst_mt = self.reset_macro_times
        chisurf.logging.info(f"Reset macro times: {rst_mt}")
        for i in range(total_files):
            # progress 0..100
            progress = int((i / total_files) * 100)
            self.progressBar.setValue(progress)
            QtWidgets.QApplication.processEvents()

            start = i * chunk_size
            stop = start + chunk_size
            if stop > total_photons:
                stop = total_photons  # leftover chunk

            c = t[start:stop]
            if rst_mt:
                n = tttrlib.TTTR()
                macro_times = c.macro_times
                micro_times = c.micro_times
                routing_channels = c.routing_channels
                event_types = c.event_types
                mt0 = int(-1 * macro_times[0])
                n.append_events(macro_times, micro_times, routing_channels, event_types, True, mt0)
            else:
                n = c
            out_name = f"{input_file.stem}_{i:05d}.ptu"
            fn = output_subfolder / out_name
            n.write(fn.as_posix(), header)

        # Finalize progress
        self.progressBar.setValue(100)

        if VERBOSE:
            QtWidgets.QMessageBox.information(
                self,
                "Splitting Complete",
                f"Created {total_files} files in:\n{output_subfolder}"
            )

        if not self.keep_original:
            input_file.unlink()

        # Re-enable UI elements
        self._set_user_input_enabled(True)

    def _set_user_input_enabled(self, enabled: bool):
        """
        Enable/Disable the user input widgets to prevent interaction during splitting.
        """
        pass
        # Adjust to your specific widget names:
        self.lineEdit.setEnabled(enabled)
        self.lineEdit_2.setEnabled(enabled)
        self.comboBox.setEnabled(enabled)
        self.spinBox.setEnabled(enabled)
        self.toolButton.setEnabled(enabled)
        self.toolButton_2.setEnabled(enabled)
        self.pushButton.setEnabled(enabled)

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def keep_original(self) -> bool:
        """Return True to keep the original file unchanged. Otherwise original file is deleted"""
        return bool(self.checkBox.isChecked())

    @property
    def photons_per_file(self) -> int:
        """Returns the current value from the spinBox as the chunk size."""
        return int(self.spinBox.value()) * 1000

    @property
    def tttr_type(self) -> str | None:
        """
        Returns the type from the comboBox or None if 'Auto' is selected.
        """
        tp = self.comboBox.currentText()
        return None if tp == "Auto" else tp

    @property
    def tttr_input_filename(self) -> Path | None:
        """
        Returns a Path object for the input file if it exists, otherwise None.
        """
        fn = self.lineEdit.text().strip()
        path = Path(fn)
        return path if path.is_file() else None

    @property
    def output_folder(self) -> Path | None:
        """
        Returns a Path object for the desired output folder, or None if invalid.
        Creates the folder if needed.
        """
        folder_str = self.lineEdit_2.text().strip()
        if not folder_str:
            return None
        p = Path(folder_str)
        # If the user typed a new folder path, we can decide to create it:
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Failed to create folder '{p}': {e}")
                return None
        return p

    @property
    def reset_macro_times(self) -> bool:
        return self.checkBox_2.isChecked()



if __name__ == "plugin":
    brick_mic_wiz = PTUSplitter()
    brick_mic_wiz.show()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    brick_mic_wiz = PTUSplitter()
    brick_mic_wiz.setWindowTitle('PTU-Splitter')
    brick_mic_wiz.show()
    sys.exit(app.exec_())
