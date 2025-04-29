from pathlib import Path
from qtpy import QtWidgets, QtCore, QtGui

import tttrlib

import chisurf.gui.decorators
import chisurf.settings

VERBOSE = False

# Unified mapping: container name → (file‐extension stem, default record type id)
_CONTAINER_INFO = {
    'PTU':              ('ptu', 4,  0),   # PQ_PTU_CONTAINER → PQ_RECORD_TYPE_HHT3v2
    'HT3':              ('ht3', 4,  1),   # PQ_HT3_CONTAINER → PQ_RECORD_TYPE_HHT3v2
    'SPC-130':          ('spc', 7,  2),   # BH_SPC130_CONTAINER → BH_RECORD_TYPE_SPC130
    'SPC-600_256':      ('spc', 8,  3),   # BH_SPC600_256_CONTAINER → BH_RECORD_TYPE_SPC600_256
    'SPC-600_4096':     ('spc', 9,  4),   # BH_SPC600_4096_CONTAINER → BH_RECORD_TYPE_SPC600_4096
    'PHOTON-HDF5':      ('hdf', 4,  5),   # PHOTON_HDF_CONTAINER → PQ_RECORD_TYPE_PHT3
    'CZ-RAW':           ('raw', 10, 6),  # CZ_CONFOCOR3_CONTAINER → CZ_RECORD_TYPE_CONFOCOR3
    'SM':               ('sm',  11, 7),  # SM_CONTAINER → SM_RECORD_TYPE
}

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
        # Fill comboBox_2 with supported output container types (default PTU)
        self.populate_output_container_types()

        # Connect your UI elements (change names if different in .ui)
        self.toolButton.clicked.connect(self.browse_and_open_input_file)
        self.toolButton_2.clicked.connect(self.browse_output_folder)
        self.pushButton.clicked.connect(self.split_file)

        # whenever the user forces a different input type,
        # update the binning UI to match SPC defaults or re‐enable
        self.comboBox_2.currentIndexChanged.connect(self._update_binning_ui)

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

    def populate_supported_types(self):
        """Populates self.comboBox (input detection) …"""
        self.comboBox.clear()
        self.comboBox.insertItem(0, "Auto")
        self.comboBox.insertItems(1, tttrlib.TTTR.get_supported_container_names())

    def populate_output_container_types(self):
        """Populate self.comboBox_2 with all supported container names, default PTU."""
        names = tttrlib.TTTR.get_supported_container_names()
        self.comboBox_2.clear()
        self.comboBox_2.addItems(names)
        # select PTU if present
        if "PTU" in names:
            self.comboBox_2.setCurrentText("PTU")

    def _update_binning_ui(self):
        """
        Whenever a new file is loaded, if it’s an SPC file
        we force the comboBox_3 to at least 8 (and visually set it).
        Otherwise we re‑enable the box so the user can pick freely.
        """
        spc_types = {'SPC-130', 'SPC-600_256', 'SPC-600_4096'}
        output_type = self.output_container

        default_bin = 8 if output_type in spc_types else None

        if default_bin is not None:
            # set to 8 if below, then disable to show it’s fixed
            try:
                current = int(self.comboBox_3.currentText())
            except ValueError:
                current = 0
            if current < default_bin:
                self.comboBox_3.setCurrentText(str(default_bin))
            self.comboBox_3.setEnabled(False)
        else:
            # non‑SPC: allow full user control
            self.comboBox_3.setEnabled(True)

    def split_file(self):
        """
        Either split into multiple files or write all photons into a single file,
        based on self.split_files.
        """
        if self._tttr is None:
            QtWidgets.QMessageBox.warning(self, "No Data Loaded",
                                          "Please load a TTTR file first.")
            return

        t = self._tttr
        total = len(t)
        if total == 0:
            QtWidgets.QMessageBox.warning(self, "Empty File", "No photons to split!")
            return

        out_folder = self.output_folder
        if out_folder is None:
            QtWidgets.QMessageBox.warning(self, "Invalid Output Folder",
                                          "Please specify a valid output folder.")
            return

        self._set_user_input_enabled(False)

        # build ranges
        if self.split_files:
            chunk = self.photons_per_file
            full, rem = divmod(total, chunk)
            n = full + (1 if rem else 0)
            ranges = [(i * chunk, min((i + 1) * chunk, total)) for i in range(n)]
        else:
            ranges = [(0, total)]
            n = 1

        in_path = self.tttr_input_filename
        subfolder = out_folder / in_path.stem
        subfolder.mkdir(parents=True, exist_ok=True)

        # prepare header
        input_header = t.header
        out_name = self.output_container
        tttr_type_id_output = _CONTAINER_INFO[out_name][2]
        tttr_type_id_input = self.input_tttr_type_id
        ext, rec, cont = _CONTAINER_INFO[out_name]

        print(f"Input file: {self.tttr_input_filename}")
        print(f"Input file type: {tttr_type_id_input}")
        print(f"Output file type: {tttr_type_id_output}")

        if tttr_type_id_input == tttr_type_id_output:
            header = input_header
            print("Split")
        else:
            header = input_header  # reuse the existing header object
            header.tttr_container_type = cont
            header.tttr_record_type = rec

            if out_name == "PTU":
                # PTU via HydraHarp wants the special tag group 0x00010304
                header.set_tag("TTResultFormat_TTTRRecType", 0x00010304, 268435464)
                header.set_tag("TTResultFormat_BitsPerRecord", 32, 268435464)
                header.set_tag("MeasDesc_RecordType", rec, 268435464)
            else:
                # default behaviour for other containers
                header.set_tag("TTResultFormat_TTTRRecType", rec)
                header.set_tag("MeasDesc_RecordType", rec)

            print(f"Transcode {in_path} > {self.tttr_type} -> {out_name}")
            print(f"out_name: {out_name}")
            print(f"out_idx:  {tttr_type_id_output}")
            print(f"ext:      {ext}")
            print(f"rec:      {rec}")
            print(f"cont:     {cont}")

        # write each range
        for i, (start, stop) in enumerate(ranges):
            self.progressBar.setValue(int((i / n) * 100))
            QtWidgets.QApplication.processEvents()

            chunk_tttr = t[start:stop]
            if self.reset_macro_times or self.tttr_type != out_name:
                mt, µt = chunk_tttr.macro_times, chunk_tttr.micro_times
                rc, et = chunk_tttr.routing_channels, chunk_tttr.event_types
                if self.reset_macro_times:
                    mt0 = int(-1 * mt[0])
                else:
                    mt0 = 0
                new_tttr = tttrlib.TTTR()
                new_tttr.append_events(mt, µt, rc, et, True, mt0)
                chunk_tttr = new_tttr

            if self.split_files:
                fname = f"{in_path.stem}_{i:05d}.{ext}"
            else:
                fname = f"{in_path.stem}_all.{ext}"

            out_fp = subfolder / fname
            chunk_tttr.write(str(out_fp), header)

        self.progressBar.setValue(100)
        if VERBOSE:
            mode = "multiple" if self.split_files else "one"
            QtWidgets.QMessageBox.information(
                self, "Done",
                f"Wrote {mode} file(s) into:\n{subfolder}"
            )

        if not self.keep_original:
            in_path.unlink()

        self._set_user_input_enabled(True)

    def _set_user_input_enabled(self, enabled: bool):
        """
        Enable/Disable the user input widgets to prevent interaction during splitting.
        """
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
    def input_tttr_type_id(self) -> int:
        if self.tttr_type is None:
            r = tttrlib.inferTTTRFileType(self.tttr_input_filename.as_posix())
        else:
            r = self.comboBox.currentIndex() - 1
        return r

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

    @property
    def output_container(self) -> str:
        """The desired output container type, e.g. 'PTU', 'HT3', …"""
        return self.comboBox_2.currentText()

    @property
    def micro_time_binning(self) -> int:
        """
        Returns the integer binning factor for micro‑times:
          • Default is 8 for any SPC container (SPC‑130, SPC‑600_256, SPC‑600_4096), 1 otherwise.
          • If the user’s choice in comboBox_3 is larger than that default, use the user’s value.
        """
        spc_types = {'SPC-130', 'SPC-600_256', 'SPC-600_4096'}
        default_bin = 1
        try:
            names = tttrlib.TTTR.get_supported_container_names()
            idx = self._tttr.header.tttr_container_type
            container = names[idx]
            if container in spc_types:
                default_bin = 8
        except Exception:
            if self.tttr_type in spc_types:
                default_bin = 8

        try:
            user_bin = int(self.comboBox_3.currentText())
        except (ValueError, TypeError):
            user_bin = default_bin

        return max(default_bin, user_bin)

    @property
    def split_files(self) -> bool:
        """
        If True (checkBox_3 checked), split into chunks.
        If False, write all photons into a single file.
        """
        return bool(self.checkBox_3.isChecked())


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
