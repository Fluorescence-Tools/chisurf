import sys
import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QScrollArea, QSpinBox,
    QComboBox, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import tttrlib  # ensure tttrlib is in your PYTHONPATH

# Container type information for file format conversion
CONTAINER_INFO = {
    'PTU':              ('ptu', 4,  0),   # PQ_PTU_CONTAINER → PQ_RECORD_TYPE_HHT3v2
    'HT3':              ('ht3', 4,  1),   # PQ_HT3_CONTAINER → PQ_RECORD_TYPE_HHT3v2
    'SPC-130':          ('spc', 7,  2),   # BH_SPC130_CONTAINER → BH_RECORD_TYPE_SPC130
    'SPC-600_256':      ('spc', 8,  3),   # BH_SPC600_256_CONTAINER → BH_RECORD_TYPE_SPC600_256
    'SPC-600_4096':     ('spc', 9,  4),   # BH_SPC600_4096_CONTAINER → BH_RECORD_TYPE_SPC600_4096
    'PHOTON-HDF5':      ('hdf', 4,  5),   # PHOTON_HDF_CONTAINER → PQ_RECORD_TYPE_PHT3
    'CZ-RAW':           ('raw', 10, 6),   # CZ_CONFOCOR3_CONTAINER → CZ_RECORD_TYPE_CONFOCOR3
    'SM':               ('sm',  11, 7),   # SM_CONTAINER → SM_RECORD_TYPE
}

class FileLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drag TTTR file or type path...")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.setText(path)
            main_win = self.window()
            if hasattr(main_win, 'load_file'):
                main_win.load_file(path)
        event.acceptProposedAction()

class AlexPTUCreator(QMainWindow):
    @property
    def tttr_filetype(self) -> str | None:
        txt = self.format_combo.currentText()
        if txt == 'Auto':
            # Try to infer file type from the current file
            if hasattr(self, 'tttr_path') and self.tttr_path and os.path.exists(self.tttr_path):
                file_type_int = tttrlib.inferTTTRFileType(self.tttr_path)
                # Update comboBox if a file type is recognized
                if file_type_int is not None and file_type_int >= 0:
                    container_names = tttrlib.TTTR.get_supported_container_names()
                    if 0 <= file_type_int < len(container_names):
                        # Return the inferred file type name
                        return container_names[file_type_int]
            # If we can't infer, return None to let tttrlib try auto-detection
            return None
        return txt

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALEX PTU Creator")
        self.resize(800, 600)

        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        self.setCentralWidget(central)

        # Controls panel
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(2, 2, 2, 2)
        controls_layout.setSpacing(2)
        main_layout.addWidget(controls, stretch=1)

        # Load/Save
        load_layout = QHBoxLayout()
        load_layout.setContentsMargins(0, 0, 0, 0)
        load_layout.setSpacing(2)
        btn_load = QPushButton("Load…")
        btn_load.clicked.connect(self.open_file_dialog)
        load_layout.addWidget(btn_load)
        self.file_edit = FileLineEdit(self)
        self.file_edit.returnPressed.connect(lambda: self.load_file(self.file_edit.text()))
        load_layout.addWidget(self.file_edit)
        controls_layout.addLayout(load_layout)

        # File format selection
        format_layout = QHBoxLayout()
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.setSpacing(2)
        format_label = QLabel("Input Format:")
        format_layout.addWidget(format_label)
        self.format_combo = QComboBox()
        self.format_combo.addItem("Auto")
        self.format_combo.addItems(tttrlib.TTTR.get_supported_container_names())
        format_layout.addWidget(self.format_combo)
        controls_layout.addLayout(format_layout)

        # Output format selection
        output_format_layout = QHBoxLayout()
        output_format_layout.setContentsMargins(0, 0, 0, 0)
        output_format_layout.setSpacing(2)
        output_format_label = QLabel("Output Format:")
        output_format_layout.addWidget(output_format_label)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(tttrlib.TTTR.get_supported_container_names())
        # Set default output format to PTU
        if "PTU" in tttrlib.TTTR.get_supported_container_names():
            self.output_format_combo.setCurrentText("PTU")
        output_format_layout.addWidget(self.output_format_combo)
        controls_layout.addLayout(output_format_layout)

        # ALEX Period input
        period_layout = QHBoxLayout()
        period_layout.setContentsMargins(0, 0, 0, 0)
        period_layout.setSpacing(2)
        period_label = QLabel("ALEX Period:")
        period_layout.addWidget(period_label)
        self.period_spinbox = QSpinBox()
        self.period_spinbox.setRange(1, 1000000)
        self.period_spinbox.setValue(8000)  # Default value from the notebook
        self.period_spinbox.valueChanged.connect(self.update_histogram)
        period_layout.addWidget(self.period_spinbox)
        controls_layout.addLayout(period_layout)

        # Period Shift input
        shift_layout = QHBoxLayout()
        shift_layout.setContentsMargins(0, 0, 0, 0)
        shift_layout.setSpacing(2)
        shift_label = QLabel("Period Shift:")
        shift_layout.addWidget(shift_label)
        self.shift_spinbox = QSpinBox()
        self.shift_spinbox.setRange(-1000000, 1000000)
        self.shift_spinbox.setValue(0)  # Default value
        self.shift_spinbox.valueChanged.connect(self.update_histogram)
        shift_layout.addWidget(self.shift_spinbox)
        controls_layout.addLayout(shift_layout)

        # Save button
        save_layout = QHBoxLayout()
        save_layout.setContentsMargins(0, 0, 0, 0)
        save_layout.setSpacing(2)
        self.btn_save = QPushButton("Save PTU…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_file)
        save_layout.addWidget(self.btn_save)
        controls_layout.addLayout(save_layout)

        # Add stretch to push controls to the top
        controls_layout.addStretch()

        # Plot
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        plot_layout.setSpacing(2)
        main_layout.addWidget(plot_container, stretch=3)

        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Micro-time bin')
        self.plot.setLabel('left', 'Counts')
        plot_layout.addWidget(self.plot)

        # state
        self.tttr = None
        self.tttr_path = None
        self.orig_mt = None
        self.routing = None

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open TTTR file", "", "TTTR Files (*.*)")
        if path:
            self.file_edit.setText(path)
            self.load_file(path)

    def load_file(self, path):
        try:
            # Store the path first so tttr_filetype can use it for inference
            self.tttr_path = path

            # Get the file type using the tttr_filetype property
            input_format = self.tttr_filetype

            # Load the TTTR file
            self.tttr = tttrlib.TTTR(path, input_format)

            # Store original data
            self.orig_mt = self.tttr.macro_times.copy()
            self.routing = self.tttr.routing_channels.copy()
            self.btn_save.setEnabled(True)

            # Update the format combo box if a format was inferred
            if input_format and self.format_combo.currentText() == "Auto":
                index = self.format_combo.findText(input_format)
                if index >= 0:
                    self.format_combo.setCurrentIndex(index)

            # Display file information
            container_type = self.tttr.header.tttr_container_type
            container_names = tttrlib.TTTR.get_supported_container_names()
            if container_type < len(container_names):
                detected_format = container_names[container_type]
                QMessageBox.information(self, "File Loaded", 
                                       f"Loaded file: {os.path.basename(path)}\nDetected format: {detected_format}")

            # Update the histogram with current settings
            self.update_histogram()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load file:\n{e}")
            return

    def update_histogram(self):
        if self.tttr is None:
            return

        # Get the current ALEX period and shift values
        alex_period = self.period_spinbox.value()
        period_shift = self.shift_spinbox.value()

        # Create a copy of the original TTTR object to avoid modifying the original
        # Use the inferred file type
        tt = tttrlib.TTTR(self.tttr_path, self.tttr_filetype)

        # Apply the ALEX to microtime conversion
        tt.alex_to_microtime(alex_period, period_shift)

        # Store the processed TTTR object for later use
        self.processed_tttr = tt

        # Get the microtime histogram
        mt_hist = np.bincount(tt.micro_times, minlength=alex_period)

        # Create edges array for step mode plotting (needs to be len(Y) + 1)
        edges = np.arange(alex_period + 1)

        # Plot the histogram
        self.plot.clear()
        self.plot.plot(edges, mt_hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 80))
        self.plot.setTitle(f'ALEX Microtime Histogram (Period: {alex_period}, Shift: {period_shift})')

    def save_file(self):
        if self.tttr is None:
            return

        # Get the output format
        output_format = self.output_format_combo.currentText()

        # Get the file extension based on the output format
        ext = CONTAINER_INFO.get(output_format, ('ptu', 0, 0))[0]

        # Get save path
        d = os.path.dirname(self.tttr_path)
        f = os.path.basename(self.tttr_path)
        base, _ = os.path.splitext(f)

        # Create default name with appropriate extension
        if hasattr(self, 'module_checkbox') and self.module_checkbox.isChecked():
            default_name = f"{base}_alex_modulo.{ext}"
        else:
            default_name = f"{base}_alex.{ext}"

        # Set up file dialog with appropriate filter
        file_filter = f"{output_format} Files (*.{ext});;All Files (*.*)"
        sp, _ = QFileDialog.getSaveFileName(
            self, f"Save ALEX {output_format} file", 
            os.path.join(d, default_name), 
            file_filter
        )

        if not sp:
            return

        try:
            # Use the processed TTTR object from update_histogram
            if not hasattr(self, 'processed_tttr') or self.processed_tttr is None:
                # If processed_tttr doesn't exist, update the histogram to create it
                self.update_histogram()

            if not hasattr(self, 'processed_tttr') or self.processed_tttr is None:
                QMessageBox.critical(self, "Error", "Failed to process TTTR data")
                return

            tt = self.processed_tttr

            # Prepare header for the output format
            if output_format != self.format_combo.currentText() and output_format != "Auto":
                # Get container type information
                ext, rec, cont = CONTAINER_INFO.get(output_format, ('ptu', 4, 0))

                # Create a new header with the appropriate container type
                header = tt.header
                header.tttr_container_type = cont
                header.tttr_record_type = rec

                if output_format == "PTU":
                    # PTU via HydraHarp wants the special tag group 0x00010304
                    header.set_tag("TTResultFormat_TTTRRecType", 0x00010304, 268435464)
                    header.set_tag("TTResultFormat_BitsPerRecord", 32, 268435464)
                    header.set_tag("MeasDesc_RecordType", rec, 268435464)
                else:
                    # Default behavior for other containers
                    header.set_tag("TTResultFormat_TTTRRecType", rec)
                    header.set_tag("MeasDesc_RecordType", rec)

                # Save the file with the modified header
                tt.write(sp, header)
            else:
                # Save the file with the default header
                tt.write(sp)

            QMessageBox.information(self, "Saved", f"Saved to:\n{sp}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot save:\n{e}")

if __name__ == 'plugin':
    w = AlexPTUCreator(); w.show()
elif __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = AlexPTUCreator(); w.show()
    sys.exit(app.exec_())
