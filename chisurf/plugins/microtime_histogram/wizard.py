import typing
from pathlib import Path
from qtpy import QtWidgets, QtCore, QtGui

import tttrlib
import chisurf.gui.decorators
import chisurf.settings

VERBOSE = False

class FileListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

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
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if Path(file_path).is_file():
                    self.add_file(file_path)
            event.acceptProposedAction()
        else:
            event.ignore()

    def add_file(self, file_path: str):
        print(f"Adding {file_path} to listwidget")
        item = QtWidgets.QListWidgetItem(file_path, self)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        self.addItem(item)

class MicrotimeHistogram(QtWidgets.QWidget):

    @chisurf.gui.decorators.init_with_ui("microtime_histogram/wizard.ui", path=chisurf.settings.plugin_path)
    def __init__(self, *args, **kwargs):

        self._tttr = None

        # Replace existing listWidget with enhanced FileListWidget
        self.listWidget = FileListWidget(self)
        self.layout().addWidget(self.listWidget)

        # Fill combo box with supported types + "Auto"
        self.populate_supported_types()

        # Connect UI elements
        self.toolButton.clicked.connect(self.browse_and_open_input_files)
        self.toolButton_2.clicked.connect(self.clear_files)
        self.pushButton.clicked.connect(self.compute_microtime_histogram)

    def populate_supported_types(self):
        """Populates the comboBox with supported container types plus an 'Auto' option."""
        self.comboBox.clear()
        self.comboBox.insertItem(0, "Auto")
        self.comboBox.insertItems(1, list(tttrlib.TTTR.get_supported_container_names()))

    def clear_files(self):
        print("Clearing files")
        self.listWidget.clear()

    def compute_microtime_histogram(self):
        print("Computing microtime histogram...")

    def browse_and_open_input_files(self):
        """File dialog for selecting multiple TTTR files."""
        dialog = QtWidgets.QFileDialog(self, "Select TTTR Files")
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if dialog.exec_():
            selected_files = dialog.selectedFiles()
            for file in selected_files:
                self.listWidget.add_file(file)

    @property
    def selected_files(self):
        return [self.listWidget.item(i).text() for i in range(self.listWidget.count()) if
                self.listWidget.item(i).checkState() == QtCore.Qt.Checked]

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
