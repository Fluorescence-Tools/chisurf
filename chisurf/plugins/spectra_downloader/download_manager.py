import sys
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets


class DownloadManagerDialog(QtWidgets.QDialog):
    """Dialog for running fluorophore-data download scripts."""

    def __init__(self, db, parent=None):
        """Create the dialog and populate available download scripts."""
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Spectra Downloader")
        self.resize(600, 400)

        self.layout = QtWidgets.QVBoxLayout(self)

        self.top_layout = QtWidgets.QHBoxLayout()
        self.top_layout.addWidget(QtWidgets.QLabel("Available Sources:"))

        self.source_combo = QtWidgets.QComboBox()
        self.top_layout.addWidget(self.source_combo)

        self.run_btn = QtWidgets.QPushButton("\U000025B6 Run Selected Script")
        self.run_btn.clicked.connect(self.run_script)
        self.top_layout.addWidget(self.run_btn)

        self.layout.addLayout(self.top_layout)

        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        font = QtGui.QFont("Courier" if QtCore.QSysInfo.productType() == "windows" else "Monospace")
        font.setStyleHint(QtGui.QFont.Monospace)
        self.log_output.setFont(font)
        self.layout.addWidget(self.log_output)

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        self.scripts = self.get_available_download_scripts()
        for name in sorted(self.scripts.keys()):
            self.source_combo.addItem(name, self.scripts[name])

        self.process = None

    def get_available_download_scripts(self):
        """Return runnable download scripts keyed by display name."""
        download_dir = Path(__file__).parent / "download"
        scripts = {}
        if download_dir.exists():
            for script_file in download_dir.glob("*.py"):
                if script_file.name == "__init__.py" or script_file.name.startswith("import_"):
                    continue
                if script_file.name.startswith("probe_"):
                    continue
                name = script_file.stem.replace("_", " ").title()
                scripts[name] = script_file
        return scripts

    def run_script(self):
        """Run the selected script against the current database path."""
        if self.process is not None:
            QtWidgets.QMessageBox.warning(self, "Running", "A script is already running.")
            return

        script_path = self.source_combo.currentData()
        if not script_path:
            return

        self.run_btn.setEnabled(False)
        self.log_output.clear()
        self.log_output.appendPlainText(f"--- Running {script_path.name} ---")

        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)

        self.process.start(sys.executable, [str(script_path), "--db", str(self.db.db_path)])

    def handle_stdout(self):
        """Append process output to the log view."""
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.log_output.moveCursor(QtGui.QTextCursor.End)
        self.log_output.insertPlainText(stdout)
        self.log_output.moveCursor(QtGui.QTextCursor.End)

    def process_finished(self):
        """Reset controls after the current script finishes."""
        self.log_output.appendPlainText("--- Finished ---")
        self.process = None
        self.run_btn.setEnabled(True)
