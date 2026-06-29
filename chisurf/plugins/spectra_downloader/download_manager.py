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

        # Scrapers write to their own staging DB; this row pushes that staging DB
        # into the connected MFDB (the explicit stage-3 integration step).
        self.push_layout = QtWidgets.QHBoxLayout()
        self.push_layout.addWidget(QtWidgets.QLabel("Staging → MFDB:"))
        self.replace_check = QtWidgets.QCheckBox("Replace existing")
        self.replace_check.setToolTip(
            "Purge the existing reference probes in the MFDB before importing "
            "(a backup is made first)."
        )
        self.push_layout.addWidget(self.replace_check)
        self.push_layout.addStretch()
        self.push_btn = QtWidgets.QPushButton("⬆ Push to MFDB")
        self.push_btn.setToolTip("Integrate the scraped staging database into the connected MFDB.")
        self.push_btn.clicked.connect(self.push_to_mfdb)
        self.push_layout.addWidget(self.push_btn)
        self.layout.addLayout(self.push_layout)

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

    # Helper modules in download/ that are NOT standalone source scrapers (they
    # need extra arguments and must not appear in the "Run script" list).
    _NON_SCRAPER_MODULES = {"archive_recovery", "merge", "count_proteins", "count_spectra"}

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
                if script_file.stem in self._NON_SCRAPER_MODULES:
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

    def push_to_mfdb(self):
        """Push the scraped staging database into the connected MFDB."""
        if self.process is not None:
            QtWidgets.QMessageBox.warning(self, "Running", "A task is already running.")
            return

        replace = self.replace_check.isChecked()
        if replace:
            ok = QtWidgets.QMessageBox.question(
                self,
                "Replace MFDB reference set",
                "This purges the existing reference probes in the connected MFDB "
                "and re-imports them from the scraped staging database.\n\n"
                "A backup of the MFDB is made first. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if ok != QtWidgets.QMessageBox.Yes:
                return

        self.run_btn.setEnabled(False)
        self.push_btn.setEnabled(False)
        self.log_output.clear()
        self.log_output.appendPlainText("--- Pushing staging DB to MFDB ---")

        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)

        args = ["-m", "chisurf.plugins.spectra_downloader.cli", "push",
                "--staging", str(self.db.db_path)]
        if replace:
            args.append("--replace")
        self.process.start(sys.executable, args)

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
        self.push_btn.setEnabled(True)
