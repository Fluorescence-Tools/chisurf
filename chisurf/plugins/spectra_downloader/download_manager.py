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

        # Shows what the selected source already has in the staging DB, with a
        # shortcut to browse that source's scraped components + spectra.
        self.info_layout = QtWidgets.QHBoxLayout()
        self.scraped_info = QtWidgets.QLabel("")
        self.scraped_info.setWordWrap(True)
        self.info_layout.addWidget(self.scraped_info, 1)
        self.browse_source_btn = QtWidgets.QPushButton("🔎 Browse this source")
        self.browse_source_btn.setToolTip("Browse the already-scraped data for the selected source.")
        self.browse_source_btn.clicked.connect(self._browse_selected_source)
        self.info_layout.addWidget(self.browse_source_btn)
        self.layout.addLayout(self.info_layout)

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
        self.browse_btn = QtWidgets.QPushButton("🔎 Browse staging DB")
        self.browse_btn.setToolTip("Inspect the scraped staging database before pushing it.")
        self.browse_btn.clicked.connect(self.browse_staging)
        self.push_layout.addWidget(self.browse_btn)
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
        self.source_combo.currentIndexChanged.connect(self._update_scraped_info)

        self.process = None
        self._update_scraped_info()

    def get_available_download_scripts(self):
        """Return runnable scrapers keyed by display label (from the registry)."""
        from chisurf.plugins.spectra_downloader.download._base import SCRAPERS
        return {spec.label: spec.module for spec in SCRAPERS}

    def run_script(self):
        """Run the selected scraper against the current database path."""
        if self.process is not None:
            QtWidgets.QMessageBox.warning(self, "Running", "A script is already running.")
            return

        module = self.source_combo.currentData()
        if not module:
            return

        self.run_btn.setEnabled(False)
        self.log_output.clear()
        self.log_output.appendPlainText(f"--- Running {module} ---")

        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)

        self.process.start(sys.executable, [
            "-m", f"chisurf.plugins.spectra_downloader.download.{module}",
            "--db", str(self.db.db_path),
        ])

    def _source_slug(self, module):
        """Canonical provenance slug for a scraper module (from the registry)."""
        from chisurf.plugins.spectra_downloader.download._base import get_scraper

        spec = get_scraper(module)
        return spec.source if spec else module

    def _update_scraped_info(self):
        """Show how much of the selected source is already in the staging DB."""
        module = self.source_combo.currentData()
        if not module:
            self.scraped_info.setText("")
            return
        slug = self._source_slug(module)
        try:
            rows = self.db.conn.execute(
                "SELECT category, COUNT(*) n FROM probes "
                "WHERE deleted_at IS NULL AND (',' || IFNULL(source,'') || ',') LIKE ? "
                "GROUP BY category ORDER BY n DESC",
                (f"%,{slug},%",),
            ).fetchall()
        except Exception:
            rows = []
        total = sum(r[1] for r in rows)
        if total:
            by_cat = ", ".join(f"{r[0]}: {r[1]}" for r in rows)
            self.scraped_info.setText(f"Already scraped ({slug}): {total} — {by_cat}")
        else:
            self.scraped_info.setText(f"Already scraped ({slug}): none yet")

    def _browse_selected_source(self):
        """Browse the already-scraped data for the currently selected source."""
        from chisurf.plugins.spectra_downloader.browser import SpectraBrowserDialog

        module = self.source_combo.currentData()
        slug = self._source_slug(module) if module else None
        SpectraBrowserDialog(self.db, self, initial_source=slug).show()

    def browse_staging(self):
        """Open the data browser on the scraped staging database."""
        from chisurf.plugins.spectra_downloader.browser import SpectraBrowserDialog

        SpectraBrowserDialog(self.db, self).show()

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
