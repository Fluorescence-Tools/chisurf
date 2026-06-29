import sys
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets


class DownloadPanel(QtWidgets.QWidget):
    """Run scrapers into the staging DB and push it to the MFDB.

    A plain ``QWidget`` so it can live as a panel inside the Spectra tool's
    navigation shell, or be wrapped in :class:`DownloadManagerDialog`.
    """

    def __init__(self, db, parent=None):
        """Create the panel and populate available download scripts."""
        super().__init__(parent)
        self.db = db

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

        # Adding the staging DB into the MFDB is its own dedicated panel
        # (endpoint + authentication) — see gui/add_to_mfdb_panel.py.

        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        font = QtGui.QFont("Courier" if QtCore.QSysInfo.productType() == "windows" else "Monospace")
        font.setStyleHint(QtGui.QFont.Monospace)
        self.log_output.setFont(font)
        self.layout.addWidget(self.log_output)

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


class DownloadManagerDialog(QtWidgets.QDialog):
    """Standalone dialog wrapper around :class:`DownloadPanel` (back-compat)."""

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectra Downloader")
        self.resize(640, 460)
        self.db = db
        layout = QtWidgets.QVBoxLayout(self)
        self.panel = DownloadPanel(db, self)
        layout.addWidget(self.panel)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
