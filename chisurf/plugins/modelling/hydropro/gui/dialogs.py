"""Auxiliary Qt dialogs for the HydroPro tool (output log, download helper)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices, QTextCursor
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class OutputDialog(QDialog):
    """Modeless dialog showing live CLI output and a progress bar."""

    def __init__(self, parent=None, total_steps: int = 0):
        super().__init__(parent)
        self.setWindowTitle("HYDRO Output")
        self.resize(900, 600)
        self.canceled = False

        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setRange(0, total_steps)
        self.progress.setValue(0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        self.clear_btn = QPushButton("Clear")
        self.save_btn = QPushButton("Save Log…")
        self.cancel_btn = QPushButton("Cancel")
        self.clear_btn.clicked.connect(self.log.clear)
        self.save_btn.clicked.connect(self._save_log)
        self.cancel_btn.clicked.connect(self._on_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

    def _on_cancel(self) -> None:
        self.canceled = True
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Cancelling after current job finishes…")

    def _save_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output log", str(Path.home() / "hydro_output.txt"), "Text files (*.txt)"
        )
        if path:
            try:
                Path(path).write_text(self.log.toPlainText(), encoding="utf-8")
            except OSError:
                pass

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_progress(self, value: int) -> None:
        self.progress.setValue(value)

    def append(self, text: str) -> None:
        self.log.appendPlainText(text)
        self.log.moveCursor(QTextCursor.End)


class DownloadInfoDialog(QDialog):
    """Prompt the user to download / select the HYDRO executable."""

    def __init__(self, download_url: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("HYDRO executable required")
        self.resize(560, 260)
        self.selected_path: Optional[Path] = None
        self.dont_show_startup = False

        msg = (
            "HYDROPRO / HYDRO++ executable is not configured.\n\n"
            "Please download the official ZIP archive and select the executable\n"
            "(hydropro10.exe or hydro++10.exe) before running calculations."
        )
        self.info_label = QLabel(msg)
        self.info_label.setWordWrap(True)
        self.path_display = QLineEdit("")
        self.path_display.setReadOnly(True)

        self.btn_download = QPushButton("Open download page")
        self.btn_select = QPushButton("Select executable…")
        self.btn_close = QPushButton("Close")
        self.chk_no_startup = QCheckBox("Don't show this on startup")

        self.btn_download.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(download_url)))
        self.btn_select.clicked.connect(self._select_exe)
        self.btn_close.clicked.connect(self.accept)
        self.chk_no_startup.toggled.connect(self._set_no_startup)

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_display)
        path_row.addWidget(self.btn_select)
        layout.addLayout(path_row)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.chk_no_startup)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_download)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    def _set_no_startup(self, checked: bool) -> None:
        self.dont_show_startup = bool(checked)

    def _select_exe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HYDRO executable", str(Path.home()),
            "Executables (*.exe);;All files (*.*)",
        )
        if path:
            self.selected_path = Path(path)
            self.path_display.setText(path)


__all__ = ["OutputDialog", "DownloadInfoDialog"]
