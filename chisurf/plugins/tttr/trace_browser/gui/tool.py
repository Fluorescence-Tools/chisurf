"""Main Trace Browser GUI entrypoint with toolbar and dock layout."""

from __future__ import annotations

from qtpy.QtWidgets import (
    QAction,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QMainWindow,
    QPlainTextEdit,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from chisurf.gui.widgets.dock_area.dock_area import DockArea
from chisurf.plugins.tttr.trace_browser import TraceBrowser
from chisurf.plugins.tttr.trace_browser.gui.client import TraceBrowserClient

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    def _persist_plugin_state(_name: str):
        def decorator(cls):
            return cls
        return decorator
    persist_plugin_state = _persist_plugin_state


@persist_plugin_state("trace_browser")
class TraceBrowserTool(QMainWindow):
    """Toolbar-backed Trace Browser window."""

    def __init__(self, parent=None):
        """Create the toolbar/dock shell and embed the Trace Browser workspace."""
        super().__init__(parent)
        self.setWindowTitle("🔎 Trace Browser")
        self._workspace = TraceBrowser(self)
        self.client = TraceBrowserClient()
        self._dock_area = DockArea(self)
        self._dock_area.addTab(self._workspace, "📈 Traces")
        self.setCentralWidget(self._dock_area)
        self._setup_toolbar()

    def __getattr__(self, name: str):
        """Delegate workspace attributes for legacy tests and callers."""
        return getattr(self._workspace, name)

    def show(self):
        """Show the window."""
        super().show()

    def raise_(self):
        """Raise the window."""
        super().raise_()

    def activateWindow(self):
        """Activate the window."""
        super().activateWindow()

    def _setup_toolbar(self) -> None:
        """Create emoji toolbar actions backed by the workspace."""
        toolbar = QToolBar("🧰 Trace Browser", self)
        toolbar.setObjectName("traceBrowserMainToolbar")
        actions = [
            ("📂 Open", self._workspace._on_pick_folder, "Pick a folder with TTTR traces"),
            ("🧹 Clear", self._workspace._on_clear, "Clear the file list"),
            ("♻ Caches", self._workspace._on_clear_caches, "Clear trace caches"),
            ("📤 Export", self._workspace._on_export, "Export selected traces"),
            ("CSV", self._workspace._on_export_csv, "Export traces as CSV"),
            ("DOCX", self._workspace._on_export_docx, "Export selected traces as DOCX"),
            ("🧠 HMM", self._workspace._on_transfer_to_analysis, "Open in Intensity Trace Analysis"),
            ("⏱ TW", self._workspace._on_transfer_to_tw, "Open in TTTR Time Window"),
            ("📊 NDX", self._workspace._on_open_in_ndxplorer, "Open in NDXplorer"),
        ]
        for text, slot, tooltip in actions:
            action = QAction(text, self)
            action.setToolTip(tooltip)
            action.triggered.connect(slot)
            toolbar.addAction(action)
        chk_subfolders = QCheckBox("Subfolders", self)
        chk_subfolders.setChecked(self._workspace.chk_subfolders.isChecked())
        chk_subfolders.setToolTip("Include subfolders when opening a folder")
        chk_subfolders.toggled.connect(self._workspace._on_subfolders_toggled)
        toolbar.addWidget(chk_subfolders)
        toolbar.addSeparator()
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        help_action = QAction("❔ Help", self)
        help_action.setToolTip("Show what Trace Browser does, how the toolbar works, and how to use the CLI")
        help_action.triggered.connect(self._show_help)
        toolbar.addAction(help_action)
        self.addToolBar(toolbar)

    def _show_help(self) -> None:
        """Show Trace Browser usage and CLI help."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Trace Browser Help")
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit(dialog)
        text.setReadOnly(True)
        text.setPlainText(
            "Trace Browser\n\n"
            "Browse PTU/TTTR trace files from a folder, rate and annotate traces, preview binned intensity traces, "
            "and export selected traces. The toolbar replaces the old inline tool buttons:\n\n"
            "• 📂 Open: choose a folder containing TTTR traces\n"
            "• 🧹 Clear: clear the current file list\n"
            "• ♻ Caches: clear in-memory and on-disk trace caches\n"
            "• 📤 Export: copy selected raw trace files\n"
            "• CSV: export binned intensity traces as CSV files\n"
            "• DOCX: export selected traces and annotations as a DOCX report\n"
            "• 🧠 HMM: open the selected trace in Intensity Trace Analysis\n"
            "• ⏱ TW: open the selected trace in TTTR Time Window\n"
            "• 📊 NDX: compute burst analysis from the current time window and open NDXplorer\n\n"
            "The GUI talks to the Trace Browser backend through RPC for file listing, metadata, trace loading, "
            "and CSV export. The command line interface is the same plugin entry point and uses Click:\n\n"
            "  trace-browser list FOLDER [--recursive]\n"
            "  trace-browser load FILE [--window-ms 10]\n"
            "  trace-browser export-csv FILE [FILE ...] --output-dir DIR [--window-ms 10]\n"
            "  trace-browser contract\n"
        )
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, dialog)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()
