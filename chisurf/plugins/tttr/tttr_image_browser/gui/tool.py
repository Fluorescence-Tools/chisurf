"""Main TTTR Image Browser GUI entrypoint with toolbar and dock layout."""

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
from chisurf.plugins.tttr.tttr_image_browser import TTTRImageBrowser
from chisurf.plugins.tttr.tttr_image_browser.gui.client import TTTRImageBrowserClient

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    def _persist_plugin_state(_name: str):
        def decorator(cls):
            return cls
        return decorator
    persist_plugin_state = _persist_plugin_state


@persist_plugin_state("tttr_image_browser")
class TTTRImageBrowserTool(QMainWindow):
    """Toolbar-backed TTTR Image Browser window."""

    def __init__(self, parent=None):
        """Create the toolbar/dock shell and embed the TTTR Image Browser workspace.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget, by default None.
        """
        super().__init__(parent)
        self.setWindowTitle("🖼 TTTR Image Browser")
        self._workspace = TTTRImageBrowser(self)
        self.client = TTTRImageBrowserClient()
        self._dock_area = DockArea(self)
        self._dock_area.addTab(self._workspace, "📈 Images")
        self.setCentralWidget(self._dock_area)
        self._setup_toolbar()
        # When embedded in Image Tools, selecting an image proactively warms the
        # imaging pipeline (background prefill) so later steps are ready.
        try:
            self._workspace.table.itemSelectionChanged.connect(self._on_image_selected)
        except Exception:
            pass

    def _on_image_selected(self) -> None:
        """Tell the imaging pipeline which image is selected (triggers prefill)."""
        coordinator = getattr(self, "_coordinator", None)
        if coordinator is None:
            return
        source = getattr(self._workspace, "_current_file", None)
        if source is None:
            try:
                paths = self._workspace._selected_paths()
                source = paths[0] if paths else None
            except Exception:
                source = None
        if source:
            try:
                coordinator.set_pipeline(source=str(source))
            except Exception:
                pass

    def __getattr__(self, name: str):
        """Delegate workspace attributes for legacy tests and callers.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Any
            The workspace attribute.
        """
        return getattr(self._workspace, name)

    def _on_next_step(self) -> None:
        """Send the current/selected image to the imaging pipeline (Intensity step)."""
        coordinator = getattr(self, "_coordinator", None)
        if coordinator is None:
            return
        source = None
        try:
            source = getattr(self._workspace, "_current_file", None)
            if source is None:
                paths = self._workspace._selected_paths()
                source = paths[0] if paths else None
        except Exception:
            source = None
        if source:
            coordinator.set_pipeline(source=str(source))
        coordinator.goto_role("pixel_intensity")
        # Auto-compute + auto-create the intensity HDF5 (the browser already
        # showed the image, so no separate manual Run is needed).
        try:
            coordinator.autorun_role("pixel_intensity")
        except Exception:
            pass

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
        toolbar = QToolBar("🧰 TTTR Image Browser", self)
        toolbar.setObjectName("tttrImageBrowserMainToolbar")
        actions = [
            ("📂 Open", self._workspace._on_pick_folder, "Pick a folder with TTTR images"),
            ("🧹 Clear", self._workspace._on_clear, "Clear the file list"),
            ("♻ Caches", self._workspace._on_clear_caches, "Clear image caches"),
            ("📤 Export", self._workspace._on_export, "Export selected image files"),
            ("TIFF", self._workspace._on_save_tiff, "Save intensity images as TIFF stacks"),
            ("DOCX", self._workspace._on_export_docx, "Export selected images as DOCX"),
        ]
        for text, slot, tooltip in actions:
            action = QAction(text, self)
            action.setToolTip(tooltip)
            action.triggered.connect(slot)
            toolbar.addAction(action)

        # Hand the browsed image off to the imaging pipeline (when embedded in
        # the Image Tools shell). Steps remain freely navigable on the left.
        toolbar.addSeparator()
        next_action = QAction("Next ▶ Intensity", self)
        next_action.setToolTip("Send the current image to the imaging pipeline (Intensity step).")
        next_action.triggered.connect(self._on_next_step)
        toolbar.addAction(next_action)

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
        help_action.setToolTip("Show what TTTR Image Browser does, how the toolbar works, and how to use the CLI")
        help_action.triggered.connect(self._show_help)
        toolbar.addAction(help_action)
        self.addToolBar(toolbar)

    def _show_help(self) -> None:
        """Show TTTR Image Browser usage and CLI help."""
        dialog = QDialog(self)
        dialog.setWindowTitle("TTTR Image Browser Help")
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit(dialog)
        text.setReadOnly(True)
        text.setPlainText(
            "TTTR Image Browser\n\n"
            "Browse TTTR files in a folder and preview intensity images for all DetectorWizard-defined "
            "detector windows. The toolbar replaces the old inline buttons:\n\n"
            "• 📂 Open: choose a folder containing TTTR images\n"
            "• 🧹 Clear: clear the current file list\n"
            "• ♻ Caches: clear in-memory and on-disk image caches\n"
            "• 📤 Export: copy selected raw image files\n"
            "• TIFF: export intensity images as TIFF stacks\n"
            "• DOCX: export selected images and annotations as a DOCX report\n\n"
            "The GUI talks to the TTTR Image Browser backend through RPC for file listing, metadata, "
            "and image loading. The command line interface uses Click:\n\n"
            "  tttr-image-browser list FOLDER [--recursive]\n"
            "  tttr-image-browser load FILE [--max-side 512]\n"
            "  tttr-image-browser export-tiff FILE [FILE ...] --output-dir DIR\n"
            "  tttr-image-browser contract\n"
        )
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, dialog)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()
