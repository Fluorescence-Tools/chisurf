from __future__ import annotations
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# Removed reference to .models since it doesn't exist in gui_parts
HAS_DETECTOR_WIZARD = True
try:
    from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage
except ImportError:
    HAS_DETECTOR_WIZARD = False

class DetectorSelectionWidget(QtWidgets.QGroupBox):
    """Widget for selecting detectors/channels."""
    selectionChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Detector/Channel Selection", parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(1, 1, 1, 1)
        self.layout.setSpacing(1)
        self.checkboxes: Dict[str, QtWidgets.QCheckBox] = {}

    def refresh(self, detector_names: List[str], tttr_data=None):
        """Refresh checkboxes based on detector names or TTTR data."""
        # Clear existing
        for cb in self.checkboxes.values():
            cb.setParent(None)
            cb.deleteLater()
        self.checkboxes.clear()

        # Add new
        for name in detector_names:
            cb = QtWidgets.QCheckBox(str(name))
            cb.setChecked(True)
            cb.stateChanged.connect(self.selectionChanged.emit)
            self.layout.addWidget(cb)
            self.checkboxes[str(name)] = cb

        if not detector_names and tttr_data:
            try:
                # Use micro_times to check if data exists, but routing is what we want
                routing_channels = sorted(set(tttr_data.routing_channels))
                for ch in routing_channels[:8]:
                    name = f"routing_{ch}"
                    cb = QtWidgets.QCheckBox(f"Routing {ch}")
                    cb.setChecked(True)
                    cb.stateChanged.connect(self.selectionChanged.emit)
                    self.layout.addWidget(cb)
                    self.checkboxes[name] = cb
            except Exception:
                pass

    def get_selected(self) -> List[str]:
        return [name for name, cb in self.checkboxes.items() if cb.isChecked()]

class SpeciesListWidget(QtWidgets.QListWidget):
    """List widget for species decays with custom behavior."""
    filesChanged = QtCore.Signal()  # Emitted when files are added/removed (invalidate cache)
    checkStateChanged = QtCore.Signal()  # Emitted when checkboxes toggle (keep cache)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        # Checkbox state changed - don't invalidate cache
        self.checkStateChanged.emit()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            paths = [pathlib.Path(url.toLocalFile()) for url in event.mimeData().urls()]
            self.add_pattern(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
            self.filesChanged.emit()

    def add_pattern(self, paths: List[pathlib.Path]) -> None:
        """Add a new decay pattern from a set of files."""
        if not paths:
            return
            
        valid_paths = [p for p in paths if p.is_file()]
        if not valid_paths:
            return

        # Create a single entry representing this pattern (one or more files)
        name = valid_paths[0].name
        if len(valid_paths) > 1:
            name += f" (+{len(valid_paths)-1} files)"
            
        item = QtWidgets.QListWidgetItem(name)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        tooltip = "\n".join([str(p.absolute()) for p in valid_paths])
        item.setToolTip(tooltip)
        # Store the list of paths in the UserRole
        item.setData(QtCore.Qt.UserRole, valid_paths)
        self.addItem(item)
        self.filesChanged.emit()
