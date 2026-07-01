"""Embeddable check-list of already-loaded ChiSurf datasets.

Dropped into the batch wizard via an ``embed`` custom section. Reads the model's
:meth:`BatchViewModel.imported_datasets` and writes the checked indices back to
``model.selected_dataset_indices`` (then calls ``model.update()``), so all state
lives on the Qt-free view-model and the wizard stays declarative.
"""

from __future__ import annotations

from qtpy import QtCore, QtWidgets


class LoadedDatasetSelector(QtWidgets.QWidget):
    """A refreshable, checkable list bound to ``model.selected_dataset_indices``."""

    _autoform_expanding = True

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._list = QtWidgets.QListWidget()
        layout.addWidget(self._list, 1)

        bar = QtWidgets.QHBoxLayout()
        refresh = QtWidgets.QToolButton()
        refresh.setText("🔄 Refresh")
        refresh.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        refresh.setToolTip("Re-scan the datasets currently loaded in ChiSurf.")
        refresh.clicked.connect(self.repopulate)
        bar.addWidget(refresh)
        bar.addStretch(1)
        layout.addLayout(bar)

        self._list.itemChanged.connect(self._commit)
        self.repopulate()

    def repopulate(self) -> None:
        """Rebuild the list from the model's imported datasets, keeping checks."""
        checked = set(getattr(self._model, "selected_dataset_indices", []) or [])
        self._list.blockSignals(True)
        self._list.clear()
        for idx, ds in enumerate(self._model.imported_datasets()):
            name = (
                getattr(ds, "name", None) or getattr(ds, "filename", None) or f"Dataset {idx + 1}"
            )
            item = QtWidgets.QListWidgetItem(f"{idx + 1}. {name}")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setData(QtCore.Qt.UserRole, idx)
            item.setCheckState(QtCore.Qt.Checked if idx in checked else QtCore.Qt.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._commit()

    def _commit(self, *_args) -> None:
        indices = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                indices.append(int(item.data(QtCore.Qt.UserRole)))
        self._model.selected_dataset_indices = indices
        try:
            self._model.update()
        except Exception:
            pass


__all__ = ["LoadedDatasetSelector"]
