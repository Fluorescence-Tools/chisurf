from __future__ import annotations

from typing import Any, Dict, Optional

from qtpy import QtWidgets, QtCore

from ..colors import _OBJECT_ID_ROLE


class ObjectsDock(QtCore.QObject):
    """Objects panel — content widget (no outer QDockWidget wrapper)."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        margins: tuple[int, int, int, int],
        spacing: int,
    ) -> None:
        super().__init__(parent)
        self._widget = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(self._widget)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)

        objects_group = QtWidgets.QGroupBox("Loaded Molecules", self._widget)
        objects_layout = QtWidgets.QVBoxLayout(objects_group)
        objects_layout.setContentsMargins(4, 8, 4, 4)
        objects_layout.setSpacing(4)

        self.object_list = QtWidgets.QListWidget(objects_group)
        self.object_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection,
        )
        objects_layout.addWidget(self.object_list)
        layout.addWidget(objects_group)

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._widget

    def create_item(
        self,
        object_id: str,
        entry: Dict[str, Any],
    ) -> QtWidgets.QListWidgetItem:
        item = QtWidgets.QListWidgetItem(entry.get("name", object_id))
        item.setFlags(
            item.flags()
            | QtCore.Qt.ItemIsUserCheckable
            | QtCore.Qt.ItemIsSelectable
        )
        item.setCheckState(QtCore.Qt.Checked)
        item.setData(_OBJECT_ID_ROLE, object_id)
        return item

    def set_current_object(self, object_id: Optional[str]) -> None:
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if item is None:
                continue
            oid = item.data(_OBJECT_ID_ROLE)
            if oid == object_id:
                self.object_list.setCurrentItem(item)
                return

    def set_item_checked(self, object_id: str, checked: bool) -> None:
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if item is None:
                continue
            oid = item.data(_OBJECT_ID_ROLE)
            if oid == object_id:
                item.setCheckState(
                    QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked,
                )
                return
