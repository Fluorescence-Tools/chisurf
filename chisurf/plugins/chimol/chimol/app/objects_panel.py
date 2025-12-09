from __future__ import annotations

from typing import Any, Dict, Optional

from qtpy import QtWidgets, QtCore

from ..colors import _OBJECT_ID_ROLE


class ObjectsDock(QtCore.QObject):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        margins: tuple[int, int, int, int],
        spacing: int,
    ) -> None:
        super().__init__(parent)
        self._dock = QtWidgets.QDockWidget("Objects", parent)
        self._dock.setObjectName("ChimolObjectsDock")
        self._dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )

        objects_group = QtWidgets.QGroupBox("Loaded Molecules", parent)
        objects_layout = QtWidgets.QVBoxLayout(objects_group)
        objects_layout.setContentsMargins(4, 8, 4, 4)
        objects_layout.setSpacing(4)

        self.object_list = QtWidgets.QListWidget(objects_group)
        self.object_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        objects_layout.addWidget(self.object_list)

        objects_widget = QtWidgets.QWidget(parent)
        objects_widget_layout = QtWidgets.QVBoxLayout(objects_widget)
        objects_widget_layout.setContentsMargins(*margins)
        objects_widget_layout.setSpacing(spacing)
        objects_widget_layout.addWidget(objects_group)

        self._dock.setWidget(objects_widget)

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget:
        return self._dock

    def create_item(self, object_id: str, entry: Dict[str, Any]) -> QtWidgets.QListWidgetItem:
        item = QtWidgets.QListWidgetItem(entry.get("name", object_id))
        item.setFlags(
            item.flags()
            | QtCore.Qt.ItemIsUserCheckable
            | QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEnabled
        )
        item.setData(_OBJECT_ID_ROLE, object_id)
        path = entry.get("path")
        if path:
            item.setToolTip(str(path))
        return item

    def find_item(self, object_id: str) -> Optional[QtWidgets.QListWidgetItem]:
        if not object_id:
            return None
        target = str(object_id)
        for row in range(self.object_list.count()):
            item = self.object_list.item(row)
            if item is None:
                continue
            data = item.data(_OBJECT_ID_ROLE)
            if str(data) == target:
                return item
        return None

    def set_item_checked(self, object_id: str, checked: bool) -> bool:
        item = self.find_item(object_id)
        if item is None:
            return False
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        return True

    def set_current_object(self, object_id: str) -> bool:
        item = self.find_item(object_id)
        if item is None:
            return False
        self.object_list.setCurrentItem(item)
        return True
