from qtpy import QtCore, QtWidgets, QtGui
from typing import Optional, List, Any
from ..io.rmf import RmfHierarchyNode


class HierarchyModel(QtCore.QAbstractItemModel):
    """Qt model to wrap RmfHierarchyNode tree."""

    def __init__(self, root_node: Optional[RmfHierarchyNode] = None, parent=None):
        super().__init__(parent)
        self._root_node = root_node

    def set_root_node(self, node: Optional[RmfHierarchyNode]):
        self.beginResetModel()
        self._root_node = node
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if not self._root_node:
            return 0
        if not parent.isValid():
            return 1  # Root
        node = parent.internalPointer()
        return len(node.children)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 1

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        node = index.internalPointer()
        if role == QtCore.Qt.DisplayRole:
            return f"{node.name} [{node.node_type}]"
        elif role == QtCore.Qt.DecorationRole:
            pass
        return None

    def index(self, row, column, parent=QtCore.QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()
        if not parent.isValid():
            return self.createIndex(row, column, self._root_node)
        parent_node = parent.internalPointer()
        if row < len(parent_node.children):
            child_node = parent_node.children[row]
            return self.createIndex(row, column, child_node)
        return QtCore.QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()
        node = index.internalPointer()
        parent_node = node.parent
        if parent_node is None or parent_node == self._root_node:
            return QtCore.QModelIndex()
        return self.createIndex(0, 0, parent_node)


class HierarchyDock(QtWidgets.QWidget):
    """Hierarchy panel — content widget (no outer QDockWidget wrapper)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._model = HierarchyModel()
        self._tree = QtWidgets.QTreeView(self)
        self._tree.setModel(self._model)
        self._tree.setHeaderHidden(True)
        self._tree.setAnimated(True)
        self._tree.setIndentation(16)
        self._tree.setExpandsOnDoubleClick(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)

    def set_hierarchy(self, root_node: Optional[RmfHierarchyNode]) -> None:
        self._model.set_root_node(root_node)
        if root_node is not None:
            self._tree.expandToDepth(1)

    @property
    def model(self) -> HierarchyModel:
        return self._model

    @property
    def tree_view(self) -> QtWidgets.QTreeView:
        return self._tree
