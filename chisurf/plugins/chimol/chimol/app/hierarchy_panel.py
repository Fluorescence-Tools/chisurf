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
            # Could add icons based on node_type
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
        if node == self._root_node or node.parent is None:
            return QtCore.QModelIndex()
        
        parent_node = node.parent
        grandparent_node = parent_node.parent
        
        if grandparent_node is None:
            # parent_node must be root
            return self.createIndex(0, 0, parent_node)
        
        # We need the row of parent_node within grandparent_node
        row = 0
        for i, child in enumerate(grandparent_node.children):
            if child == parent_node:
                row = i
                break
        return self.createIndex(row, 0, parent_node)

    # A better way is to store parent pointers in RmfHierarchyNode or use a flat map.
    # For now, let's keep it simple and just show the hierarchy.


class HierarchyDock(QtWidgets.QDockWidget):
    """Dock widget for RMF hierarchy navigation."""

    def __init__(self, parent=None):
        super().__init__("Hierarchy", parent)
        self.setObjectName("ChimolHierarchyDock")
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        self.tree_view = QtWidgets.QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.model = HierarchyModel()
        self.tree_view.setModel(self.model)
        
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.tree_view)
        
        self.setWidget(container)

    def set_hierarchy(self, root: Optional[RmfHierarchyNode]):
        self.model.set_root_node(root)
        if root:
            self.tree_view.expandToDepth(1)
