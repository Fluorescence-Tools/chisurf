from __future__ import annotations

import json
from pathlib import Path

from qtpy import QtCore, QtWidgets


class WidgetPalette(QtWidgets.QTreeWidget):
    """Palette that lists available node types grouped from a JSON file.

    Each leaf item represents a node type (e.g. "constant", "text_entry").
    Activating an item (double-click or Enter) emits ``nodeTypeActivated`` with
    the node_type string, allowing the owning editor to create a node in the
    scene.
    """

    nodeTypeActivated = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None, palette_path: str | None = None) -> None:
        super().__init__(parent)

        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setAnimated(True)
        self.setIndentation(14)

        self._load_palette(palette_path)

        self.itemActivated.connect(self._on_item_activated)
        self.itemDoubleClicked.connect(self._on_item_activated)

    # ----- Internal helpers ----------------------------------------------
    def _load_palette(self, palette_path: str | None) -> None:
        """Load groups/items from a JSON file.

        The JSON format is::

            {
              "groups": [
                {
                  "title": "Numeric",
                  "items": [
                    {"node_type": "constant", "label": "Constant"},
                    ...
                  ]
                },
                ...
              ]
            }
        """

        self.clear()

        base = Path(__file__).resolve().parent.parent  # widgets/ -> node_editor/
        theme_dir = base / "theme"
        if palette_path is None:
            path = theme_dir / "widgets_palette.json"
        else:
            p = Path(palette_path)
            path = p if p.is_absolute() else (theme_dir / p)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        groups = data.get("groups", [])
        for group in groups:
            title = str(group.get("title", "")).strip() or "Group"
            group_item = QtWidgets.QTreeWidgetItem([title])
            # Group headers are not selectable to avoid confusing activations
            flags = group_item.flags()
            group_item.setFlags(flags & ~QtCore.Qt.ItemIsSelectable)
            self.addTopLevelItem(group_item)

            items = group.get("items", [])
            for item in items:
                node_type = str(item.get("node_type", "")).strip()
                if not node_type:
                    continue
                label = str(item.get("label") or node_type).strip()
                child = QtWidgets.QTreeWidgetItem([label])
                child.setData(0, QtCore.Qt.UserRole, node_type)
                group_item.addChild(child)

            group_item.setExpanded(True)

    def _on_item_activated(self, item: QtWidgets.QTreeWidgetItem) -> None:  # type: ignore[override]
        if item is None:
            return
        node_type = item.data(0, QtCore.Qt.UserRole)
        if not node_type:
            return
        self.nodeTypeActivated.emit(str(node_type))
