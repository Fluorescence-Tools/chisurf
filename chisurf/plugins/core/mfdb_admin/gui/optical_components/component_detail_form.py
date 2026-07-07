"""AutoForm-driven read-only detail form for optical components.

Dynamically loaded from a per-component view scheme.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qtpy import QtWidgets

from chisurf.core.dataspec import load_view_spec
from chisurf.gui.autoform import AutoForm


class _ComponentFormModel:
    """Dynamic attribute bag bound to the AutoForm value sections.

    Determines fields by reading the view scheme json file.
    """

    def __init__(self, view_scheme_path: Path) -> None:
        self._view_scheme_path = view_scheme_path
        self._fields: list[str] = []

        try:
            with open(view_scheme_path, "r", encoding="utf-8") as f:
                scheme = json.load(f)
            for section in scheme.get("sections", []):
                attr = section.get("attr")
                if attr:
                    self._fields.append(attr)
                    setattr(self, attr, "")
        except Exception as e:
            print(f"Error loading view scheme fields from {view_scheme_path}: {e}")

    def view_spec(self):
        return load_view_spec(self._view_scheme_path)

    def load(self, data: dict[str, Any]) -> None:
        for name in self._fields:
            value = data.get(name)
            setattr(self, name, "" if value is None else str(value))


class ComponentDetailForm(QtWidgets.QWidget):
    """Read-only detail panel rendered by AutoForm based on a view scheme."""

    def __init__(self, view_scheme_path: Path, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = _ComponentFormModel(view_scheme_path)
        self._form = AutoForm(self._model, parent=self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._form)

    def set_data(self, data: dict[str, Any]) -> None:
        """Populate the form from a probe/component dict."""
        self._model.load(data or {})
        self._form.sync_fields()

    def clear(self) -> None:
        self.set_data({})
