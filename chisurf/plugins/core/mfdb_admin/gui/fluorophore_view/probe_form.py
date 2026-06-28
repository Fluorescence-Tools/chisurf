"""AutoForm-driven read-only detail form for a fluorophore probe.

Replaces the previous dependency on ``mfdb_admin.gui.generic_form`` with the
project's declarative AutoForm + JSON view-scheme machinery (PRD-40). The form
is rendered from ``probe_detail.view.json``; ``set_data`` loads a probe dict
into a lightweight model object and re-syncs the widgets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qtpy import QtWidgets

from chisurf.core.dataspec import load_view_spec
from chisurf.gui.autoform import AutoForm

_VIEW_SCHEME = Path(__file__).with_name("probe_detail.view.json")

# Form fields, in display order — kept in sync with probe_detail.view.json.
_FIELDS = (
    "probe_id",
    "chromophore_name",
    "category",
    "type_name",
    "abs_max",
    "em_max",
    "qy",
    "ext_coeff",
    "lifetime",
    "verification_status",
    "quality",
    "source",
    "source_ref",
    "verified_by",
    "verified_at",
    "description",
)


class _ProbeFormModel:
    """Plain attribute bag bound to the AutoForm value sections.

    AutoForm reads each ``value`` section via ``getattr(model, attr)``. The
    model intentionally has no ``fit`` attribute, so committing a value never
    nudges the fit machinery (the fields are read-only anyway).
    """

    def __init__(self) -> None:
        for name in _FIELDS:
            setattr(self, name, "")

    def view_spec(self):
        return load_view_spec(_VIEW_SCHEME)

    def load(self, data: dict[str, Any]) -> None:
        for name in _FIELDS:
            value = data.get(name)
            setattr(self, name, "" if value is None else str(value))


class ProbeDetailForm(QtWidgets.QWidget):
    """Read-only probe detail panel rendered by AutoForm.

    Drop-in replacement for ``MFDBDetailWidget('probe')`` exposing ``set_data``.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = _ProbeFormModel()
        self._form = AutoForm(self._model, parent=self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._form)

    def set_data(self, data: dict[str, Any]) -> None:
        """Populate the form from a probe dict."""
        self._model.load(data or {})
        self._form.sync_fields()

    def clear(self) -> None:
        self.set_data({})
