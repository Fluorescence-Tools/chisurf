"""AutoForm-backed entity detail form.

Drop-in replacement for :class:`generic_form.MFDBDetailWidget` that renders an
entity's :class:`entity_schema.FieldSpec` list through the project's declarative
AutoForm machinery (PRD-40) instead of a hand-rolled ``QFormLayout``.

Public surface matches ``MFDBDetailWidget`` so ``mixins.FormMixin`` can use it
unchanged: ``commitRequested`` / ``dataChanged`` signals, ``set_data(dict)``,
``get_data() -> dict`` and ``refresh_dropdowns()``.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from qtpy import QtCore, QtWidgets

import chisurf.core.dataspec as ds
from chisurf.gui.autoform import AutoForm

# Fields that are stored as JSON in the DB but edited as text (mirrors
# generic_form.MFDBDetailWidget.get_data special-casing).
_JSON_LIST_FIELDS = {"laser_wavelengths"}
_JSON_DICT_FIELDS = {"detector_channels"}


def _section_for_field(
    fs: Any,
    dropdown_providers: dict[str, Callable[[], list[tuple[str, str]]]],
) -> ds.Section:
    """Map a FieldSpec to an AutoForm section."""
    name = fs.name
    label = fs.label or name
    tooltip = getattr(fs, "tooltip", "") or ""
    placeholder = getattr(fs, "placeholder", "") or ""
    read_only = bool(getattr(fs, "readonly", False))
    widget = getattr(fs, "widget", "str")

    # Foreign-key field → dropdown whose options are fetched once at build time.
    if getattr(fs, "fk_target", None) and name in dropdown_providers:
        options: tuple[str, ...] = ()
        labels: tuple[str, ...] = ()
        try:
            pairs = dropdown_providers[name]() or []
            options = tuple(str(v) for v, _ in pairs)
            labels = tuple(str(lbl) for _, lbl in pairs)
        except Exception:
            options, labels = (), ()
        return ds.ChoiceSection(
            attr=name, label=label, options=options, labels=labels, description=tooltip
        )

    if widget == "choice":
        return ds.ChoiceSection(
            attr=name, label=label, options=tuple(getattr(fs, "choices", ())),
            description=tooltip,
        )
    if widget == "bool":
        return ds.ToggleSection(attr=name, label=label, description=tooltip)
    if widget in ("int", "float"):
        return ds.ValueSection(
            attr=name, label=label, kind=widget, read_only=read_only, description=tooltip
        )
    if widget == "text":
        return ds.ValueSection(
            attr=name, label=label, kind="text", read_only=read_only,
            placeholder=placeholder, description=tooltip,
        )
    if widget == "date":
        return ds.ValueSection(
            attr=name, label=label, kind="date", read_only=read_only, description=tooltip
        )
    # str / fallback
    return ds.ValueSection(
        attr=name, label=label, kind="str", read_only=read_only,
        placeholder=placeholder, description=tooltip,
    )


class _EntityModel:
    """Attribute bag bound to the AutoForm; notifies on field commit.

    AutoForm reads/writes each section via ``getattr``/``setattr`` on this
    object. A ``setattr`` to a known field (outside a load) means the user
    committed an edit, so we fire ``on_commit`` to drive auto-save — matching
    ``MFDBDetailWidget``'s ``commitRequested`` semantics.
    """

    def __init__(self, initial, view, on_commit) -> None:
        # ``initial`` maps each field name to a TYPED default (0 for int, 0.0 for
        # float, False for bool, "" otherwise) so the AutoForm editors build
        # without a coercion error (e.g. int("") would fail).
        object.__setattr__(self, "_field_names", set(initial))
        object.__setattr__(self, "_view", view)
        object.__setattr__(self, "_on_commit", on_commit)
        object.__setattr__(self, "_suspend", True)
        for name, value in initial.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_suspend", False)

    def view_spec(self) -> ds.ModelView:
        return self._view

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if (
            name in getattr(self, "_field_names", ())
            and not getattr(self, "_suspend", True)
        ):
            cb = getattr(self, "_on_commit", None)
            if cb is not None:
                cb()


class EntityForm(QtWidgets.QWidget):
    """AutoForm-rendered entity detail form (drop-in for MFDBDetailWidget)."""

    dataChanged = QtCore.Signal()
    commitRequested = QtCore.Signal()

    def __init__(
        self,
        field_specs: list[Any],
        dropdown_providers: dict[str, Callable[[], list[tuple[str, str]]]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._specs = list(field_specs)
        self._providers = dropdown_providers or {}
        self._spec_by_name = {fs.name: fs for fs in self._specs}

        sections = tuple(_section_for_field(fs, self._providers) for fs in self._specs)
        view = ds.ModelView(sections=sections)
        initial = {fs.name: self._default_for(fs) for fs in self._specs}
        self._model = _EntityModel(initial, view, self._on_commit)
        self._form = AutoForm(self._model, parent=self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._form)

    # -- signal plumbing -------------------------------------------------
    def _on_commit(self) -> None:
        self.dataChanged.emit()
        self.commitRequested.emit()

    def refresh_dropdowns(self) -> None:
        """No-op: FK options are fetched once at build time (see _section_for_field)."""

    # -- data in/out -----------------------------------------------------
    def set_data(self, data: dict[str, Any]) -> None:
        """Load a record dict into the form (no commit signals during load)."""
        object.__setattr__(self._model, "_suspend", True)
        try:
            for fs in self._specs:
                object.__setattr__(
                    self._model, fs.name, self._coerce_in(fs, data.get(fs.name))
                )
            self._form.sync_fields()
        finally:
            object.__setattr__(self._model, "_suspend", False)

    def get_data(self) -> dict[str, Any]:
        """Collect form values into a record dict (matching MFDBDetailWidget)."""
        out: dict[str, Any] = {}
        for fs in self._specs:
            out[fs.name] = self._coerce_out(fs, getattr(self._model, fs.name, None))
        return out

    # -- coercion (mirrors generic_form.MFDBDetailWidget) ----------------
    @staticmethod
    def _default_for(fs: Any) -> Any:
        """Typed empty default so AutoForm editors build without coercion errors."""
        widget = getattr(fs, "widget", "str")
        if widget == "int":
            return 0
        if widget == "float":
            return 0.0
        if widget == "bool":
            return False
        return ""

    @staticmethod
    def _coerce_in(fs: Any, val: Any) -> Any:
        widget = getattr(fs, "widget", "str")
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        if widget == "int":
            try:
                return int(val) if val not in (None, "") else 0
            except (TypeError, ValueError):
                return 0
        if widget == "float":
            try:
                return float(val) if val not in (None, "") else 0.0
            except (TypeError, ValueError):
                return 0.0
        if widget == "bool":
            return bool(val)
        return "" if val is None else str(val)

    def _coerce_out(self, fs: Any, raw: Any) -> Any:
        name = fs.name
        widget = getattr(fs, "widget", "str")
        if widget == "bool":
            val: Any = 1 if raw else 0
        elif widget == "int":
            try:
                val = int(raw)
            except (TypeError, ValueError):
                val = 0
        elif widget == "float":
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = 0.0
        else:
            val = (str(raw).strip() or None) if raw is not None else None

        if name in _JSON_LIST_FIELDS:
            try:
                return json.loads(val) if val else []
            except Exception:
                return []
        if name in _JSON_DICT_FIELDS:
            try:
                return json.loads(val) if val else {}
            except Exception:
                return {}
        return val
