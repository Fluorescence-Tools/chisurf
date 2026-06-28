"""Built-in registrations mapping view-spec keys to ChiSurf widgets/plots.

Importing this module wires the standard plot keys (``line``, ``residual``,
``distribution`` ...) and standard custom sections into the registry. The model
layer references these by string only; the concrete classes live here.
"""

from __future__ import annotations

from qtpy import QtCore, QtWidgets

import chisurf as cs
import chisurf.core.math.datatools
from chisurf import logging

from .registry import register_plot, register_section


# --- plots -----------------------------------------------------------------
# Registered as zero-arg factories so chisurf.gui.plots is only imported when a
# plot is actually resolved.
def _plots():
    import chisurf.gui.plots as _p

    return _p


register_plot("line", lambda: _plots().LinePlot)
register_plot("residual", lambda: _plots().ResidualPlot)
register_plot("fit_info", lambda: _plots().FitInfo)
register_plot("fit_table", lambda: _plots().FitTablePlot)
register_plot("parameter_scan", lambda: _plots().ParameterScanPlot)
register_plot("distribution", lambda: _plots().DistributionPlot)


def resolve_distribution_options(options: dict) -> dict:
    """Resolve string accessors in distribution-plot options to callables.

    The view-spec keeps accessors as names (e.g. ``"interleaved_to_two_columns"``)
    so the model stays GUI-free; here they are mapped back to the actual
    functions from :mod:`chisurf.core.math.datatools`.
    """
    resolved = dict(options)
    dist = resolved.get("distribution_options")
    if isinstance(dist, dict):
        new_dist = {}
        for name, cfg in dist.items():
            cfg = dict(cfg)
            accessor = cfg.get("accessor")
            if isinstance(accessor, str):
                cfg["accessor"] = getattr(chisurf.core.math.datatools, accessor, None)
            new_dist[name] = cfg
        resolved["distribution_options"] = new_dist
    return resolved


# --- curve inputs ----------------------------------------------------------
class CurveInputWidget(QtWidgets.QWidget):
    """Generic data-curve picker for a :class:`CurveInputSection`.

    Renders a label, a read-only name field, a "Select…" button (opening an
    :class:`ExperimentalDataSelector`) and an optional "Unload" button. Selecting
    a curve dispatches ``section.select_action`` with the chosen curve's index
    and name (under ``section.index_key`` / ``section.name_key``) plus
    ``fit_index``; unloading dispatches ``section.unload_action``. This is the
    one widget behind every curve input (IRF, background, linearization table),
    so those inputs stay authorable in ``.view.json``.
    """

    def __init__(self, model, section, parent=None):
        super().__init__(parent)
        self._model = model
        self._section = section
        self._selector = None

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        lbl = QtWidgets.QLabel(section.label)
        layout.addWidget(lbl)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setReadOnly(True)
        self.name_edit.setPlaceholderText(f"load {section.label} →")
        layout.addWidget(self.name_edit, 1)

        # compact icon-style buttons matching the hand-written widgets
        self.select_btn = QtWidgets.QToolButton()
        self.select_btn.setText("…")  # ellipsis
        self.select_btn.setToolTip(f"Select {section.label}")
        self.select_btn.clicked.connect(self._open_selector)
        layout.addWidget(self.select_btn)

        self.unload_btn = QtWidgets.QToolButton()
        self.unload_btn.setText("✕")  # ✕
        self.unload_btn.setToolTip(f"Unload {section.label}")
        self.unload_btn.clicked.connect(self._unload)
        self.unload_btn.setVisible(bool(section.unload_action))
        layout.addWidget(self.unload_btn)

        # inline FWHM readout for the IRF, like the legacy convolve widget
        self.fwhm_label = None
        if section.name_attr == "irf":
            layout.addWidget(QtWidgets.QLabel("FWHM"))
            self.fwhm_label = QtWidgets.QLineEdit()
            self.fwhm_label.setReadOnly(True)
            self.fwhm_label.setMaximumWidth(64)
            layout.addWidget(self.fwhm_label)

        self._refresh_name()
        self._refresh_fwhm()

    def _own_fit_index(self) -> int:
        try:
            fit = getattr(self._model, "fit", None)
            for i, fg in enumerate(cs.fits):
                if fg is fit or fit in list(fg):
                    return i
        except Exception:
            pass
        return 0

    def _open_selector(self):
        from chisurf.gui.widgets.experiments import ExperimentalDataSelector

        fit = getattr(self._model, "fit", None)
        try:
            experiment = fit.data.experiment.__class__
        except Exception:
            experiment = None
        self._selector = ExperimentalDataSelector(
            parent=None, change_event=self._on_change, fit=fit, experiment=experiment
        )
        self._selector.show()

    def _on_change(self):
        sel = self._selector
        if sel is None:
            return
        section = self._section
        try:
            idx = int(sel.selected_curve_index)
            name = str(sel.curve_name)
        except Exception as exc:
            logging.warning(f"CurveInputWidget: could not read selection: {exc}")
            return
        fit_index = self._own_fit_index()
        payload = {section.index_key: idx, section.name_key: name, "fit_index": int(fit_index)}
        try:
            if section.select_action:
                cs.core.actions.dispatch(name=section.select_action, payload=payload)
            cs.core.actions.dispatch(name="fit.update", payload={"fit_index": int(fit_index)})
        except Exception as exc:
            logging.warning(f"CurveInputWidget: select dispatch failed: {exc}")
        self.name_edit.setText(name)
        self._refresh_fwhm()

    def _unload(self):
        section = self._section
        if not section.unload_action:
            return
        fit_index = self._own_fit_index()
        try:
            cs.core.actions.dispatch(
                name=section.unload_action, payload={"fit_index": int(fit_index)}
            )
            cs.core.actions.dispatch(name="fit.update", payload={"fit_index": int(fit_index)})
        except Exception as exc:
            logging.warning(f"CurveInputWidget: unload dispatch failed: {exc}")
        self.name_edit.clear()
        self._refresh_fwhm()

    def _refresh_name(self):
        section = self._section
        if not (section.name_attr and section.target):
            return
        group = getattr(self._model, section.target, None)
        curve = getattr(group, section.name_attr, None) if group is not None else None
        name = getattr(curve, "name", None) or getattr(curve, "filename", None)
        if name:
            self.name_edit.setText(str(name))

    def _refresh_fwhm(self):
        if self.fwhm_label is None:
            return
        group = getattr(self._model, self._section.target, None)
        curve = getattr(group, self._section.name_attr, None) if group is not None else None
        fwhm = getattr(curve, "fwhm", None)
        try:
            self.fwhm_label.setText(f"{float(fwhm):.3f}" if fwhm is not None else "")
        except Exception:
            self.fwhm_label.setText("")


# --- choice / toggle inputs ------------------------------------------------
def _resolve_options_source(name: str):
    """Resolve a named option list (e.g. ``"window_function_types"``)."""
    sources = {
        "window_function_types": lambda: list(chisurf.core.math.signal.window_function_types),
    }
    factory = sources.get(name)
    if factory is None:
        logging.warning(f"ChoiceWidget: unknown options_source {name!r}")
        return []
    try:
        return factory()
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning(f"ChoiceWidget: options_source {name!r} failed: {exc}")
        return []


class _BoundControlMixin:
    """Shared get/set/dispatch for attribute- or action-bound controls."""

    def _apply_tooltip(self, *widgets):
        """Set the section's ``description`` as the tooltip on the given widgets.

        Qt does not propagate a parent widget's tooltip to its children, so the
        interactive editor needs its own copy for the help to show on hover.
        """
        desc = getattr(self._section, "description", "")
        if not desc:
            return
        for w in widgets:
            if w is not None:
                w.setToolTip(desc)

    def _group(self):
        target = getattr(self._section, "target", None)
        return getattr(self._model, target, None) if target else None

    def _own_fit_index(self) -> int:
        try:
            fit = getattr(self._model, "fit", None)
            for i, fg in enumerate(cs.fits):
                if fg is fit or fit in list(fg):
                    return i
        except Exception:
            pass
        return 0

    def _current_value(self):
        section = self._section
        if section.attr:
            group = self._group()
            obj = group if group is not None else self._model
            if obj is not None:
                try:
                    return getattr(obj, section.attr)
                except Exception:
                    return None
        return None

    def _commit(self, value):
        """Apply a new value via action dispatch or direct attribute set."""
        section = self._section
        fit_index = self._own_fit_index()
        try:
            if section.set_action:
                payload = dict(section.action_fixed)
                payload[section.value_key] = value
                payload["fit_index"] = int(fit_index)
                cs.core.actions.dispatch(name=section.set_action, payload=payload)
            elif section.attr:
                group = self._group()
                obj = group if group is not None else self._model
                if obj is not None:
                    setattr(obj, section.attr, value)
            # Only nudge the fit machinery when the bound object actually belongs
            # to a fit. Generic AutoForm consumers (settings/tool dialogs) have no
            # ``fit`` and must not trigger a recompute.
            if getattr(self._model, "fit", None) is not None:
                cs.core.actions.dispatch(name="fit.update", payload={"fit_index": int(fit_index)})
        except Exception as exc:
            logging.warning(f"bound control commit failed ({section.label}): {exc}")


class ChoiceWidget(_BoundControlMixin, QtWidgets.QWidget):
    """One-of-N selector for a :class:`ChoiceSection`.

    Renders inline radio buttons when ``section.style == "radio"`` (compact, like
    the hand-written convolution-type control), otherwise a combo box.
    """

    is_form_field = True

    def __init__(self, model, section, parent=None):
        super().__init__(parent)
        self._model = model
        self._section = section
        self.form_label = section.label

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        options = list(section.options)
        if not options and section.options_source:
            options = _resolve_options_source(section.options_source)
        self._options = options
        _labels = list(getattr(section, "labels", ()))
        _has_labels = bool(_labels) and len(_labels) == len(options)
        current = self._current_value()

        self.combo = None
        self._radios = []
        if section.style == "radio":
            self._button_group = QtWidgets.QButtonGroup(self)
            for i, opt in enumerate(options):
                display = _labels[i] if _has_labels else str(opt)
                rb = QtWidgets.QRadioButton(display)
                if current is not None and str(opt) == str(current):
                    rb.setChecked(True)
                rb.toggled.connect(lambda checked, v=opt: self._commit(v) if checked else None)
                self._button_group.addButton(rb)
                self._radios.append(rb)
                layout.addWidget(rb)
            layout.addStretch(1)
            self._apply_tooltip(self, *self._radios)
        else:
            self.combo = QtWidgets.QComboBox()
            for i, opt in enumerate(options):
                display = _labels[i] if _has_labels else str(opt)
                self.combo.addItem(display)
            # Match initial selection by option value, not displayed text
            if current is not None:
                for i, opt in enumerate(options):
                    if str(opt) == str(current):
                        self.combo.setCurrentIndex(i)
                        break

            def _on_index_changed(idx, _opts=options):
                if 0 <= idx < len(_opts):
                    self._commit(_opts[idx])

            self.combo.currentIndexChanged.connect(_on_index_changed)
            layout.addWidget(self.combo, 1)
            self._apply_tooltip(self, self.combo)

    def sync(self) -> None:
        """Re-read the model value into the control without firing signals."""
        cur = self._current_value()
        if cur is None:
            return
        if self.combo is not None:
            for i, opt in enumerate(self._options):
                if str(opt) == str(cur):
                    self.combo.blockSignals(True)
                    self.combo.setCurrentIndex(i)
                    self.combo.blockSignals(False)
                    break
        else:
            for opt, rb in zip(self._options, self._radios):
                if str(opt) == str(cur):
                    rb.blockSignals(True)
                    rb.setChecked(True)
                    rb.blockSignals(False)
                    break


class ToggleWidget(_BoundControlMixin, QtWidgets.QWidget):
    """Boolean checkbox for a :class:`ToggleSection`."""

    is_form_field = True

    def __init__(self, model, section, parent=None):
        super().__init__(parent)
        self._model = model
        self._section = section
        self.form_label = section.label

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        # label moves to the form's label column; checkbox sits in the field column
        self.checkbox = QtWidgets.QCheckBox()
        current = self._current_value()
        if current is not None:
            self.checkbox.setChecked(bool(current))
        self.checkbox.toggled.connect(lambda checked: self._commit(bool(checked)))
        layout.addWidget(self.checkbox)
        layout.addStretch(1)
        self._apply_tooltip(self, self.checkbox)

    def sync(self) -> None:
        """Re-read the model value into the checkbox without firing signals."""
        cur = self._current_value()
        if cur is None:
            return
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(bool(cur))
        self.checkbox.blockSignals(False)


class ToggleRowWidget(QtWidgets.QWidget):
    """Multiple boolean checkboxes on a single horizontal line.

    Used for ``ToggleRowSection`` (e.g. Pile-up / DNL / Reverse in corrections).
    Each item dict has keys ``target``, ``attr``, ``label``.
    """

    is_form_field = False

    def __init__(self, model, section, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for item in section.items:
            target = item.get("target")
            attr = item.get("attr", "")
            label = item.get("label", attr)
            group = getattr(model, target, model) if target else model
            cb = QtWidgets.QCheckBox(label)
            desc = item.get("description", "")
            if desc:
                cb.setToolTip(desc)
            cb.setChecked(bool(getattr(group, attr, False)))

            def _on_toggle(checked, g=group, a=attr, m=model):
                setattr(g, a, bool(checked))
                try:
                    m.update()
                except Exception:
                    pass

            cb.toggled.connect(_on_toggle)
            layout.addWidget(cb)
        layout.addStretch(1)


class _FocusOutPlainTextEdit(QtWidgets.QPlainTextEdit):
    """Multi-line editor that emits ``editingFinished`` on focus-out.

    Mirrors :class:`QtWidgets.QLineEdit`'s commit-on-focus-out semantics so a
    ``ValueSection`` of ``kind="text"`` commits once the user leaves the field
    rather than on every keystroke.
    """

    editingFinished = QtCore.Signal()

    def focusOutEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().focusOutEvent(event)
        self.editingFinished.emit()


class ValueWidget(_BoundControlMixin, QtWidgets.QWidget):
    """Scalar field for a :class:`ValueSection`.

    Supported ``kind`` values: ``int`` / ``float`` (spin boxes), ``str`` (line
    edit), ``text`` (multi-line plain-text edit) and ``date`` (date edit).
    """

    is_form_field = True

    def __init__(self, model, section, parent=None):
        super().__init__(parent)
        self._model = model
        self._section = section
        self.form_label = section.label

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        current = self._current_value()
        # A read-only field must never write back — its value (which may be a
        # non-editable object such as a callable) is displayed but left untouched.
        read_only = bool(getattr(section, "read_only", False))
        if section.kind == "int":
            self.editor = QtWidgets.QSpinBox()
            self.editor.setMinimum(
                int(section.minimum) if section.minimum is not None else -2_147_483_648
            )
            self.editor.setMaximum(
                int(section.maximum) if section.maximum is not None else 2_147_483_647
            )
            if section.step:
                self.editor.setSingleStep(int(section.step))
            if section.suffix:
                self.editor.setSuffix(section.suffix)
            if current is not None:
                self.editor.setValue(int(current))
            if not read_only:
                self.editor.valueChanged.connect(lambda v: self._commit(int(v)))
        elif section.kind == "float":
            self.editor = QtWidgets.QDoubleSpinBox()
            self.editor.setDecimals(int(section.decimals))
            self.editor.setMinimum(
                float(section.minimum) if section.minimum is not None else -1e308
            )
            self.editor.setMaximum(float(section.maximum) if section.maximum is not None else 1e308)
            if section.step:
                self.editor.setSingleStep(float(section.step))
            if section.suffix:
                self.editor.setSuffix(section.suffix)
            if current is not None:
                self.editor.setValue(float(current))
            if not read_only:
                self.editor.valueChanged.connect(lambda v: self._commit(float(v)))
        elif section.kind == "text":
            self.editor = _FocusOutPlainTextEdit()
            self.editor.setMinimumHeight(54)
            if section.placeholder:
                self.editor.setPlaceholderText(section.placeholder)
            if current is not None:
                self.editor.setPlainText(str(current))
            if not read_only:
                self.editor.editingFinished.connect(
                    lambda: self._commit(self.editor.toPlainText())
                )
        elif section.kind == "date":
            self.editor = QtWidgets.QDateEdit()
            self.editor.setCalendarPopup(True)
            self.editor.setDisplayFormat("yyyy-MM-dd")
            self._set_date_from(current)
            if not read_only:
                self.editor.dateChanged.connect(
                    lambda d: self._commit(d.toString("yyyy-MM-dd"))
                )
        else:  # "str"
            self.editor = QtWidgets.QLineEdit()
            if section.placeholder:
                self.editor.setPlaceholderText(section.placeholder)
            if current is not None:
                self.editor.setText(str(current))
            if not read_only:
                self.editor.editingFinished.connect(lambda: self._commit(self.editor.text()))
        if read_only:
            self.editor.setReadOnly(True)
            if isinstance(self.editor, QtWidgets.QAbstractSpinBox):
                self.editor.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        layout.addWidget(self.editor, 1)
        self._apply_tooltip(self, self.editor)

    def _set_date_from(self, value) -> None:
        """Set the QDateEdit from an ISO ``yyyy-MM-dd`` string (or leave default)."""
        if not value:
            return
        date = QtCore.QDate.fromString(str(value)[:10], "yyyy-MM-dd")
        if date.isValid():
            self.editor.setDate(date)

    def sync(self) -> None:
        """Re-read the model value into the editor without firing signals."""
        cur = self._current_value()
        if cur is None:
            return
        self.editor.blockSignals(True)
        if isinstance(self.editor, QtWidgets.QSpinBox):
            self.editor.setValue(int(cur))
        elif isinstance(self.editor, QtWidgets.QDoubleSpinBox):
            self.editor.setValue(float(cur))
        elif isinstance(self.editor, QtWidgets.QPlainTextEdit):
            self.editor.setPlainText(str(cur))
        elif isinstance(self.editor, QtWidgets.QDateEdit):
            self._set_date_from(cur)
        elif isinstance(self.editor, QtWidgets.QLineEdit):
            self.editor.setText(str(cur))
        self.editor.blockSignals(False)


# --- custom sections -------------------------------------------------------
@register_section("lifetime_amplitude_options")
class LifetimeAmplitudeOptions(QtWidgets.QWidget):
    """Header controls for a lifetime group: normalize / absolute amplitudes.

    This is the bespoke escape-hatch widget for the dynamic lifetime section.
    It edits the model's amplitude options through the action dispatcher, so the
    model remains the single source of truth and no widget reaches into compute.
    """

    def __init__(self, model=None, target: str = "lifetimes", parent=None, **options):
        super().__init__(parent)
        self._model = model
        self._group = getattr(model, target, None)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.absolute = QtWidgets.QCheckBox("Abs.")
        self.absolute.setToolTip("Take absolute value of amplitudes (no negative amplitudes).")
        self.absolute.setChecked(bool(getattr(self._group, "absolute_amplitudes", True)))
        self.absolute.clicked.connect(self._on_changed)

        self.normalize = QtWidgets.QCheckBox("Norm.")
        self.normalize.setToolTip("Normalize amplitudes so they sum to one.")
        self.normalize.setChecked(bool(getattr(self._group, "normalize_amplitudes", True)))
        self.normalize.clicked.connect(self._on_changed)

        # read/link menus (port of the legacy LifetimeWidget header controls).
        self.read_btn = QtWidgets.QToolButton()
        self.read_btn.setText("read")
        self.read_btn.setToolTip("Copy parameter values from another lifetime group.")
        self.read_menu = QtWidgets.QMenu(self.read_btn)
        self.read_menu.aboutToShow.connect(
            lambda: self._build_target_menu(self.read_menu, self._read_values)
        )
        self.read_btn.setMenu(self.read_menu)
        self.read_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self.link_btn = QtWidgets.QToolButton()
        self.link_btn.setText("link")
        self.link_btn.setToolTip("Link this lifetime group to another (shared spectrum).")
        self.link_menu = QtWidgets.QMenu(self.link_btn)
        self.link_menu.aboutToShow.connect(
            lambda: self._build_target_menu(self.link_menu, self._link_to)
        )
        self.link_btn.setMenu(self.link_menu)
        self.link_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        layout.addWidget(self.absolute)
        layout.addWidget(self.normalize)
        layout.addWidget(self.read_btn)
        layout.addWidget(self.link_btn)

    # -- read / link ---------------------------------------------------------
    def _lifetime_groups(self):
        """Yield ``(fit_index, fit, group)`` for every lifetime group in all fits.

        Operates on core :class:`Lifetime` groups (not widgets), so it works for
        both legacy and auto-rendered models.
        """
        from chisurf.core.models.tcspc.lifetime import Lifetime

        try:
            from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client

            fit_groups = get_fitting_client().get_fit_objects()
        except Exception:
            fit_groups = []
        idx = 0
        for fg in fit_groups:
            for fit in fg:
                for a in getattr(fit.model, "aggregated_parameters", []):
                    if isinstance(a, Lifetime):
                        yield idx, fit, a
            idx += 1

    def _build_target_menu(self, menu, on_pick):
        """Populate ``menu`` with selectable target lifetime groups."""
        menu.clear()
        for _idx, fit, group in self._lifetime_groups():
            if group is self._group:
                continue
            action = menu.addAction(f"{fit.name}: {group.name}")
            action.triggered.connect(lambda _checked=False, g=group: on_pick(g))

    def _own_fit_index(self):
        """Best-effort fit index of this section's model for dispatching."""
        try:
            fit = getattr(self._model, "fit", None)
            for i, fg in enumerate(cs.fits):
                if fg is fit or fit in list(fg):
                    return i
        except Exception:
            pass
        return 0

    def _read_values(self, target):
        """Copy parameter values from ``target`` into this group via dispatch."""
        group = self._group
        if group is None:
            return
        fit_index = self._own_fit_index()
        try:
            target_params = target.parameters_all_dict
            for key in group.parameter_dict:
                if key in target_params:
                    cs.core.actions.dispatch(
                        name="parameter.value",
                        payload={
                            "parameter_name": str(key),
                            "value": float(target_params[key].value),
                            "fit_index": int(fit_index),
                        },
                    )
            cs.core.actions.dispatch(name="fit.update", payload={"fit_index": int(fit_index)})
        except Exception as exc:
            logging.warning(f"Failed to read lifetime values: {exc}")

    def _link_to(self, target):
        """Link this group's spectrum to ``target`` and refresh the fit."""
        group = self._group
        if group is None:
            return
        try:
            group.link = target
            cs.core.actions.dispatch(
                name="fit.update", payload={"fit_index": int(self._own_fit_index())}
            )
        except Exception as exc:
            logging.warning(f"Failed to link lifetime group: {exc}")

    def _on_changed(self, *_):
        """Push amplitude-option changes to the model via the dispatcher."""
        group = self._group
        if group is None:
            return
        name = str(getattr(group, "name", "lifetimes"))
        try:
            cs.core.actions.dispatch(
                name="model.normalize_amplitudes",
                payload={"component_name": name, "normalize": bool(self.normalize.isChecked())},
            )
            cs.core.actions.dispatch(
                name="model.absolute_amplitudes",
                payload={"component_name": name, "absolute": bool(self.absolute.isChecked())},
            )
        except Exception as exc:  # pragma: no cover - dispatcher optional in tests
            logging.warning(f"Failed to dispatch amplitude options: {exc}")
            # Fallback: set directly on the model group.
            group.normalize_amplitudes = bool(self.normalize.isChecked())
            group.absolute_amplitudes = bool(self.absolute.isChecked())


class PlotWidget(QtWidgets.QWidget):
    """Inline plot section rendered from a declarative :class:`PlotSection`.

    Reads the data by calling ``getattr(model, section.source)()``, which must
    return a list of series mappings (``{"x", "y", "name", "color", "width",
    "style"}``). Call :meth:`refresh` (e.g. via ``AutoForm.refresh_plots``) to
    re-read the source after the model changes.
    """

    is_form_field = False

    _STYLES = {
        "solid": QtCore.Qt.SolidLine,
        "dash": QtCore.Qt.DashLine,
        "dot": QtCore.Qt.DotLine,
    }

    def __init__(self, model, section, parent=None):
        super().__init__(parent)
        import pyqtgraph as pg

        self._model = model
        self._section = section

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        if section.height:
            self.plot.setMaximumHeight(int(section.height))
        if section.x_label:
            self.plot.setLabel("bottom", section.x_label)
        if section.y_label:
            self.plot.setLabel("left", section.y_label)
        if getattr(section, "log_x", False) or section.log_y:
            try:
                self.plot.setLogMode(bool(getattr(section, "log_x", False)), bool(section.log_y))
            except Exception:
                pass
        if section.legend:
            try:
                self.plot.addLegend(offset=(-5, 5))
            except Exception:
                pass
        try:
            self.plot.getPlotItem().getViewBox().setMenuEnabled(False)
        except Exception:
            pass
        layout.addWidget(self.plot)
        if getattr(section, "description", ""):
            self.setToolTip(section.description)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the section's source method and redraw all series."""
        import pyqtgraph as pg

        source = getattr(self._model, self._section.source, None)
        if not callable(source):
            return
        try:
            series = source() or []
        except Exception as exc:  # pragma: no cover - source is model-defined
            logging.warning(f"PlotWidget: source {self._section.source!r} failed: {exc}")
            return
        self.plot.clear()
        for s in series:
            pen = pg.mkPen(
                s.get("color", "y"),
                width=int(s.get("width", 1)),
                style=self._STYLES.get(s.get("style", "solid"), QtCore.Qt.SolidLine),
            )
            kw = {"pen": pen, "name": s.get("name", "")}
            if s.get("symbol"):
                kw["symbol"] = s["symbol"]
                kw["symbolBrush"] = s.get("color", "y")
                kw["symbolSize"] = int(s.get("symbol_size", 9))
                if s.get("no_line"):
                    kw["pen"] = None
            self.plot.plot(s.get("x", []), s.get("y", []), **kw)


class LCurveWidget(QtWidgets.QWidget):
    """Reusable L-curve view (residual vs solution norm, log-log, corner marked).

    The general, declarative L-curve component: any model holding a
    :class:`chisurf.core.math.regularization.LCurveData` (as an attribute or a
    zero-arg method named by ``target``) can show it with a ``custom`` section::

        {"type": "custom", "key": "lcurve", "target": "lcurve_data", "title": "L-curve"}

    The corner (auto-selected regularization weight) is highlighted.
    """

    #: marker so :meth:`AutoForm.refresh_plots` re-reads this widget.
    AUTOFORM_REFRESH = True

    def __init__(self, model, target: str, **options):
        super().__init__()
        self._model = model
        self._target = target
        self._opts = options
        self._plot = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        try:
            import pyqtgraph as pg

            self._pg = pg
            self._plot = pg.PlotWidget()
            self._plot.setLogMode(x=True, y=True)
            self._plot.setLabel("bottom", options.get("x_label", "residual norm"))
            self._plot.setLabel("left", options.get("y_label", "solution norm"))
            self._plot.addLegend()
            lay.addWidget(self._plot)
        except Exception:  # pragma: no cover - pyqtgraph optional
            lay.addWidget(QtWidgets.QLabel("pyqtgraph not available"))
        self.refresh()

    def _data(self):
        obj = getattr(self._model, self._target, None)
        return obj() if callable(obj) else obj

    def refresh(self) -> None:
        """Re-read the model's :class:`LCurveData` and redraw."""
        if self._plot is None:
            return
        data = self._data()
        self._plot.clear()
        if data is None or getattr(data, "reg", None) is None or len(data.reg) == 0:
            return
        pg = self._pg
        self._plot.plot(
            data.residual_norm,
            data.solution_norm,
            pen=pg.mkPen("c", width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush="c",
            name="L-curve",
        )
        corner = getattr(data, "corner_point", None)
        if corner is not None:
            self._plot.plot(
                [corner[0]],
                [corner[1]],
                pen=None,
                symbol="o",
                symbolSize=12,
                symbolBrush="r",
                name="chosen",
            )


@register_section("lcurve")
def _lcurve_section_factory(model, target: str, **options):
    """Custom-section factory rendering a model's ``LCurveData`` (see :class:`LCurveWidget`)."""
    return LCurveWidget(model, target, **options)


#: Matplotlib colormaps offered by the general image widget's colour selector.
IMAGE_COLORMAPS = ["viridis", "magma", "inferno", "plasma", "cividis", "turbo", "gray"]


def apply_colormap(image_view, name: str) -> None:
    """Apply a (matplotlib) colormap by name to a pyqtgraph ImageView, best-effort.

    Reusable by any pyqtgraph image plot (AutoForm or not) so the colour handling is
    consistent across tools (2D-FLC, RICS, PDA, ...).
    """
    try:
        import pyqtgraph as pg

        try:
            cmap = pg.colormap.get(name, source="matplotlib")
        except Exception:
            cmap = pg.colormap.get(name)
        image_view.setColorMap(cmap)
    except Exception:  # pragma: no cover - colormap optional
        pass


class ImageMapWidget(QtWidgets.QWidget):
    """General 2D image dock bound to ``model.<target>()`` with an optional colour selector.

    Declare it in a view.json as a ``custom`` section so any tool can show a 2D map::

        {"type": "custom", "key": "image", "target": "spectrum_image", "title": "Map",
         "options": {"colormap": true, "colormap_attr": "colormap"}}

    The colour control lives *in the plot* (a small combo above the image), so it is
    portable and needs no separate settings panel. ``options``:

    * ``colormap`` (bool) — show the embedded colormap selector (default ``False``).
    * ``default_colormap`` (str) — initial colormap (default ``"viridis"``).
    * ``colormap_attr`` (str) — optional model attribute to read/write the chosen colormap,
      so it persists and can be shared between several image docks.
    """

    #: marker so :meth:`AutoForm.refresh_plots` re-reads this widget.
    AUTOFORM_REFRESH = True

    def __init__(
        self,
        model,
        target: str,
        *,
        colormap: bool = False,
        default_colormap: str = "viridis",
        colormap_attr: str | None = None,
        **options,
    ):
        super().__init__()
        self._model = model
        self._target = target
        self._cmap_attr = colormap_attr
        self._cmap = default_colormap
        self._image = None
        self._combo = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        if colormap:
            bar = QtWidgets.QHBoxLayout()
            bar.setContentsMargins(4, 2, 4, 0)
            bar.addStretch(1)
            bar.addWidget(QtWidgets.QLabel("colormap"))
            self._combo = QtWidgets.QComboBox()
            self._combo.addItems(IMAGE_COLORMAPS)
            self._combo.setToolTip("Colormap for this image")
            cur = self._current_cmap()
            idx = self._combo.findText(cur)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
            self._combo.currentTextChanged.connect(self._on_cmap)
            bar.addWidget(self._combo)
            lay.addLayout(bar)
        try:
            import pyqtgraph as pg

            self._image = pg.ImageView()
            self._image.ui.roiBtn.hide()
            self._image.ui.menuBtn.hide()
            lay.addWidget(self._image, 1)
        except Exception:  # pragma: no cover - pyqtgraph optional
            lay.addWidget(QtWidgets.QLabel("pyqtgraph not available"))

    def _current_cmap(self) -> str:
        if self._cmap_attr:
            return str(getattr(self._model, self._cmap_attr, self._cmap))
        return self._cmap

    def _on_cmap(self, name: str) -> None:
        self._cmap = name
        if self._cmap_attr and hasattr(self._model, self._cmap_attr):
            setattr(self._model, self._cmap_attr, name)
        if self._image is not None:
            apply_colormap(self._image, name)

    def refresh(self) -> None:
        """Re-read the model image and redraw with the current colormap."""
        if self._image is None:
            return
        import numpy as np

        obj = getattr(self._model, self._target, None)
        img = obj() if callable(obj) else obj
        if img is None:
            return
        # keep the combo in sync if the colormap is model-backed
        if self._combo is not None:
            cur = self._current_cmap()
            if cur != self._combo.currentText():
                self._combo.blockSignals(True)
                i = self._combo.findText(cur)
                if i >= 0:
                    self._combo.setCurrentIndex(i)
                self._combo.blockSignals(False)
        self._image.setImage(np.asarray(img, dtype=float), autoLevels=True)
        apply_colormap(self._image, self._current_cmap())


@register_section("image")
def _image_section_factory(model, target: str, **options):
    """Custom-section factory for a general 2D image dock (see :class:`ImageMapWidget`)."""
    return ImageMapWidget(model, target, **options)
