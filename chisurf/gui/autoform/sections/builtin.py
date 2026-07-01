"""Built-in registrations mapping view-spec keys to ChiSurf widgets/plots.

Importing this module wires the standard plot keys (``line``, ``residual``,
``distribution`` ...) and standard custom sections into the registry. The model
layer references these by string only; the concrete classes live here.
"""

from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

import chisurf as cs
import chisurf.core.math.datatools
from chisurf import logging

from .registry import register_plot, register_section


def _wrap_tooltip(text: str, width: int = 56) -> str:
    """Word-wrap a tooltip so long descriptions break over several lines.

    Existing explicit line breaks are preserved; each paragraph is wrapped to
    ``width`` characters (Qt renders newlines in plain-text tooltips).
    """
    import textwrap

    text = str(text or "").strip()
    if not text:
        return ""
    lines = []
    for para in text.splitlines():
        para = para.strip()
        lines.append(textwrap.fill(para, width=width) if para else "")
    return "\n".join(lines)


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
register_plot("residual2d", lambda: _plots().Residual2DPlot)
register_plot("lcurve", lambda: _plots().LCurvePlot)


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
                cfg["accessor"] = _resolve_accessor(accessor)
            new_dist[name] = cfg
        resolved["distribution_options"] = new_dist
    return resolved


def _resolve_accessor(accessor: str):
    """Resolve a distribution-plot accessor name to a callable.

    Bare names (e.g. ``"interleaved_to_two_columns"``) resolve against
    :mod:`chisurf.core.math.datatools`. A dotted or ``module:function`` path
    (e.g. ``"chisurf.core.models.pda.common:get_pda_distribution"``) is imported
    directly, so model-specific Qt-free accessors stay authorable in JSON.
    """
    if ":" in accessor or "." in accessor:
        import importlib

        if ":" in accessor:
            mod_name, _, func_name = accessor.partition(":")
        else:
            mod_name, _, func_name = accessor.rpartition(".")
        try:
            return getattr(importlib.import_module(mod_name), func_name, None)
        except Exception:
            return None
    return getattr(chisurf.core.math.datatools, accessor, None)


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
def _resolve_options_source(name: str, model=None):
    """Resolve a named option list.

    Prefers a model-backed source — an attribute or zero-arg method named *name*
    on *model* returning a list (so tool view-models can drive dynamic combos);
    otherwise falls back to the built-in named sources.
    """
    if model is not None and hasattr(model, name):
        try:
            src = getattr(model, name)
            return list(src() if callable(src) else src)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(f"ChoiceWidget: model options_source {name!r} failed: {exc}")
            return []
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
        desc = _wrap_tooltip(getattr(self._section, "description", ""))
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
            # Tool view-models (not in the action registry) can request a direct
            # model-method call with the new value.
            call = getattr(section, "call", "")
            if call:
                fn = getattr(self._model, call, None)
                if callable(fn):
                    fn(value)
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

        self._options = self._resolve_opts()
        current = self._current_value()

        self.combo = None
        self._radios = []
        if section.style == "radio":
            self._button_group = QtWidgets.QButtonGroup(self)
            _labels = self._labels()
            for i, opt in enumerate(self._options):
                rb = QtWidgets.QRadioButton(_labels[i])
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
            if section.editable:
                self.combo.setEditable(True)
                self.combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
            self._populate_combo(current)
            self.combo.currentIndexChanged.connect(self._on_index_changed)
            if section.editable:
                # Commit free-typed text (not necessarily among the options) on
                # focus-out / Enter, so a hand-entered model id is preserved.
                self.combo.lineEdit().editingFinished.connect(
                    lambda: self._commit(self.combo.currentText().strip())
                )
            layout.addWidget(self.combo, 1)
            self._apply_tooltip(self, self.combo)
            # Optional +/- buttons for managed (dynamic) combos driven by model methods.
            if section.add_action:
                add_btn = QtWidgets.QToolButton()
                add_btn.setText(section.add_label)
                add_btn.clicked.connect(self._on_add)
                layout.addWidget(add_btn)
            if section.remove_action:
                del_btn = QtWidgets.QToolButton()
                del_btn.setText(section.remove_label)
                del_btn.clicked.connect(self._on_remove)
                layout.addWidget(del_btn)

    def _resolve_opts(self) -> list:
        section = self._section
        options = list(section.options)
        if not options and section.options_source:
            options = _resolve_options_source(section.options_source, self._model)
        return options

    def _labels(self) -> list:
        labels = list(getattr(self._section, "labels", ()))
        if labels and len(labels) == len(self._options):
            return [str(x) for x in labels]
        return [str(o) for o in self._options]

    def _populate_combo(self, current) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        for label in self._labels():
            self.combo.addItem(label)
        matched = False
        if current is not None:
            for i, opt in enumerate(self._options):
                if str(opt) == str(current):
                    self.combo.setCurrentIndex(i)
                    matched = True
                    break
        # An editable combo may hold a value the option list does not contain
        # (a hand-typed model id); show it verbatim in the line edit.
        if not matched and current is not None and self.combo.isEditable():
            self.combo.setEditText(str(current))
        self.combo.blockSignals(False)

    def _on_index_changed(self, idx) -> None:
        if 0 <= idx < len(self._options):
            self._commit(self._options[idx])

    def _on_add(self) -> None:
        fn = getattr(self._model, self._section.add_action, None)
        if callable(fn):
            fn()
        self._rebuild_options()

    def _on_remove(self) -> None:
        fn = getattr(self._model, self._section.remove_action, None)
        if callable(fn):
            fn(self._current_value())
        self._rebuild_options()

    def _rebuild_options(self) -> None:
        """Re-read the model-backed option list and restore the current value."""
        self._options = self._resolve_opts()
        if self.combo is not None:
            self._populate_combo(self._current_value())

    def sync(self) -> None:
        """Re-read the model value (and dynamic options) without firing signals."""
        if self._section.options_source:
            self._rebuild_options()
        cur = self._current_value()
        if cur is None:
            return
        if self.combo is not None:
            if self.combo.isEditable():
                self._populate_combo(cur)
            else:
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
            desc = _wrap_tooltip(item.get("description", ""))
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


class ButtonRowWidget(QtWidgets.QWidget):
    """A horizontal row of action buttons for a :class:`ButtonRowSection`.

    Each button dict has keys ``label``, ``action`` (a zero-arg model method) and
    optional ``description``. Lets tool toolbars be authored declaratively.
    """

    is_form_field = False

    def __init__(self, model, section, parent=None):
        super().__init__(parent)
        self._model = model
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for item in section.buttons:
            btn = QtWidgets.QToolButton()
            btn.setText(item.get("label", ""))
            desc = _wrap_tooltip(item.get("description", ""))
            if desc:
                btn.setToolTip(desc)
            action = item.get("action", "")
            btn.clicked.connect(lambda checked=False, a=action: self._call(a))
            layout.addWidget(btn)
        layout.addStretch(1)

    def _call(self, action: str) -> None:
        # Flush an in-progress field edit before running the action. Fields commit
        # on focus-out (``editingFinished``), but a NoFocus tool button does not
        # blur the editor on click, so a value typed and not yet committed (e.g. an
        # API key) would otherwise be missed. Clearing focus fires that commit
        # synchronously before the action reads the model.
        focused = QtWidgets.QApplication.focusWidget()
        if focused is not None and focused is not self and self.isAncestorOf(focused) is False:
            focused.clearFocus()
        fn = getattr(self._model, action, None)
        if callable(fn):
            fn()


class InfoWidget(QtWidgets.QTextBrowser):
    """Read-only rich-text (HTML/Markdown) block for an :class:`InfoSection`.

    Shows the section's static ``text`` or, when ``source`` is set, the string
    returned by that zero-arg model method — re-read on :meth:`refresh` so live
    status panels update with the model. Opts into ``AUTOFORM_REFRESH`` so
    ``AutoForm.refresh_plots()`` keeps it current.
    """

    is_form_field = False
    AUTOFORM_REFRESH = True

    def __init__(self, model, section, parent=None):
        super().__init__(parent)
        self._model = model
        self._section = section
        self.setOpenExternalLinks(False)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        if getattr(section, "height", 0):
            self.setMinimumHeight(int(section.height))
        self.refresh()

    def _content(self) -> str:
        source = getattr(self._section, "source", "")
        if source:
            fn = getattr(self._model, source, None)
            if callable(fn):
                try:
                    return str(fn() or "")
                except Exception:  # pragma: no cover - defensive
                    logging.warning(f"InfoWidget: source {source!r} failed", exc_info=True)
                    return ""
        return str(getattr(self._section, "text", "") or "")

    def refresh(self) -> None:
        """Re-read the content (static or from ``source``) and re-render it."""
        content = self._content()
        if getattr(self._section, "is_markdown", False):
            self.setMarkdown(content)
        else:
            self.setHtml(content)


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
                self.editor.editingFinished.connect(lambda: self._commit(self.editor.toPlainText()))
        elif section.kind == "date":
            self.editor = QtWidgets.QDateEdit()
            self.editor.setCalendarPopup(True)
            self.editor.setDisplayFormat("yyyy-MM-dd")
            self._set_date_from(current)
            if not read_only:
                self.editor.dateChanged.connect(lambda d: self._commit(d.toString("yyyy-MM-dd")))
        elif section.kind in ("password", "secret"):
            self.editor = QtWidgets.QLineEdit()
            self.editor.setEchoMode(QtWidgets.QLineEdit.Password)
            if section.placeholder:
                self.editor.setPlaceholderText(section.placeholder)
            if current is not None:
                self.editor.setText(str(current))
            if not read_only:
                self.editor.editingFinished.connect(lambda: self._commit(self.editor.text()))
        elif section.kind == "file":
            self.editor = QtWidgets.QLineEdit()
            if section.placeholder:
                self.editor.setPlaceholderText(section.placeholder)
            if current is not None:
                self.editor.setText(str(current))
            if not read_only:
                self.editor.editingFinished.connect(lambda: self._commit_file(self.editor.text()))
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
        # A ``text`` field flagged ``expand`` fills spare vertical space (e.g. a
        # JSON/log preview) instead of staying at its compact minimum height; the
        # form layout reads ``_autoform_expanding`` to hand it the stretch.
        if section.kind == "text" and getattr(section, "expand", False):
            self._autoform_expanding = True
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.editor.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
        layout.addWidget(self.editor, 1)
        if section.kind == "file" and not read_only:
            browse = QtWidgets.QToolButton()
            browse.setText("…")
            browse.setToolTip("Browse…")
            browse.clicked.connect(self._browse_file)
            layout.addWidget(browse)
        # A "secret" field is a masked input with a reveal toggle (e.g. API keys),
        # while a plain "password" field stays masked with no reveal affordance.
        if section.kind == "secret":
            self.reveal = QtWidgets.QToolButton()
            self.reveal.setCheckable(True)
            self.reveal.setText("👁")
            self.reveal.setToolTip("Show / hide")
            self.reveal.toggled.connect(self._toggle_secret)
            layout.addWidget(self.reveal)
        self._apply_tooltip(self, self.editor)

    def _toggle_secret(self, checked: bool) -> None:
        """Reveal or mask a ``kind="secret"`` field's contents."""
        mode = QtWidgets.QLineEdit.Normal if checked else QtWidgets.QLineEdit.Password
        self.editor.setEchoMode(mode)

    def _commit_file(self, path: str) -> None:
        """Commit a file path only when it actually changed (avoids reloads)."""
        if path and path != str(self._current_value() or ""):
            self._commit(path)

    def _browse_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self._section.label or "Open file")
        if path:
            self.editor.setText(path)
            self._commit_file(path)

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
            self.plot.setMinimumHeight(int(section.height))
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
    """General image dock bound to ``model.<target>()``.

    Optional colour selection, brush/draw, 3D stack browsing and click-to-pick
    selection.

    Declare it in a view.json as a ``custom`` section so any tool can show a map::

        {"type": "custom", "key": "image", "target": "spectrum_image", "title": "Map",
         "options": {"colormap": true, "colormap_attr": "colormap"}}

    The image source may be 2D ``(y, x)`` or 3D ``(z, y, x)``; a 3D array is shown
    with pyqtgraph's built-in z-slider (axis 0 = slice) and the current slice is
    preserved across refreshes.

    The colour control lives *in the plot* (a small combo above the image), so it is
    portable and needs no separate settings panel. Colour ``options``:

    * ``colormap`` (bool) — show the embedded colormap selector (default ``False``).
    * ``default_colormap`` (str) — initial colormap (default ``"viridis"``).
    * ``colormap_attr`` (str) — optional model attribute to read/write the chosen colormap,
      so it persists and can be shared between several image docks.

    Brush / draw ``options`` turn the dock into a paintable pixel selector (e.g. for
    CLSM pixel selection, FLIM masks, ROI painting). Brush mode is enabled when
    ``selection_attr`` is given:

    * ``selection_attr`` (str) — model attribute holding the 2D selection mask; the
      widget reads it on refresh and writes it back while painting.
    * ``brush_kernel_source`` (str) — model method returning the draw kernel (so the
      tool owns brush size/shape/erase); falls back to a 1×1 kernel.
    * ``on_draw`` (str) — model method called after a stroke (e.g. to recompute a decay).
    * ``live_attr`` (str) — model attribute (bool) gating ``on_draw`` during a drag.

    Point-pick / overlay ``options`` (independent of brush mode) let a tool select a
    single point in the image (e.g. a bead in a PSF stack) and draw overlays:

    * ``select_attr`` (str) — model attribute that receives the picked ``(z, y, x)``
      tuple on a left-click (``z`` is the current slice; ``0`` for a 2D image). A red
      marker is drawn at the pick.
    * ``on_pick`` (str) — model method called after a pick (e.g. to fit the bead).
    * ``markers_source`` (str) — model method returning a list of ``(z, y, x)``
      points; those on the current slice are drawn as green square markers.
    * ``roi_source`` (str) — model method returning ``{"x", "y", "r", "z"}`` (or
      ``None``); draws a non-interactive yellow circle of radius ``r`` at ``(x, y)``
      when the current slice matches ``z``.
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
        selection_attr: str | None = None,
        brush_kernel_source: str | None = None,
        on_draw: str | None = None,
        live_attr: str | None = None,
        select_attr: str | None = None,
        on_pick: str | None = None,
        markers_source: str | None = None,
        roi_source: str | None = None,
        **options,
    ):
        super().__init__()
        self._model = model
        self._target = target
        self._cmap_attr = colormap_attr
        self._cmap = default_colormap
        self._image = None
        self._combo = None
        # brush state
        self._selection_attr = selection_attr
        self._brush_kernel_source = brush_kernel_source
        self._on_draw = on_draw
        self._live_attr = live_attr
        self._overlay = None
        # point-pick / overlay state
        self._select_attr = select_attr
        self._on_pick = on_pick
        self._markers_source = markers_source
        self._roi_source = roi_source
        self._pick_marker = None
        self._marker_items = []
        self._roi_item = None
        self._ndim = 2
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
            if self._selection_attr:
                self._setup_brush(pg)
            if self._select_attr or self._on_pick:
                self._image.getView().scene().sigMouseClicked.connect(self._on_clicked)
            if self._markers_source or self._roi_source:
                self._connect_slice_changed()
        except Exception:  # pragma: no cover - pyqtgraph optional
            lay.addWidget(QtWidgets.QLabel("pyqtgraph not available"))

    # ── colormap ───────────────────────────────────────────────────────
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

    # ── brush / draw ───────────────────────────────────────────────────
    def _setup_brush(self, pg) -> None:
        """Add a paintable selection overlay on top of the image."""
        self._overlay = pg.ImageItem()
        self._overlay.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        self._image.getView().addItem(self._overlay)
        self._overlay.hoverEvent = self._hover_event
        self._overlay.mouseDragEvent = self._draw_event
        self._apply_kernel()

    def _apply_kernel(self) -> None:
        if self._overlay is None:
            return
        import numpy as np

        kernel = None
        if self._brush_kernel_source:
            fn = getattr(self._model, self._brush_kernel_source, None)
            if callable(fn):
                try:
                    kernel = np.asarray(fn())
                except Exception:  # pragma: no cover - defensive
                    kernel = None
        if kernel is None:
            kernel = np.ones((1, 1))
        cx, cy = kernel.shape[0] // 2, kernel.shape[1] // 2
        self._overlay.setDrawKernel(kernel, mask=kernel, center=(cx, cy), mode="add")

    def _live(self) -> bool:
        if self._live_attr:
            return bool(getattr(self._model, self._live_attr, True))
        return True

    def _hover_event(self, event) -> None:
        if self._image is None:
            return
        base = self._image.getImageItem().image
        if base is None or event.isExit():
            self._image.getView().setToolTip("")
            return
        pos = event.pos()
        i = int(max(0, min(pos.y(), base.shape[0] - 1)))
        j = int(max(0, min(pos.x(), base.shape[1] - 1)))
        self._image.getView().setToolTip(f"pixel ({i}, {j}) = {base[i, j]:g}")

    def _draw_event(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        event.accept()
        if event.isStart():
            self._apply_kernel()
        self._overlay.drawAt(event.pos(), event)
        if self._selection_attr:
            import numpy as np

            setattr(self._model, self._selection_attr, np.asarray(self._overlay.image))
        if self._on_draw and self._live():
            fn = getattr(self._model, self._on_draw, None)
            if callable(fn):
                fn()

    # ── point pick / overlays ──────────────────────────────────────────
    def _current_z(self) -> int:
        """Return the currently displayed slice index (0 for a 2D image)."""
        if self._ndim < 3 or self._image is None:
            return 0
        try:
            return int(self._image.currentIndex)
        except Exception:
            return 0

    def _connect_slice_changed(self) -> None:
        """Redraw per-slice markers/ROI when the z-slider moves."""
        try:
            self._image.timeLine.sigPositionChanged.connect(self._redraw_overlays)
        except Exception:
            try:
                self._image.sigTimeChanged.connect(self._redraw_overlays)
            except Exception:
                pass

    def _on_clicked(self, event) -> None:
        """Left-click in the image → write ``(z, y, x)`` and call ``on_pick``."""
        if self._image is None:
            return
        item = self._image.getImageItem()
        scene_pos = event.scenePos()
        if not item.sceneBoundingRect().contains(scene_pos):
            return
        point = item.mapFromScene(scene_pos)
        x, y = int(point.x()), int(point.y())
        z = self._current_z()
        base = item.image
        if base is not None:
            ny, nx = base.shape[:2]
            if not (0 <= x < nx and 0 <= y < ny):
                return
        if self._select_attr:
            try:
                setattr(self._model, self._select_attr, (z, y, x))
            except Exception:  # pragma: no cover - defensive
                logging.warning(f"ImageMapWidget: could not set {self._select_attr!r}")
        if self._on_pick:
            fn = getattr(self._model, self._on_pick, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # pragma: no cover - model-defined
                    logging.warning(f"ImageMapWidget: on_pick {self._on_pick!r} failed")
        self._redraw_overlays()

    def _redraw_overlays(self, *args) -> None:
        """Redraw pick marker, per-slice detected-point markers and the ROI circle."""
        if self._image is None:
            return
        import pyqtgraph as pg

        view = self._image.getView()
        z = self._current_z()

        # detected-point markers (green squares) on the current slice
        for m in self._marker_items:
            view.removeItem(m)
        self._marker_items = []
        if self._markers_source:
            fn = getattr(self._model, self._markers_source, None)
            pts = fn() if callable(fn) else None
            if pts:
                xs = [int(p[2]) for p in pts if int(p[0]) == z]
                ys = [int(p[1]) for p in pts if int(p[0]) == z]
                if xs:
                    marker = pg.ScatterPlotItem(
                        xs,
                        ys,
                        pen=pg.mkPen("g", width=1),
                        brush=pg.mkBrush(0, 255, 0, 120),
                        size=8,
                        symbol="s",
                    )
                    marker.setZValue(5)
                    view.addItem(marker)
                    self._marker_items.append(marker)

        # selected-point marker (red circle)
        if self._pick_marker is not None:
            view.removeItem(self._pick_marker)
            self._pick_marker = None
        if self._select_attr:
            sel = getattr(self._model, self._select_attr, None)
            if sel is not None and int(sel[0]) == z:
                self._pick_marker = pg.ScatterPlotItem(
                    [int(sel[2])],
                    [int(sel[1])],
                    pen=pg.mkPen("r", width=2),
                    brush=None,
                    size=15,
                    symbol="o",
                )
                self._pick_marker.setZValue(9)
                view.addItem(self._pick_marker)

        # fitted lateral-FWHM circle (yellow)
        if self._roi_item is not None:
            view.removeItem(self._roi_item)
            self._roi_item = None
        if self._roi_source:
            fn = getattr(self._model, self._roi_source, None)
            roi = fn() if callable(fn) else None
            if roi and int(roi.get("z", z)) == z:
                r = float(roi["r"])
                cx, cy = float(roi["x"]), float(roi["y"])
                try:
                    self._roi_item = pg.CircleROI(
                        [cx - r, cy - r],
                        [2 * r, 2 * r],
                        pen=pg.mkPen("y", width=2),
                        movable=False,
                        resizable=False,
                    )
                    self._roi_item.setZValue(10)
                    view.addItem(self._roi_item)
                except Exception:  # pragma: no cover - CircleROI optional
                    self._roi_item = None

    # ── refresh ────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Re-read the model image (and selection) and redraw with the colormap."""
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
        data = np.asarray(img, dtype=float)
        self._ndim = data.ndim
        if data.ndim == 3:
            # Preserve the current slice across refreshes; map axes so the
            # displayed image is (y, x) = (axis 1, axis 2) of a (z, y, x) stack.
            try:
                prev = int(self._image.currentIndex)
            except Exception:
                prev = 0
            self._image.setImage(data, autoLevels=True, axes={"t": 0, "x": 2, "y": 1})
            if 0 <= prev < data.shape[0]:
                self._image.setCurrentIndex(prev)
        else:
            self._image.setImage(data, autoLevels=True)
        apply_colormap(self._image, self._current_cmap())
        if self._overlay is not None and self._selection_attr:
            sel = getattr(self._model, self._selection_attr, None)
            sel = (
                np.zeros_like(data)
                if sel is None or np.shape(sel) != data.shape
                else np.asarray(sel)
            )
            self._overlay.setImage(sel)
        if self._markers_source or self._roi_source or self._select_attr:
            self._redraw_overlays()


@register_section("image")
def _image_section_factory(model, target: str, **options):
    """Custom-section factory for a general 2D image dock (see :class:`ImageMapWidget`)."""
    return ImageMapWidget(model, target, **options)


# --- fit mixer (LifetimeMixtureModel AutoForm section) ---------------------
@register_section("fit_mixer")
class FitMixerWidget(QtWidgets.QWidget):
    """Fit-selector and fraction-parameter UI for the LifetimeMixture AutoForm section.

    Provides a combo box of existing lifetime fits, add/remove controls, a list
    of added components, and inline fraction-parameter widgets. Register it in a
    ``.view.json`` as::

        {"type": "custom", "key": "fit_mixer"}

    The model must expose the ``LifetimeMixtureModel`` API:
    ``lifetime_fits``, ``append_model(model, name)``, ``pop_model(idx)``,
    ``_fractions`` and ``model_names``.
    """

    def __init__(self, model=None, target=None, parent=None, **options):
        super().__init__(parent)
        self._model = model

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        # Toolbar: combo + refresh + name field + "all" checkbox + add button
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(2)

        self.cb = QtWidgets.QComboBox()
        self.cb.setToolTip("Select a lifetime fit to add to the mixture.")
        toolbar.addWidget(self.cb, 2)

        refresh_btn = QtWidgets.QToolButton()
        refresh_btn.setText("↻")
        refresh_btn.setToolTip("Refresh the list of available lifetime fits.")
        refresh_btn.clicked.connect(self._refresh_fit_list)
        toolbar.addWidget(refresh_btn)

        toolbar.addWidget(QtWidgets.QLabel("Name"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("fraction name…")
        self.name_edit.setMaximumWidth(90)
        self.name_edit.setToolTip("Name for the fraction parameter (default: x_N).")
        toolbar.addWidget(self.name_edit)

        self.all_cb = QtWidgets.QCheckBox("all")
        self.all_cb.setToolTip("Add all listed fits at once.")
        toolbar.addWidget(self.all_cb)

        add_btn = QtWidgets.QToolButton()
        add_btn.setText("add")
        add_btn.setToolTip("Add the selected fit to the mixture.")
        add_btn.clicked.connect(self._on_add)
        toolbar.addWidget(add_btn)

        outer.addLayout(toolbar)

        # Current component list (double-click removes)
        self.fit_list = QtWidgets.QListWidget()
        self.fit_list.setMaximumHeight(80)
        self.fit_list.setToolTip("Mixture components. Double-click a row to remove it.")
        self.fit_list.doubleClicked.connect(self._on_remove)
        outer.addWidget(self.fit_list)

        # Fraction parameter widgets (rebuilt after each add/remove)
        self._fractions_container = QtWidgets.QWidget()
        self._fractions_layout = QtWidgets.QGridLayout(self._fractions_container)
        self._fractions_layout.setContentsMargins(0, 0, 0, 0)
        self._fractions_layout.setSpacing(2)
        outer.addWidget(self._fractions_container)

        self._refresh_fit_list()
        self._rebuild_fractions()

    # -- helpers ---------------------------------------------------------------

    def _own_fit_index(self) -> int:
        try:
            import chisurf as cs

            fit = getattr(self._model, "fit", None)
            for i, fg in enumerate(cs.fits):
                if fg is fit or fit in list(fg):
                    return i
        except Exception:
            pass
        return 0

    def _dispatch_update(self) -> None:
        try:
            import chisurf as cs

            cs.core.actions.dispatch("fit.update", {"fit_index": int(self._own_fit_index())})
        except Exception:
            pass

    # -- slots -----------------------------------------------------------------

    def _refresh_fit_list(self) -> None:
        """Populate the combo box from the model's available lifetime fits."""
        self.cb.clear()
        for f in getattr(self._model, "lifetime_fits", []):
            self.cb.addItem(f.name)

    def _on_add(self) -> None:
        """Add the selected fit(s) to the mixture."""
        fits = getattr(self._model, "lifetime_fits", [])
        if not fits:
            return
        idxs = list(range(len(fits))) if self.all_cb.isChecked() else [self.cb.currentIndex()]
        for idx in idxs:
            if not (0 <= idx < len(fits)):
                continue
            f = fits[idx]
            i = self.fit_list.count() + 1
            name = self.name_edit.text().strip() or f"x_{i}"
            self.fit_list.addItem(f"{i}: {f.name}")
            try:
                self._model.append_model(f.model, name)
            except Exception:
                pass
        self._dispatch_update()
        self._rebuild_fractions()

    def _on_remove(self) -> None:
        """Remove the double-clicked fit from the mixture."""
        idx = self.fit_list.currentRow()
        if idx < 0:
            return
        self.fit_list.takeItem(idx)
        try:
            self._model.pop_model(idx)
        except Exception:
            pass
        # Renumber remaining items to keep indices consistent
        for i in range(self.fit_list.count()):
            item = self.fit_list.item(i)
            rest = item.text().split(": ", 1)[1] if ": " in item.text() else item.text()
            item.setText(f"{i + 1}: {rest}")
        self._dispatch_update()
        self._rebuild_fractions()

    def _rebuild_fractions(self) -> None:
        """Recreate the fraction-parameter widget grid from the model's state."""
        from chisurf.gui.widgets.fitting.parameter_widgets import make_fitting_parameter_widget

        layout = self._fractions_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        fractions = getattr(self._model, "_fractions", [])
        model_names = getattr(
            self._model, "model_names", [f"x{i + 1}" for i in range(len(fractions))]
        )
        if not fractions:
            return

        layout.addWidget(QtWidgets.QLabel("Fraction"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("Model"), 0, 1)
        for row, (frac, name) in enumerate(zip(fractions, model_names), start=1):
            layout.addWidget(make_fitting_parameter_widget(frac, label_text=""), row, 0)
            layout.addWidget(QtWidgets.QLabel(name), row, 1)
