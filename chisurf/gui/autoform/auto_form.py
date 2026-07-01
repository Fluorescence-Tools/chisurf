"""Render an object's editor from its UI-agnostic :class:`DataSet`/``ModelView``.

:class:`AutoForm` walks the declarative spec (today: a model's ``view_spec()``)
and builds the control panel by composition — it *has a* bound object, it is not
one. Generic sections are drawn from the parameters themselves; bespoke ones are
looked up in the section registry by key. This is the single renderer that
replaces the per-domain hand-written widgets (PRD-40). ``AutoModelWidget`` is
kept as a backwards-compatible alias.
"""

from __future__ import annotations

from qtpy import QtCore, QtWidgets

import chisurf as cs
from chisurf import logging
from chisurf.core import dataspec as vs
from chisurf.gui.widgets.fitting import make_fitting_parameter_widget

from . import sections  # noqa: F401  (side effect: populate the registry)
from .sections.registry import get_section_factory

#: How many label/field pairs are packed onto one row of the compact field grid.
FIELDS_PER_ROW = 2

ADD_BUTTON_STYLE = (
    "QPushButton { background-color: #1f7a1f; color: white; border: 1px solid #166016; "
    "border-radius: 3px; padding: 2px 8px; }"
    "QPushButton:hover { background-color: #249124; }"
)
REMOVE_BUTTON_STYLE = (
    "QPushButton { background-color: #a82020; color: white; border: 1px solid #7d1717; "
    "border-radius: 3px; padding: 2px 8px; }"
    "QPushButton:hover { background-color: #bf2626; }"
)


def _make_field_shrinkable(field) -> None:
    """Let a field's editors shrink so the form scales to narrow docks/panels.

    Spin boxes and combo boxes default to a wide intrinsic minimum (the value text
    plus arrows, or the longest combo item), which forces the compact two-column
    grid to overflow horizontally instead of sharing the available width. Drop that
    floor and let the column stretch decide the width so the editors track the panel.
    """
    field.setMinimumWidth(0)
    editors = field.findChildren(
        (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QLineEdit)
    )
    for editor in editors:
        editor.setMinimumWidth(0)
        editor.setSizePolicy(QtWidgets.QSizePolicy.Expanding, editor.sizePolicy().verticalPolicy())
        if isinstance(editor, QtWidgets.QComboBox):
            editor.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            editor.setMinimumContentsLength(3)


def _align_label_columns(param_widgets):
    """Give a batch of fitting-parameter rows one shared label width.

    Each :class:`FittingParameterWidget` is a self-contained row, so without
    this their name labels self-size and the fix/link/value/error columns end up
    ragged from row to row. Setting every label to the widest label's width (up
    to a sane cap) lines the columns up cleanly; the full name stays available
    as the label tooltip.
    """
    labels = [getattr(pw, "label", None) for pw in param_widgets]
    labels = [lbl for lbl in labels if lbl is not None]
    if not labels:
        return
    width = min(max(lbl.sizeHint().width() for lbl in labels), 120)
    for lbl in labels:
        if not lbl.toolTip():
            lbl.setToolTip(lbl.text())
        lbl.setFixedWidth(width)


class AutoForm(QtWidgets.QWidget):
    """Build a bound object's control panel from its declarative spec.

    Parameters
    ----------
    model : chisurf.core.models.model.Model
        The pure object whose editor should be rendered (a model today).
    parent : QtWidgets.QWidget, optional
        Parent widget.
    """

    @classmethod
    def from_parameter_group(cls, group, parent=None, **kwargs):
        """Render a ``FittingParameterGroup`` directly, without authored JSON.

        Thin convenience over :class:`chisurf.core.dataspec.ParameterGroupView`;
        ``kwargs`` (``title``/``n_col``/``collapsible``/``collapsed``) are
        forwarded to it.
        """
        from chisurf.core.dataspec import ParameterGroupView

        return cls(ParameterGroupView(group, **kwargs), parent=parent)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self._param_widgets = []
        self._dock_areas = []
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setAlignment(QtCore.Qt.AlignTop)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.rebuild()

    # -- public API ---------------------------------------------------------
    def rebuild(self):
        """(Re)build the whole panel from the current view-spec."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._param_widgets = []
        self._dock_areas = []

        view = self.model.view_spec()
        self._emit_sections(view.sections, self._layout.addWidget)
        # If the view contains an expanding widget (e.g. a dock area or plot), let it take
        # the spare vertical space; otherwise top-align the panels with a trailing stretch.
        expanding = False
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if w is not None and getattr(w, "_autoform_expanding", False):
                self._layout.setStretch(i, 1)
                expanding = True
        if not expanding:
            self._layout.addStretch(1)  # push panels to the top; prevent height distribution

    def sync_fields(self):
        """Re-read model values into existing field widgets without rebuilding.

        Use this after changing model attributes programmatically so the controls reflect
        the new values *without* tearing down the layout (which would, e.g., reset a dock
        arrangement). Only widgets exposing a ``sync()`` method are updated.
        """
        for w in self.findChildren(QtWidgets.QWidget):
            sync = getattr(w, "sync", None)
            if callable(sync) and getattr(w, "is_form_field", False):
                try:
                    sync()
                except Exception:
                    pass

    def refresh_plots(self):
        """Re-read and redraw every inline :class:`PlotSection` in the form.

        Call this after the model's data changes (e.g. a recompute) so embedded
        plots update without rebuilding the whole editor.
        """
        from .sections.builtin import PlotWidget

        seen = set()
        for w in self.findChildren(PlotWidget):
            seen.add(id(w))
            try:
                w.refresh()
            except Exception:
                pass
        # Also refresh custom widgets opting in via the AUTOFORM_REFRESH marker
        # (e.g. the reusable L-curve view and the 2D map docks).
        for w in self.findChildren(QtWidgets.QWidget):
            if id(w) in seen or not getattr(w, "AUTOFORM_REFRESH", False):
                continue
            try:
                w.refresh()
            except Exception:
                pass

    def _emit_sections(self, section_list, emit, fields_per_row=None):
        """Build sections, grouping consecutive simple fields into one form.

        Field sections (value / choice / toggle, marked ``is_form_field``) are
        accumulated and flushed into a single compact ``QGridLayout`` that packs
        ``fields_per_row`` label/field pairs per row (defaulting to
        ``FIELDS_PER_ROW``) to save vertical space; passing ``1`` gives a
        single-column form layout that saves horizontal space. Any other section
        (panel, parameter grid, curve input) flushes the run and is emitted
        full-width.
        """
        pending = []

        def flush():
            if not pending:
                return
            container = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(container)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(2)
            per_row = max(1, fields_per_row or FIELDS_PER_ROW)
            for i, field in enumerate(pending):
                r, c = divmod(i, per_row)
                col = c * 2
                label = QtWidgets.QLabel(getattr(field, "form_label", ""))
                label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                tip = field.toolTip()
                if tip:
                    label.setToolTip(tip)
                # Fields stretch horizontally to share the available width; the
                # field columns carry the stretch, the label columns stay fixed.
                field.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                _make_field_shrinkable(field)
                grid.addWidget(label, r, col)
                grid.addWidget(field, r, col + 1)
                grid.setColumnStretch(col + 1, 1)
            pending.clear()
            emit(container)

        for section in section_list:
            try:
                widget = self._build_section(section)
            except Exception as exc:  # pragma: no cover - defensive
                logging.error(f"AutoForm: failed to build section {section}: {exc}")
                continue
            if widget is None:
                continue
            widget.setVisible(bool(section.visible))
            # Default path for inline help: a section's ``description`` becomes
            # the widget's tooltip. Field widgets additionally set it on their
            # editor (Qt tooltips do not propagate to child widgets).
            desc = getattr(section, "description", "")
            if desc:
                widget.setToolTip(desc)
            if getattr(widget, "is_form_field", False):
                pending.append(widget)
            else:
                flush()
                emit(widget)
        flush()

    @property
    def parameter_widgets(self):
        """Flat list of parameter widgets currently rendered (for tests/sync)."""
        return list(self._param_widgets)

    # -- section dispatch ---------------------------------------------------
    def _build_section(self, section: vs.Section):
        if isinstance(section, vs.PanelSection):
            return self._build_panel(section)
        if isinstance(section, vs.DynamicGroupSection):
            return self._build_dynamic_group(section)
        if isinstance(section, vs.CurveInputSection):
            return self._build_curve_input(section)
        if isinstance(section, vs.ChoiceSection):
            from .sections.builtin import ChoiceWidget

            return ChoiceWidget(self.model, section)
        if isinstance(section, vs.ToggleSection):
            from .sections.builtin import ToggleWidget

            return ToggleWidget(self.model, section)
        if isinstance(section, vs.ToggleRowSection):
            from .sections.builtin import ToggleRowWidget

            return ToggleRowWidget(self.model, section)
        if isinstance(section, vs.ButtonRowSection):
            from .sections.builtin import ButtonRowWidget

            return ButtonRowWidget(self.model, section)
        if isinstance(section, vs.ValueSection):
            from .sections.builtin import ValueWidget

            return ValueWidget(self.model, section)
        if isinstance(section, vs.PlotSection):
            from .sections.builtin import PlotWidget

            return PlotWidget(self.model, section)
        if isinstance(section, vs.DockAreaSection):
            return self._build_dock_area(section)
        if isinstance(section, vs.WizardSection):
            return self._build_wizard(section)
        if isinstance(section, vs.InfoSection):
            from .sections.builtin import InfoWidget

            return InfoWidget(self.model, section)
        if isinstance(section, vs.ParameterGroupTableSection):
            return self._build_parameter_group_table(section)
        if isinstance(section, vs.ParameterGroupSection):
            return self._build_parameter_group(section)
        if isinstance(section, vs.CustomSection):
            return self._build_custom(section)
        logging.warning(f"AutoModelWidget: unknown section type {type(section).__name__}")
        return None

    def _resolve_group(self, target):
        group = getattr(self.model, target, None)
        if group is None:
            logging.warning(f"AutoModelWidget: target {target!r} did not resolve")
        return group

    def _make_fold_box(self, section, fallback_title: str = ""):
        """Build a :class:`CollapsibleBox` from a section's fold attributes.

        Honors ``collapsible`` / ``collapsed`` / ``collapsed_when`` when present
        (``PanelSection``, ``ParameterGroupSection``,
        ``ParameterGroupTableSection``, ``DynamicGroupSection``), falling back
        to an always-expanded box otherwise.
        """
        from chisurf.gui.widgets.collapsible_box import CollapsibleBox

        collapsed = bool(getattr(section, "collapsed", False)) or self._collapsed_when(
            getattr(section, "collapsed_when", None)
        )
        box = CollapsibleBox(section.title or fallback_title, expanded=not collapsed)
        # the header is purely cosmetic when not collapsible
        if not getattr(section, "collapsible", True):
            box._btn.setEnabled(False)
        return box

    def _build_parameter_group(self, section: vs.ParameterGroupSection):
        group = self._resolve_group(section.target)
        if group is None:
            return None
        # Most groups define their parameters as plain attributes
        # (``self._dt = FittingParameter(...)``) that only surface in
        # ``parameters_all`` after ``find_parameters()`` aggregates them. The
        # Lifetime group fills its list eagerly via ``append()``; the others do
        # not, so without this the section would render empty.
        if not list(getattr(group, "parameters_all", [])) and hasattr(group, "find_parameters"):
            try:
                group.find_parameters()
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning(
                    f"AutoModelWidget: find_parameters failed for {section.target!r}: {exc}"
                )

        if section.exclude_source:
            try:
                excluded = {id(p) for p in getattr(group, section.exclude_source)()}
            except Exception:
                excluded = set()
            params = [p for p in getattr(group, "parameters_all", []) if id(p) not in excluded]
        else:
            params = list(getattr(group, "parameters_all", []))

        grid = self._build_param_grid(params, section.n_col)
        self._param_widgets.extend(self._collect_param_widgets(grid))

        if not getattr(section, "collapsible", True):
            # No wrapper box — used when nested inside a PanelSection
            return grid

        box = self._make_fold_box(section, fallback_title=getattr(group, "name", ""))
        box.add_widget(grid)
        return box

    def _build_param_grid(self, params, n_col=None):
        """Build a bare ``QWidget`` grid from a pre-filtered parameter list."""
        import chisurf.core.settings

        if n_col is None:
            n_col = chisurf.core.settings.gui["fit_models"]["n_columns"]
        n_col = max(1, int(n_col))
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        pws = []
        for i, p in enumerate(params):
            label_text = p.__dict__.get("label_text", p.name)
            pw = make_fitting_parameter_widget(fitting_parameter=p, label_text=label_text)
            grid.addWidget(pw, i // n_col, i % n_col)
            pws.append(pw)
        _align_label_columns(pws)
        return container

    def _build_parameter_group_table(self, section: vs.ParameterGroupTableSection):
        group = self._resolve_group(section.target)
        if group is None:
            return None

        if not list(getattr(group, "parameters_all", [])) and hasattr(group, "find_parameters"):
            try:
                group.find_parameters()
            except Exception as exc:
                logging.warning(
                    f"AutoModelWidget: find_parameters failed for {section.target!r}: {exc}"
                )

        if section.exclude_source:
            try:
                excluded = {id(p) for p in getattr(group, section.exclude_source)()}
            except Exception:
                excluded = set()
            params = [p for p in getattr(group, "parameters_all", []) if id(p) not in excluded]
        else:
            params = list(getattr(group, "parameters_all", []))

        if not params:
            return None

        from chisurf.gui.autoform.sections.parameter_table import ParameterGroupTableWidget

        table = ParameterGroupTableWidget(
            params=params,
            section=section,
            on_change=self._dispatch_fit_update,
        )

        if not getattr(section, "collapsible", True):
            return table

        box = self._make_fold_box(section, fallback_title=getattr(group, "name", ""))
        box.add_widget(table)
        return box

    def _build_panel(self, section: vs.PanelSection):
        box = self._make_fold_box(section)
        self._emit_sections(section.sections, box.add_widget, fields_per_row=section.n_col)
        return box

    def _build_dock_area(self, section: vs.DockAreaSection):
        """Render a declarative dock area: each child section becomes a dock tab.

        Uses the same ChiSurf ``DockArea`` the fit windows use, so the panels are
        rearrangeable / floatable but the layout is authored in the ``.view.json``.
        """
        from chisurf.gui.widgets.dock_area import DockArea

        area = DockArea()
        self._dock_areas.append(area)
        # Let rebuild() give the dock area the spare vertical space instead of a trailing
        # stretch, so its panels fill the height.
        area._autoform_expanding = True
        area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        if getattr(section, "height", 0):
            area.setMinimumHeight(int(section.height))
        for i, child in enumerate(section.sections):
            name = (
                getattr(child, "title", None) or getattr(child, "label", None) or f"Panel {i + 1}"
            )
            try:
                if isinstance(child, vs.PanelSection):
                    # The dock tab already carries the panel's title, so render the panel's
                    # contents directly (no redundant outer collapsible) and wrap them in a
                    # scroll area so the tab expands vertically and scrolls when needed.
                    inner = QtWidgets.QWidget()
                    lay = QtWidgets.QVBoxLayout(inner)
                    lay.setContentsMargins(0, 0, 0, 0)
                    self._emit_sections(child.sections, lay.addWidget, fields_per_row=child.n_col)
                    # Give spare vertical space to an expanding child (a plot/image) if the
                    # panel has one; otherwise keep the fields top-aligned and compact with a
                    # trailing stretch (so form rows don't spread into large gaps).
                    expanding = False
                    for r in range(lay.count()):
                        w = lay.itemAt(r).widget()
                        if w is not None and getattr(w, "_autoform_expanding", False):
                            lay.setStretch(r, 1)
                            expanding = True
                    if not expanding:
                        lay.addStretch(1)
                    inner.setSizePolicy(
                        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
                    )
                    widget = QtWidgets.QScrollArea()
                    widget.setWidgetResizable(True)
                    widget.setFrameShape(QtWidgets.QFrame.NoFrame)
                    widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                    widget.setWidget(inner)
                else:
                    widget = self._build_section(child)
            except Exception:
                widget = None
            if widget is None:
                continue
            try:
                area.add_panel(widget, str(name))
            except Exception:
                pass
        # Remember the user's dock arrangement across sessions when the view asks
        # for it (all panels are added by now, so restore-on-show can find them).
        if getattr(section, "persist", ""):
            try:
                area.enable_persistence(section.persist)
            except Exception:
                pass
        return area

    def _build_wizard(self, section: vs.WizardSection):
        """Render a directed two-column wizard from a :class:`WizardSection`.

        Each step's body is built with the same ``_emit_sections`` used everywhere
        else, so the step controls bind to this form's model. The step-completion
        predicate reuses the ``{target, attr, equals}`` condition evaluator so a
        ``complete_when`` gates *Next* (in a linear wizard) and drives the ✓ mark.
        """
        from .sections.wizard_section import WizardWidget

        pages = []
        for step in section.steps:
            body = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(body)
            lay.setContentsMargins(0, 0, 0, 0)
            self._emit_sections(step.sections, lay.addWidget)
            # Give spare vertical space to an expanding child (an embedded editor
            # or plot); otherwise keep the fields top-aligned with a trailing stretch.
            expanding = False
            for r in range(lay.count()):
                w = lay.itemAt(r).widget()
                if w is not None and getattr(w, "_autoform_expanding", False):
                    lay.setStretch(r, 1)
                    expanding = True
            if not expanding:
                lay.addStretch(1)
            pages.append(body)

        def _is_complete(index, _steps=section.steps):
            cond = getattr(_steps[index], "complete_when", None)
            # A step with no explicit condition is treated as complete (an
            # informational step never blocks a linear wizard).
            return True if not cond else self._collapsed_when(cond)

        return WizardWidget(section, pages, _is_complete)

    def _collapsed_when(self, cond) -> bool:
        """Evaluate a ``{target, attr, equals}`` fold condition against the model."""
        if not cond:
            return False
        try:
            target = cond.get("target")
            group = getattr(self.model, target) if target else self.model
            value = getattr(group, cond["attr"])
            return str(value).lower() == str(cond["equals"]).lower()
        except Exception:
            return False

    def _build_curve_input(self, section: vs.CurveInputSection):
        from .sections.builtin import CurveInputWidget

        return CurveInputWidget(self.model, section)

    def _build_custom(self, section: vs.CustomSection):
        factory = get_section_factory(section.key)
        if factory is None:
            logging.warning(f"AutoModelWidget: no custom section registered for {section.key!r}")
            return None
        return factory(model=self.model, target=section.target, **dict(section.options))

    def _build_dynamic_group(self, section: vs.DynamicGroupSection):
        group = self._resolve_group(section.target)
        if group is None:
            return None

        # Outer container always holds header + rows (with or without CollapsibleBox)
        content = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(content)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(0)

        # header: add/del + any registered header widgets
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(0)
        add_btn = QtWidgets.QPushButton(section.add_label)
        add_btn.setStyleSheet(ADD_BUTTON_STYLE)
        del_btn = QtWidgets.QPushButton(section.remove_label)
        del_btn.setStyleSheet(REMOVE_BUTTON_STYLE)
        header.addWidget(add_btn)
        header.addWidget(del_btn)
        for key in section.header_keys:
            factory = get_section_factory(key)
            if factory is not None:
                try:
                    header.addWidget(factory(model=self.model, target=section.target))
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning(f"AutoModelWidget: header {key!r} failed: {exc}")
        outer.addLayout(header)

        # rows host: VBox for per-component fold groups, Grid for flat params
        rows_host = QtWidgets.QWidget()
        if section.component_title:
            rows_layout = QtWidgets.QVBoxLayout(rows_host)
        else:
            rows_layout = QtWidgets.QGridLayout(rows_host)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(0)
        outer.addWidget(rows_host)

        def _row_params():
            if section.rows_source:
                fn = getattr(group, section.rows_source, None)
                return list(fn()) if callable(fn) else []
            return list(getattr(group, "parameters_all", []))

        def render_rows():
            while rows_layout.count():
                item = rows_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
            params = _row_params()
            width = max(1, int(section.row_width))
            if section.component_title:
                # Each row_width chunk of params gets its own CollapsibleBox
                from chisurf.gui.widgets.collapsible_box import CollapsibleBox

                n = len(params) // width if width else 0
                for comp_idx in range(n):
                    chunk = params[comp_idx * width : (comp_idx + 1) * width]
                    comp_box = CollapsibleBox(
                        f"{section.component_title} {comp_idx + 1}", expanded=True
                    )
                    row_widget = QtWidgets.QWidget()
                    row_grid = QtWidgets.QGridLayout(row_widget)
                    row_grid.setContentsMargins(0, 0, 0, 0)
                    row_grid.setSpacing(0)
                    for j, p in enumerate(chunk):
                        label = p.__dict__.get("label_text", p.name)
                        pw = make_fitting_parameter_widget(fitting_parameter=p, label_text=label)
                        row_grid.addWidget(pw, 0, j)
                        self._param_widgets.append(pw)
                    comp_box.add_widget(row_widget)
                    rows_layout.addWidget(comp_box)
            else:
                batch = []
                for i, p in enumerate(params):
                    label = p.__dict__.get("label_text", p.name)
                    pw = make_fitting_parameter_widget(fitting_parameter=p, label_text=label)
                    rows_layout.addWidget(pw, i // width, i % width)
                    self._param_widgets.append(pw)
                    batch.append(pw)
                _align_label_columns(batch)

        def on_add():
            add_fn = getattr(group, section.append_method, None)
            if callable(add_fn):
                add_fn()
            self._dispatch_fit_update()
            render_rows()

        def on_del():
            if len(_row_params()) // max(1, section.row_width) > section.min_rows:
                del_fn = getattr(group, section.remove_method, None)
                if callable(del_fn):
                    del_fn()
                    self._dispatch_fit_update()
                    render_rows()

        add_btn.clicked.connect(on_add)
        del_btn.clicked.connect(on_del)
        render_rows()

        if not getattr(section, "collapsible", True):
            # No wrapper box — used when section is already inside a PanelSection
            return content

        box = self._make_fold_box(section, fallback_title=getattr(group, "name", ""))
        box.add_widget(content)
        return box

    # -- helpers ------------------------------------------------------------
    def _dispatch_fit_update(self):
        try:
            fit = getattr(self.model, "fit", None)
            fits = cs.fits if hasattr(cs, "fits") else []
            idx = next((i for i, f in enumerate(fits) if f is fit), 0)
            cs.core.actions.dispatch(name="fit.update", payload={"fit_index": int(idx)})
        except Exception:  # pragma: no cover - dispatcher optional in tests
            pass

    @staticmethod
    def _collect_param_widgets(group_widget):
        from chisurf.gui.widgets.fitting import FittingParameterWidget

        return group_widget.findChildren(FittingParameterWidget)


#: Backwards-compatible alias from the PRD-38 model-only name.
AutoModelWidget = AutoForm
