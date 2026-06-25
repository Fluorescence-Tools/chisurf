"""Render a model's editor from its UI-agnostic :class:`ModelView`.

:class:`AutoModelWidget` walks the ``view_spec()`` returned by a pure
:class:`~chisurf.core.models.model.Model` and builds the control panel by
composition — it *has a* model, it is not one. Generic sections are drawn from
the parameters themselves; bespoke ones are looked up in the section registry by
key. This is the single renderer that replaces the per-model hand-written
widgets.
"""
from __future__ import annotations

from qtpy import QtCore, QtWidgets

import chisurf as cs
from chisurf import logging
from chisurf.core.models import view_spec as vs
from chisurf.gui.widgets.fitting import (
    make_fitting_parameter_widget,
    make_fitting_parameter_group_widget,
)
from . import sections  # ensures builtin registrations are imported
from .sections.registry import get_section_factory


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


class AutoModelWidget(QtWidgets.QWidget):
    """Build a model's control panel from ``model.view_spec()``.

    Parameters
    ----------
    model : chisurf.core.models.model.Model
        The pure model whose editor should be rendered.
    parent : QtWidgets.QWidget, optional
        Parent widget.
    """

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self._param_widgets = []
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

        view = self.model.view_spec()
        for section in view.sections:
            try:
                widget = self._build_section(section)
            except Exception as exc:  # pragma: no cover - defensive
                logging.error(f"AutoModelWidget: failed to build section {section}: {exc}")
                continue
            if widget is not None:
                widget.setVisible(bool(section.visible))
                self._layout.addWidget(widget)

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
                logging.warning(f"AutoModelWidget: find_parameters failed for {section.target!r}: {exc}")
        widget = make_fitting_parameter_group_widget(group, n_col=section.n_col)
        if section.title:
            widget.setTitle(section.title)
        self._param_widgets.extend(self._collect_param_widgets(widget))
        return widget

    def _build_panel(self, section: vs.PanelSection):
        from chisurf.gui.widgets.collapsible_box import CollapsibleBox

        collapsed = bool(section.collapsed) or self._collapsed_when(section.collapsed_when)
        box = CollapsibleBox(
            section.title or "",
            expanded=not collapsed,
        )
        # the header is purely cosmetic when not collapsible
        if not section.collapsible:
            box._btn.setEnabled(False)
        for child in section.sections:
            try:
                widget = self._build_section(child)
            except Exception as exc:  # pragma: no cover - defensive
                logging.error(f"AutoModelWidget: failed to build {child}: {exc}")
                continue
            if widget is not None:
                widget.setVisible(bool(child.visible))
                box.add_widget(widget)
        return box

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

        box = QtWidgets.QGroupBox(section.title or getattr(group, "name", ""))
        outer = QtWidgets.QVBoxLayout(box)
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

        rows_host = QtWidgets.QWidget()
        rows_layout = QtWidgets.QGridLayout(rows_host)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(0)
        outer.addWidget(rows_host)

        def render_rows():
            while rows_layout.count():
                item = rows_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
            params = list(getattr(group, "parameters_all", []))
            width = max(1, int(section.row_width))
            for i, p in enumerate(params):
                label = p.__dict__.get("label_text", p.name)
                pw = make_fitting_parameter_widget(fitting_parameter=p, label_text=label)
                rows_layout.addWidget(pw, i // width, i % width)
                self._param_widgets.append(pw)

        def on_add():
            group.append()
            self._dispatch_fit_update()
            render_rows()

        def on_del():
            if len(getattr(group, "parameters_all", [])) // max(1, section.row_width) > section.min_rows:
                if hasattr(group, "pop"):
                    group.pop()
                    self._dispatch_fit_update()
                    render_rows()

        add_btn.clicked.connect(on_add)
        del_btn.clicked.connect(on_del)
        render_rows()
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
