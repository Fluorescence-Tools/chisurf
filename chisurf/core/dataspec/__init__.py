"""UI-agnostic description of an editor (a declarative dataset → editor spec).

This package is the chisurf-native, guidata-like vocabulary for declaring *how
a structured, typed object wants to be edited* — without importing or
referencing any GUI toolkit (PRD-40). It is the contract that enforces the
strict split between data/computation (``chisurf/core``) and presentation
(``chisurf/gui``):

* An object returns a :class:`ModelView` describing its editor. It is plain,
  picklable data — dataclasses of strings, numbers and dicts.
* Anything that needs a real widget (a bespoke panel, a plot) is referenced by
  a *string key* and a *target attribute name*, never by a widget class. The
  GUI side owns a registry that maps those keys to concrete widgets.

Because this package is pure data, an entire editor can be inspected and
asserted in a headless test with Qt not even installed::

    spec = model.view_spec()
    assert any(s.target == "lifetimes" for s in spec.sections)

It was lifted out of ``chisurf.core.models.view_spec`` (PRD-38) so editors for
models, settings, metadata and tool panels can all be declared the same way;
``chisurf.core.models.view_spec`` remains as a backwards-compatible shim. See
:mod:`chisurf.gui.widgets.models.auto_model_widget` for the renderer that
consumes these descriptors.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

from chisurf import typing


@dataclasses.dataclass(frozen=True)
class Section:
    """Base type for every section in a :class:`ModelView`.

    A section describes one block of the editor. Subclasses carry the
    information the GUI needs to build that block. The ``target`` field, when
    set, names an attribute on the bound object (resolved with ``getattr``)
    so the descriptor itself stays free of object references.
    """

    #: Attribute name on the model resolved by the renderer (e.g. ``"lifetimes"``).
    target: typing.Optional[str] = None
    #: Optional title shown for the section. Falls back to the group name.
    title: typing.Optional[str] = None
    #: Whether the section is initially visible (e.g. nuisances may be hidden).
    visible: bool = True
    #: Human-readable description of the control. By default the renderer maps
    #: this to the widget's tooltip, so every section type can carry inline help
    #: straight from the ``.view.json``.
    description: str = ""


@dataclasses.dataclass(frozen=True)
class ParameterGroupSection(Section):
    """A static grid of the parameters of one :class:`FittingParameterGroup`.

    The renderer resolves ``target`` to a parameter group on the model and
    lays out every parameter using the existing parameter-widget factory. All
    per-parameter metadata (label, bounds, fixed, units) already lives on the
    parameters, so no further description is required here.
    """

    #: Number of columns in the grid, or ``None`` for the configured default.
    n_col: typing.Optional[int] = None
    #: Optional group method returning parameters to *exclude* from this grid
    #: (e.g. ``"_rotation_parameters"`` so anisotropy's fixed params render here
    #: while the variable rotation components render in a dynamic group).
    exclude_source: typing.Optional[str] = None
    #: Whether the section header can fold/unfold its parameters.
    collapsible: bool = True
    #: Initial fold state (``True`` = start collapsed).
    collapsed: bool = False
    #: Optional condition that folds the section: ``{target, attr, equals}``.
    collapsed_when: typing.Optional[typing.Mapping[str, typing.Any]] = None


@dataclasses.dataclass(frozen=True)
class DynamicGroupSection(Section):
    """A variable-length list of parameter rows with add/remove controls.

    Used by models whose component count is user-controlled (lifetimes,
    Gaussian distance distributions, FRET rates, ...). The renderer drives the
    add/remove buttons through the *model's own* ``append``/``pop`` methods, so
    the structural definition of "what a row is" stays exclusively in the
    model.
    """

    #: Labels of the controls; purely cosmetic.
    add_label: str = "add"
    remove_label: str = "del"
    #: Number of parameters that make up one logical row (e.g. 2 for an
    #: amplitude/lifetime pair). The renderer lays ``parameters_all`` out in a
    #: grid this many columns wide so paired parameters share a row.
    row_width: int = 1
    #: Minimum number of rows that must remain.
    min_rows: int = 1
    #: Optional keys of registered extra controls rendered in the header
    #: (e.g. ``("lifetime_link",)`` for the link/read menus).
    header_keys: typing.Tuple[str, ...] = ()
    #: Group method called to add a component (default ``"append"``; anisotropy
    #: uses ``"add_rotation"``).
    append_method: str = "append"
    #: Group method called to remove the last component (default ``"pop"``;
    #: anisotropy uses ``"remove_rotation"``).
    remove_method: str = "pop"
    #: Optional group method returning *only* the parameters that make up the
    #: rows (default: all of ``parameters_all``). Lets a group with fixed plus
    #: variable parameters expose only the variable ones here.
    rows_source: typing.Optional[str] = None
    #: Whether the section wraps its rows in a wrapper CollapsibleBox.
    #: Set False when the section sits inside a PanelSection that already provides
    #: the fold chrome (e.g. the Lifetimes and Anisotropy panels).
    collapsible: bool = True
    #: Initial fold state (``True`` = start collapsed).
    collapsed: bool = False
    #: Optional condition that folds the section: ``{target, attr, equals}``.
    collapsed_when: typing.Optional[typing.Mapping[str, typing.Any]] = None
    #: When set, each ``row_width`` group of params is rendered in its own
    #: CollapsibleBox titled ``"{component_title} {n}"``.  Requires
    #: ``collapsible=False`` (the outer box is suppressed; components provide
    #: the per-item fold chrome instead).
    component_title: typing.Optional[str] = None


@dataclasses.dataclass(frozen=True)
class CurveInputSection(Section):
    """A data-curve picker bound to the model via an action (e.g. the IRF).

    Some groups need a *curve* input, not a scalar parameter: convolution needs
    an IRF, generic needs a background curve, corrections needs a linearization
    table. They all share one shape — an experimental-data selector whose
    selection dispatches an action carrying the chosen curve's index and name,
    plus an optional "unload" action. Encoding that shape here (rather than a
    bespoke widget) keeps these inputs authorable in the ``.view.json`` while the
    GUI owns the single generic selector widget.

    The renderer dispatches ``select_action`` with payload
    ``{index_key: <idx>, name_key: <name>, "fit_index": <i>}`` on selection and
    ``unload_action`` (if set) on unload.
    """

    #: Label shown next to the selector (e.g. ``"IRF"``).
    label: str = "Curve"
    #: Action dispatched when a curve is selected. Required to be useful.
    select_action: str = ""
    #: Action dispatched when the curve is unloaded. Optional.
    unload_action: str = ""
    #: Payload key carrying the selected curve's dataset index.
    index_key: str = "idx"
    #: Payload key carrying the selected curve's name.
    name_key: str = "name"
    #: Optional attribute (on the target group) holding the current curve, read
    #: to display its name; e.g. ``"irf"``.
    name_attr: typing.Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PanelSection(Section):
    """A foldable group box that contains an ordered list of child sections.

    This is how the editor is organised into the familiar collapsible blocks
    (Convolution, Generic, Corrections, Lifetimes, Anisotropy). Children are
    rendered in declaration order, so authoring the special controls (a curve
    input, a choice) *before* the ``parameter_group`` places them at the top of
    the panel — matching the hand-written widgets.

    ``collapsed`` sets the initial fold state; ``collapsed_when`` folds the panel
    when a model attribute has a given value (e.g. fold anisotropy under magic
    angle): ``{"target": "anisotropy", "attr": "polarization_type",
    "equals": "vm"}``.
    """

    #: Whether the panel header can fold/unfold its contents.
    collapsible: bool = True
    #: Initial fold state (``True`` = start collapsed).
    collapsed: bool = False
    #: Optional condition that folds the panel: ``{target, attr, equals}``.
    collapsed_when: typing.Optional[typing.Mapping[str, typing.Any]] = None
    #: Ordered child sections rendered inside the panel.
    sections: typing.Tuple[Section, ...] = ()


@dataclasses.dataclass(frozen=True)
class ChoiceSection(Section):
    """A one-of-N selector (combo/radio) bound to a model attribute or action.

    Covers the bespoke enum controls the hand-written widgets carried: the
    convolution type (``per``/``exp``/``full``), the linearization smoothing
    window, the polarization type. Either it binds directly to ``attr`` on the
    ``target`` group (``setattr`` then a model update), or it dispatches
    ``set_action`` with ``{**action_fixed, value_key: <choice>, "fit_index": i}``.
    Options are listed inline, or sourced from a named list via
    ``options_source`` (e.g. ``"window_function_types"``).
    """

    label: str = "Choice"
    #: Attribute on the target group to get/set (direct-binding mode).
    attr: typing.Optional[str] = None
    #: Inline option values (the values written to the model attribute).
    options: typing.Tuple[str, ...] = ()
    #: Optional display labels; when set, labels[i] is shown but options[i] is committed.
    labels: typing.Tuple[str, ...] = ()
    #: Named source for options resolved GUI-side (e.g. ``"window_function_types"``).
    options_source: typing.Optional[str] = None
    #: Action dispatched on change instead of ``setattr`` (action-binding mode).
    set_action: str = ""
    #: Fixed payload merged into the dispatch (e.g. ``{"correction_type": "..."}``).
    action_fixed: typing.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)
    #: Payload key carrying the chosen value.
    value_key: str = "value"
    #: Render hint: ``"combo"`` or ``"radio"``.
    style: str = "combo"


@dataclasses.dataclass(frozen=True)
class ToggleSection(Section):
    """A boolean checkbox bound to a model attribute or action.

    Covers ``do_convolution`` and the on/off correction flags. Like
    :class:`ChoiceSection` it either sets ``attr`` on the ``target`` group or
    dispatches ``set_action`` with the boolean under ``value_key``.
    """

    label: str = "Enabled"
    attr: typing.Optional[str] = None
    set_action: str = ""
    action_fixed: typing.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)
    value_key: str = "value"


@dataclasses.dataclass(frozen=True)
class ValueSection(Section):
    """A single scalar field (int / float / string) bound to the object.

    The generic typed-field primitive (the ``IntItem`` / ``FloatItem`` /
    ``StringItem`` of PRD-40), selected via ``kind``. Like :class:`ChoiceSection`
    and :class:`ToggleSection` it either writes ``attr`` on the ``target`` group
    (``setattr``) or dispatches ``set_action`` with the value under
    ``value_key``. The numeric bounds/step/decimals/suffix only apply to the
    ``int``/``float`` kinds; ``placeholder`` only to ``str``.
    """

    label: str = "Value"
    #: One of ``"int"``, ``"float"``, ``"str"``.
    kind: str = "str"
    #: Attribute on the target group to get/set (direct-binding mode).
    attr: typing.Optional[str] = None
    #: Action dispatched on change instead of ``setattr`` (action-binding mode).
    set_action: str = ""
    #: Fixed payload merged into the dispatch.
    action_fixed: typing.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)
    #: Payload key carrying the new value.
    value_key: str = "value"
    #: Numeric constraints (int/float kinds).
    minimum: typing.Optional[float] = None
    maximum: typing.Optional[float] = None
    step: typing.Optional[float] = None
    decimals: int = 3
    suffix: str = ""
    #: Placeholder text (str kind).
    placeholder: str = ""
    #: Render the field as read-only (display/output field).
    read_only: bool = False


@dataclasses.dataclass(frozen=True)
class ToggleRowSection(Section):
    """Multiple boolean checkboxes rendered on a single horizontal line.

    Replaces a run of individual :class:`ToggleSection` items when they share
    a line (e.g. Pile-up / DNL / Reverse in the corrections panel).
    Each entry in ``items`` is a mapping with keys ``target``, ``attr``, ``label``.
    """

    items: typing.Tuple[typing.Mapping[str, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class CustomSection(Section):
    """A bespoke, hand-written widget referenced by string ``key``.

    This is the escape hatch that keeps unlimited customization possible while
    preserving the boundary: the model only emits a key and a target attribute;
    the concrete widget is registered on the GUI side under that key. The
    optional ``options`` dict is passed through to the widget unchanged.
    """

    #: Registry key resolved on the GUI side. Required for custom sections.
    key: str = ""
    #: Free-form options forwarded to the custom widget constructor.
    options: typing.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class PlotSection(Section):
    """An inline plot embedded in the editor, declared entirely in the spec.

    The model exposes a method named ``source`` that returns the data to plot
    as a list of *series* mappings, each with keys ``x``, ``y`` and optionally
    ``name``, ``color``, ``width`` and ``style`` (``"solid"``/``"dash"``/
    ``"dot"``). The renderer calls ``getattr(model, source)()`` and can be asked
    to re-read it via ``AutoForm.refresh_plots()`` when the model changes.
    """

    #: Name of the model method returning the list of series mappings.
    source: str = ""
    x_label: str = ""
    y_label: str = ""
    #: Maximum plot height in pixels (0 = unconstrained).
    height: int = 0
    log_x: bool = False
    log_y: bool = False
    legend: bool = True


@dataclasses.dataclass(frozen=True)
class DockAreaSection(Section):
    """A rearrangeable/floatable dock area hosting child sections as panels.

    Each child section (typically a :class:`PlotSection` or a panel) becomes a
    drag-and-drop dock tab in a ChiSurf ``DockArea`` — the same mechanism the fit
    windows use for their plots, but declared entirely in the ``.view.json``.
    """

    #: Child sections, each rendered as one dock panel.
    sections: typing.Tuple["Section", ...] = ()
    #: Optional minimum height (pixels) for the dock area (0 = unconstrained).
    height: int = 0


@dataclasses.dataclass(frozen=True)
class PlotSpec:
    """A plot referenced by string ``key`` plus toolkit-agnostic ``options``.

    Replaces direct references to plot classes (e.g. ``cs.gui.plots.LinePlot``)
    that would otherwise leak the GUI into the model. The GUI registry maps
    ``key`` -> plot class.
    """

    #: Registry key, e.g. ``"line"``, ``"residual"``, ``"distribution"``.
    key: str
    #: Options forwarded to the plot, e.g. axis scales and labels.
    options: typing.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ModelView:
    """Complete, UI-agnostic description of an editor.

    Returned by :meth:`chisurf.core.models.model.Model.view_spec`. Contains the
    ordered list of editor sections and the ordered list of plots.
    """

    sections: typing.Tuple[Section, ...] = ()
    plots: typing.Tuple[PlotSpec, ...] = ()

    def flat_sections(self) -> typing.List["Section"]:
        """Return all sections in depth-first order, recursing into panels.

        Allows callers to find any section type regardless of nesting depth.
        """
        result: typing.List[Section] = []
        def _walk(secs):
            for s in secs:
                result.append(s)
                if isinstance(s, PanelSection):
                    _walk(s.sections)
        _walk(self.sections)
        return result

    def section_targets(self) -> typing.List[str]:
        """Return the resolved target attribute names of all sections (including nested).

        Convenience accessor for headless tests and introspection.
        """
        return [s.target for s in self.flat_sections() if s.target is not None]


#: Maps the ``"type"`` field of a JSON section to its dataclass.
_SECTION_TYPES = {
    "panel": PanelSection,
    "parameter_group": ParameterGroupSection,
    "dynamic_group": DynamicGroupSection,
    "curve_input": CurveInputSection,
    "choice": ChoiceSection,
    "toggle": ToggleSection,
    "toggle_row": ToggleRowSection,
    "value": ValueSection,
    "custom": CustomSection,
    "plot": PlotSection,
    "dock_area": DockAreaSection,
}


def _section_from_dict(d: typing.Mapping[str, typing.Any]) -> Section:
    """Build a :class:`Section` from a JSON dict, validating its ``type``."""
    kind = d.get("type", "parameter_group")
    cls = _SECTION_TYPES.get(kind)
    if cls is None:
        raise ValueError(
            f"unknown section type {kind!r}; expected one of {sorted(_SECTION_TYPES)}"
        )
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in d.items() if k in fields}
    # tuples for frozen/hashable dataclasses
    if "header_keys" in kwargs and kwargs["header_keys"] is not None:
        kwargs["header_keys"] = tuple(kwargs["header_keys"])
    if "options" in kwargs and kwargs["options"] is not None:
        kwargs["options"] = tuple(kwargs["options"])
    if "labels" in kwargs and kwargs["labels"] is not None:
        kwargs["labels"] = tuple(kwargs["labels"])
    # panels nest child sections — parse them recursively
    if "sections" in kwargs and kwargs["sections"] is not None:
        kwargs["sections"] = tuple(_section_from_dict(s) for s in kwargs["sections"])
    # ToggleRowSection items must be a tuple of plain dicts
    if "items" in kwargs and kwargs["items"] is not None:
        kwargs["items"] = tuple(dict(item) for item in kwargs["items"])
    return cls(**kwargs)


@dataclasses.dataclass
class ParameterGroupView:
    """Adapter that makes a ``FittingParameterGroup`` renderable with no JSON.

    PRD-40 calls for "a ``FittingParameterGroup`` *is* a ``DataSet`` view
    automatically". This is that adapter: wrap any parameter group and the
    result exposes a :meth:`view_spec` describing a single
    :class:`ParameterGroupSection`, so ``AutoForm(ParameterGroupView(group))``
    renders the group's parameters (with their bounds/fixed/link metadata)
    without authoring a ``.view.json``.

    The group is duck-typed (only ``parameters_all``/``name`` are read), so this
    stays free of any GUI or fitting-internals import and the module remains
    Qt-free.

    Parameters
    ----------
    group : object
        A ``FittingParameterGroup`` (or anything the renderer's
        ``ParameterGroupSection`` path accepts).
    title : str or None
        Header text; defaults to the group's ``name``.
    n_col : int or None
        Grid column count, or ``None`` for the configured default.
    collapsible, collapsed : bool
        Foldable-section behaviour, mirroring :class:`ParameterGroupSection`.
    """

    group: typing.Any
    title: typing.Optional[str] = None
    n_col: typing.Optional[int] = None
    collapsible: bool = True
    collapsed: bool = False

    def view_spec(self) -> ModelView:
        """Return a one-section :class:`ModelView` over the wrapped group."""
        title = self.title if self.title is not None else getattr(self.group, "name", "")
        return ModelView(
            sections=(
                ParameterGroupSection(
                    target="group",
                    title=title,
                    n_col=self.n_col,
                    collapsible=self.collapsible,
                    collapsed=self.collapsed,
                ),
            )
        )


def load_view_spec(data: typing.Union[str, pathlib.Path, typing.Mapping]) -> ModelView:
    """Build a :class:`ModelView` from JSON data, a file path, or a dict.

    This is the loader behind the user-editable ``<model>.view.json`` files that
    accompany each computational model. It stays free of any GUI toolkit: the
    result is the same plain-data :class:`ModelView` the GUI renderer consumes.

    Parameters
    ----------
    data : str or pathlib.Path or mapping
        A path to a ``.view.json`` file, or an already-parsed mapping.

    Returns
    -------
    ModelView
        The parsed, validated editor description.
    """
    if isinstance(data, (str, pathlib.Path)):
        with open(data, encoding="utf-8") as fh:
            data = json.load(fh)
    if not isinstance(data, typing.Mapping):
        raise TypeError(f"view spec must be a mapping, got {type(data).__name__}")

    sections = tuple(_section_from_dict(s) for s in data.get("sections", ()))
    plots = tuple(
        PlotSpec(key=p["key"], options=dict(p.get("options", {})))
        for p in data.get("plots", ())
    )
    return ModelView(sections=sections, plots=plots)


@dataclasses.dataclass(frozen=True)
class FittingParameterSection(Section):
    """A single FittingParameter control: value spinbox + fix/link/bounds row.

    The AutoForm renderer resolves ``target`` to a ``FittingParameter`` attribute
    on the model, then builds a ``FittingParameterWidget`` for it.
    Per-parameter display options mirror ``FittingParameterWidget`` kwargs.
    """

    #: Display label; falls back to the parameter's own name when empty.
    label: str = ""
    #: Hide the name label (useful when the name is shown in a surrounding grid).
    hide_label: bool = False
    #: Hide the error estimate field.
    hide_error: bool = False
    #: Disable bounds controls entirely.
    hide_bounds: bool = False
    #: Hide the link checkbox.
    hide_link: bool = False
    #: Whether the fix checkbox is shown.
    fixable: bool = True
    #: Significant digits shown in the value spinbox.
    decimals: int = 4
    #: Unit suffix appended to the displayed value.
    suffix: str = ""


__all__ = [
    "Section",
    "ParameterGroupSection",
    "DynamicGroupSection",
    "CurveInputSection",
    "PanelSection",
    "ChoiceSection",
    "ToggleSection",
    "ValueSection",
    "CustomSection",
    "FittingParameterSection",
    "PlotSpec",
    "ModelView",
    "ParameterGroupView",
    "load_view_spec",
]
