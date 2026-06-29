"""Headless tests for the model view-spec layer (no Qt required)."""
from __future__ import annotations

import numpy as np

from chisurf.core.models import view_spec as vs


def test_view_spec_vocabulary_is_pure_data():
    """The view-spec dataclasses are plain, comparable, hashable data."""
    view = vs.ModelView(
        sections=(
            vs.ParameterGroupSection(target="generic", title="Generic"),
            vs.DynamicGroupSection(target="lifetimes", title="Lifetimes", row_width=2),
            vs.CustomSection(key="my_panel", target="anisotropy"),
        ),
        plots=(vs.PlotSpec("line", {"y_label": "counts"}),),
    )
    assert view.section_targets() == ["generic", "lifetimes", "anisotropy"]
    # frozen dataclasses are hashable / equality-comparable
    assert vs.PlotSpec("line") == vs.PlotSpec("line")


def _make_lifetime_model():
    """Build a LifetimeModel against a tiny in-memory fit, or skip."""
    import pytest
    try:
        import chisurf.core.fitting.fit as fit_mod
        from chisurf.core.data import DataCurve
        from chisurf.core.models.tcspc.lifetime import LifetimeModel
    except Exception as exc:  # pragma: no cover - import guard
        pytest.skip(f"lifetime model import failed: {exc}")

    x = np.linspace(0, 25, 256)
    y = np.ones_like(x)
    try:
        data = DataCurve(x=x, y=y)
        fit = fit_mod.Fit(model_class=LifetimeModel, data=data)
        model = fit.model
    except Exception as exc:  # pragma: no cover - construction guard
        pytest.skip(f"lifetime model construction failed: {exc}")
    return model


def test_lifetime_model_view_spec_structure():
    """LifetimeModel exposes its editor as data with the expected sections."""
    model = _make_lifetime_model()
    spec = model.view_spec()

    assert isinstance(spec, vs.ModelView)
    # nuisances + dynamic lifetimes + anisotropy
    targets = spec.section_targets()
    for expected in ("convolve", "generic", "corrections", "lifetimes", "anisotropy"):
        assert expected in targets, f"missing section target {expected!r}"

    # the lifetimes section is dynamic with paired (amplitude, lifetime) rows
    lifetimes = next(s for s in spec.flat_sections() if s.target == "lifetimes")
    assert isinstance(lifetimes, vs.DynamicGroupSection)
    assert lifetimes.row_width == 2
    assert "lifetime_amplitude_options" in lifetimes.header_keys

    # plot keys are strings, never GUI classes
    plot_keys = [p.key for p in spec.plots]
    assert "line" in plot_keys and "residual" in plot_keys
    for p in spec.plots:
        assert isinstance(p.key, str)

    # every section target resolves to a real attribute on the model
    for target in targets:
        assert hasattr(model, target), f"unresolved target {target!r}"


def test_curve_input_section_loads_from_json():
    """The curve_input section type round-trips through the JSON loader as pure
    data (no Qt), carrying the action names and payload keys (PRD-38)."""
    spec = vs.load_view_spec({
        "sections": [
            {"type": "curve_input", "target": "convolve", "label": "IRF",
             "select_action": "model.change_irf", "unload_action": "model.unload_irf",
             "index_key": "irf_idx", "name_key": "irf_name", "name_attr": "irf"},
        ],
        "plots": [],
    })
    sec = spec.sections[0]
    assert isinstance(sec, vs.CurveInputSection)
    assert sec.target == "convolve" and sec.label == "IRF"
    assert sec.select_action == "model.change_irf"
    assert sec.unload_action == "model.unload_irf"
    assert sec.index_key == "irf_idx" and sec.name_key == "irf_name"
    assert sec.name_attr == "irf"


def test_lifetime_view_has_irf_curve_input():
    """The Lifetime editor declares an IRF curve input so the model can be
    given an instrument response and actually compute a convolved fit."""
    model = _make_lifetime_model()
    spec = model.view_spec()
    curve_inputs = [s for s in spec.flat_sections() if isinstance(s, vs.CurveInputSection)]
    irf = next((s for s in curve_inputs if s.select_action == "model.change_irf"), None)
    assert irf is not None, "Lifetime view spec must expose an IRF curve input"
    assert irf.target == "convolve"


def test_choice_and_toggle_sections_load_from_json():
    """choice/toggle section types round-trip as pure data with their binding
    fields (attr- or action-bound)."""
    spec = vs.load_view_spec({
        "sections": [
            {"type": "choice", "target": "convolve", "attr": "mode", "label": "Type",
             "options": ["per", "exp", "full"]},
            {"type": "toggle", "target": "convolve", "attr": "do_convolution", "label": "Convolve"},
            {"type": "choice", "target": "corrections", "attr": "window_function",
             "label": "Smoothing", "options_source": "window_function_types"},
        ],
        "plots": [],
    })
    choice, toggle, smoothing = spec.sections
    assert isinstance(choice, vs.ChoiceSection)
    assert choice.attr == "mode" and choice.options == ("per", "exp", "full")
    assert isinstance(toggle, vs.ToggleSection) and toggle.attr == "do_convolution"
    assert smoothing.options_source == "window_function_types"


def test_lifetime_view_exposes_bespoke_controls():
    """The Lifetime view declares the bespoke controls the hand-written widget
    had: convolution type + on/off, smoothing, correction toggles, polarization."""
    spec = _make_lifetime_model().view_spec()
    choices = [s for s in spec.flat_sections() if isinstance(s, vs.ChoiceSection)]
    toggles = [s for s in spec.flat_sections() if isinstance(s, vs.ToggleSection)]
    choice_attrs = {s.attr for s in choices}
    toggle_attrs = {s.attr for s in toggles}
    assert {"mode", "window_function", "polarization_type"} <= choice_attrs
    assert "do_convolution" in toggle_attrs
    # Pile-up / DNL / Reverse are now in a ToggleRowSection (all on one line)
    toggle_row_attrs = {
        item["attr"]
        for s in spec.flat_sections()
        if isinstance(s, vs.ToggleRowSection)
        for item in s.items
    }
    assert {"correct_pile_up", "correct_dnl", "reverse"} <= toggle_row_attrs


def test_parameter_group_view_adapter_builds_a_section():
    """PRD-40 Task 4: a FittingParameterGroup renders via ParameterGroupView with
    no JSON — the adapter yields a one-section ModelView targeting the group."""
    from chisurf.core import dataspec as ds
    from chisurf.core.fitting.parameter import FittingParameter, FittingParameterGroup

    group = FittingParameterGroup(
        name="kinetics",
        parameters=[
            FittingParameter(name="k1", value=1.0),
            FittingParameter(name="k2", value=2.0),
        ],
    )

    view = ds.ParameterGroupView(group, collapsed=True).view_spec()
    assert isinstance(view, ds.ModelView)
    assert len(view.sections) == 1
    section = view.sections[0]
    assert isinstance(section, ds.ParameterGroupSection)
    assert section.target == "group"        # resolved by AutoForm via getattr
    assert section.title == "kinetics"       # defaults to the group's name
    assert section.collapsed is True
    # the adapter exposes the group at the resolved attribute name
    assert ds.ParameterGroupView(group).group is group


def test_value_section_loads_from_json():
    """PRD-40: the generic scalar field (int/float/str) parses from JSON."""
    spec = vs.load_view_spec({
        "sections": [
            {"type": "value", "label": "N bins", "kind": "int",
             "target": "setup", "attr": "n_bins", "minimum": 1, "maximum": 64},
            {"type": "value", "label": "Name", "kind": "str",
             "target": "setup", "attr": "name", "placeholder": "untitled"},
        ],
        "plots": [],
    })
    n_bins, name = spec.sections
    assert isinstance(n_bins, vs.ValueSection)
    assert n_bins.kind == "int" and n_bins.attr == "n_bins"
    assert (n_bins.minimum, n_bins.maximum) == (1, 64)
    assert name.kind == "str" and name.placeholder == "untitled"


def test_parameter_group_table_section_round_trips_from_json():
    """PRD-44: ParameterGroupTableSection parses from JSON with column
    subsetting and folds through flat_sections()."""
    spec = vs.load_view_spec({
        "sections": [
            {
                "type": "parameter_group_table",
                "target": "convolve",
                "collapsible": False,
                "columns": ["name", "value", "fixed", "error"],
            },
            {
                "type": "panel",
                "title": "Outer",
                "sections": [
                    {
                        "type": "parameter_group_table",
                        "target": "lifetimes",
                        "columns": ["name", "value"],
                    },
                ],
            },
        ],
        "plots": [],
    })
    assert len(spec.sections) == 2

    table = spec.sections[0]
    assert isinstance(table, vs.ParameterGroupTableSection)
    assert table.target == "convolve"
    assert table.collapsible is False
    assert table.columns == ("name", "value", "fixed", "error")

    # nested inside a panel
    panel = spec.sections[1]
    assert isinstance(panel, vs.PanelSection)
    nested = panel.sections[0]
    assert isinstance(nested, vs.ParameterGroupTableSection)
    assert nested.target == "lifetimes"
    assert nested.columns == ("name", "value")

    # flat_sections covers both top-level and nested
    flat = spec.flat_sections()
    table_ids = [id(s) for s in flat if isinstance(s, vs.ParameterGroupTableSection)]
    assert len(table_ids) == 2
    assert id(table) in table_ids
    assert id(nested) in table_ids

    # section_targets resolves correctly
    assert "convolve" in spec.section_targets()
    assert "lifetimes" in spec.section_targets()

    # empty columns means all columns
    spec2 = vs.load_view_spec({
        "sections": [
            {"type": "parameter_group_table", "target": "g", "columns": []},
        ],
    })
    assert spec2.sections[0].columns == ()


def test_parameter_group_table_section_hashable():
    """ParameterGroupTableSection is frozen and hashable like the rest."""
    s1 = vs.ParameterGroupTableSection(target="a", columns=("name", "value"))
    s2 = vs.ParameterGroupTableSection(target="a", columns=("name", "value"))
    s3 = vs.ParameterGroupTableSection(target="a", columns=("name",))
    assert s1 == s2
    assert s1 != s3
    assert hash(s1) == hash(s2)

    # works in sets
    _ = {s1, s2, s3}  # no error
    assert len({s1, s2, s3}) == 2


def _make_mixture_model():
    """Build a LifetimeMixtureNewModel against a tiny in-memory fit, or skip."""
    import pytest
    try:
        import chisurf.core.fitting.fit as fit_mod
        from chisurf.core.data import DataCurve
        from chisurf.core.models.tcspc.lifetime import LifetimeMixtureNewModel
    except Exception as exc:
        pytest.skip(f"mixture model import failed: {exc}")
    x = np.linspace(0, 25, 256)
    data = DataCurve(x=x, y=np.ones_like(x))
    try:
        fit = fit_mod.Fit(model_class=LifetimeMixtureNewModel, data=data)
        return fit.model
    except Exception as exc:
        pytest.skip(f"mixture model construction failed: {exc}")


def test_mix_model_view_spec_structure():
    """LifetimeMixtureNewModel exposes a pure-data view spec from mix_model.view.json."""
    model = _make_mixture_model()
    spec = model.view_spec()

    assert isinstance(spec, vs.ModelView)
    targets = spec.section_targets()
    for expected in ("convolve", "generic", "corrections"):
        assert expected in targets, f"missing section target {expected!r}"

    # The mixture panel declares a custom fit_mixer section
    customs = [s for s in spec.flat_sections() if isinstance(s, vs.CustomSection)]
    mixer = next((s for s in customs if s.key == "fit_mixer"), None)
    assert mixer is not None, "mix_model.view.json must have a fit_mixer custom section"

    # Plot keys are pure strings
    plot_keys = [p.key for p in spec.plots]
    assert "line" in plot_keys and "residual" in plot_keys and "distribution" in plot_keys


def test_mix_model_has_irf_curve_input():
    """The mix model editor declares an IRF curve input so convolution works."""
    model = _make_mixture_model()
    spec = model.view_spec()
    irf = next(
        (s for s in spec.flat_sections()
         if isinstance(s, vs.CurveInputSection) and s.select_action == "model.change_irf"),
        None,
    )
    assert irf is not None, "mix_model.view.json must expose an IRF curve input"


def test_mix_model_view_spec_file_exists():
    """mix_model.view.json is on disk next to lifetime.py."""
    import pathlib
    import inspect
    from chisurf.core.models.tcspc.lifetime import LifetimeMixtureNewModel

    src = inspect.getfile(LifetimeMixtureNewModel)
    json_path = pathlib.Path(src).parent / "mix_model.view.json"
    assert json_path.exists(), f"mix_model.view.json not found at {json_path}"
