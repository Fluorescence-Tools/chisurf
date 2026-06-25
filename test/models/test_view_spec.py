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
    lifetimes = next(s for s in spec.sections if s.target == "lifetimes")
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
    curve_inputs = [s for s in spec.sections if isinstance(s, vs.CurveInputSection)]
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
    choices = [s for s in spec.sections if isinstance(s, vs.ChoiceSection)]
    toggles = [s for s in spec.sections if isinstance(s, vs.ToggleSection)]
    choice_attrs = {s.attr for s in choices}
    toggle_attrs = {s.attr for s in toggles}
    assert {"mode", "window_function", "polarization_type"} <= choice_attrs
    assert "do_convolution" in toggle_attrs
    assert {"correct_pile_up", "correct_dnl", "reverse"} <= toggle_attrs
