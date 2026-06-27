"""Headless tests for the dict-backed ``SettingsView`` adapter (no Qt required).

``SettingsView`` lets ``AutoForm`` render an arbitrary settings dict: nested
dicts become collapsible panels, scalars become typed fields, and everything is
bound back to the *live* dict so edits round-trip in place. These tests assert
the pure-data half of that contract — structure derivation, attribute binding,
in-place mutation, the change callback, and preservation of non-scalar values.
"""

from __future__ import annotations

from chisurf.core import dataspec as vs


def _settings():
    def accessor(x):  # a callable value — must survive untouched
        return x

    return {
        "Lifetime": {
            "attribute": "lifetime_spectrum",
            "accessor": accessor,
            "curve_options": {"stepMode": False, "symbol": "o", "fillLevel": 0.0},
        },
        "threshold": 5,
        "scale": 1.25,
        "enabled": True,
        "name": "default",
    }


def test_structure_maps_dicts_to_panels_and_scalars_to_typed_fields():
    view = vs.SettingsView(_settings()).view_spec()
    # one nested dict -> a collapsible panel, four scalars -> typed fields
    panels = [s for s in view.sections if isinstance(s, vs.PanelSection)]
    assert len(panels) == 1
    assert panels[0].title == "Lifetime"
    assert panels[0].collapsible is True

    kinds = {
        s.attr: (type(s).__name__, getattr(s, "kind", None))
        for s in view.sections
        if not isinstance(s, vs.PanelSection)
    }
    assert kinds["threshold"] == ("ValueSection", "int")
    assert kinds["scale"] == ("ValueSection", "float")
    assert kinds["enabled"][0] == "ToggleSection"
    assert kinds["name"] == ("ValueSection", "str")


def test_bool_is_a_toggle_not_an_int_field():
    # bool subclasses int — it must be detected first.
    view = vs.SettingsView({"flag": True}).view_spec()
    assert isinstance(view.sections[0], vs.ToggleSection)


def test_callable_value_is_read_only_and_untouched():
    original = lambda x: x  # noqa: E731 - a callable value to round-trip
    data = {"Lifetime": {"accessor": original, "label": "ok"}}
    view = vs.SettingsView(data).view_spec()
    panel = next(s for s in view.sections if isinstance(s, vs.PanelSection))
    accessor_field = next(s for s in panel.sections if s.attr == "accessor")
    # non-scalar values render read-only and are never bound/written.
    assert accessor_field.read_only is True
    assert data["Lifetime"]["accessor"] is original


def test_groups_bind_and_mutate_the_live_dict_with_callback():
    data = _settings()
    fired = []
    view = vs.SettingsView(data, on_change=lambda: fired.append(1))

    # AutoForm resolves a section's target via getattr(model, target), then
    # writes via setattr(group, attr, value). Replicate that here.
    sec = next(
        s for s in view.view_spec().sections
        if not isinstance(s, vs.PanelSection) and s.attr == "threshold"
    )
    group = getattr(view, sec.target)
    setattr(group, sec.attr, 42)

    assert data["threshold"] == 42          # mutated in place
    assert fired == [1]                     # callback fired once
    assert getattr(group, "threshold") == 42  # readable back through the group


def test_nested_panel_edits_reach_the_inner_dict():
    data = _settings()
    view = vs.SettingsView(data)
    panel = next(s for s in view.view_spec().sections if isinstance(s, vs.PanelSection))
    inner = next(s for s in panel.sections if isinstance(s, vs.PanelSection))  # curve_options
    field = next(s for s in inner.sections if s.attr == "fillLevel")
    group = getattr(view, field.target)
    setattr(group, "fillLevel", 3.5)
    assert data["Lifetime"]["curve_options"]["fillLevel"] == 3.5


def test_unknown_attribute_raises_so_fit_resolves_to_none():
    # AutoForm does getattr(model, "fit", None); SettingsView must not invent it.
    view = vs.SettingsView(_settings())
    assert getattr(view, "fit", None) is None


def test_empty_dict_yields_empty_view():
    view = vs.SettingsView({}).view_spec()
    assert view.sections == ()
