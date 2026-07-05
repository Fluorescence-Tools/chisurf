"""Headless tests for the RPC-method → AutoForm bridge (no Qt required).

Plugins declare their RPC interface in ``manifest.json`` (``rpc_methods``);
:class:`chisurf.core.dataspec.RpcMethodView` turns a method's JSON-Schema
``params_schema`` into a renderable view. These tests assert the pure-data
half of the contract: the manifest's new method-level ``description`` field,
the schema→section mapping, description→tooltip plumbing (via
``Section.description``), live value binding, and the ``params()`` round-trip.
"""

from __future__ import annotations

from chisurf.core import dataspec as vs
from chisurf.core.plugin.manifest import PluginManifest, validate_manifest


def _method():
    return {
        "name": "pch.compute",
        "summary": "Compute a PCH histogram.",
        "description": "Compute a photon counting histogram from a loaded TTTR file.",
        "params_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Path of the TTTR file to analyse.",
                },
                "channels": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Routing channels to include.",
                },
                "bin_time_us": {
                    "type": "number",
                    "default": 10.0,
                    "minimum": 0.0,
                    "description": "Counting bin width in microseconds.",
                },
                "n_components": {"type": "integer", "default": 1},
                "normalize": {"type": "boolean", "default": True},
                "mode": {"type": "string", "enum": ["fast", "exact"], "default": "fast"},
            },
            "required": ["filename"],
        },
    }


# -- manifest schema ----------------------------------------------------------


def test_manifest_parses_and_roundtrips_method_description():
    manifest = PluginManifest.from_dict({"id": "pch", "version": "1.0", "rpc_methods": [_method()]})
    spec = manifest.rpc_methods[0]
    assert spec.summary == "Compute a PCH histogram."
    assert spec.description.startswith("Compute a photon counting histogram")
    assert manifest.to_dict()["rpc_methods"][0]["description"] == spec.description


def test_manifest_validation_rejects_non_string_description():
    data = {"id": "p", "version": "1", "rpc_methods": [{"name": "m", "description": 3}]}
    assert any("description" in e for e in validate_manifest(data))
    data["rpc_methods"][0]["description"] = "ok"
    assert validate_manifest(data) == []


# -- schema → sections --------------------------------------------------------


def test_panel_carries_method_description_as_tooltip_text():
    view = vs.RpcMethodView(_method()).view_spec()
    panel = view.sections[0]
    assert isinstance(panel, vs.PanelSection)
    assert panel.title == "pch.compute"
    # AutoForm maps Section.description to the widget tooltip.
    assert panel.description.startswith("Compute a photon counting histogram")


def test_summary_is_the_tooltip_fallback_when_description_missing():
    method = _method()
    method.pop("description")
    panel = vs.RpcMethodView(method).view_spec().sections[0]
    assert panel.description == "Compute a PCH histogram."


def test_property_descriptions_become_field_descriptions():
    panel = vs.RpcMethodView(_method()).view_spec().sections[0]
    by_attr = {s.attr: s for s in panel.sections}
    # per-parameter JSON-Schema description → Section.description (→ tooltip);
    # required parameters are flagged in the help text.
    assert by_attr["filename"].description == "Path of the TTTR file to analyse. (required)"
    assert by_attr["bin_time_us"].description == "Counting bin width in microseconds."


def test_types_map_to_typed_sections():
    panel = vs.RpcMethodView(_method()).view_spec().sections[0]
    by_attr = {s.attr: s for s in panel.sections}
    assert isinstance(by_attr["normalize"], vs.ToggleSection)
    assert isinstance(by_attr["mode"], vs.ChoiceSection)
    assert by_attr["mode"].options == ("fast", "exact")
    assert by_attr["n_components"].kind == "int"
    assert by_attr["bin_time_us"].kind == "float"
    assert by_attr["bin_time_us"].minimum == 0.0
    assert by_attr["channels"].kind == "str"  # arrays edit as JSON text
    assert by_attr["filename"].kind == "str"


# -- binding & params() -------------------------------------------------------


def test_edits_bind_like_autoform_and_params_round_trips():
    fired = []
    view = vs.RpcMethodView(_method(), on_change=lambda: fired.append(1))
    panel = view.view_spec().sections[0]
    field = next(s for s in panel.sections if s.attr == "filename")
    # AutoForm writes via getattr(model, target) then setattr(group, attr, v).
    group = getattr(view, field.target)
    setattr(group, "filename", "/data/a.ptu")
    setattr(group, "channels", "[0, 2]")
    assert fired == [1, 1]

    params = view.params()
    assert params["filename"] == "/data/a.ptu"
    assert params["channels"] == [0, 2]  # JSON text parsed back to a list
    assert params["bin_time_us"] == 10.0  # schema default
    assert params["normalize"] is True
    assert params["mode"] == "fast"


def test_empty_optionals_are_omitted_but_defaults_kept():
    method = _method()
    view = vs.RpcMethodView(method)
    params = view.params()
    assert "channels" not in params  # optional, left empty
    assert "filename" in view._values  # required stays bound even if empty


def test_untouched_optional_numbers_are_not_sent_as_zero():
    # An untouched spin box holds a 0 placeholder — that must not reach the
    # callee (e.g. micro_time_max=0 would gate away every photon).
    method = {
        "name": "m",
        "params_schema": {
            "type": "object",
            "properties": {"micro_time_max": {"type": "integer"}},
        },
    }
    view = vs.RpcMethodView(method)
    assert view.params() == {}
    view._params_group.micro_time_max = 0  # an explicit user edit of 0 counts
    assert view.params() == {"micro_time_max": 0}


def test_initial_values_override_defaults_and_serialize_json_props():
    view = vs.RpcMethodView(_method(), values={"channels": [1, 3], "bin_time_us": 5.0})
    assert view._values["channels"] == "[1, 3]"  # held as text for the editor
    params = view.params()
    assert params["channels"] == [1, 3]
    assert params["bin_time_us"] == 5.0


def test_non_string_enum_values_round_trip():
    method = {
        "name": "m",
        "params_schema": {
            "type": "object",
            "properties": {"order": {"type": "integer", "enum": [1, 2, 4], "default": 2}},
        },
    }
    view = vs.RpcMethodView(method)
    group = view._params_group
    setattr(group, "order", "4")  # a choice widget commits the string label
    assert view.params()["order"] == 4
