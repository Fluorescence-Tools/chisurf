"""Render a plugin RPC method's ``params_schema`` as an AutoForm editor.

Plugins declare their RPC interface in ``manifest.json`` (``rpc_methods``, see
:class:`chisurf.core.plugin.manifest.RPCMethodSpec`). Each method carries a
JSON-Schema ``params_schema``; :class:`RpcMethodView` is the Qt-free adapter
that turns that schema into a :class:`~chisurf.core.dataspec.ModelView` so
``AutoForm(RpcMethodView(method))`` renders a parameter form for the call.

Help text flows into Qt tooltips through the existing ``Section.description``
channel: the method's ``description`` (falling back to ``summary``) becomes the
panel tooltip, and each property's standard JSON-Schema ``description`` becomes
the tooltip of its field. Like :class:`~chisurf.core.dataspec.SettingsView`,
this module is pure data — headless tests can assert the whole editor without
Qt installed.
"""

from __future__ import annotations

import json

from chisurf import typing

from . import (
    ChoiceSection,
    ModelView,
    PanelSection,
    Section,
    ToggleSection,
    ValueSection,
    _SettingsGroup,
)

#: JSON-Schema types edited as JSON text (round-tripped through ``params()``).
_JSON_KINDS = ("array", "object")


def _method_field(method: typing.Any, key: str, default: typing.Any = None) -> typing.Any:
    """Read *key* from an ``RPCMethodSpec``-like object or a plain mapping."""
    if isinstance(method, typing.Mapping):
        return method.get(key, default)
    return getattr(method, key, default)


class _ParamsGroup(_SettingsGroup):
    """A :class:`_SettingsGroup` that records which parameters were edited.

    ``params()`` must not send optional parameters the user never touched —
    the zero placeholder an empty spin box holds is not a real value (e.g. an
    untouched ``micro_time_max: 0`` would gate away every photon).
    """

    def __init__(self, data, on_change, touched: typing.Set[str]):
        super().__init__(data, on_change)
        object.__setattr__(self, "_touched", touched)

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_touched").add(name)
        super().__setattr__(name, value)


class RpcMethodView:
    """Bind an RPC method's parameters so AutoForm can render and edit them.

    Parameters
    ----------
    method : RPCMethodSpec or mapping
        The RPC method declaration (an ``rpc_methods`` manifest entry). Only
        ``name``, ``summary``, ``description`` and ``params_schema`` are read,
        so both the parsed dataclass and the raw JSON dict work.
    values : mapping, optional
        Initial parameter values, overriding the schema ``default``s.
    on_change : callable, optional
        Invoked with no arguments after any parameter edit.
    title : str, optional
        Panel title; defaults to the method name.
    """

    def __init__(
        self,
        method: typing.Any,
        *,
        values: typing.Optional[typing.Mapping[str, typing.Any]] = None,
        on_change: typing.Optional[typing.Callable] = None,
        title: typing.Optional[str] = None,
    ):
        self.method_name = str(_method_field(method, "name", "") or "")
        self.summary = str(_method_field(method, "summary", "") or "")
        self.description = str(_method_field(method, "description", "") or "")
        schema = _method_field(method, "params_schema") or {}

        properties: typing.Mapping[str, typing.Any] = schema.get("properties", {}) or {}
        self._required = tuple(schema.get("required", ()) or ())
        #: Properties whose declared type is array/object — edited as JSON text.
        self._json_props: typing.Set[str] = set()
        #: Original enum values per property (choice widgets commit strings).
        self._enums: typing.Dict[str, typing.Tuple[typing.Any, ...]] = {}

        self._values: typing.Dict[str, typing.Any] = {}
        #: Parameters carrying a real value from the start (schema default or
        #: caller-supplied), as opposed to a mere type placeholder.
        self._explicit: typing.Set[str] = set()
        #: Parameters the user edited through the form.
        self._touched: typing.Set[str] = set()
        sections: typing.List[Section] = []
        for prop_name, prop in properties.items():
            if not isinstance(prop, typing.Mapping):
                prop = {}
            sections.append(self._field_section(prop_name, prop))
            self._values[prop_name] = self._initial_value(prop_name, prop, values)
            if "default" in prop or (values is not None and prop_name in values):
                self._explicit.add(prop_name)

        # AutoForm resolves each section's ``target`` via ``getattr(model, target)``;
        # the sections all point at this one live binding group.
        self._params_group = _ParamsGroup(self._values, on_change, self._touched)
        panel = PanelSection(
            title=title if title is not None else self.method_name,
            description=self.description or self.summary,
            n_col=1,
            sections=tuple(sections),
        )
        self._view = ModelView(sections=(panel,))

    def view_spec(self) -> ModelView:
        """Return the derived editor description (one panel per RPC method)."""
        return self._view

    # -- values ---------------------------------------------------------------
    def params(self) -> typing.Dict[str, typing.Any]:
        """Return the current parameter values, ready to pass to the RPC call.

        Array/object parameters edited as JSON text are parsed back. Optional
        parameters are only included when they carry a real value — a schema
        default, a caller-supplied initial value, or a user edit — so untouched
        placeholders never reach the callee and its own defaults apply.
        """
        out: typing.Dict[str, typing.Any] = {}
        for key, value in self._values.items():
            optional = key not in self._required
            if optional and key not in self._explicit and key not in self._touched:
                continue
            if key in self._enums and isinstance(value, str):
                value = next((orig for orig in self._enums[key] if str(orig) == value), value)
            if key in self._json_props and isinstance(value, str):
                text = value.strip()
                if not text:
                    value = None
                else:
                    try:
                        value = json.loads(text)
                    except json.JSONDecodeError:
                        pass  # hand the raw text to the caller's validation
            if value in (None, "") and optional:
                continue
            out[key] = value
        return out

    # -- builder ----------------------------------------------------------------
    def _initial_value(self, name, prop, overrides):
        if overrides is not None and name in overrides:
            value = overrides[name]
        elif "default" in prop:
            value = prop["default"]
        else:
            value = None
        if value is not None:
            # JSON-edited (array/object) params are held as text in the field.
            if name in self._json_props and not isinstance(value, str):
                return json.dumps(value)
            return value
        kind = prop.get("type")
        if kind == "boolean":
            return False
        if kind == "integer":
            return 0
        if kind == "number":
            return 0.0
        return ""

    def _field_section(self, name: str, prop: typing.Mapping[str, typing.Any]) -> Section:
        label = str(prop.get("title", name))
        description = str(prop.get("description", "") or "")
        if name in self._required and description:
            description += " (required)"
        elif name in self._required:
            description = "(required)"
        common = dict(target="_params_group", attr=name, description=description)

        if "enum" in prop:
            self._enums[name] = tuple(prop["enum"])
            options = tuple(str(v) for v in prop["enum"])
            return ChoiceSection(label=label, options=options, **common)

        kind = prop.get("type")
        if kind == "boolean":
            return ToggleSection(label=label, **common)
        if kind == "integer":
            return ValueSection(
                label=label,
                kind="int",
                minimum=prop.get("minimum"),
                maximum=prop.get("maximum"),
                **common,
            )
        if kind == "number":
            return ValueSection(
                label=label,
                kind="float",
                minimum=prop.get("minimum"),
                maximum=prop.get("maximum"),
                decimals=6,
                **common,
            )
        if kind in _JSON_KINDS:
            self._json_props.add(name)
            return ValueSection(label=label, kind="str", placeholder="JSON", **common)
        # strings and untyped properties: a plain line edit; ``format`` may ask
        # for the file picker (line edit + browse button).
        value_kind = "file" if prop.get("format") in ("file", "path", "uri") else "str"
        return ValueSection(label=label, kind=value_kind, **common)
