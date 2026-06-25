# PRD-40: A chisurf-native declarative dataset → editor framework (guidata-like)

## Goal

One reusable way to **declare a structured, typed dataset once and get its editor
generated automatically** — across models, experiment setups, settings, tool
panels — instead of the ~5 ad-hoc "type → widget" mappers that exist today.
Inspired by [guidata](https://guidata.readthedocs.io) (`DataSet` of typed
`Item`s → auto Qt form), but **chisurf-native** so it understands the domain
types guidata cannot: `FittingParameter` (bounds/fixed/link/error), `DataCurve`
selectors, and chinet node binding.

## Evidence (why) — the same problem is solved 5 times

- `gui/widgets/settings_editor.py` dispatches on `type(value)`
  (`bool→QCheckBox`, `int→QSpinBox`, `float→QDoubleSpinBox`, `str→QLineEdit`,
  `list→QLineEdit`, path→file picker). A hand-rolled item→widget mapper.
- `gui/widgets/metadata_editor.py` — bespoke key/value combo editor.
- `gui/widgets/parameter_editor/`, `gui/widgets/node_editor/` — more of the same.
- `core/fitting/parameter.py` `FittingParameter`/`FittingParameterGroup` +
  `parameter_widgets.py` — a typed numeric item (bounds/fixed/link) with its own
  hand-written widget; `parameter_registry.json` already externalises the
  metadata (label/bounds/units), proving the declaration wants to be *data*.
- **PRD-38** `core/models/view_spec.py` + `gui/.../auto_model_widget.py` — the
  model-specific instance: a JSON-authored tree of section/item descriptors
  (`parameter_group`, `choice`, `toggle`, `curve_input`, `panel`) rendered by a
  generic `AutoModelWidget` with a string-keyed registry escape hatch.

PRD-38 already built ~70% of the general machinery — it is just scoped to models.
This PRD lifts that machinery out from under `models/` so everything can use it.

## Design

Three layers, one-directional `gui → core` (the PRD-38 boundary, generalised):

1. **`core/dataspec/` (pure data, no Qt)** — the vocabulary.
   - `Item` subtypes: `FloatItem, IntItem, StringItem, BoolItem, ChoiceItem,
     FileItem, CurveItem, GroupItem, ListItem`, plus `CustomItem(key)` (escape
     hatch). Each carries label, help, default, constraints, and a **binding**
     (the attribute/accessor on the bound object) — never a widget reference.
   - `DataSet` — an ordered, possibly nested tree of items (== today's `ModelView`
     generalised; `PanelSection`→`GroupItem`, `DynamicGroupSection`→`ListItem`,
     `ChoiceSection`→`ChoiceItem`, `ToggleSection`→`BoolItem`,
     `CurveInputSection`→`CurveItem`, `ParameterGroupSection`→a `GroupItem` of
     `FloatItem`s sourced from the `FittingParameterGroup`).
   - `load_dataspec(json|dict)` — the authoring surface (today's `load_view_spec`).
   - An adapter so a `FittingParameterGroup` *is* a `DataSet` view automatically
     (its parameters → `FloatItem`s with bounds/fixed/link), so existing models
     need no re-authoring.

2. **`gui/autoform/` (renderer)** — `AutoForm(dataset, bound_object)` builds the
   Qt editor by composition (today's `AutoModelWidget`, renamed/generalised). A
   registry maps item-type/`CustomItem` keys → concrete widgets (today's
   `sections/registry.py` + `builtin.py`, lifted).

3. **Consumers** — `Model.view_spec()` returns a `DataSet`; the settings editor,
   metadata editor and experiment-setup panels are migrated onto `AutoForm` one
   at a time, deleting their bespoke type→widget code.

### Boundary rule (enforced, generalised from PRD-38)

`core/dataspec/**` must never import a GUI toolkit. AST CI test, as for
`core/models/**`.

## API (sketch)

```python
# core/dataspec
FloatItem("dt", bind="convolve.dt", min=0.0, unit="ns")
ChoiceItem("Type", bind="convolve.mode", choices=["per","exp","full"], style="radio")
BoolItem("Convolve", bind="convolve.do_convolution")
CurveItem("IRF", select_action="model.change_irf", unload_action="model.unload_irf")
ListItem("Lifetimes", bind="lifetimes", item=GroupItem([...]), add="append", remove="pop")
GroupItem("Convolution", items=[...], collapsible=True, collapsed=False,
          collapsed_when={"bind": "anisotropy.polarization_type", "equals": "vm"})
DataSet(items=[...])

# gui/autoform
form = AutoForm(dataset, bound_object=model)   # QWidget
autoform.register_item("curve", CurveItemWidget)
```

## Tasks (incremental — each independently shippable)

1. Land PRD-38 fully for the Lifetime model (panels, foldable, anisotropy list,
   compact controls). This is the working prototype the framework is extracted
   from. — *in progress*
2. Lift `core/models/view_spec.py` → `core/dataspec/` (rename sections→items,
   keep a thin `view_spec` shim re-exporting for models). Boundary test moves
   with it. No behaviour change.
3. Lift `gui/widgets/models/auto_model_widget.py` + `sections/` →
   `gui/autoform/`. `AutoModelWidget` becomes `AutoForm`; `build_model_editor`
   delegates.
4. `FittingParameterGroup`→`DataSet` adapter so any param group renders via
   `AutoForm` with no JSON.
5. Migrate **one** non-model consumer (settings editor is the cleanest) onto
   `AutoForm`; delete its `type(value)` ladder. Proves generality.
6. Migrate metadata editor + experiment-setup panels; delete bespoke editors.

## Definition of Done

- A single `AutoForm(dataset, obj)` renders models, settings and metadata; the
  per-domain type→widget code is deleted.
- A new typed field anywhere = add one `Item` to a declaration (or JSON), no new
  widget code, unless it is genuinely bespoke (then one `CustomItem` + registry
  entry).
- `core/dataspec/**` is Qt-free (AST test green). Headless dataspec tests +
  offscreen `AutoForm` tests (the PRD-38 `/test-model-editor` path, generalised).

## Build vs. adopt

Adopting guidata itself was considered and rejected: it cannot represent
`FittingParameter` semantics (bounds/fixed/link/error/global-link), `DataCurve`
selection, or chinet-node binding, which are the whole point here. A native,
~small framework (PRD-38 is most of it already) gives that integration and keeps
the data declarations authorable as the JSON files users already edit.

## Relationship

Supersedes the model-only scope of **PRD-38** by generalising its machinery;
complements **PRD-23** (thin view-only widgets) and **PRD-26** (declarative
generation). PRD-38 continues to completion as the reference implementation;
this PRD is the extraction that follows.
