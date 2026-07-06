---
type: PRD
prd: "38"
title: "PRD-38: Model/UI Split — view-spec JSON drives auto-generated model editors"
description: Splits a fitting model's compute definition from its editor by describing the editor in a co-located JSON view spec that a generic GUI renderer turns into the control panel.
status: in-progress
phase: "unassigned"
resource: chisurf/core/models/
tags: [prd, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-38 makes a fitting model's editor the automatic result of its computational definition instead of a hand-written per-model widget that duplicates the model's structure and welds Qt to the compute side. Each model stays in one place (parameters plus `update_model` in `chisurf/core/models`, Qt-free), its editor is described in a hand-editable `<model>.view.json` file, and the GUI renders that spec by composition via `AutoModelWidget`. A strict, AST-CI-enforced boundary keeps `core/models/**` from importing any GUI toolkit, while a string-keyed registry provides an escape hatch for bespoke custom sections. The view-spec vocabulary has grown to parameter groups, dynamic groups, curve inputs, choices, toggles, and custom sections plus plots.

# Status
In-progress (unassigned phase, STATUS TABLE authoritative). Data spine, boundary test, generic renderer, live wiring, and several section types are done; migrating the remaining structured-model widget family (FRET, anisotropy, FCS, PDA) and dropping `plot_classes` remain.

# Goal
Make a fitting model's **editor the automatic result of its computational definition**. Maintain each model in one place (compute + parameters in `chisurf/core/models`), describe its editor in a **user-editable JSON file that accompanies the model**, and have the GUI render that editor generically. Eliminate the hand-written, per-model widget that today duplicates the model's structure and welds Qt to the compute side.

# Evidence (why)

Today model and widget are not split — they are the **same object** via multiple inheritance, and structure is declared twice:

- `class LifetimeWidget(Lifetime, QtWidgets.QWidget)` and `class LifetimeModelWidgetBase(ModelWidget, LifetimeModel)` (`chisurf/gui/widgets/models/tcspc/lifetime.py:45,319`). The widget *is* the model; compute and Qt live in one class.
- `Lifetime.append/pop/__init__` (`core/models/tcspc/lifetime.py:166,206`) define component structure; `LifetimeWidget.append/pop/update` (`gui/.../lifetime.py:287,311,48`) re-implement the *same* structure just to spawn/destroy a widget per parameter and copy values back with `for w, v in zip(...)`.
- `plot_classes` references GUI plot classes from inside the model-widget (`gui/.../lifetime.py:321`) — a hard GUI dependency on what should be model.
- `parameter_registry.json` exists because parameter metadata (label, bounds, description) is scattered across `__init__`s and has to be *scraped back out*.

Yet the pieces for "model declares, GUI derives" already exist and are unused by the hand-written models: a generic renderer (`FittingParameterGroupWidget`, `parameter_widgets.py:1658`), the `parameter.controller = widget` binding, and two declarative model factories (`function_to_model_decorator`, `tcspc.models.json`). This PRD generalises that path to all structured models and adds the missing **strict boundary** + **JSON authoring surface**.

Relationship: complements **PRD-23** (thin view-only widgets, construction smoke tests) and **PRD-26** (declarative generation from data). This PRD applies the same "declare once, generate the surface" principle to model editors.

# Design

Strict three-layer split with a one-directional dependency (`gui` → `core`, never the reverse):

1. **Compute (pure, `core/models/**`)** — parameters + `update_model`. No Qt. Enforced by an AST CI test, not convention.
2. **View spec (data)** — a `ModelView` tree of section/plot descriptors. Authored as a `<model>.view.json` file co-located with the compute model and meant to be hand-edited. The dataclasses are the *schema* that validates it. Custom UI and target parameter groups are referenced by **string**, never by a widget class.
3. **Presentation (`gui/**`)** — `AutoModelWidget(model)` builds the control panel by **composition** (has-a model). A registry maps string keys → concrete plot classes and bespoke section widgets.

Customisation is preserved without leaking Qt into the model: a section of type `custom` carries a `key`; the GUI registry owns the hand-written widget under that key (e.g. the lifetime amplitude options header, the link/read menus, the r(t) panel). Unlimited customisation, zero boundary violation.

## The boundary rule (enforced)

`chisurf/core/models/**` must never import `qtpy`, `PyQt5/6`, `PySide2/6`, or `chisurf.gui`. A stray import fails CI.

# API

Core (pure data; `core/models/view_spec.py`):

```python
ModelView(sections: tuple[Section, ...], plots: tuple[PlotSpec, ...])
Section subtypes: ParameterGroupSection, DynamicGroupSection, CustomSection
PlotSpec(key: str, options: dict)
load_view_spec(path | dict) -> ModelView        # JSON loader + validation
```

Model hook (`core/models/model.py`):

```python
class Model:
    view_spec_file: str | None = None   # co-located <name>.view.json
    def view_spec(self) -> ModelView    # loads file, else auto-derives
```

GUI (`gui/widgets/models/`):

```python
AutoModelWidget(model)                                  # renders the editor
sections.registry.register_plot(key, factory)
sections.registry.register_section(key)                 # decorator
sections.registry.resolve_plot_specs(view) -> [(cls, options)]  # legacy bridge
```

## JSON shape (`lifetime.view.json`)

```json
{
  "sections": [
    {"type": "parameter_group", "target": "convolve", "title": "Convolution"},
    {"type": "dynamic_group", "target": "lifetimes", "title": "Lifetimes",
     "row_width": 2, "header_keys": ["lifetime_amplitude_options"]}
  ],
  "plots": [{"key": "line", "options": {"y_label": "counts"}}]
}
```

`target` names a model attribute resolved with `getattr`; `key` resolves in the GUI registry. No code in the file.

# Tasks

1. View-spec vocabulary + JSON loader (`view_spec.py`), pure data. — DONE
2. AST boundary CI test for `core/models/**`. — DONE
3. `Model.view_spec()` JSON discovery + auto-derive fallback. — DONE
4. `lifetime.view.json`; `LifetimeModel.view_spec_file`; seed ≥1 component on the compute side. — DONE
5. `AutoModelWidget` + section/plot registry + builtin registrations. — DONE
6. Offscreen-Qt render/add-del tests + headless view-spec tests. — DONE
7. **Live wiring**: GUI builds `AutoModelWidget(fit.model)` instead of inheriting the model; `fit_subwindow` plots via `resolve_plot_specs(view_spec())` instead of `fit.model.plot_classes`; port link/read menus + r(t) panel to registered custom sections. — DONE
8. Migrate remaining structured models (FRET, anisotropy, FCS, PDA) to `view.json`; delete their hand-written widgets.
9. Remove `plot_classes` from models once all consumers read `view_spec().plots`.

# Definition of Done

- A registered model class instantiated by `Fit` carries no Qt; the editor is a separate `AutoModelWidget`. Project save/load and fitting are unaffected.
- Editing `lifetime.view.json` (reorder sections, retitle, add/remove a plot) changes the live editor with no Python change.
- Custom UI (amplitude options, link/read, r(t)) works, referenced by key.
- Boundary test green; per-model widget files for migrated models deleted.

# Definition of Clean

- No `for w, v in zip(widgets, values)` value copy-back; model is the single source of truth, controllers re-render via the existing binding.
- No model imports a plot/widget class. No widget subclasses a model.
- Parameter metadata (label/bounds/units) lives on the parameter constructor, sourced from `parameter_registry.json`; not duplicated in widgets.

# Implementation status

**Increment 1 (data spine + boundary) — DONE.** `core/models/view_spec.py` (dataclasses + `load_view_spec`), `Model.view_spec()` with `view_spec_file` discovery and auto-derive fallback, `test/architecture/test_model_ui_boundary.py` (AST check, green). `core/models/**` is already Qt-free and now locked.

**Increment 2 (Lifetime authored as JSON) — DONE.** `core/models/tcspc/lifetime.view.json`; `LifetimeModel.view_spec_file = "lifetime.view.json"`; the model seeds one lifetime component (a zero-component lifetime model is invalid). The hand-written Python `view_spec()` was removed in favour of the data file.

**Increment 3 (generic renderer) — DONE.** `gui/widgets/models/auto_model_widget.py::AutoModelWidget` (composition), `gui/widgets/models/sections/{registry,builtin}.py` (plot keys + the `lifetime_amplitude_options` custom header, `resolve_plot_specs` bridge).

**Tests — DONE.** Run in the `arm64` conda env with `-p no:cov -o addopts=""`:
- non-GUI: `test/models/test_view_spec.py test/architecture/` (3 pass).
- offscreen Qt: `QT_QPA_PLATFORM=offscreen ... test/gui/test_auto_model_widget.py` (4 pass). Must run in its own pytest process — mixing GUI and non-GUI modules segfaults at Qt teardown. Existing `test/fitting/test_models_regression.py` unchanged (its one failure, `ParseModel()` with no fit, predates this work).

**Increment 4 (live wiring) — DONE (seams + additive entry); custom-section ports pending.** The live app now routes every model through a compatibility seam, `gui/widgets/models/model_editor.py`:
- `build_model_editor(fit.model)` — returns the model itself when it is already a widget (legacy, unchanged), else an `AutoModelWidget`. Wired at `main.py` (`modelLayout.addWidget`).
- `model_plot_specs(fit.model)` — plots from `view_spec().plots` via `resolve_plot_specs` (with distribution-accessor resolution), falling back to `plot_classes`. Wired at `fit_subwindow.py:156`.
Because all currently-registered models are widgets, both take the legacy branch → zero behaviour change. A pure model `LifetimeModelAuto` (same compute as `LifetimeModel`, shares `lifetime.view.json`) is registered **additively** in `experiment_configs.yaml` as menu entry **"Lifetime (auto-UI)"**, so the data-driven editor can be opened live alongside the hand-written one. Verified headless end-to-end (resolve → Fit → pure model → `AutoModelWidget` + 6 resolved plots; full fit lifecycle: `update_model`/`get_curves`/`chi2r`/residuals). Test: `test_registered_auto_lifetime_model_wires_live`.

**Increment 4b (link/read menus) — DONE.** The lifetime read/link header controls were ported from `LifetimeWidget` into the registered `lifetime_amplitude_options` section (`gui/widgets/models/sections/builtin.py`), operating on **core** `Lifetime` groups (`group.link = target`, value-copy via the action dispatcher) so they work for both legacy and auto-rendered models. Offscreen test: `test_lifetime_header_has_read_link_controls`.

**Increment 4c (plot-reference modes to core) — DONE.** The overlay modes (total/peak photons, donor reference, r(t) anisotropy) were relocated **verbatim** (programmatic AST extraction, no hand-copying) from the widgets to core:
- core `Anisotropy` gained the 6 diagnostics helpers (`_shift_trace_to_reference`, `_fit_timeshift`, `_fit_bg_level`, `_curve_bg_level`, `_extract_vv_vh_raw_for_diag`, `_extract_vv_vh_model_for_diag`).
- core `LifetimeModel` gained the plot-reference block (`_tcspc_reference_window` … `get_plot_reference_modes`); core `lifetime.py` now imports `plot_transforms`.
The widget copies were removed; legacy widgets inherit the methods (verified: `LifetimeModelWidget`/`AnisotropyWidget` still construct and expose the same 3 modes; all TCSPC model widgets import; `EtModelFreeWidget` never had them, no regression). Boundary test still green (the moved code is Qt-free), regression unchanged, and the pure `LifetimeModelAuto` now exposes `get_plot_reference_modes()` headless.

**Increment 4d (code view opens model + view.json) — DONE.** The fit window's plot↔code toggle (`FitSubWindow.show_code_view`) now also lists the model's `*.view.json` in the file picker and opens it as a second editor tab next to the model source (`CodeEditor.open_file` is tab-based and detects JSON), so "Code" shows both the computation and its editor layout. Resolver: `source_jump.resolve_model_view_spec_path`. Test: `test_code_view_resolves_model_view_json`.

**Increment 4e (code view targets the COMPUTE class, not the widget) — DONE.** Screenshot bug: for a *legacy* `LifetimeModelWidget`, "Code" opened the GUI widget `.py` (and missed the json) because `show_code_view`/`save_model_code` used `self.fit.model.__class__` — the widget. Added `source_jump.resolve_compute_model_class` (walks `type(model).__mro__`, returns the most-derived `core.models.model.Model` subclass that is **not** a Qt widget). `show_code_view` now opens `core/models/tcspc/lifetime.py` + `lifetime.view.json` for both legacy and pure entries. `save_model_code` resolves/reloads the compute module and only hot-swaps `instance.__class__` when the live object *is* the pure compute model (swapping a widget's class to the pure model would strip its Qt behaviour). Test: `test_code_view_legacy_widget_resolves_to_compute_model`.

**Increment 5 (flip primary Lifetime entry to the pure model) — DONE.** The `experiment_configs.yaml` "Lifetime" entry now registers `chisurf.core.models.tcspc.lifetime.LifetimeModel` (the pure compute model) — its editor is auto-generated by `AutoModelWidget`. The temporary `LifetimeModelAuto` demo class and its config line were removed (the pure model is now primary). The dead `LifetimeModelWidget` standalone class + its `tcspc/__init__` re-export were deleted (~54 lines). **`gui/widgets/models/tcspc/lifetime.py` was NOT deleted**: it still defines the *shared bases* `LifetimeWidget` and `LifetimeModelWidgetBase` (and `LifetimeMixtureModelWidget`) that the FRET/Gaussian/PDDEM/WLC/fret_rate/lifetime_mix widget family inherits. That file shrinks to its real size only once that family is migrated (task 8). Verified headless: dependent widgets still import; pure `LifetimeModel` → `Fit` → `AutoModelWidget` + 6 plots; 11/11 GUI + 3 boundary/view-spec tests green; models-regression unchanged (1 pre-existing `ParseModel()` failure).

**Increment 5b (flip fallout fixes, found by live GUI run) — DONE.** Running the flipped app surfaced two regressions, both fixed:
- *"Lifetime" missing from the model combobox.* The user copy `~/.chisurf/experiment_configs.yaml` **replaces** (not merges) the default model list and still pinned the deleted `...gui.widgets.models.tcspc.LifetimeModelWidget`, which resolved to `None` → entry dropped. Fix: a back-compat **alias** `LifetimeModelWidget = LifetimeModel` in the widget module (+ `__init__` re-export). Old configs/pickled projects now resolve to the pure model (GUI builds an `AutoModelWidget`). *General rule: deleting a registered model class needs a deprecation alias — user configs and projects pin class paths.*
- *`Failed to load view spec 'lifetime.view.json' for FRETrateModelWidget`.* The FRET/Gaussian/etc. widgets inherit `view_spec_file` from `LifetimeModel`, and `Model.view_spec()` resolved it against `type(self)`'s module (the FRET widget's dir). Fixes: (a) `Model.view_spec()` resolves `view_spec_file` against the **declaring** class's module (MRO `__dict__` walk); (b) `model_plot_specs()` now returns `plot_classes` directly for any `QWidget` model, so a legacy widget never inherits the lifetime view-spec's plots and `view_spec()` isn't consulted for widgets at all. Tests: `test_lifetime_model_widget_is_backcompat_alias`, `test_legacy_widget_does_not_inherit_lifetime_plots`.

**Increment 5c (real runtime wiring — the model panel) — DONE.** The flip crashed live: `add_fit failed ... addWidget(): argument 1 has unexpected type 'LifetimeModel'`. Increment 4 wired the seam at `main.py modelLayout.addWidget`, but the **actual** runtime add-fit path is `macros/core_fit.py::add_fit` (`gui.modelLayout.addWidget(fit.model)`), which was never routed through the seam — and several sites manipulate `fit.model` as a widget. Fixes:
- `model_editor.py` gained editor **caching + lifecycle helpers**: `build_model_editor` now caches the `AutoModelWidget` on the model (`_chisurf_model_editor`, an underscore attr → skipped by view-spec auto-derive, not pickled), with an `_is_alive()` guard so a layout-cleared (deleted) editor is rebuilt rather than reused. New `model_editor_widget` / `show_model_editor` / `hide_model_editor`.
- Wired the real sites: `core_fit.py` add (`addWidget(build_model_editor(fit.model))`), fit-switch show/hide (`show/hide_model_editor`), and the two report screenshots (`model_editor_widget(...)`, skip if `None`); `main.py:412 subWindowActivated` (`show_model_editor`). Post-fit value refresh needs no new wiring — `Model.update()`/`finalize()` are pure-core and propagate to the bound controllers.
- `Model.update()` now skips parameter groups without an `update()` method (a GUI-widget-only method) instead of logging a warning every fit iteration.
Tests: `test_add_fit_display_path_wires_pure_model` (+ existing 13). Verified headless: pure model → build editor → `addWidget` → `update`/`finalize`/`rebuild`, no crash, no per-iteration warning spam.

**Increment 6 (curve inputs + empty-group fix + test path) — DONE.** The live GUI exposed that convolve/generic/corrections/anisotropy sections rendered empty and the flipped model "did not compute". Two causes, both fixed:
- *Empty sections:* those groups define parameters as plain attributes (`self._dt = FittingParameter(...)`) that only surface in `parameters_all` after `find_parameters()`. `AutoModelWidget._build_parameter_group` now calls it when the list is empty, so the ~11 convolve params (etc.) render.
- *No way to compute (curve inputs):* these groups need a **data curve** (IRF, background, linearization table), not scalar parameters. Per the chosen direction, the view-spec vocabulary was **extended** with a first-class `CurveInputSection` (`view_spec.py`: label, select_action, unload_action, index_key, name_key, name_attr; `curve_input` in `_SECTION_TYPES`). The GUI renders it via one generic `CurveInputWidget` (`sections/builtin.py`): an `ExperimentalDataSelector` whose selection dispatches `select_action` with `{index_key, name_key, fit_index}` then `fit.update`, with an optional unload. `lifetime.view.json` now declares the **IRF** input (`model.change_irf`/`model.unload_irf`) and the **linearization table** (`model.set_linearization`/`model.unload_lintable`). *Gap:* generic background has no clean set-action (the old widget set `_background_curve` directly) — it needs a new `model.set_background_curve` action before becoming a curve_input.

**Testing path established (user mandate).** "I do not want to test all changes manually." Added `test/gui/test_model_editor_integration.py` — a headless end-to-end smoke test that walks the real path (resolve model by name from config → build `Fit` → `build_model_editor` → every section populated → IRF present → model computes a finite decay); each assertion maps to one of the GUI bugs this increment fixed. Codified as the **`/test-model-editor` skill** (`.claude/skills/test-model-editor/SKILL.md`): run order, env gotchas (arm64, `-p no:cov`, offscreen-own-process, cosmetic teardown segfault), and the rule to extend the integration test when adding a model/section. Pure-data `CurveInputSection` round-trip lives in `test/models/test_view_spec.py`.

**Increment 7 (choice/toggle vocabulary) — DONE.** Gaps were enumerated by *self-inspected offscreen screenshots* of the editor (not by the user). Added two more first-class section types: `ChoiceSection` (enum combo, attr- or action-bound) and `ToggleSection` (bool checkbox). One generic `ChoiceWidget` / `ToggleWidget` (`_BoundControlMixin`) sets the bound attribute on the target group (or dispatches an action) then triggers a fit update; option lists may be inline or from a named source (`window_function_types`). `lifetime.view.json` now declares: convolution **type** (`convolve.mode`), **do-convolution**, the linearization **smoothing** window (`corrections.window_function`), the **pile-up / DNL / reverse** correction toggles, and the **polarization** type (`anisotropy.polarization_type`). Verified by screenshot + write-through tests. Vocabulary is now: `parameter_group`, `dynamic_group`, `curve_input`, `choice`, `toggle`, `custom`, plus plots.

**Remaining Lifetime parity gaps (tracked, not blocking):**
- anisotropy **rotation components** add/remove — core `Anisotropy` has the `_rhos`/`_bs` lists but no `append`/`pop` (only `Lifetime` does); add those to core, then a `dynamic_group` section renders it.
- generic **background curve** — no clean set-action (the old widget set `_background_curve` directly); add a `model.set_background_curve` action, then a `curve_input`.
- convolve **FWHM** read-only display (minor).

**START NEXT — close the 3 tracked parity gaps, then migrate the widget family.** Remaining: (1) GUI eyeball the flipped "Lifetime" editor in the running app; (2) task 8 — migrate the structured models that still inherit `LifetimeModelWidgetBase` (FRET rate/structure, Gaussian, PDDEM, WLC, lifetime_mix, anisotropy) to `view.json` + custom sections, which is what finally lets `gui/widgets/models/tcspc/lifetime.py` (and the others) be deleted; (3) task 9 — drop `plot_classes` once all consumers read `view_spec().plots`.

# Relationships
- Complements PRD-23 (thin view-only widgets) and PRD-26 (declarative generation from data).
- Generalized by [PRD-40](prd-40.md), which lifts this machinery out from under `models/` into a reusable `core/dataspec` + `gui/autoform` framework.
- Provides the numeric-input consumer that [PRD-42](prd-42.md) supplies a dependency-free replacement for.
- Realizes the [GUI & AutoForm](/subsystems/gui-autoform.md) direction over the [Core target](/specs/core.md).
