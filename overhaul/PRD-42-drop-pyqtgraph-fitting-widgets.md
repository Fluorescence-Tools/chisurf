# PRD-42: Confine pyqtgraph to plot-only widgets

## Goal

Restrict `pyqtgraph` usage to plot canvas code only (`PlotWidget`, `mkPen`, etc.).
Non-plot uses — numeric input (`pg.SpinBox`) and hierarchical parameter trees
(`pg.parametertree`) — must be replaced with Qt-native equivalents.

This enforces a clean dependency rule: **pyqtgraph is a plot library, not a
widget toolkit**. Input widgets that accidentally pull in pyqtgraph cause
unnecessary import-time cost and make the dependency harder to reason about.

## Scope — files to fix

| File | pyqtgraph symbols used |
|------|------------------------|
| `chisurf/gui/widgets/fitting/parameter_widgets.py` | `pg.SpinBox` (×7 — value, lb, ub editors in both the main widget and detail popup) |
| `chisurf/gui/widgets/fitting/fit_subwindow.py` | import only, no `pg.*` call — safe to delete the import |
| `chisurf/gui/widgets/fitting/fit_controller.py` | import only — delete |
| `chisurf/gui/widgets/fitting/fit_list.py` | import only — delete |
| `chisurf/gui/widgets/general.py` | `pg.SpinBox` in `make_widgets_from_yaml` helper |
| `chisurf/gui/widgets/parameter_editor/parameter_editor.py` | `pg.parametertree.ParameterTree`, `pg.parametertree.Parameter`, `pg.parametertree.parameterTypes.{ListParameter,GroupParameter}` |

**Allowed** (plot canvas — keep as-is): `chisurf/gui/autoform/sections/builtin.py`
uses `pg.PlotWidget` / `pg.mkPen` — these are plot uses and are explicitly permitted.

## What `pg.SpinBox` provides (constraints on replacement)

pyqtgraph's `SpinBox` extends `QAbstractSpinBox` with:

- **Decimal stepping** (`dec=True`) — step size scales with the current value
  magnitude (×1.01 / ×0.99 per click).
- **Scientific-notation display** — renders `1.23e-04` rather than `0.000123`.
- **Suffix** — unit label appended to the displayed text.
- **`finite=False`** — allows ±inf as a valid value.
- **`editingFinished`** signal — identical to `QAbstractSpinBox`.

The replacement must match this API surface so call sites require minimal change.

## Replacement: `ScientificDoubleSpinBox`

A new widget in `chisurf/gui/widgets/scientific_spinbox.py`:

```python
class ScientificDoubleSpinBox(QtWidgets.QAbstractSpinBox):
    """QAbstractSpinBox with decimal stepping, scientific-notation display,
    optional unit suffix, and optional ±inf support.

    Drop-in for pg.SpinBox at the call sites in parameter_widgets.py.
    Exposes the same constructor kwargs used there:
      dec, decimals, suffix, finite
    and the same .value() / .setValue() / .editingFinished interface.
    """
```

Key implementation notes:

- Subclass `QAbstractSpinBox`; render via a `QLineEdit` (already provided by
  the base class).
- Validate with `QDoubleValidator` extended to accept `inf`/`-inf` strings when
  `finite=False`.
- `stepBy(steps)` override: multiply current value by `1.01 ** steps` (decimal
  step), clamped to configured min/max.
- Display: `format(value, f'.{decimals}e') + suffix` written into the line edit
  on every value change.
- Keep `opts` dict attribute populated (`{'decimals': ..., 'compactHeight': ...}`)
  so the popup code reading `self.controller.widget_lower_bound.opts` keeps
  working without further change.

## Replacement: `ParameterEditorWidget`

`parameter_editor.py` wraps `pg.parametertree.ParameterTree` to display/edit
a nested settings dict. Usage is internal (settings dialog); it does not need to
stay structurally identical.

Replace with a `QTreeWidget`-based implementation:

- `dict_to_parameter_tree(origin)` (already present) produces a flat list-of-dicts
  describing the tree — reuse that output as input to a recursive `QTreeWidget`
  builder.
- Leaf nodes get an inline editor widget (inserted via `QTreeWidget.setItemWidget`):
  `bool` → `QCheckBox`, `int` → `QSpinBox`, `float` → `ScientificDoubleSpinBox`,
  `str` → `QLineEdit`, `list` → `QComboBox`.
- `parameter_tree_to_dict` must still work; rewrite it to walk `QTreeWidgetItem`s
  instead of `pg.Parameter` nodes.
- Keep the public API of `ParameterEditorWidget` unchanged:
  `load_settings(path)`, `save_settings(path)`, `set_parameter(dict)`.

## Migration path (task order)

1. **`ScientificDoubleSpinBox`** — implement and unit-test in isolation
   (`test/test_scientific_spinbox.py`): value round-trip, decimal stepping,
   inf support, suffix display.
2. **`parameter_widgets.py`** — swap `pg.SpinBox(...)` → `ScientificDoubleSpinBox(...)`.
   Remove `import pyqtgraph as pg`. Run `/test-model-editor` skill to verify no
   regressions.
3. **`general.py`** — swap the one `pg.SpinBox(value=value)` call; remove
   local `import pyqtgraph as pg`.
4. **`fit_subwindow.py`, `fit_controller.py`, `fit_list.py`** — delete the dead
   `import pyqtgraph as pg` lines.
5. **`parameter_editor.py`** — implement `QTreeWidget`-based replacement;
   remove all three `pyqtgraph` imports. Test with the settings dialog.
6. **Verify** — `grep -r "import pyqtgraph" chisurf/gui/widgets/` must return
   only `autoform/sections/builtin.py` (plot canvas — expected). Run full non-GUI test suite.

## Success criteria

- `grep -r "import pyqtgraph" chisurf/gui/widgets/fitting/` → empty.
- `grep -r "import pyqtgraph" chisurf/gui/widgets/parameter_editor/` → empty.
- `grep -r "import pyqtgraph" chisurf/gui/widgets/general.py` → empty.
- `FittingParameterWidget` still displays and edits value/bounds correctly
  (verified by `/test-model-editor` skill).
- `ParameterEditorWidget` round-trips a settings JSON file without data loss.
- Existing test suite (`pixi run test`) passes without new failures.

## AutoForm alignment (long-term direction)

`FittingParameterWidget` should eventually be a first-class item in the
PRD-40 AutoForm / DataSpec framework, not a hand-wired standalone widget:

- Add a `FittingParameterSection` (or extend `ParameterGroupSection`) in
  `chisurf/core/dataspec/` that represents a **single** fitting parameter:
  its value, bounds, fixed flag, link, unit suffix, and description.
- Register `FittingParameterWidget` (the concrete renderer) in
  `chisurf/gui/autoform/sections/registry.py` under key `"fitting_parameter"`.
- When AutoForm encounters a `FittingParameterSection`, it delegates to the
  registry; the registered factory returns a `FittingParameterWidget` instance.
- This lets model `.view.json` files declare individual parameter overrides
  (custom label, hide bounds, etc.) through the same dataspec vocabulary
  already used for groups, curves, and panels.

`ScientificDoubleSpinBox` (introduced in this PRD) becomes the canonical
numeric input inside that renderer. Because it is a plain `QLineEdit`
subclass with no external dependencies, AutoForm can compose it freely.

This alignment work is tracked under PRD-40; PRD-42 delivers the prerequisite
(a compact, dependency-free input widget and a clean `FittingParameterWidget`
that AutoForm can wrap).

## Non-goals

- Removing pyqtgraph from plot widgets (`autoform/sections/builtin.py`, curve
  plots) — those uses are intentional and stay.
- Matching every visual detail of `pg.SpinBox` (compact height mode,
  hover-highlight) — functional equivalence is sufficient.
