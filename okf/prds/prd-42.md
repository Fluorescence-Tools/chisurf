---
type: PRD
prd: "42"
title: "PRD-42: Confine pyqtgraph to plot-only widgets"
description: Restricts pyqtgraph to plot-canvas code and replaces its non-plot uses (numeric spin boxes, parameter trees) with dependency-free Qt-native equivalents.
status: planned
phase: "unassigned"
resource: overhaul/PRD-42-drop-pyqtgraph-fitting-widgets.md
tags: [prd, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-42 enforces a clean dependency rule that pyqtgraph is a plot library, not a widget toolkit. It replaces the non-plot uses — `pg.SpinBox` numeric inputs and `pg.parametertree` hierarchical trees across the fitting widgets, a shared helper, and the parameter editor — with Qt-native equivalents: a new `ScientificDoubleSpinBox` (decimal stepping, scientific-notation display, unit suffix, optional infinities) that is drop-in for the spin box call sites, and a `QTreeWidget`-based `ParameterEditorWidget`. Plot-canvas uses of pyqtgraph are explicitly kept. The dependency-free spin box is intended to become the canonical numeric input inside the [PRD-40](prd-40.md) AutoForm/DataSpec framework.

# Status
Planned (unassigned phase, STATUS TABLE authoritative). Scope, replacement widgets, and a task-ordered migration path with grep-based success criteria are specified.

# Relationships
- Delivers a prerequisite for [PRD-40](prd-40.md): a compact, dependency-free input widget and a clean `FittingParameterWidget` that AutoForm can wrap under a `fitting_parameter` item type.
- Uses the [PRD-38](prd-38.md) `/test-model-editor` path to verify the fitting-widget swaps cause no regressions.
- Advances the [GUI & AutoForm](/subsystems/gui-autoform.md) direction.

# Source
- Primary: `overhaul/PRD-42-drop-pyqtgraph-fitting-widgets.md`
