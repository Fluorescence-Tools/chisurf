---
type: PRD
prd: "38"
title: "PRD-38: Model/UI Split — view-spec JSON drives auto-generated model editors"
description: Splits a fitting model's compute definition from its editor by describing the editor in a co-located JSON view spec that a generic GUI renderer turns into the control panel.
status: in-progress
phase: "unassigned"
resource: overhaul/PRD-38-model-view-spec-split.md
tags: [prd, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-38 makes a fitting model's editor the automatic result of its computational definition instead of a hand-written per-model widget that duplicates the model's structure and welds Qt to the compute side. Each model stays in one place (parameters plus `update_model` in `chisurf/core/models`, Qt-free), its editor is described in a hand-editable `<model>.view.json` file, and the GUI renders that spec by composition via `AutoModelWidget`. A strict, AST-CI-enforced boundary keeps `core/models/**` from importing any GUI toolkit, while a string-keyed registry provides an escape hatch for bespoke custom sections. The view-spec vocabulary has grown to parameter groups, dynamic groups, curve inputs, choices, toggles, and custom sections plus plots.

# Status
In-progress (unassigned phase, STATUS TABLE authoritative). Data spine, boundary test, generic renderer, live wiring, and several section types are done; migrating the remaining structured-model widget family (FRET, anisotropy, FCS, PDA) and dropping `plot_classes` remain.

# Relationships
- Complements PRD-23 (thin view-only widgets) and PRD-26 (declarative generation from data).
- Generalized by [PRD-40](prd-40.md), which lifts this machinery out from under `models/` into a reusable `core/dataspec` + `gui/autoform` framework.
- Provides the numeric-input consumer that [PRD-42](prd-42.md) supplies a dependency-free replacement for.
- Realizes the [GUI & AutoForm](/subsystems/gui-autoform.md) direction over the [Core target](/specs/core.md).

# Source
- Primary: `overhaul/PRD-38-model-view-spec-split.md`
