---
type: PRD
prd: "40"
title: "PRD-40: A ChiSurf-native declarative dataset-to-editor framework"
description: Provides one reusable way to declare a typed dataset once and auto-generate its Qt editor across models, settings, and tool panels, replacing the several ad-hoc type-to-widget mappers.
status: planned
phase: "unassigned"
resource: overhaul/PRD-40-declarative-dataset-editors.md
tags: [prd, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-40 lifts the model-scoped machinery of [PRD-38](prd-38.md) into a general, ChiSurf-native framework so any structured, typed dataset can be declared once and get its editor generated automatically, replacing the roughly five ad-hoc type-to-widget mappers that exist today. It is inspired by a declarative form-generation library but stays native because it must understand domain types that library cannot — `FittingParameter` semantics (bounds/fixed/link/error), `DataCurve` selectors, and chinet-node binding. The design is three layers: a Qt-free `core/dataspec/` item vocabulary and `DataSet` tree, a `gui/autoform/` renderer (`AutoForm`) with a string-keyed registry, and consumers migrated one at a time onto `AutoForm` while their bespoke widget code is deleted. An enforced AST boundary keeps `core/dataspec/**` GUI-free.

# Status
Planned (unassigned phase, STATUS TABLE authoritative). The spec relocation, renderer lift, parameter-group adapter, generic scalar field, and one non-model consumer migration are recorded as done; the settings/metadata tree-editors are noted as a poor fit, so a fixed-form consumer and experiment-setup panels remain.

# Relationships
- Supersedes the model-only scope of [PRD-38](prd-38.md) by generalizing its machinery; PRD-38 continues to completion as the reference implementation.
- Complements PRD-23 (thin view-only widgets) and PRD-26 (declarative generation).
- Receives from [PRD-42](prd-42.md) the dependency-free numeric input and the clean `FittingParameterWidget` that a `fitting_parameter` item type wraps.
- Realizes the [GUI & AutoForm](/subsystems/gui-autoform.md) direction.

# Source
- Primary: `overhaul/PRD-40-declarative-dataset-editors.md`
