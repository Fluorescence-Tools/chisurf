---
type: PRD
prd: "46"
title: "PRD-46: Scripts as first-class citizens in the test pipeline"
description: Brings shipped example scripts into the automated test suite by running headless/process scripts under a shebang-driven runner with numeric assertions, plus unit tests for the public model API they exercise.
status: planned
phase: "unassigned"
resource: overhaul/PRD-46-script-testing-pipeline.md
tags: [prd, core]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-46 brings ChiSurf's runnable example scripts, which exercise the core model API end-to-end but sit outside CI, into the automated test suite so regressions in the public model API are caught automatically. A parametrized pytest runner discovers `scripts/*.py` and runs each according to its shebang: process scripts as subprocesses with stdout and numeric-output assertions, and console/ipython scripts headlessly by exec against a thin `cs` namespace stub that provides the real core but no GUI. It also adds unit tests for the public model API surface (`chain_length`, `persistence_length`, mixture `fractions` setter) and wires a `test-scripts` task into the default test suite. Scripts stay runnable interactively without modification — the harness is a thin wrapper.

# Status
Planned (unassigned phase, STATUS TABLE authoritative). Runner design, headless stub, numeric assertions, unit tests, and CI integration with acceptance criteria are specified.

# Relationships
- Adds a headless test path for scripts, complementing the separate GUI test task.
- Covers the public model API surface exposed through the model work related to [PRD-38](prd-38.md)/[PRD-40](prd-40.md).
- Reinforces the [Core target](/specs/core.md) by asserting the stable model API.

# Source
- Primary: `overhaul/PRD-46-script-testing-pipeline.md`
