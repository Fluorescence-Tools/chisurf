---
type: PRD
prd: "23"
title: "PRD-23: Thin Widgets / View–API Separation"
description: Makes GUI widgets pure view — no data processing, no database or acquisition-library calls, no side effects on construction — with mandatory construction smoke tests and a shared dockable-tool base.
status: in-progress
phase: "cross-cutting"
resource: overhaul/PRD-23-thin-widgets-view-api.md
tags: [prd, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-23 makes GUI widgets pure view: they call only the api/RPC client, hold no database or photon-library logic, and perform no side effects (especially no database writes) on construction. It mandates a construction smoke test per tool to catch missing-import and side-effect-on-init regressions, and introduces a shared dockable-tool base implementing Load/Save, drag-drop, dock management, and connectivity-aware MFDB behaviour once instead of per plugin. This removes the class of bugs where logic hidden inside a widget ships broken, and enforces the GUI half of the transformer contract.

# Status
In-progress (cross-cutting, STATUS TABLE authoritative). The shared `ChisurfDockTool` base, residual-logic extraction, construction smoke tests, and a repo-wide static guard against database writes in widget `__init__` have landed for the reference transformers plus a third tool; remaining dockable tools migrate opportunistically.

# Relationships
- Enforces the GUI half of [PRD-16](prd-16.md) (transformer contract); adds construction smoke tests as a conformance checklist item.
- Generalizes fixes made to the reference-transformer tools.
- Leans on the completed MVC controller separation so state lives in the model/controller.
- Targets the [Plugins target](/specs/plugins.md); widgets talk only through the [RPC target](/specs/rpc.md).

# Source
- Primary: `overhaul/PRD-23-thin-widgets-view-api.md`
