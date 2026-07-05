---
type: PRD
prd: "32"
title: "PRD-32: Acquisition Standard Output Folder"
description: Adds a single user-configurable standard output folder to acquisition so new measurements have a predictable save location.
status: planned
phase: "cross-cutting"
resource: overhaul/PRD-32-acquisition-output-folder.md
tags: [prd, acquisition]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Gives acquisition a single, user-configurable standard output folder so new measurements have a predictable save location without the user picking a directory every time. The setting is made explicit in the Setup surface, persisted in `gui.acquisition`, used to prefill the acquisition dock, and resolved as the default runtime destination (creating the folder before a run writes files). Intentionally narrow: no direct MFDB registration, no device-format changes, no project-scoped output-tree policy.

# Status
Planned per the authoritative status table, though the document states the configuration/runtime path is implemented in the current branch (settings panel field, dock default, start-time resolution).

# Relationships
- Deliberately kept separate from [PRD-33](prd-33.md) (acquisition→MFDB registration), which remains the more complex database path requiring sample ownership/provenance decisions first.
- Relates to acquisition/acq and the [plugin system](/architecture/plugin-system.md).

# Source
- Primary: `overhaul/PRD-32-acquisition-output-folder.md`
