---
type: PRD
prd: "32"
title: "PRD-32: Acquisition Standard Output Folder"
description: Adds a single user-configurable standard output folder to acquisition so new measurements have a predictable save location.
status: planned
phase: "cross-cutting"
resource: chisurf/plugins/core/acq
tags: [prd, acquisition]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Gives acquisition a single, user-configurable standard output folder so new measurements have a predictable save location without the user picking a directory every time. The setting is made explicit in the Setup surface, persisted in `gui.acquisition`, used to prefill the acquisition dock, and resolved as the default runtime destination (creating the folder before a run writes files). Intentionally narrow: no direct MFDB registration, no device-format changes, no project-scoped output-tree policy.

# Status
Planned per the authoritative status table, though the configuration/runtime path is implemented in the current branch: the acquisition settings panel exposes a standard output folder, the acquisition dock reads it by default, and new runs resolve their output location from that setting before starting.

# Goal

Give acquisition a single, user-configurable standard output folder so new
measurements have a predictable save location without making the user pick a
directory every time.

# Why

The current acquisition flow already has an ad-hoc output-path field in the dock,
but the setting is not defined in the central setup surface. That makes the save
location easy to miss and hard to standardize across sessions.

This PRD makes the setting explicit in the setup UI and treats it as the default
runtime destination for acquisition output.

# Scope

- Add a standard output-folder field to acquisition settings in Setup.
- Persist the setting in `gui.acquisition`.
- Prefill the acquisition dock from the saved setting.
- Resolve the output folder at acquisition start if the dock is empty.
- Create the destination folder before the run writes files.

# Non-goals

- Direct MFDB registration.
- Changing the device-specific file formats.
- Introducing a new project-scoped output tree policy.

# Definition of Done

- [ ] Acquisition settings expose a standard output-folder field.
- [ ] The folder persists across restarts.
- [ ] The acquisition dock defaults to the saved folder.
- [ ] Acquisition start uses the saved folder when the dock field is empty.
- [ ] The output folder is created before a run writes files.

# Notes

This PRD is intentionally narrow. The MFDB alternative is a separate, more
complex PRD ([PRD-33](prd-33.md)) because it needs sample ownership and
provenance decisions first.

# Relationships
- Deliberately kept separate from [PRD-33](prd-33.md) (acquisition→MFDB registration), which remains the more complex database path requiring sample ownership/provenance decisions first.
- Relates to acquisition/acq and the [plugin system](/architecture/plugin-system.md).
