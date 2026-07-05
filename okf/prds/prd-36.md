---
type: PRD
prd: "36"
title: "PRD-36: Dockable-Tool Base Migration Tracker"
description: Tracks the per-tool rollout of the shared dockable-tool base across remaining QMainWindow plugin tools so drag-drop, dock, geometry, and MFDB-connectivity boilerplate is implemented once.
status: in-progress
phase: "cross-cutting"
resource: overhaul/PRD-36-dock-tool-migration-tracker.md
tags: [prd, gui, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Tracks the incremental rollout of the shared dockable-tool base (`ChisurfDockTool` + `PathDropListWidget`) across every remaining `QMainWindow` plugin tool, so the path drag-drop, docking, window-geometry persistence, and lazy MFDB-connectivity boilerplate is implemented once rather than re-forked per tool. It documents the per-tool migration recipe (subclass the base, swap the drop widget, delete duplicated drop handlers, route MFDB acquisition through the base, lazy-load the GUI tool, add an offscreen construction smoke test), lists tools already migrated, and enumerates the priority-A drag-drop and priority-B plain-window backlog. Non-`QMainWindow` wizard tools are out of scope for this base.

# Status
In progress. The base, smoke-test pattern, and the repo-wide read-only-construction guard exist; three reference tools are migrated and a backlog of ~20 tools remains.

# Relationships
- The rollout backlog for [PRD-23](prd-23.md) Task 1 (thin dockable tools); each migration also advances its smoke-test and read-only-construction tasks.
- Touches the [plugin system](/architecture/plugin-system.md) and the [Plugins target](/specs/plugins.md); read-only-construction rule keeps tools from opening [MFDB (current)](/architecture/mfdb.md) connections on init.

# Source
- Primary: `overhaul/PRD-36-dock-tool-migration-tracker.md`
