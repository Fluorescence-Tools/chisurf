---
type: PRD
prd: "detector-setup"
title: "PRD (detector-setup): Centralized Detector Setup Selection"
description: Replace the full detector/PIE-window wizard page embedded across 15+ plugin UIs with a lightweight setup-selector widget that opens the full editor on demand.
status: planned
phase: "unassigned"
resource: overhaul/PRD_centralized_detector_setup_selection.md
tags: [prd, gui, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
The full-featured detector/PIE-window definition wizard page is instantiated in 15+ plugin UIs, cluttering windows where detector setup is secondary, duplicating complexity, and multiplying maintenance burden as any wizard change propagates to every embedding site. This PRD introduces a lightweight `SetupSelectorWidget` — a combo box of saved setups, a "Setup Editor…" button that opens the full wizard in a modal dialog, and a tooltip summarizing the selected setup. The full wizard page remains available for the settings tool, the onboarding wizard, and the on-demand modal dialog. Usage sites are migrated in priority order (direct embeds, layout-adaptation sites, wizard-page replacements, and near-pattern simplifications), leaving the setups file format and MFDB integration unchanged.

# Status
Planned / unassigned (STATUS TABLE authoritative). New widget plus staged P0–P3 migration across plugin UIs; a small set of sites remain unchanged by design.

# Relationships
- Cross-cutting GUI simplification across many plugins in the [plugin system](/architecture/plugin-system.md).
- Aligns with the short-label/tooltip and centralized-editing conventions in [GUI & AutoForm](/subsystems/gui-autoform.md).
- Preserves the existing detector-setups backend and MFDB integration ([MFDB (current)](/architecture/mfdb.md)).

# Source
- Primary: `overhaul/PRD_centralized_detector_setup_selection.md`
