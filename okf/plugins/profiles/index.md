---
type: Plugin Profiles Index
title: High-priority plugin profiles
description: Evidence records for plugin documentation and follow-up work.
resource: chisurf/plugins/
tags: [plugins, documentation, worklist]
timestamp: '2026-07-05T00:00:00Z'
---

# Purpose

These profiles populate OKF with concrete plugin-level evidence for the highest-risk
documentation targets. They are not replacements for plugin-local `README.md` files.
They are work records that say what exists now, what a maintainer must verify, and
which local README or `docs/` files should be written next.

Each profile follows the same shape:

- identity from `manifest.json`
- architecture and entrypoint evidence from the plugin tree
- data/provenance impact
- verification surface
- documentation work still needed

# Profiles

| Plugin | Why it is prioritized |
| --- | --- |
| [MFDB Admin](mfdb-admin.md) | Central database/provenance admin surface with a large RPC contract. |
| [Sample Database](sample-database.md) | Legacy/root MFDB surface that overlaps newer MFDB admin services. |
| [Database Connector](database-connector.md) | Backup/reset/import/export services touch user databases and external files. |
| [Project Browser](project-browser.md) | Project archive/restore/import/export can mutate persistent project state. |
| [Light Path Simulator](lightpath-simulator.md) | Full API/core/backend/RPC/CLI/GUI stack and a good client-server reference. |
| [Code Editor](code-editor.md) | Document mutation, lint/fix RPC, optional agent/LSP execution surfaces. |
| [HydroPro](hydropro.md) | External executable workflow with GUI, CLI, RPC, and generated output files. |
| [Trace Browser](trace-browser.md) | TTTR folder browsing with ratings, annotations, previews, and CSV export. |
| [TTTR Time Windows](tttr-time-windows.md) | TTTR-to-BID file generation used by downstream burst workflows. |
| [Pixel Phasor](pixel-phasor.md) | Phasor-FLIM analysis API/RPC/CLI/GUI used by imaging workflows. |

# Use

When documenting a plugin, start from the relevant profile, then write or update the
plugin-local `README.md` using [Plugin documentation standard](../documentation-standard.md).
Do not copy claims blindly: re-check the manifest, entrypoints, and tests before
declaring a plugin stable or complete.
