---
type: Plugin Group
title: Core tools plugins
description: Infrastructure plugins — settings, onboarding, plugin/model management, MFDB admin and user management, project browsing, updates, acquisition and batch analysis.
resource: chisurf/plugins/core/
tags: [plugins, infrastructure]
timestamp: '2026-07-05T00:00:00Z'
---

Core tools are the host-side infrastructure: configuration, onboarding, database and
user administration, project I/O, updates, data acquisition, and batch runs. They live
under `chisurf/plugins/core/` and follow the same manifest contract as feature plugins,
so ChiSurf's own plumbing is packaged as plugins too.

| Plugin dir | Display name | What it does |
| --- | --- | --- |
| `core/setup` | Setup:Settings | Unified Settings dialog that hosts several config panels. |
| `core/boarding` | Help:Boarding Wizard | First-run onboarding wizard (rebuilt as a `boarding.view.json` directed stepper). |
| `core/plugin_manager` | Setup:Plugins | Enable/disable and inspect installed plugins. |
| `core/plugin_check` | Tools:Miscellaneous:Plugin-Check | Startup-error test harness across all plugins; reports pass/fail/skip. |
| `core/model_manager` | Setup:Models | Manage fitting models. |
| `core/mfdb_admin` | Tools:MFDB Admin | Manage the Multiparametric Fluorescence Database: samples, experiments, setups, data, provenance, project archives, fluorophore curation. |
| `core/user_editor` | Setup:User Editor | Manage users registered in the MFDB. |
| `core/switch_user` | Setup:Switch User | Switch the active MFDB user for the session. |
| `core/database_connector` | Core:Database Connector | Source/user DB resolution, migration, backup/reset, FLR CIF import/export. |
| `core/project_browser` | Tools:Open Project | Browse, save, restore, export/import projects via MFDB with version control. |
| `core/updater` | Setup:Updates & Packages | Update checker/installer and conda package manager (panels inside Settings). |
| `core/acq` | Main:Tools:Acquisition | Single-molecule acquisition from TCSPC hardware or the built-in tttrlib photon simulator. |
| `core/batch_analysis` | Main:Tools:Batch-Analysis | Apply one template fit to many datasets/files and export consolidated results. |
| `core/globalview` | Main:Tools:Global View | Interactive network graph of parameter relationships across fits. |
| `core/lightpath_simulator` | Spectroscopy:Light Path Simulator | Compute crosstalk and R₀ overlap integrals for an optical path. |
| `sample_database` | Legacy:Sample Database | Retired prerelease MFDB surface; active work belongs in `core/mfdb_admin`. |
| `ai_settings` | Tools:AI Settings | Root-level AI provider/backend configuration tool. |

Every tool is discovered by `manifest.json`, activated with a `PluginContext`, and
renders declaratively ([plugin system](/architecture/plugin-system.md),
[Plugins target](/specs/plugins.md), [GUI & AutoForm](/subsystems/gui-autoform.md)).
The database-facing tools (`mfdb_admin`, `user_editor`, `switch_user`,
`project_browser`, `database_connector`) are front-ends over the provenance store —
see [MFDB](/architecture/mfdb.md) and [PRD-02b](/prds/prd-02b.md). Acquisition tracks
[PRD-32](/prds/prd-32.md)/[PRD-33](/prds/prd-33.md). Because this plumbing is itself
plugins, it exercises the same discovery/lifecycle rules the [Plugins target](/specs/plugins.md)
demands of feature code.
