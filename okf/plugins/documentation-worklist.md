---
type: Plugin Documentation Worklist
title: Plugin documentation worklist
description: Screening of ChiSurf plugins for documentation coverage and priority.
resource: chisurf/plugins/
tags: [plugins, documentation, worklist]
timestamp: '2026-07-05T00:00:00Z'
---

# Screening Summary

Current inventory from `chisurf/plugins/`:

| Metric | Count |
| --- | ---: |
| Discoverable plugin manifests | 85 |
| Plugin roots without `README.md` | 37 |
| Plugin roots with a `docs/` folder | 3 |
| Plugins with visible API/core plus backend/server/rpc layering | 29 |

The documentation backlog is therefore not just missing READMEs. Several high-impact
plugins have manifest/RPC surfaces but no local human contract, while some older
READMEs describe user features without the architecture, MFDB, and verification
sections needed for maintainers.

# Priority Rules

Document plugins in this order:

1. **Data or provenance risk:** plugins that write MFDB, mutate files, archive projects,
   or register operations.
2. **Large RPC/API surface:** plugins with services, CLI, or many manifest RPC methods.
3. **Reference architecture:** plugins that should become examples for future work.
4. **User-facing scientific workflows:** plugins with complex input/output assumptions.
5. **Small GUI tools:** simple calculators and setup panels after higher-risk plugins.

# P0 Work

These should get full README coverage first, and `docs/CONTRACT.md` where RPC/API is
substantial.

| Plugin | Current gap | Why it matters | Recommended docs |
| --- | --- | --- | --- |
| `core/mfdb_admin` | README plus contract/status now exist; per-method schemas, workflow docs, and AutoForm migration remain. | Central database/provenance admin surface; many PRDs depend on it. | Extend `docs/CONTRACT.md`, add `docs/WORKFLOWS.md`, migrate ordinary panels to AutoForm JSON specs. |
| `sample_database` | Retired prerelease surface; hidden/deprecated manifest now declares no entrypoints or RPC methods. | Avoids documenting/supporting a duplicate MFDB namespace. | Delete/quarantine remaining wrapper files after confirming no live callers. |
| `core/database_connector` | README and temporary-MFDB smoke tests now exist; destructive-path schemas/tests remain. | Backup/reset/import/export touches user data. | Extend service contract and add backup/reset/import/export tests. |
| `core/project_browser` | README and sample-data-backed service/headless-GUI tests now exist; deeper branch/import tests remain. | Project persistence is high-impact and easy to misuse. | Add workflow examples, MFDB artifact diagrams, and import/branch coverage. |
| `core/lightpath_simulator` | No README; full api/core/backend/rpc/cli/gui stack. | Good client-server reference and scientific workflow. | `README.md`, `docs/CONTRACT.md`, verification commands. |
| `core/code_editor` | No README; backend services. | Executes user scripts/macros; safety and state need clarity. | `README.md` with execution model and service methods. |
| `modelling/hydropro` | No README; CLI/RPC/GUI and external executable assumptions. | External binary dependencies and generated outputs need explicit docs. | `README.md`, CLI examples, external-tool setup. |
| `tttr/trace_browser` | No README; api/core/backend/cli/gui stack. | File/folder ratings and exports are workflow-heavy. | `README.md`, input/output and persistence notes. |
| `tttr/tttr_time_windows` | No README; API/server/CLI/GUI stack. | Produces BIDs/time-window outputs used downstream. | `README.md`, `docs/CONTRACT.md`. |
| `microscopy/img_pixel_phasor` | No README; backend/CLI/services. | Phasor-FLIM outputs feed current PRD work. | `README.md`, workflow and output format notes. |

# P1 Work

These need at least the standard README, usually without a separate docs folder.

| Plugin | Current gap | Recommended focus |
| --- | --- | --- |
| `burst/burst_fcs_correlator` | No README; core/backend/gui split. | Explain burstwise FCS input folders, output curves, service calls. |
| `fcs/fcs_filter_calculator` | No README; backend/CLI/gui split. | Explain lifetime-filter math, input decays, output filters. |
| `microscopy/img_pixel_mle` | No README; API/backend/CLI/gui split. | Explain pixel-wise MLE workflow and HDF5/image outputs. |
| `fret_line` | No README; core/backend/gui split. | Explain FRET-line models, RPC/service status, ndxplorer overlays. |
| `spectra_downloader` | No README and missing manifest id. | Clarify staging-to-MFDB workflow and fix manifest identity separately. |
| `calculator/phasor_calculator` | No README; user-facing calculator. | Small README with phasor formulae, controls, and tests. |
| `core/user_editor` | No README; MFDB user mutation. | Safety rules, backend-enforced immutability constraints. |
| `core/setup` | No README; settings mutation. | Settings panels, persisted keys, verification. |
| `core/boarding` | No README; onboarding workflow. | First-run state, page model, tests. |
| `tttr/audifier` | No README. | Input formats, audio output, preview limits. |
| `tttr/tttr_lut_tools` | No README. | LUT generation inputs/outputs and setup interaction. |
| `traj/traj_tools` | No README. | Aggregator README linking trajectory subtools. |

# P2 Work

These are smaller, older, or lower-risk GUI tools missing a README.

| Plugin | Recommended focus |
| --- | --- |
| `ai_settings` | Provider/backend settings, stored configuration keys. |
| `burst/burst_analysis` | Aggregator shell and included panels. |
| `burst/burst_browser` | Accepted burst table formats and plot actions. |
| `core/plugin_check` | Startup-check semantics and failure categories. |
| `core/switch_user` | Active-user switch behavior and MFDB implications. |
| `fcs/fcs_channel_preset` | Detector/channel preset workflow. |
| `fluorescence_decay/lifetime_analysis` | Aggregator shell and included decay tools. |
| `microscopy/imaging_tools` | Imaging toolbox aggregator. |
| `microscopy/img_calibration` | IRF/background calibration workflow. |
| `microscopy/img_pixel_intensity` | Standard image-HDF5 creation. |
| `microscopy/img_pixel_micro_time` | Mean micro-time map workflow. |
| `microscopy/img_pixel_nb` | Number-and-brightness map workflow. |
| `misc/breakout_game` | Optional/demo status and why it ships. |
| `modelling/structure_tools` | Aggregator shell and included modelling tools. |
| `tttr/tttr_toolbox` | Aggregator shell and included TTTR tools. |

# Existing Strong References

Use these as models, but normalize them to the standard format over time:

| Plugin | Strength |
| --- | --- |
| `burst/burst_selection` | Best reference for API/backend/CLI/GUI/MFDB layering; already has `docs/CONTRACT.md`, `docs/REFERENCE_IMPLEMENTATION.md`, and `docs/STATUS.md`. |
| `modelling/fret` | Good domain README plus CLI/API docs. Needs MFDB/provenance and verification sections to match the standard. |
| `fluorescence_decay/maxent_decay` | Has a docs folder and layered structure; should be checked against the new README order. |
| `pch` | Existing README plus GUI/CLI/service split; good candidate for a quick standardization pass. |

# Documentation Patch Pattern

For each plugin:

1. Read `manifest.json`, `README.md` if present, and entrypoint files.
2. Fill the standard README sections from code evidence, not assumptions.
3. If the plugin has RPC/CLI/API surfaces, add `docs/CONTRACT.md` or link an existing one.
4. Add or update verification commands using the plugin's own tests.
5. Do not claim client-server or MFDB provenance unless the code implements it.
