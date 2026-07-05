---
type: Plugin Documentation Standard
title: Plugin documentation standard
description: Required format for documenting individual ChiSurf plugins.
resource: chisurf/plugins/
tags: [plugins, documentation, manifest]
timestamp: '2026-07-05T00:00:00Z'
---

# Purpose

Each first-class plugin should have a local `README.md` that explains what the plugin
does, how users run it, and how maintainers verify or extend it. Group-level OKF pages
summarize domains; the plugin README is the source closest to the code.

# Required files

| File | Required when | Purpose |
| --- | --- | --- |
| `manifest.json` | Every discoverable plugin | Machine-readable identity, menu placement, entrypoints, RPC methods, state namespace, dependencies. |
| `README.md` | Every discoverable plugin | Human-readable plugin contract and workflow. |
| `docs/` | Complex plugins only | Detailed guides, contracts, screenshots, API/CLI references, migration notes. |
| `docs/STATUS.md` | Long-lived migrations/workflows | Current state, known gaps, verification evidence. |
| `docs/CONTRACT.md` | Plugins with API/RPC/CLI surfaces | Stable request/result/event contract. |

# README Format

Use this section order for every plugin README:

```markdown
# <Plugin Display Name>

Short one-paragraph summary: what the plugin does and who uses it.

## Status

| Field | Value |
| --- | --- |
| Plugin id | `<manifest id>` |
| Menu path | `<manifest display_name>` |
| Category | `<manifest categories>` |
| Maturity | `stable` / `active` / `experimental` / `deprecated` |
| Architecture | `legacy-qt` / `layered` / `client-server` / `declarative-ui` |
| MFDB | `none` / `reads` / `writes` / `full provenance` |

## User Workflows

1. Primary workflow.
2. Secondary workflow.
3. Batch/headless workflow, if present.

## Inputs And Outputs

| Kind | Formats | Notes |
| --- | --- | --- |
| Input | `.ptu`, `.spc`, `.csv`, ... | Required assumptions. |
| Output | `.bur`, `.h5`, MFDB artifact, ... | Where results are written. |

## UI Surface

Where it appears, key panels/actions, drag-drop behavior, and settings that persist.

## API, CLI, And RPC

| Surface | Entry point / method | Purpose |
| --- | --- | --- |
| Python API | `...` | Pure call or DTO contract. |
| CLI | `...` | Command and main options. |
| RPC | `plugin.method` | JSON-safe service call. |

## Architecture

Map the plugin's layers:
- `api/`: DTOs, serialization, public contract.
- `core/`: pure computation and IO helpers without Qt/server imports.
- `backend/`, `server/`, or `rpc/`: JSON-RPC adapters and service registration.
- `cli/`: command-line wrapper.
- `gui/`: Qt or AutoForm view only.
- `tests/` or `test/`: behavior and smoke tests.

State any known rule breaks explicitly.

## MFDB And Provenance

Say whether the plugin reads/writes MFDB, which artifact kinds it creates, which
operation type it records, and whether registration is best-effort or fail-loud.

## Verification

List the focused tests and any manual checks:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest <tests>
```

## Limitations And Open Work

Short bullets only. Link PRDs or TODOs instead of writing long plans inline.

## Related Files

- `manifest.json`
- `api/...`
- `backend/...`
- `gui/...`
- `tests/...`
```

# Optional docs folder format

Use `docs/` when a README would become too long. Keep filenames predictable:

| File | Contents |
| --- | --- |
| `docs/CONTRACT.md` | DTOs, JSON schemas, RPC methods, events, state patches. |
| `docs/CLI.md` | Commands, examples, exit behavior. |
| `docs/API.md` | Python API or service API reference. |
| `docs/WORKFLOWS.md` | User workflows with input/output examples. |
| `docs/STATUS.md` | Completed work, remaining gaps, verification. |
| `docs/MIGRATION.md` | Legacy-to-standard migration notes. |

# Documentation maturity labels

| Label | Meaning |
| --- | --- |
| `none` | No README and no docs folder. |
| `stub` | README exists but only gives a short feature list or launch note. |
| `usable` | README covers workflows, inputs/outputs, and tests. |
| `contracted` | README plus API/RPC/CLI contract docs for nontrivial surfaces. |
| `reference` | Complete docs plus current status and migration/extension guidance. |

# Minimum Acceptance

A plugin documentation update is complete when:

- `README.md` follows the required section order.
- The README matches `manifest.json` id, display name, entrypoints, and RPC method names.
- Inputs, outputs, and persistence side effects are explicit.
- Tests or manual verification commands are listed.
- Known gaps are stated without claiming completion.
