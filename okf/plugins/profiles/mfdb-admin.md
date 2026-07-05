---
type: Plugin Profile
title: MFDB Admin plugin
description: OKF profile for the MFDB administration plugin.
resource: chisurf/plugins/core/mfdb_admin/
tags: [plugins, mfdb, provenance, rpc]
timestamp: '2026-07-05T00:00:00Z'
---

# Identity

| Field | Value |
| --- | --- |
| Plugin id | `mfdb_admin` |
| Display name | `Tools:MFDB Admin` |
| Categories | `Tools`, `Fluorescence`, `Database` |
| Version | `1.1.0` |
| State namespace | `mfdb_admin` |
| Local README | `chisurf/plugins/core/mfdb_admin/README.md` |

The manifest describes this as the administration surface for samples, experiments,
setups, raw/processed data, provenance, project archives, and fluorophore curation.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| GUI | `gui/tool.py`, entity docks, lifecycle/protocol/study/reagent/calibration views, optical-component forms. |
| Backend services | `backend/services.py` plus auth, password, measurement, ndxplorer, and fluorophore service modules. |
| API placeholder | `api/` exists but no substantial public contract is visible from the file inventory. |
| CLI placeholder | `cli/` exists but the manifest exposes GUI and services, not a CLI command. |
| Tests | `test/test_*_handlers.py`, `test/test_*_view.py`, navigation, session SSO, and optical-component tests. GUI behavior coverage is tracked in `docs/GUI_TEST_COVERAGE.md`. |

The manifest lists 126 `mfdb.*` RPC methods. `backend/services.py` defines
`register_services` and many handler functions for sample, protocol, study,
reagent, calibration, lifecycle, project, object, and dataset operations.

UI direction: MFDB Admin should use JSON view specs and AutoForm for ordinary
forms, detail panels, tables, wizard steps, and toolbars. When AutoForm lacks a
needed primitive, implement it in the shared AutoForm/dataspec layer first rather
than adding another one-off MFDB Qt widget.

# Data And Provenance Impact

This is a P0 documentation target because it is the broadest MFDB mutation and
inspection surface. It can create, edit, delete, browse, and validate database
entities used by provenance-aware workflows. Its docs must make clear which handlers
are mutating, which are read-only, and which operations are fail-loud versus
best-effort.

# Verification Surface

Focused test command:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/core/mfdb_admin/test
```

Current full-suite status in the arm64 Qt environment is not green: 84 passed,
3 failed, 6 errors. The failures/errors are in structured sample handler tests
that import `chisurf.core.mfdb.orm` and require `sqlalchemy`.

GUI coverage status:

- Headless interaction tested: workflow views for studies, protocols, lifecycle,
  calibrations, reagent lots, pipelines, the connection dialog, and core
  EntityDock paths for Samples, Experiments, Users, Objects, Projects, and
  Branches.
- Partially tested: spectra/optical components.
- Construction-only and behavior-untested: aggregate panels, sample metadata,
  provenance graph, import/export, measurements, and remaining generic
  `EntityDock` panels unless a narrower test says otherwise.

Construction-only entries are not done. Manual testing has shown failures in GUI
elements, so `docs/GUI_TEST_COVERAGE.md` is the source of truth for what must be
treated as untested.

# Documentation Work

- Keep `chisurf/plugins/core/mfdb_admin/README.md` current as the user-facing overview.
- Extend `docs/CONTRACT.md` from method-family coverage to per-method schemas for high-risk mutations.
- Add `docs/WORKFLOWS.md` for common admin tasks: browse sample, edit setup, inspect provenance, import/export project.
- Keep `docs/STATUS.md` synchronized with manifest/service drift findings.
- Keep `docs/GUI_TEST_COVERAGE.md` synchronized with real headless GUI tests.
- Explicitly document destructive actions, auth/session assumptions, and user-safety constraints.
- Continue migrating manual GUI panels to AutoForm JSON specs, starting with simple
  browse/list panels that fit the shared table primitive.
