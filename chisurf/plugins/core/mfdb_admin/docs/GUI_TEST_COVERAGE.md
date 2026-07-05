# MFDB Admin GUI Test Coverage

Last reviewed: 2026-07-05.

This file is the conservative truth table for MFDB Admin GUI confidence. If a
surface is only constructed in a headless test, it is not marked done for behavior.
Manual testing has shown that multiple elements can fail despite passing
construction tests, so construction-only entries must be treated as untested until
there is a headless interaction test with sample data.

## Coverage Levels

| Level | Meaning |
| --- | --- |
| Headless interaction tested | A test constructs the widget, drives at least one user-visible action, and asserts UI/service state. |
| Construction tested | A test builds the widget/panel without raising. This does not prove the controls work. |
| Untested | No meaningful GUI test evidence. Treat as not done. |

## Shell And Aggregate Panels

| Surface | Current evidence | Status |
| --- | --- | --- |
| MFDB Admin navigation shell | `test_admin_navigation.py` verifies `MFDBWidget` is a `NavigationPanelTool`, exposes nav/stack widgets, and lists expected panels. | Construction tested. |
| Overview | Built through `test_every_panel_builds`. | Untested for behavior. |
| All items | Built through `test_every_panel_builds`. | Untested for behavior. |
| Measurements | Built through `test_every_panel_builds`. | Untested for behavior. |
| Provenance Graph | Built through `test_every_panel_builds`. | Untested for behavior. |
| Import / Export | Built through `test_every_panel_builds`. | Untested for behavior. |

## Generic Entity Panels

These panels use `EntityDock`. `test_every_panel_builds` constructs each panel and
`test_entity_dock_uses_autoform` verifies the sample panel uses `EntityForm`.
That does not prove list refresh, row selection, create, edit, auto-save, delete,
extra actions, FK jumps, or validation.

| Entity panel | Current status |
| --- | --- |
| Samples | Construction tested; behavior untested. |
| Sample Conditions | Construction tested; behavior untested. |
| Entities | Construction tested; behavior untested. |
| Probes | Construction tested; behavior untested. |
| Label Positions | Construction tested; behavior untested. |
| FRET Pairs | Construction tested; behavior untested. |
| Experiments | Construction tested; behavior untested. |
| Experiment Types | Construction tested; behavior untested. |
| Setups | Construction tested; behavior untested. |
| Detector Channels | Construction tested; behavior untested. |
| PIE Windows | Construction tested; behavior untested. |
| FCS Pairs | Construction tested; behavior untested. |
| Devices | Construction tested; behavior untested. |
| Raw Data | Construction tested; behavior untested. |
| Processing Runs | Construction tested; behavior untested. |
| Processed Products | Construction tested; behavior untested. |
| Analyses | Construction tested; behavior untested. |
| Objects | Construction tested; behavior untested. |
| Projects | Construction tested; behavior untested. |
| Branches | Construction tested; behavior untested. |
| Users | Construction tested; behavior untested. |

## Dedicated Panels

| Surface | Current evidence | Status |
| --- | --- | --- |
| Sample Metadata | Built through `test_every_panel_builds`. | Untested for behavior. |
| Spectra / optical components | `test_fluorophore_panel_and_rpc_integrated` constructs the integrated dock and verifies `fluorophores.list`; `test_optical_components.py` covers spectrum display widgets. | Partially tested; full curation workflows untested. |
| Studies | `test_studies_view.py` lists studies, selects fields/members, and creates a study. | Headless interaction tested. |
| Protocols | `test_protocols_view.py` lists latest protocols, selects versions/schema, and creates a protocol. | Headless interaction tested. |
| Lifecycle | `test_lifecycle_view.py` loads definitions, refreshes state/next transitions, applies a transition, and surfaces illegal transitions. | Headless interaction tested. |
| Calibrations | `test_calibrations_view.py` lists calibrations, creates a calibration, rejects non-numeric values, and shows stale uses. | Headless interaction tested. |
| Reagent Lots | `test_reagents_view.py` lists lots, filters kind/expired state, selects detail, creates a lot, and rejects blank names. | Headless interaction tested. |
| Pipelines | `test_pipelines_view.py` lists pipelines and shows structure/runs on selection. | Headless interaction tested. |
| Connection/Auth dialog | `test_connection_dialog.py` verifies password masking, collapsible advanced options, and value round-trip. | Headless interaction tested. |

## Required Next GUI Tests

Priority order:

1. Add sample-data-backed interaction tests for `Samples`, `Experiments`, `Raw Data`,
   `Processed Products`, `Objects`, `Projects`, and `Users`.
2. Add tests for `Import / Export`, `Measurements`, and `Provenance Graph`, because
   those are workflow surfaces where construction-only coverage is especially weak.
3. For each generic `EntityDock`, drive row selection and at least one supported
   mutation or explicit read-only action.
4. Keep every new ordinary form/table test on AutoForm/JSON view specs where
   possible. If AutoForm lacks a needed primitive, add it to the shared AutoForm
   layer before adding MFDB-specific widget code.

## Current Smoke Command

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  test/gui/test_autoform_table_section.py \
  chisurf/plugins/core/mfdb_admin/test/test_admin_navigation.py \
  chisurf/plugins/core/mfdb_admin/test/test_autoform_entity_form.py \
  chisurf/plugins/core/mfdb_admin/test/test_connection_dialog.py \
  chisurf/plugins/core/mfdb_admin/test/test_session_sso.py \
  chisurf/plugins/core/database_connector/test \
  chisurf/plugins/core/project_browser/test
```

Last focused run on 2026-07-05 in the arm64 Qt environment: 23 passed, 2 warnings.

The full `chisurf/plugins/core/mfdb_admin/test` suite is not green in that
environment: 80 passed, 3 failed, 6 errors. The failures/errors are structured
sample handler paths requiring `sqlalchemy`, not additional GUI interaction
evidence.
