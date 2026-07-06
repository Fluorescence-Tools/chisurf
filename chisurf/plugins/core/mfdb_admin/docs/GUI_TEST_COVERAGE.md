# MFDB Admin GUI Test Coverage

Last reviewed: 2026-07-06.

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
| Measurements | `test_measurements_panel_refreshes_and_filters_seeded_data` refreshes the panel from seeded raw data, processing-run, and processed-product records, then exercises kind/search filters. | Headless interaction tested. |
| Provenance Graph | `test_provenance_graph_loads_seeded_processing_graph` loads a seeded raw -> processing -> processed provenance graph, verifies edge-table rows, node-editor graph contents, and edge-details selection. | Headless interaction tested. |
| Import / Export | `test_import_export_panel_validates_previews_and_exports_seeded_sample` validates a seeded sample, previews flrCIF text, confirms export despite validation warnings, writes sample CIF, exports the sample table, and imports a minimal CIF fixture through the file field. | Headless interaction tested. |

## Generic Entity Panels

These panels use `EntityDock`. `test_every_panel_builds` constructs each panel and
`test_entity_dock_uses_autoform` verifies the sample panel uses `EntityForm`.
That does not prove list refresh, row selection, create, edit, auto-save, delete,
extra actions, FK jumps, or validation.

| Entity panel | Current status |
| --- | --- |
| Samples | Headless interaction tested for sample-data-backed refresh, row selection, AutoForm load, and auto-save of `description` in `test_entity_dock_gui_interactions.py`. Delete, FK jumps, and structured-sample workflows remain untested. |
| Sample Conditions | Headless interaction tested for refresh, row selection, AutoForm load, New-button creation, auto-save of `ph`, checked-row delete, and delete confirmation in `test_entity_dock_gui_interactions.py`. Sample-link workflows remain untested. |
| Entities | Headless interaction tested for sample-data-backed refresh, row selection, and AutoForm load in `test_entity_dock_gui_interactions.py`. Create/edit/delete workflows remain untested. |
| Probes | Headless interaction tested for sample-data-backed refresh, row selection, AutoForm load, and auto-save of `description` in `test_entity_dock_gui_interactions.py`. Optical-property editing, create, and delete workflows remain untested. |
| Label Positions | Headless interaction tested for sample-data-backed refresh, row selection, and AutoForm load in `test_entity_dock_gui_interactions.py`. Create/edit/delete workflows remain untested. |
| FRET Pairs | Headless interaction tested for sample-data-backed refresh, row selection, and AutoForm load in `test_entity_dock_gui_interactions.py`. Create/edit/delete workflows remain untested. |
| Experiments | Headless interaction tested for sample-data-backed refresh, row selection, AutoForm load, and auto-save of `status` in `test_entity_dock_gui_interactions.py`. Delete, FK jumps, and experiment data subflows remain untested. |
| Experiment Types | Headless interaction tested for refresh, row selection, AutoForm load, New-button creation, auto-save of `description`, checked-row delete, and delete confirmation in `test_entity_dock_gui_interactions.py`. Experiment-type linkage workflows remain untested. |
| Setups | Headless interaction tested for refresh, row selection, AutoForm load, and auto-save of `details` in `test_entity_dock_gui_interactions.py`. Validation and delete workflows remain untested. |
| Detector Channels | Headless interaction tested for setup-data-backed refresh, row selection, and AutoForm load in `test_entity_dock_gui_interactions.py`. Create/edit/delete workflows remain untested. |
| PIE Windows | Headless interaction tested for setup-data-backed refresh, row selection, and AutoForm load in `test_entity_dock_gui_interactions.py`. Create/edit/delete workflows remain untested. |
| FCS Pairs | Headless interaction tested for setup-data-backed refresh, row selection, and AutoForm load in `test_entity_dock_gui_interactions.py`. Create/edit/delete workflows remain untested. |
| Devices | Headless interaction tested for refresh, row selection, AutoForm load, New-button creation, auto-save of `model`/`name`, checked-row delete, and delete confirmation in `test_entity_dock_gui_interactions.py`. Device linkage workflows remain untested. |
| Raw Data | Headless interaction tested for raw-artifact-backed refresh, row selection, AutoForm load, Copy ID, Reveal/Open, provenance-seed action, validation-status update, confirmed soft delete, table refresh, and operation-link soft delete through the visible EntityDock in `test_entity_dock_gui_interactions.py`. File-content parsing/preview remains untested. |
| Processing Runs | Headless interaction tested for operation-backed refresh, row selection, and AutoForm load in `test_entity_dock_gui_interactions.py`. Run detail/provenance workflows remain untested. |
| Processed Products | Headless interaction tested for processed-artifact-backed refresh, row selection, AutoForm load, Copy ID, Reveal/Open, provenance-seed action, validation-status update, confirmed soft delete, table refresh, and operation-link soft delete through the visible EntityDock in `test_entity_dock_gui_interactions.py`. File-content parsing/preview remains untested. |
| Analyses | Headless interaction tested for analysis-operation-backed refresh, row selection, AutoForm load, Copy ID, Details drilldown with parameter and output-product tables, and provenance-seed action through the visible EntityDock in `test_entity_dock_gui_interactions.py`. Non-empty input-product and grouped-fit drilldowns remain untested. |
| Objects | Headless interaction tested for object-store-backed refresh, row selection, AutoForm load, Copy UUID, Reveal, and confirmed delete through the visible Objects EntityDock in `test_entity_dock_gui_interactions.py`. Direct blob-content download/open remains untested. |
| Projects | Headless interaction tested for project-operation-backed refresh, row selection, and AutoForm load in `test_entity_dock_gui_interactions.py`. Restore/export/version workflows remain covered by Project Browser tests, not this admin panel. |
| Branches | Headless interaction tested for branch refresh, row selection, AutoForm load, New-button creation, auto-save of `description`, checked-row delete, and delete confirmation in `test_entity_dock_gui_interactions.py`. Branch fork/time-travel workflows remain untested. |
| Users | Headless interaction tested for refresh, row selection, AutoForm load, New-button creation, auto-save of `display_name`, rename through editable `user_id`, checked-row delete, delete confirmation, and built-in `user_default` delete protection in `test_entity_dock_gui_interactions.py`. Password change and force-delete workflows remain untested in GUI. |

## Dedicated Panels

| Surface | Current evidence | Status |
| --- | --- | --- |
| Sample Metadata | `test_sample_metadata_dock_loads_edits_and_saves_metadata` loads a seeded sample, edits a metadata row through the detail form, saves, and verifies MFDB state. | Headless interaction tested. |
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

1. Add tests for create/delete paths on writable EntityDock panels, especially
   user password/force-delete flows and non-branch entity mutation paths.
2. Add tests for analysis non-empty input-product/grouped-fit drilldowns and
   raw/processed file-content parsing or preview paths.
3. Add deeper import tests for FLR-rich CIF/mmCIF files with entities, probes,
   positions, and analyses.
4. For each generic `EntityDock`, drive row selection and at least one supported
   mutation or explicit read-only action.
5. Keep every new ordinary form/table test on AutoForm/JSON view specs where
   possible. If AutoForm lacks a needed primitive, add it to the shared AutoForm
   layer before adding MFDB-specific widget code.

## Current Smoke Command

```bash
PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  test/core/test_mfdb_vendor_package.py \
  test/gui/test_autoform_table_section.py \
  chisurf/plugins/core/mfdb_admin/test/test_admin_navigation.py \
  chisurf/plugins/core/mfdb_admin/test/test_autoform_entity_form.py \
  chisurf/plugins/core/mfdb_admin/test/test_connection_dialog.py \
  chisurf/plugins/core/mfdb_admin/test/test_entity_dock_gui_interactions.py \
  chisurf/plugins/core/mfdb_admin/test/test_session_sso.py \
  chisurf/plugins/core/database_connector/test \
  chisurf/plugins/core/project_browser/test
```

Last focused run on 2026-07-06 in the arm64 Qt environment: 58 passed, 2 warnings.

The full `chisurf/plugins/core/mfdb_admin/test` suite is green in that
environment on 2026-07-06: 111 passed, 2 warnings.
