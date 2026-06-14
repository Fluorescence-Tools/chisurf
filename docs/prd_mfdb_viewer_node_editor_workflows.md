# PRD: MFDB Viewer + Future-Ready Node Editor Workflows

## Summary

Enhance the MFDB viewer so users can inspect the full experiment data chain:
experiments -> raw TTTR files -> processing runs such as Burst Selection ->
processed products such as `.bur`, decays, FCS, PDA, HDF5/ZIP outputs ->
analysis/project records -> provenance dependencies.

The provenance graph must be displayed with the existing ChiSurf node editor.
While touching the node editor, make it reusable for future user-defined
workflows by separating demo behavior from embeddable editor infrastructure,
preserving stable graph metadata, and adding non-breaking workflow extension
points.

## Hard Requirements

- [x] Keep this PRD at `docs/prd_mfdb_viewer_node_editor_workflows.md`.
- [x] Mark checklist items as implementation progresses.
- [x] Do not add a new graph-rendering dependency.
- [x] Do not add an MFDB schema migration for this task.
- [ ] Do not break existing node editor examples or default behavior.
      Follow-up review found a default `NodeEditorWidget` layout regression
      after introducing `NodeViewerWidget`.
- [x] Do not remove `sample_database` compatibility wrappers.
- [x] Final implementation report must reference this PRD and list completed
      and remaining checklist items.

## Current Facts To Respect

- [x] MFDB viewer lives mainly in `chisurf/plugins/core/mfdb_admin/gui/tool.py`.
- [x] MFDB client lives in `chisurf/plugins/core/mfdb_admin/gui/client.py`.
- [x] `MFDBClient` production default uses `ZmqClient`; `inprocess=True` is explicit.
- [x] `SampleDatabaseClient` remains a legacy in-process compatibility client.
- [x] MFDB GUI no longer imports `FluorescenceDatabase` or `resolve_database_path`
      directly in `chisurf/plugins/core/mfdb_admin/gui/tool.py`.
- [x] Provenance/data RPC handlers live in
      `chisurf/plugins/core/mfdb_admin/backend/measurement_services.py`.
- [x] Node editor widget lives in `chisurf/gui/widgets/node_editor/editor.py`.
- [x] Node scene serialization lives in `chisurf/gui/widgets/node_editor/scene.py`.
- [x] Node type registry lives in `chisurf/gui/widgets/node_editor/registry.py`.
- [x] Headless node graph model lives in `chisurf/gui/widgets/node_editor/graph.py`.
- [x] Existing MFDB UI bug: `raw_data.list` returns `raw_data`, but the UI
      reads `raw_datasets`.
- [x] Existing MFDB UI bug: `processed_data.list` returns `processed_data`,
      but the UI reads `processed_datasets`.
- [x] Existing MFDB UI bug: provenance upstream/downstream handlers require
      `node_type` and `node_id`, but the UI sends only `seed_id`.
- [x] Existing node editor behavior: `NodeEditorWidget()` always builds the
      example graph and side panel.
- [x] Existing node editor bug: `NodeScene.from_dict()` drops loaded node IDs
      because it does not pass `id=node_id` into `NodeModel`.
- [x] Existing node editor detail: edge port indexes are combined
      `port_items` indexes, not separate input/output indexes. For MFDB
      adapter nodes with one input and one output, provenance edges must use
      `source_port=1` and `target_port=0`.

## Product Requirements

- [x] The MFDB viewer shows experiment-linked raw data, processing runs,
      processed products, analyses, and provenance.
- [x] Users can select a sample and see related experiments.
- [x] Users can select an experiment and see related raw data, processing
      runs, processed products, analyses, and provenance.
- [x] Raw data table shows TTTR-like inputs, including `.spc`, `.ptu`, `.bh`,
      `.ht3`, photon HDF5, checksum, validation status, storage mode, and
      path/URL/folder.
- [x] Processing runs table shows Burst Selection and other registered runs
      with status, settings, selected setup, detectors, windows, counts, and
      errors.
- [x] Processed products table shows product type and location for `.bur`,
      decay, FCS, PDA, IRF, HDF5, ZIP, GMM summary, spectra, and fit results.
- [x] Selecting raw data, a processing run, processed data, or analysis can
      seed a provenance trace.
- [x] Provenance tab shows an edge table, JSON/details panel, and embedded node
      editor graph.
- [x] Provenance graph is read-only in the MFDB viewer.
- [x] The same node editor infrastructure remains suitable for future editable
      user-defined workflows.

## Node Editor Implementation Plan

### 1. Make `NodeEditorWidget` reusable instead of demo-only

Edit `chisurf/gui/widgets/node_editor/editor.py`.

- [x] Add backward-compatible constructor kwargs:
  - [x] `build_example: bool = True`
  - [x] `show_side_panel: bool = True`
  - [x] `show_timeline: bool = True`
  - [x] `read_only: bool = False`
  - [x] `graph_purpose: str = "example"`
- [ ] Keep current behavior when no args are passed.
      Current follow-up review found the default side-panel/editor layout is
      broken by a double top-level layout installation.
- [x] Only call `_build_example_graph()` when `build_example=True`.
- [x] Only create the palette/JSON side panel when `show_side_panel=True`.
- [x] Only create timeline controls when `show_timeline=True`.
- [x] Pass `read_only` into `NodeScene`.
- [x] Store `self.graph_purpose`.
- [x] Add `load_graph_dict(self, data: dict[str, Any]) -> None`.
- [x] `load_graph_dict` must call `self.scene.from_dict(data)` and then
      `self.fit_graph()`.
- [x] Add `graph_dict(self) -> dict[str, Any]` as a thin wrapper around
      `self.scene.to_dict()`.
- [x] Add `fit_graph(self) -> None` that calls `self.view.fit_all()`.
- [x] Add `nodeSelected = QtCore.Signal(dict)`.
- [x] Add `edgeSelected = QtCore.Signal(dict)`.
- [x] Emit `nodeSelected` with the selected node model config when node
      selection changes.
- [x] Emit `edgeSelected` with selected edge config when edge selection
      changes.
- [ ] Keep existing JSON load/save buttons working when `show_side_panel=True`.
      Verify visually and with a layout regression test after fixing the
      `NodeEditorWidget` / `NodeViewerWidget` inheritance layout.

### 2. Add read-only interaction mode

Edit `chisurf/gui/widgets/node_editor/scene.py`.

- [x] Add `read_only: bool = False` to `NodeScene.__init__`.
- [x] Store `self.read_only`.
- [x] In read-only mode, block edge creation by port drag.
- [x] In read-only mode, block Delete key deletion.
- [x] In read-only mode, block paste.
- [x] In read-only mode, block duplicate.
- [x] In read-only mode, block add-node shortcuts.
- [x] In read-only mode, block node movement.
- [x] In read-only mode, hide or disable context-menu mutation actions:
  - [x] Add node
  - [x] Delete node
  - [x] Duplicate node
  - [x] Delete connection
  - [x] Align
  - [x] Undo
  - [x] Redo
- [x] In read-only mode, keep context-menu view actions:
  - [x] Fit to view
  - [x] Zoom in
  - [x] Zoom out
  - [x] Reset zoom
  - [x] Help
- [x] Make read-only node movement impossible by clearing `ItemIsMovable` on
      loaded nodes when `scene.read_only=True`.
- [x] Do not remove node or edge selection flags in read-only mode.
- [x] Keep panning and zooming available in read-only mode.

### 3. Preserve graph identity and metadata

Edit `chisurf/gui/widgets/node_editor/scene.py` and
`chisurf/gui/widgets/node_editor/edge_item.py`.

- [x] In `NodeScene.from_dict()`, construct `NodeModel(..., id=node_id)`.
- [x] Preserve top-level `meta` from loaded graph on the scene.
- [x] Include top-level `meta` in `to_dict()` if present.
- [x] Preserve unknown node `config` keys exactly.
- [x] Preserve unknown edge `config` keys exactly.
- [x] Keep existing edge color override behavior.
- [x] Keep schema version `1`.
- [x] Add optional graph metadata fields, but never require them:
  - [x] `meta.purpose`: `"example" | "provenance_view" | "workflow"`
  - [x] `meta.workflow_id`
  - [x] `meta.created_by`
  - [x] `meta.schema_name`
  - [x] `meta.read_only`

### 4. Add future workflow extension points

Edit `chisurf/gui/widgets/node_editor/registry.py`.

- [x] Extend `NodeType` with optional fields:
  - [x] `description: str = ""`
  - [x] `tags: list[str] = field(default_factory=list)`
  - [x] `config_schema: dict | None = None`
  - [x] `runtime: str | None = None`
  - [x] `operation: str | None = None`
  - [x] `executor: Callable | None = None`
- [x] Keep all new fields optional so existing registrations still work.
- [x] Add `NodeRegistry.unregister(type_id: str) -> None`.
- [x] Add `NodeRegistry.replace(node_type: NodeType) -> None`.
- [x] Add `NodeRegistry.by_category() -> dict[str, list[NodeType]]`.
- [x] Add `NodeRegistry.workflow_types() -> list[NodeType]`, returning nodes
      with `runtime` or `executor`.
- [x] Do not implement a full workflow runtime in this task.
- [x] Do make graph data rich enough that a future runtime can execute it.

### 5. Add headless graph helpers for workflows

Edit `chisurf/gui/widgets/node_editor/graph.py`.

- [x] Align `GraphDef.to_dict()` with current scene schema:
  - [x] use `pos: [x, y]`, not separate `x`/`y`
  - [x] include `collapsed`
  - [x] include `version`
  - [x] include optional `meta`
- [x] Add `GraphDef.from_scene_dict(data)`.
- [x] Add `GraphDef.to_scene_dict()`.
- [x] Add `topological_node_ids() -> list[str]`.
- [x] Add `validate_acyclic() -> bool`.
- [x] Add `incoming_edges(node_id)`.
- [x] Add `outgoing_edges(node_id)`.
- [x] Add tests proving helpers work without Qt.
- [x] Keep this runtime-neutral; no plugin-specific execution here.

## MFDB Provenance Graph Adapter

Create `chisurf/plugins/core/mfdb_admin/gui/provenance_graph.py`.

### Public API

- [x] Add `mfdb_graph_to_node_editor_graph(graph: dict[str, Any]) -> dict[str, Any]`.
- [x] Add `node_key(node_type: str, node_id: str) -> str`.
- [x] Add `record_title(node: dict[str, Any]) -> str`.
- [x] Add `record_kind(node: dict[str, Any]) -> str`.
- [x] Add `layout_nodes(nodes, edges) -> dict[str, tuple[float, float]]`.

### Input shape

Adapter input is the result of `provenance.graph.export`.

- [x] `graph["nodes"]` contains records with at least `node_type` and `node_id`.
- [x] `graph["edges"]` contains records with:
  - [x] `source_node_type`
  - [x] `source_node_id`
  - [x] `target_node_type`
  - [x] `target_node_id`
  - [x] `relationship_type`
  - [x] optional `edge_id`
  - [x] optional `metadata`

### Output shape

- [x] Top-level output includes:
  - [x] `version: 1`
  - [x] `meta.purpose: "provenance_view"`
  - [x] `meta.schema_name: "mfdb.provenance.node_editor.v1"`
  - [x] `nodes`
  - [x] `edges`
- [x] Node IDs are stable: `f"{node_type}:{node_id}"`.
- [x] Each MFDB node uses:
  - [x] `type: "mfdb_record"`
  - [x] one input port: `{"name": "in", "type": "mfdb"}`
  - [x] one output port: `{"name": "out", "type": "mfdb"}`
  - [x] `config.record` containing the original MFDB node
  - [x] `config.node_type`
  - [x] `config.node_id`
  - [x] `config.workflow_runtime: None`
  - [x] `collapsed: False`
- [x] Each MFDB edge uses:
  - [x] `source = f"{source_node_type}:{source_node_id}"`
  - [x] `target = f"{target_node_type}:{target_node_id}"`
  - [x] `source_port = 1`
  - [x] `target_port = 0`
  - [x] `config.edge_id`
  - [x] `config.relationship_type`
  - [x] `config.metadata`
- [x] Important implementation detail: use `source_port=1` because node editor
      ports are stored as combined input+output items.
- [x] Skip malformed edges whose source or target node is missing.
- [x] Do not crash on an empty graph; return valid empty node editor graph.

### Layout rules

- [x] Do not require `networkx`.
- [x] Compute left-to-right dependency levels from directed edges.
- [x] Unconnected nodes go to level 0.
- [x] Position columns:
  - [x] `x = level * 260`
  - [x] `y = row_index * 130`
- [x] Sort nodes deterministically by `(level, kind, title, id)`.

### Visual conventions

- [x] Color edge config by relation:
  - [x] `input_to`: blue
  - [x] `produced`: green
  - [x] `parameter_of`: yellow
  - [x] `derived_from`: gray
  - [x] unknown: light gray
- [x] Node titles:
  - [x] raw data: `Raw: <data_type or suffix>`
  - [x] processing run: `Process: <processing_type>`
  - [x] analysis run: `Analysis: <analysis_type or model_name>`
  - [x] processed data: `Product: <product_type>`
  - [x] parameter: `Parameter: <name>`
  - [x] fallback: `<node_type>: <node_id>`

## MFDB Client Changes

Edit `chisurf/plugins/core/mfdb_admin/gui/client.py`.

- [x] Add typed helper `list_raw_data(experiment_id=None, data_type=None)`.
- [x] Add typed helper `get_raw_data(raw_data_id)`.
- [x] Add typed helper `list_processing_runs(experiment_id=None, status=None)`.
- [x] Add typed helper `get_processing_run(processing_id)`.
- [x] Add typed helper `list_processed_data(processing_id=None, product_type=None)`.
- [x] Add typed helper `get_processed_data(processed_data_id)`.
- [x] Add typed helper `list_analysis_runs(experiment_id=None, analysis_type=None)`.
- [x] Add typed helper `get_analysis_run(analysis_id)`.
- [x] Add typed helper `dependencies_upstream(node_type, node_id)`.
- [x] Add typed helper `dependencies_downstream(node_type, node_id)`.
- [x] Add typed helper `list_provenance_edges(**filters)`.
- [x] Add typed helper `export_provenance_graph(seed_node_type, seed_node_id, target_path=None)`.
- [x] Keep `_call()` unchanged for compatibility.

## MFDB Viewer Changes

Edit `chisurf/plugins/core/mfdb_admin/gui/tool.py`.

### Data table fixes

- [x] Fix raw data list key:
  - [x] use `raw_data`
  - [x] keep fallback `raw_datasets`
- [x] Fix processed data list key:
  - [x] use `processed_data`
  - [x] keep fallback `processed_datasets`
- [x] Fix provenance calls:
  - [x] send `node_type`
  - [x] send `node_id`
  - [x] stop sending only `seed_id`

### Selection state

- [x] Add state fields:
  - [x] `current_sample_id`
  - [x] `current_experiment_id`
  - [x] `current_provenance_seed_type`
  - [x] `current_provenance_seed_id`
- [x] Selecting sample refreshes experiments.
- [x] Selecting experiment refreshes raw data, processing runs, processed
      products, and analyses.
- [x] Add scope controls where useful:
  - [x] All
  - [x] Current sample
  - [x] Current experiment

### Raw data tab

- [x] Show columns:
  - [x] raw data id
  - [x] experiment id
  - [x] data type
  - [x] storage mode
  - [x] path/url/folder
  - [x] validation
  - [x] checksum
  - [x] acquired at
- [x] Add JSON details panel.
- [x] Add buttons:
  - [x] Open
  - [x] Copy ID
  - [x] Use as provenance seed
- [x] Seed type is `raw_data`.

### Processing runs tab

- [x] Show columns:
  - [x] processing id
  - [x] experiment id
  - [x] type
  - [x] started at
  - [x] status
  - [x] raw count
  - [x] product count
  - [x] operator
- [x] Details panel shows full processing run JSON.
- [x] Add buttons:
  - [x] Copy ID
  - [x] Use as provenance seed
- [x] Seed type is `processing_run`.

### Processed products tab

- [x] Add product type filter:
  - [x] all
  - [x] bur
  - [x] tcspc_decay
  - [x] fcs_correlation
  - [x] pda_histogram
  - [x] irf_curve
  - [x] hdf5
  - [x] zip
  - [x] gmm_summary
  - [x] spectra
  - [x] fit_results
- [x] Show columns:
  - [x] processed data id
  - [x] processing id
  - [x] product type
  - [x] storage mode
  - [x] path/url/folder
  - [x] validation
  - [x] checksum
  - [x] row count
- [x] Details panel shows full processed data JSON.
- [x] Add buttons:
  - [x] Open
  - [x] Open in NDXplorer
  - [x] Copy ID
  - [x] Use as provenance seed
- [x] Seed type is `processed_data`.

### Analyses tab

- [x] Use typed `list_analysis_runs()`.
- [x] Preserve existing project filtering behavior for the project tab.
- [x] Details panel shows full analysis run JSON.
- [x] Add buttons:
  - [x] Copy ID
  - [x] Use as provenance seed
- [x] Seed type is `analysis_run`.

### Provenance controls and graph dock

- [x] Replace existing simple table with separate provenance controls and a
      draggable graph dock.
- [x] Provenance controls contain toolbar fields and export/load actions.
- [x] Provenance graph dock contains a splitter with:
  - [x] edge table
  - [x] embedded node editor
  - [x] JSON/details panel
- [x] Dock right-click context menu is enabled in basic mode.
- [x] Dock context menu exposes hidden dock tabs for reopening.
- [x] Seed/load actions show the provenance graph dock if it was hidden.
- [x] Embed node editor as:
      `NodeEditorWidget(build_example=False, show_side_panel=False, show_timeline=False, read_only=True, graph_purpose="provenance_view")`.
- [x] Toolbar fields:
  - [x] node type combo
  - [x] node id edit
  - [x] Load upstream
  - [x] Load downstream
  - [x] Load full graph
  - [x] Export JSON
  - [x] Export ZIP
- [x] Node type combo values:
  - [x] raw_data
  - [x] processing_run
  - [x] processed_data
  - [x] analysis_run
  - [x] analysis_parameter
- [x] `Load upstream` calls `dependencies_upstream(node_type, node_id)`.
- [x] `Load downstream` calls `dependencies_downstream(node_type, node_id)`.
- [x] `Load full graph` calls `export_provenance_graph`, converts via
      `mfdb_graph_to_node_editor_graph`, loads with `node_editor.load_graph_dict`,
      and calls `node_editor.fit_graph`.
- [x] Edge table shows:
  - [x] edge id
  - [x] source type
  - [x] source id
  - [x] relationship
  - [x] target type
  - [x] target id
  - [x] operation/processing id
- [x] Clicking an edge row shows full edge JSON.
- [x] Connect node editor `nodeSelected` to details panel.
- [x] Connect node editor `edgeSelected` to details panel.

## Backend Changes

Only add backend work if the UI cannot get required fields from existing
services.

- [x] Do not add schema migration.
- [x] Do not rename existing RPCs.
- [x] If needed, add optional `experiment_id` filter to `processed_data.list`;
      otherwise use client-side filtering by processing IDs.
- [x] If needed, add optional `processing_type` filter to processing-run list;
      otherwise leave existing Burst Selection-specific list behavior alone.
- [x] Add condition and probe RPC helpers behind the MFDB client boundary.
- [x] Keep compatibility for `sample_database` wrappers.

## Tests

### Node editor tests

Add or update tests under `chisurf/gui/widgets/node_editor/tests`.

- [x] Default constructor still creates the current example graph.
- [x] `NodeEditorWidget(build_example=False)` starts empty.
- [x] `show_side_panel=False` hides palette/JSON panel.
- [x] `show_timeline=False` hides timeline controls.
- [x] `show_timeline=False` keeps undo/redo state available.
- [x] `read_only=True` prevents edge creation by port drag.
- [x] `read_only=True` prevents Delete from removing selected nodes/edges.
- [x] `read_only=True` blocks paste and duplicate.
- [x] `read_only=True` keeps node selection working.
- [x] `from_dict()` preserves node IDs after `to_dict()`.
- [x] Edge config metadata round-trips.
- [x] Top-level graph `meta` round-trips.
- [x] `load_graph_dict()` loads a valid graph and does not crash.
- [x] `GraphDef` can parse and emit scene schema without Qt.
- [x] `GraphDef.topological_node_ids()` works for a simple workflow DAG.

### MFDB adapter tests

- [x] Empty MFDB graph converts to `{version: 1, nodes: [], edges: []}` with
      provenance metadata.
- [x] Raw -> processing -> product graph converts to 3 nodes and 2 edges.
- [x] Every converted node has exactly one input and one output.
- [x] Every converted edge uses `source_port=1` and `target_port=0`.
- [x] Converted graph passes node editor validation.
- [x] Missing source/target edge is skipped.
- [x] Edge-only dependency responses synthesize endpoint nodes.
- [x] Layout is deterministic.
- [x] Relationship colors are assigned.
- [x] Raw node titles can infer suffix from stored location fields.

### MFDB plugin tests

Add or update tests under `test/plugins`.

- [x] Raw table consumes RPC key `raw_data`.
- [x] Raw table still accepts fallback key `raw_datasets`.
- [x] Processed table consumes RPC key `processed_data`.
- [x] Processed table still accepts fallback key `processed_datasets`.
- [x] Upstream provenance call sends `node_type` and `node_id`.
- [x] Downstream provenance call sends `node_type` and `node_id`.
- [x] Full graph loads through adapter into node editor.
- [x] Selecting raw data sets provenance seed to `raw_data`.
- [x] Selecting processing run sets seed to `processing_run`.
- [x] Selecting processed product sets seed to `processed_data`.
- [x] Selecting analysis sets seed to `analysis_run`.
- [x] Full provenance export responses unwrap the backend `graph` payload.
- [x] Provenance edge helper consumes backend `provenance_edges` key.
- [x] Processed products are experiment-scoped through processing runs.
- [x] Empty-experiment processed-product scoping returns an empty list.
- [x] Processed-product row count reads the top-level `row_count` field.
- [x] URL-backed raw/processed locations open with their original URL scheme.
- [x] `MFDBClient()` constructs a ZMQ client by default and does not import backend services.
- [x] Sample condition save/get services round-trip through the dispatcher.
- [x] Probe list service includes optical properties.

### Integration tests

- [ ] Create temporary MFDB.
- [ ] Add sample.
- [ ] Add experiment.
- [ ] Register raw `.spc` or `.ptu`.
- [ ] Record Burst Selection run with `.bur`, `.fcs`, `.pda`, and `.dec`
      products.
- [ ] Export provenance graph from a processed product.
- [ ] Verify graph contains raw data, processing run, and processed products.
- [ ] Verify converted node editor graph validates.
- [ ] Verify no Qt crash constructing `MFDBWidget` headlessly.

## Manual QA Checklist

- [ ] Open MFDB plugin.
- [ ] Select a sample.
- [ ] Confirm experiments update.
- [ ] Select an experiment.
- [ ] Confirm raw data table updates.
- [ ] Confirm processing runs update.
- [ ] Confirm processed products update.
- [ ] Select a `.bur` product and seed provenance.
- [ ] Click Load full graph.
- [ ] Confirm graph shows raw input -> Burst Selection -> products.
- [ ] Confirm graph is read-only:
  - [ ] cannot drag nodes
  - [ ] cannot create edges
  - [ ] cannot delete nodes
- [ ] Confirm pan, zoom, and fit still work.
- [ ] Export provenance JSON.
- [ ] Export provenance ZIP.
- [ ] Open local raw/product path from table.
- [ ] Open compatible product in NDXplorer.

## Required Verification Commands

Run from `/Users/tpeulen/dev/chisurf`.

- [ ] `python -m pytest chisurf/gui/widgets/node_editor/tests/test_round_trip.py`
- [ ] `python -m pytest chisurf/gui/widgets/node_editor/tests/test_validation.py`
- [ ] `python -m pytest chisurf/gui/widgets/node_editor/tests/test_delete.py chisurf/gui/widgets/node_editor/tests/test_port_types.py`
- [ ] `python -m pytest test/plugins/test_sample_database_plugin.py`
- [ ] `python -m pytest test/fio/test_fdb_provenance.py test/fio/test_fdb_analysis.py`
- [ ] If Qt environment supports it, run the new MFDB widget smoke test.

Arm64 conda verification completed with:

- [x] `QT_QPA_PLATFORM=offscreen /Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest --no-cov chisurf/gui/widgets/node_editor/tests/test_round_trip.py chisurf/gui/widgets/node_editor/tests/test_validation.py chisurf/gui/widgets/node_editor/tests/test_delete.py chisurf/gui/widgets/node_editor/tests/test_port_types.py chisurf/gui/widgets/node_editor/tests/test_read_only.py chisurf/gui/widgets/node_editor/tests/test_graph_headless.py test/plugins/test_sample_database_plugin.py test/fio/test_fdb_provenance.py test/fio/test_fdb_analysis.py test/plugins/test_provenance_graph_adapter.py` passed with `58 passed, 32 warnings`.
- [x] `/Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m ruff check --select F,I,W chisurf/gui/widgets/node_editor/edge_item.py chisurf/gui/widgets/node_editor/editor.py chisurf/gui/widgets/node_editor/graph.py chisurf/gui/widgets/node_editor/registry.py chisurf/gui/widgets/node_editor/scene.py chisurf/plugins/core/mfdb_admin/gui/client.py chisurf/plugins/core/mfdb_admin/gui/provenance_graph.py chisurf/plugins/core/mfdb_admin/gui/tool.py test/plugins/test_provenance_graph_adapter.py test/plugins/test_sample_database_plugin.py chisurf/gui/widgets/node_editor/tests/test_round_trip.py chisurf/gui/widgets/node_editor/tests/test_read_only.py chisurf/gui/widgets/node_editor/tests/test_graph_headless.py` passed.

## Done Definition

- [x] PRD exists at `docs/prd_mfdb_viewer_node_editor_workflows.md`.
- [x] MFDB viewer shows experiments, raw data, processing runs, processed
      products, analyses, and provenance.
- [x] Provenance graph renders in embedded node editor.
- [x] Node editor embedding is read-only and reusable.
- [x] Node editor graph schema remains suitable for future user-defined
      workflows.
- [x] Current MFDB RPC key mismatches are fixed.
- [x] Provenance RPC calls use correct parameters.
- [x] All new tests pass.
- [x] Existing node editor and MFDB plugin tests pass.
- [x] No schema migration.
- [x] No new dependency.

## Assumptions

- [x] This task prepares workflow infrastructure but does not implement a full
      user workflow execution engine.
- [x] Future workflows will use the same node editor schema with
      `meta.purpose="workflow"`.
- [x] MFDB provenance uses `meta.purpose="provenance_view"` and must remain
      read-only.
- [x] Node `config` is the extension point for plugin-specific workflow
      settings.
- [x] `NodeType.runtime`, `NodeType.operation`, and `NodeType.executor` are
      optional future hooks, not required for MFDB provenance display.

---

# Addendum: NodeViewer Abstraction + ZMQ JSON-RPC Boundary

## Review Result: 2026-06-13

Review scope:

- [x] `chisurf/plugins/core/mfdb_admin/gui/tool.py`
- [x] `chisurf/plugins/core/mfdb_admin/gui/client.py`
- [x] `chisurf/plugins/core/mfdb_admin/gui/provenance_graph.py`
- [x] `chisurf/gui/widgets/node_editor/editor.py`
- [x] `chisurf/gui/widgets/node_editor/graph.py`
- [x] `chisurf/gui/widgets/node_editor/scene.py`
- [x] `test/plugins/test_sample_database_plugin.py`
- [x] `test/plugins/test_provenance_graph_adapter.py`
- [x] `chisurf/gui/widgets/node_editor/tests/test_read_only.py`
- [x] `chisurf/gui/widgets/node_editor/tests/test_graph_headless.py`

Current review decision:

- [x] Previous functional blockers for provenance display are fixed:
  - [x] Full graph export responses are unwrapped before display.
  - [x] Edge-only upstream/downstream dependency responses synthesize endpoint nodes.
  - [x] `list_provenance_edges()` reads the backend `provenance_edges` key.
  - [x] `show_timeline=False` keeps undo/redo state available.
  - [x] Raw and processed URL locations preserve their URL scheme when opened.
  - [x] Processed products are scoped through processing runs.
  - [x] Empty-experiment processed-product scoping returns an empty list.
  - [x] Processed-product row count reads the top-level `row_count` field.
- [x] Architecture blocker resolved: `MFDBClient` now defaults to `ZmqClient`;
      `inprocess=True` is explicit and `SampleDatabaseClient` forces in-process legacy compatibility.
- [x] Architecture blocker resolved: `NodeViewerWidget` now formalizes the
      read-only graph display API separately from the editable
      `NodeEditorWidget`, and `NodeEditorWidget` remains compatible with that
      API.

Findings fixed in this implementation pass:

- [x] **P1: MFDB GUI no longer bypasses RPC for condition/probe operations.**
      `chisurf/plugins/core/mfdb_admin/gui/tool.py` now uses typed `MFDBClient`
      helpers for condition save/lookup and probe loading, and no longer imports
      `FluorescenceDatabase` or `resolve_database_path` in that file.
  - [x] Condition save in `save_condition()` calls `self.client.save_sample_condition()`.
  - [x] Condition lookup in `_auto_fill_condition_details()` calls `self.client.get_sample_condition()`.
  - [x] Probe loading in `fill_probes()` calls `self.client.list_probes()`.
  - [x] `FluorescenceDatabase` and `resolve_database_path` imports were removed
        from `chisurf/plugins/core/mfdb_admin/gui/tool.py`.
- [x] **P1: `MFDBClient` now defaults to a ZMQ JSON-RPC client.**
      `chisurf/plugins/core/mfdb_admin/gui/client.py` builds `ZmqClient` by default;
      in-process construction is available only with `inprocess=True` or via
      explicit dependency injection. `SampleDatabaseClient` forces the legacy
      in-process compatibility path.
  - [x] Added `MFDBClient._make_zmq_client()`.
  - [x] Renamed in-process factory wording to `_make_inprocess_client()`.
  - [x] Kept dependency injection for tests and fakes.
  - [x] Added a test proving default construction uses ZMQ and does not import
        plugin backend services.
- [x] **P2: A read-only `NodeViewerWidget` abstraction is now available.**
      `chisurf/gui/widgets/node_editor/node_viewer.py` owns the embeddable
      read-only graph display API (`load_graph_dict`, `graph_dict`, `fit_graph`,
      `clear_graph`, selection signals), while `NodeEditorWidget` remains an
      editable subclass for workflow authoring/demo behavior.
  - [x] Added `NodeViewerWidget`.
  - [x] Kept `NodeEditorWidget` compatible with the NodeViewer API.
  - [x] Added tests for read-only viewer behavior, serialization, selection
        signals, and editor compatibility.

Verification performed during review:

- [x] `python -m pytest test/plugins/test_provenance_graph_adapter.py test/plugins/test_sample_database_plugin.py chisurf/gui/widgets/node_editor/tests/test_graph_headless.py -q`
      passed with `25 passed`.
- [x] `QT_QPA_PLATFORM=offscreen /Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest --no-cov chisurf/gui/widgets/node_editor/tests/test_read_only.py -q`
      passed with `6 passed`.
- [x] `QT_QPA_PLATFORM=offscreen /Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest --no-cov chisurf/gui/widgets/node_editor/tests/test_node_viewer.py chisurf/gui/widgets/node_editor/tests/test_read_only.py chisurf/gui/widgets/node_editor/tests/test_round_trip.py -q`
      passed with `17 passed, 1 warning`.
- [x] `QT_QPA_PLATFORM=offscreen /Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest --no-cov chisurf/gui/widgets/node_editor/tests/test_round_trip.py chisurf/gui/widgets/node_editor/tests/test_validation.py chisurf/gui/widgets/node_editor/tests/test_delete.py chisurf/gui/widgets/node_editor/tests/test_port_types.py chisurf/gui/widgets/node_editor/tests/test_read_only.py chisurf/gui/widgets/node_editor/tests/test_graph_headless.py test/plugins/test_sample_database_plugin.py test/fio/test_fdb_provenance.py test/fio/test_fdb_analysis.py test/plugins/test_provenance_graph_adapter.py`
      passed with `60 passed, 36 warnings`.
- [x] `/Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m ruff check --select F,I,W chisurf/gui/widgets/node_editor/node_viewer.py chisurf/gui/widgets/node_editor/editor.py chisurf/gui/widgets/node_editor/tests/test_node_viewer.py chisurf/gui/widgets/node_editor/edge_item.py chisurf/gui/widgets/node_editor/graph.py chisurf/gui/widgets/node_editor/registry.py chisurf/gui/widgets/node_editor/scene.py chisurf/plugins/core/mfdb_admin/gui/client.py chisurf/plugins/core/mfdb_admin/gui/provenance_graph.py chisurf/plugins/core/mfdb_admin/gui/tool.py chisurf/plugins/core/mfdb_admin/backend/services.py chisurf/plugins/sample_database/gui/client.py test/plugins/test_provenance_graph_adapter.py test/plugins/test_sample_database_plugin.py chisurf/gui/widgets/node_editor/tests/test_round_trip.py chisurf/gui/widgets/node_editor/tests/test_read_only.py chisurf/gui/widgets/node_editor/tests/test_graph_headless.py`
      passed.
- [x] `git diff --check -- chisurf/gui/widgets/node_editor chisurf/plugins/core/mfdb_admin chisurf/plugins/sample_database test/plugins/test_provenance_graph_adapter.py test/plugins/test_sample_database_plugin.py docs/prd_mfdb_viewer_node_editor_workflows.md`
      passed with no output.

## Follow-up Review Result: 2026-06-13

Review scope:

- [x] `chisurf/gui/widgets/node_editor/node_viewer.py`
- [x] `chisurf/gui/widgets/node_editor/editor.py`
- [x] `chisurf/gui/widgets/node_editor/tests/test_node_viewer.py`
- [x] `chisurf/gui/widgets/node_editor/tests/test_read_only.py`
- [x] `chisurf/plugins/core/mfdb_admin/gui/client.py`
- [x] `chisurf/plugins/core/mfdb_admin/gui/tool.py`
- [x] `chisurf/plugins/core/mfdb_admin/backend/services.py`
- [x] `chisurf/plugins/core/mfdb_admin/manifest.json`
- [x] `test/plugins/test_sample_database_plugin.py`
- [x] `docs/prd_mfdb_viewer_node_editor_workflows.md`

Current review decision:

- [x] **ACCEPTED WITH FOLLOW-UP SCOPE.**
- [x] The ZMQ client default and condition/probe RPC refactor are materially
      improved.
- [x] `NodeEditorWidget` default layout regression is fixed.
- [x] `NodeViewerWidget` now exposes the promised abstract viewer API surface
      used by the MFDB viewer path.
- [x] Runtime transport error handling is implemented for the ZMQ default.
- [ ] The full `nodeviewer.graph.*` provider API remains future work and is still
      unchecked below.

Findings:

- [x] **P1: `NodeEditorWidget` no longer breaks default layout after becoming a
      `NodeViewerWidget` subclass.**
      `NodeViewerWidget` now owns only the reusable `viewer_panel`; it does not
      install a top-level layout when subclassed. `NodeEditorWidget` installs the
      editor top-level layout once and adds `self.viewer_panel`, so the default
      side panel, palette, JSON editor, and splitter remain managed.
  - [x] Reproduction command no longer prints the Qt layout warning for
        `NodeEditorWidget(build_example=False, show_side_panel=True, show_timeline=False)`.
  - [x] Added regression coverage for the default side-panel layout and managed
        palette/JSON widgets.

- [x] **P2: `NodeViewerWidget` now implements the abstract viewer API required
      by this PRD.**
      The widget now exposes `show_toolbar`, `graph_purpose`, optional `client`,
      `selected_node()`, `selected_edge()`, `graphLoaded`, and `graphLoadFailed`.
      Successful graph loads emit `graphLoaded`; JSON/dict load failures emit a
      structured `graphLoadFailed` payload.
  - [x] Added constructor args `show_toolbar=False`,
        `graph_purpose="provenance_view"`, and `client=None`.
  - [x] Added `selected_node()` and `selected_edge()`.
  - [x] Added `graphLoaded(dict)` and `graphLoadFailed(dict)` signals.
  - [x] Added tests for each API element.

- [x] **P2: MFDB ZMQ transport errors are handled at widget startup.**
      `MFDBWidget` now accepts an injectable client, keeps a Refresh action
      enabled while disconnected, disables data-dependent toolbar actions and the
      main splitter during transport failure, and shows the transport error in
      `status_label` without raising during construction.
  - [x] Added a top-level refresh error handler.
  - [x] Added `MFDBWidget(client=...)` test injection.
  - [x] Added a fake-client timeout test proving construction remains usable.

- [ ] **P2: NodeViewer JSON-RPC provider API is still not implemented.**
      This is acceptable only if explicitly scoped as future work. Current MFDB
      provenance display still calls `provenance.dependencies.*` and
      `provenance.graph.export` through `MFDBClient`, not `nodeviewer.graph.get`.
  - [ ] Required PRD clarity:
    - [ ] Keep this as an unchecked future-phase requirement unless implemented.
    - [ ] Do not mark the addendum done until `nodeviewer.providers.list`,
          `nodeviewer.graph.get`, `nodeviewer.graph.validate`, and
          `nodeviewer.graph.layout` exist and are tested.

Verification performed during this follow-up review:

- [x] Targeted test command passed:
      `QT_QPA_PLATFORM=offscreen /Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest --no-cov chisurf/gui/widgets/node_editor/tests/test_node_viewer.py chisurf/gui/widgets/node_editor/tests/test_read_only.py test/plugins/test_sample_database_plugin.py test/plugins/test_provenance_graph_adapter.py chisurf/gui/widgets/node_editor/tests/test_graph_headless.py -q`
      with `40 passed, 17 warnings`.
- [x] Manual constructor/layout check no longer finds the Qt layout warning for
      `NodeEditorWidget(build_example=False, show_side_panel=True, show_timeline=False)`.
- [ ] Existing tests do not cover the default side-panel layout regression.

Follow-up implementation checklist:

- [x] Fix `NodeEditorWidget` / `NodeViewerWidget` layout ownership.
- [x] Add a default `NodeEditorWidget()` layout regression test.
- [x] Complete the promised `NodeViewerWidget` abstract API or mark missing API
      elements as future scope.
- [x] Add startup transport-error handling for `MFDBWidget`.
- [ ] Keep the nodeviewer JSON-RPC provider work unchecked until implemented.
- [x] Re-run the targeted arm64 Qt test suite.
- [ ] Re-run the broader node editor round-trip/validation/delete/port tests.
- [ ] Re-run `ruff check --select F,I,W` on touched node editor and MFDB files.

## Spec Sheet: Architecture

### Non-Negotiable Transport Rule

- [ ] All runtime communication between GUI code and backend/core data services
      must happen via ZMQ JSON-RPC.
- [ ] GUI code must not import database repositories, backend service modules,
      service dispatchers, or plugin backend registrars in production paths.
- [ ] GUI code may import pure schema, graph, and validation modules that have
      no database, server, Qt, or transport side effects.
- [ ] Tests may inject fakes or `InProcessClient`, but tests must make this
      explicit.
- [ ] JSON-RPC request and response payloads must be JSON-serializable plain
      dict/list/scalar values.
- [ ] Long-running workflow execution must use JSON-RPC command methods plus
      ZMQ PUB/SUB progress events, not direct Python callbacks across the
      GUI/backend boundary.

### Layer Responsibilities

Core layer:

- [ ] Own graph schema definitions and validation.
- [ ] Own deterministic layout helpers that do not depend on Qt.
- [ ] Own workflow graph analysis helpers such as cycle checks, topological
      ordering, incoming edges, and outgoing edges.
- [ ] Own provider-neutral value objects for node, edge, graph, selection, and
      validation errors.
- [ ] Must not import Qt.
- [ ] Must not import ZMQ.
- [ ] Must not import MFDB repository or plugin backend modules.

API / JSON-RPC layer:

- [ ] Own JSON-RPC method names, request schemas, response schemas, and error
      codes.
- [ ] Own the server-side provider registry for graph providers.
- [ ] Own the ZMQ transport client used by GUI clients.
- [ ] Convert backend/core exceptions into JSON-RPC errors.
- [ ] Keep transport payloads stable and versioned.
- [ ] Expose both read-only graph viewing methods and future editable workflow
      methods.

GUI layer:

- [ ] Own Qt widgets only.
- [ ] Call typed client methods only.
- [ ] Never open repositories or database files directly.
- [ ] Never register backend services directly.
- [ ] Treat `NodeViewer` as an embeddable read-only graph display component.
- [ ] Treat `NodeEditorWidget` as an editable superset used for workflow
      authoring.
- [ ] Convert user gestures into JSON-RPC requests through client APIs.

Plugin/provider layer:

- [ ] MFDB provider maps MFDB provenance records into the shared node graph
      schema.
- [ ] Future workflow provider maps saved workflow graph records into the same
      shared node graph schema.
- [ ] Providers live behind JSON-RPC handlers.
- [ ] Providers may use repositories, database handles, and backend services
      because they run server-side.

## Proposed Module Layout

Core modules:

- [ ] Create `chisurf/core/nodeviewer/__init__.py`.
- [ ] Create `chisurf/core/nodeviewer/schema.py`.
- [ ] Create `chisurf/core/nodeviewer/validation.py`.
- [ ] Create `chisurf/core/nodeviewer/layout.py`.
- [ ] Create `chisurf/core/nodeviewer/providers.py`.
- [ ] Move or wrap current headless graph helpers from
      `chisurf/gui/widgets/node_editor/graph.py` into core without breaking
      existing imports.
- [ ] Keep compatibility import shims in `chisurf/gui/widgets/node_editor/graph.py`
      until all callers are migrated.

API / transport modules:

- [ ] Create `chisurf/server/services/nodeviewer.py`.
- [ ] Register nodeviewer JSON-RPC methods in the server dispatcher.
- [ ] Add method specs to `chisurf/server/server_methods.json`.
- [ ] Add client method specs to `chisurf/server/client_methods.json` if the
      client catalogue expects them.
- [ ] Use `chisurf/server/transport/zmq.py::ZmqClient` for production GUI
      transport.
- [ ] Add a small typed client wrapper, for example
      `chisurf/core/nodeviewer/client.py` or
      `chisurf/gui/widgets/node_editor/client.py`, that only calls JSON-RPC.

GUI modules:

- [ ] Create `chisurf/gui/widgets/node_editor/node_viewer.py` or
      `chisurf/gui/widgets/node_viewer.py`.
- [ ] Implement `NodeViewerWidget` as a thin read-only wrapper around the
      existing scene/view infrastructure.
- [ ] Keep `NodeEditorWidget` backward compatible.
- [ ] Make `NodeEditorWidget` reuse `NodeViewerWidget` behavior rather than
      duplicating graph loading, selection, fitting, and read-only logic.
- [ ] Keep existing import path `chisurf.gui.widgets.node_editor.editor.NodeEditorWidget`.

MFDB provider modules:

- [ ] Keep `chisurf/plugins/core/mfdb_admin/gui/provenance_graph.py` temporarily for
      GUI compatibility.
- [ ] Move provider-neutral conversion helpers to core or server-side provider
      modules.
- [ ] Create `chisurf/plugins/core/mfdb_admin/backend/nodeviewer_provider.py`.
- [ ] Register the MFDB provenance provider under provider id
      `mfdb.provenance`.
- [ ] Ensure provider output is the shared NodeViewer graph schema.

## Proposed JSON-RPC API

All methods below must be served over ZMQ JSON-RPC in production.

Provider discovery:

- [ ] `nodeviewer.providers.list`
  - [ ] Params: `{}`.
  - [ ] Result: `{"providers": [{"provider_id": str, "label": str, "read_only": bool, "supports_workflows": bool}]}`

Graph retrieval:

- [ ] `nodeviewer.graph.get`
  - [ ] Params:
        `{"provider_id": str, "query": dict, "layout": "provider" | "auto" | "none"}`
  - [ ] Result:
        `{"graph": NodeViewerGraph, "warnings": list[str]}`
  - [ ] MFDB example query:
        `{"seed_node_type": "processed_data", "seed_node_id": "prod_1", "direction": "full"}`
  - [ ] Supported MFDB directions:
    - [ ] `upstream`
    - [ ] `downstream`
    - [ ] `full`

Graph validation:

- [ ] `nodeviewer.graph.validate`
  - [ ] Params: `{"graph": NodeViewerGraph, "purpose": "provenance_view" | "workflow"}`
  - [ ] Result:
        `{"valid": bool, "errors": list[dict], "warnings": list[dict]}`

Graph layout:

- [ ] `nodeviewer.graph.layout`
  - [ ] Params:
        `{"graph": NodeViewerGraph, "algorithm": "dependency_columns"}`
  - [ ] Result:
        `{"graph": NodeViewerGraph}`

Workflow authoring, future phase:

- [ ] `workflow.graph.create`
- [ ] `workflow.graph.get`
- [ ] `workflow.graph.save`
- [ ] `workflow.graph.validate`
- [ ] `workflow.graph.delete`
- [ ] `workflow.run.start`
- [ ] `workflow.run.status`
- [ ] `workflow.run.cancel`
- [ ] `workflow.run.result.get`

Workflow progress events, future phase:

- [ ] Publish progress on ZMQ PUB/SUB topic prefix `workflow.run.`.
- [ ] Event payload must include:
  - [ ] `run_id`
  - [ ] `workflow_id`
  - [ ] `node_id`
  - [ ] `state`
  - [ ] `message`
  - [ ] `timestamp`
  - [ ] optional `progress`
  - [ ] optional `result_ref`

## NodeViewer Graph Schema

Top-level graph:

- [ ] `version: 1`
- [ ] `meta: dict`
- [ ] `nodes: list[NodeViewerNode]`
- [ ] `edges: list[NodeViewerEdge]`

Required `meta` keys:

- [ ] `purpose`: one of:
  - [ ] `provenance_view`
  - [ ] `workflow`
  - [ ] `example`
- [ ] `schema_name`: stable dotted schema id.
- [ ] `read_only`: bool.

Recommended `meta` keys:

- [ ] `provider_id`
- [ ] `graph_id`
- [ ] `workflow_id`
- [ ] `created_by`
- [ ] `created_at`
- [ ] `source`
- [ ] `transport`: must be `zmq-json-rpc` for runtime-loaded graphs.

Node object:

- [ ] `id: str`
- [ ] `title: str`
- [ ] `type: str`
- [ ] `inputs: list[PortSpec]`
- [ ] `outputs: list[PortSpec]`
- [ ] `config: dict`
- [ ] `pos: [float, float]`
- [ ] `collapsed: bool`
- [ ] optional `z: float`

Edge object:

- [ ] `source: str`
- [ ] `source_port: int`
- [ ] `target: str`
- [ ] `target_port: int`
- [ ] optional `config: dict`

Port object:

- [ ] string shorthand remains supported for backward compatibility.
- [ ] dict form supports:
  - [ ] `name: str`
  - [ ] optional `type: str`
  - [ ] optional `fixed: bool`
  - [ ] optional `min: float`
  - [ ] optional `max: float`

## Abstract NodeViewer API

Create a small API that both provenance views and future workflow editors can
share.

`NodeViewerWidget` requirements:

- [ ] Constructor accepts:
  - [ ] `read_only: bool = True`
  - [ ] `show_toolbar: bool = False`
  - [ ] `graph_purpose: str = "provenance_view"`
  - [ ] optional `client`
- [ ] Public methods:
  - [ ] `load_graph_dict(graph: dict) -> None`
  - [ ] `graph_dict() -> dict`
  - [ ] `clear_graph() -> None`
  - [ ] `fit_graph() -> None`
  - [ ] `set_read_only(read_only: bool) -> None`
  - [ ] `selected_node() -> dict | None`
  - [ ] `selected_edge() -> dict | None`
- [ ] Signals:
  - [ ] `nodeSelected(dict)`
  - [ ] `edgeSelected(dict)`
  - [ ] `graphLoaded(dict)`
  - [ ] `graphLoadFailed(dict)`
- [ ] It must be usable without MFDB imports.
- [ ] It must be usable without workflow runtime imports.

`NodeEditorWidget` requirements:

- [ ] Remains backward compatible with current constructor defaults.
- [ ] Can be used as `NodeViewerWidget` in read-only mode.
- [ ] Adds editing affordances only when `read_only=False`.
- [ ] Keeps undo/redo available even if timeline UI is hidden.
- [ ] Maintains workflow metadata without stripping unknown keys.
- [ ] Exposes future workflow authoring hooks but does not execute workflows
      directly.

## MFDB GUI Refactor Checklist

- [ ] `MFDBClient` production default uses `ZmqClient`.
- [ ] `MFDBClient` accepts host/port or a configured JSON-RPC endpoint.
- [ ] `MFDBClient` accepts an injected fake/in-process client for tests.
- [ ] `MFDBClient` never imports `chisurf.plugins.core.mfdb_admin.backend`.
- [ ] `MFDBClient` never registers services.
- [ ] `MFDBWidget` uses only `MFDBClient` and pure GUI/core helpers.
- [ ] `MFDBWidget` has no direct imports from `chisurf.core.fio.mmcif.db`.
- [ ] `MFDBWidget` has no direct imports from `chisurf.core.mfdb.repository`.
- [ ] Add JSON-RPC methods for sample conditions:
  - [ ] `mfdb.sample_conditions.get`
  - [ ] `mfdb.sample_conditions.save`
  - [ ] `mfdb.sample_conditions.list`
- [ ] Add JSON-RPC methods for probes:
  - [ ] `mfdb.probes.list`
  - [ ] `mfdb.probes.get`
  - [ ] optional `mfdb.probes.optical_properties.get`
- [ ] Update GUI condition form to use client helpers.
- [ ] Update GUI probe table to use client helpers.
- [ ] Add tests that fail if GUI imports repository/database modules.

## MFDB Provenance NodeViewer Checklist

- [ ] Add provider id `mfdb.provenance`.
- [ ] Server-side provider accepts:
  - [ ] `seed_node_type`
  - [ ] `seed_node_id`
  - [ ] `direction`
- [ ] Provider maps `direction="upstream"` to upstream dependency graph.
- [ ] Provider maps `direction="downstream"` to downstream dependency graph.
- [ ] Provider maps `direction="full"` to exported provenance graph.
- [ ] Provider returns shared NodeViewer graph schema.
- [ ] Provider includes `meta.provider_id = "mfdb.provenance"`.
- [ ] Provider includes `meta.transport = "zmq-json-rpc"`.
- [ ] GUI provenance tab calls `nodeviewer.graph.get` instead of directly
      calling `provenance.dependencies.*` for display.
- [ ] Existing MFDB provenance JSON/ZIP export RPCs remain available.
- [ ] Existing typed provenance helpers can remain if they are implemented over
      ZMQ JSON-RPC and are used for non-NodeViewer workflows.

## Future Workflow Editor Checklist

- [ ] Persist workflow graphs server-side through JSON-RPC.
- [ ] Validate workflow graphs server-side before saving.
- [ ] Keep client-side validation for immediate UX feedback only.
- [ ] Workflow execution runs server-side.
- [ ] GUI starts workflow execution through JSON-RPC.
- [ ] GUI receives execution progress through ZMQ PUB/SUB events.
- [ ] Nodes declare runtime behavior through `NodeType.runtime`,
      `NodeType.operation`, or server-registered operation ids.
- [ ] GUI does not import executor callables.
- [ ] GUI can inspect workflow results through JSON-RPC result references.
- [ ] Workflow graph metadata records:
  - [ ] `workflow_id`
  - [ ] `version`
  - [ ] `created_by`
  - [ ] `updated_at`
  - [ ] `last_run_id`

## New Verification Checklist

Transport boundary tests:

- [ ] Test that `MFDBClient()` production path constructs a ZMQ JSON-RPC client.
- [ ] Test that `MFDBClient(fake_client)` still works for unit tests.
- [ ] Test that importing `chisurf.plugins.core.mfdb_admin.gui.tool` does not import
      MFDB repository modules.
- [ ] Test that `MFDBWidget.save_condition()` calls a client method, not a
      repository.
- [ ] Test that `MFDBWidget.fill_probes()` calls a client method, not a
      repository.

NodeViewer API tests:

- [ ] Test `NodeViewerWidget` can load a graph without MFDB imports.
- [ ] Test `NodeViewerWidget` emits `nodeSelected`.
- [ ] Test `NodeViewerWidget` emits `edgeSelected`.
- [ ] Test `NodeViewerWidget(read_only=True)` blocks graph mutation.
- [ ] Test `NodeEditorWidget(read_only=False)` remains editable.
- [ ] Test `NodeEditorWidget` remains backward compatible with old default
      constructor behavior.

JSON-RPC tests:

- [ ] Test `nodeviewer.providers.list`.
- [ ] Test `nodeviewer.graph.get` for `mfdb.provenance` full graph.
- [ ] Test `nodeviewer.graph.get` for upstream dependencies.
- [ ] Test `nodeviewer.graph.get` for downstream dependencies.
- [ ] Test `nodeviewer.graph.validate` rejects malformed edges.
- [ ] Test `nodeviewer.graph.layout` is deterministic.

Manual QA:

- [ ] Start ChiSurf ZMQ JSON-RPC server.
- [ ] Open MFDB plugin as a GUI client.
- [ ] Verify sample/experiment/raw/processed/provenance views load through RPC.
- [ ] Disconnect server and verify GUI shows a transport error, not a crash.
- [ ] Reconnect server and verify refresh works.
- [ ] Load provenance graph through `NodeViewer`.
- [ ] Confirm graph metadata includes `transport = "zmq-json-rpc"`.
- [ ] Confirm no direct database file is opened by the GUI process.

## Updated Done Definition For This Addendum

- [ ] Runtime GUI/backend communication uses ZMQ JSON-RPC.
- [ ] In-process dispatch is limited to explicit tests or explicit development
      fallback code.
- [ ] MFDB GUI contains no direct database/repository/backend service imports.
- [ ] `NodeViewerWidget` exists as a provider-neutral graph viewer abstraction.
- [ ] `NodeViewerWidget` implements the complete abstract API in this PRD:
      constructor args, selected-node/edge accessors, load success/failure
      signals, and tests.
- [ ] `NodeEditorWidget` remains backward compatible and can still support
      future user-defined workflow editing.
- [ ] `NodeEditorWidget()` default construction installs exactly one valid
      top-level layout and preserves the existing example side panel.
- [ ] `NodeEditorWidget()` default construction emits no Qt layout warnings.
- [ ] MFDB provenance display can load through `nodeviewer.graph.get`.
- [ ] If `nodeviewer.graph.get` remains future scope, the PRD and completion
      report explicitly say that MFDB provenance still uses the legacy
      provenance RPCs over ZMQ JSON-RPC.
- [ ] `MFDBWidget` handles disconnected ZMQ transport during startup without
      raising out of `__init__`.
- [ ] All new transport-boundary tests pass.
- [ ] All new layout-regression tests pass.
- [ ] Existing MFDB provenance and node editor tests still pass.
