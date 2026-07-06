---
type: PRD
prd: "29"
title: "PRD-29: Visual Burst Programming — Node-Graph Editor for Burst Analysis"
description: Turns the existing node editor into a visual programming canvas for composing, running, previewing, and provenance-recording burst-analysis pipelines.
status: planned
phase: "2"
resource: chisurf/gui/widgets/node_editor/
tags: [prd, fret, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Turns the existing node editor into a visual programming language for burst analysis: a canvas where users graphically compose pipelines (load TTTR → filter → select bursts → compute FRET → fit decays → export). Each node is a typed, executable operation following the transformer contract; the graph runs live via the reactive chinet runtime with intermediate previews. Every execution is recorded in MFDB with full provenance (operations, artifacts, parameters, edges), and pipelines are saved as MFDB artifacts that reference data by UUID so they are shareable and re-executable, including headlessly. Re-execution with changed parameters creates a branch rather than mutating the original.

# Status
Planned. Merges three existing pieces (node editor, chinet runtime, MFDB); phased plan covers burst node types, provenance recording, live interactivity, and polish including headless execution.

# Goal
Turn the existing node editor into a **visual programming language for burst analysis** — a node-graph canvas where users compose burst pipelines graphically: load TTTR data → apply filters → select bursts → compute FRET → fit decays → output results. Each node is a **typed, executable operation**; the graph runs live, previews intermediate data, and every execution is recorded with full provenance in MFDB.

Three existing pieces merge into one workflow:

| Component | Role in this PRD |
|-----------|------------------|
| **Node editor** (`chisurf/gui/widgets/node_editor/`) | The visual canvas — graph creation, port wiring, live preview, parameter editing |
| **chinet** (`modules/chinet/`) | The reactive runtime — topological evaluation, dataflow propagation, lazy recomputation |
| **MFDB** (`chisurf/core/mfdb/`) | The persistence + provenance layer — artifact storage, operation recording, pipeline serialization |

# Background

## What exists today
**Node editor.** A full visual graph editor with:
- Qt `QGraphicsScene`/`QGraphicsView` with dark theme, bezier edges, port snapping
- DAG validation (cycle detection via networkx), undo/redo via full-graph snapshots
- Node registry (`NodeType` + `PortSpec`), JSON serialization (schema v1)
- `chinet_eval.py` evaluates PT (parameter transform) nodes via chinet
- Already used by the lightpath simulator (optical nodes) and provenance viewer (read-only)

**chinet.** A reactive dataflow runtime with:
- Port-Node-Session architecture, lazy evaluation, reactive propagation
- Canonical schema (`chinet.session.v1`) for serialization
- `MFDBChinetBackend` for transparent persistence
- `Parameter` → `chinet.Port` bridge used by all fitting models

**MFDB.** The measurement database with:
- Artifact/operation/provenance model (`mfdb_artifact`, `mfdb_operation`, `mfdb_edge`)
- Burst pipeline registration ([PRD-04](prd-04.md)): raw TTTR → burst_selection → burst_table
- Result registry ([PRD-03](prd-03.md)): one-call storage for typed burst/fit data
- Object store for blob data (burst tables, decay curves, etc.)
- Planned transformer contract ([PRD-16](prd-16.md)) and pipeline engine ([PRD-22](prd-22.md))

## What is missing
| Gap | Current state |
|-----|---------------|
| Burst-specific node types | None exist — only math/chinet/optical nodes |
| Burst data as port values | No typed port for "burst table" or "TTTR stream" |
| Live preview of intermediate data | `PtPlotWidget` exists for chinet plots, no burst-specific preview |
| Pipeline-as-document | Node editor saves JSON graphs; not persisted as MFDB artifacts |
| Execution records provenance | `chinet_eval.py` doesn't record to MFDB |
| Reactive recompute on parameter change | Works for PT nodes, not connected to burst pipeline |

# Design

## Architecture overview
```
┌──────────────────────────────────────────────────────────────┐
│                    Node Editor Canvas                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ Load     │    │ Burst        │    │ Compute      │        │
│  │ TTTR     │───▶│ Select       │───▶│ FRET         │───▶    │
│  │ (file)   │    │ (thresholds) │    │ (γ = 1.0)   │        │
│  └──────────┘    └──────────────┘    └──────────────┘        │
│       │                 │                  │                  │
│       ▼                 ▼                  ▼                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ Preview  │    │ Burst Table  │    │ FRET Dist.   │        │
│  │ scatter  │    │ widget       │    │ plot widget  │        │
│  └──────────┘    └──────────────┘    └──────────────┘        │
├──────────────────────────────────────────────────────────────┤
│  chinet Runtime (evaluates graph in topological order)       │
├──────────────────────────────────────────────────────────────┤
│  MFDB (persists pipeline + records each execution)           │
└──────────────────────────────────────────────────────────────┘
```

## Node types (burst domain)
Each node type follows the [PRD-16](prd-16.md) transformer contract: typed input/output ports + `.dic`-declared parameters + pure `transform` function.

| Node Type ID | Category | Inputs | Outputs | Parameters | Widget Factory |
|---|---|---|---|---|---|
| `load_tttr` | Data | (none) | `TTTRStream` | file_path, file_type | File picker |
| `photon_filter` | Filter | `TTTRStream` | `TTTRStream` | dT min/max, microtime range, routing chs | Range editors |
| `burst_select` | Burst | `TTTRStream` | `BurstTable` | T window, threshold FRET, min photons, ... | Slider/combo panel |
| `burst_filter` | Burst | `BurstTable` | `BurstTable` | E min/max, S min/max, size range | Range sliders |
| `compute_fret` | Analysis | `BurstTable` | `FRETResult` | gamma, donor_only_lifetime | Numeric inputs |
| `fit_decays` | Analysis | `BurstTable` | `FitResult` | model (exponential, stretched), n_components | Model picker |
| `fit_fret_dist` | Analysis | `BurstTable` | `DistanceResult` | model (Gaussian, logNormal, MLE), n_components | Model picker |
| `anisotropy` | Analysis | `TTTRStream` | `AnisotropyCurve` | time_range, bin_width | Range slider |
| `fcs_correlate` | Analysis | `TTTRStream` | `FCSCurve` | tau_min, tau_max, binning_per_decade | Range sliders |
| `export_csv` | Output | `BurstTable` / `FRETResult` / `FitResult` | (none) | file_path, columns | File save dialog |
| `export_figure` | Output | `BurstTable` / `FRETResult` / `FitResult` | (none) | file_path, format, dpi | File save dialog |

## Port types (data model)
```
TTTRStream    — reference to loaded TTTR data (file path + metadata)
BurstTable    — burst table (columns: start, stop, E, S, duration, n_photons, ...)
FRETResult    — FRET efficiency distribution (E_values, counts, fit_params)
FitResult     — decay fit (decay_curve, fit_curve, residuals, parameters)
DistanceResult — distance distribution (r_values, P(r), fit_params)
AnisotropyCurve — time-resolved anisotropy (time, r(t), fit_params)
FCSCurve      — FCS correlation curve (tau, G(t), fit_params)
```
Each port type maps to an MFDB artifact kind (from `payload_models.py`).

## Graph evaluation (chinet integration)
1. When the user adds/wires a burst node, `NodeEditorWidget` registers it as a `chinet.Node` with matching ports.
2. Chinet handles topological ordering and reactive propagation:
   - Changing a parameter slider → marks downstream nodes invalid → re-evaluates
   - Replacing an upstream file → cascading recompute
3. The evaluation calls each node's `transform(inputs, parameters)` (pure function — no Qt, no DB).
4. Results are stored as port values (numpy arrays + metadata).
5. Preview widgets read result port values for display.

## MFDB recording (provenance integration)
When the user executes or auto-evaluates the graph:

1. **Pipeline definition** is saved as a `chinet.session.v1` schema → MFDB artifact (`artifact_kind="burst_pipeline"`).
2. **Each node execution** is recorded as an `mfdb_operation` with:
   - `operation_type` = the node type (e.g. `"burst_selection"`)
   - `input_artifacts` = MFDB refs to upstream node outputs
   - `output_artifacts` = MFDB refs to this node's results
   - `parameters` = current node parameter values → `mfdb_parameter`
3. **Provenance edges** connect operations into the MFDB graph:
   ```
   load_tttr(artifact) ──input_to──▶ burst_select(operation)
   burst_select(operation) ──produced──▶ burst_table(artifact)
   burst_table(artifact) ──input_to──▶ compute_fret(operation)
   ```
4. **Re-execution** creates a new branch ([PRD-27](prd-27.md) branching) — the original pipeline is immutable; "what-if" variants are branches.

## Node-graph live feedback
| Feature | Implementation |
|---------|---------------|
| **Preview on hover** | Hovering a port shows a tooltip with value summary |
| **Side panel preview** | Selecting a node shows its output in a preview dock (scatter plot, table, etc.) |
| **Inline mini-plots** | Nodes like `compute_fret` show a mini histogram inside the node body |
| **Parameter sliders** | Numeric parameters use `InlineLabeledSlider` — changing a slider immediately recomputes downstream (chinet reactive), like a live number slider |
| **Visual dataflow** | Edge coloring changes when data flows (dim = stale, bright = valid) |
| **Error highlighting** | Failed nodes turn red with error message in tooltip |

## Pipeline as a shareable document
- A pipeline graph is serialized to the existing node editor JSON schema (+ burst node types) and wrapped in a `chinet.session.v1` schema.
- Saved as an MFDB artifact (`artifact_kind="burst_pipeline"`).
- Can be loaded from the node editor's "Open..." dialog → loads from MFDB.
- Can be shared: the pipeline document references data by MFDB UUID, so anyone with access to the same DB can open and re-execute it.
- **Versioning**: each save creates a new artifact version ([PRD-27](prd-27.md) append-only).

# Implementation plan

## Phase 1: Burst node types + runtime
1. **Define port type enums** (`TTTRStream`, `BurstTable`, `FRETResult`, etc.) in the node editor or a shared module.
2. **Implement burst node types** as `NodeType` registrations:
   - `load_tttr` — wraps existing `tttrlib.TTTR` file loading
   - `burst_select` — wraps burst selection logic (reuse from burst_selection plugin)
   - `compute_fret` — wraps FRET computation (E, S from burst table columns)
   - `fit_decays` — wraps decay fit (reuse from fitting framework)
   - Preview nodes for scatter plots, histograms, tables
3. **Connect to chinet evaluation** — extend `chinet_eval.py` so burst nodes are evaluated by chinet, not just PT nodes. Each burst node's `transform` is wrapped as a `chinet.Node` callback.
4. **Add preview widgets** (mini-plots inside nodes, side panel views).

## Phase 2: MFDB provenance recording
5. **Pipeline persistence** — save/load the graph as `chinet.session.v1` → `mfdb_artifact` (`artifact_kind="burst_pipeline"`).
6. **Per-node operation recording** — after chinet evaluation, call `register_result()` for each output artifact and `record_operation()` for each node execution. Use `mfdb_operation_artifact` to link inputs/outputs.
7. **Provenance edge wiring** — call `add_edge()` to connect nodes in the MFDB graph, enabling the provenance viewer to display the pipeline.
8. **Branching** — when the user tweaks parameters and re-executes, create a new branch ([PRD-27](prd-27.md)) so the original execution is preserved.

## Phase 3: Node-graph interactivity
9. **Reactive recompute** — wire `InlineLabeledSlider` changes to chinet invalidation, so moving a slider immediately updates downstream nodes.
10. **Live preview** — implement side panel that shows the selected node's output (burst table scatter plot, FRET histogram, decay fit overlay).
11. **Visual dataflow feedback** — edge dim/bright on validity, node border color changes (green = computed, yellow = stale, red = error).
12. **Preview caching** — cache intermediate results so reconnecting an edge doesn't recompute the entire graph.

## Phase 4: Polish
13. **Burst-specific node palette** — a categorized palette (Data, Filter, Burst, Analysis, Output) for drag-add, like a component tab.
14. **Preset pipelines** — ship example graphs (e.g., "basic FRET analysis", "FCS correlation", "anisotropy decay") as MFDB seed data.
15. **CLI/headless execution** — reuse [PRD-22](prd-22.md)'s headless runner: load a pipeline artifact from MFDB, evaluate, record. This makes burst analysis reproducible from the command line.
16. **Export pipeline as script** — generate Python script from the graph (each node → function call, edges → data flow).

# Relationship to existing PRDs
| PRD | Relationship |
|-----|-------------|
| [PRD-03](prd-03.md) (result registry) | `register_result()` stores burst node outputs as MFDB artifacts |
| [PRD-04](prd-04.md) (burst pipeline) | `burst_select` node wraps the established burst selection pipeline |
| [PRD-11](prd-11.md) (operation abstraction) | Each node maps to an `operation_type`; parameters go to `mfdb_parameter` |
| [PRD-16](prd-16.md) (transformer contract) | Each node's `transform` follows the typed input/output/parameter contract |
| [PRD-21](prd-21.md) (lineage API) | The graph's provenance edges are queryable via the lineage API |
| [PRD-22](prd-22.md) (pipeline engine) | Shares the headless runner; this PRD adds the **visual** editor on top |
| [PRD-27](prd-27.md) (event-sourced core) | Pipeline versions and re-executions use branching |

[PRD-22](prd-22.md) defers the GUI; this PRD delivers it. The two share the same runner and transformer contract — this PRD is the GUI counterpart of [PRD-22](prd-22.md)'s headless engine.

# Non-goals
- **General-purpose visual programming** — scoped to burst analysis. (The architecture is generic enough to extend, but this PRD does not define a generic visual language.)
- **Removing existing plugins** — burst node types supplement, not replace, the existing burst selection/FRET/fitting plugin UIs.
- **Real-time streaming** — pipelines run on loaded files, not streaming data.
- **Auto-layout** — the existing spring layout is sufficient; no dedicated graph layout engine.

# Definition of Done
- [ ] At least 4 burst node types exist: `load_tttr`, `burst_select`, `compute_fret`, `export_csv`.
- [ ] A 3-node pipeline (load → burst_select → compute_fret) can be built visually and executed.
- [ ] Changing a parameter slider recomputes downstream nodes (reactive).
- [ ] Pipeline is persisted as an MFDB artifact and can be reloaded.
- [ ] Each execution is recorded in MFDB with full provenance (operations, artifacts, parameters, edges).
- [ ] The provenance viewer (mfdb_admin) displays the pipeline graph.
- [ ] Headless execution: `python -m chisurf run-pipeline <pipeline_uuid>` reproduces the same results.

# Definition of Clean
- Burst node types are registered in `NodeRegistry` with zero Qt imports in the `transform` function (pure data in/out).
- chinet integration uses the existing `chinet_eval.py` path — no fork of the evaluation logic.
- MFDB recording uses `register_result()` / `record_operation()` — no raw SQL in the node editor.
- Port types are dataclasses in a shared module, not ad-hoc dicts.
- Tests exist for: node type registration, graph evaluation, pipeline round-trip, MFDB recording, reactive recompute.

# Relationships
- Delivers the visual counterpart to [PRD-22](prd-22.md) (pipeline engine), sharing the headless runner and transformer contract.
- Nodes map to operations and typed parameters per [PRD-11](prd-11.md) and follow the [PRD-16](prd-16.md) transformer contract.
- Stores outputs via [PRD-03](prd-03.md) result registry; edges queryable via [PRD-21](prd-21.md) lineage API; versioning/branching via [PRD-27](prd-27.md).
- The `burst_select` node wraps the [PRD-04](prd-04.md) burst pipeline.
- Uses [MFDB (current)](/architecture/mfdb.md); modeled on a visual node-graph editor's interaction style.
