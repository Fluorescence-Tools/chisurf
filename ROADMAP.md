# ROADMAP

This roadmap is derived from `TODO.md`, `AGENTS.md`, and `AGENTS_PLAN.md` and is intended to be the actionable execution plan (ranked by complexity/work).

## Dependency Triage (Blocking vs Not Blocking)

### Blocking (core app / core workflows)

- GUI/runtime: `PyQt5`, `qtpy`, `sip`, `pyqtgraph`
- Numerics: `numpy<2.0`, `scipy`, `numba`, `numexpr`
- Broad core utilities: `PyYAML`, `tables`, `click`, `typing-extensions`, `deprecation`
- In-repo native deps that block major workflows if missing:
  - `chinet` (`modules/chinet`): parameter engine used by `chisurf/parameter.py` and node-editor integrations
  - `tttrlib` (`modules/tttrlib`): TTTR readers + many TTTR/burst/FCS plugins and TCSPC TTTR paths

### Not blocking (optional or plugin-only workflows)

- `python-docx` (docx export/report generation)
- `mdtraj` (trajectory-related tooling/plugins)
- `qtconsole`, `ipython` (embedded console/Jupyter-adjacent UX)
- `matplotlib` (some plotting/export paths)
- `emcee` (sampling/MCMC features)
- Optional extra: `scikit-learn` (`[project.optional-dependencies].ml`)
- In-repo plugin deps (only block their plugins): `ndxplorer`, `quest`, `clsmview`, `labellib`

## Workstreams (Ranked)

Legend: Complexity 1-5, Work S/M/L/XL, Progress: `[ ]` Pending, `[/]` In Progress, `[x]` Done

### 1) Full-Fidelity Project Save/Restore v3 (Primary Plan) (60% Complete)

Source: `AGENTS_PLAN.md` Phases 1-6.

- [/] Phase 1: Identity + registry foundations (Complexity 4, Work L)
  - [x] Add/verify stable UIDs for all persistent entities (`unique_identifier`).
  - [x] Implement registry module (new) for O(1) UID-to-object resolution.
  - [ ] Ensure parameter UID reattachment via stable `param_path` matching on model rebuild.

- [/] Phase 2: Linking correctness rewrite (Complexity 5, Work XL)
  - [/] Eliminate fit-index-based mutations from GUI handlers.
  - [/] Capture link endpoints by `source_param_uid`/`target_param_uid`; restore links as an explicit UID graph.
  - [/] Add cycle detection and endpoint validation before applying links.

- [x] Phase 3: Project v3 save pipeline (Complexity 5, Work XL)
  - [x] New bundle format: `project.json` manifest + `arrays.npz` + optional `blobs/`.
  - [x] Deterministic serialization ordering and strict schema versioning.
  - [x] Require model state contracts: `get_constructor_state/get_state/set_state`.

- [x] Phase 4: Project v3 load pipeline (Complexity 5, Work XL)
  - [x] Deterministic load order: datasets -> fits -> models -> params -> links -> UI.
  - [x] Strict UID integrity checking and actionable failure messages.

- [/] Phase 5: UI 1:1 restore hardening (Complexity 5, Work XL)
  - [x] Use `saveGeometry/saveState` + stable `objectName` for docks/toolbars.
  - [/] Two-pass restore: recreate widgets first, then `restoreState`, then semantic UI state.

- [/] Phase 6: Performance + reliability (Complexity 3, Work M)
  - [x] Transactional save (temp + atomic replace), batched redraw suppression during restore, timing instrumentation.

### 2) Operation History + Replay + Action Core (50% Complete)

Source: `AGENTS.md` + `AGENTS_PLAN.md` Phases 7-10. *(Cross-ref: `TODO.md` -> Medium Priority -> History Replay)*

- [/] Expand structured history coverage (Complexity 4, Work L)
  - [/] Ensure all state-changing actions emit structured events with UIDs.
  - [ ] Persist history stream in project bundle (`history.jsonl`).

- [/] Undo/redo “time travel” hardening (Complexity 4, Work L)
  - [/] Expand replay coverage from navigation/parameter subset toward full scientific state (Phase 10).
  - [ ] Consolidate finalize/update paths so replay produces consistent GUI/model refresh.

- [x] MVC action dispatcher core (Complexity 5, Work XL)
  - [x] Add GUI-independent action registry/dispatcher with schema validation + dedupe/coalescing.
  - [x] Migrate a vertical slice first (parameter set/link/unlink), then fits/datasets, then project lifecycle.

### 3) Plugin Check System (Safety + Correctness)

Source: `TODO.md` (High Priority). *(Cross-ref: `TODO.md` -> High Priority -> Plugin Check System)*

- [/] Replace string matching with AST-based detection (Complexity 3, Work M)
  - [/] Parse plugin source via `ast.parse` and inspect `ast.Call` to detect `eval/exec`, `subprocess.*`, `os.system`, etc. (Temporary string spacing fix applied)
  - [ ] Allowlist safe patterns like `ast.literal_eval`, `json.loads`, and `yaml.safe_load`.
  - [ ] Add tests covering safe-vs-dangerous edge cases.

### 4) GUI/UX Improvements

*(Cross-ref: `TODO.md` -> Medium Priority -> Plots / UX, Parameters / UX, Unified UX, TCSPC)*

- [ ] Fit progress window width explosion on long fit names (Complexity 1, Work S)
- [ ] Auto-enable group display for line plots of FitGroups (Work S)
- [ ] Lineplot: reduce legend/metrics footprint for multi-fit groups (Work M)
- [ ] Output parameters should be visually distinct (Work S)
- [ ] Unify selection UX for single-molecule setups and plugin settings (Work M)
- [ ] Add steady-state anisotropy output (`r_ss`) to the anisotropy widget (Work S)

### 5) Documentation, Release, Licensing, Performance (Product Hygiene)

Source: `TODO.md` (Medium Priority). *(Cross-ref: `TODO.md` -> Medium Priority -> Release Process / Versioning, Performance, Documentation, Licensing)*

- [ ] Release workflow + versioning docs (Complexity 2, Work M)
- [ ] Startup performance profiling + lazy init plan (Complexity 3, Work M)
- [ ] Plugin dev docs (security + check behavior) (Complexity 2, Work S/M)
- [ ] License inventory + effective license baseline (Complexity 3, Work M/L)

### 6) Node Based Models (Longterm)

- [ ] Make node editor and node-based models the default approach for modeling.

### 7) AI Settings Migration

*(Cross-ref: `TODO.md` -> High Priority -> AI Settings Migration)*

- [/] Migrate ALL Chato API settings to centralized AI settings plugin
  - [x] Provider, Base URL, API Key, Models, Text/Code Embedding, Temperature
  - [ ] RAG Settings, remove hardcoded settings, import from ai_settings


## Known Bugs / Risk Register (Keep Visible)

Source: `BUGS.md` (note: cross-check against `CHANGELOG.md` because some items may already be fixed).

- Project lifecycle: save -> close -> open can fail due to missing `chisurf.cs` in some flows.
- Parameter widget edits sometimes fail to fully propagate finalize/update consistently.
- Native access violation during fit creation (Windows crash) in `chinet`-adjacent parameter init paths.
- Jupyter integration reliability is currently poor/disabled by default.
- TTTR Image Browser `.docx` default save folder mismatch.
