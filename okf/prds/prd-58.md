---
type: PRD
prd: "58"
title: "PRD-58: FRET Plugin as a Strict FPS + OLGA Superset"
description: Make the fret modelling plugin a strict superset of the legacy FRET-positioning (FPS) and optimal-label-selection (OLGA) tools — embedded fps.json editing with live AV preview, the full refine/bootstrap/sample/evaluate workflows, and integrated informative-pair selection.
status: draft
phase: "feature track"
resource: chisurf/plugins/modelling/fret/
tags: [prd, plugins, modelling, fret]
timestamp: '2026-07-06T00:00:00Z'
---

# Summary
The `fret` and `fps_json_editor` modelling plugins work but do not know about each
other: the editor produces `fps.json` label/distance definitions with no way to run
AV calculations, docking, screening or pair selection, and the `fret` engine has a
complete docking/screening core and CLI but its wizard does not reuse the editor.
Several legacy workflows are also missing (refine, bootstrap, a working Metropolis
`sample`, an evaluator graph, informative pair selection). This PRD makes
`chisurf/plugins/modelling/fret/` a **strict superset of the FPS and OLGA tools**
by (1) embedding an improved `LabelStructure` editor inside the `fret` wizard with
a live accessible-volume preview and no file round-trip, (2) completing the missing
FPS workflows, and (3) adding the OLGA evaluator graph and informative pair
selection. GUI and CLI stay thin; all computation lives in importable core modules.

# Status
Draft — the plan predates approval ("awaiting approval before any code is written",
2026-06-07). Partially realized since: the `fret/evaluators/` subpackage has landed
(base + positions/distance/fret_efficiency/chi2/residuals/geometry/av_metrics), and
the plugin ships `TARGETS.md` and `CHANGELOG_FRET.md`. The requirements, phased plan,
and acceptance criteria below remain the authoritative roadmap for the remaining work.

# Goal
Keep all computation in importable, GUI-free core modules so the plugin is a strict
superset of the legacy FPS (FRET-positioning) and OLGA (optimal-label-selection)
tools, reachable identically from CLI and GUI.

# Current state

- `fps_json_editor` — a working `LabelStructure` widget: loads a PDB via
  `chisurf.core.structure.Structure`, picks an attachment atom, configures AV
  parameters (linker length/width, radii, resolution), builds the `Positions`
  entry, defines distances (R0, d, error±, type), manages score groups (`χ²`
  key) and `FlexFit` data, and saves/loads `fps.json`. It lacks an AV preview,
  a "Run AV" button, any connection to the `fret` engine, score-set management
  buttons, a `body_id` field, and a tabular distance view.
- `fret` engine (working): `av.py` (AV1/AV3 via an external labelling library
  primary, external biophysical-modeling framework fallback), `clash.py` (numba
  vdW), `distance.py` (Rmp/RDAMean/RDAMeanE/chi2/histogram), `engine.py`
  (Verlet-spring rigid body), `docking.py`, `screening.py` (multi-threaded),
  `results.py`, `io.py` (fps.json + PDB), `__main__.py` (CLI: `info`/`dock`/
  `screen`). `sampling.py` has a bug (`rst.position_a` should be
  `rst.global_position_a(bodies)`); the `sample` CLI exits with an error. 20
  tests pass. `wizard.py` does not embed `LabelStructure`.

# Requirements

| ID | Requirement |
|---|---|
| R01 | `LabelStructure` embeddable as a widget in the `fret` wizard, sharing one fps.json data model |
| R02 | `LabelStructure` shows a live AV preview (n_points, volume, mean position) for a selected attachment point |
| R03 | Add/Remove score-set button pair |
| R04 | `body_id` field per position (multi-body docking) |
| R05 | Distances shown in a table (not a flat list) with all key columns |
| R06 | Wizard embeds `LabelStructure` on a Project/Edit tab |
| R07 | Docking/screening launched from the wizard use the in-editor fps.json, no file round-trip |
| R08 | CLI `info-backends` reports availability/version of both AV backends |
| R09 | `dock`/`screen`/`sample` accept `--av-backend auto\|labellib\|imp-bff` |
| R10 | `refine` CLI + `run_refinement()` (ported from FPS `Refinement.cs`), tested |
| R11 | `bootstrap` CLI + `run_bootstrap()` (ported from FPS `ErrorEstimation.cs`), tested |
| R12 | `sample` Metropolis bug fixed (`rst.global_position_a(bodies)`) |
| R13 | Output writers: viewer `.pml` script, R-table, chi²-table, tested |
| R14 | RMSD vs reference with optional Kabsch superposition, tested |
| R15 | OLGA evaluator base classes + all 15 evaluator types |
| R16 | `evaluate` CLI supports `--pdb`, `--pdb-dir`, `--top`+`--traj` |
| R17 | `pair_selection.py` with NaN preprocessing + report writer |
| R18 | `select-pairs` CLI subcommand |
| R19 | All new public functions carry NumPy-style docstrings |
| R20 | No conda env changes via pip except `pip install --no-deps labellib` |

**Non-goals:** display-requiring GUI tests, hard trajectory-library dependency
(optional only), the fallback AV backend on Windows (skip gracefully), GPU batch.

# Architecture

`LabelStructure` is owned by `fps_json_editor` and **imported** by `fret/wizard.py`;
both share a live Python dict (`positions`, `distances`, `score_sets`) via a
`fps_json_payload` getter/setter — no file round-trip. The wizard grows tabs:
Edit fps.json (embedded editor), Docking, Screening, Evaluators (new), Pair
Selection (new), Results.

New/changed `fret` files: `__main__.py` (+info-backends/refine/bootstrap/
select-pairs/evaluate), `av.py` (+`select_backend`), `io.py` (+evaluator JSON,
`compute_rmsd`), `results.py` (+bootstrap result, `.pml`/R-table/chi²-table
writers), `sampling.py` (bug fix), new `refine.py`, `bootstrap.py`,
`pair_selection.py`, and the `evaluators/` subpackage. The 15 evaluator types span
`positions`, `distance` (+distribution), `fret_efficiency`, `chi2` (+reduced/
contribution), `residuals`, `geometry` (Euler/transform/min-distance), and
`av_metrics` (size/sphere-overlap), all deriving from an `Evaluator` ABC with
`EvaluatorResult`, `EvaluationStorage` (to_dataframe/to_csv) and JSON round-trip
under the fps.json `Evaluators` key.

# Phased plan

1. **Bugs + backend** — fix Metropolis; `select_backend()`; `info-backends`;
   `--av-backend` on all subcommands.
2. **FPS parity** — `refine`, `bootstrap`, fix `sample`; output writers; RMSD.
3. **OLGA evaluators** — base classes, 15 evaluator types, JSON round-trip,
   `evaluate` CLI.
4. **OLGA pair selection** — `pair_selection.py` (drop columns with NaN fraction
   > 0.20; fill from closest-RMSD frame), reusing
   `chisurf/plugins/traj/fret_pair_selection/olga_greedy.py`
   (`select_informative_pairs`); OLGA-compatible tab-separated report;
   `select-pairs` CLI.
5. **Editor + wizard** — AV preview, score-set buttons, distance table,
   `body_id`, `fps_json_payload`; embed in the wizard.
6. **Integration QA** — parity checklist, examples, changelog.

Each phase: implement → test → confirm green → proceed.

# Final CLI surface
`info-backends`, `info`, `dock` (`[--av-backend][--n-trials][--ref-pdb]`),
`refine`, `bootstrap`, `sample`, `screen`, `evaluate`
(`--pdb\|--pdb-dir\|--top+--traj`), `select-pairs`.

# Hard rules
Never pip-install compiled packages except `pip install --no-deps labellib`; NumPy
docstrings on new public API; keep all previous tests green each phase; never rename
the `fps.json` keys `Positions`, `Distances`, `χ²`; GUI/CLI stay thin (compute in
core modules).

# Relationships
- Concrete build-out of the FRET docking plugin summarized in
  [Modelling plugins](/plugins/modelling.md); shares the `LabelStructure` widget
  with `fps_json_editor`.
- Contributes to the multiparameter-fluorescence feature-parity track
  ([PRD-49](prd-49.md)) by absorbing the legacy FPS/OLGA tool capabilities.
- Its AV backends are an instance of the external-framework boundary drawn in
  [PRD-47](prd-47.md).
