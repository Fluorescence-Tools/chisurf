# Changelog: FRET Superset Plugin

This document outlines all changes, enhancements, and features introduced in the chiSurf FRET modeling plugin (superset of FPS and OLGA) and the refactored `fps_json_editor`.

## FRET Plugin Backend Parity

- **Metropolis Sampling Fix:** Resolved `rst.position_a`/`rst.position_b` bug by mapping references to `rst.global_position_a(bodies)` and `rst.global_position_b(bodies)`.
- **Refinement Cycle (`run_refinement`):** Added iterative dock-and-recompute refinement matching the legacy FPS `Refinement.cs` logic.
- **Parametric Bootstrap (`run_bootstrap`):** Added parametric error estimation to calculate model distance standard deviations under perturbed experimental measurements.
- **Kabsch RMSD Alignment:** Implemented coordinate superposition and RMSD calculations.
- **Output Writers:** Added PML script exporter for PyMOL, along with R-table and chi2-table tab-separated value format writers.
- **OLGA Evaluator Graph:** Implemented 13 evaluators covering accessible volume size, volume, distances, FRET efficiencies, chi², residuals, translation, rotation, and minimal inter-body clashes.
- **Informative Pair Selection:** Integrated the OLGA greedy algorithm to select optimal FRET labeling pairs under evolutionary coordinate changes.

## Command Line Interface (CLI)

Exposed new CLI commands on `python -m chisurf.plugins.modelling.fret`:
- `info-backends`: Details available AV calculation backends (LabelLib, IMP.bff).
- `refine`: Run SpringEngine docking refinement.
- `bootstrap`: Perform parametric bootstrap error estimation.
- `evaluate`: Run OLGA-style evaluators on PDB structures, directories, or trajectories.
- `select-pairs`: Select optimal experimental label pairs.
- Added `--av-backend auto|labellib|imp-bff` flags to all applicable commands.

## GUI Refactoring & Chimol Integration

- **Monolithic Split:** Refactored the 708-line `label_structure.py` into decoupled sub-modules:
  - `model.py`: Pure-Python data model for `fps.json` payload manipulation.
  - `av_worker.py`: Non-blocking AV calculations running on a background `QThread`.
  - `position_panel.py`: Atom selection, AV properties, body index fields, and a live 3D structure/AV preview.
  - `distance_panel.py`: Distance inputs and an editable `QTableWidget` replacing the old list widget.
  - `flexfit_panel.py`: Dedicated layout and callbacks for FlexFit residues and bonds.
- **Chimol Renderer Extension:** Added public point-cloud rendering APIs (`add_point_overlay`, `update_point_overlay`, `remove_point_overlay`, `clear_point_overlays`, `add_sphere`) to the 3D viewer.
- **Wizard Integration:** Embedded the complete parameter editor, evaluators tool, and pair selection tool directly as tabs inside the FRET Docking & Screening Wizard.
