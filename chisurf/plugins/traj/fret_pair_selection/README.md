# Optimal FRET Pair Selection (Olga-style)

This ChiSurf plugin selects an informative set of FRET pairs from a structural ensemble (trajectory) using an Olga-style greedy experiment planning algorithm.

The core idea is:

- Given an ensemble (frames) and many candidate labeling pairs,
- compute FRET efficiencies `E` for each candidate pair in each frame,
- compute the pairwise RMSD matrix between frames,
- greedily pick the next pair that minimizes the expected mean RMSD ("precision") under a chi-squared tail weighting.

## Where it lives

- Plugin package: `chisurf.plugins.traj.fret_pair_selection`
- GUI entry point: `wizard.py` (`FRETPairSelectionWindow`)
- Core selection algorithm: `olga_greedy.py` (`select_informative_pairs`)
- Trajectory + fps.json helpers: `traj_utils.py`

## Requirements

- `mdtraj` (trajectory loading, RMSD, distances)
- `numpy`
- `numba` (speeds up the greedy selector)
- `PyQt5` / `qtpy` / `pyqtgraph` (GUI)

## Inputs

### 1) Trajectory

Any MDTraj-supported trajectory. For formats like DCD you typically also need a topology (PDB).

### 2) Candidate labeling positions & candidate pairs

You can define candidates in two ways:

#### A) Recommended: `*.fps.json`

Provide an `*.fps.json` created with ChiSurf's **Structure:FPS JSON Editor**.

This plugin currently uses the following sections:

- `Positions` (required)
  - Each position must define at least:
    - `residue_seq_number` (PDB `resSeq`)
    - `atom_name` (attachment atom; `CA` fallback is used if missing)
    - `chain_identifier` (optional; if empty, first matching chain is used)

- Candidate pair list is derived in this order:
  - `Mean FRET Efficiencies` (if present): treated as a *candidate list* (`position1_name`, `position2_name`, `Forster_radius`).
  - Otherwise, if `Distances` exists and looks like a placeholder candidate list (e.g. `distance`/`error_*` are `< 0`): use `position1_name`, `position2_name`, `Forster_radius`.
  - Otherwise: **all pairwise combinations** of positions (Olga-style candidate pool).

If a pair entry has no `Forster_radius`, the GUI `R0` value is used.

#### B) Manual residue selection (fallback)

If no fps.json is provided, you can enter residue ranges (PDB `resSeq`) and an attachment atom (`CB` with fallback to `CA`). Candidate pairs are then all combinations of those sites.

### 3) Algorithm parameters

- `Expected E error`: the assumed absolute experimental uncertainty in `E`.
- `Max pairs`: how many pairs the greedy selection will choose.
- `RMSD atom selection`: MDTraj atom selection string used for the RMSD matrix (default `name CA`).
- `Stride`: trajectory subsampling; the RMSD matrix scales as `N_frames^2`, so clustering/subsampling is strongly recommended.

## Outputs

### GUI table

- Ranked list of selected pairs
- The corresponding "precision decay" value after each added pair (expected mean RMSD in Å)

### Plot

- Precision decay curve vs number of pairs added

### Export

Export writes a tab-separated text file with:

- `Pair_added`
- `Pair`
- `<<RMSD>>/A`

## How to run (GUI)

1. Start ChiSurf
2. Open: **Plugins → Structure → Optimal FRET Pair Selection**
3. Load trajectory (+ topology if needed)
4. Optionally load `*.fps.json`
5. Click **Run selection**
6. Optionally **Export**

## Headless usage

You can run the selector without the GUI using:

- `chisurf.plugins.traj.fret_pair_selection.olga_greedy.select_informative_pairs(effs, rmsds, err, max_pairs, ...)`

Where:

- `effs` is `(n_frames, n_pairs)` float array
- `rmsds` is `(n_frames, n_frames)` float array in Å

## Notes / current limitations

- When `fps.json` is provided and **Use AV backend** is enabled, the plugin computes mean FRET efficiencies by **sampling dye positions from accessible volumes (AVs)**:
  - **Windows:** prefers **LabelLib** (`chisurf.structure.av.BasicAV`)
  - **macOS/Linux:** prefers **IMP.bff** (`quest.lib.imp_av`)
- If the preferred AV backend is not available, the plugin falls back to computing `E` from **attachment atom distances**:
  - `E = 1 / (1 + (R/R0)^6)`
- AV-based per-frame efficiencies can be computationally expensive; use **clustering / subsampling** (and ChiSurf `Stride`) for large ensembles.

## References

- Dimura, M. et al. Quantitative FRET studies and integrative modeling unravel the structure and dynamics of biomolecular systems. *Curr Opin Struct Biol*. 2016.
- Dimura, M. et al. Automated and optimally FRET-assisted structural modeling. *Nat Commun*. 2020;11:5394. https://doi.org/10.1038/s41467-020-19023-1
