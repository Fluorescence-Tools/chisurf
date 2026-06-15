# PMI-Compatible RMF Output And Chimol Display PRD

## Goal

Make all ChiSurf structure-related RMF output PMI-compatible so generated RMF/RMF3 files can be opened correctly in ChimeraX and displayed correctly in Chimol. ProteinMC and future structure generators should write frame metadata and scores in a PMI-compatible way so Chimol can inspect trajectories while they are being generated.

## Scope

This work covers RMF output produced by ChiSurf structure/modeling code, including ProteinMC and shared structure RMF writers.

Primary source areas:

- `chisurf/plugins/modelling/proteinmc/`
- `chisurf/core/models/structure/`
- `chisurf/plugins/chimol/chimol/io/rmf.py`
- `chisurf/plugins/chimol/chimol/renderer/chimol_state.py`
- `chisurf/plugins/chimol/chimol/renderer/view.py`
- `chisurf/plugins/chimol/chimol/app/`

Reference-only source areas:

- `/Users/tpeulen/dev/imp/modules/pmi/pyext/src/output.py`
- `/Users/tpeulen/dev/imp/modules/pmi/pyext/src/topology/__init__.py`
- `/Users/tpeulen/dev/imp/modules/pmi/pyext/src/dof/__init__.py`
- `/Users/tpeulen/dev/imp/modules/pmi/pyext/src/macros.py`
- `thirdparty/rmf_chimerax/src/io.py`
- `thirdparty/rmf_chimerax/src/tool.py`
- `thirdparty/rmf_chimerax/src/cmd.py`

Do not add `thirdparty/rmf_chimerax` as a dependency. Code patterns may be adapted when useful.

## Non-Goals

- Do not replace ProteinMC's current torsion Monte Carlo engine in the first implementation.
- Do not require PMI for all ChiSurf installs unless RMF writing or PMI compatibility is requested.
- Do not make Chimol depend on ChimeraX.
- Do not implement full PMI `ReplicaExchange` sampling until PMI-compatible output and display are stable.

## Definitions

PMI-compatible RMF means:

- The RMF hierarchy uses standard IMP/PMI-compatible hierarchy decorators: system/state/molecule or hierarchy root, chain, residue, atom/particle, coordinates, radii, and representations.
- The file is readable by standard IMP/RMF tooling and ChimeraX RMF support.
- Frame metadata is stored using the PMI `stat` category pattern from `IMP.pmi.output.Output`.
- Restraints/features that are written to RMF use standard `IMP.rmf.add_restraints`, `IMP.rmf.add_geometries`, or RMF representation nodes where possible.
- Optional ChiSurf-specific metadata is additive and harmless to external readers.

Live inspection means:

- A partially generated RMF can be opened or refreshed in Chimol without crashing.
- Chimol can show coordinates, hierarchy, restraints/features, provenance, and frame score plots for frames written so far.

## Current State

ProteinMC currently writes RMF via `chisurf/plugins/modelling/proteinmc/rmf.py`.

Known limitations:

- The writer builds a minimal IMP hierarchy manually.
- It saves coordinate frames but not PMI `stat` metadata.
- It does not embed `energy`, `labeling_energy`, `rmsd`, `drmsd`, `iteration`, `accepted`, or `rejected` values.
- Similar RMF writer code exists in `chisurf/core/models/structure/rmf.py`.
- Similar ProteinMC runner code exists in `chisurf/core/models/structure/proteinmc.py`.

Chimol currently reads RMF via `chisurf/plugins/chimol/chimol/io/rmf.py`.

Known limitations:

- It loads coordinates, hierarchy, radii, simple provenance, bonds, and two-particle representation restraints.
- It does not read PMI `stat` frame metadata.
- It does not expose general RMF features, frame score series, alternatives, resolutions, or richer provenance.
- RMF plotting UI does not exist yet.

## Product Requirements

1. All new ChiSurf structure RMF output must be PMI-compatible by default.
2. Existing RMF writer call sites must keep working with minimal API changes.
3. ProteinMC RMF output must open in ChimeraX using ChimeraX RMF support.
4. ProteinMC RMF output must open in Chimol with correct coordinates, hierarchy, and frame count.
5. ProteinMC RMF output must include frame-aligned PMI `stat` metadata for scores and progress.
6. Chimol must read PMI `stat` metadata into frame series suitable for plotting.
7. Chimol must provide a plot option for frame series such as energy, labeling energy, RMSD, dRMSD, accepted, rejected, and iteration.
8. Chimol must be able to refresh or reopen a growing RMF file during generation.
9. Optional PMI imports must be guarded and produce clear errors when unavailable.
10. Tests must cover writing, reading, Chimol metadata extraction, and basic plot-model behavior.

## Architecture Overview

Use a shared RMF writer implementation for structure trajectories.

Recommended module layout:

- `chisurf/core/models/structure/rmf.py`: canonical shared PMI-compatible RMF writer and metadata helpers.
- `chisurf/plugins/modelling/proteinmc/rmf.py`: thin compatibility import or ProteinMC-specific adapter around the shared writer.
- `chisurf/plugins/modelling/proteinmc/model.py`: ProteinMC runner passes frame metadata into the writer.
- `chisurf/plugins/chimol/chimol/io/rmf.py`: loader extracts hierarchy, coordinates, features, provenance, and PMI `stat` frame series.
- `chisurf/plugins/chimol/chimol/app/rmf_panel.py`: RMF feature/provenance/plot dock.

The shared writer should support two levels of compatibility:

- Basic IMP-compatible hierarchy built manually with `IMP.atom.Hierarchy`, `IMP.atom.Chain`, `IMP.atom.Residue`, `IMP.atom.Atom`, and `IMP.core.XYZR`.
- PMI-style stat metadata following `IMP.pmi.output.Output.init_rmf()` and `write_rmf()` behavior.

Avoid using PMI topology classes for the initial writer unless necessary. Manual IMP hierarchy plus PMI `stat` category is simpler and easier to keep compatible with existing structures.

## RMF Stat Metadata Format

Use category `stat`, matching PMI output.

For each scalar output key:

- Create RMF key with `rh.get_key(cat, output_key, tag)`.
- Use `RMF.float_tag` for floats.
- Use `RMF.int_tag` for integers.
- Use `RMF.string_tag` for strings.
- Set values on `rh.get_root_node()` after `IMP.rmf.save_frame()`.
- Flush the RMF handle after each written frame for live inspection.

Required keys for ProteinMC:

- `Total_Score`
- `ProteinMC_Energy`
- `ProteinMC_Labeling_Energy`
- `ProteinMC_RMSD`
- `ProteinMC_dRMSD`
- `ProteinMC_Iteration`
- `ProteinMC_Accepted`
- `ProteinMC_Rejected`
- `rmf_file`
- `rmf_frame_index`

Optional keys:

- `ProteinMC_KT`
- `ProteinMC_Move_Scale`
- `ProteinMC_Frame_Name`
- `ChiSurf_Version`
- `ProteinMC_Settings_JSON`

Chimol should treat all numeric `stat` values with one value per frame as plottable frame series.

## Step 1: Inventory And Consolidate RMF Writers

Tasks:

- Search for all RMF writers and `IMP.rmf.save_frame` calls in ChiSurf.
- Identify duplicated structure/ProteinMC writer code.
- Choose one canonical shared writer location.
- Convert plugin-local writer modules to import or subclass the shared writer.
- Keep backward-compatible imports unless no code uses them.

Suggested commands:

```bash
python -m pytest chisurf/plugins/modelling/proteinmc/test/test_plugin.py
```

Acceptance criteria:

- Only one implementation owns core RMF hierarchy/stat writing logic.
- Existing ProteinMC tests still pass.
- Existing imports from `chisurf.plugins.modelling.proteinmc.rmf` still work.

## Step 2: Implement PMI-Compatible Stat Writer Helper

Tasks:

- Add a helper class, for example `RmfStatWriter`.
- It receives an RMF file handle and a list/dict of output keys.
- It creates RMF `stat` keys using the PMI pattern.
- It writes per-frame values to the root node.
- It flushes after each frame.

Reference behavior:

- `/Users/tpeulen/dev/imp/modules/pmi/pyext/src/output.py`, methods `init_rmf()` and `write_rmf()`.

Pseudo-API:

```python
class RmfStatWriter:
    def __init__(self, rmf_handle, initial_output: dict[str, object]) -> None: ...
    def write(self, values: dict[str, object]) -> None: ...
```

Acceptance criteria:

- Float, int, and string values are written with matching RMF tags.
- Unknown or unsupported values are converted to strings.
- Writing stats is optional and does not break coordinate-only RMF output.

## Step 3: Update Shared Structure RMF Writer

Tasks:

- Ensure the hierarchy uses standard IMP decorators.
- Ensure blank chain IDs are replaced with a valid chain ID such as `A`.
- Preserve atom names, residue names, residue IDs, chain IDs, radii, and coordinates.
- Add optional `append(..., metadata: dict[str, object] | None = None)` support.
- Save coordinates with `IMP.rmf.save_frame()` before writing root `stat` values.
- Call `flush()` after metadata is written.

Acceptance criteria:

- Output opens in ChimeraX.
- Output opens in Chimol.
- Existing coordinate frame loading still works.
- Frame metadata can be omitted.

## Step 4: Update ProteinMC To Write PMI Stats

Tasks:

- Update ProteinMC `_append_frame()` so RMSD and dRMSD are computed before the RMF frame is written.
- Pass metadata to the RMF writer for every written frame.
- Include initial frame metadata.
- Use `Total_Score` and `ProteinMC_Energy` consistently. For now both can equal the weighted ProteinMC energy.
- Include `ProteinMC_Labeling_Energy` even when no labeling potential is active, using `0.0`.

Metadata map:

```python
{
    "Total_Score": float(energy),
    "ProteinMC_Energy": float(energy),
    "ProteinMC_Labeling_Energy": float(labeling_energy),
    "ProteinMC_RMSD": float(rmsd),
    "ProteinMC_dRMSD": float(drmsd),
    "ProteinMC_Iteration": int(iteration),
    "ProteinMC_Accepted": int(accepted),
    "ProteinMC_Rejected": int(rejected),
}
```

Acceptance criteria:

- ProteinMC RMF has one metadata value per coordinate frame.
- `rmf_frame_index` matches frame order.
- Chimol reads the score series.

## Step 5: Add Chimol PMI Stat Reader

Tasks:

- Extend `chisurf/plugins/chimol/chimol/io/rmf.py`.
- During frame iteration, read root-node values from RMF category `stat`.
- Build `rmf_frame_series: dict[str, np.ndarray]` for numeric keys.
- Preserve string values in `rmf_frame_metadata` if useful.
- Return these from `load_rmf_full()`.

Important details:

- RMF API availability can vary. Guard `get_category`, `get_keys`, `get_name`, and `get_value` calls.
- Some files may not have `stat`; return an empty dict.
- A stat key may be absent for some frames; fill numeric missing values with `np.nan`.

Acceptance criteria:

- `load_rmf_full(path)["rmf_frame_series"]` contains ProteinMC stat arrays.
- Existing RMF files without `stat` still load.
- Corrupt or partially written stat values do not prevent coordinate loading.

## Step 6: Add Chimol State Plumbing

Tasks:

- Extend `_MolViewObjectState` with:
  - `rmf_features`
  - `rmf_frame_series`
  - `rmf_frame_metadata`
  - `rmf_resolutions`
- Extend `MolView.set_rmf_data()` to accept these fields.
- Update RMF load path in `molview_main_window.py` to pass these fields.
- Update object switching so RMF UI panels refresh with active object state.

Acceptance criteria:

- Active object state exposes RMF frame series.
- Object switching updates hierarchy and RMF panel without stale data.

## Step 7: Add Chimol RMF Panel With Plotting

Tasks:

- Add `chisurf/plugins/chimol/chimol/app/rmf_panel.py`.
- Provide a dock with:
  - frame-series combo box,
  - line plot widget,
  - current-frame marker,
  - min/max/current value labels,
  - optional feature/provenance tabs.
- Use a lightweight Qt `QWidget.paintEvent()` plot first to avoid adding dependencies.
- Refresh the current-frame marker from the viewer/timeline timer.

Acceptance criteria:

- Opening a ProteinMC RMF shows plottable series.
- Selecting `Total_Score` or `ProteinMC_Energy` plots values versus frame index.
- Changing the Chimol frame updates the marker and current value.
- Empty series show a clear `No RMF frame series` message.

## Step 8: Add Live RMF Refresh Support In Chimol

Tasks:

- Add a refresh action for active RMF objects.
- Track source path in object state/store.
- Re-run `load_rmf_full()` on refresh and replace frames/metadata for the active object.
- Preserve camera/view state and active frame where possible.
- Add optional file-watch timer later if manual refresh is stable.

Acceptance criteria:

- A ProteinMC RMF can be opened while generation is running.
- Manual refresh loads newly written frames.
- If the file is temporarily unreadable because it is being flushed/written, Chimol shows a non-fatal warning and keeps the previous data.

## Step 9: Improve RMF Feature And Provenance Loading

Tasks:

- Adapt data-model ideas from `thirdparty/rmf_chimerax/src/io.py`.
- Represent RMF features separately from the structural hierarchy.
- Keep two-particle representation features as drawable restraints.
- Add richer provenance extraction for available factories:
  - `StructureProvenanceConstFactory`
  - `SampleProvenanceConstFactory`
  - `ScriptProvenanceConstFactory`
  - `SoftwareProvenanceConstFactory`
- Add resolution and alternatives extraction where factories are available.

Acceptance criteria:

- Chimol hierarchy remains structural, not polluted by provenance/feature nodes.
- Feature and provenance panels show meaningful names.
- Missing factories are harmless.

## Step 10: Add Commands And GUI Hooks

Tasks:

- Extend `chisurf/plugins/chimol/chimol/cmd/rmf.py` with:
  - `rmf_hierarchy [object_id]`
  - `rmf_features [object_id]`
  - `rmf_provenance [object_id]`
  - `rmf_scores [object_id]`
  - `rmf_refresh [object_id]`
- Add GUI menu/toolbutton hooks for RMF refresh and plot panel visibility.

Acceptance criteria:

- Commands work in tests with mock viewer state where possible.
- GUI actions call the same underlying functions as commands.

## Step 11: ChimeraX Validation

Tasks:

- Generate a small ProteinMC RMF3 file.
- Open it in ChimeraX with the RMF plugin available.
- Verify hierarchy, frame count, and coordinate display.
- Verify no warnings caused by invalid hierarchy decorators or malformed stat keys.

Manual validation command example:

```bash
csc proteinmc 148L --output /tmp/proteinmc_pmi.rmf3 --n-iter 2 --n-out 1 --n-written 1
```

Acceptance criteria:

- ChimeraX opens the generated file.
- ChimeraX displays coordinates for all frames it loads.
- Chimol opens the same file and displays score plots.

## Step 12: Tests

ProteinMC tests:

- Writer replaces blank chain IDs.
- Writer creates PMI `stat` keys.
- Runner writes frame metadata.
- Metadata length equals number of coordinate frames.

Chimol RMF loader tests:

- Reads frames from PMI-compatible ProteinMC RMF.
- Reads `stat` values into `rmf_frame_series`.
- Handles RMF without `stat`.
- Handles partially missing stat values.

Chimol plot tests:

- Plot model/widget accepts empty data.
- Plot model/widget accepts one point.
- Plot model/widget accepts NaN values.
- Current-frame marker clamps to available range.

Use guards:

```python
pytest.importorskip("IMP")
pytest.importorskip("RMF")
```

Recommended targeted test commands:

```bash
python -m pytest chisurf/plugins/modelling/proteinmc/test/test_plugin.py
python -m pytest chisurf/plugins/chimol/test/
```

## Step 13: Documentation

Tasks:

- Document that ChiSurf structure RMF output is PMI-compatible.
- Document ProteinMC RMF frame score keys.
- Document Chimol RMF refresh and plot workflows.
- Mention that ChimeraX RMF plugin is a validation target, not a dependency.

Acceptance criteria:

- User can generate a ProteinMC RMF and inspect it in Chimol while generation continues.
- User can open the same RMF in ChimeraX.

## Implementation Notes

- Prefer minimal shared helpers over large abstractions.
- Do not add backward-compatibility code unless an import path or API is currently used.
- Keep optional IMP/PMI imports lazy so non-RMF workflows are unaffected.
- Flush RMF files after each frame to support live inspection.
- Be defensive when reading RMF. External RMF files vary widely in available factories and metadata.

## Open Questions For The Implementing Agent

1. Should the canonical writer live in `chisurf/core/models/structure/rmf.py` or a new lower-level module such as `chisurf/core/structure/rmf.py`?
2. Should ProteinMC's plugin-local RMF module remain a compatibility wrapper permanently?
3. Should live inspection be manual refresh first, or should file watching be implemented immediately after stable refresh?
4. Should all ChiSurf-specific frame keys be prefixed with `ChiSurf_` or only ProteinMC-specific keys with `ProteinMC_`?
5. Should future PMI-native sampling be exposed as `--engine pmi` after this output/display work is complete?

## Done Criteria

The task is complete when:

- ProteinMC writes PMI-compatible RMF3 files by default.
- Shared structure RMF output uses the same PMI-compatible writer path.
- Generated files open in ChimeraX.
- Generated files open in Chimol.
- Chimol displays frame score plots from PMI `stat` metadata.
- Chimol can refresh a growing RMF file during generation.
- Targeted ProteinMC and Chimol tests pass.
