# PRD-31: ndXplorer Headless CLI (parameter-based burst filtering + imaging)

## Status

Proposed — **someone else implements**. This PRD was written after wiring and
testing the `raw+sample → Burst Selection → ndXplorer` CLI handoff (see PRD-28);
the missing leg is a *headless* ndXplorer that can run its two core workflows —
**burst filtering** and **imaging** — without the GUI.

## Goal

Give ndXplorer a **headless CLI** for its two main jobs, both currently
GUI-only:

1. **Burst filter** — select a subset of bursts by parameter ranges/gates (e.g.
   proximity ratio, stoichiometry, lifetime, count rate) and emit a *filtered
   burst selection*.
2. **Imaging** — load image-axis data (FLIM / intensity / per-pixel parameter
   maps, CLSM TTTR), apply parameter gates and/or pixel ROIs, and **export
   images / parameter maps** (and optionally a masked photon/burst sub-selection).

Both run with no window shown, print JSON to stdout, and integrate with MFDB.
This closes the CLI round trip (burst leg shown):

```bash
# 1) raw + sample  ->  Burst Selection  (already works, PRD-28 / BS CLI --mfdb)
csc burst-selection analyze m000.spc m001.spc \
    --filetype SPC-130 --detectors-json det.json \
    --mfdb --db mfdb.sqlite --sample-name "DNA burst sample"
#   => registers raw inputs (sample-linked), burst tables, and ONE output-folder
#      group artifact; prints mfdb_artifacts with that artifact id.

# 2) Burst Selection  ->  ndXplorer  (THIS PRD — does not exist yet)
ndxplorer filter --from-mfdb <output_folder_artifact_id> --db mfdb.sqlite \
    --select "proximity_ratio:0.30-0.70" --select "n_photons:50-" \
    --to-mfdb --sample-id <sample_id>
#   => writes a FILTERED burst selection (a new burst folder / burst_table)
#      referencing the same TTTR files, registered under the same sample.
```

## Why

ndXplorer is chisurf's burst/MFD **and imaging** explorer: it ingests burst
selections (`open_files(file_type="burst_dir", …)`), does parameter-based gating
(column selection, 1D/2D histograms, polygon/range gates), and renders image-axis
data (FLIM / intensity / per-pixel parameter maps with pixel ROI selection). But
its **CLI is only a GUI launcher**:

`modules/ndxplorer/ndxplorer/__main__.py:main` accepts `--file/--folder/
--test-data/--processed-data-id/--experiment-id/--zmq-port/--debug` and then
calls `win.show(); app.exec_()`. There is no headless mode that loads data,
applies a gate/ROI, and writes a filtered selection or an exported image. So
neither the burst-filter nor the imaging workflow can participate in scripted/CI
pipelines or in the MFDB round trip non-interactively.

## Current state (verified)

- **BS → MFDB works headlessly.** `csc burst-selection analyze … --mfdb`
  registers raw inputs linked to a sample, burst tables, and a single
  output-folder `external_reference` (`data_format="directory"`). Verified on
  `bh_spc132_sm_dna`: 2 inputs, 2 burst tables, 1 group; the group resolves via
  `MFDatabase.open_dataset` (what `mfdb.datasets.open` / the ndXplorer launcher
  call) to the on-disk burstwise folder containing `m000.bur`/`m001.bur`
  co-located with the TTTR files.
- **`.bur` files reference photons by index in the original TTTR file** (PRD-28).
  A filtered selection must therefore stay a *reference next to the same TTTR*
  (a burst sub-selection / new `.bur` folder beside the TTTR), never an
  object-store copy — otherwise the photon-index linkage breaks.
- **Shared-DB requirement (learned in testing).** The in-process `MFDBClient`
  used by the chisurf-side launcher opens the **configured** database, not an
  arbitrary `--db` path. For a real round trip, BS and ndXplorer must point at
  the *same* MFDB (use the configured DB, or thread the same DB path through
  both CLIs). The headless ndXplorer CLI must accept an explicit DB/source so it
  reads what BS wrote.

## Existing pieces to reuse (do not reinvent)

- **ndXplorer ingest:** `NDXplorer.open_files(file_type="burst_dir",
  file_handles=<path>, append=False)` and
  `ndxplorer.__main__.open_path_like_drop(ndx, path)`.
- **ndXplorer parameters/gating:** its column/parameter model and selection
  (`ui/column_selection_dialog.py`, `ui/parameter_editor.py`, the gate/region
  logic behind the GUI histograms). The headless filter should call the same
  selection core the GUI uses — not a parallel reimplementation.
- **ndXplorer imaging:** the image-axis handling and pixel selection it already
  has — `core/plot_main.py:check_and_set_image_axes`, the pixel ROI/brush masking
  in `utils/mask_helpers.py` (`create_pixel_radius_brush_kernel`, …), and the
  report-tool image rendering (`report_tool.py:_update_image_preview`,
  `_open_report_images_for_folder`). The headless imaging mode must drive this
  same machinery, not a second imaging stack.
- **chisurf glue (keep `modules/ndxplorer` chisurf-free):**
  `chisurf/plugins/ndxplorer/mfdb_launcher.py` already resolves an MFDB artifact
  to a local path (`resolve_dataset_path`) and opens it. MFDB write-back can
  reuse `chisurf.core.mfdb.result_registry.register_result` /
  `BurstMFDBPipeline`.
- **BS CLI `--mfdb`** (this session) — the upstream half; the contract is the
  printed `mfdb_artifacts.output_folder` artifact id.

## Constraints

- **`modules/ndxplorer` stays dependency-free of chisurf.** The pure parameter
  filter primitive (open burst folder → apply parameter gates → write filtered
  burst folder) lives in `modules/ndxplorer`; all MFDB resolution/registration
  lives in `chisurf/plugins/ndxplorer`.
- **Preserve photon-index linkage** (write filtered `.bur`/sub-selection beside
  the same TTTR; reference, don't copy).
- **No GUI.** Must run under `QT_QPA_PLATFORM=offscreen` with no window shown
  (parameters may still need a `QApplication`; construct one without `.show()`).
- **Deterministic, scriptable output**: print the written path / new artifact id
  as JSON to stdout (mirrors BS CLI), so pipelines can chain on it.

## Design

### A. Pure filter primitive (in `modules/ndxplorer`, chisurf-free)

Add a headless subcommand, e.g. `ndxplorer filter`:

- Inputs: `--folder/--file` (a burst selection) and one or more `--select
  "<param>:<min>-<max>"` gates (open-ended `min-` / `-max` allowed), plus
  `--out <dir>` for the filtered burst folder.
- Loads the burst selection through the same path as a GUI drop
  (`open_path_like_drop` / `open_files`), reads the per-burst parameter table,
  evaluates the gates (AND across `--select`, range per parameter), and writes a
  filtered burst folder referencing the same TTTR (the surviving bursts only).
- Emits JSON: `{"input": …, "n_in": …, "n_out": …, "selected": …, "out": …}`.
- Parameter names should match ndXplorer's column ids (document the canonical set
  — proximity ratio, stoichiometry, lifetime, duration, count rate, n_photons).

### B. chisurf MFDB wrapper (in `chisurf/plugins/ndxplorer`)

Add a chisurf-side CLI (e.g. `csc ndxplorer filter`) that:
1. Resolves `--from-mfdb <artifact_id>` (or `--folder`) to a local burst path via
   `resolve_dataset_path` (the configured/`--db` database).
2. Calls the A-primitive to produce a filtered burst folder beside the TTTR.
3. With `--to-mfdb`, registers the filtered folder as a new burst selection
   (operation_type e.g. `"burst_filter"`, parent = the source burst selection,
   linked to `--sample-id`), reusing `register_result`/the burst pipeline so the
   single-group + real-name conventions (PRD-28) hold.
4. Prints the new artifact id / path as JSON.

### D. Imaging primitive (in `modules/ndxplorer`, chisurf-free)

ndXplorer is also used for **imaging** (FLIM / intensity / per-pixel parameter
maps from CLSM TTTR, and report images). Add a headless subcommand, e.g.
`ndxplorer image`, that drives the same image machinery without a window:

- Inputs: `--file/--folder` (CLSM TTTR, an MFD-HDF5 with image axes, or a folder
  of report images), the image axes / parameter to render (e.g.
  `--map intensity|lifetime|<param>`), optional binning/frame range, and optional
  selection: parameter gates (`--select "<param>:<min>-<max>"`) and/or a pixel
  ROI (`--roi <mask.png|polygon.json>`).
- Builds the image via `check_and_set_image_axes` + the existing renderer, applies
  the gate/ROI masking (`mask_helpers`), and **exports**:
  - the rendered image / parameter map (`--out img.png` / `--out-tiff map.tiff`,
    16-bit/float preserved for quantitative maps), and/or
  - a **masked sub-selection** (the photons/bursts inside the ROI/gate) written
    beside the source TTTR, so it can re-enter the burst/MFDB flow.
- Emits JSON: `{"input": …, "map": …, "shape": [h, w], "n_selected_px": …,
  "out": …}`.
- Must be deterministic and headless (`QT_QPA_PLATFORM=offscreen`, no `.show()`).

### E. chisurf MFDB wrapper for imaging (in `chisurf/plugins/ndxplorer`)

Mirror section B for images: `csc ndxplorer image --from-mfdb <id>` resolves the
source via `resolve_dataset_path`, runs the D-primitive, and with `--to-mfdb`
registers the exported image / parameter map (kind e.g. `"image"` /
`"processed_data"`) and any masked sub-selection under `--sample-id`, parented to
the source artifact.

### F. Round-trip parity

Both modes must produce outputs that re-open identically in the GUI and resolve
through `mfdb.datasets.open`, so `raw+sample → BS → ndXplorer filter` and
`raw(+sample) → ndXplorer image` work the same in scripts and by hand.

## Tasks

1. `modules/ndxplorer`: headless `filter` subcommand (open burst selection →
   parameter gates → write filtered burst folder), no GUI shown; JSON stdout.
2. `modules/ndxplorer`: headless `image` subcommand (load image-axis data →
   render map → apply gate/ROI → export image/parameter map and/or masked
   sub-selection), no GUI shown; JSON stdout.
3. Define and document the canonical parameter/column ids usable in `--select`
   and the supported image maps (`intensity`, `lifetime`, named parameters).
4. `chisurf/plugins/ndxplorer`: `filter` and `image` CLI wrappers doing MFDB
   resolve → primitive → optional MFDB write-back under the sample; JSON stdout.
5. Tests: (a) primitive filter on a bundled burst folder
   (`modules/ndxplorer/test/mfd/...`) asserting `n_out < n_in` for a tight gate
   with index-linkage preserved; (b) primitive image render on a CLSM/MFD-HDF5
   fixture asserting a non-empty map of the expected shape and that an ROI mask
   reduces `n_selected_px`; (c) chisurf round trips for both modes under the
   hermetic harness.
6. Docs: a one-page recipe showing both "ndXplorer as a burst filter" and
   "ndXplorer headless imaging" pipelines from `raw+sample`.

## Definition of Done

- [ ] `ndxplorer filter --folder <burst_dir> --select "proximity_ratio:0.3-0.7"
      --out <dir>` writes a filtered burst selection headlessly (no window).
- [ ] `ndxplorer image --file <clsm.ptu> --map lifetime --out map.tiff` renders
      and exports a parameter map headlessly; `--roi`/`--select` masks it.
- [ ] `csc ndxplorer filter|image --from-mfdb <id> --to-mfdb --sample-id <id>`
      completes the MFDB round trip; outputs re-open in ndXplorer and resolve via
      `mfdb.datasets.open`.
- [ ] `modules/ndxplorer` has no chisurf imports; MFDB logic is chisurf-side only.
- [ ] Burst/masked outputs reference the same TTTR (photon-index linkage intact).
- [ ] Tests + the recipes pass.

## Definition of Clean

Reuse ndXplorer's existing open + selection + imaging core (no parallel gating or
rendering engine); keep `modules/ndxplorer` chisurf-free; MFDB
resolution/registration only in `chisurf/plugins/ndxplorer`; JSON stdout like the
BS CLI; identity/scope via the PRD-17 resolver for any MFDB write-back.

## Relationship

Completes the CLI leg of **PRD-28** (ndXplorer ↔ MFDB burst integration).
Complements **PRD-30** (Unix-pipe CLI composability) — PRD-30 makes the TTTR/burst
CLIs pipe-friendly; this PRD makes ndXplorer a headless filter stage usable in
those pipelines. Honors **PRD-23** (thin widgets / logic in the API, not the GUI)
and **PRD-17** (identity) for MFDB write-back.
