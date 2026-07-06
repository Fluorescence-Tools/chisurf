---
type: PRD
prd: "31"
title: "PRD-31: Headless CLI for the Companion Photon-Data Exploration Tool"
description: Adds a windowless CLI to the companion exploration tool for parameter-based burst filtering and imaging, integrated with MFDB.
status: planned
phase: "2"
resource: chisurf/plugins/ndxplorer
tags: [prd, imaging, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Gives the companion photon-data exploration tool a headless CLI for its two core jobs, both previously GUI-only: burst filtering (select a subset of bursts by parameter ranges/gates and emit a filtered burst selection) and imaging (render intensity or per-pixel parameter maps from image-axis/CLSM data, apply gates/ROIs, and export images or a masked sub-selection). Both run with no window, print JSON to stdout, and complete the MFDB round trip. Pure primitives live in the chisurf-free external module; MFDB resolution and write-back live in the ChiSurf-side wrapper. Filtered/masked outputs stay a reference beside the same TTTR so photon-index linkage is preserved.

# Status
Planned overall, though implementation of the headless CLI landed 2026-06-27: the two primitives and both wrappers exist; the `image` MFDB round trip is implemented but not yet covered by a ChiSurf-side test. A separate, completed (2026-06-24) GUI-migration variant carrying the same number is folded below as "# Companion: pyqtgraph migration".

## Implementation state (headless CLI, 2026-06-27)

Both primitives + both chisurf wrappers landed.

- **Primitives (chisurf-free, `modules/ndxplorer/ndxplorer/cli.py`):**
  `ndxplorer filter` (burst folder → `--select`/`--query` gates → filtered
  `.bst` folder referencing the same TTTR) and `ndxplorer image` (HDF5/burst
  image axes → intensity or named-parameter mean map → float32 TIFF / normalized
  PNG, with `--roi` TIFF mask + `--select` gates + `--out-selection` masked
  sub-selection). JSON to stdout; registered as click subcommands in
  `__main__.py` with pre-import offscreen handling.
- **chisurf wrappers (`chisurf/plugins/ndxplorer/cli.py`, `cli_entrypoint`
  `ndxplorer=…:cli` so `csc ndxplorer …` works):** resolve `--from-mfdb` via
  `resolve_database_path`, call the primitive over subprocess (keeps the submodule
  chisurf-free), and with `--to-mfdb` register the result via `register_result`
  (`operation_type="burst_filter"`, parented to the source, under `--sample-id`).
- **No chisurf imports in `modules/ndxplorer`** (DoD constraint verified).
- **Tests green:** `modules/ndxplorer/ndxplorer/tests/test_cli.py` (5 — filter
  gate, filter query, image intensity, image lifetime-TIFF, image ROI mask) +
  `test/fio/test_ndxplorer_cli.py` (chisurf filter round trip, 1).
- **Recipe doc:** `docs/ndxplorer_headless_cli.md` (both pipelines from raw+sample).

**Remaining gap:** the `csc ndxplorer image` MFDB round trip is implemented but
not yet covered by a chisurf-side test (needs an image artifact fixture under the
hermetic harness); the `filter` round trip is tested. Everything else in the DoD
is met.

# Goal

Give the companion exploration tool a **headless CLI** for its two main jobs,
both otherwise GUI-only:

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

# 2) Burst Selection  ->  exploration tool  (THIS PRD)
ndxplorer filter --from-mfdb <output_folder_artifact_id> --db mfdb.sqlite \
    --select "proximity_ratio:0.30-0.70" --select "n_photons:50-" \
    --to-mfdb --sample-id <sample_id>
#   => writes a FILTERED burst selection (a new burst folder / burst_table)
#      referencing the same TTTR files, registered under the same sample.
```

# Why

The companion exploration tool is chisurf's burst/MFD **and imaging** explorer:
it ingests burst selections (`open_files(file_type="burst_dir", …)`), does
parameter-based gating (column selection, 1D/2D histograms, polygon/range gates),
and renders image-axis data (FLIM / intensity / per-pixel parameter maps with
pixel ROI selection). But its **CLI is only a GUI launcher**:

`modules/ndxplorer/ndxplorer/__main__.py:main` accepts `--file/--folder/
--test-data/--processed-data-id/--experiment-id/--zmq-port/--debug` and then
calls `win.show(); app.exec_()`. There is no headless mode that loads data,
applies a gate/ROI, and writes a filtered selection or an exported image. So
neither the burst-filter nor the imaging workflow can participate in scripted/CI
pipelines or in the MFDB round trip non-interactively.

# Current state (verified)

- **BS → MFDB works headlessly.** `csc burst-selection analyze … --mfdb`
  registers raw inputs linked to a sample, burst tables, and a single
  output-folder `external_reference` (`data_format="directory"`). Verified on
  `bh_spc132_sm_dna`: 2 inputs, 2 burst tables, 1 group; the group resolves via
  `MFDatabase.open_dataset` (what `mfdb.datasets.open` / the launcher call) to
  the on-disk burstwise folder containing `m000.bur`/`m001.bur` co-located with
  the TTTR files.
- **`.bur` files reference photons by index in the original TTTR file** (PRD-28).
  A filtered selection must therefore stay a *reference next to the same TTTR*
  (a burst sub-selection / new `.bur` folder beside the TTTR), never an
  object-store copy — otherwise the photon-index linkage breaks.
- **Shared-DB requirement (learned in testing).** The in-process `MFDBClient`
  used by the chisurf-side launcher opens the **configured** database, not an
  arbitrary `--db` path. For a real round trip, BS and the exploration tool must
  point at the *same* MFDB (use the configured DB, or thread the same DB path
  through both CLIs). The headless CLI must accept an explicit DB/source so it
  reads what BS wrote.

# Existing pieces to reuse (do not reinvent)

- **Ingest:** `NDXplorer.open_files(file_type="burst_dir",
  file_handles=<path>, append=False)` and
  `ndxplorer.__main__.open_path_like_drop(ndx, path)`.
- **Parameters/gating:** the column/parameter model and selection
  (`ui/column_selection_dialog.py`, `ui/parameter_editor.py`, the gate/region
  logic behind the GUI histograms). The headless filter should call the same
  selection core the GUI uses — not a parallel reimplementation.
- **Imaging:** the image-axis handling and pixel selection it already has —
  `core/plot_main.py:check_and_set_image_axes`, the pixel ROI/brush masking in
  `utils/mask_helpers.py` (`create_pixel_radius_brush_kernel`, …), and the
  report-tool image rendering (`report_tool.py:_update_image_preview`,
  `_open_report_images_for_folder`). The headless imaging mode must drive this
  same machinery, not a second imaging stack.
- **chisurf glue (keep `modules/ndxplorer` chisurf-free):**
  `chisurf/plugins/ndxplorer/mfdb_launcher.py` already resolves an MFDB artifact
  to a local path (`resolve_dataset_path`) and opens it. MFDB write-back can
  reuse `chisurf.core.mfdb.result_registry.register_result` / `BurstMFDBPipeline`.
- **BS CLI `--mfdb`** — the upstream half; the contract is the printed
  `mfdb_artifacts.output_folder` artifact id.

# Constraints

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

# Design

## A. Pure filter primitive (in `modules/ndxplorer`, chisurf-free)

Add a headless subcommand, e.g. `ndxplorer filter`:

- Inputs: `--folder/--file` (a burst selection) and one or more `--select
  "<param>:<min>-<max>"` gates (open-ended `min-` / `-max` allowed), plus
  `--out <dir>` for the filtered burst folder.
- Loads the burst selection through the same path as a GUI drop
  (`open_path_like_drop` / `open_files`), reads the per-burst parameter table,
  evaluates the gates (AND across `--select`, range per parameter), and writes a
  filtered burst folder referencing the same TTTR (the surviving bursts only).
- Emits JSON: `{"input": …, "n_in": …, "n_out": …, "selected": …, "out": …}`.
- Parameter names should match the tool's column ids (document the canonical set
  — proximity ratio, stoichiometry, lifetime, duration, count rate, n_photons).

## B. chisurf MFDB wrapper (in `chisurf/plugins/ndxplorer`)

Add a chisurf-side CLI (e.g. `csc ndxplorer filter`) that:
1. Resolves `--from-mfdb <artifact_id>` (or `--folder`) to a local burst path via
   `resolve_dataset_path` (the configured/`--db` database).
2. Calls the A-primitive to produce a filtered burst folder beside the TTTR.
3. With `--to-mfdb`, registers the filtered folder as a new burst selection
   (operation_type e.g. `"burst_filter"`, parent = the source burst selection,
   linked to `--sample-id`), reusing `register_result`/the burst pipeline so the
   single-group + real-name conventions (PRD-28) hold.
4. Prints the new artifact id / path as JSON.

## D. Imaging primitive (in `modules/ndxplorer`, chisurf-free)

The tool is also used for **imaging** (FLIM / intensity / per-pixel parameter
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

## E. chisurf MFDB wrapper for imaging (in `chisurf/plugins/ndxplorer`)

Mirror section B for images: `csc ndxplorer image --from-mfdb <id>` resolves the
source via `resolve_dataset_path`, runs the D-primitive, and with `--to-mfdb`
registers the exported image / parameter map (kind e.g. `"image"` /
`"processed_data"`) and any masked sub-selection under `--sample-id`, parented to
the source artifact.

## F. Round-trip parity

Both modes must produce outputs that re-open identically in the GUI and resolve
through `mfdb.datasets.open`, so `raw+sample → BS → filter` and
`raw(+sample) → image` work the same in scripts and by hand.

# Tasks

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
6. Docs: a one-page recipe showing both "exploration tool as a burst filter" and
   "headless imaging" pipelines from `raw+sample`.

# Definition of Done

- [x] `ndxplorer filter --folder <burst_dir> --select "proximity_ratio:0.3-0.7"
      --out <dir>` writes a filtered burst selection headlessly (no window).
- [x] `ndxplorer image --file <clsm.ptu> --map lifetime --out map.tiff` renders
      and exports a parameter map headlessly; `--roi`/`--select` masks it.
- [~] `csc ndxplorer filter|image --from-mfdb <id> --to-mfdb --sample-id <id>`
      completes the MFDB round trip; outputs resolve via `mfdb.datasets.open`.
      **filter tested; image wrapper implemented but not yet tested.**
- [x] `modules/ndxplorer` has no chisurf imports; MFDB logic is chisurf-side only.
- [x] Burst/masked outputs reference the same TTTR (photon-index linkage intact).
- [x] Tests + the recipes pass.

# Definition of Clean

Reuse the tool's existing open + selection + imaging core (no parallel gating or
rendering engine); keep `modules/ndxplorer` chisurf-free; MFDB
resolution/registration only in `chisurf/plugins/ndxplorer`; JSON stdout like the
BS CLI; identity/scope via the PRD-17 resolver for any MFDB write-back.

# Companion: pyqtgraph migration

A separate, completed (2026-06-24) GUI-migration effort carried the same PRD
number. It replaced a legacy Qt plotting toolkit (and, as a side effect, most
matplotlib usage) in the exploration tool's GUI with **pyqtgraph** (already a
dependency), unifying on a single actively-maintained visualization stack,
removing a heavy unmaintained C-extension chain, and reducing startup time.
This migration is **done**.

## Motivation

The tool's GUI previously depended on **three** plotting stacks:

| Stack | Used for | Status |
|-------|----------|--------|
| **legacy Qt plotting toolkit** | Marginal 1D histograms, optional 2D backend, curve overlays, colormaps, fixed image item | Heavy, unmaintained upstream, not in pyproject.toml (conda-only) |
| **matplotlib** | Colormap application, report generation, color generation | Heavy, GUI-irrelevant for most operations |
| **pyqtgraph** | UMAP 2D/3D scatter, parameter-tree editor | Already present, minimal, fast |

The legacy toolkit was not listed in `pyproject.toml` (only `environment.yml`),
was wrapped in try/except across 8+ files, added unnecessary complexity, and
blocked standalone packaging. Consolidating on pyqtgraph removed an entire
C-extension dependency chain, unified on one library, reduced startup time, and
made colormaps/histograms/curves/images consistent.

## What was migrated

1. **Marginal 1D histograms (highest priority).** Standalone legacy curve-dialog
   windows for the `g_xplot`/`g_yplot`/`g_zplot` 1D projections became embedded
   `pg.PlotWidget` dock widgets, each showing the marginal histogram as a
   `PlotDataItem` with fill under the curve; axis sync via `PlotItem.setXLink`/
   `setYLink` instead of manual sync in `plot_helpers.py`.
2. **2D histogram display.** The two former backends (a custom `QPainter`-based
   `SimpleImageWidget` using matplotlib colormaps, and the legacy-toolkit image
   dialog + fixed image item, selected by `NDXPLORER_2D_BACKEND`) were replaced
   by a single pyqtgraph backend: `pg.ImageItem` inside a `pg.PlotWidget` with
   `setLevels`, `pg.colormap.*`, native axes/zoom/pan, `LinearRegionItem` for
   slice/marginal selection, and `ROI` for rectangular/elliptical regions. This
   eliminated `SimpleImageWidget` and simplified `DrawingOverlayWidget` (overlay
   on a `GraphicsView` instead of raw `QPainter` coordinates).
3. **Curve overlays.** `curve_overlay.py`'s legacy curve plot/items became a
   `pg.PlotWidget` + `pg.PlotDataItem` overlay (transparent background, stacked
   over / sharing axes with the main 2D image); the `CurveEvaluator` logic (which
   just produces (x, y) arrays) stayed unchanged; progress signals use native Qt
   signals.
4. **Colormaps.** `colormaps.py` switched from the legacy toolkit's colormap
   list/LUT (+ matplotlib fallback) to `pyqtgraph.colormap` (`pg.colormap.get`,
   `listMaps`, custom `pg.ColorMap(pos, color)`), which ships viridis/magma/
   inferno/plasma/jet/gray etc. and supports user-defined maps — zero legacy or
   matplotlib imports.
5. **Fixed image item.** The legacy-toolkit `FixedImageItem` (with matplotlib
   colormap fallback) was eliminated; `pg.ImageItem` handles display with
   built-in colormap support.
6. **Report generation (matplotlib) kept, deliberately.** `report_tool.py` still
   uses matplotlib for publication-quality PNG report figures — an output-format
   choice, not a GUI dependency. matplotlib remains a dependency for the report
   tool but is **no longer required for GUI operation**.

Files created: `plotting/pg_image_widget.py` (`PGImageWidget` — pyqtgraph 2D
image display with colormap/levels/crosshair, self-contained < 200 lines).
Files modified: `plotting/plot_helpers.py`, `plotting/colormaps.py`,
`plotting/curve_overlay.py`, `plotting/image_items.py`, `core/plot_main.py`,
`pyproject.toml`, `environment.yml`; removed `plotting/simple_image_widget.py`
and the `NDXPLORER_2D_BACKEND` env var.

## Migration outcome (Definition of Done — all met)

- [x] Marginal 1D histograms render in `PlotWidget` dock widgets with linked axes.
- [x] 2D histogram image renders with correct colormap, levels, aspect ratio.
- [x] Curve overlays render and update correctly.
- [x] `NDXPLORER_2D_BACKEND` removed — single pyqtgraph backend.
- [x] The legacy Qt plotting toolkit is no longer imported anywhere in the codebase.
- [x] matplotlib is no longer imported during normal GUI operation (only in
      `report_tool.py`).
- [x] All existing colormaps still work (same visual output).
- [x] Startup time noticeably faster (no legacy-toolkit import).
- [x] Tests pass; no behavioural change visible to the end user.

**Non-goals of the migration:** removing matplotlib from the report tool;
changing the UMAP visualization (already pyqtgraph); changing the data
processing pipeline (only the *display* of results changed); API breakage for
the companion imaging-viewer plugin or external importers (internal plotting
only). **Remaining out-of-scope work tracked separately:**
`chisurf/gui/plots/surfaceplot/` and `chisurf/gui/plots/global_tcspc/` still use
the legacy toolkit (not part of the exploration tool).

# Relationships
- Completes the CLI leg of [PRD-28](prd-28.md) (companion-tool ↔ MFDB burst integration).
- Complements [PRD-30](prd-30.md) (Unix-pipe CLI composability) by making the exploration tool a headless filter/imaging stage.
- Honors [PRD-23](prd-23.md) (thin widgets) and [PRD-17](prd-17.md) (identity) for MFDB write-back.
- Registers results via [PRD-03](prd-03.md) result registry; resolves through [MFDB (current)](/architecture/mfdb.md) and the [plugin system](/architecture/plugin-system.md).
