# Batch-Analysis

Apply one **pre-optimised template fit** to many datasets or files in a single
pass and export the consolidated results.

Before batch processing, load a representative dataset, create a fit for it and
**manually optimise its parameters** — those parameter values seed every run, so
good starting values are what make the batch results reliable.

## Workflow (GUI wizard)

1. **Welcome** — overview.
2. **Loaded data** *(optional)* — tick datasets already loaded in ChiSurf.
3. **Files & fit** — drop files (or a folder) and pick the template fit.
4. **Run** — choose a results CSV path and run. Each item is restored to the
   template parameters, fitted, and its parameters + reduced χ² recorded.
5. **Results** — a per-parameter results table.

Alongside the CSV the tool writes a **DOCX report** (per-item screenshots +
consolidated table) and a **ZIP** of the per-run numeric exports.

## Architecture (new plugin standard)

```
batch_analysis/
  manifest.json           plugin metadata + gui/cli entrypoints
  core/runner.py          Qt-free batch runner + CSV/DOCX/ZIP exporters
  gui/view_model.py       BatchViewModel (state, info sources, run action)
  gui/tool.py             AutoForm host (BatchProcessingWizard)
  gui/loaded_datasets.py  embedded dataset check-list widget
  batch.view.json         declarative wizard layout (AutoForm)
  cli/main.py             `batch-analysis run` / `batch-analysis report`
  test/                   headless tests (no Qt / no live session)
```

The numeric work lives entirely in `core/runner.py` (no Qt), so it is exercised
by headless tests and reused by both the GUI and the CLI.

## CLI

```bash
# from a ChiSurf session (csc), fit a set of files with template fit 0
csc batch-analysis run --list-fits
csc batch-analysis run --file a.sm --file b.sm --fit-index 0 -o results.csv

# regenerate a DOCX report from an existing results CSV (fully headless)
csc batch-analysis report results.csv --docx report.docx
```
