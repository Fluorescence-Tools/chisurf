---
type: Reference
title: Known issues & recurring gotchas
description: Curated open functional bugs and cross-cutting engineering pitfalls distilled from working bug logs.
resource: chisurf/
tags: [bugs, gotchas, pitfalls, gui, qt]
timestamp: '2026-07-06T00:00:00Z'
---

# Purpose

This concept distills durable content out of ad-hoc working bug logs (formerly in
a top-level `notes/` folder) so the knowledge survives without keeping a raw log
in the tree. Two things are worth preserving from those logs: **functional bugs
that were still open** when the log was last touched, and **recurring root-cause
patterns** that keep re-biting across subsystems.

This is not a live issue tracker. Architecture-divergence items (spec violations,
review-found correctness bugs, schema/manifest drift) live in the
[cleanup backlog](/specs/assessment.md); this page is for runtime/functional
behaviour. The open list below was captured around **June 2026** — re-verify each
against the current tree before acting, as some may already be fixed.

# Recurring gotchas (durable — verify before you re-break them)

These are the patterns; each caused more than one bug.

- **Qt Python-wrapper vs C++ lifetime.** `RuntimeError: wrapped C/C++ object …
  has been deleted` on window close. The Python wrapper outlives the destroyed
  C++ widget; a shared `_new_closeEvent` in `chisurf/gui/misc_helpers.py` calls
  through to a dead object. Any close-event / save-window-state handler must
  guard against an already-deleted widget (`sip.isdeleted`). Recurs across
  plugins (microtime histogram, MLE lifetime wizard).
- **`@property`-backed reader settings silently drop on serialize.**
  `serialize_reader_state()` / `_reader_settings_dict()` iterate
  `reader.__dict__` and skip `_`-prefixed keys. A `@property` (e.g. `is_jordi`)
  is not in `__dict__`; only its private backing field is, and that is filtered
  out — so the setting fails to persist across restart with no error. Any new
  `@property` on an `ExperimentReader` needs the serializer to look for the
  public property. PRD-40 aims to make AutoForm/`.view.json` own persistence and
  remove this class of bug.
- **`DataCurve[key]` returns 5 items, not 4** (since masks were added). Every
  consumer (`calculate_weighted_residuals`, `DataCurve.save`) must unpack 5. A
  `Curve.__getitem__` regression that returned `y` where `x` was expected
  produced huge residuals / wrong χ² rather than a crash — easy to miss.
- **Refresh derived outputs explicitly.** Updating input parameters and plots
  does *not* refresh derived/output parameters. After any fit switch or parameter
  edit, call `model.finalize()` and refresh the output-parameter controllers.
- **Propagate anisotropy calibration as a unit.** `g_factor`, `l1`, `l2` must
  travel together reader → dataset/group metadata → model kwargs → parameters;
  propagating `g_factor` alone was a repeated bug. Values may be scalar floats or
  parameter objects — handle both (no `.value` on a float). Coerce NaN/inf to
  finite before constructing parameters (non-finite calibration caused Windows
  access-violation crashes on project load). Steady-state anisotropy uses
  `r = (VV - VH)/(g*VV + 2*VH)` and should prefer dual-channel VV/VH.
- **pyqtgraph text overlays + threading.** Before `setHtml`/`updateTextPos` on a
  `TextItem`, verify both it and its backing `QGraphicsTextItem` are alive
  (`sip.isdeleted`); dangling items crash on Windows. Prefer plain `setText`.
  Qt-owning objects (MCP server, GUI executor) must be created/owned/destroyed on
  the main thread.
- **macOS ARM64 text rendering.** Emojis/special symbols in UI strings can hit
  CoreText paths that SIGBUS (`EXC_ARM_DA_ALIGN`); keep UI strings ASCII-safe on
  that platform. (Note this tension with the general emoji-toolbutton preference.)
- **Keep hot-path logging at DEBUG.** Logging a full DataFrame repr / per-redraw
  traces at INFO on every recompute was itself a major interactive slowdown.
  `resizeEvent` should rescale cached data (debounced), never re-bin/recompute.
- **File-save dialogs seed from the data location.** Default the save dir to the
  loaded data file's folder / `cs.working_path` (not Qt's last-used dir) and
  update `cs.working_path` after save. `chisurf/gui/main.py` "Save Fit" is the
  reference.
- **Plugin menu-callback contract.** The menu system `exec`s a plugin file with
  `__name__ == "plugin"`; without an explicit `if __name__ == "plugin": load()`
  guard the plugin silently does nothing.
- **Plugin `.ui` paths must be plugin-relative and existence-checked** (cached
  resolver) before `uic.loadUi`; stale relative paths break silently after folder
  moves. Contract tests cover this — keep them.
- **Batch/action dispatch must not swallow per-item exceptions.** A batch
  `fit.add` that swallowed callback errors reported success while creating no
  fits. Route default actions to `cs.current_fit`, not fit index 0.
- **macOS QComboBox popups render translucent.** Node/embedded combos need an
  explicit opaque palette + `setAutoFillBackground(True)` and cleared
  `WA_TranslucentBackground` on the view; QSS `QComboBox QAbstractItemView`
  alone does not fix the popup window. (Seen in the light-path node editor — see
  [Light Path Simulator](/plugins/profiles/lightpath-simulator.md).)

# Open functional issues (re-verify against current tree)

Grouped by area; captured June 2026.

**Project save / restore**
- Project round-trip: decay/lifetime models do not save & reload (critical).
- Save→close→open resolves the active window via `chisurf.cs` after teardown
  removed it → `AttributeError: module 'chisurf' has no attribute 'cs'`.
- Dataset removal no longer confirms and closes dependent fits.
- TCSPC "stacked" (`is_jordi`) not persisted across restart (see the
  `@property` gotcha above).
- Undo/redo does not destroy stashed dataset/fit UI windows
  (`history_replay.apply_entity_lifecycle`) — stale windows linger.

**Fit creation / fitting**
- Access violation (`0xC0000005`) on fit creation after dataset load, in
  TCSPC/anisotropy chinet init — keep parameter init scalar/type-safe into the
  native layer (no catchable Python exception at the crash boundary).
- FCS MaxEnt fit does not autorange on creation → zero points in range →
  non-computable out of the box.
- Parameter-widget edits do not fully propagate to model/chinet.

**Plugins / tools**
- NDXplorer stays blocked after data load; clear-then-reload shows no data — a
  proper `reinit()` (reset caches, plot objects, combos) is needed rather than a
  partial `clear_plots`.
- TraceBrowser does not populate its file list on folder open (severe).
- Correlator/FCS channel-preset saving does not work.
- TTTR Image Browser exports to the wrong folder; anisotropy "Save CSV" ignores
  the loaded-data location (see file-save gotcha).
- BH / SPC-130 micro-time resolution is wrong and not corrected on load (data
  correctness).
- mfdb-admin: cancelling the password prompt still opens the UI.

**Environment**
- Built-in Jupyter/notebook integration is disabled/broken; the notebook menu is
  missing from the ribbon.

# Deferred enhancements

- **Per-item tooltips in the metadata-key combobox** (`chisurf/gui/plots/
  fitinfo.py`, keys from `chisurf/core/fio/mmcif/db/pdbx_metadata.py`). Multiple
  Qt approaches (`Qt.ToolTipRole`, `QStandardItem.setToolTip`, delegates, event
  filters) either broke selection or showed no tooltip on a ~6.7k-item model.
  Deferred; descriptions are shown as the combo/value-cell tooltip after a key is
  chosen. If revisited, try a custom popup `QListView`/delegate applied *after*
  `showPopup()`, or a side-panel hint instead of dropdown tooltips.
