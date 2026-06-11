# Burst Selection Plot Crash/Empty-Plot Issue

## What the user sees

1. Clicking a file in the file list triggers a `RuntimeError: wrapped C/C++ object of type ViewBox has been deleted` inside `BurstSelectionTool.update_burst_plots()` at `gui/tool.py`. The stack trace shows the crash in `self.dt_plot.plot()` -> `ViewBox.addItem()`.

2. After various patches, the plots (MCS, Decay, Burst length, Filter) appear empty — only the dT (delta-macro-time) plot shows data. Or in some variants, only the Filter plot works.

## Root cause

### 1. Deleted ViewBox
The dt_plot's internal `PlotItem.viewBox` C++ object gets deleted (likely by Qt layout destruction during hot-reload, save/restore of dock layout, or aggressive tab-closing in `DockArea`). The Python `PlotWidget` wrapper still exists, but the underlying ViewBox is gone, so `plot()` crashes.

### 2. Docked / not docked confusion
MCS, Decay, and Burst length plots are **inside QSplitters** (not direct tabs). `DockArea._all_widgets` only stores top-level widgets (tabs + splitters), not the page widgets inside splitters.  
- `DockArea.indexOf(plot)` returns -1 for these plots.  
- `DockArea._find_widget_by_key()` could not find them during layout restore.  
- Our own `_plot_widget_is_docked()` also failed to find them.

After restart, the saved dock layout could not be fully restored: MCS/Decay/Burst plots were silently dropped.

### 3. Deleted widget objects
After hot-reload, old `PlotWidget` Python wrappers point to deleted C++ widgets.  
- `sip.isdeleted(old)` is true.  
- `_replace_dock_widget()` early-returns without inserting the new widget.  
- The new `PlotWidget` ends up detached from the layout → invisible.

## What we tried / What is in the current code

**`chisurf/plugins/burst/burst_selection/gui/tool.py`**

| Method | What it does |
|--------|-------------|
| `_is_qt_object_deleted()` | Uses `qt_sip.isdeleted()` to detect stale Qt wrappers. |
| `_plot_widget_is_usable()` | Checks whether `PlotWidget.getPlotItem().vb` is alive. |
| `_plot_widget_is_docked()` | Returns True if the widget should be visible: either it is in the dock tree, OR the diagnostic feature is `initial_enabled` and not in `_closed_diagnostic_plots`. |
| `_ensure_plot_widget()` | Recreates a `PlotWidget` if its ViewBox is dead. If the old widget was deleted, calls `DockArea.addTab()` to re-add it. Otherwise delegates to `_replace_dock_widget()`. |
| `_replace_dock_widget()` | Replaces a widget in a DockArea tab, a QSplitter, or a plain QLayout. Updates `_all_widgets`/`_tab_names` bookkeeping. |
| `_replace_dock_tab()` | Replaces within a `DockTabWidget` preserving tab order and tooltips. |
| `_remove_plot_item()` | Removes a `PlotDataItem` while ignoring deleted wrappers. |
| `update_burst_plots()` | Calls `_ensure_dt_plot()` then `_ensure_plot_widget()` for each diagnostic plot that is docked. |

**`chisurf/gui/widgets/dock_area/dock_area.py`**

| Method | What is patched |
|--------|-----------------|
| `_iter_nested_widgets()` | New helper that yields a widget and all page widgets inside its nested `DockSplitter` children. |
| `_find_widget_by_key()` | Now searches nested splitter children via `_iter_nested_widgets()`. Without this, layout restore could not find MCS/Decay/Burst plot widgets. |
| `_find_widget_by_tab_name()` | Same nested search. |

## Still not right

Despite all patches, after restart only the Filter (direct tab) plot displays. The root cause is **not isolated** — three interacting problems produce different symptoms depending on restart vs hot-reload:

1. **`DockArea` bookkeeping** — `_all_widgets` does not include page widgets inside splitters. Many methods (`count`, `currentIndex`, `widget`, `indexOf`, `_find_*`) assume all page widgets are top-level tabs. A proper fix would add every page widget to `_all_widgets` regardless of nesting depth and adjust `addTab`/`removeTab` to handle nested insertion. This is invasive.

2. **QSettings stale layout state** — The saved layout may refer to widgets by stored tab names that no longer match after hot-reload. Our nested search may still fail if `_tab_names` mapping is stale.

3. **`_plot_widget_is_docked()` heuristic** — Using `initial_enabled + not closed` as fallback is fragile: it doesn't account for the user having manually rearranged or hidden tabs.

## Recommended approach for the next person

1. **Simplify**: Instead of replacing stale plot widgets, rebuild the entire dock area on first `update_burst_plots` call if any ViewBox is dead. This avoids partial-bookkeeping bugs.

2. **Or, fix `DockArea` hierarchy**:
   - Ensure every page widget is registered in `_all_widgets` (including splitter children).
   - Change `findChildren(DockTabWidget)` calls to walk `_root_widget` instead.
   - Add `DockArea.insertTab(index, widget, name)` to preserve tab order after replacement.

3. **Remove QSettings duck-layout entirely** if it keeps causing more problems than it solves.

4. **Test with `QT_QPA_PLATFORM=offscreen`** (macOS needs `QT_QPA_PLATFORM_PLUGIN_PATH` pointing at `lib/python3.10/site-packages/PyQt6/Qt/plugins` or conda's `qt-main` plugins). Instantiate `BurstSelectionTool`, call `update_burst_plots()` with dummy data, check each plot has items.
