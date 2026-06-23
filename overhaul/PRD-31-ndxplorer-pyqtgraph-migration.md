# PRD-31: ndXplorer GUI — Replace guiqwt / matplotlib with pyqtgraph

## Goal

Replace **guiqwt** (and as a side effect most **matplotlib** usage) in ndXplorer
with **pyqtgraph**, which is already a dependency. This eliminates the heavy
guiqwt + qwt stack, removes matplotlib as a runtime GUI dependency, and makes
ndXplorer's visualization stack uniform (pyqtgraph only).

## Motivation

ndXplorer currently depends on **three** plotting stacks:

| Stack | Used for | Status |
|-------|----------|--------|
| **guiqwt** (+ qwt) | Marginal 1D histograms, optional 2D backend, curve overlays, colormaps, `FixedImageItem` | Heavy, unmaintained upstream, not in pyproject.toml (missing dep) |
| **matplotlib** | Colormap application, report generation, color generation | Heavy, GUI-irrelevant for most operations |
| **pyqtgraph** | UMAP 2D/3D scatter, `ParameterTree` editor | Already present, minimal, fast |

guiqwt is:
- **Not listed in pyproject.toml** — only in `environment.yml` (conda)
- **Heavily wrapped with try/except** throughout the codebase (8+ files)
- **Unnecessary complexity** — marginal histograms, curve overlays, and 2D image
  display are all things pyqtgraph does well, faster, and with fewer lines
- **A blocker for standalone packaging** — guiqwt/qwt are hard to distribute

Replacing guiqwt + matplotlib with pyqtgraph:
- Removes an entire C-extension dependency chain (qwt, guiqwt)
- Unifies on a single actively-maintained visualization library
- Reduces startup time (fewer imports)
- Makes colormaps, histograms, curves, and images consistent

## Design

### 1. Marginal 1D histograms (highest priority)

**Current:** `plot_helpers.py` creates `CurveDialog` (guiqwt) instances for
`g_xplot`, `g_yplot`, `g_zplot` — three standalone windows showing 1D
projections of the 2D histogram.

**Target:** Replace each `CurveDialog` with a `pyqtgraph.PlotWidget` embedded
in the main window layout or a lightweight `QDockWidget`. Each plot shows the
same marginal histogram data as a `PlotDataItem` with fill under the curve.

| Current (guiqwt) | Target (pyqtgraph) |
|---|---|
| `CurveDialog` (standalone window) | `PlotWidget` in a `QDockWidget` |
| `guiqwt.curve.CurveItem` | `pg.PlotDataItem(pen, fillLevel=0)` |
| `guiqwt.styles.CurveStyle` | `pg.mkPen(color, width)` |
| Manual axis sync in `plot_helpers.py` | `PlotItem.setXLink` / `setYLink` for syncing |

### 2. 2D histogram display (image)

**Current:** Two backends controlled by `NDXPLORER_2D_BACKEND`:
- `"simple"` (default): `SimpleImageWidget` — custom `QWidget` using
  `QPainter.drawImage` with matplotlib colormaps
- `"guiqwt"`: `guiqwt.image.ImageDialog` + `FixedImageItem`

**Target:** A single pyqtgraph backend. Replace both with `pg.ImageItem`
inside a `pg.PlotWidget`:

- `ImageItem.setImage(data)` + `setLevels(vmin, vmax)` for display
- `pg.colormap.*` for colormap application (replaces matplotlib colormaps)
- `PlotWidget` handles axes, zoom, pan natively
- `LinearRegionItem` for slice/marginal region selection
- `ROI` for rectangular/elliptical selection regions

This eliminates `SimpleImageWidget` entirely and makes `DrawingOverlayWidget`
simpler (overlay on a `GraphicsView` instead of raw `QPainter` coordinates).

### 3. Curve overlays

**Current:** `curve_overlay.py` uses:
- `guiqwt.curve.CurvePlot` / `guiqwt.curve.CurveItem` for the overlay plot
- `guiqwt.signals` for curve computation signals

**Target:** `pg.PlotWidget` + `pg.PlotDataItem`:
- Each user-defined curve becomes a `PlotDataItem` added to the overlay plot
- The overlay plot is stacked on top of (or shares axes with) the main 2D image
- `CurveEvaluator` logic stays unchanged — it produces (x, y) arrays regardless
- Signal/slot for progress uses Qt's native signals, not guiqwt-specific ones

### 4. Colormaps

**Current:** `colormaps.py` uses:
- `guiqwt.colormap.get_colormap_list()` for the colormap combo
- `guiqwt.colormap.QwtLinearColorMap` for LUT generation
- `matplotlib.pyplot.get_cmap()` as fallback

**Target:** `pyqtgraph.colormap`:
- `pg.colormap.get(name)` for built-in colormaps
- `pg.colormap.listMaps()` for the combo box
- Custom colormaps via `pg.ColorMap(pos, color)` 
- `colormaps.py` can be rewritten to have zero imports from guiqwt or matplotlib

Note: pyqtgraph ships with a solid set of default colormaps (viridis, magma,
inferno, plasma, jet, gray, etc.) and supports user-defined ones.

### 5. FixedImageItem

**Current:** `image_items.py` — `FixedImageItem` inherits from
`guiqwt.image.ImageItem`, with matplotlib colormap fallback. This is only used
when the guiqwt backend is active.

**Target:** Eliminate `FixedImageItem`. When the pyqtgraph backend is used,
`pg.ImageItem` handles image display with built-in colormap support. The
`__init__.py` export of `FixedImageItem` can be deprecated or aliased to a
lightweight adapter.

### 6. Report generation (matplotlib)

**Current:** `report_tool.py` uses `matplotlib.pyplot` figures for generating
report images (PNG export of histograms with annotations).

**Target:** Keep matplotlib for report generation only — it is a deliberate
output format choice (publication-quality figures), not a GUI dependency.
This is a **non-goal** of this PRD; matplotlib remains in dependencies for
the report tool but is **no longer required for GUI operation**.

## Implementation plan

### Phase 1: Marginal 1D histogram replacement

1. **Replace `CurveDialog` with `PlotWidget`** in `plot_helpers.py`:
   - `g_xplot`, `g_yplot`, `g_zplot` become `QDockWidget` instances containing
     `pg.PlotWidget`
   - Data update uses `plot_item.setData(x, y)` instead of `curve.setData()`
   - Axis synchronisation uses `PlotItem.setXLink()` / `setYLink()`
   - Style parity: matching colors, fill under curve, axis labels

2. **Remove guiqwt import chain** from `plot_helpers.py`:
   - `_ensure_guiqwt()` no longer needed for marginal plots
   - Remove `QwtPlot`, `QwtPlotCanvas`, `CurveDialog` references

### Phase 2: 2D image backend replacement

3. **Add pyqtgraph image widget** as a new backend:
   - Create `plotting/pg_image_widget.py` with a `PGImageWidget` class that
     wraps `pg.PlotWidget` + `pg.ImageItem`
   - Support the same API as `SimpleImageWidget` (set_data, levels, colormap,
     aspect ratio, crosshair cursor)

4. **Replace colormap backend** in `colormaps.py`:
   - Switch from `guiqwt.colormap` + matplotlib to `pyqtgraph.colormap`
   - Keep the same fallback behaviour but with pyqtgraph as primary
   - Ensure the colormap combo box lists pyqtgraph maps

5. **Remove `SimpleImageWidget`** after the pyqtgraph backend is verified:
   - Remove `plotting/simple_image_widget.py`
   - Remove `NDXPLORER_2D_BACKEND` env var — only one backend
   - Remove matplotlib import from image display code

### Phase 3: Curve overlay replacement

6. **Replace `CurvePlot`/`CurveItem` with `PlotWidget`/`PlotDataItem`** in
   `curve_overlay.py`:
   - The overlay plot becomes a `pg.PlotWidget` with transparent background,
     stacked over the main 2D image (or sharing axes)
   - `CurveEvaluator` output feeds `PlotDataItem.setData()`
   - Remove guiqwt-specific signal connections

### Phase 4: Cleanup

7. **Remove guiqwt and qwt from all imports**:
   - `core/plot_main.py` — remove `_ensure_guiqwt()` call and lazy import
   - `plotting/image_items.py` — remove or deprecate `FixedImageItem`
   - `plotting/__init__.py` — clean up re-exports
   - `plotting/colormaps.py` — remove all guiqwt/matplotlib code paths

8. **Update dependencies** in `pyproject.toml`:
   - Remove `guiqwt` if listed (was never in pyproject.toml, only environment.yml)
   - Keep `matplotlib` only for report generation (`report_tool.py`)
   - `pyqtgraph` stays as-is

9. **Remove `environment.yml` guiqwt reference** — guiqwt is no longer required

## Files to modify

| File | Change |
|------|--------|
| `plotting/plot_helpers.py` | Replace `CurveDialog` with `PlotWidget` for marginal histograms |
| `plotting/plot_helpers.py` | Remove `_ensure_guiqwt()`, `QwtPlot`, `QwtPlotCanvas` |
| `plotting/colormaps.py` | Replace guiqwt/matplotlib colormaps with pyqtgraph colormaps |
| `plotting/curve_overlay.py` | Replace `CurvePlot`/`CurveItem` with `PlotWidget`/`PlotDataItem` |
| `plotting/simple_image_widget.py` | Remove (replaced by new pyqtgraph image widget) |
| `plotting/image_items.py` | Remove or deprecate `FixedImageItem` |
| `core/plot_main.py` | Remove `_ensure_guiqwt()`, simplify init |
| `pyproject.toml` | Keep pyqtgraph, keep matplotlib only for report |
| `environment.yml` | Remove guiqwt |

## Files to create

| File | Content |
|------|---------|
| `plotting/pg_image_widget.py` | `PGImageWidget` — pyqtgraph-based 2D image display with colormap, levels, crosshair |

## Non-goals

- **Removing matplotlib from the report tool** — report generation is a
  deliberate PDF/PNG export feature, not a GUI dependency.
- **Changing the UMAP visualization** — already uses pyqtgraph; no change needed.
- **Changing the data processing pipeline** — histogram computation,
  clustering, filtering stay exactly as-is. Only the *display* of results
  changes.
- **API breakage for plugins** — the napari plugin and any external code
  importing from ndXplorer should see no API change (internal plotting only).

## Definition of Done

- [ ] Marginal 1D histograms render correctly in `PlotWidget` dock widgets
- [ ] 2D histogram image renders with correct colormap, levels, aspect ratio
- [ ] Curve overlays render and update correctly
- [ ] `NDXPLORER_2D_BACKEND` env var removed — single pyqtgraph backend
- [ ] `guiqwt` is no longer imported anywhere in the codebase
- [ ] `matplotlib` is no longer imported during normal GUI operation (only in `report_tool.py`)
- [ ] All existing colormaps still work (same visual output)
- [ ] Startup time is noticeably faster (no guiqwt/qwt import)
- [ ] Tests pass

## Definition of Clean

- Zero `guiqwt` imports anywhere — no try/except, no lazy fallback
- `matplotlib` imported only inside `report_tool.py`
- The pyqtgraph image widget is self-contained in one file (< 200 lines)
- Marginal histograms are embedded dock widgets with linked axes (not standalone windows)
- No behavioural change visible to the end user — same functionality, faster startup
