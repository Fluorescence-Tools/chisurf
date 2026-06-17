# Photon Counting Histogram (PCH) Analysis Plugin

Analyze the distribution of photon counts in fluorescence time traces to extract molecular brightness and occupancy.

## Architecture

```
pch/
├── manifest.json           # Plugin manifest (entrypoints, RPC methods, metadata)
├── __init__.py             # Bootstrap — loads manifest, exports PCHApp
├── api/
│   ├── models.py           # PchSettings, PchResult, FitResult dataclasses
│   └── algorithms.py       # Pure PCH math (numba-accelerated)
├── backend/
│   └── services.py         # RPC handlers: pch.load_tttr, pch.compute, pch.fit
├── gui/
│   ├── client.py           # PCHClient wrapping InProcessClient
│   └── tool.py             # PCHApp QMainWindow (toolbar, splitter layout)
├── cli/
│   └── main.py             # click CLI: pch analyze, pch refit
└── tests/
    ├── test_algorithms.py
    ├── test_manifest.py
    └── test_services.py
```

## Usage

1. From the ChiSurf menu: **Plugins > Spectroscopy > Single-Molecule > PCH**
2. Click **Load TTTR** to select a file
3. Adjust channels, bin time, micro-time range in the Data Settings panel
4. Click **Compute PCH** (toolbar) to build the histogram
5. Set number of species and initial guesses in Model Fit
6. Click **Fit Model** (toolbar) to fit
7. Drag the region selector on the histogram to recompute χ² for a sub-range
8. Click **Save Results** (toolbar) to export NPZ/CSV/PNG/TXT

### CLI

```bash
python -m chisurf pch analyze data.ptu --components 2 --bin-time 50
python -m chisurf pch refit results.npz --components 3 --json
```

## Dependencies

- ttrolib (TTTR file I/O)
- numpy, scipy, numba (computation)
- qtpy, pyqtgraph (GUI)
- click (CLI)

## Author

Thomas-Otavio Peulen — thomas.peulen@tu-dortmund.de
