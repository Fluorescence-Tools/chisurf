# MaxEnt decay Python API quickstart

This document summarizes the Python-facing API and example notebooks for the
MaxEnt TCSPC lifetime / FRET plugin.

The implementation lives under the plugin package
`chisurf.plugins.fluorescence_decay.maxent_decay` and exposes a small, stable
namespace via the `fmem` subpackage.

## Modules and layout

- `chisurf.plugins.fluorescence_decay.maxent_decay.fmem.core`
  Numerical implementation of the low-level solvers.

- `chisurf.plugins.fluorescence_decay.maxent_decay.fmem.api`
  Convenience helpers for calling the solvers from Python scripts or
  notebooks (grid builders and `run_*_mem_*` functions).

- `chisurf.plugins.fluorescence_decay.maxent_decay.notebooks/`
  Example Jupyter notebooks that exercise the Python API:
  - `maxent_lifetime_example.ipynb`
  - `maxent_fret_example.ipynb`

## Typical imports

Lifetime mode:

```python
from chisurf.plugins.fluorescence_decay.maxent_decay.fmem import (
    build_tau_grid,
    run_lifetime_mem_from_arrays,
)

# or explicitly from the API module
from chisurf.plugins.fluorescence_decay.maxent_decay.fmem.api import (
    build_tau_grid,
    run_lifetime_mem_from_files,
)
```

FRET distance mode:

```python
from chisurf.plugins.fluorescence_decay.maxent_decay.fmem import (
    build_distance_grid,
    run_fret_mem_from_arrays,
)

R0 = 52.0
R = build_distance_grid(R0=R0, r_min_frac=0.1, r_max_frac=3.0, r_bins=96)
```

## Notes on units

- **`dt`**
  Time step per detector channel (typically in ns).
- **`timeshift`**
  Expressed in **detector channels (samples)**, not in time units. Fractional
  values are allowed.
- **`fitrange`**
  A tuple `(start, stop)` of detector channel indices.
- **FRET distance grid `R`**
  Distances are in Å, but the grid is typically defined via
  `r_min_frac`/`r_max_frac` as **fractions of `R0`** so that the same settings
  remain meaningful when `R0` changes.

The result dictionaries returned by these helpers follow the structure of
`solve_lifetime_mem` / `solve_fret_mem` and contain, among others:

- `p` – recovered distribution (over `tau` or `R`).
- `tau` or `R` – lifetime or distance grid.
- `Fi`, `y`, `sigma`, `fitrange` – basis matrix, data segment, weights, and
  fit range used internally.

Refer to the example notebooks in `notebooks/` for end-to-end usage,
including synthetic data generation and plotting of decay, fit and
recovered distributions.
