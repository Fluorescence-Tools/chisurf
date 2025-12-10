# MaxEnt decay Python API quickstart

This document summarizes the Python-facing API and example notebooks for the
MaxEnt TCSPC lifetime / FRET plugin.

The implementation lives under the plugin package
`chisurf.plugins.fluorescence_decay.maxent_decay` and exposes a small, stable
namespace via the `fmem` subpackage.

## Modules and layout

- `chisurf.plugins.fluorescence_decay.maxent_decay.fmem.core`
  Thin wrapper that re-exports the low-level solvers from `core.py`.

- `chisurf.plugins.fluorescence_decay.maxent_decay.fmem.api`
  Convenience helpers for calling the solvers from Python scripts or
  notebooks (grid builders and `run_*_mem_*` functions).

- `chisurf.plugins.fluorescence_decay.maxent_decay.fmem.cli`
  Click-based command-line interface used by the `csc maxent-decay` subcommand.

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
```

The result dictionaries returned by these helpers follow the structure of
`mem_vin4_lifetime` / `mem_vin4_fret` and contain, among others:

- `p` – recovered distribution (over `tau` or `R`).
- `tau` or `R` – lifetime or distance grid.
- `Fi`, `y`, `sigma`, `fitrange` – basis matrix, data segment, weights, and
  fit range used internally.

Refer to the example notebooks in `notebooks/` for end-to-end usage,
including synthetic data generation and plotting of decay, fit and
recovered distributions.
