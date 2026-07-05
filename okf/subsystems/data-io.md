---
type: Subsystem
title: Data IO
description: File loading and the format registry — TTTR photon data via tttrlib, ASCII/FCS curves, structures, and slow-storage staging.
resource: chisurf/core/fio/
tags: [io, core]
timestamp: '2026-07-05T00:00:00Z'
---

# Scope

`chisurf/core/fio/` is the Qt-free IO layer: it turns files on disk (or a slow
network share) into the domain objects consumed by [Core](/subsystems/core.md)
and the fitting models.

| Module | Handles |
| --- | --- |
| `fluorescence/tttr.py`, `photons.py`, `burst.py` | single-photon (TTTR) streams |
| `fluorescence/fcs/*` | correlation curves (Kristine, ALV, ConfoCor3, PyCorrFit, ISS, …) |
| `fluorescence/tcspc.py`, `sdtfile.py`, `bhfiles.py`, `thdfile.py` | TCSPC decays / B&H |
| `ascii.py`, `jordi.py`, `zipped.py` | generic text / Jordi curves / gz-bz2 wrappers |
| `structure/coordinates.py`, `density.py` | PDB / density structures |
| `mmcif/` | mmCIF importer + database resolver (feeds [MFDB](/architecture/mfdb.md)) |
| `staging.py` | slow/network-file staging (below) |

# Format registry (`chisurf/core/file_formats.py` + `file_formats.json`)

`FILE_FORMATS` maps an extension to `{name, description, tttrlib_container,
reading_routine, experiments}`. Extension → experiment routing:

| Ext | Format | tttrlib container | Experiments |
| --- | --- | --- | --- |
| `.ptu` / `.ht3` | PicoQuant PTU/HT3 | PTU / HT3 | pda, tcspc, fcs, pch, rics |
| `.spc` | Becker & Hickl SPC | SPC-130 | pda, tcspc, fcs |
| `.h5` / `.hdf5` | Photon-HDF5 | PHOTON-HDF5 | pda, tcspc, fcs, pch, rics |
| `.pt3`, `.t3r`, `.sm`, `.raw` | PicoQuant / SM / CZ-RAW | (varies) | tcspc (+fcs/pda) |
| `.csv` / `.txt` / `.fcs` | text / ISS FCS | — | fcs, pda |

# Photon / TTTR data (tttrlib)

Time-tagged single-photon records are parsed by the compiled `tttrlib`
container (see [compiled modules](/subsystems/compiled-modules.md)); ~30 modules
across `core/fio`, `core/experiments`, and models depend on it. Per-domain
readers wrap it: `experiments/tcspc/tttr_reader.py`, `experiments/rics/tttr_loader.py`,
`experiments/{pch,pda,fcs}/reader.py`, `experiments/deer/reader.py`.

# Slow-storage staging (`chisurf/core/fio/staging.py`)

Because `tttrlib.TTTR(path)` is a single blocking C++ call with **no progress
hook**, multi-GB reads from a slow share would freeze the caller. `staging`:

1. Probes source throughput via a small head read.
2. If *slow* + large, stream-copies to a local temp with a chunked loop that
   emits `progress_cb` (bytes/%/MB·s/ETA), then parses the local copy.
3. If *fast*, returns the original path unchanged (no copy).

`staged_source(path)` is a context manager (temp is ephemeral, auto-deleted);
`open_tttr(path)` stages+parses+cleans in one call. It is **Qt-free** — reusable
from the [server](/architecture/server.md)/CLI; the GUI
(`chisurf/gui/widgets/staged_loading`) only adds a progress dialog. Tunables live
in the `data_loading` settings section; `StagingCancelled` reports user cancel.

See also: [overview](/overview.md), [macros & CLI](/subsystems/macros-cli.md),
[history](/subsystems/history.md).
