# Burst H2MM Analysis Plugin

Photon-by-photon Hidden Markov Model (**H2MM**) analysis of single-molecule
FRET burst data. H2MM fits a hidden Markov model directly to photon arrival
times and colours *within* bursts, resolving sub-burst conformational dynamics
on the microsecond timescale that burst-averaged methods (FRET histograms, BVA)
cannot see.

The numerical core is a self-contained **Numba** re-implementation of the
algorithm of Pirchi *et al.* (J. Phys. Chem. B 2016, 120, 13065) and the
reference `H2MM_C` library by P. D. Harris — no external H2MM dependency is
required.

## What it does

- Fits `π` (initial state), `A` (one-step transition matrix) and `B` (emission
  matrix) over a range of state counts.
- Selects the number of states by **BIC** or **ICL** (Viterbi complete-data).
- Reports per-state apparent FRET, Viterbi state populations, transition rates
  (1/s), a **transition-density plot** (E before vs E after), and per-state
  **dwell-time distributions**.

## Architecture (client–server standard)

```
core/        Qt-free numeric core
  h2mm.py        Numba engine: A^Δt + ρ caches, scaled forward-backward,
                 Baum-Welch EM, Viterbi, BIC, model factory, simulator
  photons.py     .bur + tttrlib → per-burst (macro_time, stream) arrays
  analysis.py    state scan, BIC/ICL selection, dwell/transition diagnostics
api/         models.py (H2mmSettings/H2mmResult), contract.py, serialization.py
backend/     services.py — register_services + RPC handlers + shared runner
cli/         main.py — `chisurf h2mm compute`
gui/         client.py (H2mmClient), tool.py (H2mmTool: toolbar/tabs/plots)
```

The GUI never touches the database or heavy compute directly; it calls the
backend through `H2mmClient` (in-process `ServiceDispatcher` by default, or a
remote ZMQ server), exactly like the other burst plugins.

## The `A^Δt` trick

Between two photons separated by `Δt` macro-time ticks the hidden state makes
`Δt` unobserved transitions, so the engine substitutes `A**Δt` wherever a
standard HMM uses `A`. Powers and the transition-count tensor `ρ(Δt)` are
computed once per **unique** `Δt` (associative binary exponentiation with
row renormalisation) and cached, so cost scales with the number of *photons*,
not clock ticks.

## Photon streams

A *stream* is a detector category (e.g. donor "green", acceptor "red"), defined
by routing channels plus optional micro-time windows (PIE/ALEX). The first two
streams are treated as donor/acceptor for apparent-FRET reporting. Photons
matching no stream are dropped.

## Usage

**In the pipeline** — the plugin appears as step *6. H2MM* in the integrated
**Burst Analysis** tool; it inherits the burst folder and channel definitions
from earlier steps.

**Standalone GUI** — launch `H2mmTool`, pick a folder of `.bur` files, set the
donor/acceptor detectors and the state range, then **Run**.

**CLI**

```bash
chisurf h2mm compute /path/to/burst_folder \
    --file-type SPC-130 --donor-channels 0,8 --acceptor-channels 1,9 \
    --min-states 1 --max-states 3 --criterion bic
```

**Python / RPC**

```python
from chisurf.plugins.burst.burst_h2mm.gui.client import H2mmClient
client = H2mmClient()
result = client.compute(analysis_folder="…", settings={"max_states": 4})
```

## Status

Marked **experimental**: the Numba engine is validated against simulated data
(ground-truth recovery, monotonic log-likelihood, BIC model selection — see
`tests/`) but has not yet been cross-checked against `H2MM_C` on experimental
measurements.
