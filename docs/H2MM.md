# Photon-by-photon HMM (H2MM)

The `burst_h2mm` plugin fits a **Hidden Markov Model directly to photon arrival
times and colours** within single-molecule FRET bursts, resolving sub-burst
FRET-state dynamics on the microsecond scale — down to time-scales far below the
burst duration that a burst-averaged FRET histogram cannot see. It implements the
H2MM algorithm of Pirchi *et al.* (J. Phys. Chem. B **120**, 13065, 2016) as a
self-contained, Qt-free `numpy`/`numba` engine, cross-validated against the
reference `H2MM_C` library.

## Theory

### The model

An H2MM model $\lambda = \{\pi, A, B\}$ over `n_states` hidden states and
`n_streams` photon streams (detector categories, e.g. donor/acceptor) is:

* $\pi$ — `(n_states,)` initial-state probabilities;
* $A$ — `(n_states, n_states)` **row-stochastic** transition matrix for **one
  base time unit** ($A_{ij} = P(\text{state } j \text{ at } t{+}1 \mid \text{state } i \text{ at } t)$);
* $B$ — `(n_states, n_streams)` **row-stochastic** emission matrix
  ($B_{ik} = P(\text{stream } k \mid \text{state } i)$).

Each hidden state has a characteristic **apparent FRET efficiency**
$E = B_{\text{acceptor}} / (B_{\text{donor}} + B_{\text{acceptor}})$.

### Variable inter-photon times — the key trick

Photons arrive at irregular integer macro-times $t_1 < t_2 < \dots$. Between two
consecutive photons the hidden chain performs $\Delta t$ *unobserved*
transitions, so wherever a standard HMM uses $A$, H2MM uses $A^{\Delta t}$. The
engine computes each power (and the expected transition-count tensor
$\rho(\Delta t)$) **once per unique $\Delta t$** by binary exponentiation, so cost
scales with the number of *photons*, not the number of clock ticks — this is what
makes photon-by-photon HMM tractable.

### Fitting and model selection

Parameters are estimated by **Baum-Welch** (scaled forward–backward EM). The
number of states is chosen by scanning `min_states..max_states` and minimising

* **BIC** $= -2\ln L + k\ln N_\text{photons}$, or
* **ICL** $= -2\ln L_\text{path} + k\ln N_\text{photons}$ (uses the Viterbi path),

where $k$ is the number of free parameters. The most-likely per-photon state
path is recovered by **Viterbi** decoding, giving per-state dwell-time
distributions and a transition-density plot.

## Compute engines

The same fit is available through several **engines** (trade exactness for
speed); model selection always scores fitted models by BIC/ICL — only *how* each
model is fitted changes.

| engine | description | accuracy |
| --- | --- | --- |
| `em` (default) | Exact Baum-Welch with SQUAREM acceleration | exact MLE |
| `em-float32` | The same EM in `float32` | ≈ (small round-off) |
| `surrogate` | Amortised neural estimator — one forward pass, no iteration | approximate |
| `surrogate-refine` | Surrogate seed + a few EM polish maps | ≈ exact |

The exact engine runs several-fold faster than `H2MM_C` and reaches the identical
maximum-likelihood estimate. The **surrogate** engine (see below) is an
*optional*, opt-in simulation-based-inference fast path: it trades a little
statistical precision for a large speed-up and is appropriate for exploration on
over-determined data (many bursts) — use `em` for final numbers and model
selection. An opt-in scan `patience` early-stops the state-count scan once the
criterion clearly turns upward (safe, ~1.6× faster).

### The neural surrogate

Because the H2MM generative model is a fast, exact simulator, a small network can
be trained once to map a dataset's summary features directly to
$(\pi, A, B)$ — replacing the iterative EM with a single forward pass. It ships
disabled with no pretrained model; train one for your `(n_states, n_streams)` and
Δt regime and pass it via `--surrogate`. See the design note *PRD-60* for the
rationale (notably: seeding EM near the optimum does **not** speed it up, because
EM's cost is the finite-sample last mile to the dataset-specific MLE — so the
network must *replace* EM, not initialise it).

## Inputs and outputs

**Input.** A folder of Seidel-style `.bur` burst files whose rows index into raw
`tttrlib.TTTR` photon data. Each photon is mapped to a *stream* from its routing
channel (and optional micro-time window).

**Outputs** (written to `<folder>/h2mm/`):

* `h2mm_result.json` — the model, per-state FRET, populations, transition rates,
  dwell means, and the BIC/ICL scan;
* `h2mm_photons.h5` — a **per-photon table** with the Viterbi `State`, macro/micro
  time, channel and stream;
* `h2mm_bursts.csv` — a **per-burst summary** (photon count, dominant state,
  transition count, mean FRET).

The photon and burst tables are plain numeric tables (time column
`Mean Macro Time (s)`) that open directly in **ndxplorer (ndX)** through its
CSV / MFD-HDF5 readers — colour the per-photon scatter by `State` to visualise the
recovered state trajectory.

## Usage

### GUI

Launch the H2MM tool, select a `.bur` folder, assign donor/acceptor detectors,
choose the state range, criterion, and engine, then **Run**. Results appear as a
FRET-state plot, transition-density plot, model-selection curve, and dwell-time
distributions.

### CLI

```bash
h2mm compute /path/to/analysis \
    --file-type SPC-130 \
    --donor-channels 0,8 --acceptor-channels 1,9 \
    --min-states 1 --max-states 4 --criterion bic \
    --engine em --patience 1
```

### Python API

```python
from chisurf.plugins.burst.burst_h2mm.core import h2mm, analysis

data = h2mm.prepare_bursts(times, streams, n_streams=2)      # engine layout
ana = analysis.analyze(data, state_counts=(1, 2, 3, 4),
                       criterion="bic", engine="em", patience=1)
print(ana.best.n_states, ana.fret)
path, icl = h2mm.viterbi(ana.best.model, data)               # per-photon states
```

The plugin ships its own examples under
`chisurf/plugins/burst/burst_h2mm/examples/`:

* `H2MM_01_Simulated_smFRET.ipynb` — an end-to-end tutorial that simulates smFRET
  photons with `tttrlib`, fits H2MM, compares the engines, and exports the ndX
  tables;
* `generate_example_data.py` — writes a **real, loadable** example dataset (a
  Photon-HDF5 `tttrlib` file + a `.bur` burst table) so the GUI and CLI can be
  run on genuine files:

  ```bash
  python -m chisurf.plugins.burst.burst_h2mm.examples.generate_example_data --out ./h2mm_example
  h2mm compute ./h2mm_example --file-type auto
  ```
