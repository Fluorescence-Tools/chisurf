---
type: PRD
prd: "60"
title: "PRD-60: Amortised (Surrogate) Neural Estimator for H2MM"
description: An optional simulation-based-inference fast path that predicts H2MM parameters in one neural-network forward pass instead of iterating Baum-Welch EM, for large single-molecule FRET datasets.
status: draft
phase: "unassigned"
resource: chisurf/plugins/burst/burst_h2mm/core/surrogate.py
tags: [prd, burst, h2mm, machine-learning, performance]
timestamp: '2026-07-09T00:00:00Z'
---

# Summary

The `burst_h2mm` numba engine already fits H2MM ~4–16× faster than the reference
C implementation (deferred-ρ E-step, two-pass forward-backward, SQUAREM
acceleration, parallel Viterbi). This PRD proposes an **optional** further path
for the large-dataset regime: an **amortised neural estimator** that predicts the
H2MM parameters `(prior, trans, obs)` in a **single forward pass** of a small
network, trained once on data drawn from the H2MM generative model — i.e.
**simulation-based inference (SBI)**. It is a *replacement* for EM on a given
dataset, not an accelerator of it, and returns an **approximate** MLE. It ships
disabled and un-pretrained; EM remains the exact default.

# Motivation — why the remaining speed-up must be algorithmic

Fitting cost is `maps × N_photons × n²`. `N` (all bursts) and `n²` are already
minimal, so the only lever left for a large speed-up on the full dataset is the
**map count**. Two measured facts frame the design:

* **Initialisation cannot cut the map count.** Starting Baum-Welch from the
  *exact generative parameters* still took ~39 maps versus ~42 from a random
  start (3-state, 8000 bursts); only starting from *this dataset's* converged
  MLE reached the ~6-map floor. EM spends its effort on the finite-sample "last
  mile" to the dataset-specific MLE. **Consequence: a neural network used to
  *initialise* EM does not help** — even an oracle initialiser saves ~7%.
* **The data over-determine the model.** Fitting on 5 % of bursts recovered the
  same per-state FRET to ΔE ≤ 0.005 (10× faster); 2 % → ΔE ≈ 0.01 (19×). The
  parameters are pinned long before all photons are used.

Together these say: the fast, correct-enough estimate exists (over-determined
data), but EM must iterate to *reproduce* it. A network that learns the
data→parameter map **outputs that estimate directly, skipping the iteration**.

# Proof of concept

A scikit-learn `MLPRegressor` trained on 2 500 simulated 2-state datasets
(permutation-invariant summary features → parameters) recovered held-out
per-state FRET with **mean abs error 0.028–0.032**, versus **0.047–0.052** for a
single EM fit against the same ground truth — i.e. the surrogate was *more*
accurate than a single-restart MLE (it behaves like a Bayes estimator under the
training prior) and ~5× faster on small data, with the gap widening on large
data (inference cost is ~constant in `N`; EM scales with `maps × N`).

# Design

Implemented (draft) in `core/surrogate.py`, Qt-free, `numpy` + `scikit-learn`
only (already a ChiSurf dependency); `surrogate_available()` gates on the
backend so callers fall back to EM when it is absent.

* **Features** — `extract_features(data)`: fixed-length, permutation-invariant
  over bursts — windowed local-FRET histogram + quantiles (emission structure),
  photon-lag autocorrelation of the FRET signal (kinetics), inter-photon-Δt
  stats. Versioned (`FEATURES_VERSION`) so stale cached models are rejected.
* **Encoding** — targets are the canonically-ordered (states sorted by
  stream-0 emission) parameters; tiny off-diagonal transition probabilities are
  regressed in `log10` space. `_decode` reconstructs a valid row-stochastic
  model (clip + renormalise + diagonal repair).
* **Training** — `train_surrogate(...)` draws random models over a realistic
  FRET/kinetics prior, simulates each with a fast O(photons) sampler
  (`_fast_simulate`, states drawn at photon times via cached `A^Δt` — identical
  in distribution to the tick-by-tick simulator for the *observed* photons),
  extracts features, and fits a standardised MLP. `SurrogateModel.save/load`
  pickles the net + scalers + regime metadata.
* **Estimation** — `estimate_model(data, n_states, surrogate, refine_iters=0)`:
  one forward pass; `refine_iters > 0` optionally polishes with a few Baum-Welch
  maps toward the exact MLE (trading the speed-up back for exactness). Wired as
  the optional `surrogate=`/`refine_iters=` arguments of
  `h2mm.fit_states` (default `None` → EM unchanged).

# Scope, status, and open items

* **Status: draft / experimental, opt-in, no pretrained model shipped.** A
  surrogate is specific to a `(n_states, n_streams)` and the burst-length /
  Δt regime it was trained on; users train and cache their own.
* **Accuracy is approximate** — appropriate for exploration and routine fits on
  over-determined data; use EM (or `refine_iters`) for final/publication numbers
  and **always** do model selection (BIC/ICL) with EM on full data, since those
  depend on the total log-likelihood.
* **Open items**: model-registry/caching convention and a training CLI; a
  richer set-based architecture (DeepSets/transformer over bursts) if MLP
  accuracy plateaus; posterior/uncertainty output (proper SBI, e.g. neural
  posterior estimation) rather than a point estimate; generalisation across
  Δt/burst-length regimes; GUI exposure in the H2MM tool. Model *selection* by a
  classifier head (predict `n_states`) is a natural extension.

Related: the `burst_h2mm` engine and its optimisations are documented in
[burst plugins](/plugins/burst.md).
