---
type: Reference
title: "Modelling / ProteinMC roadmap notes"
description: Durable forward-looking modelling, ProteinMC and sampling feature ideas for the structural/FRET-modelling tools.
tags: [reference, modelling, roadmap]
timestamp: '2026-07-06T00:00:00Z'
---

# Reference

Durable, forward-looking feature ideas for ChiSurf's structural- and
FRET-modelling stack (ProteinMC, sampling, accessible-volume tooling). These
are design intentions, not committed work; treat them as a backlog that
complements the maintained [Modelling plugins](/plugins/modelling.md) group and
[PRD-58](/prds/prd-58.md) (FRET plugin as an FPS + OLGA superset).

# Roadmap

## ProteinMC → external biophysical framework

The long-term direction is to retire the custom ProteinMC representation and
energy machinery in favour of an established, general-purpose **external
biophysical modeling framework**.

- **Energy terms.** Reimplement the currently used/custom energy terms on top
  of the external framework rather than maintaining a bespoke scoring stack.
  This includes adding a missing **Go-model (structure-based) potential**.
- **Sampling backends.** Decouple sampling from any single engine — the current
  MCMC path should not be the only option. Add support for additional sampling
  methods, including **replica-exchange** sampling (parallel/"#Run" runs should
  map onto replica exchange rather than naive independent parallel runs, which
  are unreliable today).
- **Coarse-grained force field.** Add support for a **Martini-style**
  coarse-grained force field via the external framework's facilities.
- **Rigid-body docking.** Add restrained **rigid-body docking** to ProteinMC.
  This subsumes and is intended to retire the legacy FRET-positioning (FPS)
  code path.

## Sampling, ensembles and FRET-scored refinement

- **Chi-square coupling.** Allow the chi-square of other fits to be added as an
  energy term, so structural sampling can be driven by experimental fit
  quality.
- **Ensemble sampling.** Sample multiple structures in parallel and expose the
  resulting inter-dye **distance distributions** as first-class outputs.
- **Scoring against FRET posteriors.** Score structures/ensembles against
  **PDA-derived posterior distance-distribution densities** (e.g. ucFRET
  posteriors). This requires extending the `fps.json` format to carry the
  distribution information and its uncertainties, then scoring an ensemble
  against those densities.
- **Ensemble-refinement model.** Provide a dedicated ensemble-refinement model
  (distinct from ProteinMC) that loads an ensemble with prior probabilities and
  scores it against multiple experiments simultaneously.

## Accessible-volume / `fps.json` format support

- Support the **full `fps.json` schema** as used by the external FRET-restraint
  library — multiple scoring groups (user-selectable), definitions of mobile
  amino-acid segments, and the other richer fields. Bring the in-app `fps.json`
  editor up to full parity with that format.
- **Expose inter-fluorophore distances as parameters** so fits can tap into
  modelled distances and link them into sampling (parameter linking across the
  modelling ↔ fitting boundary).

## In-app model editing ("front face / back face")

Make it possible to develop and refine fits and models from *within* ChiSurf,
without a separate development environment.

- Give each fit/model a visible **"front face" and hidden "back face"**: a view
  that displays the actual code backing a model, with the option to edit the
  fit in place. This is expected to require a substantial architecture change
  (no backward-compatibility constraint), but every path must remain tested.
- Objective: iteratively develop and modify models and fits inside the running
  application.

## Parameter-transform adapter models

Introduce **parameter-transform models** — models whose parameters exist purely
to link (adapt) between other models.

- Build adapters for accessible-volume observables: `dRDAE`, `RDA`, `dRMP`,
  and FRET efficiency. Example: a user inputs a distance between mean dye
  positions and the adapter outputs the mean distance or the mean
  FRET-averaged distance.
- Purpose: compose more complex global models — e.g. combine **PDA (RDAE)** with
  **TCSPC (RDA)** so both experiment types share the same underlying structural
  parameters.

# Known bug

- **PDA parameter scan is broken.** Running a PDA parameter scan reproducibly
  fails; see the archived screenshot `Pasted Graphic.png` captured against the
  PDA modelling UI. Needs investigation and a regression test once fixed.
