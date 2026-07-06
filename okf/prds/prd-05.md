---
type: PRD
prd: "05"
title: "PRD-05: Calibration Provenance"
description: Track calibration parameters in MFDB with links to the reference measurements they derive from
status: in-progress
phase: "4"
resource: chisurf/core/mfdb
tags: [prd, mfdb, fret]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Calibration parameters — g-factor, gamma, crosstalk, direct excitation, donor
lifetime, Förster radius — are tracked in MFDB with links to the reference
measurements they were derived from, so that when a calibration value changes,
every downstream fit that used it can be identified. Today these live as free
fitting-parameter objects with no record of where their values came from. The
PRD adds a calibration data model and provenance edges, and allows recording
values that are predicted from a setup's structured optical path with a
distinct "computed-from-optics" provenance source, so predicted and measured
crosstalk/R₀ can be compared.

# Status
In progress. Depends on the result registry for storing calibration records and
wiring provenance edges; some values can be sourced from an optical-configuration
setup rather than a measurement.

# Goal
Calibration parameters (g-factor, gamma, crosstalk, direct excitation, Förster
radius) are tracked in MFDB with links to the reference measurements they came
from. When a calibration value changes, all downstream fits that used it can be
identified.

# Background
Relevant code:
- `chisurf/core/models/tcspc/lifetime.py` — `background_curve`, `scatter`, `t_bg`, `t_exp`
- `chisurf/core/models/tcspc/anisotropy.py` — `g_factor`, `l1`, `l2`
- `chisurf/core/models/tcspc/fret.py` — `R0`, `tauD0`, `kappa2`
- `chisurf/core/models/pda/nusiance.py` — `crosstalk`, `gamma`, `direct_excitation`
- `chisurf/core/fluorescence/fret/__init__.py` — intensity-based FRET corrections
- `chisurf/plugins/jordi_g_factor/` — g-factor calculator plugin
- `chisurf/core/mfdb/result_registry.py` — the result registry (PRD-03)

# What are calibration parameters
In smFRET, before real distances can be recovered, these correction parameters
are needed:

| Parameter | What | How measured |
|-----------|------|-------------|
| g-factor | Detector polarization sensitivity ratio | Measure fast-rotating dye (e.g. Rhodamine 110), tail-match VV/VH |
| gamma | Detection efficiency ratio (donor/acceptor channels) | Donor-only sample + DA sample comparison |
| crosstalk (alpha) | Donor leakage into acceptor channel | Donor-only measurement, count ratio |
| Direct excitation (delta) | Acceptor excited by donor laser | Acceptor-only measurement |
| tau_D0 | Donor-only lifetime | Fit donor-only decay |
| R0 | Förster radius | Computed from spectra (see [PRD-06](prd-06.md)) |

Currently these are free `FittingParameter` objects; nobody tracks where their
values came from.

**Computed from the optical configuration ([PRD-08](prd-08.md)).** `crosstalk` and
`R0` can also be *predicted* from a setup's structured optical path
(filters/dichroics/detectors + dye spectra) by the Light Path Simulator plugin
(`chisurf/plugins/core/lightpath_simulator/backend/crosstalk.py`), which already
computes spectral crosstalk and R₀ overlap integrals. Record such values with a
"computed-from-optics" provenance source (vs. measured), so a setup's predicted
vs. measured crosstalk/R₀ can be compared. This makes the optical config a
first-class calibration source.

# Design and tasks

## Task 1: Calibration data model
`chisurf/core/mfdb/models.py` — a `CalibrationRecord` dataclass for calibration
records. Calibration values come from two sources: derived from a reference
measurement (`method="tail_matching"`, `"intensity_ratio"`, … →
`source_artifact_id` points to the reference measurement artifact); or entered
manually without backing data (`method="user_provided"` → empty
`source_artifact_id`/`source_file`). "God given" values (e.g. R0 from literature)
are stored with `method="user_provided"` and optionally a note citing the source.
Fields: `calibration_type` (`g_factor`/`gamma`/`crosstalk`/`direct_excitation`/
`donor_lifetime`/`forster_radius`), `value`, `error`, `source_file`,
`source_artifact_id`, `sample_id`, `method`, `notes`.

## Task 2: Register calibrations when computed
- **g-factor** (`chisurf/plugins/jordi_g_factor/`): after the g-factor is
  computed and displayed, call `register_calibration(...)` with
  `calibration_type="g_factor"` and `parent_artifact_id=<fast-rotating dye
  measurement>`. Wrap in try/except so the plugin still works without MFDB.
- **Donor lifetime** (`chisurf/plugins/fluorescence_decay/lltf/`): after a simple
  lifetime fit on a dataset tagged donor-only (sample has a donor probe but no
  acceptor probe), register `tau_D0` with `calibration_type="donor_lifetime"`.
- **Crosstalk / direct excitation**: typically entered manually or from intensity
  ratios. Provide a "Save to MFDB" button in the PDA nuisance parameter widget
  (`chisurf/gui/widgets/models/`).

## Task 2b: Register "god given" calibration values
Not all calibration parameters come from reference measurements — users enter
literature values (e.g. R0 = 54 Å for Alexa488-Alexa647) or values from prior
experiments. `register_calibration()` handles this with `method="user_provided"`
and an empty `parent_artifact_id`. The function stores `calibration_type` in
metadata and accepts `method` (how the value was determined) and `notes`
(free-text, e.g. a literature citation for user-provided values). Any calibration
widget should offer a "Save to MFDB" button with a small text input for an
optional citation.

## Task 3: Link fits to their calibration sources
`chisurf/core/mfdb/project_archiver.py` — in `_archive_fits()`, after creating the
fit operation and recording parameters, check whether any parameter matches a
known calibration record (`g_factor`, `gamma`, `crosstalk`, `direct_excitation`,
`delta`, `alpha`, `R0`, `forster_radius`, `tauD0`, `tau_D0`). If found, create a
`calibrated_by` edge from the fit operation to the calibration artifact.

## Task 4: Link background curves to MFDB
In `_archive_fits()`, if the fit state includes a background curve file, register
it as a raw-measurement artifact (`purpose="background_reference"`) and add a
`calibrated_by` edge. Prerequisite: `chisurf/core/project/fit_state.py`
`_model_to_state()` must serialize the background curve file path.

## Task 5: Staleness detection
`chisurf/core/mfdb/staleness.py` — find fits that use outdated calibrations. A use
is "stale" if it has a `calibrated_by` edge to a calibration artifact and a newer
calibration artifact of the same `calibration_type` exists. Returns the fit
operation, calibration type, used vs. latest artifact ids, and used vs. latest
values.

## Task 6: Tests
`test/fio/test_calibration_provenance.py` — register-calibration creates a
`calibration_data` artifact; a derived calibration gets a `derived_from` edge to
its reference measurement; user-provided calibrations store without a parent.

# Implementation status — headless core landed; GUI/archiver wiring deferred
The headless calibration-provenance core is in place and tested; it completes the
calibration-change impact loop PRD-21 Task 4 left open (`Lineage.impact_of` /
`what_used` already forward-listed `calibrated_by`).

- `calibrated_by` is now in the `relationship_type` vocabulary
  (`mfdb_flr_ext.dic`); `add_edge(..., relationship_type="calibrated_by")`
  validates. The column's CHECK is a *negative* constraint, so no DB migration is
  needed.
- `register_calibration` (in `result_registry.py`) already supports `method`/
  `notes` and the `user_provided` (no-parent) path; the `calibration_data` kind
  and `calibration` operation type already exist.
- `chisurf/core/mfdb/staleness.py`: `record_calibration_use` (the
  consumer→calibration `calibrated_by` edge) and `find_stale_calibration_uses` (a
  use is stale when a newer calibration of the same `calibration_type` exists;
  recency by `rowid`).
- Reworked from the PRD's pre-refactor sketch (`db.con`→`db.conn`,
  `source_id`→`source_node_id`, `parameter_name`→`name`,
  `metadata`→`metadata_json`).
- Tests: `test/fio/test_calibration_provenance.py` (6).

**`CalibrationRecord` dataclass — intentionally not added.**
`register_calibration`'s signature + the artifact `metadata_json` already capture
every field; a standalone, unconsumed dataclass would be dead code against the
"human maintainable" bar. Revisit only if a typed read-back path needs it.

# Definition of Done
- [x] `register_calibration()` supports `method` and `notes` parameters for manual entry
- [x] User-provided calibrations (R0 from literature, etc.) can be stored without a parent
- [x] `staleness.py` can find fits using outdated calibrations (`find_stale_calibration_uses`)
- [x] Headless usage-link API (`record_calibration_use`) creates `calibrated_by` edges
- [x] All headless tests pass (including user_provided calibration + the PRD-21 loop)
- [x] mfdb-admin **Calibrations view** (`gui/calibrations_view.py`) lists calibrations,
      registers literature/user-provided values (the "god-given" Task 2b path), and
      surfaces **stale uses** (`find_stale_calibration_uses`) — PRD-05's goal made visible.
      Over `mfdb.calibrations.*` handlers + `MFDBClient` methods + `staleness.list_calibrations`.
      Headless-tested (`test_calibration_handlers.py` 4, `test_calibrations_view.py` 4) and
      screenshot-verified offscreen.
- [ ] g-factor plugin / PDA-nuisance per-widget "Save to MFDB" buttons _(GUI; the admin
      view now covers manual registration, so these are convenience-only)_
- [ ] Fit archiver auto-links via `calibrated_by` _(deferred — the spec's value-matching
      heuristic is fragile; the explicit `record_calibration_use` API is the maintainable
      path, and the admin view surfaces the result)_
- [ ] `CalibrationRecord` dataclass — deferred (see note; avoids dead code)

# Relationships
- Depends on the [PRD-03](prd-03.md) result registry; consumes samples/FRET pairs from [PRD-02](prd-02.md).
- Predicted values relate to an optical-configuration setup extended from [PRD-04](prd-04.md).
- Records calibration provenance in the [MFDB (current)](/architecture/mfdb.md) store toward the [MFDB target](/specs/mfdb.md).
