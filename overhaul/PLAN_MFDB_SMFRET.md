# MFDB smFRET Archiving Plan

## Vision

MFDB becomes the authoritative archive for single-molecule FRET experiments -- from raw
photon streams through calibration, burst selection, decay construction, and model fitting
with full parameter dependency tracking. Every analysis step is a provenance node; every
parameter decision is traceable. The database is designed for eventual public sharing of
fluorescence experiments, but the immediate priority is making ChiSurf's own save/load
cycle work correctly and completely.

## Architecture Overview

```
chisurf.gui  <--ZMQ-->  chisurf.server  <-->  MFDB (SQLite + object store)
                              |
                           chinet (parameter dependency graph)
```

- **GUI**: Qt frontend, sends JSON-RPC commands via ZMQ
- **Server**: `ChiSurfServer` with `ServiceDispatcher`, `EventBus`, `JobManager`
- **MFDB**: Three-generation schema (v28) with content-addressed object store
- **chinet**: DAG-based parameter linking across fits, persisted via `MFDBChinetBackend`

---

## Current State Assessment

### What Works

| Component | Status | Notes |
|-----------|--------|-------|
| MFDB schema (v28) | Stable | 3 generations of tables, progressive migration |
| Object store | Working | MD5-deduplicated, UUID-referenced content blobs |
| `.csp` project save/load | Working | ZIP archive with project.json + chinet session |
| TTTR-to-histogram | Working | tttrlib-based, multi-channel |
| Burst selection plugin | Working | Multiple algorithms (sliding window, BOCPD, Kalman), PIE, GMM, BVA |
| TCSPC fitting models | Working | Multi-exp lifetime, FRET (Gaussian/discrete/WLC/structure), PDDEM, MaxEnt |
| Parameter linking (runtime) | Working | FittingParameter.link within/across fits |
| Auth/ACL system | Working | Session tokens, groups, Unix-style permissions |
| mfdb-admin GUI | Working | Browse, inspect, provenance graph viewer |
| Seed fluorophore data | Minimal | 7 probes, 3-point placeholder spectra |

### What's Broken or Missing

| Issue | Severity | Location |
|-------|----------|----------|
| **MFDB archive loses fit structure** -- id, name, model_name, plot_state, fit_range stripped | Critical | `project_archiver.py:290-295` |
| **MFDB restore produces wrong format** -- flat fit_state list instead of fit group records | Critical | `project_archiver.py:596-600` |
| **Chinet sessions restored but discarded** by restore handler | Critical | `project_browser/backend/services.py:387` |
| **Parameters/edges stored but never read back** on restore | Critical | `project_archiver.py:527-601` |
| **No FLR metadata linkage** -- sample, dyes, instrument, conditions not connected to project archive | High | `project_archiver.py` (entire) |
| **UI state, experiments, extra metadata not persisted** to MFDB | High | `project_archiver.py:162-178` |
| **Burst pipeline disconnected** from project save flow | High | `pipeline.py` (entire) |
| **Burst pipeline vocabulary mismatches** -- `artifact_type` vs `artifact_kind`, `"local"` vs `"local_file"`, `"success"` vs `"succeeded"` | Medium | `pipeline.py:56,58,151` |
| **Dataset role filtering may drop all datasets** on restore | Medium | `project_archiver.py:580` |
| **Multiple datasets may collapse** to single key on restore | Medium | `project_archiver.py:584` |
| **No MFDB round-trip test** exists | Medium | (missing) |
| **No burst-to-decay pipeline** -- can't build TCSPC histograms from burst subpopulations | High | (missing) |
| **No unified calibration workflow** -- each model handles its own correction parameters | Medium | scattered |

---

## Objectives

### Objective 1: Fix MFDB Project Round-Trip (Immediate)

Make `archive_project_to_mfdb()` -> `restore_project_from_artifacts()` produce a project
that is functionally identical to what `.csp` save/load produces.

### Objective 2: Connect smFRET Workflow to MFDB (Near-term)

Wire burst selection, decay computation, calibrations, and sample metadata into the
provenance graph so that every analysis step from raw TTTR to final FRET distances is
archived.

### Objective 3: Enrich Fluorophore/Sample Database (Medium-term)

Populate MFDB with real spectral data, expand probe catalog, and connect to external
fluorophore databases for Forster radius computation.

### Objective 4: Public Database Readiness (Long-term)

Schema stability, export formats, access control, and documentation for sharing MFDB
databases across labs.

---

## PRD 1: Fix MFDB Project Archive Round-Trip

### Problem

Projects archived to MFDB cannot be restored to a working state. The archiver strips
structural metadata and the restore path produces a payload format incompatible with the
project loader.

### Requirements

#### 1.1 Preserve Full Fit Group Structure

**Current**: `archive_project_to_mfdb()` stores only `fit_state_payload` (the inner
parameter snapshot). The fit group envelope -- `id`, `name`, `model_name`, `local_fits`
list structure, `plot_state`, `fit_range` -- is discarded.

**Required**: Store the complete fit record as returned by `make_fit_record()` in
`fit_state.py`. The fit_result artifact's `metadata_json` must include:
- `fit_id` (UID)
- `fit_name`
- `model_name` (reader/model class identifier)
- `plot_state` (axis ranges, log scale, visible curves)
- `fit_range` (data range used for chi-squared)
- `local_fits` count (for fit groups)

**Files to modify**:
- `chisurf/core/mfdb/project_archiver.py` -- `_archive_fits()` (around line 290)
- Store envelope metadata in the operation record or as a separate `fit_metadata` artifact

#### 1.2 Persist Project-Level Metadata

**Current**: Only `project_id`, `version_number`, `fit_count`, `dataset_count` stored.

**Required**: The project operation metadata must also include:
- `chisurf_version`
- `project_format_version`
- `description` (from project meta block)
- `created` timestamp
- `ui_state` (experiment index, setup index, dataset layout, fit index)
- `experiments` dict (experiment configurations)

**Files to modify**:
- `chisurf/core/mfdb/project_archiver.py` -- `archive_project_to_mfdb()` metadata dict (line 162)
- Consider storing `ui_state` and `experiments` as separate artifacts of kind `project_metadata` to avoid bloating the operation record

#### 1.3 Fix Dataset Restore

**Current**: Two bugs in dataset restoration:
1. `role` field may not be present in the joined query result, causing the `kind == "processed_data" and role == "dataset"` filter to silently drop all datasets
2. Fallback `ds_id = ds_id.get("ds_id", role)` collapses multiple datasets to key `"dataset"`

**Required**:
- Verify that `db.get_operation_artifacts()` returns the `role` field from `mfdb_operation_artifact`
- If not, join explicitly or query `mfdb_operation_artifact` separately
- Use a unique dataset identifier (original UID or artifact UUID) as the dict key, never `role`
- Store the dataset's original UID, filename, reader class, and reader config in `metadata_json`

**Files to modify**:
- `chisurf/core/mfdb/project_archiver.py` -- `restore_project_from_artifacts()` lines 575-590
- `chisurf/core/mfdb/repository.py` -- `get_operation_artifacts()` if role field is missing from result

#### 1.4 Restore Chinet Sessions and Parameter Dependencies

**Current**: `restore_project_from_artifacts()` returns `chinet_sessions` but the
`restore_project_handler()` in the project browser plugin discards them. Parameters and
`parameter_depends_on` edges in `mfdb_edge` are never read back.

**Required**:
- `restore_project_from_artifacts()` must query `mfdb_parameter` for each fit operation
  and include parameter records in the returned fit data
- `parameter_depends_on` edges must be queried and returned as a dependency map
- `restore_project_handler()` must pass chinet session data to the project loader
- The project loader must re-establish `FittingParameter.link` connections using the
  dependency map, handling the ordering problem (linked-to fit must exist before the
  linking fit)

**Files to modify**:
- `chisurf/core/mfdb/project_archiver.py` -- `restore_project_from_artifacts()` (line 527+)
- `chisurf/plugins/core/project_browser/backend/services.py` -- `restore_project_handler()`
- `chisurf/core/project/fit_state.py` -- `apply_state_to_fit()` must handle MFDB-sourced dependency info

#### 1.5 Round-Trip Test

**Required**: A test that:
1. Creates a project with 2+ datasets and 2+ fits (at least one with cross-fit parameter links)
2. Calls `archive_project_to_mfdb()`
3. Calls `restore_project_from_artifacts()`
4. Verifies: dataset count, dataset UIDs, dataset curve data (x/y arrays), fit count,
   fit UIDs, fit names, model names, parameter values, parameter bounds, parameter
   fixed states, parameter link targets, plot state, fit range
5. Verifies provenance edges exist: `project_contains`, `derived_from`, `parameter_depends_on`

**Files to create**:
- `test/fio/test_mfdb_project_roundtrip.py`

### Acceptance Criteria

- `archive_project_to_mfdb()` followed by `restore_project_from_artifacts()` produces a
  payload that, when loaded via the standard project loader, results in the same number
  of datasets and fits with identical parameter values (within floating-point tolerance)
- Cross-fit parameter links are re-established after restore
- The round-trip test passes in CI

---

## PRD 2: Connect Burst Pipeline to MFDB Provenance

### Problem

The burst selection plugin produces `.bur` files and summary DataFrames but these
artifacts are invisible to the project archiver. The `BurstPipeline` class in
`pipeline.py` is standalone and uses incorrect vocabulary constants.

### Requirements

#### 2.1 Fix BurstPipeline Vocabulary

**Current mismatches**:
- `artifact_type` -> should be `artifact_kind` (line 56)
- `storage_mode="local"` -> should be `"local_file"` (line 58)
- `status="success"` -> should be `"succeeded"` (line 151)

**Files to modify**: `chisurf/core/mfdb/pipeline.py` lines 56, 58, 151

#### 2.2 Integrate Burst Selection Results into Project Archive

**Required**: When a project is archived to MFDB:
1. Each burst selection run should be recorded as an `mfdb_operation` of type `burst_selection`
2. Input: `raw_measurement` artifact (the TTTR file)
3. Output: `burst_data` artifact (the `.bur` file or DataFrame, stored in object store)
4. Operation metadata: detection algorithm, parameters (min_photons, time_window,
   photon_window), channel definitions, PIE window definitions, filter settings
5. `derived_from` edge from burst_data to raw_measurement
6. `project_contains` edge from project operation to burst_selection operation

**Design decision**: The burst selection plugin already runs independently. The integration
point should be in `archive_project_to_mfdb()` -- it needs a way to discover which burst
selections exist for the project's datasets. Options:
- (A) The burst plugin registers its results in a runtime registry that the archiver queries
- (B) The archiver accepts burst results as an explicit parameter
- (C) The burst plugin archives directly to MFDB when it runs, and the project archiver
  links to existing artifacts

Recommend **(C)** -- the burst plugin should call `BurstPipeline` (or equivalent) to
archive its results to MFDB immediately when they are produced, and the project archiver
should discover and link existing burst artifacts by querying for artifacts derived from
the project's raw measurement artifacts.

**Files to modify**:
- `chisurf/plugins/burst/burst_selection/api/selection.py` -- add MFDB archiving after burst selection
- `chisurf/core/mfdb/pipeline.py` -- fix vocabulary, add `version_id` parameter for project linkage
- `chisurf/core/mfdb/project_archiver.py` -- discover and link burst artifacts

#### 2.3 Record Burst-to-Decay Derivation (Future)

When burst-to-decay computation is implemented, it should be recorded as:
- Operation type: `histogram_construction`
- Input: `burst_data` artifact + `raw_measurement` artifact
- Output: `processed_data` artifact (the decay histogram)
- Metadata: which burst subpopulation, channel selection, micro-time binning
- Edge: `derived_from` linking decay to burst selection

This is a placeholder requirement -- the burst-to-decay pipeline itself is out of scope
for this PRD.

### Acceptance Criteria

- Burst selection results appear as artifacts in MFDB with correct provenance
- `mfdb-admin` provenance graph shows the burst selection step between raw data and
  processed data
- No vocabulary mismatches in pipeline.py

---

## PRD 3: Link FLR Metadata to Project Archive

### Problem

When a project is archived to MFDB, no sample, dye, instrument, or experimental
condition metadata is attached. The FLR tables (`probe_types`, `probes`, `entities`,
`flr_sample`, `flr_experiment`, `flr_fret_forster_radius`, etc.) exist in the schema
but are never populated during the project archive workflow.

### Requirements

#### 3.1 Sample Association

**Required**: Each project archive must link to an `mfdb_sample` record. The sample
defines:
- Name and description
- Protein/DNA entity (sequence, mutations, labeling positions)
- Attached probes (donor/acceptor dye assignments)
- Conditions (buffer, pH, temperature, salt concentration)

**Workflow**: The user must specify the sample when creating or archiving a project.
The GUI should provide:
- A sample picker that queries existing `mfdb_sample` records
- A "New Sample" dialog that creates the sample record with entity, probes, conditions
- The selected sample_id is stored in the project operation metadata and linked via
  `mfdb_edge` with relationship type `measured_sample`

**Files to modify**:
- `chisurf/core/mfdb/project_archiver.py` -- accept `sample_id` parameter, create edge
- `chisurf/gui/widgets/` -- new sample picker widget (or extend mfdb-admin)
- `chisurf/core/mfdb/api.py` -- add `create_sample_with_probes()` convenience function

#### 3.2 Experiment Association

**Current**: `_ensure_experiment()` in `chinet_adapter.py` creates a placeholder
experiment record, but only from `store_chinet_session()`, not from the project archiver.

**Required**: Each project archive must link to an `mfdb_experiment` record:
- Experiment type (smFRET, ensemble TCSPC, FCS, etc.)
- Instrument/device reference
- Date/time
- Operator (user reference)
- Link to sample

The `mfdb_experiment` table already exists. The project archiver needs to accept
`experiment_id` and create a `measured_in` edge.

**Files to modify**:
- `chisurf/core/mfdb/project_archiver.py` -- accept `experiment_id`, create edge
- `chisurf/core/mfdb/api.py` -- add `create_experiment()` if not already present

#### 3.3 Dye/Probe Properties Linkage

**Required**: When a sample has attached probes (donor/acceptor), the archive must:
- Link to `probes` table entries with full optical properties
- If the donor-acceptor pair has a `flr_fret_forster_radius` entry, link it
- Store the Forster radius used in the fit as a parameter and link to the dye pair

**Workflow**: The Forster radius `R0` is currently a free `FittingParameter` in
`FRETParameters`. On archive, the system should:
1. Check if the fit's R0 value matches a known dye pair's Forster radius
2. If yes, create a `calibrated_by` edge to the `flr_fret_forster_radius` record
3. Store donor/acceptor probe IDs in the fit operation metadata

**Files to modify**:
- `chisurf/core/mfdb/project_archiver.py` -- probe linkage logic in `_archive_fits()`
- `chisurf/core/mfdb/repository.py` -- add `find_forster_radius_by_probes()` query

#### 3.4 Instrument/Setup Linkage

**Required**: Each raw measurement artifact should link to an `mfdb_setup` record:
- Instrument type (confocal, TIRF, wide-field)
- Detector configuration (channels, spectral windows)
- Excitation wavelengths and powers
- Time resolution (TAC range, micro-time bins)

The `mfdb_setup` table exists. The TTTR header JSON (already captured by `TCSPCTTTRReader`)
contains most of this information. On archive, extract instrument metadata from the TTTR
header and create/reuse a setup record.

**Files to modify**:
- `chisurf/core/mfdb/project_archiver.py` -- extract setup from dataset metadata
- `chisurf/core/experiments/tcspc/tttr_reader.py` -- ensure header JSON is in dataset metadata

#### 3.5 Calibration Parameter Provenance

**Required**: Correction parameters (gamma, crosstalk, direct excitation, g-factor) must
be recorded as `mfdb_parameter` entries with provenance links to how they were determined:

- **g-factor**: If determined from a reference measurement with fast-rotating dye, create
  an edge from the g-factor parameter to the reference measurement artifact
- **gamma factor**: If determined from donor-only/DA comparison, link to both measurements
- **crosstalk/direct excitation**: Link to calibration measurement if applicable

**Current state**: These parameters exist as `FittingParameter` objects but their origin
is not tracked. The `flr_fret_calibration_parameters` table exists but is never populated
from the fitting workflow.

**Files to modify**:
- `chisurf/core/mfdb/project_archiver.py` -- detect calibration parameters in fit state,
  store in `flr_fret_calibration_parameters` and `mfdb_parameter`
- `chisurf/core/models/tcspc/anisotropy.py` -- expose g-factor source info
- `chisurf/core/models/pda/nusiance.py` -- expose calibration source info

#### 3.6 Buffer/Reference Measurement Linkage

**Required**: When a fit uses a background curve (buffer measurement), the archive must:
1. Store the background curve file as a `raw_measurement` artifact (if not already stored)
2. Create a `calibrated_by` edge from the fit operation to the background artifact
3. Store `t_bg` and `t_exp` as parameters on the fit operation

**Current**: `Generic.background_curve` is a `Curve` object with a file path. The
fit_state captures the curve UID but not the file path or MFDB artifact reference.

**Files to modify**:
- `chisurf/core/mfdb/project_archiver.py` -- detect background curve in fit state, archive as artifact
- `chisurf/core/project/fit_state.py` -- include background curve file path in state

### Acceptance Criteria

- A project archived with sample/experiment/instrument metadata shows the full
  provenance chain in mfdb-admin: sample -> experiment -> raw data -> burst selection ->
  decay -> fit -> parameters
- Dye properties and Forster radius are linked to fit results
- Calibration parameters have provenance to their source measurements
- Buffer measurements appear as artifacts with `calibrated_by` edges

---

## PRD 4: Expand Fluorophore Database

### Problem

The seed database has only 7 probes with 3-point placeholder spectra. Real smFRET
analysis requires accurate spectral data for Forster radius computation (overlap integral).

### Requirements

#### 4.1 Import Real Spectral Data

**Required**: Populate the `spectra` table with real absorption and emission spectra for
common smFRET dyes:
- Alexa Fluor series (350, 405, 488, 532, 546, 555, 568, 594, 633, 647, 660, 680, 700, 750)
- ATTO series (488, 532, 550, 565, 590, 594, 610, 620, 633, 647N, 655, 680, 700)
- Cy series (Cy3, Cy3B, Cy5, Cy5.5, Cy7)
- Common intrinsic probes (Trp, NADH, FAD)

Data sources:
- FPbase (public fluorophore database)
- Manufacturer specification sheets
- Published literature values

Store as wavelength/intensity arrays (1 nm resolution, normalized) in the `spectra` table.

#### 4.2 Forster Radius Computation

**Required**: Add a function that computes R0 from:
- Donor emission spectrum
- Acceptor absorption spectrum
- Donor quantum yield
- Acceptor extinction coefficient
- Refractive index
- Orientation factor (kappa^2)

Using the standard overlap integral formula. Store computed R0 values in
`flr_fret_forster_radius` table.

**Files to create/modify**:
- `chisurf/core/fluorescence/fret/forster.py` -- overlap integral computation
- `chisurf/core/mfdb/seed_data.py` -- expanded seed data

#### 4.3 Probe Search and Selection UI

**Required**: A widget that lets users:
- Search probes by name, type, wavelength range
- View absorption/emission spectra
- Select donor/acceptor pairs
- See computed Forster radius for the pair
- Add custom probes with uploaded spectra

### Acceptance Criteria

- 50+ common fluorophores with real spectral data in the seed database
- Forster radius auto-computed for any donor-acceptor pair from spectral overlap
- User can find and select dyes when defining a sample

---

## PRD 5: smFRET Analysis Workflow Template

### Problem

There is no guided workflow for smFRET data processing. Users must manually configure
each step. The processing chain has implicit dependencies that are not enforced or tracked.

### Requirements

#### 5.1 Define the Canonical smFRET Workflow

```
1. Define Sample
   - Protein/DNA sequence
   - Labeling positions (cysteine mutations, etc.)
   - Donor/acceptor dye pair
   - Buffer conditions

2. Load Reference Measurements
   a. Donor-only sample -> determine donor lifetime (tau_D0)
   b. Buffer/blank -> background level
   c. Fast-rotating dye (e.g., Rhodamine 110) -> g-factor for anisotropy
   d. Known-concentration dye -> confocal volume calibration (FCS)

3. Determine Calibration Parameters
   a. gamma factor (from donor-only + DA samples, or from lifetime)
   b. Crosstalk alpha (from donor-only, acceptor channel leakage)
   c. Direct excitation delta (from acceptor-only)
   d. Detection efficiency ratio

4. Load DA Measurement (TTTR)
   a. Configure PIE windows (if applicable)
   b. Select channels (donor prompt, acceptor prompt, donor delay, acceptor delay)

5. Burst Selection
   a. Choose algorithm and parameters
   b. Apply photon filters
   c. Inspect burst statistics (size, duration, count rate)
   d. Apply population selection (GMM, manual gates)

6. Burstwise Analysis
   a. Compute proximity ratio / FRET efficiency histograms
   b. Apply corrections (gamma, crosstalk, direct excitation)
   c. Compute corrected FRET efficiencies
   d. BVA (Burst Variance Analysis) for dynamics detection

7. Sub-ensemble TCSPC
   a. Build decay histograms from burst subpopulations
   b. Fit donor decay in DA sample -> extract distance distributions
   c. Compare with donor-only decay
   d. Multi-component analysis (static vs dynamic FRET)

8. Model Fitting
   a. Choose distance model (Gaussian, WLC, discrete, structure-based)
   b. Set Forster radius from dye pair
   c. Fit with proper IRF convolution
   d. Cross-fit parameter linking (shared R0, tau_D0 across fits)
   e. Global fit if multiple constructs

9. Archive to MFDB
   a. All artifacts linked with provenance
   b. All parameters recorded with uncertainties
   c. Cross-references to calibrations, samples, dyes
```

#### 5.2 Workflow as MFDB Operations

Each step above maps to one or more `mfdb_operation` records:

| Step | Operation Type | Input Artifacts | Output Artifacts |
|------|---------------|-----------------|------------------|
| Load TTTR | `measurement` | -- | `raw_measurement` |
| Burst selection | `burst_selection` | `raw_measurement` | `burst_data` |
| Population gating | `population_selection` | `burst_data` | `burst_data` (filtered) |
| Decay construction | `histogram_construction` | `burst_data` + `raw_measurement` | `processed_data` (decay) |
| Background subtraction | `background_correction` | `processed_data` + `raw_measurement` (buffer) | `processed_data` (corrected) |
| Model fit | `local_fit` | `processed_data` | `fit_result` |
| Calibration | `calibration` | `raw_measurement` (ref) | `calibration_data` |
| Global fit | `global_fit` | multiple `fit_result` | `fit_result` (global) |

#### 5.3 Dependency Tracking

The workflow template should define which steps depend on which other steps. When a
calibration parameter changes (e.g., g-factor re-determined), all downstream analysis
steps should be flagged as stale. This is tracked via `mfdb_edge` relationships:

- `derived_from`: data lineage (decay derived from burst selection)
- `calibrated_by`: calibration provenance (fit uses g-factor from reference)
- `parameter_depends_on`: parameter linking (fit shares R0 with another fit)
- `supersedes`: version lineage (new analysis supersedes old)

#### 5.4 User-Facing Workflow Widget (Future)

A wizard-style widget that guides users through the workflow. Each step:
- Shows what data/parameters are needed
- Pre-fills values from MFDB if available (e.g., dye pair -> R0)
- Validates completeness before proceeding
- Archives results to MFDB on completion

This is a large GUI effort and is out of scope for the immediate work, but the
MFDB schema and operation types must support it.

### Acceptance Criteria

- All operation types from the workflow are registered in `OPERATION_TYPES` vocabulary
- The provenance graph for a complete smFRET analysis is representable in MFDB
- A test demonstrating the full workflow chain as MFDB operations exists

---

## PRD 6: mfdb-admin Enhancements for Workflow Inspection

### Problem

mfdb-admin can browse the database but does not provide workflow-oriented views for
inspecting smFRET analysis chains.

### Requirements

#### 6.1 Workflow Tree View

A tree view showing the analysis chain for a selected project:
```
Project "DNA hairpin smFRET"
  +-- Sample: HP3-Cy3-Cy5
  |     +-- Entity: DNA hairpin (sequence...)
  |     +-- Donor: Cy3 (QY=0.15, abs_max=550nm)
  |     +-- Acceptor: Cy5 (QY=0.27, abs_max=649nm)
  |     +-- R0 = 54 A (kappa2=2/3, n=1.33)
  +-- Calibrations
  |     +-- g-factor = 1.02 (from Rh110 measurement)
  |     +-- gamma = 0.85 (from donor-only)
  |     +-- crosstalk = 0.03
  +-- Measurement: DA_sample_001.ptu
  |     +-- Burst selection (APBS, M=50, T=500us)
  |     |     +-- 15,234 bursts detected
  |     |     +-- GMM: 2 populations (E=0.35, E=0.82)
  |     +-- Decay: donor channel (population 1)
  |     |     +-- Fit: 2-Gaussian FRET
  |     |     |     +-- R1 = 52.3 +/- 1.2 A
  |     |     |     +-- R2 = 68.1 +/- 2.3 A
  |     |     |     +-- chi2_r = 1.03
  |     +-- Decay: donor channel (population 2)
  |           +-- Fit: 1-Gaussian FRET
  |                 +-- R = 38.5 +/- 0.8 A
  |                 +-- chi2_r = 1.01
  +-- Fit dependencies (chinet graph)
        +-- tau_D0 linked across all fits
        +-- R0 linked across all fits
```

#### 6.2 Parameter Comparison View

A table showing parameters across fits/versions/branches:
- Compare same parameter across different constructs (e.g., R_DA for different
  labeling positions on the same protein)
- Show parameter evolution across project versions
- Highlight parameters that differ from linked source

#### 6.3 Staleness Detection

When a calibration parameter or reference measurement is updated, highlight all
downstream fits that used the old value. Use the `mfdb_edge` provenance graph to
trace dependencies.

### Acceptance Criteria

- mfdb-admin shows a tree view of the analysis workflow for any project
- Parameters can be compared across fits in a table view
- Stale analysis steps are visually flagged

---

## Implementation Priority

### Phase 1: Make It Work (PRD 1)
**Goal**: MFDB project round-trip produces correct results.

1. ~~Fix fit group structure preservation in archiver~~ ✅ (fit_id, name, model_name, plot_state, fit_range, local_fits_count in artifact metadata)
2. ~~Fix dataset restore (role filtering, key collision)~~ ✅ (_parse_ds_id + multi-level fallback chain)
3. ~~Restore chinet sessions and parameter dependencies~~ ✅ (restore_project_from_artifacts queries fit artifacts and parameters, version-scoped)
4. ~~Persist project-level metadata~~ ✅ (chisurf_version, project_format_version, description, created, ui_state, experiments)
5. ~~Write round-trip test~~ ✅ (all 6 tests passing)

**Completed tasks (Phase 1, 2026-06-17):**

- **R7-1**: Scoped fit artifact query to version-specific operations using LIKE pattern `fit_{version_id}:%`
- **R7-2**: Updated _restore_parameter_links_from_edges to use `parts[0].startswith("fit_")` instead of `parts[0] == "fit"` and fixed operation_id matching logic
- **R7-3**: Made dependency_edges functional by adding dependency_edges and parameters fields to Project dataclass, updating Project.from_dict to extract and pass dependency_edges, modifying apply_state_to_fit call sites in core_fit.py to pass dependency_edges filtered by fit operation_id
- **R7-4**: Changed fit_operation_ids from list to set to avoid duplicates
- **R7-5**: Improved roundtrip tests to verify actual content (dataset x/y arrays, parameter values)
- Removed incorrect object_store_root parameter from MFDatabase constructor calls (6 occurrences)
- Wired parameters + dependency_edges through _reconstruct_payload in services.py
- Implemented _restore_parameter_links_from_edges function in fit_state.py
- Fixed restore_project_from_artifacts to query fit artifacts from specific version
- Added operation_id to fit artifact query result
- Converted fit_operation_ids set to list for SQL IN clause compatibility

**All 106 MFDB tests passing** (roundtrip, integration, chinet adapter, credentials, user management, object store, auth)

**Files modified:**
- `chisurf/core/mfdb/project_archiver.py`: Fixed fit artifact query with version-scoping, added operation_id column, fixed set/list conversion
- `chisurf/core/project/fit_state.py`: Implemented _restore_parameter_links_from_edges function
- `chisurf/core/project/project.py`: Added dependency_edges and parameters fields to Project dataclass
- `chisurf/macros/core_fit.py`: Updated apply_state_to_fit calls to pass dependency_edges
- `chisurf/plugins/core/project_browser/backend/services.py`: Added parameters and dependency_edges to _reconstruct_payload return
- `test/fio/test_mfdb_project_roundtrip.py`: Updated test assertions for curves structure

### Phase 2a: SQLAlchemy MFDB Mapping (PRD-020) — NEW
**Goal**: Establish a deliberate SQLAlchemy mapping boundary before adding more schema-heavy dictionary and sample behavior.

1. Add SQLAlchemy as an explicit project dependency
2. Create `chisurf/core/mfdb/orm/` with base/session helpers
3. Map sample/probe/FRET/vocabulary tables without making ORM the schema source of truth
4. Fix sample-scoped FRET pair modeling (R14-1)
5. Preserve existing `MFDatabase` public APIs while moving sample graph internals behind the adapter
6. Add schema/mapping consistency tests

**Files**: `chisurf/core/mfdb/orm/*`, `chisurf/core/mfdb/schema.py`, `chisurf/core/mfdb/sample_manager.py`, `chisurf/core/mfdb/repository.py`
**PRD**: `overhaul/PRD-020-sqlalchemy-mfdb-mapping.md`

### Phase 2b: mmCIF Dictionary Infrastructure (PRD-02a) — NEW
**Goal**: Parse bundled mmCIF/flrCIF .dic files for vocabulary validation.

1. Rewrite `pdbx_metadata.py` → `MmcifDictionary` class parsing all 7 .dic files
2. Extract categories, items, descriptions, types, enumerations
3. JSON cache with auto-regeneration when .dic files change
4. `validate_value()` for runtime field validation
5. CLI introspection (`python -m chisurf.core.mfdb.pdbx_metadata`)

**Files**: `chisurf/core/mfdb/pdbx_metadata.py`, `chisurf/core/mfdb/data/update_dictionaries.sh`
**Depends on**: Phase 2a (SQLAlchemy MFDB mapping)
**PRD**: `overhaul/PRD-02a-mmcif-dictionary-infrastructure.md`

### Phase 2c: Deep Sample Description (PRD-02) — REWRITTEN
**Goal**: Sample creation populates full flrCIF data model with vocabulary validation.

1. `SampleDefinition` uses Optional (not sentinels), `ProbeDefinition` dataclass
2. `create_sample` populates all flr_* tables (entity, probes, positions, conditions, Förster radius)
3. `get_sample_full_description` returns complete structured graph
4. Vocabulary validation against parsed .dic enumerations
5. `validate_sample_for_export` checks flrCIF export readiness
6. flrCIF round-trip test: create → export → import → compare

**Depends on**: Phase 2a (SQLAlchemy MFDB mapping), Phase 2b (dictionary infrastructure)
**PRD**: `overhaul/PRD-02-sample-tracking.md`

### Phase 2c': Admin Overhaul (PRD-02b) — NEW
**Goal**: Manual inspection and editing of the PRD-02 data model through the mfdb-admin GUI before progressing to downstream PRDs.

1. Wire `sample_manager` functions (`create_sample`, `get_sample_full_description`, `validate_sample_for_export`) into admin RPC handlers
2. Overhaul Sample tab: structured editor for entities, probes, FRET pairs, condition
3. Make Entities/Probes/Positions tabs editable with add/delete
4. New FRET Pairs tab: view/add/edit Förster radius records per sample
5. Full-description preview panel and export validation display
6. PDBx key autocomplete in Metadata tab

**Depends on**: Phase 2c (PRD-02 sample tracking)
**Blocks**: Phase 2d (pipeline), Phase 3 (metadata), Phase 4 (workflow inspection)
**PRD**: `overhaul/PRD-02b-mfdb-admin-overhaul.md`

### Phase 2d: Connect the Pipeline (PRD 2-orig + PRD 3.1-3.2)
**Goal**: smFRET workflow steps appear in MFDB provenance.

1. Fix burst pipeline vocabulary — NOT DONE (`pipeline.py` still has `artifact_type` instead of `artifact_kind`, `"local"` instead of `"local_file"`, `"success"` instead of `"succeeded"`)
2. Burst selection archives to MFDB on completion
3. Project archiver discovers and links burst artifacts
4. Sample and experiment association (using PRD-02 infrastructure)

### Phase 3: Enrich Metadata (PRD 3.3-3.6 + PRD 4)
**Goal**: Full FLR metadata linkage and fluorophore catalog.

1. Dye/probe properties linkage (uses PRD-02 ProbeDefinition)
2. Instrument/setup from TTTR headers
3. Calibration parameter provenance
4. Buffer measurement linkage
5. Expand fluorophore database with real spectra

### Phase 4: Workflow and Inspection (PRD 5 + PRD 6)
**Goal**: Guided workflows and rich inspection tools.

1. Define operation types for full smFRET workflow
2. Workflow dependency tracking in MFDB
3. mfdb-admin workflow tree view
4. Parameter comparison and staleness detection
5. (Future) Wizard-style workflow widget

---

## Ideas — Evaluate Later

### Object Store: Deduplication for Sliced/Derived Data

**Context:** TTTR source files are essentially random photon timestamps — CDC
(content-defined chunking à la Duplicacy) won't find shared chunks across
different measurements. However, *sliced* or *derived* data (burst selections,
sub-ranges, processed histograms) may appear identically across multiple
projects or analyses.

**Current state:** Whole-file MD5 dedup works well for identical blobs. No
sub-file dedup exists.

**Possible directions:**
- Evaluate whether sliced photon data (burst windows, time ranges) produces
  identical derived blobs often enough to justify better dedup
- If yes: rolling-hash chunking (Buzhash) with numba `@njit` inner loop for
  the hash pass, chunk manifests per logical object, reassembly on read
- Compression (zstd) on all blobs regardless — stacks with any dedup strategy
- Artifacts are immutable from the user's perspective — they are internal
  storage, never directly accessed or modified by users

**Prerequisite:** Real-world usage data on object store growth patterns. Don't
optimize before measuring.

---

## TODO: Serialization Layer — ZMQ-MessagePack

### Problem

The GUI-server communication currently uses JSON-RPC over ZMQ. JSON serialization
is slow for large binary payloads (TCSPC histograms, burst arrays, spectra) and
adds unnecessary base64 encoding overhead for numpy arrays.

### Design Principle

Three serialization boundaries, each with its own format:

| Boundary | Format | Rationale |
|----------|--------|-----------|
| **Inside Python** | Native Python dicts, numpy arrays | No serialization overhead. Functions accept/return plain objects. |
| **ZMQ client ↔ server** | MessagePack | Binary-efficient, handles bytes natively. No base64 needed for arrays. Faster than JSON for large payloads. |
| **User-facing files** (.csp, exports, logs) | JSON | Human-readable, inspectable, diffable. Users should be able to open and read these. |

### Requirements

1. Replace JSON-RPC encoding on the ZMQ transport with MessagePack
2. Keep all user-facing serialization (`.csp` project files, MFDB export archives,
   log output, config files) as JSON
3. Internal Python code stays on native dicts — no JSON string intermediaries
   between functions in the same process
4. `_json_dumps` / `_json_loads` remain for MFDB metadata storage (SQLite text
   columns) and user-facing file I/O
5. Add `msgpack` as a dependency

### Files to modify

- `chisurf/server/zmq_transport.py` — switch wire format to msgpack
- `chisurf/gui/client.py` (or equivalent ZMQ client) — match wire format
- `chisurf/core/experiments/core/serialize.py` — provide msgpack-aware array
  serialization (pack numpy arrays as raw bytes + dtype + shape header, not
  base64 strings)
- Plugin RPC handlers — no changes needed if the transport layer handles
  serialization transparently

### Scope

This is a transport-layer change. It should not affect:
- MFDB storage (SQLite metadata stays JSON text)
- `.csp` file format (stays JSON inside ZIP)
- Test fixtures or assertions that compare dict structures
- Any function signature — callers send/receive Python dicts before and after

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `chisurf/core/mfdb/schema.py` | Database schema (v28), all table definitions |
| `chisurf/core/mfdb/models.py` | Dataclasses and vocabulary constants |
| `chisurf/core/mfdb/repository.py` | Database access layer (MFDatabase class) |
| `chisurf/core/mfdb/project_archiver.py` | Project -> MFDB archiving and restoration |
| `chisurf/core/mfdb/chinet_adapter.py` | Chinet <-> MFDB persistence backend |
| `chisurf/core/mfdb/pipeline.py` | Burst pipeline MFDB integration |
| `chisurf/core/mfdb/api.py` | RPC-style API with auth integration |
| `chisurf/core/mfdb/graph.py` | Provenance graph traversal |
| `chisurf/core/mfdb/object_store.py` | Content-addressed blob storage |
| `chisurf/core/mfdb/database_resolver.py` | DB path resolution and backup |
| `chisurf/core/mfdb/seed_data.py` | Curated fluorophore seed data |
| `chisurf/core/mfdb/auth/` | RBAC, sessions, ACLs |
| `chisurf/core/project/project.py` | Project dataclass and .csp save/load |
| `chisurf/core/project/archive.py` | ZIP archive format with MFDB layer |
| `chisurf/core/project/fit_state.py` | Fit parameter serialization |
| `chisurf/core/project/registry.py` | Runtime UID registry |
| `chisurf/core/experiments/tcspc/tttr_reader.py` | TTTR file loading |
| `chisurf/core/models/tcspc/lifetime.py` | Multi-exponential lifetime models |
| `chisurf/core/models/tcspc/fret.py` | FRET distance distribution models |
| `chisurf/core/models/pda/nusiance.py` | PDA correction parameters |
| `chisurf/core/fluorescence/fret/__init__.py` | Intensity-based FRET functions |
| `chisurf/core/fluorescence/burst/` | Burst detection algorithms |
| `chisurf/plugins/burst/burst_selection/` | Burst selection plugin |
| `chisurf/plugins/core/mfdb_admin/` | Database admin GUI plugin |
| `chisurf/server/services/projects.py` | Server-side project handling |
| `chisurf/core/actions/project_actions.py` | Project action dispatch |
