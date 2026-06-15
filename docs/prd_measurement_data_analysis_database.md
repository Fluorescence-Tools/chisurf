# Product Requirements Document (PRD) & Checklist: fdb

This document defines the requirements, scope, architecture, and current implementation progress of `fdb` (Fluorescence Database for Chemical Biology workflows in ChiSurf).

Architecture migration note: the canonical target for consolidating the current
phase-grown implementation is defined in
[`prd_fdb_architecture_migration.md`](prd_fdb_architecture_migration.md).

---

## 📋 Status Checklist

### Phase 0 - Project framing and compatibility
- [x] Rename the scope and project language to `fdb`
- [x] Keep existing sample database APIs and GUI names working during the transition
- [x] Document database records as provenance metadata

### Phase 1 - Burst Selection TTTR provenance slice
- [x] Add schema migration (v13) for raw data references, processing runs, processed products, and provenance edges
- [x] Add repository methods for raw TTTR records, burst processing runs, processed products, and provenance edge queries
- [x] Add JSON-RPC service methods: `raw_data.*`, `processing.burst_selection.*`, `processed_data.*`, `provenance.*`, and `archive.*`
- [x] Add `run` service path to execute/record Burst Selection and record settings/outputs/provenance
- [x] Add archive manifest export and registration for one burst-processing run
- [x] Add automated unit tests verifying the Phase 1 provenance chain and migration

### Phase 2 - ndxplorer as the next consumer
- [x] Add service methods to query and load registered burst data products (`.bur` folder, HDF5, ZIP) into ndxplorer
- [x] Add service methods to record ndxplorer selections, GMM clustering results, and plot/table exports back to the database
- [x] Add automated unit tests for ndxplorer service methods (headless loading and recording of selection/GMM results)
- [x] Record ndxplorer analysis and selection mask provenance as downstream records

### Phase 3 - Setup definitions and generalized measurement records
- [x] Add setup definition tables (instruments, optical paths, lasers, detectors, ALEX/MFD settings, timing calibration)
- [x] Link measurements/experiments to setups while preserving existing experiment records
- [x] Add validation helpers for setup configurations required by TTTR/burst-wise workflows

### Phase 4 - General processing and processed-data model
- [x] Generalize Phase 1 raw/process/product pattern to FCS, microtime histograms, decays, spectra, anisotropy, and fit curves
- [x] Add APIs for upstream/downstream dependency queries across raw and processed data

### Phase 5 - Analysis, model, fit, and parameter provenance
- [x] Add analysis run records that consume processed data and produce model/fit results
- [x] Store model definitions, versions, optimizer settings, convergence, and parameter dependencies
- [x] Store fitted parameter values, errors, confidence intervals, covariance, and dataset-to-parameter mappings

### Phase 6 - Archive and exchange formats
- [x] Add JSON/JSONL provenance graph export
- [x] Extend mmCIF / FLR CIF export where the mapping is stable
- [x] Add SQLite backup archive export
- [x] Add ZIP/TAR archive export with manifest, checksums, and optional external data copies

### Phase 7 - GUI expansion
- [x] Evolve Sample Database window into a Measurement and Data Analysis Database window
- [x] Add focused views for setups, raw data, processing runs, processed products, provenance, and analyses
- [x] Add "open in ndxplorer" actions for registered burst products

### Phase 8 - Server and web readiness
- [x] Ensure all service methods are JSON-RPC-compatible and callable through local dispatch and ZMQ
- [x] Keep service interfaces stable enough for a future HTTP/websocket layer
- [x] Add audit logging for create/update/delete/archive actions


### Phase 9 - Project Archive/Restore
- [x] Add "to db" menu item under `File` in ChiSurf to archive the current project to the `fdb` database.
- [x] Implement archiving of the entire analysis project state, including all ChiNet parameters, fit parameters, fit parameter widgets, and model types.
- [x] Map the fit structure and interdependence of parameters introduced by the user during analysis.
- [x] Ensure fits have UUIDs and are correctly linked across each other to preserve provenance.
- [x] Add option in the database explorer to restore analysis projects directly from the database.
- [x] Track how the analysis was created by saving the preprocessing steps from raw data to the final analysis project.

---

## 🛠️ Phase 2 Technical Design

### Service: `ndxplorer.load_burst_product`
Loads a registered burst product by its `processed_data_id` from the database. It handles folder, file, and ZIP sources, parses them using the existing `ndxplorer.io.reader`, and returns a dictionary with parameter names and values.

**Request Parameters:**
- `processed_data_id`: The ID of the processed product record.

**Response Structure:**
```json
{
  "ok": true,
  "parameter_names": ["Mean Macro Time (s)", "N_ph", ...],
  "values": {
    "__ndarray__": true,
    "dtype": "float32",
    "data": [...]
  },
  "processed_data_id": "prod_123"
}
```

### Service: `ndxplorer.record_analysis`
Persists the results of an ndxplorer operations run (e.g. selection gating, clustering, GMM) back to the database. It creates an `fdb_processing_run` record representing the analysis step, links it to the input processed product(s) via `input_to` provenance edges, and registers the output products (e.g. selection masks, summaries) via `produced` edges.

**Request Parameters:**
- `experiment_id`: The ID of the associated experiment.
- `input_processed_data_ids`: A list of processed product IDs that served as inputs.
- `analysis_type`: Type of analysis, e.g. `"selection"`, `"gmm_clustering"`.
- `settings`: Dict containing parameters/gates.
- `products`: List of output product specifications to register (e.g., selection mask, GMM summaries).
- `operator_user_id` (optional): User ID.
- `software_version` (optional): Version of ndxplorer/ChiSurf.
- `status` (optional): `"succeeded"`, `"failed"`, etc.

**Response Structure:**
```json
{
  "ok": true,
  "processing_run": {
    "processing_id": "proc_456",
    ...
  }
}
```

---

## 🛠️ Phase 4 Technical Design

### Service: `processing.run.record`
Registers and records a generic processing run of any type (e.g., FCS correlation, TCSPC fitting). Consumes lists of raw and/or processed data input IDs, registers the outputs, and establishes all incoming/outgoing provenance edges in the dependency graph.

### Services: `provenance.dependencies.upstream` / `downstream`
Performs a recursive query (using SQLite recursive Common Table Expressions) starting from a specified seed node, tracing upstream to identify raw/intermediate ancestors, or downstream to identify all derived results/products.

---

## 🛠️ Phase 9 Technical Design

### Project Archival
The objective is to provide a save/restore mechanism for full ChiSurf analysis projects using the `fdb` database instead of a simple project file. This ensures strong provenance, linking the analysis project state directly back to the raw/processed data it was derived from.

**Archival workflow (`File -> to db`):**
1. **Traverse State:** Serialize the current state of the ChiSurf project, capturing all ChiNet models, fit parameter settings, interdependent constraints, UI widget states, and metadata.
2. **Provenance Tracking:** Map the current project state to its upstream preprocessing steps. Identify which raw and processed data records in `fdb` were used.
3. **Database Insertion:** Create a new composite analysis record in `fdb_analysis_run` (or an overarching project container record). All fits within the project must have UUIDs and record their interdependence via `fdb_provenance_edge`. All parameters must be stored in `fdb_analysis_parameter`.
4. **Linking:** Link the project record downstream from the relevant data records to maintain an unbroken provenance chain from raw TTTR files to the final project state.

### Project Restoration
**Restoration workflow (via Database Explorer):**
1. **Query Database:** Display archived projects in the GUI Database Explorer alongside standard processing runs.
2. **Deserialize State:** When a user selects a project to restore, load the project record, fetch all associated parameters and fit structures, and reconstruct the ChiNet models and widget states.
3. **Reconstruct Data:** If necessary, pull the corresponding raw/processed data into memory to allow the user to immediately resume analysis seamlessly.
4. **Maintain UUIDs:** Ensure that restored fits and parameters keep their original UUIDs so that subsequent saves ("updates") correctly version or link back to the same conceptual entities.
