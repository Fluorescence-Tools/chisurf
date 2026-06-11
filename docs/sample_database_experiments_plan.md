# Sample Database Experiments Plan

## Goal
Extend the built-in ChiSurf sample database from a sample/probe registry into a lightweight LIMS that preserves links between samples, experiments, operators, devices, metadata, and raw experimental data.

## Core Principles
- Keep curated/source data separate from the user-editable database.
- Put the database connector plugin in core so all database-backed plugins use the same resolver, migration, backup, repository, and JSON-RPC facade.
- Prefer FLR CIF / PDBx / PDB-IHM-compatible metadata where possible.
- Preserve provenance: sample -> experiment -> data -> operator/device/project.
- Support both small embedded data and external raw data references.
- Keep the GUI backend JSON-RPC compatible for later ZMQ/web service use.
- File loading should eventually be able to open experiment data directly from database links.

## Data Model

### Experiment Types
Users can define experiment types such as:

- Imaging
  - FLIM
- Single-molecule
  - ALEX
  - MFD
- TCSPC
- FCS
- Spectra
- Other/custom types

Represented in the database as user-definable records with optional category, description, and details.

### Experiments
Experiments should be linked to:

- sample
- experiment type
- project
- operator/user
- device/instrument
- start/end time
- status
- optional metadata key/value pairs

Experiments should also carry FLR CIF/PDBx/PDB-IHM-compatible metadata when available.

### Experiment Data
Experiment data can be:

- embedded in the database for small datatypes:
  - FCS
  - TCSPC
  - spectra
  - FLIM metadata
  - ALEX/MFD summaries
- linked to external raw data:
  - URL
  - file path
  - folder path
  - raw data directory

The database should record storage mode, file path, URL, folder path, MIME type, size, checksum, and metadata.

## GUI Plan
Add sample database GUI sections for:

1. Experiment types
   - create/edit/delete custom types
   - optional category and description

2. Experiments
   - list experiments
   - filter by sample/project/type/status
   - edit experiment metadata
   - link experiment to sample, type, user, and device

3. Experiment data
   - add embedded data for small datasets
   - add file/folder/URL links for raw data
   - open linked files from the database
   - preserve metadata with the experiment

4. File loader integration
   - expose database experiment data links to ChiSurf's file-loading path
   - allow opening an experiment from the database and loading its linked data
   - later support direct embedded data loading where appropriate

## Backend/API Plan
Add JSON-RPC-compatible service methods for:

- experiment type CRUD
- experiment CRUD
- experiment data CRUD
- experiment data resolution/opening
- sample-linked experiment listing
- export/import of experiment metadata

## Implementation Phases

### Phase 0 - Core database connector plugin
- Add a core database connector plugin responsible for:
  - resolving source/user database paths
  - copying the curated source database into the user database location
  - backing up user databases before migrations or resets
  - opening SQLite connections with migrations applied
  - exposing repository/import/export helpers through JSON-RPC-compatible services
- Register core plugin manifests before optional GUI plugins.
- Keep the connector independent of the sample database UI; sample database and future experiment modules should call the connector instead of opening DB paths directly.
- Status: implemented.

### Phase 1 - Database foundation
- Add schema tables for experiment types, experiments, experiment data, and optional key/value metadata.
- Add repository models and CRUD methods.
- Add migration with backup-before-migration.
- Seed curated sample database with example experiment types and example experiments.
- Status: implemented.

### Phase 2 - Sample database uses connector
- Refactor sample database services to obtain the active repository from the core database connector.
- Keep source/user DB resolver and reset/backup behavior inside the connector.
- Add tests proving the connector initializes a user DB, applies migrations, and exposes repository methods.

### Phase 3 - GUI management
- Add experiment type management tab.
- Add experiment management tab.
- Add experiment data link/embedding editor.
- Persist GUI layout and selected records.

### Phase 4 - File opening integration
- Add service method to resolve experiment data into local file paths or embedded payloads.
- Add GUI action to open linked data files from the database.
- Add minimal integration point for ChiSurf file loaders to consume database experiment links.

### Phase 5 - FLR CIF/PDBx/PDB-IHM alignment
- Map database experiment metadata to FLR CIF/PDBx/PDB-IHM concepts where possible.
- Add export support for experiment metadata.
- Add import support for experiment metadata from FLR CIF/PDBx/PDB-IHM files.

### Phase 6 - LIMS workflows
- Improve project/user/device metadata.
- Improve provenance tracking.
- Add validation and migration helpers.
- Add server/web JSON-RPC support using the existing service layer.

## Notes
- Do not implement all phases at once.
- Keep changes incremental and testable.
- Avoid writing to the curated source database.
- Always preserve existing user databases through migration backups.
