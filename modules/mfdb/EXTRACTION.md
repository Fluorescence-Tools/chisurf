# MFDB extraction state

Current state as of 2026-07-06:

- Source package moved to `modules/mfdb/src/mfdb`.
- MFDB dictionaries (`*.dic`) live under `modules/mfdb/src/mfdb/data/`; the
  old `chisurf/core/mfdb/data` copy has been removed.
- MFDB Admin lives inside the MFDB package at `mfdb.admin` as the optional admin
  application. The ChiSurf `core/mfdb_admin` plugin is a thin manifest/wrapper
  shim around `mfdb.admin`.
- ChiSurf package discovery includes `mfdb*` from `modules/mfdb/src`.
- `chisurf.core.mfdb` is a transitional facade for existing imports.
- Loaded `mfdb.*` submodules are aliased under `chisurf.core.mfdb.*` so
  transitional imports do not create duplicate class objects.
- MFDB Admin, Database Connector, and Project Browser import `mfdb` directly in
  their active backend/test paths. MFDB Admin app code imports through
  `mfdb.admin`.
- Structured sample creation uses the SQLite repository path by default. The
  SQLAlchemy ORM adapter remains available through the optional `orm` extra.
- Structured sample full-description reads have a stable raw-SQL fallback shape,
  including nullable condition fields when no condition row exists.
- Runtime path/default-user configuration lives in `mfdb.config`; `mfdb` no
  longer imports ChiSurf settings for database path, object-store root, or the
  default user.
- `mfdb.result_registry` resolves through explicit DB arguments,
  `set_global_db`, or the MFDB-configured database path; it no longer imports
  the ChiSurf `database_connector` singleton.
- `mfdb.payload_models.GenericCurve` now serializes/deserializes plain curve
  arrays without importing ChiSurf `DataCurve`.
- `mfdb.seed_data` computes seed Förster overlap/radius values locally and no
  longer imports ChiSurf fluorescence helpers.
- `mfdb.project_archiver` encodes project curve arrays locally and no longer
  imports ChiSurf experiment serialization helpers.
- The fluorophore reference `spectra.db` is bundled under
  `modules/mfdb/src/mfdb/data/`; `MFDatabase.import_reference_set()` resolves
  that package-local file by default or an explicit `MFDB_REFERENCE_SPECTRA_DB`.
- Probe-type reseeding uses a non-destructive upsert so imported probes keep
  valid `probe_types` foreign keys.

The package is not yet independently clean. Remaining ChiSurf dependencies must
be removed before publishing to `github.com/fluorescence-tools/mfdb`:

- `mfdb.chinet_adapter` imports ChiSurf settings and fit-state helpers.

Next extraction steps:

1. Move ChiSurf-specific fit-state and ChiNet adapters behind adapter modules
   outside the core `mfdb` package.
2. Run a standalone package test command from `modules/mfdb` without importing
   `chisurf`.
3. Delete the `chisurf.core.mfdb` facade after all callers use `mfdb`.
