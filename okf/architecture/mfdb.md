---
type: Architecture
title: MFDB — Metadata / Provenance Store
description: SQLite-backed metadata and provenance store with canonical tables generated from mmCIF dictionaries.
resource: modules/mfdb/src/mfdb/
tags: [mfdb, metadata, provenance, sqlite, mmcif]
timestamp: '2026-07-06T00:00:00Z'
---

# Purpose

`modules/mfdb/src/mfdb/` is the vendored SQLite-backed metadata and
provenance package. It is versioned with migrations, has ACL/auth, and models a
provenance DAG (operations -> artifacts -> edges). MFDB dictionary files
(`*.dic`) live in `modules/mfdb/src/mfdb/data/`; the old
`chisurf/core/mfdb/data` dictionary copy has been removed.

MFDB Admin is part of this package as the optional `mfdb.admin` application
(`modules/mfdb/src/mfdb/admin`). The ChiSurf `core/mfdb_admin` plugin is a thin
manifest/wrapper shim around `mfdb.admin`, not the owner of the implementation.

`chisurf.core.mfdb` is now a prerelease compatibility facade. New ChiSurf code
should import `mfdb` directly so the package can later move to
`github.com/fluorescence-tools/mfdb` without another application-wide import
cutover. During the transition, the facade aliases loaded `mfdb.*` submodules
under `chisurf.core.mfdb.*` so old imports do not create duplicate class objects.

Current extraction boundary: runtime path/default-user/object-store
configuration, result registration, generic payload serialization, project
curve-array encoding, and seed Förster overlap/radius calculations are MFDB
local. Remaining ChiSurf-specific imports are concentrated in ChiNet adapter
code.

The fluorophore reference `spectra.db` used by seeded/reference imports is now
bundled under `modules/mfdb/src/mfdb/data/`. `MFDatabase.import_reference_set()`
uses that package-local database by default or an explicit
`MFDB_REFERENCE_SPECTRA_DB` override.

Known design issue addressed during extraction: probe-type reseeding now uses a
non-destructive upsert instead of `INSERT OR REPLACE`, preserving `probe_types`
primary keys referenced by imported probes.

Structured sample creation uses the SQLite repository path by default; the
SQLAlchemy ORM adapter is optional. Raw-SQL full-description reads expose a
stable shape, including nullable condition fields when no condition row exists.

# Schema authority

The canonical tables are **generated from mmCIF dictionaries** (`data/*.dic` —
the wwPDB/PDB-IHM family plus the local `mfdb_flr_ext.dic`) by
`schema_from_dictionary.py`. Treat the `.dic` dictionaries as the schema
authority: change them and regenerate; do **not** hand-edit generated DDL.

## Dictionary programming rules

For flrCIF/PDBx fields the bundled `.dic` files are the source of truth; Python may
reflect tables and call repository helpers but must not become the authority for
item names, schema aliases, enums, defaults, descriptions, or mandatory flags. When
upstream flrCIF is incomplete, add store-local definitions to
`modules/mfdb/src/mfdb/data/mfdb_flr_ext.dic` **before** writing Python that
persists or validates the field. Every persisted dictionary item declares
`_item.name`, `_item.category_id`, `_item_type.code`, `_item.mandatory_code`,
`_item_enumeration.value` (controlled values), `_item_default.value` (defaults), and
`_mfdb_schema.table_name` / `_mfdb_schema.column_name`.

**Idempotent identity rows.** Rows representing identity-like objects are looked up
by their natural key before insert (e.g. chemical descriptors keyed by
`(descriptor_type, descriptor, program, program_version)`; vocabulary rows by
declared name). Prefer a DB uniqueness constraint; where legacy schema makes that
impractical, repository code provides an idempotent upsert and tests prove repeated
writes reuse the same row. A dictionary change is not complete until tests prove
items map to live columns, enums/defaults drive behaviour, and repeated writes do
not duplicate identity rows.

# Package layout

The package (`modules/mfdb/src/mfdb/`) is organized into concern subpackages,
not a flat module list. Top level holds only the facade/entry surface —
`repository.py` (`MFDatabase`), `api.py`, `models.py`, `config.py`,
`chinet_adapter.py`, and `__init__.py` — over these groups:

- `schema/` — dictionary-driven schema engine: `schema`, `schema_from_dictionary`,
  `dictionary_schema_map`, `pdbx_metadata`, `dao` (`DictionaryDao`),
  `vocabulary_loader`, `docs_generator`, `_sqlutil`.
- `store/` — persistence primitives: `object_store`, `payload_codec`,
  `payload_models`, `database_resolver`, `transactions`.
- `provenance/` — DAG + results: `lineage`, `graph`, `result_registry`,
  `compute_spec`, `operation_parameters`.
- `samples/` — sample domain: `sample_manager`, `sample_requests`,
  `external_refs`, `reagents`, `importer`, `seed_data`.
- `lifecycle/` — state/events/audit: `lifecycle`, `event_log`, `events`, `staleness`.
- `security/` — identity & access: `auth` (Principal, token→principal, ACL/permissions,
  sessions), `credentials`, `session`, `boundary_validation`, `base`, plus the
  **pluggable authentication** layer (PRD-59): `auth_providers` (`AuthProvider` protocol +
  `LocalAuthProvider` / `LdapAuthProvider`) and `login` (the `login()` orchestrator that
  authenticates → resolves/JIT-provisions the MFDB user via `flr_sample_users.auth_provider`/
  `external_id` → maps directory groups → mints the session). The `mfdb.security.auth.login`
  RPC and `mfdb-admin auth` CLI both route through `login()`; `ldap3` is an optional/lazy
  `[ldap]` dependency, so `security/` still imports with only `src` on the path.
- `project/` — `project_archiver`.
- `adapters/` — bridges to external systems / host apps: `chinet` (ChiSurf fit
  sessions → MFDB), with electronic-lab-notebook adapters (eLabFTW, …) to follow.
  Adapters may depend on their target but keep those imports lazy.
- `queries/` — per-concern **`MFDatabase` mixins** (god-class breakup, PRD-26):
  `artifacts` (artifact/operation/edge/provenance-graph core), `analysis`,
  `samples`, `setups`, `probes`, `objects`, `parameters`, `protocols`, `studies`,
  `branches`, `lifecycle`, `experiments`, `users`. `repository.py` is a thin
  composition of these over `self.conn`/`self.dao`/`self.lineage` — reduced from
  ~6,400 lines to ~1,850 (init/connection/properties, migration, audit,
  vocabulary, experiment key-values, pdbx metadata remain as the core).
- `admin/` — the admin RPC service (`backend/`, `cli/`); chisurf-free and import-
  clean. The chisurf-coupled admin **GUI** lives in the ChiSurf plugin
  (`chisurf/plugins/core/mfdb_admin/gui/`), not in the package.
- `data/` — bundled `.dic` dictionaries and config JSON.

The package has its own **hermetic, standalone test suite** at
`modules/mfdb/tests/` (isolated via `MFDB_SETTINGS_DIR`, no chisurf import) —
run it with `cd modules/mfdb && PYTHONPATH=src pytest`. ChiSurf↔MFDB
integration tests (admin GUI client, chinet/parameter-registry alignment, etc.)
stay in the ChiSurf `test/` tree. This boundary is enforced by construction:
the core package imports cleanly with only `src` on the path (PRD-24).

# API

`api.py` is the transport-agnostic function API for the store.

## Data access — one engine, raw SQL is an antipattern

CRUD goes through the dictionary-driven **`DictionaryDao`** (`db.dao`), which
whitelists every identifier against the reflected schema and binds all values:
`db.dao.insert / upsert / get / list / update / soft_delete`. New code **must**
use it. Hand-written / f-string / raw `db.conn.execute("INSERT …")` SQL for
create-read-update-delete is an **antipattern** — it duplicates the one engine,
bypasses schema-whitelisting and audit-column handling, and drifts from the
`.dic` source of truth (PRD-26/INC-05). Any remaining raw-SQL CRUD in the
repository is legacy debt being migrated onto the DAO, not a pattern to copy.

Raw SQL is reserved for genuinely **bespoke** statements the single-table DAO
cannot express — multi-table joins, graph/lineage traversal, aggregates, export,
`INSERT … SELECT` bulk copies, conditional/multi-column `WHERE` updates,
composite-key soft-deletes on PK-less junctions, resurrecting updates (reset
`deleted_at`), and intentional hard deletes — kept as organized methods (or
`# raw`-flagged blocks) on the relevant concern mixin. When in doubt: a
single-table CRUD shape is DAO; a join/traversal/bulk/conditional is raw.
`DictionaryDao.insert` works on keyless tables too (UNIQUE-only junctions like
`mfdb_group_member`): with no resolvable primary key it returns the last rowid
rather than raising, so even junctions can be written through the DAO.

# Object store & provenance-aware readers

MFDB has a content-addressed **object store** (`object_store.py`): blobs are
stored under `{root}/{md5[:2]}/{md5[2:4]}/{md5}` with streaming-MD5 dedup and
refcounting, shared across users, `root` defaulting to `~/.chisurf/objects`.
Blobs are addressed by UUID; the repository exposes
`put/get/get_info/delete/list_object` and the admin app surfaces them as
`mfdb.objects.*` RPC methods and an "Objects" browser.

Experiment readers can record provenance automatically. The `ExperimentReader`
base (`chisurf/core/experiments/core/reader.py`) has provenance hooks
(`operation_type`, `artifact_kind_source`/`_derived`, and instance `db` /
`object_store` / `record_provenance`). When `get_data()` runs with a `db` set it:
registers the source file(s) as objects, calls the subclass `read()`, serializes
the derived curves to a language-agnostic JSON+base64 envelope
(`chisurf/core/experiments/core/serialize.py`), registers those, and records an
`mfdb_operation` linking source → derived artifacts. The provenance DAG is
`source object → raw_data artifact → operation → processed_data artifact →
derived object`. Provenance is **opt-in and backward compatible**: readers still
work when `db` is unset, and direct `read()` calls skip provenance entirely.
TCSPC (TTTR + curve), PDA, PCH, and FCS readers declare provenance operation
types. This object-store/provenance layer is the foundation the MFDB overhaul
(the [PRDs](/prds/index.md)) builds on.

# Related work

Curation of the fluorophore database (GUI/RPC/CLI) now lives in the optional
`mfdb.admin` application. ChiSurf discovers that app through the
`core/mfdb_admin` plugin shim. MFDB is also the prototype for the broader
fdb4chembio (NFDI4Chem) fluorescence databank effort.

# Note on OKF

This [OKF](/index.md) bundle is a plain-markdown knowledge layer that sits
*beside* the code; MFDB is the runtime provenance store. OKF could serve as
an interchange/export format for MFDB provenance in the future.

# Citations

[1] [ChiSurf architecture doc](/references/architecture-doc.md)
