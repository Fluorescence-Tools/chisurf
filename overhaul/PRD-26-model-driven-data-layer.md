# PRD-26: Model-Driven Data Layer — the `.dic` generates the system (Architecture J)

## Goal

Extend "dictionary dictates schema" to "dictionary dictates the **system**":
generate the **repository/DAO**, the admin **entity registry**, **RPC parameter
validation**, and **API/schema docs** from the same `.dic`, so the hand-maintained
surface (and the drift between schema, ORM, admin, and validation) largely
disappears.

## Evidence (why)

The `.dic` already generates DDL (PRD-04) and, after PRD-19, vocab + the whole
canonical schema declaratively. But the layers *above* the schema are still
hand-written and drift from it:

- The admin `entity_registry` / `entity_schema` derive `FieldSpec` from the `.dic`
  for *forms*, but the entity list, columns, and CRUD are still partly hand-coded.
- RPC parameter validation and request models are hand-written per handler.
- Repository methods are hand-written SQL strings (N+1, f-string injection risk,
  inconsistent with the dictionary types).

Every hand-maintained copy of "what the fields are" is a drift/bug surface (the
`QComboBox`/field bugs, the operation-parameter bag).

## Design

A code/metadata generator over the parsed dictionary (`MmcifDictionary` +
`DictionarySchemaMap` + the PRD-19 reconcile), producing:

1. **Typed DAO / repository accessors** per category: generated
   `get/list/insert/update(soft-delete)` with typed columns, parameterised SQL (no
   f-strings), and consistent scoping hooks (owner/visibility, soft-delete). Hand
   SQL remains only for genuinely bespoke queries (lineage, browse).
2. **Admin entity registry** generated from the `.dic`: which categories are
   editable, their columns, labels, enums, and FK targets — replacing the
   hand-maintained `EntitySpec` list (the `FieldSpec` derivation already proves it).
3. **RPC parameter validation**: request/parameter schemas derived from the `.dic`
   (types, units, bounds, required, repeatable — reusing PRD-11
   `mfdb_operation_parameter_def`) so validation is at the boundary and
   dictionary-typed, not hand-written per handler.
4. **API + schema docs**: a generated reference of tables, columns, types, vocab,
   and operation parameter schemas — one document, always in sync.

The generator is the single extension point: adding/altering a field in the `.dic`
updates schema, DAO, admin, validation, and docs together.

## Tasks

1. Generator core over `MmcifDictionary`: emit typed DAO accessors per category
   (parameterised SQL, soft-delete, scoping).
2. Migrate the repository to generated accessors where the query is CRUD-shaped;
   keep bespoke queries explicit.
3. Generate the admin entity registry from the `.dic`; retire the hand-maintained
   `EntitySpec` list.
4. Derive RPC parameter validation from the `.dic` (with PRD-11 param schemas).
5. Generate API/schema docs from the dictionary.
6. Tests: a `.dic` field edit propagates to DAO + admin + validation + docs with no
   hand edits; generated SQL is parameterised; round-trip CRUD per category.

## Definition of Done

- [ ] DAO accessors, admin registry, RPC validation, and docs are generated from
      the `.dic`; the hand-maintained copies are removed.
- [ ] Adding/altering a field is a `.dic` edit only; all layers update together.
- [ ] Generated data access is parameterised (no f-string SQL); CRUD round-trips.

## Definition of Clean

`.dic` is the single spec for the whole stack; no hand-maintained field lists or
f-string SQL; generated + bespoke-query separation is explicit; behavior-asserting
tests that a single `.dic` edit propagates everywhere.

## Implementation status

**Task 1 (DAO generator core) — landed.** `chisurf/core/mfdb/dao.py` provides
`DictionaryDao`: generic, fully-parameterised CRUD (`insert/get/list/update/
soft_delete`) over any dictionary-declared table. Table/column identifiers are
**whitelisted against the live schema** (built via `DictionarySchemaMap` /
`introspect_sqlite_schema`), so identifiers never carry caller input and all values
are bound parameters — closing the f-string-SQL/injection surface. Conventions are
applied automatically when a table declares them: `deleted_at` soft-delete (default
reads hide deleted rows; `soft_delete` falls back to hard `DELETE` when absent) and
an `updated_at` touch on update. Constructors: `from_connection` (introspect a live
`MFDatabase.conn`), `from_dictionary_map`, `from_db_path`. Covered by
`test/fio/test_dao.py` (CRUD round trip + soft-delete on `flr_sample`,
unknown-table/column rejection, PK-immutable, and a parameterisation/no-injection
assertion).

**Task 2 (migrate repository CRUD onto the DAO) — in progress.** `MFDatabase`
exposes a lazy `dao` accessor (`DictionaryDao.from_connection(self.conn)`, built after
schema reconcile). Migrated so far — the `_row_to_dict(SELECT * … WHERE pk=?)`
get-by-PK family, each behaviour-equivalent (`_row_to_dict` ≡ `dict(row)`;
`include_deleted=True` preserves the historical "return regardless of soft-delete"
semantics) and replacing a hand SELECT:
- `get_artifact` → `dao.get("mfdb_artifact", …, include_deleted=True)`
- `get_operation` → `dao.get("mfdb_operation", …, include_deleted=True)`
- `get_parameter` → `dao.get("mfdb_parameter", …, include_deleted=True)`
- `get_object_info` → `dao.get("mfdb_object", …, include_deleted=True)`

Verified: all operation/parameter/object/registry/burst-pipeline suites pass; the
only reds are pre-existing migration/owner-backfill failures (v35/v39), A/B-confirmed
unchanged with the migration reverted. Plus a dedicated `get_artifact` soft-delete
regression test.

Soft-delete family migrated: `DictionaryDao.soft_delete` gained an optional
`deleted_at` value (defaults to `CURRENT_TIMESTAMP`), so the single-table
`delete_citation`/`delete_probe`/`delete_spectrum`/`delete_setup` now delegate to
`dao.soft_delete(table, id, pk_column=…, deleted_at=_utc_now())` — exact marker
format preserved; the only delta is the now-idempotent `AND deleted_at IS NULL` guard
(callers ignore the rowcount). Verified by `test_fdb_setups` + dao tests.

**Still deferred (need care):**
- `get_sample` returns a `sqlite3.Row` with an explicit column subset (not `SELECT *`/
  dict) — migrating changes the return type/shape; audit callers first.
- `delete_parameter` does a dual-key UPDATE (`parameter_uuid` *and* `parameter_id`) —
  legacy two-column delete, not a single `soft_delete`.

**Task 5 (generate API/schema docs from the `.dic`) — landed.**
`chisurf/core/mfdb/docs_generator.py` renders a Markdown reference straight from the
dictionary: tables → columns (type, required, FK, allowed-values, description),
controlled vocabularies (from `.dic` enumerations + enum-details), and the PRD-11
operation parameter schemas (from `data/operation_parameter_defs.json`). It is
dictionary-only (no DB needed; default scope = `flr_*`/`mfdb_*` tables) and can be
restricted to a live `MFDatabase` schema via `tables=…` so the doc never drifts from
what exists. `write_schema_reference(path, …)` emits the file. Covered by
`test/fio/test_docs_generator.py` (live-table coverage, no-drift/no-extras when
restricted, default-namespace scope, determinism).

**Task 3 (admin entity registry) — largely already satisfied.** `mfdb_admin`'s
`entity_schema.py` already derives the per-entity `FieldSpec` (columns/types/enums)
from the `.dic`; `entity_registry.py` is intentionally *wiring-only* (rpc namespace,
title, group), which is genuine UI metadata not derivable from the dictionary. No
generation needed beyond what exists.

**Task 4 (derive RPC/boundary parameter validation from the `.dic`) — landed.**
`chisurf/core/mfdb/boundary_validation.py` provides `DictionaryValidator`: built from
`MmcifDictionary` (per-table `ColumnRule`s carrying `type_code`/`mandatory`/
`enumerations`), it validates a boundary payload against the dictionary-declared
columns — rejecting unknown columns, missing mandatory ones, type-incoherent values,
and out-of-vocabulary values (`validate(table, values, *, require_mandatory,
allow_unknown)`; no-op for undeclared tables; partial-update friendly). The value
side is one shared kernel, `check_value_type`, normalising the mmCIF `type_code`s
(and PRD-11 `value_type`s) to `int`/`float`/`bool`/`date`/`str` and accepting
stringly-typed payloads. The PRD-11 operation-parameter validation now reuses that
kernel: `validate_operation_parameters` additionally enforces declared `value_type`
and numeric `lower_bound`/`upper_bound` (unwrapping rich `{value,…}` dicts and
role-indexed lists), so the wired boundary in `result_registry` is now
type/bound-checked, not just name-checked. Covered by
`test/fio/test_boundary_validation.py` (unknown/missing/type/enum rejection,
partial + allow-unknown, unknown-table no-op, the shared `check_value_type` kernel,
and op-param out-of-bounds/wrong-type/valid-rich-repeatable).

Remaining: continue **Task 2** (harder CRUD families — inserts; `get_sample` shape;
`delete_parameter` dual-key). Tasks 1, 3, 4, 5 are landed.

## Relationship

Builds directly on **PRD-19** (canonical schema + reconcile generator) and **PRD-11**
(operation parameter schemas). Subsumes most of the hand-maintained admin/validation
work in PRD-02b and the per-plugin parameter handling. Do after PRD-19, alongside or
just after PRD-11.
