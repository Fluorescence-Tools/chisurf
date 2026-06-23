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

**Deferred (need care, not behaviour-equivalent yet):**
- `get_sample` returns a `sqlite3.Row` with an explicit column subset (not `SELECT *`/
  dict) — migrating changes the return type/shape; audit callers first.
- The `delete_*` soft-delete methods set `deleted_at` via `_utc_now()` (Python ISO
  string), whereas `DictionaryDao.soft_delete` uses `CURRENT_TIMESTAMP`; some cascade
  across related tables. Align the DAO's soft-delete timestamp/cascade story before
  migrating these.

Remaining: continue **Task 2** (CRUD-shaped methods, each with an equivalence check;
keep bespoke lineage/browse SQL explicit); **Tasks 3–5** generate the admin entity
registry, RPC parameter validation, and API/schema docs from the `.dic`.

## Relationship

Builds directly on **PRD-19** (canonical schema + reconcile generator) and **PRD-11**
(operation parameter schemas). Subsumes most of the hand-maintained admin/validation
work in PRD-02b and the per-plugin parameter handling. Do after PRD-19, alongside or
just after PRD-11.
