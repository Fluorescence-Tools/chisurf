# MFDB Dictionary Programming Rules

## Dictionary Authority

For PRD-02/flrCIF/PDBx fields, the bundled dictionary files are the source of
truth. Python code may reflect database tables and call repository helpers, but
must not become the authority for dictionary item names, schema aliases, enums,
defaults, descriptions, or mandatory flags.

When upstream flrCIF is incomplete or still changing, add ChiSurf/MFDB-specific
definitions to `chisurf/core/mfdb/data/mfdb_flr_ext.dic`. Do this before adding
Python code that persists or validates the field.

Every persisted dictionary item should declare:

- `_item.name`
- `_item.category_id`
- `_item_type.code`
- `_item.mandatory_code`
- `_item_enumeration.value` for controlled values
- `_item_default.value` for defaulted values
- `_mfdb_schema.table_name`
- `_mfdb_schema.column_name`

## Idempotent Identity Rows

Rows that represent identity-like objects must be looked up by their natural key
before insert. Reuse existing IDs rather than creating duplicate rows.

Examples:

- Chemical descriptors reuse normalized `(descriptor_type, descriptor, program,
  program_version)`.
- Probe/type vocabulary rows reuse their declared names.
- Any future dictionary-backed identity table must define the natural key before
  repository code inserts rows.

Prefer a database uniqueness constraint when the schema can safely enforce the
natural key. When legacy schema constraints make that impractical, repository
code must provide an idempotent upsert and tests must prove repeated writes reuse
the same row.

## Review Gate

A PRD-02/flrCIF change is not complete until tests prove that dictionary items
map to live database columns, dictionary enums/defaults drive code behavior, and
identity rows are not duplicated by repeated writes.
