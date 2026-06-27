# PRD-39: Sequence Provenance & External References (UniProt / PDB)

**Depends on:** [PRD-02: Sample Tracking](PRD-02-sample-tracking.md), [PRD-02a: mmCIF Dictionary Infrastructure](PRD-02a-mmcif-dictionary-infrastructure.md), [PRD-02c: flrCIF Alignment](PRD-02c-flrcif-alignment.md)

## Goal

Every protein (or nucleic-acid) entity in a sample should be traceable to its
source database record — **UniProt** for the canonical sequence and organism,
**PDB** for the structure — and every engineered mutation (above all the
**cysteine substitutions** used as dye-attachment points, e.g. S48C, S131C)
should be recorded as **structured, exportable flrCIF/PDBx data**, not buried in a
free-text note or implied by a per-probe flag.

This uses the **standard PDBx `struct_ref` category family** (already present in
the bundled dictionaries) wherever it exists, so a sample exports as valid flrCIF
and round-trips through `export_flr_cif` / `import_structure_file`.

### Why this matters

PRD-02 (DONE) built the sample graph: entities, `entity_poly_seq` sequences,
probes, and `flr_poly_probe_position` rows carrying a `mutation_flag` and an
`auth_name` ("S48C") **per probe position**. But there is currently **no way to
document the construct itself** — which UniProt entry it derives from, which PDB
structure it corresponds to, the complete mutation list, or whether the stored
`sequence` is wild-type or the labeled mutant.

Concretely, today:

- `EntityDefinition` (`chisurf/core/mfdb/models.py`) has only `name`,
  `entity_type`, `sequence`, `details` — **no `uniprot_accession`, `pdb_id`,
  `organism`**.
- There are **no `struct_ref` / `struct_ref_seq` / `struct_ref_seq_dif` tables**
  in `chisurf/core/mfdb/schema.py`, and no UniProt/SIFTS code anywhere in
  `chisurf/core/mfdb/`.

A smFRET sample is meaningless without "what protein is this, what was mutated and
why, and where do the canonical sequence/structure live." This PRD answers that.

## Background — what already exists

**Read starting:**

| File | What to look at |
|------|-----------------|
| `chisurf/core/mfdb/data/mmcif_std.dic`, `mmcif_pdbx_v50.dic` | `struct_ref`, `struct_ref_seq`, `struct_ref_seq_dif` categories — **the standard mechanism for this exact problem** (see below). |
| `chisurf/core/mfdb/models.py` | `EntityDefinition`, `ProbeDefinition` (`seq_id`, `comp_id`, `mutation_flag`, `auth_name`), `ENTITY_TYPES`. |
| `chisurf/core/mfdb/schema.py` | `entities`, `entity_poly_seq`, `flr_poly_probe_position`; soft-delete + audit column conventions to mirror. |
| `chisurf/core/mfdb/sample_manager.py` | `create_sample` (`:31`), `get_sample_full_description` (`:1199`), `set_sample_metadata`, `validate_sample_for_export`. |
| `chisurf/core/mfdb/repository.py` | `export_flr_cif` (`:2505`), `export_flr_cif_to_text` (`:2918`). |
| `chisurf/core/mfdb/importer.py` | `import_structure_file` (`:14`). |
| `chisurf/core/mfdb/orm/models.py` | reflected ORM classes; new tables need reflection here. |
| `chisurf/core/fio/structure/coordinates.py` | `fetch_pdb` (`:247`), `fetch_pdb_string` (`:233`) — reuse the network/error style for downloads. |

### The standard categories are already in our dictionaries

`grep` over `mmcif_std.dic` / `mmcif_pdbx_v50.dic` confirms the full family ships:

- **`struct_ref`** — the cross-reference. Key items: `id`, `entity_id`,
  `db_name` (`UNP` = UniProt, `PDB`), `db_code`, `pdbx_db_accession`,
  `pdbx_db_isoform`.
- **`struct_ref_seq`** — alignment of the construct sequence against the reference
  DB sequence. Items: `align_id`, `ref_id`, `seq_align_beg`/`seq_align_end`
  (construct numbering), `db_align_beg`/`db_align_end` (reference numbering),
  `pdbx_db_accession`, `pdbx_seq_align_beg`.
- **`struct_ref_seq_dif`** — **per-residue differences**, the natural home for
  engineered mutations. Items: `align_id`, `seq_num`, `mon_id` (construct residue,
  e.g. `CYS`), `db_mon_id` (reference residue, e.g. `SER`),
  `details` (e.g. `"ENGINEERED MUTATION"`), `pdbx_seq_db_name`,
  `pdbx_seq_db_accession_code`, `pdbx_ordinal`.

**Decision (LOCKED):** reuse these standard categories; do **not** invent custom
columns for cross-references or mutations.

### Worked example — T4 Lysozyme, Cy3B/ATTO647N

```
Entity: "T4 Lysozyme" (protein, construct sequence as measured)
├── struct_ref (UNP)
│   ├── db_name=UNP, db_code=LYS_BPT4, pdbx_db_accession=P00720
│   └── struct_ref_seq: construct 1..164 ↔ UniProt 1..164
│       ├── struct_ref_seq_dif: seq_num=48  mon_id=CYS db_mon_id=SER
│       │     details="ENGINEERED MUTATION"   (S48C — Cy3B attachment)
│       └── struct_ref_seq_dif: seq_num=131 mon_id=CYS db_mon_id=SER
│             details="ENGINEERED MUTATION"   (S131C — ATTO647N attachment)
├── struct_ref (PDB)
│   └── db_name=PDB, pdbx_db_accession=2LZM   (structure reference, chain A)
└── (PRD-02) flr_poly_probe_position
    ├── seq_id=48,  comp_id=CYS, mutation_flag=yes, auth_name="S48C"   ← Cy3B
    └── seq_id=131, comp_id=CYS, mutation_flag=yes, auth_name="S131C"  ← ATTO647N
```

The PRD-02 probe positions and the new `struct_ref_seq_dif` rows describe the same
cysteines from two angles; Design §4 keeps them consistent.

## Design

### 1. Reuse standard PDBx categories — no custom invention

Map all cross-references and mutations onto `struct_ref` / `struct_ref_seq` /
`struct_ref_seq_dif`. Custom `.dic` extension is **out of scope**: every field we
need already exists in the standard categories above.

### 2. New schema tables

**File:** `chisurf/core/mfdb/schema.py`

Add `struct_ref`, `struct_ref_seq`, `struct_ref_seq_dif`, keyed by `entity_id`
(FK → `entities`), each with the same soft-delete (`deleted_at`) and audit columns
as sibling flrCIF tables. Then reflect them in
`chisurf/core/mfdb/orm/models.py` alongside `Entity` / `EntityPolySeq`, with the
entity → struct_ref → struct_ref_seq → struct_ref_seq_dif relationship chain.

### 3. Dataclass extensions

**File:** `chisurf/core/mfdb/models.py`

Extend `EntityDefinition` (additive, all optional — backward compatible):

```python
@dataclasses.dataclass
class EntityDefinition:
    name: str = ""
    entity_type: str = ""
    sequence: str = ""                    # construct (as-measured) sequence
    details: str = ""
    # ── external references ──
    uniprot_accession: Optional[str] = None   # e.g. "P00720"  → struct_ref UNP
    pdb_id: Optional[str] = None              # e.g. "2LZM"    → struct_ref PDB
    pdb_chain_id: Optional[str] = None        # author chain in the PDB entry
    organism: Optional[str] = None            # from UniProt
    reference_sequence: Optional[str] = None  # canonical/WT seq from UniProt
    mutations: list["MutationDefinition"] = dataclasses.field(default_factory=list)
```

New `MutationDefinition`:

```python
@dataclasses.dataclass
class MutationDefinition:
    """One difference between the construct and its reference DB sequence.

    Maps to one ``struct_ref_seq_dif`` row. The common case is a cysteine
    substitution introduced for dye labeling (S48C → CYS replaces SER).
    """
    seq_id: int                       # residue number (construct numbering)
    mut_comp_id: str                  # construct residue, e.g. "CYS"  (mon_id)
    wt_comp_id: str = ""              # reference residue, e.g. "SER"   (db_mon_id)
    auth_name: str = ""              # author label, e.g. "S48C"
    kind: str = "engineered_mutation" # → struct_ref_seq_dif.details
    rationale: str = ""              # e.g. "cysteine labeling"
```

`sequence` stays the construct (as-measured) sequence; `reference_sequence` holds
the canonical UniProt sequence. `kind` vocabulary: `engineered_mutation`
(default), `conflict`, `insertion`, `deletion`, `variant`.

### 4. Consistency with PRD-02 probe positions

A labeled-cysteine probe position
(`flr_poly_probe_position.mutation_flag="yes"`) MUST correspond to a
`struct_ref_seq_dif`/`MutationDefinition` on the same entity at the same `seq_id`.
Extend `validate_sample_for_export` (`sample_manager.py`) to **warn** (not reject)
when:

- a probe sits on `mutation_flag="yes"` with no matching mutation record, or
- a mutation's `mut_comp_id` ≠ the probe's `comp_id` at that `seq_id`, or
- `auth_name` disagrees between the two records.

### 5. UniProt fetch service

**New file:** `chisurf/core/mfdb/external_refs.py`

Functions to fetch the canonical sequence + organism + entry name from the UniProt
REST API:

```python
def fetch_uniprot(accession: str, *, cache_dir: Path | None = None) -> dict | None:
    """Fetch sequence + organism + entry_name from UniProt REST.

    GET https://rest.uniprot.org/uniprotkb/{accession}.json (and .fasta).
    Result cached on disk under the mfdb data dir. Returns None on network
    failure so callers degrade to manual entry (offline/headless-safe).
    """
```

Mirror the network/error-handling style of `fetch_pdb_string` / `fetch_pdb`
(`chisurf/core/fio/structure/coordinates.py`), and reuse those for PDB downloads.
All fetches are optional; failures never block sample creation.

### 6. PDB ↔ UniProt mapping (SIFTS)

For `pdb_id` + `pdb_chain_id` → UniProt accession + residue offset, query the EBI
SIFTS endpoint:

```
GET https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}
```

Use the returned chain mapping to set `struct_ref_seq` alignment offsets, so author
/ PDB numbering (S48C) maps correctly into UniProt numbering. Cached like §5.

### 7. Auto-diff algorithm

Given the construct `sequence` and the fetched `reference_sequence`, align and emit
one `struct_ref_seq_dif` (and `MutationDefinition`) per mismatch, defaulting
`details="ENGINEERED MUTATION"`. Auto-detected mutations populate
`EntityDefinition.mutations`; **user edits always win** (re-running the diff never
clobbers manually entered records). Note the alignment dependency choice
(Biopython/`parasail` global alignment vs. a small vendored aligner) as an open
implementation decision; a simple Needleman–Wunsch is sufficient for
single-residue substitutions.

### 8. create_sample / full-description / flrCIF wiring

- `create_sample` (`sample_manager.py:31`): persist `struct_ref`,
  `struct_ref_seq`, `struct_ref_seq_dif` for each entity.
- `get_sample_full_description` (`sample_manager.py:1199`): add `external_refs`
  (UniProt/PDB accessions, organism) and `mutations` to each entity in the
  returned graph.
- `export_flr_cif` (`repository.py:2505`) + `import_structure_file`
  (`importer.py:14`): round-trip the three `struct_ref*` categories.

### 9. GUI (headless-first)

Everything above is usable through the headless API/CLI first (project rule: every
feature needs a non-GUI path). Then extend `SamplePicker`: per-entity UniProt/PDB
fields with a **Fetch** button, a mutations table, and a **"diff vs. reference"**
action that previews detected mutations before they are written.

## Tasks

1. **Schema + ORM** — add `struct_ref` / `struct_ref_seq` / `struct_ref_seq_dif`
   to `schema.py`; reflect in `orm/models.py`. (§2)
2. **Dataclasses** — extend `EntityDefinition`; add `MutationDefinition`. (§3)
3. **UniProt fetch** — `external_refs.py::fetch_uniprot` + disk cache. (§5)
4. **SIFTS mapping** — PDB↔UniProt chain/offset resolver. (§6)
5. **Auto-diff** — construct-vs-reference alignment → `struct_ref_seq_dif`. (§7)
6. **Consistency validator** — extend `validate_sample_for_export`. (§4)
7. **Persistence + full description** — wire `create_sample` /
   `get_sample_full_description`. (§8)
8. **flrCIF round-trip** — export/import the `struct_ref*` categories. (§8)
9. **GUI** — `SamplePicker` fields, Fetch button, diff preview. (§9)

## Definition Done

**Data model:**
- [x] `struct_ref`, `struct_ref_seq`, `struct_ref_seq_dif` tables exist + reflected in ORM
- [x] `EntityDefinition` has `uniprot_accession`, `pdb_id`, `pdb_chain_id`, `organism`, `reference_sequence`, `mutations` (all optional; backward compatible)
- [x] `MutationDefinition` dataclass maps 1:1 to a `struct_ref_seq_dif` row

**Behavior:**
- [x] `fetch_uniprot` returns sequence + organism, caches on disk, returns `None` (not raises) offline
- [x] SIFTS mapping resolves PDB chain → UniProt accession + offset
- [x] Auto-diff emits `struct_ref_seq_dif` rows for substitutions; cysteine mutations get `details="ENGINEERED MUTATION"`; manual edits preserved on re-run
- [x] `validate_sample_for_export` warns on probe↔mutation inconsistencies
- [x] `create_sample` persists the new tables; `get_sample_full_description` surfaces `external_refs` + `mutations`
- [x] flrCIF export/import round-trips `struct_ref*`
- [x] **Headless path:** a sample with UniProt/PDB refs + mutations is fully creatable and inspectable via the API/CLI with no GUI

**GUI:**
- [x] `SamplePicker` per-entity UniProt/PDB fields + Fetch button + mutations table + diff preview

## Definition Clean

- [x] Tests (run in the `arm64` conda env): UniProt/SIFTS fetch tested offline via
      primed disk cache (`tmp_path/uniprot_*.json`, `sifts_*.json`) + empty-input
      `None` — equivalent to HTTP mocking, no network; auto-diff on known WT→mutant
      pairs **incl. T4L S48C/S131C** (`test_diff_detects_engineered_cysteines`);
      flrCIF round-trip (`test_flr_cif_round_trip_preserves_struct_ref`);
      probe↔mutation consistency validator. **39 pass** (`test_external_refs.py` 18
      + `test_sample_manager.py` 21), verified 2026-06-27.
- [x] All network access optional + cached; no test hits the live network
      (`fetch_uniprot`/`fetch_sifts_uniprot_mapping` read the cache first via
      `_fetch_json`; tests prime the cache so `urllib` is never invoked).
- [x] No custom `.dic` extension introduced — standard `struct_ref` /
      `struct_ref_seq` / `struct_ref_seq_dif` categories only (0 `struct_ref`
      occurrences in `mfdb_flr_ext.dic`).

## Implementation status

**All tasks complete (2026-06-27):**
- [x] Task 1 — `struct_ref` / `struct_ref_seq` / `struct_ref_seq_dif` tables added
      to `chisurf/core/mfdb/schema.py` (`CREATE_TABLES_SQL`, after `entity_poly_seq`).
- [x] Task 2 — `EntityDefinition` extended (`uniprot_accession`, `pdb_id`,
      `pdb_chain_id`, `organism`, `reference_sequence`, `mutations`) + new
      `MutationDefinition` dataclass in `chisurf/core/mfdb/models.py`.
- [x] Task 3 — `fetch_uniprot()` in `chisurf/core/mfdb/external_refs.py`,
      offline-safe with disk cache.
- [x] Task 4 — SIFTS PDB↔UniProt chain/offset mapping
      (`fetch_sifts_uniprot_mapping` in `external_refs.py`, offline-safe, shares
      the `_fetch_json` cache helper with `fetch_uniprot`).
- [x] Task 5 — `diff_sequences()` auto-diff wired into `_persist_external_refs`.
      Auto-fills `struct_ref_seq_dif` when `reference_sequence` is set and
      `mutations` is empty; manual mutations preserved. Tested in
      `test_auto_diff_wired_in_create_sample`.
- [x] Task 6 — `validate_sample_for_export` warns on probe↔mutation inconsistencies.
- [x] Task 7 — `create_sample` persists the tables (`_persist_external_refs` in
      `orm/sample_repository.py`) and `get_sample_full_description` surfaces them
      (`_attach_external_refs` carries `external_refs` + `mutations` per entity).
- [x] Task 8 — flrCIF round-trip: export emits `_struct_ref` / `_struct_ref_seq` /
      `_struct_ref_seq_dif` loops; importer parses them back via pdbx reader.
      Tested in `test_flr_cif_round_trip_preserves_struct_ref`.
- [x] Task 9 — `SamplePicker` GUI (`_SampleDefinitionDialog`) with per-entity
      UniProt/PDB fields, Fetch button, editable mutations table, and
      "Diff vs Reference" preview.
- [x] All tests green: `test/fio/test_external_refs.py` (18 tests) +
      `test/fio/test_sample_manager.py` (21 tests) = **39 pass, 0 fail**.

**IMPLEMENTED (2026-06-27):**

- [x] **Task 5 — auto-diff wiring into `create_sample`.** `_persist_external_refs`
  in `orm/sample_repository.py` now calls `diff_sequences` when `reference_sequence`
  is present but `mutations` is empty. Manual mutations are preserved (the auto-diff
  only fires when the list is empty). Tested in `test_auto_diff_wired_in_create_sample`.

- [x] **Task 8 — flrCIF round-trip for `struct_ref*`.** Export (`repository.py:2712`):
  emits `_struct_ref`, `_struct_ref_seq`, `_struct_ref_seq_dif` loops from the three
  tables. Import (`importer.py`): parses those categories back via the pdbx reader.
  Verified in `test_flr_cif_round_trip_preserves_struct_ref` which exports to CIF
  text and parses the struct_ref* categories back using `pdbx.reader.PdbxReader`.

- [x] **Task 9 — `SamplePicker` GUI.** Extended `_SampleDefinitionDialog` with:
  per-entity UniProt accession + PDB ID + chain ID fields; **Fetch** button
  (calls `fetch_uniprot`, populates organism + reference sequence); mutations
  table (QTableWidget, editable, add/clear); **"Diff vs Reference"** button
  (previews `diff_sequences` output before writing to the table).
  Headless path remains the canonical API; the GUI is thin wiring.

**Run tests with:** `conda activate arm64 && python -m pytest
test/fio/test_external_refs.py test/fio/test_sample_manager.py -q -p no:cacheprovider`
(the `INTERNALERROR: Can't combine branch coverage data` line at the end is a
coverage-merge artifact from running two files together, not a failure — the
`N passed` line is what matters).

## Relationship

- **Depends on** PRD-02 / PRD-02a / PRD-02c — consumes their entity/sequence/probe
  model and the dictionary infrastructure.
- **Adjacent to** PRD-06 (fluorophore database) and PRD-33
  (acquisition→MFDB registration) — samples those flows create would carry these
  external references and mutation records.
