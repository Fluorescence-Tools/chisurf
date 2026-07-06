---
type: PRD
prd: "02"
title: "PRD-02: Sample Tracking — Deep Sample Description"
description: Link every dataset and result to a full atomistic, flrCIF-aligned sample description
status: done
phase: "foundation"
resource: modules/mfdb/src/mfdb/
tags: [prd, mfdb, fret]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Every dataset and analysis result in MFDB is linked to a sample, where a sample
is a full atomistic description — the biomolecule and its sequence, labeling
positions, fluorescent probes and their photophysical properties, FRET pairs,
and buffer conditions — modeled as a graph across the flr_* tables and
exportable as valid flrCIF. It restructures `SampleDefinition` into typed
entity, probe, and FRET-pair sub-models (probes carry no intrinsic
donor/acceptor role; that is pair-relative), auto-populates spectra for known
dyes, and makes `create_sample` populate the full flrCIF data model plus a
lightweight index row. Supports 2/3/4-color and homo-FRET, single-label FCS, and
multi-chain complexes.

# Status
Done. A companion GUI verification gate ([PRD-02b](prd-02b.md)) exists so the
structured data can be inspected and confirmed before downstream wiring.
Vocabulary validation uses the `MmcifDictionary` API from [PRD-02a](prd-02a.md)
to check field values against the parsed `.dic` files at runtime.

# Goal
Every dataset and analysis result in MFDB is linked to a **sample** — not just a
name, but a full atomistic description of what was measured: the biomolecule,
its sequence, the labeling positions, the fluorescent probes, their
photophysical properties, and the buffer conditions. The description must use
PDBx/pdbihm/flrCIF vocabulary wherever a standard category exists, and must be
exportable as a valid flrCIF file.

**Why this matters:** without proper sample descriptions, data is an orphan. An
smFRET distance measurement is meaningless without knowing which protein, which
mutant, which dyes, at which positions, in what buffer. The flrCIF standard
(doi:10.1038/s41592-021-01145-3) was designed for exactly this, and ChiSurf
already has the schema tables — they just need to be wired into sample creation.

# What a sample really is (flrCIF data model)
A sample in smFRET is a **graph** of related entities, not a flat record. For
example a "T4L-heterodimer-3color" sample:
- **Entities** (`entities` → `entity_poly_seq`): e.g. Entity 0 "T4 Lysozyme"
  (protein, 164 residues), Entity 1 "DNA ruler".
- **Entity assembly** (`flr_entity_assembly`, `struct_asym`): Chain A → Entity 0,
  Chain B → Entity 1.
- **Probes** with positions (`flr_poly_probe_position`) and properties
  (`optical_properties`, `spectra`): e.g. Cy3B at entity 0 / chain A / residue 48
  / atom CB (mutation S48C), ATTO647N at residue 131, AF488 on the DNA at
  residue 5.
- **FRET pairs** (`flr_fret_forster_radius`): pair 0 probe 0→1 R₀=5.1 nm, pair 1
  probe 2→0 R₀=5.0 nm.
- **Condition** (`flr_sample_condition`): PBS pH 7.4, 150 mM NaCl, 25 °C.
- **Default spectra**: auto-populated from a built-in library when not measured.

Key position fields (from flrCIF `flr_poly_probe_position`): `entity_id`,
`asym_id` (chain), `seq_id` (residue number), `comp_id` (residue name),
`atom_id` (attachment atom), `mutation_flag`, `modification_flag`, `auth_name`.
This is stored across 8+ tables (entities, entity_poly_seq, probes,
optical_properties, flr_poly_probe_position, flr_sample_probe,
flr_sample_condition, spectra) plus `flr_sample` and `flr_fret_forster_radius`.

**Two sample tables — which is canonical?** `flr_sample` (rich columns, used by
`export_flr_cif`, `importer.py`, `seed_example.py`, and repository CRUD) is the
authoritative table. `mfdb_sample` (minimal: display_name, sample_type,
metadata_json) is a lightweight index for quick lookups and display.
`sample_manager.py` must populate **both** plus the related flr_* tables.

# Tasks

## Task 1: Restructure `SampleDefinition` with proper typing (`models.py`)
- Replace sentinel defaults with `Optional`: `donor_position`, `acceptor_position`
  → `Optional[int] = None`; `ph`, `temperature_k`, `salt_concentration_m` →
  `Optional[float] = None`.
- Add `EntityDefinition` (`name`, `entity_type` ∈ `ENTITY_TYPES`, `sequence`) →
  maps to `entities` + `entity_poly_seq`. `SampleDefinition.entities` becomes a
  `list[EntityDefinition]`.
- Add `ProbeDefinition` with the full flrCIF position model (`entity_index`,
  `seq_id`, `comp_id`, `atom_id`, `asym_id`, `mutation_flag`,
  `modification_flag`, `auth_name`), photophysical scalars (abs/em peak λ,
  quantum yield, extinction coefficient), full absorption/emission spectra
  arrays (source of truth for spectral overlap), and chemical descriptors
  (SMILES/InChI for chromophore, reactive form, and linker; `probe_origin`,
  `probe_link_type`, `chromophore_center_atom`). **No donor/acceptor label** —
  role is pair-relative. `__post_init__` auto-populates missing fields from
  `DEFAULT_FLUOROPHORE_SPECTRA` (a bundled JSON library keyed by dye name);
  experimental values always override defaults (per-field `None` check).
- Add `FretPairDefinition` (`probe_1_index` = donor role, `probe_2_index` =
  acceptor role, `forster_radius_nm`, `reduced_forster_radius_nm`,
  `kappa_squared` default 0.666667, `refractive_index` default 1.4,
  `overlap_integral`). R₀ can be computed from spectral overlap J(λ) via
  `compute_forster_radius(...)` when both probes carry full spectra.
- `SampleDefinition.probes` and `.fret_pairs` become lists. Legacy flat
  `donor`/`acceptor`/`entity_name` fields are removed but accepted for backward
  compat and converted internally (`donor_probe_name` → `probes[0]`, etc.).
- `__post_init__` validation: `entity_type` in `ENTITY_TYPES`; each
  `probe.entity_index < len(entities)`; each FRET-pair probe index
  `< len(probes)`.

**Probe → DB mapping:** name → `probes.chromophore_name`; abs/em scalars →
`optical_properties`; spectra → `spectra` (spectrum_type absorption/emission);
SMILES/InChI → `ihm_chemical_component_descriptor` (FK from probes); position
fields → `flr_poly_probe_position`. The `fluorophore_type` on `flr_sample_probe`
is **derived from FRET-pair context**, not set on the probe: probe only as
`probe_1` → "donor", only as `probe_2` → "acceptor", both (relay dye) or no
pairs → "unspecified".

**Supported configurations:** standard 2-color (1 entity, 2 probes, 1 pair);
3-color (3 probes, pairs 0→1, 1→2); 4-color; homo-FRET (same dye at two
positions); multi-chain complex (protein + DNA, probes on different
`entity_index`/`asym_id`); homodimer (1 entity, chains A/B); FCS/single-label
(1 probe, 0 pairs). Chain + entity is required because in a protein-DNA complex
residue 5 on the protein and residue 5 on the DNA are different positions; SMILES
disambiguates dye chemistry (maleimide vs. NHS vs. azide under one dye name).

## Task 2: `create_sample()` populates the flr_* tables (`sample_manager.py`)
Create all entities (+ `entity_poly_seq` rows), probes (via `find_or_add_probe`,
warn on unknown dye names), optical properties, spectra, probe positions
(resolving `entity_index` → `entity_id`, with all flrCIF position fields), the
`flr_sample` record, `flr_sample_probe` mappings (deriving `fluorophore_type`),
the `flr_fret_forster_radius` rows, the `mfdb_sample` index, and the entity
assembly. **API cheat sheet (avoid repeat bugs from PRD v1):** connection is
`db.conn` (not `db.con`); use `with db._transaction():` (not `db.conn.commit()`);
`MFDatabase(db_path)` takes no `object_store_root`; edge columns are
`source_node_id`/`target_node_id`; row access is `row["col"]`/`dict(row)`; use
`db.add_edge()` for `mfdb_edge` inserts.

## Task 3: Vocabulary validation on sample creation
`entity_type` → hard `ValueError` on mismatch; probe names → `logging.warning`
(custom dyes are valid); buffer components freeform; `ph` 0–14, `temperature_k`
> 0, `quantum_yield` 0–1 if set → `ValueError` otherwise.

## Task 4: `get_sample_full_description(db, sample_id)`
Return the complete structured graph by joining `flr_sample` → condition →
entity assembly → entities → `entity_poly_seq` → `flr_sample_probe` → probes →
positions → optical properties → `flr_fret_forster_radius`. Result contains
`entities` (list), `condition`, `probes` (each with position, properties,
`has_default_spectra`, derived `fluorophore_type`), `fret_pairs`, and
`key_values`.

## Task 5: PDBx key-value metadata (`flr_sample_key_value`)
`set_sample_metadata(db, sample_id, key, value, details=None)` for
`category.attribute` keys (validated against the dictionary when available);
auto-populate `flr.solvent_phase`, `flr.num_of_probes`, `pdbx.entity_type`,
`chisurf.sample_origin` during `create_sample`; `suggest_pdbx_keys(prefix)` for
GUI autocomplete.

## Task 6: Seed `mfdb_vocabulary` (`schema.py` `bootstrap_vocabulary`)
Seed `entity_type`, `fluorophore_type` (donor/acceptor/unspecified),
`solvent_phase`, and `sample_type` — matching the constants in `models.py`
(single source of truth).

## Task 7: `SamplePicker` dialog (`chisurf/gui/widgets/sample_picker.py`)
Group fields into Entities / Probes / FRET pairs / Condition tables with add/
remove buttons and no probe-count limit; editable probe-name combo populated
from `COMMON_PROBE_NAMES` that auto-fills photophysical values from
`DEFAULT_FLUOROPHORE_SPECTRA` (with a "(default)" indicator); entity combo per
probe row; a completeness indicator showing which flrCIF categories are filled.

## Task 8: Export validation
`validate_sample_for_export(db, sample_id)` returns a list of warnings (empty =
export-ready). Required for minimal flrCIF: ≥1 entity, ≥2 probes with positions,
sample-probe mappings with `fluorophore_type`, a condition (pH + temperature).
Recommended: entity sequence, optical properties, Förster radius.

## Task 9: Link datasets to samples in the project archiver
Already implemented: `_archive_datasets()` calls `link_artifact_to_sample()`
when a dataset carries `sample_id`. Verify it works with the restructured
`SampleDefinition`.

## Task 10: Tests (`test/fio/test_sample_manager.py`)
Cover flr_* population, entity-type validation, unknown-probe warning, 3-color,
homo-FRET, single-probe/FCS, 4-color, multi-entity protein-DNA, homodimer
(same residue on chains A/B), default-spectra auto-population, default-spectra
non-override, unknown-dye no-default, entity-index out-of-range, export
validation (complete + incomplete), flrCIF round-trip (create → export →
import → compare), full-description structure, and pH None-vs-0.0 handling.

# Definition of Done (abridged)
Data model: `EntityDefinition`; `entities`/`probes`/`fret_pairs` as lists;
`Optional` instead of sentinels; full flrCIF `ProbeDefinition` position model with
photophysics + spectra + SMILES/InChI and no donor/acceptor label;
`FretPairDefinition`; index validation. Default spectra: bundled JSON,
auto-populate known dyes, experimental overrides, unknown dyes stay `None`.
Database: `fluorophore_type` derived from pairs; `compute_forster_radius` from
spectral overlap; `create_sample` populates all flr_* tables plus `mfdb_sample`;
vocabulary validation; `get_sample_full_description`; `validate_sample_for_export`;
PDBx key-value support; seeded `mfdb_vocabulary`; N-entity/N-probe/N-pair
`SamplePicker`; `pH=None` stored NULL and `pH=0.0` preserved. Tests: multi-entity,
homodimer, 3-color, homo-FRET, single-probe, default-spectra, entity-index
validation, flrCIF round-trip, and all existing tests pass.

# Implementation notes (as built)
Data was separated from code: vocabularies live in JSON under
`modules/mfdb/src/mfdb/data/` (`entity_types.json`, `probe_names.json`,
`buffer_components.json`, `sample_condition_fields.json`,
`default_fluorophore_spectra.json`, `probe_properties.json`) with a
`vocabulary_loader.py` module (`get_entity_types`, `get_probe_names`, …,
`reload_vocabulary`) and fallback values in `models.py`. Sentinel values became
`Optional`; a `validate_vocabulary` flag (default off for backward compat) gates
entity/probe checks. Canonical input objects were added in `sample_requests.py`
(`SampleCreateRequest`, `SampleUpdateRequest`, `SampleQueryRequest`,
`SampleLinkRequest`, `SampleUnlinkRequest`, `SampleSearchRequest`), following the
Request pattern used by other plugins, with `to_sample_definition()`. Exports for
the new constants, loader functions, sample-manager functions, and Request
classes were added to the package `__init__.py`.

# Review conclusion
PRD-02 was hardened across a long, multi-round code review (documented history:
31 rounds spanning Phase 1 round-trip through the full PRD-02 series). Early
rounds caught PRD-pseudocode bugs (`db.con` vs `db.conn`, wrong `mfdb_edge`
column names, manual commits, sentinel-value pitfalls) which the implementation
avoided, then genuine implementation bugs: FRET pairs were initially globally
scoped by probe IDs (cross-sample leakage and UNIQUE-constraint failures) and
were re-scoped to the sample (see [PRD-020](prd-020.md) R14-1 correction);
`get_sample_full_description` join/column errors; a relay-dye
`fluorophore_type` ordering bug (made order-independent); a regression where a
`probes`-only sample without explicit `entities` was rejected (fixed by
auto-creating a default entity); metadata-index serialization that dropped
new-format fields; and request-layer classes that hard-rejected custom dye names
(changed to warn). The final verdict across the PRD-02 series was **APPROVE**:
the sample layer moved from unvalidated JSON blobs to strict, flrCIF-aligned
entity-probe-condition graphs. Focused suites pass in isolation
(`test/fio/test_sample_manager.py` 18/18); a couple of GUI integration tasks
(the Sample Picker widget and its dataset-import integration) were deferred as
non-blocking for the data layer.

# Relationships
- Depends on [PRD-020](prd-020.md) (ORM boundary) and [PRD-02a](prd-02a.md) (dictionary validation).
- GUI counterpart / verification gate: [PRD-02b](prd-02b.md); export alignment: [PRD-02c](prd-02c.md).
- Populates sample tables in the [MFDB (current)](/architecture/mfdb.md) store toward the [MFDB target](/specs/mfdb.md).
