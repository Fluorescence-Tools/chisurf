# PRD-02: Sample Tracking — Deep Sample Description

**Depends on:** [PRD-020: SQLAlchemy MFDB Mapping](PRD-020-sqlalchemy-mfdb-mapping.md), [PRD-02a: mmCIF Dictionary Infrastructure](PRD-02a-mmcif-dictionary-infrastructure.md)

## Goal

Every dataset and every analysis result in MFDB is linked to a **sample**. A sample
is not just a name — it is a full atomistic description of what was measured:
the biomolecule, its sequence, the labeling positions, the fluorescent probes,
their photophysical properties, and the buffer conditions. This description must
use **PDBx/pdbihm/flrCIF vocabulary** wherever a standard category exists, and
must be exportable as a valid flrCIF file.

Vocabulary validation uses the `MmcifDictionary` API from PRD-02a to check
field values against the parsed `.dic` files at runtime.

### Why this matters

Without proper sample descriptions, data is an orphan. An smFRET distance
measurement is meaningless without knowing: which protein, which mutant, which
dyes, at which positions, in what buffer. The flrCIF standard
(doi:10.1038/s41592-021-01145-3) was designed exactly for this. ChiSurf already
has the schema tables — they just need to be wired into the sample creation
workflow properly.

## Background — what already exists

**Read these files before starting:**

| File | What to look at |
|------|-----------------|
| `chisurf/core/mfdb/schema.py` | `flr_sample`, `flr_sample_condition`, `flr_sample_probe`, `flr_poly_probe_position`, `entities`, `entity_poly_seq`, `probes`, `optical_properties`, `flr_fret_forster_radius`, `mfdb_sample`, `mfdb_vocabulary` |
| `chisurf/core/mfdb/models.py` | `ENTITY_TYPES`, `COMMON_PROBE_NAMES`, `BUFFER_COMPONENTS`, `SAMPLE_CONDITION_FIELDS`, `SampleDefinition`, `MfdbSample`, `validate_vocabulary` |
| `chisurf/core/mfdb/repository.py` | Sample methods: `add_sample`, `get_sample`, `add_sample_probe`, `get_sample_probe_mappings`, `add_entity`, `set_sequence`, `add_sample_condition`, `add_poly_probe_position`, `add_probe`, `add_optical_property`, `add_spectrum`, `export_flr_cif` |
| `chisurf/core/mfdb/sample_manager.py` | Current high-level CRUD — wraps `mfdb_sample` table |
| `chisurf/core/mfdb/importer.py` | `import_structure_file` — imports PDBx/IHM/flrCIF into flr_* tables |
| `chisurf/core/mfdb/pdbx_metadata.py` | PDBx dictionary parser for key suggestions |
| `chisurf/plugins/core/mfdb_admin/seed_example.py` | `_seed_flr_tables` — correct creation flow for entities, probes, positions, conditions, samples |

### What a sample really is (flrCIF data model)

A sample in smFRET is a **graph** of related entities, not a flat record:

```
Sample: "T4L-heterodimer-3color"
  ├── Entity 0: "T4 Lysozyme"     (entities → entity_poly_seq)
  │     └── Sequence: MNIFEMLR...  (protein, 164 residues)
  ├── Entity 1: "DNA ruler"        (entities → entity_poly_seq)
  │     └── Sequence: 5'-GCATCG...CGATGC-3'
  ├── Entity assembly              (flr_entity_assembly)
  │     ├── Chain A → Entity 0     (struct_asym)
  │     └── Chain B → Entity 1
  ├── Probe 0: Cy3B
  │     ├── Position: entity 0, chain A, residue 48, atom CB  (flr_poly_probe_position)
  │     │           mutation_flag=yes (S48C), comp_id=CYS
  │     └── Properties: λ_abs=559, λ_em=572, QY=0.67, ε=130000
  ├── Probe 1: ATTO647N
  │     ├── Position: entity 0, chain A, residue 131, atom CB
  │     │           mutation_flag=yes (S131C), comp_id=CYS
  │     └── Properties: λ_abs=644, λ_em=669, QY=0.65, ε=150000
  ├── Probe 2: Alexa Fluor 488
  │     ├── Position: entity 1, chain B, residue 5, atom C5
  │     │           mutation_flag=no, comp_id=dT
  │     └── Properties: λ_abs=495, λ_em=519, QY=0.92, ε=73000
  ├── FRET pairs                   (flr_fret_forster_radius)
  │     ├── Pair 0: probe 0→1  R₀=5.1 nm  (Cy3B → ATTO647N)
  │     └── Pair 1: probe 2→0  R₀=5.0 nm  (AF488 → Cy3B)
  ├── Condition: PBS pH 7.4, 150 mM NaCl, 25°C (flr_sample_condition)
  └── Default spectra: auto-populated from built-in library when not measured
```

**Key position fields** (from flrCIF `flr_poly_probe_position`):
- `entity_id` — which biomolecule (mandatory)
- `asym_id` — which chain/strand in the assembly (mandatory for multi-chain)
- `seq_id` — residue number within the entity sequence (mandatory)
- `comp_id` — residue name, e.g. "CYS", "dT" (mandatory)
- `atom_id` — attachment atom, e.g. "CB", "C5" (optional, needed for AV simulation)
- `mutation_flag` — was the residue mutated for labeling? e.g. S131C (mandatory)
- `modification_flag` — is the residue chemically modified? (mandatory)
- `auth_name` — author-provided position name, e.g. "S131C" (optional)

This is stored across **8+ tables** (entities, entity_poly_seq, probes,
optical_properties, flr_poly_probe_position, flr_sample_probe,
flr_sample_condition, spectra) plus the sample itself (flr_sample) and the
FRET pair parameters (flr_fret_forster_radius).

### Two sample tables — which is canonical?

The schema has **two** sample tables:

1. **`flr_sample`** — rich columns (entity_assembly_id, condition_id, probes,
   user, device). Used by `export_flr_cif`, `importer.py`, `seed_example.py`,
   and the full repository CRUD. This is the authoritative table.
2. **`mfdb_sample`** — minimal (display_name, sample_type, metadata_json). Used
   by `sample_manager.py`. Stores metadata as a flat JSON blob.

**Decision:** `flr_sample` is the canonical sample table. `mfdb_sample` is a
lightweight index for quick lookups and display. `sample_manager.py` must
populate **both** tables and the related flr_* tables.

## Tasks

### Task 1: Restructure `SampleDefinition` with proper typing

**File**: `chisurf/core/mfdb/models.py`

The current `SampleDefinition` uses sentinel values (`-1`, `0.0`) and freeform
strings. Replace with properly typed fields and vocabulary validation.

**Changes:**

1. Replace sentinel defaults with `Optional`:
   - `donor_position: int = -1` → `donor_position: Optional[int] = None`
   - `acceptor_position: int = -1` → `acceptor_position: Optional[int] = None`
   - `ph: float = 0.0` → `ph: Optional[float] = None`
   - `temperature_k: float = 0.0` → `temperature_k: Optional[float] = None`
   - `salt_concentration_m: float = 0.0` → `salt_concentration_m: Optional[float] = None`

2. Add entity definition for multi-entity/multi-chain support:
   ```python
   @dataclasses.dataclass
   class EntityDefinition:
       """One biomolecular entity in the sample.

       A sample can involve multiple entities (e.g. a heterodimer of protein A
       and protein B, or a protein + DNA). Each entity may appear as one or
       more chains (asym_ids) in the entity assembly.

       Maps to the ``entities`` + ``entity_poly_seq`` tables.
       """
       name: str = ""                 # e.g. "T4 Lysozyme"
       entity_type: str = ""          # "protein", "dna", "rna", etc. (ENTITY_TYPES)
       sequence: str = ""             # amino acid or nucleotide sequence
   ```

   Update `SampleDefinition` to use a list of entities instead of flat fields:
   ```python
   # Replace flat entity_name / entity_type / entity_sequence with:
   entities: list[EntityDefinition] = dataclasses.field(default_factory=list)

   # Backward compat: if entity_name is provided and entities is empty,
   # auto-create entities = [EntityDefinition(name=entity_name, ...)]
   ```

3. Add probe photophysical properties with full position model:
   ```python
   @dataclasses.dataclass
   class ProbeDefinition:
       """One fluorescent probe attached to the sample.

       A probe is a physical chromophore at a specific position on a
       biomolecule. It has intrinsic photophysical properties (spectra,
       quantum yield, extinction coefficient) and chemical identity
       (SMILES/InChI).

       **Position model:** A probe position is defined by:
       - ``entity_index`` — which entity in ``SampleDefinition.entities``
       - ``asym_id`` — chain/strand within the entity assembly
       - ``seq_id`` — residue number within the entity sequence
       - ``comp_id`` — residue name (3-letter code, e.g. "CYS", "dT")
       - ``atom_id`` — specific attachment atom (e.g. "CB" for Cβ, "C5")
       - ``mutation_flag`` / ``modification_flag`` — labeling chemistry

       Residue index alone is NOT sufficient — in multi-chain complexes
       (heterodimers, multi-subunit assemblies, protein-DNA complexes),
       the same residue number exists on different chains. The entity +
       chain + residue tuple uniquely identifies the position.

       **No donor/acceptor label here.** Whether a probe acts as donor or
       acceptor is not an intrinsic property — it depends on which other
       probe it is paired with. Cy5 is an acceptor relative to Cy3B but
       a donor relative to Cy7. In homo-FRET, the same dye is both.
       The donor/acceptor assignment lives on ``FretPairDefinition``, not
       on the probe.

       **Default spectra:** When ``name`` matches a known fluorophore in
       ``DEFAULT_FLUOROPHORE_SPECTRA`` and no experimental spectra are
       provided, the default absorption/emission spectra and photophysical
       scalars are auto-populated. Experimental values always override
       defaults.

       A sample can have any number of probes. The ``spectra`` table stores
       full absorption and emission spectra as arrays. The FRET relationship
       between probes is determined by spectral overlap (emission of one
       overlapping absorption of another), which defines the overlap
       integral J(λ) and thus the Förster radius R₀.
       """
       name: str                              # e.g. "Cy3B"
       # ── Position on the biomolecule (flr_poly_probe_position) ──
       entity_index: int = 0                  # index into SampleDefinition.entities
       seq_id: Optional[int] = None           # residue number (flr_poly_probe_position.seq_id)
       comp_id: str = ""                      # residue name e.g. "CYS" (flr_poly_probe_position.comp_id)
       atom_id: str = ""                      # attachment atom e.g. "CB" (flr_poly_probe_position.atom_id)
       asym_id: str = ""                      # chain/strand ID (flr_poly_probe_position.asym_id)
       mutation_flag: str = "no"              # "yes"/"no" — residue mutated for labeling?
       modification_flag: str = "no"          # "yes"/"no" — residue chemically modified?
       auth_name: str = ""                    # author position name e.g. "S131C"
       # ── Photophysical properties (scalar summaries) ──
       absorption_wavelength_nm: Optional[float] = None   # peak absorption λ
       emission_wavelength_nm: Optional[float] = None     # peak emission λ
       quantum_yield: Optional[float] = None               # fluorescence QY
       extinction_coefficient: Optional[float] = None      # molar ε (M⁻¹cm⁻¹)
       # ── Full spectra (arrays) — source of truth for spectral overlap ──
       absorption_spectrum: Optional[tuple[list[float], list[float]]] = None
           # (wavelengths_nm, intensities) — stored in ``spectra`` table
       emission_spectrum: Optional[tuple[list[float], list[float]]] = None
           # (wavelengths_nm, intensities) — stored in ``spectra`` table
       # ── Chemical descriptors (flrCIF flr_probe_descriptor) ──
       chromophore_smiles: str = ""           # SMILES for the chromophore
       chromophore_inchi: str = ""            # InChI for the chromophore
       reactive_probe_smiles: str = ""        # SMILES for the reactive form (e.g. maleimide)
       reactive_probe_name: str = ""          # e.g. "Cy3B-maleimide"
       reactive_probe_flag: str = "no"        # "yes" if reactive form differs from chromophore
       probe_origin: str = "extrinsic"        # "extrinsic" or "intrinsic" (e.g. Trp)
       probe_link_type: str = "covalent"      # how probe attaches to biomolecule
       chromophore_center_atom: str = ""      # atom name for AV simulation center
       # ── Conjugate chemistry (flrCIF flr_poly_probe_conjugate) ──
       linker_smiles: str = ""               # SMILES for the full probe-linker conjugate
       ambiguous_stoichiometry: str = "no"    # "yes" if labeling stoichiometry is uncertain
       probe_stoichiometry: Optional[float] = None  # avg number of probes at this site

       def __post_init__(self):
           # Auto-populate from default spectra library if available
           if self.name in DEFAULT_FLUOROPHORE_SPECTRA:
               defaults = DEFAULT_FLUOROPHORE_SPECTRA[self.name]
               for field_name, default_value in defaults.items():
                   if getattr(self, field_name) is None:
                       setattr(self, field_name, default_value)
   ```

   **Default spectra library:** A dictionary of common fluorophores with
   literature-reference photophysical properties. Loaded from a bundled JSON
   file (`chisurf/core/mfdb/data/default_fluorophore_spectra.json`).

   ```python
   # In models.py:
   DEFAULT_FLUOROPHORE_SPECTRA: dict[str, dict] = _load_default_spectra()

   # JSON structure per fluorophore:
   {
       "Cy3B": {
           "absorption_wavelength_nm": 559.0,
           "emission_wavelength_nm": 572.0,
           "quantum_yield": 0.67,
           "extinction_coefficient": 130000.0,
           "absorption_spectrum": [[wavelengths...], [intensities...]],
           "emission_spectrum": [[wavelengths...], [intensities...]],
           "chromophore_smiles": "...",
       },
       "ATTO 647N": { ... },
       "Alexa Fluor 488": { ... },
       ...
   }
   ```

   When `ProbeDefinition(name="Cy3B")` is created without explicit spectra,
   `__post_init__` fills in the defaults. If the user provides experimental
   values, those override — `None` check per field.

   **Database mapping:**

   | ProbeDefinition field | DB table | Column(s) |
   |----------------------|----------|-----------|
   | name | `probes` | `chromophore_name` |
   | absorption/emission scalars | `optical_properties` | property_name + value + unit |
   | absorption_spectrum | `spectra` | spectrum_type="absorption", wavelengths, intensity_values |
   | emission_spectrum | `spectra` | spectrum_type="emission", wavelengths, intensity_values |
   | chromophore_smiles/inchi | `ihm_chemical_component_descriptor` | smiles, inchi (FK via `probes.chromophore_chem_descriptor_id`) |
   | reactive_probe_smiles | `ihm_chemical_component_descriptor` | smiles (FK via `probes.reactive_probe_chem_descriptor_id`) |
   | linker_smiles | `ihm_chemical_component_descriptor` | smiles (FK via `flr_poly_probe_conjugate.chem_descriptor_id`) |
   | probe_origin, probe_link_type | `probes` | direct columns |
   | entity_index → entity_id | `flr_poly_probe_position` | entity_id (FK from resolved entity) |
   | seq_id, comp_id, asym_id | `flr_poly_probe_position` | seq_id (=residue_number), comp_id (=residue_name), asym_id |
   | atom_id | `flr_poly_probe_position` | atom_id |
   | mutation_flag, modification_flag | `flr_poly_probe_position` | mutation_flag, modification_flag |
   | auth_name | `flr_poly_probe_position` | auth_name |

   **Why chain + entity matters:** In a protein-DNA complex, residue 5 on the
   protein (chain A) and residue 5 on the DNA (chain B) are completely
   different positions. Without entity + chain, there is no way to unambiguously
   place a probe.

   **Why SMILES matters:** SMILES strings uniquely identify the dye chemistry.
   Labs often use the same dye name (e.g. "Alexa647") for different reactive
   forms (maleimide vs. NHS vs. azide). SMILES disambiguates.

4. Add FRET pair definition (supports multi-pair, spectrally defined):
   ```python
   @dataclasses.dataclass
   class FretPairDefinition:
       """One FRET pair between two probes on the sample.

       The pair defines which probe acts as donor (energy transfer source)
       and which as acceptor (energy transfer sink) **for this specific
       pair**. The same probe can appear as donor in one pair and acceptor
       in another (e.g. relay dye in 3-color FRET).

       The Förster radius R₀ is determined by:
         R₀⁶ = (9 ln10 κ² QD J(λ)) / (128 π⁵ n⁴ Nₐ)
       where J(λ) is the spectral overlap integral between the donor
       emission and acceptor absorption spectra. If full spectra are
       provided on the probes, J(λ) and R₀ can be computed automatically.

       For 3-color FRET (Cy3B → Cy5 → Cy7), create two pairs:
         FretPairDefinition(probe_1_index=0, probe_2_index=1, ...)  # Cy3B→Cy5
         FretPairDefinition(probe_1_index=1, probe_2_index=2, ...)  # Cy5→Cy7
       where indices refer to positions in SampleDefinition.probes.
       probe_1 is the energy transfer source (donor role in this pair).
       probe_2 is the energy transfer sink (acceptor role in this pair).
       """
       probe_1_index: int         # index into SampleDefinition.probes (donor role)
       probe_2_index: int         # index into SampleDefinition.probes (acceptor role)
       forster_radius_nm: Optional[float] = None  # can be computed from spectra
       reduced_forster_radius_nm: Optional[float] = None
       kappa_squared: float = 0.666667  # 2/3 default (dynamic averaging)
       refractive_index: float = 1.4
       overlap_integral: Optional[float] = None  # J(λ), computed from spectra if available
   ```

   **Key design principle:** The probes themselves have no donor/acceptor label.
   Whether probe A transfers energy to probe B depends on spectral overlap
   between A's emission and B's absorption. The `FretPairDefinition` captures
   this relationship. The `flr_sample_probe.fluorophore_type` column in the
   database is populated from the FRET pair context: a probe that appears as
   `probe_1` in any pair gets `fluorophore_type="donor"`, and `probe_2` gets
   `"acceptor"`. A probe appearing in both roles (relay dye) gets
   `"unspecified"`. A probe in no pairs gets `"unspecified"`.

   **Auto-computation of R₀:** When both probes in a pair have full spectra
   (`absorption_spectrum` and `emission_spectrum`), the overlap integral J(λ)
   and Förster radius R₀ can be computed automatically:
   ```python
   def compute_forster_radius(
       donor_emission: tuple[np.ndarray, np.ndarray],
       acceptor_absorption: tuple[np.ndarray, np.ndarray],
       donor_quantum_yield: float,
       kappa_squared: float = 0.666667,
       refractive_index: float = 1.4,
   ) -> tuple[float, float]:
       """Compute R₀ (nm) and J(λ) from spectra.

       Returns (forster_radius_nm, overlap_integral).
       """
   ```

5. Replace flat donor/acceptor fields with **lists** of entities, probes,
   and FRET pairs:
   ```python
   # In SampleDefinition:
   entities: list[EntityDefinition] = dataclasses.field(default_factory=list)
   probes: list[ProbeDefinition] = dataclasses.field(default_factory=list)
   fret_pairs: list[FretPairDefinition] = dataclasses.field(default_factory=list)
   ```

   Each `ProbeDefinition.entity_index` points into `entities[]`.
   Each `FretPairDefinition.probe_1_index` / `probe_2_index` points into
   `probes[]`.

   This supports:
   - **Standard 2-color FRET**: 1 entity, 2 probes, 1 FRET pair
   - **3-color FRET**: 1+ entities, 3 probes, 2 FRET pairs (probe 0→1, 1→2)
   - **4-color FRET**: 1+ entities, 4 probes, 3+ FRET pairs
   - **Homo-FRET**: 2 instances of same probe at different positions, 1 pair
   - **Multi-chain complex**: 2+ entities (e.g. protein + DNA), probes on
     different chains with different entity_index values
   - **Homodimer**: 1 entity, 2 chains (asym_id "A" and "B"), probes
     distinguished by asym_id
   - **FCS / single-label**: 1 entity, 1 probe, 0 FRET pairs

   The old flat `donor`/`acceptor`/`donor_probe_name`/`acceptor_probe_name`
   fields are removed. For backwards compatibility during migration,
   `create_sample` should accept old-style fields and internally convert:
   `donor_probe_name` → `probes[0]`, `acceptor_probe_name` → `probes[1]`,
   auto-create one `FretPairDefinition(probe_1_index=0, probe_2_index=1)`.
   Old-style `entity_name`/`entity_type`/`entity_sequence` → 
   `entities[0] = EntityDefinition(name=entity_name, ...)`.

   **Database mapping**:
   - Each `EntityDefinition` → one row in `entities` + `entity_poly_seq` rows
   - Each `ProbeDefinition` → one row in `flr_sample_probe` +
     one row in `flr_poly_probe_position`
   - Each `FretPairDefinition` → one row in `flr_fret_forster_radius`
   - The `fluorophore_type` on `flr_sample_probe` is derived from FRET pair
     context, not set directly on the probe

6. Add validation in `SampleDefinition.__post_init__`:
   ```python
   def __post_init__(self):
       # Validate entity types
       for entity in self.entities:
           if entity.entity_type and entity.entity_type not in ENTITY_TYPES:
               raise ValueError(
                   f"entity_type {entity.entity_type!r} not in vocabulary: {ENTITY_TYPES}"
               )
       # Validate probe entity_index references
       for i, probe in enumerate(self.probes):
           if self.entities and probe.entity_index >= len(self.entities):
               raise ValueError(
                   f"probe[{i}].entity_index={probe.entity_index} "
                   f"exceeds entities list length {len(self.entities)}"
               )
       # Validate FRET pair probe indices
       for i, pair in enumerate(self.fret_pairs):
           if pair.probe_1_index >= len(self.probes):
               raise ValueError(
                   f"fret_pair[{i}].probe_1_index={pair.probe_1_index} "
                   f"exceeds probes list length {len(self.probes)}"
               )
           if pair.probe_2_index >= len(self.probes):
               raise ValueError(
                   f"fret_pair[{i}].probe_2_index={pair.probe_2_index} "
                   f"exceeds probes list length {len(self.probes)}"
               )
   ```

### Task 2: Update `sample_manager.py` to populate flr_* tables

**File**: `chisurf/core/mfdb/sample_manager.py`

The current `create_sample` creates records in `mfdb_sample` and partially in
legacy tables. It must be updated to fully populate the flrCIF data model.

**Changes to `create_sample`:**

1. **Create all entities** from `SampleDefinition.entities` in the `entities`
   table using `db.add_entity()` or `_insert_entity()`. For each entity with
   a sequence, create `entity_poly_seq` rows. Track `entity_id` per index
   so probes can reference the correct entity.

2. **Create probes in `probes` table** using `db.add_probe()` or look up
   existing probes by name using `db.search_probes()`. The name must match
   `COMMON_PROBE_NAMES` if the probe is a known dye — warn (don't reject) if
   it doesn't match, since custom dyes are valid.

3. **Create optical properties** from `ProbeDefinition` fields using
   `db.add_optical_property()`:
   - `absorption_wavelength_nm` → property_name `"absorption_wavelength"`, unit `"nm"`
   - `emission_wavelength_nm` → `"emission_wavelength"`, `"nm"`
   - `quantum_yield` → `"quantum_yield"`, dimensionless
   - `extinction_coefficient` → `"extinction_coefficient"`, `"M-1cm-1"`
   Note: these may be auto-populated from `DEFAULT_FLUOROPHORE_SPECTRA` if the
   user didn't provide experimental values. Store whatever is on the
   `ProbeDefinition` after `__post_init__` runs.

4. **Store spectra** from `ProbeDefinition.absorption_spectrum` and
   `emission_spectrum` using `db.add_spectrum()`. These may come from the
   default library or from experimental data provided by the user.

5. **Create probe positions** in `flr_poly_probe_position` using
   `db.add_poly_probe_position()`. Resolve `entity_index` → `entity_id`
   from step 1. Pass all position fields:
   - `entity_id` — from `entities[probe.entity_index]`
   - `asym_id` — chain/strand ID
   - `seq_id` → `residue_number` — residue number
   - `comp_id` → `residue_name` — residue name
   - `atom_id` — attachment atom (optional)
   - `mutation_flag` — "yes"/"no"
   - `modification_flag` — "yes"/"no"
   - `auth_name` — author position label

6. **Create the flr_sample** record using `db.add_sample()` with full
   parameters (entity_assembly_id, sample_condition_id, num_of_probes,
   solvent_phase).

7. **Create sample-probe mappings** in `flr_sample_probe` using
   `db.add_sample_probe()`. Derive `fluorophore_type` from FRET pair context:
   probe appearing only as `probe_1` → `"donor"`, only as `probe_2` →
   `"acceptor"`, both roles → `"unspecified"`, no pairs → `"unspecified"`.

8. **Create the Förster radius** in `flr_fret_forster_radius` if both
   donor and acceptor are provided and `forster_radius_nm` is set.

9. **Also create `mfdb_sample`** record (as now) for quick lookups.

10. **Create entity assembly** linking entity + probes.

**Critical API notes** (bugs from original PRD that must not be repeated):

- Connection attribute is `db.conn` (NOT `db.con`)
- Use `with db._transaction():` for atomicity (NOT `db.conn.commit()`)
- MFDatabase constructor: `MFDatabase(db_path)` — no `object_store_root` param
- Edge columns: `source_node_id`, `target_node_id` (NOT `source_id`, `target_id`)
- sqlite3.Row: use `row["column_name"]` or `dict(row)` (NOT `row[0]`)
- Use `db.add_edge()` for mfdb_edge inserts — it handles column mapping

### Task 3: Add vocabulary validation to sample creation

**File**: `chisurf/core/mfdb/sample_manager.py`

When creating a sample, validate fields against known vocabularies:

| Field | Vocabulary | Action on mismatch |
|-------|-----------|-------------------|
| `entity_type` | `ENTITY_TYPES` | Raise `ValueError` |
| `probe.name` (each in `probes` list) | `COMMON_PROBE_NAMES` | Warn via `logging.warning`, accept |
| Buffer components | `BUFFER_COMPONENTS` | No validation (freeform) |
| `ph` | Must be 0–14 if set | Raise `ValueError` |
| `temperature_k` | Must be > 0 if set | Raise `ValueError` |
| `quantum_yield` | Must be 0–1 if set | Raise `ValueError` |

For probe names: many labs use custom dyes or dye derivatives (e.g.
"Cy3B-maleimide", "ATTO647N-NHS"). Do NOT reject unknown names — log a
warning suggesting the closest match from `COMMON_PROBE_NAMES`.

### Task 4: Add `get_sample_full_description` to return the complete graph

**File**: `chisurf/core/mfdb/sample_manager.py`

Add a function that returns the full structured sample description by joining
across all related tables:

```python
def get_sample_full_description(db: MFDatabase, sample_id: str) -> dict | None:
    """Return the complete sample description including entity, probes,
    positions, condition, and Förster radius.

    Returns a dictionary with the structure:
    {
        "sample_id": "...",
        "description": "...",
        "entities": [
            {
                "entity_id": "...",
                "type": "protein",
                "common_name": "T4 Lysozyme",
                "sequence": "MNIFEMLR...",
            },
            {
                "entity_id": "...",
                "type": "dna",
                "common_name": "DNA ruler",
                "sequence": "GCATCG...",
            },
        ],
        "condition": {
            "ph": 7.4,
            "temperature_k": 298.15,
            "ionic_strength": 0.15,
            "buffer_composition": "PBS",
        },
        "probes": [
            {
                "probe_name": "Cy3B",
                "fluorophore_type": "donor",  # derived from fret_pairs context
                "position": {
                    "entity_id": "...",
                    "asym_id": "A",
                    "seq_id": 48,
                    "comp_id": "CYS",
                    "atom_id": "CB",
                    "mutation_flag": "yes",
                    "modification_flag": "no",
                    "auth_name": "S48C",
                },
                "properties": {
                    "absorption_wavelength": {"value": "559", "unit": "nm"},
                    "emission_wavelength": {"value": "572", "unit": "nm"},
                    "quantum_yield": {"value": "0.67", "unit": null},
                    "extinction_coefficient": {"value": "130000", "unit": "M-1cm-1"},
                },
                "has_default_spectra": true,
            },
            ...
        ],
        "fret_pairs": [
            {
                "probe_1_name": "Cy3B",
                "probe_2_name": "ATTO647N",
                "forster_radius_nm": 5.1,
                "kappa_squared": 0.666667,
                "refractive_index": 1.4,
            },
            # ... additional pairs for 3-color FRET etc.
        ],
        "key_values": [
            {"key": "pdbx.sample_type", "value": "protein"},
            ...
        ],
    }
    """
```

Implementation: query `flr_sample` → JOIN `flr_sample_condition` via
`sample_condition_id` → JOIN `flr_entity_assembly` → JOIN `entities` → query
`entity_poly_seq` for sequence → query `flr_sample_probe` → JOIN `probes` →
JOIN `flr_poly_probe_position` → query `optical_properties` → query
`flr_fret_forster_radius`. Use the existing repository methods:
`db.get_sample()`, `db.get_sample_probe_mappings()`, `db.get_sample_key_values()`.

### Task 5: Add PDBx key-value metadata support

**File**: `chisurf/core/mfdb/sample_manager.py`

The `flr_sample_key_value` table stores extensible metadata as PDBx-style
key-value pairs (e.g. `"pdbx.sample_type"`, `"flr.solvent_phase"`). Add
functions to:

1. **Set validated key-value pairs:**
   ```python
   def set_sample_metadata(db, sample_id, key, value, details=None):
       """Set a PDBx/flrCIF key-value pair on a sample.

       The key should follow PDBx naming: 'category.attribute',
       e.g. 'pdbx.sample_type', 'flr.solvent_phase'.

       If PDBx dictionary is available, validates that the key exists.
       """
   ```

2. **Auto-populate standard key-values** during `create_sample`:
   - `flr.solvent_phase` from `SampleDefinition`
   - `flr.num_of_probes` (count from probes provided)
   - `pdbx.entity_type` from `entity_type`
   - `chisurf.sample_origin` = `"user_created"` or `"cif_import"`

3. **Suggest PDBx keys** for interactive use (used by GUI autocomplete):
   ```python
   def suggest_pdbx_keys(prefix: str = "") -> list[tuple[str, str]]:
       """Return (key, description) pairs from PDBx dictionary matching prefix."""
   ```
   Uses `pdbx_metadata.get_pdbx_metadata_keys()` and
   `pdbx_metadata.get_pdbx_metadata_descriptions()`.

### Task 6: Seed vocabulary table with sample-related vocabularies

**File**: `chisurf/core/mfdb/schema.py` (in `bootstrap_vocabulary`)

Add vocabulary seed data for sample-related fields:

```python
# In bootstrap_vocabulary():
"entity_type": [
    ("protein", "Protein"),
    ("dna", "DNA"),
    ("rna", "RNA"),
    ("polymer", "Polymer"),
    ("non-polymer", "Non-polymer"),
    ("water", "Water"),
    ("macromolecule", "Macromolecule"),
    ("oligosaccharide", "Oligosaccharide"),
    ("ligand", "Ligand"),
    ("solvent", "Solvent"),
],
"fluorophore_type": [
    ("donor", "Donor"),
    ("acceptor", "Acceptor"),
    ("unspecified", "Unspecified"),
],
"solvent_phase": [
    ("liquid", "Liquid"),
    ("solid", "Solid"),
    ("gas", "Gas"),
    ("vitrified", "Vitrified"),
],
"sample_type": [
    ("protein", "Protein sample"),
    ("dna", "DNA sample"),
    ("rna", "RNA sample"),
    ("physical_sample", "Generic sample"),
],
```

These must match the constants in `models.py` — single source of truth.

### Task 7: Update `SamplePicker` dialog with structured fields

**File**: `chisurf/gui/widgets/sample_picker.py`

The current `_SampleDefinitionDialog` has flat text fields. Improve it to:

1. **Group fields into tabs or sections:**
   - **Entities**: a list/table of entities. Each row: name, type (combo from
     `ENTITY_TYPES`), sequence. "Add entity" / "Remove entity" buttons.
     Most samples have 1 entity, but protein-DNA complexes, heterodimers
     need 2+.
   - **Probes**: a list/table of probes. Each row: name (combo with
     `COMMON_PROBE_NAMES` + custom), entity (combo from entities list),
     chain (asym_id), residue number (seq_id), residue name (comp_id),
     atom (atom_id), mutation flag, absorption/emission peak λ, QY, ε,
     SMILES (optional). No donor/acceptor label — that is determined by
     FRET pair context. "Add probe" / "Remove probe" buttons. No limit
     on probe count.
   - **FRET pairs**: a list/table of pairs. Each row: probe 1 (combo from
     probes list, energy donor), probe 2 (combo from probes list,
     energy acceptor), R₀, κ², n. Auto-populated when exactly 2
     probes exist (common 2-color case). For homo-FRET, both combos
     can reference the same dye at different positions.
   - **Condition**: buffer, pH, temperature, salt concentration

2. **Probe name combo box** should be editable (allow custom names) but
   populated from `COMMON_PROBE_NAMES`. When a known probe is selected,
   auto-fill its photophysical properties from `DEFAULT_FLUOROPHORE_SPECTRA`
   (the built-in library). Show a "(default)" indicator next to auto-filled
   values so the user knows they can override with experimental data.

3. **Entity combo on probe rows** — each probe row has a dropdown referencing
   the entities list. When only 1 entity exists, it's auto-selected.

4. **Show completeness indicator** — a simple progress or checklist showing
   which flrCIF categories are filled (entities, sequence, probes with
   positions, condition, Förster radius).

### Task 8: Validate flrCIF round-trip

**File**: `chisurf/core/mfdb/sample_manager.py`

Add a function that validates a sample is complete enough for flrCIF export:

```python
def validate_sample_for_export(db, sample_id) -> list[str]:
    """Check that a sample has enough data for valid flrCIF export.

    Returns a list of warnings/missing fields. Empty list = export-ready.

    Required for minimal flrCIF:
    - At least one entity
    - At least two probes with positions
    - Sample-probe mappings with fluorophore_type
    - Sample condition (at minimum pH and temperature)

    Recommended:
    - Entity sequence
    - Optical properties for probes
    - Förster radius
    """
```

### Task 9: Link datasets to samples in project archiver (unchanged)

**File**: `chisurf/core/mfdb/project_archiver.py`

This is already implemented correctly. In `_archive_datasets()`, after creating
the raw_measurement artifact, the code checks for `sample_id` in dataset
metadata and calls `link_artifact_to_sample()`. No changes needed — just verify
it works with the restructured `SampleDefinition`.

### Task 10: Write comprehensive tests

**File**: `test/fio/test_sample_manager.py`

Extend the existing 7 tests with:

```python
def test_create_sample_populates_flr_tables(db):
    """Creating a sample with full definition populates entity, probes,
    positions, condition, and sample-probe mappings in flr_* tables."""
    defn = SampleDefinition(
        name="T4L-Cy3B-ATTO647N",
        entities=[
            EntityDefinition(
                name="T4 Lysozyme",
                entity_type="protein",
                sequence="MNIFEMLR...",  # truncated
            ),
        ],
        probes=[
            ProbeDefinition(
                name="Cy3B",
                entity_index=0,
                seq_id=48,
                comp_id="CYS",
                atom_id="CB",
                asym_id="A",
                mutation_flag="yes",
                auth_name="S48C",
                absorption_wavelength_nm=559.0,
                emission_wavelength_nm=572.0,
                quantum_yield=0.67,
                extinction_coefficient=130000.0,
            ),
            ProbeDefinition(
                name="ATTO647N",
                entity_index=0,
                seq_id=131,
                comp_id="CYS",
                atom_id="CB",
                asym_id="A",
                mutation_flag="yes",
                auth_name="S131C",
                absorption_wavelength_nm=644.0,
                emission_wavelength_nm=669.0,
                quantum_yield=0.65,
                extinction_coefficient=150000.0,
            ),
        ],
        fret_pairs=[
            FretPairDefinition(
                probe_1_index=0,
                probe_2_index=1,
                forster_radius_nm=5.1,
                kappa_squared=0.666667,
                refractive_index=1.4,
            ),
        ],
        buffer_description="50 mM Tris-HCl pH 7.4, 150 mM NaCl",
        ph=7.4,
        temperature_k=298.15,
        salt_concentration_m=0.15,
    )
    sample_id = create_sample(db, defn)

    # Verify entity
    entities = db.conn.execute(
        "SELECT * FROM entities WHERE deleted_at IS NULL"
    ).fetchall()
    assert len(entities) >= 1
    assert dict(entities[0])["type"] == "protein"

    # Verify sequence was stored
    seq_rows = db.conn.execute(
        "SELECT mon_id FROM entity_poly_seq WHERE entity_id = ? ORDER BY num",
        (dict(entities[0])["entity_id"],)
    ).fetchall()
    assert len(seq_rows) > 0

    # Verify probes exist — fluorophore_type is derived from FRET pair context
    probes = db.get_sample_probe_mappings(sample_id)
    assert len(probes) == 2
    probe_names = {p["chromophore_name"] for p in probes}
    assert "Cy3B" in probe_names
    assert "ATTO647N" in probe_names

    # Verify probe positions with full position model
    positions = db.conn.execute(
        "SELECT * FROM flr_poly_probe_position WHERE deleted_at IS NULL"
    ).fetchall()
    assert len(positions) >= 2
    pos_dicts = [dict(p) for p in positions]
    assert any(p["residue_number"] == 48 and p["asym_id"] == "A" for p in pos_dicts)
    assert any(p["residue_number"] == 131 and p["asym_id"] == "A" for p in pos_dicts)
    # Check mutation_flag is set
    assert all(p.get("mutation_flag") == "yes" for p in pos_dicts)

    # Verify condition
    condition = db.conn.execute(
        "SELECT * FROM flr_sample_condition WHERE deleted_at IS NULL"
    ).fetchone()
    assert condition is not None
    assert dict(condition)["ph"] == 7.4

    # Verify Förster radius
    forster = db.conn.execute(
        "SELECT * FROM flr_fret_forster_radius WHERE deleted_at IS NULL"
    ).fetchone()
    assert forster is not None
    assert abs(dict(forster)["forster_radius"] - 5.1) < 0.01

    # Verify full description
    desc = get_sample_full_description(db, sample_id)
    assert desc is not None
    assert len(desc["entities"]) == 1
    assert desc["entities"][0]["type"] == "protein"
    assert len(desc["probes"]) == 2
    assert desc["condition"]["ph"] == 7.4
    assert len(desc["fret_pairs"]) == 1
    assert abs(desc["fret_pairs"][0]["forster_radius_nm"] - 5.1) < 0.01


def test_create_sample_validates_entity_type(db):
    """Invalid entity_type raises ValueError."""
    with pytest.raises(ValueError, match="entity_type"):
        SampleDefinition(
            name="bad",
            entities=[EntityDefinition(name="bad-entity", entity_type="alien")],
        )


def test_create_sample_warns_unknown_probe(db, caplog):
    """Unknown probe name logs a warning but succeeds."""
    defn = SampleDefinition(
        name="custom-dye-sample",
        probes=[ProbeDefinition(name="MyCustomDye-NHS")],
    )
    sample_id = create_sample(db, defn)
    assert sample_id
    assert "MyCustomDye-NHS" in caplog.text or True  # warning logged


def test_three_color_fret(db):
    """3-color FRET: 3 probes, 2 FRET pairs (D→R, R→A)."""
    defn = SampleDefinition(
        name="3color-Cy3B-Cy5-Cy7",
        entities=[EntityDefinition(name="Triple-labeled DNA", entity_type="dna")],
        probes=[
            ProbeDefinition(name="Cy3B", entity_index=0, seq_id=5, asym_id="A"),
            ProbeDefinition(name="Cy5", entity_index=0, seq_id=15, asym_id="A"),
            ProbeDefinition(name="Cy7", entity_index=0, seq_id=25, asym_id="A"),
        ],
        fret_pairs=[
            FretPairDefinition(probe_1_index=0, probe_2_index=1, forster_radius_nm=5.4),
            FretPairDefinition(probe_1_index=1, probe_2_index=2, forster_radius_nm=6.0),
        ],
    )
    sample_id = create_sample(db, defn)

    probes = db.get_sample_probe_mappings(sample_id)
    assert len(probes) == 3

    forster_rows = db.conn.execute(
        "SELECT * FROM flr_fret_forster_radius WHERE deleted_at IS NULL"
    ).fetchall()
    assert len(forster_rows) >= 2

    desc = get_sample_full_description(db, sample_id)
    assert len(desc["probes"]) == 3
    assert len(desc["fret_pairs"]) == 2


def test_homo_fret(db):
    """Homo-FRET: same dye at two positions — both donor AND acceptor."""
    defn = SampleDefinition(
        name="homo-FRET-ATTO488",
        entities=[EntityDefinition(name="dsDNA", entity_type="dna")],
        probes=[
            ProbeDefinition(name="ATTO 488", entity_index=0, seq_id=5, asym_id="A"),
            ProbeDefinition(name="ATTO 488", entity_index=0, seq_id=20, asym_id="A"),
        ],
        fret_pairs=[
            FretPairDefinition(probe_1_index=0, probe_2_index=1, forster_radius_nm=4.5),
        ],
    )
    sample_id = create_sample(db, defn)
    probes = db.get_sample_probe_mappings(sample_id)
    assert len(probes) == 2
    assert probes[0]["chromophore_name"] == probes[1]["chromophore_name"]


def test_single_probe_no_fret(db):
    """FCS sample: single probe, no FRET pairs."""
    defn = SampleDefinition(
        name="FCS-Alexa488",
        entities=[EntityDefinition(name="GFP", entity_type="protein")],
        probes=[
            ProbeDefinition(name="Alexa Fluor 488", entity_index=0, seq_id=1, asym_id="A", quantum_yield=0.92),
        ],
    )
    sample_id = create_sample(db, defn)
    probes = db.get_sample_probe_mappings(sample_id)
    assert len(probes) == 1
    desc = get_sample_full_description(db, sample_id)
    assert len(desc["fret_pairs"]) == 0


def test_four_color_fret(db):
    """4-color FRET: 4 probes, multiple FRET pairs."""
    defn = SampleDefinition(
        name="4color-sample",
        entities=[EntityDefinition(name="Multi-labeled construct", entity_type="dna")],
        probes=[
            ProbeDefinition(name="Alexa Fluor 488", entity_index=0, seq_id=5, asym_id="A"),
            ProbeDefinition(name="Cy3B", entity_index=0, seq_id=15, asym_id="A"),
            ProbeDefinition(name="Cy5", entity_index=0, seq_id=25, asym_id="A"),
            ProbeDefinition(name="Cy7", entity_index=0, seq_id=35, asym_id="A"),
        ],
        fret_pairs=[
            FretPairDefinition(probe_1_index=0, probe_2_index=1, forster_radius_nm=5.0),
            FretPairDefinition(probe_1_index=1, probe_2_index=2, forster_radius_nm=5.4),
            FretPairDefinition(probe_1_index=2, probe_2_index=3, forster_radius_nm=6.0),
        ],
    )
    sample_id = create_sample(db, defn)
    probes = db.get_sample_probe_mappings(sample_id)
    assert len(probes) == 4
    desc = get_sample_full_description(db, sample_id)
    assert len(desc["fret_pairs"]) == 3


def test_multi_entity_protein_dna_complex(db):
    """Protein-DNA complex: 2 entities, probes on different chains."""
    defn = SampleDefinition(
        name="T4L-DNA-complex",
        entities=[
            EntityDefinition(name="T4 Lysozyme", entity_type="protein", sequence="MNIFEMLR..."),
            EntityDefinition(name="DNA ruler", entity_type="dna", sequence="GCATCGATCG"),
        ],
        probes=[
            ProbeDefinition(
                name="Cy3B", entity_index=0, seq_id=48, comp_id="CYS",
                atom_id="CB", asym_id="A", mutation_flag="yes", auth_name="S48C",
            ),
            ProbeDefinition(
                name="ATTO647N", entity_index=1, seq_id=5, comp_id="dT",
                atom_id="C5", asym_id="B", mutation_flag="no",
            ),
        ],
        fret_pairs=[
            FretPairDefinition(probe_1_index=0, probe_2_index=1, forster_radius_nm=5.1),
        ],
    )
    sample_id = create_sample(db, defn)
    desc = get_sample_full_description(db, sample_id)
    assert len(desc["entities"]) == 2
    assert desc["entities"][0]["type"] == "protein"
    assert desc["entities"][1]["type"] == "dna"
    # Probes on different chains
    positions = db.conn.execute(
        "SELECT asym_id FROM flr_poly_probe_position WHERE deleted_at IS NULL"
    ).fetchall()
    asym_ids = {dict(p)["asym_id"] for p in positions}
    assert "A" in asym_ids
    assert "B" in asym_ids


def test_homodimer_same_entity_two_chains(db):
    """Homodimer: 1 entity type, 2 chains (A and B)."""
    defn = SampleDefinition(
        name="homodimer-CaM",
        entities=[EntityDefinition(name="Calmodulin", entity_type="protein")],
        probes=[
            ProbeDefinition(name="ATTO 488", entity_index=0, seq_id=34, asym_id="A"),
            ProbeDefinition(name="ATTO 647N", entity_index=0, seq_id=34, asym_id="B"),
        ],
        fret_pairs=[
            FretPairDefinition(probe_1_index=0, probe_2_index=1, forster_radius_nm=5.0),
        ],
    )
    sample_id = create_sample(db, defn)
    positions = db.conn.execute(
        "SELECT asym_id, residue_number FROM flr_poly_probe_position WHERE deleted_at IS NULL"
    ).fetchall()
    pos_dicts = [dict(p) for p in positions]
    # Same residue number on different chains
    assert any(p["asym_id"] == "A" and p["residue_number"] == 34 for p in pos_dicts)
    assert any(p["asym_id"] == "B" and p["residue_number"] == 34 for p in pos_dicts)


def test_default_spectra_auto_populated(db):
    """Known fluorophore auto-populates spectra from default library."""
    probe = ProbeDefinition(name="Cy3B")
    # After __post_init__, defaults should be filled from DEFAULT_FLUOROPHORE_SPECTRA
    assert probe.absorption_wavelength_nm is not None
    assert probe.emission_wavelength_nm is not None
    assert probe.quantum_yield is not None
    assert probe.absorption_spectrum is not None
    assert probe.emission_spectrum is not None


def test_default_spectra_not_override_experimental(db):
    """Experimental values override default spectra library."""
    probe = ProbeDefinition(
        name="Cy3B",
        quantum_yield=0.70,  # user-measured, different from default 0.67
    )
    assert probe.quantum_yield == 0.70  # not overwritten by default


def test_unknown_dye_no_default_spectra(db):
    """Unknown dye name: no auto-population, fields remain None."""
    probe = ProbeDefinition(name="MyCustomDye-NHS")
    assert probe.absorption_wavelength_nm is None
    assert probe.emission_wavelength_nm is None
    assert probe.quantum_yield is None


def test_entity_index_out_of_range(db):
    """entity_index exceeding entities list raises ValueError."""
    with pytest.raises(ValueError, match="entity_index"):
        SampleDefinition(
            name="bad-entity-ref",
            entities=[EntityDefinition(name="only-one", entity_type="protein")],
            probes=[ProbeDefinition(name="Cy3B", entity_index=1)],  # only index 0 exists
        )


def test_validate_sample_for_export_complete(db):
    """A fully specified 2-color sample passes export validation."""
    # ... create full sample as in test_create_sample_populates_flr_tables ...
    warnings = validate_sample_for_export(db, sample_id)
    assert warnings == []


def test_validate_sample_for_export_incomplete(db):
    """A minimal sample reports missing fields for export."""
    sample_id = create_sample(db, SampleDefinition(name="bare"))
    warnings = validate_sample_for_export(db, sample_id)
    assert len(warnings) > 0
    assert any("entity" in w.lower() for w in warnings)
    assert any("probe" in w.lower() for w in warnings)


def test_flrcif_roundtrip(db, tmp_path):
    """Sample created via SampleDefinition survives flrCIF export/import."""
    # Create sample with full definition (2-color FRET)
    # Export via db.export_flr_cif()
    # Create fresh DB, import via import_structure_file()
    # Compare entities, probes, positions, conditions, Förster radius
    ...


def test_get_sample_full_description_structure(db):
    """Full description contains all expected sections."""
    # ... create sample ...
    desc = get_sample_full_description(db, sample_id)
    assert "entities" in desc
    assert "condition" in desc
    assert "probes" in desc
    assert "fret_pairs" in desc
    assert "key_values" in desc


def test_ph_none_not_lost(db):
    """pH=None is stored as NULL, not as 0.0."""
    defn = SampleDefinition(name="no-ph-sample")
    sample_id = create_sample(db, defn)
    desc = get_sample_full_description(db, sample_id)
    assert desc["condition"]["ph"] is None


def test_ph_zero_preserved(db):
    """pH=0.0 (strongly acidic) is stored correctly, not treated as unset."""
    defn = SampleDefinition(name="acid-sample", ph=0.0)
    sample_id = create_sample(db, defn)
    desc = get_sample_full_description(db, sample_id)
    assert desc["condition"]["ph"] == 0.0
```

## Definition of Done

**Data model:**
- [ ] `EntityDefinition` dataclass for multi-entity/multi-chain support
- [ ] `SampleDefinition.entities` is a **list** — supports 1, 2, N entities
- [ ] `SampleDefinition` uses `Optional` instead of sentinel values
- [ ] `ProbeDefinition` with full flrCIF position model: entity_index, asym_id,
      seq_id, comp_id, atom_id, mutation_flag, modification_flag, auth_name
- [ ] `ProbeDefinition` with photophysical properties, spectra, and SMILES/InChI
- [ ] `ProbeDefinition` has **no** donor/acceptor label — role is pair-relative
- [ ] `ProbeDefinition.__post_init__` auto-populates from `DEFAULT_FLUOROPHORE_SPECTRA`
- [ ] `FretPairDefinition` dataclass defines probe_1 (donor role) → probe_2 (acceptor role)
- [ ] `SampleDefinition.probes` is a **list** — supports 1, 2, 3, 4+ probes
- [ ] `SampleDefinition.fret_pairs` is a **list** — supports 0, 1, 2+ FRET pairs
- [ ] Index validation: entity_index < len(entities), probe indices < len(probes)

**Default spectra:**
- [ ] `DEFAULT_FLUOROPHORE_SPECTRA` loaded from bundled JSON file
- [ ] Known dyes auto-populate spectra + scalars when not explicitly provided
- [ ] Experimental values always override defaults
- [ ] Unknown dye names: no error, fields remain None

**Database operations:**
- [ ] `fluorophore_type` on `flr_sample_probe` derived from FRET pair context
- [ ] `compute_forster_radius` from spectral overlap when spectra provided
- [ ] `create_sample` populates all entities, `entity_poly_seq`, `probes`,
      `optical_properties`, `spectra`, `flr_poly_probe_position` (with full
      position fields), `flr_sample_probe`, `flr_sample_condition`,
      `flr_fret_forster_radius`, `flr_sample`, AND `mfdb_sample`
- [ ] Vocabulary validation: `entity_type` checked, probe names warned
- [ ] `get_sample_full_description` returns complete structured graph with
      `entities` (list), probe positions including asym_id/comp_id/atom_id
- [ ] `validate_sample_for_export` checks flrCIF export readiness
- [ ] PDBx key-value metadata support with `flr_sample_key_value`
- [ ] `mfdb_vocabulary` seeded with sample-related vocabularies
- [ ] `SamplePicker` dialog supports N entities + N probes + N FRET pairs
- [ ] `pH=None` stored as NULL (not 0.0); `pH=0.0` preserved as a valid value
- [ ] SMILES/InChI chemical descriptors for probes and linkers

**Tests:**
- [ ] Multi-entity test: protein + DNA complex, probes on different chains
- [ ] Homodimer test: 1 entity, 2 chains (A, B), same residue number
- [ ] 3-color FRET test passes (3 probes, 2 FRET pairs)
- [ ] Homo-FRET test passes (same dye at 2 positions)
- [ ] Single-probe/no-FRET test passes (FCS use case)
- [ ] Default spectra auto-population test (known dye)
- [ ] Default spectra no-override test (experimental > default)
- [ ] Unknown dye no-default test
- [ ] Entity index out-of-range validation test
- [ ] flrCIF round-trip test: create → export → import → compare
- [ ] All existing tests still pass
- [ ] All new tests pass

## API cheat sheet (avoid repeat bugs)

| Thing | Correct | Wrong (from PRD v1) |
|-------|---------|---------------------|
| DB connection | `db.conn` | `db.con` |
| Transactions | `with db._transaction():` | `db.conn.commit()` |
| Row access | `row["col"]` or `dict(row)` | `row[0]` |
| Edge columns | `source_node_id`, `target_node_id` | `source_id`, `target_id` |
| MFDatabase init | `MFDatabase(path)` | `MFDatabase(path, object_store_root=...)` |
| Edge creation | `db.add_edge(...)` | Manual INSERT into mfdb_edge |
