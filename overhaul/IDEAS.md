# Ideas — MFDB Overhaul

---

## IDEA-01: fps_json_editor as starting point for flrCIF sample definer

**Date:** 2026-06-18
**Status:** Idea
**Related:** PRD-02 Task 7, PRD-02a, PRD-04 Task 5, PRD-06 Task 1, PRD-07 Step 4, PLAN Phase 2b/5.4

### Observation

The `fps_json_editor` plugin (`chisurf/plugins/modelling/fps_json_editor/`)
already implements much of what a flrCIF-grade sample definer needs:

- **PDB structure loading** with atom-level selection (`position_panel.py`)
- **Labeling position picking** — residue, chain, atom via interactive 3D view
  (MolView) and PDB selectors
- **Accessible Volume (AV) simulation** — computes dye-accessible clouds per
  position, which requires exactly the probe attachment info (chain, residue,
  atom, linker length) that flrCIF `flr_poly_probe_position` stores
- **Distance restraints** between labeled positions (`distance_panel.py`) —
  analogous to `flr_fret_distance_restraint`
- **JSON data model** (`model.py` — `FpsJsonModel`) with positions, distances,
  score sets — structurally similar to a `SampleDefinition` with probes and
  FRET pairs
- **FlexFit panel** for flexible residue ranges — maps to
  `ihm_entity_poly_segment`

### Idea

Use fps_json_editor as the implementation skeleton for a richer
`SampleDefinitionDialog` that produces `SampleDefinition` objects (PRD-02)
backed by the flrCIF data model:

1. **Position picking** → `ProbeDefinition` fields: `entity_index`, `asym_id`,
   `seq_id`, `comp_id`, `atom_id`, `mutation_flag`. The existing atom picker
   already extracts chain/residue/atom — wire it to populate `ProbeDefinition`
   instead of fps.json position dicts.

2. **Probe properties** → extend the position panel with a dye-selection combo
   (from `COMMON_PROBE_NAMES` / `DEFAULT_FLUOROPHORE_SPECTRA`). When a known
   dye is selected, auto-fill AV parameters (linker length, dye radii) AND
   photophysical properties (spectra, QY, extinction coefficient).

3. **Distance panel** → `FretPairDefinition`. The existing distance restraint
   UI pairs two positions and stores R₀ — map this to `probe_1_index`,
   `probe_2_index`, `forster_radius_nm`, `kappa_squared`, `refractive_index`.
   Add auto-computation of R₀ from spectral overlap when both probes have
   spectra.

4. **Output format** → instead of (or in addition to) fps.json, emit a
   `SampleDefinition` that `create_sample()` writes to the MFDB flr_* tables.
   The fps.json format remains available as an export for FPS/Olga
   compatibility.

5. **flrCIF/PDBx integration** → use `MmcifDictionary` (PRD-02a) to validate
   and suggest field values. The position panel could show valid `comp_id`
   values from the dictionary, validate `atom_id` against `chem_comp_atom`,
   and flag unknown entity types.

6. **Multi-entity support** → the current fps_json_editor loads one PDB. For
   protein-DNA complexes or multi-subunit assemblies, extend to load multiple
   entities (or parse chains from one PDB into separate `EntityDefinition`
   objects).

### What already works vs. what needs extension

| Capability | fps_json_editor status | Needed for sample definer |
|-----------|----------------------|--------------------------|
| Load PDB structure | Done | Reuse as-is |
| Pick residue/chain/atom | Done | Wire to `ProbeDefinition` |
| AV simulation | Done | Reuse, link to probe attachment |
| Dye selection by name | Not present | Add combo from `COMMON_PROBE_NAMES` |
| Photophysical properties | Not present | Add from `DEFAULT_FLUOROPHORE_SPECTRA` |
| Full spectra (abs/em) | Not present | Add spectrum viewer widget |
| Distance restraints | Done (fps.json format) | Map to `FretPairDefinition` |
| R₀ from spectral overlap | Not present | Add `compute_forster_radius` |
| SMILES/InChI | Not present | Add chemistry fields |
| Buffer/condition | Not present | Add condition panel |
| flrCIF export | Not present | Via `create_sample()` → `export_flr_cif()` |
| Multi-entity | Not present | Extend structure loader |

### Integration path

The cleanest approach is probably NOT to modify fps_json_editor itself, but to
create a new `sample_definer` plugin that **reuses** its sub-widgets:

- Import `PositionPanel` for atom picking
- Import `MolView` for 3D visualization
- Import `AVWorker` for accessible volume computation
- Add new panels for: dye selection, photophysics, condition, FRET pairs
- Output: `SampleDefinition` → MFDB, with optional fps.json export

This keeps fps_json_editor focused on FPS/Olga workflows while the sample
definer handles the full flrCIF data model.

### Where this plugs into the PRDs

| PRD | Section | What it says now | How fps_json_editor changes it |
|-----|---------|-----------------|-------------------------------|
| **PRD-02** | Task 7 | "Update `SamplePicker` dialog with structured fields" in `sample_picker.py` — flat form with entities/probes/pairs tabs | Replace the flat `_SampleDefinitionDialog` with a structure-aware editor reusing `PositionPanel` for atom-level probe placement. The "Probes" tab becomes a 3D position picker, not a text form. |
| **PRD-02** | Task 7.2 | "Probe name combo box: auto-fill photophysical properties from `DEFAULT_FLUOROPHORE_SPECTRA`" | Extend: when a dye is selected AND a structure is loaded, also auto-fill AV parameters (linker length, dye radii) and show the accessible volume cloud in MolView. |
| **PRD-02a** | Task 5 | "`suggest_values()` / `validate_value()` from mmCIF dictionary" | Wire dictionary validation into the position panel: `comp_id` validated against `entity_poly_seq.mon_id`, `atom_id` against `chem_comp_atom`, `entity_type` from `ENTITY_TYPES` vocabulary. |
| **PRD-04** | Task 5 | "Add a `SamplePicker` widget to the burst selection GUI" | The sample definer could serve as both the picker (browse existing) AND definer (create new) — burst plugins get one widget instead of a separate picker + dialog. |
| **PRD-06** | Task 1 | "Förster radius calculator from spectral overlap" (`forster.py`) | The distance panel (from fps_json_editor) already pairs positions and stores R₀. Wire `compute_forster_radius()` from PRD-06 so that when both probes have spectra, R₀ is auto-computed and shown inline. |
| **PRD-06** | Task 2-3 | "Expand seed data with real spectra" / "Lookup R₀ by dye pair" | The sample definer becomes the primary consumer: user picks Cy3B + ATTO647N → spectra loaded from DB → overlap integral computed → R₀ shown → `FretPairDefinition` populated. |
| **PRD-07** | Step 4 | "Add a `SamplePicker` widget to each plugin" | If the sample definer plugin exists, all plugins that need sample association can embed or invoke it via a common API (`show_sample_definer_dialog()`), not just a simple picker. |
| **PLAN** | Phase 5.4 | "User-Facing Workflow Widget (Future)" — wizard that pre-fills from MFDB | The sample definer is the first panel of this wizard: define the sample → pick probes on structure → set conditions → proceed to measurement. Structure + AV visualization makes it the natural entry point. |

### GUI phasing

The sample definer GUI is a **Phase 3+ effort** — it depends on:
- Phase 2a: mmCIF dictionary infrastructure (for vocabulary validation)
- Phase 2b: `SampleDefinition` / `ProbeDefinition` / `FretPairDefinition`
  dataclasses and `create_sample()` backend (PRD-02)
- PRD-06 Task 1: `compute_forster_radius()` for auto R₀

The fps_json_editor sub-widgets (`PositionPanel`, `MolView`, `AVWorker`) are
available now and don't need modification — only composition into a new host
widget.


---

## IDEA-02: OME optical-path schema as template for MFDB instrument provenance

**Date:** 2026-06-19
**Status:** Proposal → PRD-08
**Related:** PRD-05 (calibration), PRD-06 (fluorophore DB), PRD-08 (optical config)

### Observation

A confocal smFRET/TCSPC photon travels through a well-defined sequence of
optical elements before being registered. Without structured metadata
documenting this path, datasets cannot be:

- reproduced on a different instrument without guesswork
- compared across labs (different filter transmissions → different effective
  detection windows → different apparent FRET efficiencies)
- traced to their calibration history (e.g., which IRF file was measured on
  which laser/filter combination)

The current MFDB captures the *instrument* only as two sparse tables:
- `flr_instrument` — just `instrument_id` + `instrument_name`
- `flr_inst_setting` — generic key-value pairs

The `mfdb_setup` table stores the actual optical configuration inside JSON
blobs (`configuration_json`, `detectors_json`, `timing_calibration_json`, etc.)
which are opaque to queries and the admin GUI.

---

### Key concept from OME: FilterSet vs LightPath

The Open Microscopy Environment (OME) Consortium solved this with a two-level
separation:

| Concept | OME name | Scope | Ordering |
|---------|----------|-------|----------|
| Physical hardware installed on the instrument | **FilterSet** | Instrument level — defined once | Unordered collection |
| Active optical path for one acquisition channel | **LightPath** | Channel level — per acquisition | Ordered light-travel sequence |

This cleanly separates:
- "My confocal has a 532 nm laser, a z473rdc dichroic, and two SPADs"
  (hardware spec, stored once in the instrument record)
- "In this FRET measurement, excited at 532 nm, emission through a
  550–620 nm bandpass to SPAD-1 for donor channel"
  (active channel, stored per setup/measurement)

---

### Photon journey in confocal smFRET (what to capture)

```
[Excitation source]  →  [Exc. filter (opt.)]  →  [Dichroic]
                                                       ↓
                                                  [Objective]
                                                       ↓
                                                   [Sample]
                                                       ↓
                                              [Dichroic / beam splitter]
                                                       ↓
                                            [Em. filter (bandpass)]
                                                       ↓
                                                  [Detector]
```

For a two-channel smFRET setup (donor + acceptor), two parallel emission paths
share one excitation source and one primary dichroic.

---

### What already exists (reuse)

| Existing entity | What it covers | Disposition |
|-----------------|---------------|-------------|
| `flr_instrument` | Instrument identity (name, ID) | Keep — top-level anchor |
| `flr_inst_setting` | Key-value setting overrides | Keep — for non-structured extras |
| `mfdb_setup` | Setup configuration (JSON blobs) | Keep — backward compat; structured tables become canonical |
| `flr_experiment → setup_definition_id` | FK experiment → setup | Already wired |

---

### New tables proposed (PRD-08)

**Hardware components** — defined once per instrument, FK → `flr_instrument`:

```
mfdb_light_source      wavelength_nm, power_mw, pulse_width_ps, rep_rate_mhz,
                       source_type (laser/led/lamp), brand, model, serial

mfdb_optical_filter    filter_type (bandpass/longpass/shortpass/notch),
                       center_wavelength_nm, bandwidth_nm, transmission_pct,
                       optical_density, brand, model, catalog_no

mfdb_dichroic          cutoff_wavelength_nm, dichroic_type
                       (long-pass/short-pass/multi-band),
                       reflection_band_json, brand, model

mfdb_objective         numerical_aperture, magnification,
                       immersion_medium (water/oil/air/glycerol),
                       brand, model, working_distance_mm

mfdb_detector          detector_type (SPAD/APD/PMT/EMCCD/CMOS),
                       active_area_um, dead_time_ns, timing_resolution_ps,
                       dark_count_rate_hz, qe_at_peak, brand, model, serial
```

**Channel definition** — one detection path, FK → `mfdb_setup`:

```
mfdb_optical_channel   setup_id, channel_name,
                       channel_role (donor/acceptor/vv/vh/scatter),
                       light_source_id, exc_filter_id (nullable),
                       dichroic_id (nullable), objective_id,
                       em_filter_id (nullable), detector_id,
                       excitation_wavelength_nm (override),
                       emission_center_nm, emission_bandwidth_nm
```

**Per-measurement settings** — actual acquisition overrides:

```
mfdb_channel_setting   channel_id, measurement_id,
                       laser_power_pct, laser_power_mw,
                       detector_gain, integration_time_ms,
                       count_rate_hz, notes
```

---

### Migration from existing JSON blobs

`mfdb_setup` already has `configuration_json` and `detectors_json`. A
migration helper reads those blobs and populates the new tables. The blobs
are kept for backward compatibility; structured tables are canonical for new
data.

---

### Community standards alignment (web-search verified, 2026-06-19)

- **flrCIF `FLR_INSTRUMENT` + `FLR_INST_SETTING`** (ihmwg/flrCIF, RCSB):
  Both categories are intentionally minimal stubs — only `id` (int) and
  `details` (text) — with a note in the dictionary that they "will be extended
  in the future." The Python-IHM library confirms this limitation explicitly.
  On export, `mfdb_optical_channel` fields are serialized into
  `FLR_INST_SETTING` key-value rows as the current interoperability path.
- **OME-XML `Channel / LightPath / FilterSet`** (OME Model 6.3.1):
  The most complete formal schema for optical path description today.
  Key entities: `Laser` (Type, LaserMedium, Wavelength, RepetitionRate,
  FrequencyMultiplication, PockelCell), `Filter` (Type, CutIn/CutOut +
  Tolerance, Transmittance), `Dichroic`, `Objective` (LensNA,
  CalibratedMagnification, Correction, Immersion, WorkingDistance),
  `Detector` (Type, Gain, AmplificationGain, Binning, ReadOutRate),
  `LightSourceSettings` (Attenuation, Wavelength), `DetectorSettings`
  (Gain, Offset, Binning, Voltage, ReadOutRate).
  FilterSet lives at Instrument scope (unordered); LightPath lives at Channel
  scope (ordered, active subset). OMERO stores these in PostgreSQL via
  Hibernate ORM with optimistic locking (version column per row).
- **NBO-Q / LiMi-Model** (WU-BIMAC/NBOMicroscopyMetadataSpecs):
  Tier system for confocal — Tier 1 (minimal): instrument type + objective NA
  + channel excitation/emission wavelengths; Tier 2 (recommended): + filter
  specs + detector type; Tier 3 (complete): + full filter transmittance curves,
  laser pulse parameters, detector gain/bias.
  Renaming from NBO-Q → LiMi-Model underway (2024–2025). PHD (Persistent
  Hardware Descriptor) project funded September 2025 — assigns globally unique
  IDs to physical microscope components.
- **QUAREP-LiMi WG11** (*J. Cell Biology*, 2024): Published minimal reporting
  requirements for light microscopy. Tier 2 checklist — what the proposed
  schema covers — includes filter manufacturer + catalog number, detector
  model, laser wavelength + power, objective NA/magnification/immersion.
- **OME-NGFF (Zarr v3)**: Cloud-native format; future CLSM images stored as
  Zarr will reference `mfdb_optical_channel` for channel metadata via a
  chisurf namespace in `zarr.json`.

### TCSPC / smFRET specific additions beyond standard OME

The web search confirmed that OME-XML covers confocal generally but the
following are critical for TCSPC/smFRET and absent from OME:

| Field | Why needed for TCSPC |
|-------|---------------------|
| `Laser.RepetitionRate` (MHz) | Sets TAC range, max resolvable lifetime |
| `Laser.PulseWidth` (ps) | IRF width, convolution kernel |
| TCSPC board model + vendor | TAC linearity, ADC resolution, dead-time |
| TAC range (ns) + resolution (ps/channel) | Nanotime axis calibration |
| Sync source (laser trigger / internal) | Anchors nanotime axis |
| Fluorophore → channel assignment (D/A/ALEX) | Maps detector to biology |

These are captured in `mfdb_channel_setting` (per-measurement) and
`mfdb_light_source.pulse_width_ps` / `rep_rate_mhz` (hardware).

---

### Community tools (integration opportunities)

| Tool | Primary function | Integration |
|------|-----------------|-------------|
| **Micro-Meta App** (WU-BIMAC) | Interactive hardware-map GUI → `Microscope.json` | Import its JSON to seed `mfdb_light_source`, `mfdb_detector`, etc. |
| **MethodsJ2** (ABIF-McGill) | Auto-extract metadata from file headers via Bio-Formats | Could populate `mfdb_channel_setting` from TTTR headers on measurement import |
| **SSBD Ontology** (openssbd) | RDF/OWL semantic model linking specimens to imaging modalities | Future: link `mfdb_detector` type to FBbi terms |

---

### Design principles

1. **Separate hardware from acquisition**: Components defined once per
   instrument; `mfdb_optical_channel` selects which are active; `mfdb_channel_setting`
   records per-measurement overrides.
2. **Nullable FKs**: Not every setup has an excitation filter; unknown
   components can be omitted. Partial documentation is better than nothing.
3. **`flr_instrument` as the anchor**: All hardware components FK to
   `flr_instrument`, preserving the existing identity layer.
4. **JSON blobs as fallback**: `mfdb_setup.configuration_json` remains as
   the unstructured catch-all for setups not yet migrated.
5. **flrCIF export via key-value round-trip**: Structured channel fields are
   serialized to `FLR_INST_SETTING` rows on flrCIF export, maintaining
   standard compliance without requiring a dictionary extension.

*Note: This entry distills the OME community landscape (OME-XML, OMERO,
OME-NGFF, QUAREP-LiMi, NBO-Q, Micro-Meta App, MethodsJ2, SSBD) that was
originally captured as a raw unformatted text block. The structured design
is fully elaborated in PRD-08.*
