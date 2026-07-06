---
type: PRD
prd: "08"
title: "PRD-08: Optical Configuration Schema"
description: Replace opaque setup JSON blobs with structured, queryable tables describing the full optical path from source to detector.
status: planned
phase: "4"
resource: chisurf/core/mfdb
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Setup configuration currently lives in opaque JSON blobs that SQL and the GUI cannot query. This PRD adds structured hardware-component tables (light sources, optical filters, dichroics, objectives, detectors, a TCSPC board) plus channel-level tables (`mfdb_optical_channel`, `mfdb_channel_setting`) modeled on the OME FilterSet/LightPath pattern, so every photon's path — which laser, filters, objective, detector, at what power/gain — becomes traceable per measurement. It includes a migration from existing blobs, `.dic` dictionary entries and enumerations, admin-GUI entity registration, and flrCIF export. The optical-path node-graph tool becomes the authoring/visualization/validation front-end and derives crosstalk and R0 as setup-level values.

# Status
Planned. Schema tables, migration helper, dictionary entries, admin registration, flrCIF export, and the node-graph front-end integration are all pending.

# Goal
Replace opaque JSON blobs in `mfdb_setup` with structured, queryable tables that
describe the complete optical path from excitation source through to detector for
every confocal smFRET / TCSPC setup. Every photon's journey becomes traceable:
which laser, which filters, which objective, which detector, at what power and
gain, in which measurement.

# Background
Relevant code:
- `chisurf/core/mfdb/schema.py` — current `flr_instrument`, `flr_inst_setting`,
  `mfdb_setup` definitions; migration helpers
- `chisurf/core/mfdb/repository.py` — `MFDatabase` methods
- `chisurf/plugins/core/mfdb_admin/gui/entity_registry.py` — how new entities
  appear in the admin GUI

## What already exists

| Table | Content | Gap |
|-------|---------|-----|
| `flr_instrument` | `instrument_id`, `instrument_name`, `details` | No component breakdown |
| `flr_inst_setting` | key-value pairs for instrument | Untyped, not queryable |
| `mfdb_setup` | Full setup config stored as JSON blobs | Opaque to SQL queries and GUI |
| `flr_experiment` | Links to `setup_definition_id` | Already wired |

The flrCIF standard (`FLR_INSTRUMENT` / `FLR_INST_SETTING`) is intentionally
minimal — free-text only, "will be extended in the future." This PRD defines the
chisurf extension.

**Reconciliation with PRD-04.** PRD-04 introduces the reading/processing channel
base table `mfdb_setup_detector_channel` (detector channels: `channels`,
`micro_time_ranges`, `g_factor`, …) and `mfdb_setup_pie_window`. This PRD's
`mfdb_optical_channel` is the *spectroscopic extension* of that detection channel,
**not** a parallel concept. `mfdb_optical_channel` carries a
`detector_channel_id REFERENCES mfdb_setup_detector_channel(...)` link (populated
by the Task 3 migration) so the two tables are joined, not duplicated. Do not
re-derive channels from `detectors_json` if the structured base table already
exists.

# Design: OME FilterSet / LightPath pattern
Following the OME-XML model:
- **Hardware level** (FilterSet equivalent): each physical component defined once
  per instrument — `mfdb_light_source`, `mfdb_optical_filter`, `mfdb_dichroic`,
  `mfdb_objective`, `mfdb_detector`.
- **Channel level** (LightPath equivalent): `mfdb_optical_channel` says which
  hardware components are active in a given setup and in what role.
- **Measurement level**: `mfdb_channel_setting` captures per-measurement overrides
  (laser power, detector gain, count rate).

# Tasks

## Task 1: Hardware component tables (schema.py)
Add to `CREATE_STATEMENTS`, after `flr_inst_setting`. All tables carry
`created_at`/`updated_at`/`deleted_at` audit columns and an
`instrument_id REFERENCES flr_instrument(instrument_id)`.

- **`mfdb_light_source`** (PK `light_source_id`): `source_type` (laser/led/lamp/
  other), `brand`, `model`, `serial`, `wavelength_nm`, `bandwidth_nm`, `power_mw`,
  `pulse_width_ps`, `rep_rate_mhz`, `frequency_multiplication` (SHG=2/THG=3),
  `tunable`, `details`.
- **`mfdb_optical_filter`** (PK `filter_id`): `filter_type` (bandpass/longpass/
  shortpass/notch/multipass), `brand`, `model`, `catalog_no`,
  `center_wavelength_nm`, `bandwidth_nm`, `cut_in_nm`, `cut_out_nm`,
  `cut_in_tolerance_nm`, `cut_out_tolerance_nm`, `transmission_pct`,
  `optical_density`, `details`.
- **`mfdb_dichroic`** (PK `dichroic_id`): `dichroic_type` (long-pass/short-pass/
  multi-band/beam-splitter), `brand`, `model`, `catalog_no`,
  `cutoff_wavelength_nm`, `reflection_band_json`, `details`.
- **`mfdb_objective`** (PK `objective_id`): `brand`, `model`,
  `numerical_aperture` NOT NULL, `magnification`, `immersion_medium`,
  `correction`, `working_distance_mm`, `details`.
- **`mfdb_detector`** (PK `detector_id`): `detector_type` (SPAD/APD/PMT/EMCCD/
  CMOS/SNSPD), `brand`, `model`, `serial`, `active_area_um`, `dead_time_ns`,
  `timing_resolution_ps`, `dark_count_rate_hz`, `qe_at_peak`, `afterpulsing_pct`,
  `details`.
- **`mfdb_tcspc_board`** (PK `board_id`, critical for TCSPC/smFRET, absent from
  OME): `brand`, `model`, `serial`, `tac_range_ns`, `tac_resolution_ps`,
  `n_channels`, `sync_source` (laser_trigger/internal/external), `details`.

Add per-instrument indexes for each table.

## Task 2: Optical channel tables (schema.py)
- **`mfdb_optical_channel`** (PK `channel_id`, `setup_id` NOT NULL REFERENCES
  `mfdb_setup(setup_id)` ON DELETE CASCADE): `channel_name`, `channel_role`
  (donor/acceptor/vv/vh/scatter/reference), `sort_order`, and FK columns
  `light_source_id`, `exc_filter_id`, `dichroic_id`, `objective_id`,
  `em_filter_id`, `detector_id`, `tcspc_board_id`; plus
  `excitation_wavelength_nm`, `emission_center_nm`, `emission_bandwidth_nm`,
  `fluorophore` (associated dye name), `details`.
- **`mfdb_channel_setting`** (PK `setting_id` AUTOINCREMENT, `channel_id` NOT NULL
  REFERENCES `mfdb_optical_channel` ON DELETE CASCADE): `measurement_id` (FK to
  raw_data/`mfdb_artifact`), `laser_power_pct`, `laser_power_mw`, `detector_gain`,
  `detector_voltage`, `integration_time_ms`, `count_rate_hz`, `rep_rate_mhz`,
  `tac_range_ns`, `notes`.

Add indexes on `mfdb_optical_channel(setup_id)`,
`mfdb_channel_setting(channel_id)`, `mfdb_channel_setting(measurement_id)`.

## Task 3: Migration helper (schema.py)
Add `_migrate_setup_to_optical_channels(conn, setup_id)` that reads existing
`mfdb_setup.detectors_json` / `configuration_json` blobs and populates
`mfdb_optical_channel` rows (parses a `{channel_name: {detector_type, role,
emission_center_nm, …}}` dict; inserts with generated ids; stamps "Migrated from
detectors_json"). Runs as part of the schema upgrade path (version bump after the
current highest version); must not drop existing data.

## Task 4: Register new tables in the entity registry
`chisurf/plugins/core/mfdb_admin/gui/entity_registry.py` — add `EntitySpec`
entries so the admin GUI shows tabs for each new table under the "Instrument"
group (`mfdb_light_source`, `mfdb_optical_filter`, `mfdb_dichroic`,
`mfdb_objective`, `mfdb_detector`, `mfdb_tcspc_board`) and the "Setup" group
(`mfdb_optical_channel`), each `writable=True` with the appropriate `id_field` and
display columns.

## Task 5: flrCIF export serialization
`chisurf/core/mfdb/pdbx_metadata.py` (or a new `optical_export.py`) — add
`export_optical_channels_to_flr_inst_setting(db, setup_id, instrument_id)` that
serializes `mfdb_optical_channel` rows into `FLR_INST_SETTING` key-value dicts
(keys like `channel_1_name`, `channel_1_role`, `channel_1_excitation_nm`, …), one
block per channel, suitable for insertion into `flr_inst_setting`.

## Task 6: Dictionary entries (`mfdb_flr_ext.dic`)
Add `save_` category + `save__<table>.<col>` item blocks for each new category so
the admin GUI can resolve field descriptions and enumerate valid values. Include
`_item_enumeration.value` loops for `mfdb_light_source.source_type` (laser/led/
lamp/other), `mfdb_optical_filter.filter_type` (bandpass/longpass/shortpass/notch/
multipass/dichroic), and `mfdb_optical_channel.channel_role` (donor/acceptor/vv/
vh/scatter/reference/other). Each column block carries the
`_chisurf_schema.table_name`/`column_name` pair, `_item_type.code`, and units
where applicable (`nm`, `MHz`).

## Task 7: Tests
`test/fio/test_optical_configuration.py` — all eight tables exist; a pulsed laser
inserts and reads back (`wavelength_nm`, `rep_rate_mhz`); an optical channel joins
its light-source and detector components; a per-measurement `mfdb_channel_setting`
reads back its `laser_power_pct`.

## Task 8: Light Path Simulator as the optical-config front-end
The Light Path Simulator plugin (`chisurf/plugins/core/lightpath_simulator/`)
already models the full optical path as a node graph (excitation sources, filters,
dichroics/beam-splitters, detectors, fluorophores) and computes derived quantities
(spectral **crosstalk** and **R₀ overlap integrals**, `backend/crosstalk.py`). Its
`backend/mmcif_export.py` already declares dataclasses that mirror this PRD's
hardware components:

| Simulator dataclass | This PRD's structured table |
|---|---|
| `LaserLine` (excitation source) | `mfdb_light_source` |
| `FilterSetting` (excitation_filter / emission_filter / dichroic / splitter) | `mfdb_optical_filter` / `mfdb_dichroic` |
| `DetectorSetting` (detector channel) | `mfdb_detector` + `mfdb_optical_channel` |
| `FluorophoreSetting` | `flr_probe_list` (existing) |
| `InstrumentSetting` (composite) | `flr_instrument` + `mfdb_channel_setting` |

Today it serializes these to the **untyped** `flr_inst_setting` blob — exactly the
gap this PRD closes. So the simulator becomes the natural **authoring,
visualization, and validation front-end** for the structured optical config:
1. **Author / edit** a setup's optics in the node graph and **persist to the
   structured tables** (reuse `build_instrument_setting(node_states, db)` but
   target the typed tables).
2. **Visualize** a stored setup: load its optical-config rows from MFDB back into
   the node graph.
3. **Validate / derive**: the simulator already computes crosstalk + R₀ overlap
   from the optical config (filter/dichroic transmission × **dye spectra from
   [PRD-06](prd-06.md)**); surface those as setup-level derived values feeding
   [PRD-05](prd-05.md) (γ / crosstalk / R₀) and as a sanity check on detector/
   filter choices.
4. **Link, don't duplicate**: `mfdb_optical_channel.detector_channel_id`
   references the detection channel base table (`mfdb_setup_detector_channel`); the
   simulator edits the spectroscopic extension, not a parallel channel concept.

Scope note: keep the pure optics model (`backend/simulator.py`, `crosstalk.py`)
chisurf-DB-free; MFDB read/write lives in the plugin's backend services, mirroring
the PRD-08 repository methods.

# Definition of Done
- [ ] Tables `mfdb_light_source`, `mfdb_optical_filter`, `mfdb_dichroic`,
      `mfdb_objective`, `mfdb_detector`, `mfdb_tcspc_board` exist in schema
- [ ] Tables `mfdb_optical_channel`, `mfdb_channel_setting` exist in schema
- [ ] Schema version bumped; migration path does not drop existing data
- [ ] Migration helper `_migrate_setup_to_optical_channels` exists
- [ ] New tables appear in mfdb-admin GUI under "Instrument" group
- [ ] flrCIF export serializes channels to `FLR_INST_SETTING` key-value rows
- [ ] Dictionary (`.dic` file) has save blocks for source_type and channel_role enumerations
- [ ] Light Path Simulator writes/reads a setup's optics to/from the structured
      tables (not the `flr_inst_setting` blob), can visualize a stored setup's
      light path, and surfaces crosstalk / R₀ as setup-level derived values
      (PRD-05 feed)
- [ ] All tests in `test_optical_configuration.py` pass

# References
- flrCIF category index (RCSB) — confirms `FLR_INSTRUMENT` / `FLR_INST_SETTING`
  are free-text stubs.
- OME Filter and FilterSet model (v6.3.1) — the FilterSet/LightPath pattern this
  schema follows.
- QUAREP-LiMi WG11 minimal microscopy-metadata checklist (J. Cell Biology, 2024).
- Community microscopy-metadata specifications (NBO-Q / LiMi model).

# Relationships
- Extends the detection-channel base tables introduced with the burst pipeline work; `mfdb_optical_channel` links to them rather than duplicating channels.
- Consumes dye spectra from [PRD-06](prd-06.md) (optical configuration) and feeds R0/crosstalk into the calibration provenance layer.
- Independent of the operation spine ([PRD-11](prd-11.md)/[PRD-16](prd-16.md)) per the sequencing note in [PRD-11](prd-11.md).
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).
