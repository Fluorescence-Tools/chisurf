# PRD-08: Optical Configuration Schema

## Goal

Replace opaque JSON blobs in `mfdb_setup` with structured, queryable tables
that describe the complete optical path from excitation source through to
detector for every confocal smFRET / TCSPC setup. Every photon's journey
becomes traceable: which laser, which filters, which objective, which detector,
at what power and gain, in which measurement.

## Background

Read these files before starting:
- `chisurf/core/mfdb/schema.py` — current `flr_instrument`, `flr_inst_setting`,
  `mfdb_setup` definitions; migration helpers
- `chisurf/core/mfdb/repository.py` — `MFDatabase` methods
- `overhaul/IDEAS.md` — IDEA-02 for design rationale and community alignment
- `chisurf/plugins/core/mfdb_admin/gui/entity_registry.py` — how new entities
  appear in the admin GUI

## Context: What Already Exists

| Table | Content | Gap |
|-------|---------|-----|
| `flr_instrument` | `instrument_id`, `instrument_name`, `details` | No component breakdown |
| `flr_inst_setting` | key-value pairs for instrument | Untyped, not queryable |
| `mfdb_setup` | Full setup config stored as JSON blobs | Opaque to SQL queries and GUI |
| `flr_experiment` | Links to `setup_definition_id` | Already wired |

The standard (flrCIF `FLR_INSTRUMENT` / `FLR_INST_SETTING`) is intentionally
minimal — free-text only, "will be extended in the future." This PRD defines
the chisurf extension.

> Reconciliation with PRD-04: PRD-04's prerequisite introduces the
> reading/processing channel base table `mfdb_setup_detector_channel` (detector
> channels: `channels`, `micro_time_ranges`, `g_factor`, …) and
> `mfdb_setup_pie_window`. This PRD's `mfdb_optical_channel` is the
> *spectroscopic extension* of that detection channel, **not** a parallel
> concept. `mfdb_optical_channel` should carry a
> `detector_channel_id REFERENCES mfdb_setup_detector_channel(...)` link (and the
> migration in Task 3 should populate it) so the two tables are joined, not
> duplicated. Do not re-derive channels from `detectors_json` if the
> structured base table already exists.

## Design: OME FilterSet / LightPath Pattern

Following the OME-XML model (OME v6.3.1):

- **Hardware level** (FilterSet equivalent): Each physical component is defined
  once per instrument. Tables: `mfdb_light_source`, `mfdb_optical_filter`,
  `mfdb_dichroic`, `mfdb_objective`, `mfdb_detector`.
- **Channel level** (LightPath equivalent): `mfdb_optical_channel` says which
  hardware components are active in a given setup and in what role.
- **Measurement level**: `mfdb_channel_setting` captures per-measurement
  overrides (laser power, detector gain, count rate).

## Tasks

### Task 1: Add Hardware Component Tables to schema.py

**File**: `chisurf/core/mfdb/schema.py`

Add these CREATE TABLE statements to `CREATE_STATEMENTS` list, after the
existing `flr_inst_setting` block:

```python
"""CREATE TABLE IF NOT EXISTS mfdb_light_source (
    light_source_id TEXT PRIMARY KEY,
    instrument_id TEXT REFERENCES flr_instrument(instrument_id),
    source_type TEXT NOT NULL,          -- laser / led / lamp / other
    brand TEXT,
    model TEXT,
    serial TEXT,
    wavelength_nm REAL,                 -- nominal / center wavelength
    bandwidth_nm REAL,                  -- for LED / broadband sources
    power_mw REAL,                      -- max rated output power
    pulse_width_ps REAL,                -- for pulsed lasers (TCSPC)
    rep_rate_mhz REAL,                  -- repetition rate (TCSPC)
    frequency_multiplication INTEGER DEFAULT 1,  -- SHG=2, THG=3
    tunable INTEGER DEFAULT 0,          -- boolean
    details TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT
)""",

"""CREATE TABLE IF NOT EXISTS mfdb_optical_filter (
    filter_id TEXT PRIMARY KEY,
    instrument_id TEXT REFERENCES flr_instrument(instrument_id),
    filter_type TEXT NOT NULL,          -- bandpass / longpass / shortpass / notch / multipass
    brand TEXT,
    model TEXT,
    catalog_no TEXT,
    center_wavelength_nm REAL,          -- for bandpass: center
    bandwidth_nm REAL,                  -- FWHM for bandpass
    cut_in_nm REAL,                     -- lower edge
    cut_out_nm REAL,                    -- upper edge
    cut_in_tolerance_nm REAL,
    cut_out_tolerance_nm REAL,
    transmission_pct REAL,              -- peak transmission 0-100
    optical_density REAL,               -- blocking OD
    details TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT
)""",

"""CREATE TABLE IF NOT EXISTS mfdb_dichroic (
    dichroic_id TEXT PRIMARY KEY,
    instrument_id TEXT REFERENCES flr_instrument(instrument_id),
    dichroic_type TEXT NOT NULL,        -- long-pass / short-pass / multi-band / beam-splitter
    brand TEXT,
    model TEXT,
    catalog_no TEXT,
    cutoff_wavelength_nm REAL,          -- primary cut-on/cut-off nm
    reflection_band_json TEXT,          -- JSON list of [lo_nm, hi_nm] bands reflected
    details TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT
)""",

"""CREATE TABLE IF NOT EXISTS mfdb_objective (
    objective_id TEXT PRIMARY KEY,
    instrument_id TEXT REFERENCES flr_instrument(instrument_id),
    brand TEXT,
    model TEXT,
    numerical_aperture REAL NOT NULL,
    magnification REAL,
    immersion_medium TEXT,              -- water / oil / air / glycerol / silicon
    correction TEXT,                    -- Plan-Apo / Plan-Fluor / etc.
    working_distance_mm REAL,
    details TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT
)""",

"""CREATE TABLE IF NOT EXISTS mfdb_detector (
    detector_id TEXT PRIMARY KEY,
    instrument_id TEXT REFERENCES flr_instrument(instrument_id),
    detector_type TEXT NOT NULL,        -- SPAD / APD / PMT / EMCCD / CMOS / SNSPD
    brand TEXT,
    model TEXT,
    serial TEXT,
    active_area_um REAL,
    dead_time_ns REAL,
    timing_resolution_ps REAL,         -- IRF FWHM or TTS
    dark_count_rate_hz REAL,
    qe_at_peak REAL,                   -- peak quantum efficiency 0-1
    afterpulsing_pct REAL,
    details TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT
)""",
```

Also add a TCSPC board table (critical for TCSPC/smFRET, absent from OME):

```python
"""CREATE TABLE IF NOT EXISTS mfdb_tcspc_board (
    board_id TEXT PRIMARY KEY,
    instrument_id TEXT REFERENCES flr_instrument(instrument_id),
    brand TEXT,
    model TEXT,
    serial TEXT,
    tac_range_ns REAL,                 -- full TAC range in nanoseconds
    tac_resolution_ps REAL,            -- ps per channel
    n_channels INTEGER,                -- number of TCSPC channels (detectors)
    sync_source TEXT,                  -- laser_trigger / internal / external
    details TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT
)""",
```

Add indexes after the existing instrument indexes:

```python
"CREATE INDEX IF NOT EXISTS idx_mfdb_light_source_instrument ON mfdb_light_source (instrument_id)",
"CREATE INDEX IF NOT EXISTS idx_mfdb_optical_filter_instrument ON mfdb_optical_filter (instrument_id)",
"CREATE INDEX IF NOT EXISTS idx_mfdb_dichroic_instrument ON mfdb_dichroic (instrument_id)",
"CREATE INDEX IF NOT EXISTS idx_mfdb_detector_instrument ON mfdb_detector (instrument_id)",
"CREATE INDEX IF NOT EXISTS idx_mfdb_objective_instrument ON mfdb_objective (instrument_id)",
```

### Task 2: Add Optical Channel Tables

Still in `chisurf/core/mfdb/schema.py`, add after hardware component tables:

```python
"""CREATE TABLE IF NOT EXISTS mfdb_optical_channel (
    channel_id TEXT PRIMARY KEY,
    setup_id TEXT NOT NULL REFERENCES mfdb_setup(setup_id) ON DELETE CASCADE,
    channel_name TEXT NOT NULL,
    channel_role TEXT,                 -- donor / acceptor / vv / vh / scatter / reference
    sort_order INTEGER DEFAULT 0,
    light_source_id TEXT REFERENCES mfdb_light_source(light_source_id),
    exc_filter_id TEXT REFERENCES mfdb_optical_filter(filter_id),
    dichroic_id TEXT REFERENCES mfdb_dichroic(dichroic_id),
    objective_id TEXT REFERENCES mfdb_objective(objective_id),
    em_filter_id TEXT REFERENCES mfdb_optical_filter(filter_id),
    detector_id TEXT REFERENCES mfdb_detector(detector_id),
    tcspc_board_id TEXT REFERENCES mfdb_tcspc_board(board_id),
    excitation_wavelength_nm REAL,     -- override if light source is broadband/tunable
    emission_center_nm REAL,           -- effective emission center (from filter)
    emission_bandwidth_nm REAL,        -- effective emission bandwidth (from filter)
    fluorophore TEXT,                  -- associated fluorophore name (e.g. Cy3B, ATTO647N)
    details TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT
)""",

"""CREATE TABLE IF NOT EXISTS mfdb_channel_setting (
    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL REFERENCES mfdb_optical_channel(channel_id) ON DELETE CASCADE,
    measurement_id TEXT,               -- FK to raw_data / mfdb_artifact
    laser_power_pct REAL,              -- % of max rated power
    laser_power_mw REAL,               -- actual power at sample (µW or mW)
    detector_gain REAL,
    detector_voltage REAL,
    integration_time_ms REAL,
    count_rate_hz REAL,                -- measured count rate (photons/s)
    rep_rate_mhz REAL,                 -- actual rep rate used (may differ from source spec)
    tac_range_ns REAL,                 -- TAC range for this measurement
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT
)""",
```

Add indexes:

```python
"CREATE INDEX IF NOT EXISTS idx_mfdb_optical_channel_setup ON mfdb_optical_channel (setup_id)",
"CREATE INDEX IF NOT EXISTS idx_mfdb_channel_setting_channel ON mfdb_channel_setting (channel_id)",
"CREATE INDEX IF NOT EXISTS idx_mfdb_channel_setting_measurement ON mfdb_channel_setting (measurement_id)",
```

### Task 3: Migration Helper

**File**: `chisurf/core/mfdb/schema.py`

Add a migration function that reads existing `mfdb_setup.detectors_json` and
`configuration_json` blobs and populates the new tables. This runs as part of
the schema upgrade path (version bump after current highest version):

```python
def _migrate_setup_to_optical_channels(conn, setup_id: str) -> int:
    """Parse mfdb_setup JSON blobs and create mfdb_optical_channel rows.

    Returns the number of channels created.
    """
    import json as _json

    row = conn.execute(
        "SELECT detectors_json, configuration_json FROM mfdb_setup WHERE setup_id = ?",
        (setup_id,)
    ).fetchone()
    if not row:
        return 0

    detectors_json, config_json = row
    channels_created = 0

    # Parse detectors_json — typically a dict of {channel_name: {detector_type, ...}}
    if detectors_json:
        try:
            detectors = _json.loads(detectors_json)
            if isinstance(detectors, dict):
                for name, det in detectors.items():
                    conn.execute(
                        """INSERT OR IGNORE INTO mfdb_optical_channel
                           (channel_id, setup_id, channel_name, channel_role,
                            emission_center_nm, emission_bandwidth_nm, notes)
                           VALUES (lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?)""",
                        (
                            setup_id,
                            name,
                            det.get("role", ""),
                            det.get("emission_center_nm"),
                            det.get("emission_bandwidth_nm"),
                            "Migrated from detectors_json",
                        )
                    )
                    channels_created += 1
        except Exception:
            pass

    return channels_created
```

### Task 4: Register New Tables in Entity Registry

**File**: `chisurf/plugins/core/mfdb_admin/gui/entity_registry.py`

Add entries so the mfdb-admin GUI shows tabs for the new tables. Add after the
`flr_inst_setting` block:

```python
EntitySpec(
    key="mfdb_light_source",
    title="Light Sources",
    category="mfdb_light_source",
    id_field="light_source_id",
    writable=True,
    columns=[
        ("light_source_id", "ID"),
        ("source_type", "Type"),
        ("wavelength_nm", "λ (nm)"),
        ("rep_rate_mhz", "Rep rate (MHz)"),
        ("pulse_width_ps", "Pulse width (ps)"),
        ("brand", "Brand"),
        ("model", "Model"),
    ],
    group="Instrument",
),
EntitySpec(
    key="mfdb_optical_filter",
    title="Optical Filters",
    category="mfdb_optical_filter",
    id_field="filter_id",
    writable=True,
    columns=[
        ("filter_id", "ID"),
        ("filter_type", "Type"),
        ("center_wavelength_nm", "Center (nm)"),
        ("bandwidth_nm", "BW (nm)"),
        ("cut_in_nm", "Cut-in (nm)"),
        ("cut_out_nm", "Cut-out (nm)"),
        ("brand", "Brand"),
        ("catalog_no", "Catalog #"),
    ],
    group="Instrument",
),
EntitySpec(
    key="mfdb_dichroic",
    title="Dichroics",
    category="mfdb_dichroic",
    id_field="dichroic_id",
    writable=True,
    columns=[
        ("dichroic_id", "ID"),
        ("dichroic_type", "Type"),
        ("cutoff_wavelength_nm", "Cutoff (nm)"),
        ("brand", "Brand"),
        ("model", "Model"),
    ],
    group="Instrument",
),
EntitySpec(
    key="mfdb_objective",
    title="Objectives",
    category="mfdb_objective",
    id_field="objective_id",
    writable=True,
    columns=[
        ("objective_id", "ID"),
        ("numerical_aperture", "NA"),
        ("magnification", "Mag"),
        ("immersion_medium", "Immersion"),
        ("brand", "Brand"),
        ("model", "Model"),
    ],
    group="Instrument",
),
EntitySpec(
    key="mfdb_detector",
    title="Detectors",
    category="mfdb_detector",
    id_field="detector_id",
    writable=True,
    columns=[
        ("detector_id", "ID"),
        ("detector_type", "Type"),
        ("dead_time_ns", "Dead time (ns)"),
        ("timing_resolution_ps", "Timing res. (ps)"),
        ("brand", "Brand"),
        ("model", "Model"),
        ("serial", "Serial"),
    ],
    group="Instrument",
),
EntitySpec(
    key="mfdb_tcspc_board",
    title="TCSPC Boards",
    category="mfdb_tcspc_board",
    id_field="board_id",
    writable=True,
    columns=[
        ("board_id", "ID"),
        ("brand", "Brand"),
        ("model", "Model"),
        ("tac_range_ns", "TAC range (ns)"),
        ("tac_resolution_ps", "TAC res. (ps)"),
        ("n_channels", "Channels"),
    ],
    group="Instrument",
),
EntitySpec(
    key="mfdb_optical_channel",
    title="Optical Channels",
    category="mfdb_optical_channel",
    id_field="channel_id",
    writable=True,
    columns=[
        ("channel_id", "ID"),
        ("setup_id", "Setup"),
        ("channel_name", "Name"),
        ("channel_role", "Role"),
        ("excitation_wavelength_nm", "Ex (nm)"),
        ("emission_center_nm", "Em center (nm)"),
        ("fluorophore", "Fluorophore"),
    ],
    group="Setup",
),
```

### Task 5: Add flrCIF Export Serialization

**File**: `chisurf/core/mfdb/pdbx_metadata.py` (or a new
`chisurf/core/mfdb/optical_export.py`)

Add a function that serializes `mfdb_optical_channel` rows into
`FLR_INST_SETTING` key-value rows for flrCIF export:

```python
def export_optical_channels_to_flr_inst_setting(
    db: "MFDatabase",
    setup_id: str,
    instrument_id: str,
) -> list[dict]:
    """Convert mfdb_optical_channel rows to FLR_INST_SETTING key-value dicts.

    Each channel produces a block of key-value rows with keys like:
        channel_1_name, channel_1_role, channel_1_excitation_nm, etc.

    Returns a list of dicts suitable for insertion into flr_inst_setting.
    """
    rows = db.con.execute(
        "SELECT * FROM mfdb_optical_channel WHERE setup_id = ? ORDER BY sort_order",
        (setup_id,)
    ).fetchall()
    cols = [d[0] for d in db.con.description]

    settings = []
    for i, row in enumerate(rows, 1):
        ch = dict(zip(cols, row))
        prefix = f"channel_{i}_"
        for key in ("channel_name", "channel_role", "excitation_wavelength_nm",
                    "emission_center_nm", "emission_bandwidth_nm", "fluorophore"):
            val = ch.get(key)
            if val is not None:
                settings.append({
                    "instrument_id": instrument_id,
                    "setting_name": prefix + key,
                    "setting_value": str(val),
                    "details": f"optical channel {i}",
                })
    return settings
```

### Task 6: Dictionary Entries (mfdb_flr_ext.dic)

**File**: `chisurf/core/mfdb/data/mfdb_flr_ext.dic`

Add save blocks for each new category so the admin GUI can resolve field
descriptions and enumerate valid values:

```
save_mfdb_light_source
   _category.id              mfdb_light_source
   _category.description
;     ChiSurf extension: structured light source (laser/LED/lamp) hardware record.
;     Replaces free-text in flr_inst_setting for structured optical path description.
;
   _category.mandatory_code  no
   _category_key.name        "_mfdb_light_source.light_source_id"

save__mfdb_light_source.source_type
   _item.name                "_mfdb_light_source.source_type"
   _item.category_id         mfdb_light_source
   _item.mandatory_code      yes
   _item_type.code           ucode
   _chisurf_schema.table_name  mfdb_light_source
   _chisurf_schema.column_name source_type
   _item_description.description
;     Type of light source.
;
   loop_
   _item_enumeration.value
     laser
     led
     lamp
     other

save__mfdb_light_source.wavelength_nm
   _item.name                "_mfdb_light_source.wavelength_nm"
   _item.category_id         mfdb_light_source
   _item.mandatory_code      no
   _item_type.code           float
   _chisurf_schema.table_name  mfdb_light_source
   _chisurf_schema.column_name wavelength_nm
   _item_units.code          nm
   _item_description.description
;     Nominal (or center) wavelength of the light source in nanometres.
;

save__mfdb_light_source.rep_rate_mhz
   _item.name                "_mfdb_light_source.rep_rate_mhz"
   _item.category_id         mfdb_light_source
   _item.mandatory_code      no
   _item_type.code           float
   _chisurf_schema.table_name  mfdb_light_source
   _chisurf_schema.column_name rep_rate_mhz
   _item_units.code          MHz
   _item_description.description
;     Pulse repetition rate in MHz. Critical for TCSPC; sets the maximum
;     resolvable fluorescence lifetime (TAC range < 1/rep_rate).
;

save_mfdb_optical_filter
   _category.id              mfdb_optical_filter
   _category.description
;     ChiSurf extension: optical bandpass, longpass, shortpass, or notch filter.
;
   _category.mandatory_code  no
   _category_key.name        "_mfdb_optical_filter.filter_id"

save__mfdb_optical_filter.filter_type
   _item.name                "_mfdb_optical_filter.filter_type"
   _item.category_id         mfdb_optical_filter
   _item.mandatory_code      yes
   _item_type.code           ucode
   _chisurf_schema.table_name  mfdb_optical_filter
   _chisurf_schema.column_name filter_type
   loop_
   _item_enumeration.value
     bandpass
     longpass
     shortpass
     notch
     multipass
     dichroic

save_mfdb_optical_channel
   _category.id              mfdb_optical_channel
   _category.description
;     ChiSurf extension: one detection channel — the ordered path from one
;     excitation source through filters and objective to one detector.
;     Mirrors OME-XML LightPath at the Channel level.
;
   _category.mandatory_code  no
   _category_key.name        "_mfdb_optical_channel.channel_id"

save__mfdb_optical_channel.channel_role
   _item.name                "_mfdb_optical_channel.channel_role"
   _item.category_id         mfdb_optical_channel
   _item.mandatory_code      no
   _item_type.code           ucode
   _chisurf_schema.table_name  mfdb_optical_channel
   _chisurf_schema.column_name channel_role
   _item_description.description
;     Functional role of this detection channel in the smFRET experiment.
;
   loop_
   _item_enumeration.value
     donor
     acceptor
     vv
     vh
     scatter
     reference
     other
```

### Task 7: Write Tests

**File to create**: `test/fio/test_optical_configuration.py`

```python
"""Tests for optical configuration schema (PRD-08)."""
import json
import os
import tempfile
import pytest

from chisurf.core.mfdb.repository import MFDatabase


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        obj_root = os.path.join(tmpdir, "objects")
        os.makedirs(obj_root)
        database = MFDatabase(db_path, object_store_root=obj_root)
        yield database
        database.close()


def test_tables_exist(db):
    tables = {r[0] for r in db.con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for t in ("mfdb_light_source", "mfdb_optical_filter", "mfdb_dichroic",
              "mfdb_objective", "mfdb_detector", "mfdb_tcspc_board",
              "mfdb_optical_channel", "mfdb_channel_setting"):
        assert t in tables, f"Missing table: {t}"


def test_insert_pulsed_laser(db):
    db.con.execute(
        """INSERT INTO mfdb_light_source
           (light_source_id, source_type, wavelength_nm, rep_rate_mhz,
            pulse_width_ps, brand, model)
           VALUES ('ls_532', 'laser', 532.0, 80.0, 100.0, 'PicoQuant', 'PDL 800-B')"""
    )
    db.con.commit()
    row = db.con.execute(
        "SELECT wavelength_nm, rep_rate_mhz FROM mfdb_light_source WHERE light_source_id='ls_532'"
    ).fetchone()
    assert row[0] == 532.0
    assert row[1] == 80.0


def test_optical_channel_links_components(db):
    # Setup minimal instrument + setup
    db.con.execute(
        "INSERT INTO flr_instrument (instrument_id, instrument_name) VALUES ('inst_1', 'Test Confocal')"
    )
    db.con.execute(
        "INSERT INTO mfdb_setup (setup_id, name) VALUES ('setup_1', 'FRET setup')"
    )
    db.con.execute(
        """INSERT INTO mfdb_light_source
           (light_source_id, instrument_id, source_type, wavelength_nm)
           VALUES ('ls_1', 'inst_1', 'laser', 532.0)"""
    )
    db.con.execute(
        """INSERT INTO mfdb_detector
           (detector_id, instrument_id, detector_type, dead_time_ns)
           VALUES ('det_1', 'inst_1', 'SPAD', 22.0)"""
    )
    db.con.execute(
        """INSERT INTO mfdb_optical_channel
           (channel_id, setup_id, channel_name, channel_role,
            light_source_id, detector_id, emission_center_nm)
           VALUES ('ch_1', 'setup_1', 'Donor', 'donor', 'ls_1', 'det_1', 580.0)"""
    )
    db.con.commit()

    row = db.con.execute(
        """SELECT c.channel_name, ls.wavelength_nm, det.dead_time_ns
           FROM mfdb_optical_channel c
           JOIN mfdb_light_source ls ON ls.light_source_id = c.light_source_id
           JOIN mfdb_detector det ON det.detector_id = c.detector_id
           WHERE c.channel_id = 'ch_1'"""
    ).fetchone()
    assert row[0] == 'Donor'
    assert row[1] == 532.0
    assert row[2] == 22.0


def test_channel_setting_per_measurement(db):
    db.con.execute(
        "INSERT INTO mfdb_setup (setup_id, name) VALUES ('setup_2', 'FRET setup 2')"
    )
    db.con.execute(
        """INSERT INTO mfdb_optical_channel
           (channel_id, setup_id, channel_name)
           VALUES ('ch_2', 'setup_2', 'Acceptor')"""
    )
    db.con.execute(
        """INSERT INTO mfdb_channel_setting
           (channel_id, measurement_id, laser_power_pct, count_rate_hz)
           VALUES ('ch_2', 'meas_001', 10.0, 50000.0)"""
    )
    db.con.commit()

    row = db.con.execute(
        "SELECT laser_power_pct FROM mfdb_channel_setting WHERE measurement_id='meas_001'"
    ).fetchone()
    assert row[0] == 10.0
```

## Definition of Done

- [ ] Tables `mfdb_light_source`, `mfdb_optical_filter`, `mfdb_dichroic`,
      `mfdb_objective`, `mfdb_detector`, `mfdb_tcspc_board` exist in schema
- [ ] Tables `mfdb_optical_channel`, `mfdb_channel_setting` exist in schema
- [ ] Schema version bumped; migration path does not drop existing data
- [ ] Migration helper `_migrate_setup_to_optical_channels` exists
- [ ] New tables appear in mfdb-admin GUI under "Instrument" group
- [ ] flrCIF export serializes channels to `FLR_INST_SETTING` key-value rows
- [ ] Dictionary (`.dic` file) has save blocks for source_type and channel_role enumerations
- [ ] All tests in `test_optical_configuration.py` pass

## References

- [flrCIF category index (RCSB)](https://mmcif.rcsb.org/dictionaries/mmcif_ihm_flr_ext.dic/Categories/index.html) — confirms FLR_INSTRUMENT / FLR_INST_SETTING are free-text stubs
- [OME Filter and FilterSet (v6.3.1)](https://docs.openmicroscopy.org/ome-model/6.3.1/developers/filter-and-filterset.html)
- [QUAREP-LiMi WG11 minimal checklist (J. Cell Biology, 2024)](https://www.doi.org/10.1083/jcb.202107093)
- [WU-BIMAC/NBOMicroscopyMetadataSpecs (NBO-Q / LiMi-Model)](https://github.com/WU-BIMAC/NBOMicroscopyMetadataSpecs)
- `overhaul/IDEAS.md` — IDEA-02 for design rationale
