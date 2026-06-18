# PRD-02b: MFDB Admin Plugin Overhaul — Manual Inspection & Editing

**Depends on:** [PRD-02: Sample Tracking](PRD-02-sample-tracking.md)
**Blocks:** PRD-03 (result registry), PRD-04 (burst pipeline), PRD-07 (plugin integration)
**Related:** [PRD-06: mfdb-admin Workflow Inspection](../overhaul/PLAN_MFDB_SMFRET.md#prd-6) (future — tree views, staleness detection)

## Goal

The mfdb-admin plugin must be able to **inspect, add, and edit** every record
the PRD-02 data model produces. Before progressing to downstream PRDs (result
registry, burst pipeline, plugin integration), the user needs to manually
verify that the sample/probe/entity/FRET data is correct in the database. The
current admin GUI predates the PRD-02 rewrite and cannot display or edit the
new structured data.

### Why this matters

PRD-02 added `EntityDefinition`, multi-probe `ProbeDefinition` with flrCIF
position fields, `FretPairDefinition`, default spectra auto-population, and
vocabulary validation. None of this is visible or editable in the admin GUI.
The existing Sample tab shows raw `flr_sample` fields (sample_id, uuid,
num_probes, solvent_phase) — it cannot:

- Show or edit entities (name, type, sequence) for a sample
- Show or edit probe positions (entity, chain, residue, atom, mutation flag)
- Show or edit FRET pairs (donor↔acceptor, R₀, κ², n, overlap integral)
- Show or edit optical properties (spectra, QY, extinction coefficient)
- Create a sample using `SampleDefinition` / `SampleCreateRequest`
- View `get_sample_full_description()` output
- Run `validate_sample_for_export()` and display results
- Show or edit PDBx/flrCIF key-value metadata per sample

Without this, the only way to verify PRD-02 correctness is through test code
or raw SQL — unacceptable for a scientific application where the user must
see and confirm what the database contains.

### Where this fits

```
Phase 1: Round-trip (PRD-01) ........... DONE
Phase 2a: SQLAlchemy mapping (PRD-020) . planned
Phase 2b: mmCIF dictionary (PRD-02a) ... implemented
Phase 2c: Sample tracking (PRD-02) ..... implemented, under review
Phase 2c': Admin overhaul (PRD-02b) .... THIS PRD  ← manual verification gate
Phase 2d: Connect pipeline ............. blocked on 02b verification
Phase 3: Enrich metadata ............... blocked
Phase 4: Workflow inspection (PRD-06) .. future
```

PRD-02b is the **manual verification gate** between implementing the data model
(PRD-02) and trusting it enough to wire into downstream workflows. It is a
Phase 2c' deliverable — same phase as PRD-02, just the GUI counterpart.

## Background — what already exists

**Read these files before starting:**

| File | What to look at |
|------|-----------------|
| `chisurf/plugins/core/mfdb_admin/gui/tool.py` (4569 lines) | `MFDBWidget` with 21 tabs. Sample/Condition/Entities/Probes/Positions tabs exist but show only raw table columns |
| `chisurf/plugins/core/mfdb_admin/gui/client.py` (752 lines) | `MFDBClient` RPC wrapper. Has `list_samples`, `get_sample`, `save_sample`, `list_probes`, `get_probe`, `save_sample_condition`, `save_sample_key_values`, `export_sample` |
| `chisurf/plugins/core/mfdb_admin/backend/services.py` (1330 lines) | RPC handlers. `save_sample_handler` uses raw SQL, not `create_sample()`. Missing: full-description handler, export-validation handler, probe-edit handler |
| `chisurf/plugins/core/mfdb_admin/backend/measurement_services.py` (1850 lines) | Raw data, processing runs, provenance edges. Works but not sample-aware in the PRD-02 sense |
| `chisurf/core/mfdb/sample_manager.py` | `create_sample()`, `get_sample_full_description()`, `validate_sample_for_export()`, `set_sample_metadata()`, `suggest_pdbx_keys()` |
| `chisurf/core/mfdb/models.py` | `EntityDefinition`, `ProbeDefinition`, `FretPairDefinition`, `SampleDefinition`, `DEFAULT_FLUOROPHORE_SPECTRA`, `compute_forster_radius()` |
| `chisurf/core/mfdb/sample_requests.py` | `SampleCreateRequest`, `SampleUpdateRequest` — the canonical input for API/GUI sample creation |

### Current admin GUI tabs (what works vs. what's missing)

| Tab | Current state | PRD-02b target |
|-----|---------------|----------------|
| **Sample** | Form for `flr_sample` fields (id, uuid, description, num_probes, solvent, condition_id, assembly_id). New/Save/Delete/Clear. | Full `SampleDefinition` editor with entities, probes, FRET pairs. Create via `SampleCreateRequest`. Show full description. |
| **Condition** | Form for `flr_sample_condition` (pH, temp, ionic strength, buffer, details). Save/Clear. | Works — keep, minor polish |
| **Entities** | Read-only 5-column table (id, type, description, name, sequence). No add/edit/delete. | Editable table with add/delete. Show entity→probe relationships. |
| **Probes** | Read-only 8-column table (id, name, category, origin, link, abs, em, QY). No add/edit/delete. | Editable with all PRD-02 fields. Show optical properties and spectra inline. |
| **Label positions** | Read-only 9-column table (sample_probe_id, sample, probe_id, probe, entity, chain, residue, type, desc). | Editable with all flrCIF fields (atom_id, mutation_flag, modification_flag, auth_name). |
| **Metadata** | Key-value editor per sample. | Add PDBx vocabulary autocomplete from `suggest_pdbx_keys()`. |
| **Import/Export** | Import file, export sample (flrCIF), export table (CSV/XLSX). | Add export validation results display. |
| **All items** | Master table of all samples with columns. | Add full-description preview panel. |
| **FRET pairs** | **Does not exist** | NEW tab: show/add/edit Förster radius records per sample |
| **Spectra** | **Does not exist** | NEW tab or inline: show absorption/emission spectra plots per probe |

## Tasks

---

### Task 1: Wire `sample_manager` into the admin backend

**Why**: The current `save_sample_handler` (`services.py:798-869`) builds
samples via raw repository calls (`db.add_sample()`, `db.add_entity()`,
`db.clear_sample_probes()`, `db.add_sample_probe()`). It bypasses the PRD-02
`create_sample()` function entirely — which means optical properties, spectra,
FRET pairs, new position fields (atom_id, mutation_flag, modification_flag),
and vocabulary validation are all ignored on save. Similarly,
`get_sample_handler` (`services.py:267-271`) uses `db.get_sample_full()` which
only appends `key_values` to the raw `flr_sample` row — it does NOT use
`get_sample_full_description()` which returns the full nested structure with
entities, probes, positions, optical properties, and FRET pairs.

#### 1.1 New RPC handlers in `services.py`

Add the following handler functions to `services.py`. Each must follow the
existing pattern: accept `(params_dict, auth=None)`, open `MFDatabase`, call
`_require_auth`, call `sample_manager` function, return result dict.

| RPC name | Handler function | Calls | Returns |
|----------|-----------------|-------|---------|
| `mfdb.samples.full_description` | `get_sample_full_description_handler(sample_id, auth)` | `get_sample_full_description(db, sample_id)` from `sample_manager.py` | `{"description": <nested dict>}` |
| `mfdb.samples.validate_export` | `validate_sample_export_handler(sample_id, auth)` | `validate_sample_for_export(db, sample_id)` from `sample_manager.py` | `{"warnings": [...], "valid": bool}` |
| `mfdb.samples.create_structured` | `create_structured_sample_handler(sample_data, auth)` | Builds `SampleDefinition` from dict → `create_sample(db, definition)` | `{"sample_id": str, "description": <full description>}` |
| `mfdb.entities.save` | `save_entity_handler(entity, auth)` | `db.add_entity(...)` + `db.set_sequence(entity_id, seq)` (fix R15-1: currently `add_entity` accepts sequence but never stores it) | `{"entity": <entity dict>}` |
| `mfdb.entities.delete` | `delete_entity_handler(entity_id, auth)` | `db.conn.execute("UPDATE entities SET deleted_at = ... WHERE entity_id = ?")` | `{"ok": True}` |
| `mfdb.entities.list` | `list_entities_handler(sample_id=None, auth)` | If `sample_id`: entities for that sample via join. Else: all non-deleted entities. | `{"entities": [...]}` |
| `mfdb.probes.save` | `save_probe_handler(probe, auth)` | `db.find_or_add_probe(...)` with all chemical fields (probe_origin, probe_link_type, reactive_probe_flag, reactive_probe_name, center_atom) | `{"probe": <probe dict>}` |
| `mfdb.probes.optical_properties.save` | `save_probe_optical_properties_handler(probe_id, properties, auth)` | `db.set_probe_optical_property(probe_id, name, value, unit)` for each | `{"optical_properties": [...]}` |
| `mfdb.probes.positions.list` | `list_probe_positions_handler(sample_id=None, probe_id=None, auth)` | `db.get_sample_probe_mappings(sample_id)` or by probe_id | `{"positions": [...]}` |
| `mfdb.fret_pairs.list` | `list_fret_pairs_handler(sample_id, auth)` | `SELECT * FROM flr_fret_forster_radius WHERE sample_id = ?` | `{"fret_pairs": [...]}` |
| `mfdb.fret_pairs.save` | `save_fret_pair_handler(pair, auth)` | `db.add_fret_forster_radius(...)` (fix R15-6: currently ignores forster_radius_id param) | `{"fret_pair": <pair dict>}` |
| `mfdb.fret_pairs.delete` | `delete_fret_pair_handler(forster_radius_id, auth)` | `DELETE FROM flr_fret_forster_radius WHERE forster_radius_id = ?` | `{"ok": True}` |
| `mfdb.pdbx.suggest_keys` | `suggest_pdbx_keys_handler(prefix)` | `suggest_pdbx_keys(prefix)` from `sample_manager.py` | `{"keys": [...]}` |
| `mfdb.pdbx.validate_value` | `validate_pdbx_value_handler(key, value)` | `MmcifDictionary` `.validate_value()` | `{"valid": bool, "message": str}` |

**Files to modify**: `services.py` — add 14 handler functions after the
existing `delete_sample_handler` (line 872).

**Imports to add** at top of `services.py`:
```python
from chisurf.core.mfdb.sample_manager import (
    create_sample,
    get_sample_full_description,
    validate_sample_for_export,
    suggest_pdbx_keys,
)
from chisurf.core.mfdb.models import SampleDefinition, SampleCreateRequest
```

#### 1.2 Register new handlers

**File**: `services.py` → `register_services()` (line 136).

Add entries to the `{name: handler}` dict (line 136-172):
```python
"samples.full_description": get_sample_full_description_handler,
"samples.validate_export": validate_sample_export_handler,
"samples.create_structured": create_structured_sample_handler,
"entities.list": list_entities_handler,
"entities.save": save_entity_handler,
"entities.delete": delete_entity_handler,
"probes.save": save_probe_handler,
"probes.optical_properties.save": save_probe_optical_properties_handler,
"probes.positions.list": list_probe_positions_handler,
"fret_pairs.list": list_fret_pairs_handler,
"fret_pairs.save": save_fret_pair_handler,
"fret_pairs.delete": delete_fret_pair_handler,
"pdbx.suggest_keys": suggest_pdbx_keys_handler,
"pdbx.validate_value": validate_pdbx_value_handler,
```

Also add to the `sample_database.*` backward-compat alias loop (line 120-128).

**File**: `manifest.json` — add 14 entries after line 62 following the existing
`{"name": "mfdb.probes.optical_properties.get"}` pattern.

#### 1.3 Update `save_sample_handler` to delegate to `create_sample()`

**File**: `services.py:798-869`

Current flow (raw SQL):
```
sample dict → db.add_entity() → db.add_sample() → db.clear_sample_key_values()
→ db.clear_sample_probes() → db.add_sample_probe()
```

New flow:
```
sample dict → detect if structured (has 'probes' list with position fields)
  → YES: build SampleDefinition → create_sample(db, definition)
  → NO (legacy flat dict): keep current raw SQL path for backward compat
```

What the raw SQL path currently **misses** that `create_sample()` handles:
- Optical properties per probe (abs_max, em_max, QY, ext_coeff)
- Spectra data (absorption/emission arrays)
- FRET pairs (Förster radius records)
- Position fields: atom_id, mutation_flag, modification_flag, auth_name
- Entity sequence storage (R15-1 bug: `add_entity()` accepts but never stores)
- Probe chemical fields: reactive_probe_flag, reactive_probe_name, center_atom
- Vocabulary validation for entity_type, solvent_phase, probe_origin

#### 1.4 New client methods in `client.py`

**File**: `client.py` — add after `get_probe_optical_properties()` (line 108).

Add one method per new handler, following the existing pattern:
```python
def get_sample_full_description(self, sample_id: str) -> dict[str, Any]:
    return self._call("mfdb.samples.full_description", {"sample_id": sample_id}).get("description", {})

def validate_sample_export(self, sample_id: str) -> dict[str, Any]:
    return self._call("mfdb.samples.validate_export", {"sample_id": sample_id})

def create_structured_sample(self, sample_data: dict[str, Any]) -> dict[str, Any]:
    return self._call("mfdb.samples.create_structured", {"sample_data": sample_data})

def list_entities(self, sample_id: str | None = None) -> list[dict[str, Any]]:
    return self._call("mfdb.entities.list", {"sample_id": sample_id}).get("entities", [])

def save_entity(self, entity: dict[str, Any]) -> dict[str, Any]:
    return self._call("mfdb.entities.save", {"entity": entity}).get("entity", {})

def delete_entity(self, entity_id: str) -> dict[str, Any]:
    return self._call("mfdb.entities.delete", {"entity_id": entity_id})

def save_probe(self, probe: dict[str, Any]) -> dict[str, Any]:
    return self._call("mfdb.probes.save", {"probe": probe}).get("probe", {})

def save_probe_optical_properties(self, probe_id: int, properties: list[dict]) -> dict[str, Any]:
    return self._call("mfdb.probes.optical_properties.save", {"probe_id": probe_id, "properties": properties})

def list_probe_positions(self, sample_id: str | None = None, probe_id: int | None = None) -> list[dict[str, Any]]:
    return self._call("mfdb.probes.positions.list", {"sample_id": sample_id, "probe_id": probe_id}).get("positions", [])

def list_fret_pairs(self, sample_id: str) -> list[dict[str, Any]]:
    return self._call("mfdb.fret_pairs.list", {"sample_id": sample_id}).get("fret_pairs", [])

def save_fret_pair(self, pair: dict[str, Any]) -> dict[str, Any]:
    return self._call("mfdb.fret_pairs.save", {"pair": pair}).get("fret_pair", {})

def delete_fret_pair(self, forster_radius_id: int) -> dict[str, Any]:
    return self._call("mfdb.fret_pairs.delete", {"forster_radius_id": forster_radius_id})

def suggest_pdbx_keys(self, prefix: str) -> list[str]:
    return self._call("mfdb.pdbx.suggest_keys", {"prefix": prefix}).get("keys", [])

def validate_pdbx_value(self, key: str, value: str) -> dict[str, Any]:
    return self._call("mfdb.pdbx.validate_value", {"key": key, "value": value})
```

**Total**: 16 new client methods.

---

### Task 2: Overhaul the Sample tab

**Why**: The current `sample_tab()` (`tool.py:1299-1350`) is a flat
`QFormLayout` with 12 raw `flr_sample` fields. It cannot display or create
structured samples with entities, probes, positions, or FRET pairs. The
`load_sample()` method (`tool.py:2148-2187`) populates from `client.get_sample()`
which uses `db.get_sample_full()` (minimal — just flr_sample + key_values).
The `collect_sample()` method (`tool.py:2290-2354`) collects entities and
sample_probes from the GUI but misses probe chemical fields, FRET pairs, and
all new position fields (atom_id, mutation_flag, modification_flag).

#### 2.1 Restructure `sample_tab()` layout

**File**: `tool.py:1299-1350`

Replace the flat `QFormLayout` with a `QSplitter(Qt.Vertical)` containing:

1. **Top panel**: Sample header form (keep existing fields: sample_id, uuid,
   description, details, num_probes, solvent_phase, condition_id, assembly_id,
   project_id, measured_by, device, measured_at). Use `QFormLayout` as now.

2. **Bottom panel**: `QTabWidget` with sub-tabs:
   - "Entities" — inline entities editor (see 2.2)
   - "Probes & Positions" — inline probes editor (see 2.3)
   - "FRET Pairs" — inline FRET pairs editor (see 2.4)
   - "Condition" — embed the existing condition form (see 2.5)
   - "Full Description" — JSON/tree preview (see 2.6)

3. **Action buttons bar** between top and bottom panels (see 2.7).

#### 2.2 Entities sub-panel in Sample tab

**Current state**: `entities_tab()` (`tool.py:1481-1492`) creates a read-only
5-column `QTableWidget`. `fill_entities()` (`tool.py:2213-2233`) has dead code
(`if False else ""` for sequence column — line 2220). Entity data comes from
`sample.get("entities", [])` in `load_sample()` (line 2173), but the current
`get_sample_handler` returns entities from `db.get_sample_full()` which does
NOT populate entity sequences (R15-1).

**Changes needed**:
- Create `_entities_sub_panel()` method returning a `QWidget` with:
  - `QTableWidget` with columns: entity_id, name, type, sequence (truncated), details
  - Make type column a `QComboBox` delegate with values from `ENTITY_TYPES`:
    `["polymer", "non-polymer", "macrolide", "water", "branched"]`
  - "➕ Add entity" button → inserts new empty row
  - "🗑 Remove" button → removes selected row (with confirmation if linked)
- `fill_entities()` must call `client.list_entities(sample_id=current_sample_id)`
  instead of relying on `sample.get("entities")` — this uses the new handler
  which actually returns sequences
- `collect_sample()` (`tool.py:2290-2308`) must also collect `sequence` from
  the table (currently not collected)

#### 2.3 Probes & Positions sub-panel in Sample tab

**Current state**: `probes_tab()` (`tool.py:1494-1505`) shows 8 columns
(id, name, category, origin, link, abs, em, QY) — all read-only.
`fill_probes()` (`tool.py:2235-2257`) loads ALL probes from the database
globally via `client.list_probes()`, not scoped to the current sample.
`positions_tab()` (`tool.py:1507-1528`) shows 9 columns — read-only.
`fill_positions()` (`tool.py:2259-2278`) shows sample_probes but misses
atom_id, mutation_flag, modification_flag, auth_name columns.

**Changes needed**:
- Create `_probes_sub_panel()` method returning a `QWidget` with:
  - Top: Probe table with editable columns:
    - probe_name (combo from `COMMON_PROBE_NAMES` + freetext)
    - entity (combo from current entities list, maps to entity_index)
    - seq_id (int), comp_id, asym_id, atom_id
    - mutation_flag (checkbox), modification_flag (checkbox)
    - abs_nm, em_nm, QY (read-only, populated from optical_properties)
  - When a known probe name is selected from combo, auto-fill abs/em/QY
    from `DEFAULT_FLUOROPHORE_SPECTRA` dict (`models.py`)
  - "➕ Add probe" button → inserts new row
  - "🗑 Remove" button → removes selected row
  - Bottom: Selected probe optical properties detail (expandable)
- `fill_probes()` must be rewritten to call
  `client.list_probe_positions(sample_id=current_sample_id)` instead of
  global `client.list_probes()` — shows only probes attached to this sample
- `collect_sample()` (`tool.py:2309-2326`) currently collects only
  sample_probe_id, probe_id, fluorophore_type, description from positions
  table. Must add: entity_id, asym_id, seq_id, comp_id, atom_id,
  mutation_flag, modification_flag

#### 2.4 FRET Pairs sub-panel in Sample tab

**Current state**: No FRET pairs UI exists anywhere in the admin GUI.

**Changes needed**:
- Create `_fret_pairs_sub_panel()` method returning a `QWidget` with:
  - `QTableWidget` with columns: donor (combo from current probes), acceptor
    (combo from current probes), R₀ (nm), κ², n_refr, overlap_integral
  - "➕ Add pair" button → inserts new row with donor/acceptor combos
  - "🗑 Remove" button → removes selected row
  - "♻️ Recompute R₀" button → calls `compute_forster_radius()` from
    `models.py` and updates the R₀ cell (nice-to-have)
- `fill_fret_pairs()` method → calls `client.list_fret_pairs(sample_id)`
- `collect_fret_pairs()` method → collects rows from the table
- Wire into `collect_sample()` — add `"fret_pairs"` key to the returned dict

#### 2.5 Embed Condition in Sample tab

**Current state**: `condition_tab()` (`tool.py:1400-1479`) is a separate tab
with its own form (pH, temp, ionic_strength, buffer_composition, details).
Works correctly.

**Changes needed**:
- Move the condition form into the Sample tab's sub-tab widget as a
  "Condition" sub-tab
- Keep the standalone Condition tab in the main tabs as well (for quick access)
- Both views should share the same underlying widgets (or sync on load/save)

#### 2.6 Full Description preview in Sample tab

**Current state**: No way to see `get_sample_full_description()` output.

**Changes needed**:
- Create `_full_description_panel()` method returning a `QWidget` with:
  - `QPlainTextEdit` (read-only) showing JSON-formatted full description
  - "🔄 Refresh" button → calls `client.get_sample_full_description(sample_id)`
    and formats with `json.dumps(result, indent=2)`
  - "📋 Copy" button → copies JSON to clipboard
  - "✅ Validate" button → calls `client.validate_sample_export(sample_id)`
    and shows warnings in a `QListWidget` below the JSON view

#### 2.7 Update action buttons

**File**: `tool.py:1322-1331` (current button row: New, Save, Delete, Clear)

**Changes needed**:
- Keep: New, Delete, Clear
- Change "💾 Save" to call the new `create_structured_sample_handler` when
  the sample has structured data (entities/probes/fret_pairs), falling back to
  the existing `save_sample()` for legacy flat saves
- Add: "📄 Full Description" → switches to Full Description sub-tab and
  refreshes
- Add: "✅ Validate" → calls validate_export and shows inline results

#### 2.8 Update `load_sample()` to use full description

**File**: `tool.py:2148-2187`

**Current**: `self.client.get_sample(sample_id)` → flat dict from
`db.get_sample_full()`.

**Change to**: `self.client.get_sample_full_description(sample_id)` → nested
dict with entities (including sequences), probes (with optical properties and
positions), FRET pairs, condition, and key_values.

Update all `fill_*` calls to use the nested structure:
- `fill_entities(desc.get("entities", []))` — now includes sequence data
- `fill_probes_positions(desc.get("probes", []))` — new method, replaces
  separate `fill_probes()` + `fill_positions()` calls
- `fill_fret_pairs(desc.get("fret_pairs", []))` — new
- Condition and key_values populate as before

#### 2.9 Update `collect_sample()` to emit structured dict

**File**: `tool.py:2290-2354`

Currently returns a flat dict matching the raw `flr_sample` table. Must be
extended to include:

```python
{
    "sample_id": ...,
    "description": ...,
    # ... existing flat fields ...
    "entities": [
        {"entity_id": ..., "type": ..., "sequence": ..., "common_name": ...},
    ],
    "probes": [
        {
            "chromophore_name": ...,
            "entity_index": 0,
            "seq_id": ..., "comp_id": ..., "asym_id": ..., "atom_id": ...,
            "mutation_flag": ..., "modification_flag": ...,
        },
    ],
    "fret_pairs": [
        {"donor_index": 0, "acceptor_index": 1, "forster_radius": ..., "kappa_squared": ..., "n_refr": ...},
    ],
    "condition": {...},  # existing
    "key_values": [...],  # existing
}
```

---

### Task 3: Overhaul the Entities tab (standalone)

**Why**: `entities_tab()` (`tool.py:1481-1492`) is a bare 5-column read-only
`QTableWidget` with no buttons. `fill_entities()` (`tool.py:2213-2233`) has
dead code on line 2220 (`if False else ""` — the sequence column is always
empty). No add, edit, or delete capability exists.

#### 3.1 Add control buttons

**File**: `tool.py:1481-1492`

Add a button row above the table:
- "🔄 Refresh" → calls `client.list_entities()` (new handler from Task 1)
- "➕ New entity" → opens inline row or dialog with fields:
  - entity_id (text), name (text), type (combo: polymer, non-polymer,
    macrolide, water, branched), sequence (multiline `QPlainTextEdit`),
    details (text)
- "✏️ Edit" → double-click enables editing on the selected row
- "🗑 Delete" → calls `client.delete_entity(entity_id)` with confirmation
  dialog: "Entity X is referenced by N probe positions. Delete anyway?"
- "🔍 Show probes" → filters the Probes tab to show only probes attached to
  the selected entity

#### 3.2 Fix `fill_entities()` dead code

**File**: `tool.py:2213-2233`

Remove the dead code block:
```python
# CURRENT (broken):
sequence = (
    self.client.get_sample(self.sample_id_edit.text()).get("sequence", "")
    if False
    else ""
)

# REPLACE WITH:
sequence = entity.get("sequence", "")
```

This requires the new `list_entities_handler` to actually return sequence data
(which in turn depends on R15-1 fix: `add_entity()` must store sequences).

#### 3.3 Make table editable

Set `QTableWidget` edit triggers to `DoubleClicked | SelectedClicked` instead
of the implicit default. When a cell is edited and focus leaves the row, call
`client.save_entity(row_data)`.

#### 3.4 Add entity count to tab label

Show entity count in the tab title: "Entities (3)" — update on refresh.

---

### Task 4: Overhaul the Probes tab (standalone)

**Why**: `probes_tab()` (`tool.py:1494-1505`) shows 8 columns, all read-only.
`fill_probes()` (`tool.py:2235-2257`) loads ALL probes globally via
`client.list_probes()` — not filtered by the currently selected sample.
No add/edit/delete buttons. Missing PRD-02 chemical fields (reactive_probe_flag,
reactive_probe_name, center_atom) and extended optical properties.

#### 4.1 Expand table columns

**File**: `tool.py:1499-1501`

Current columns: `["id", "name", "category", "origin", "link", "abs", "em", "QY"]`

New columns: `["id", "name", "category", "origin", "link_type",
"reactive", "center_atom", "abs_nm", "em_nm", "QY", "ext_coeff"]`

#### 4.2 Add control buttons

Add button row above table:
- "🔄 Refresh" → reloads probe list
- "➕ New probe" → dialog with:
  - chromophore_name (combo from `COMMON_PROBE_NAMES` keys + freetext)
  - When known name selected, auto-fill fields from `DEFAULT_FLUOROPHORE_SPECTRA`
  - category, probe_origin, probe_link_type (combos from vocabulary)
  - reactive_probe_flag (checkbox), reactive_probe_name, center_atom
- "✏️ Edit" → double-click to modify selected probe
- "🗑 Delete" → calls appropriate delete (confirm if referenced)
- "📊 Optical properties" → expands/shows side panel with all optical
  property rows for selected probe
- "🔬 Spectra" → (nice-to-have) plots absorption/emission spectra

#### 4.3 Scope probes to sample

**File**: `tool.py:2235-2257`

`fill_probes()` currently calls `self.client.list_probes()` which returns
ALL probes in the database. Change to:
- When a sample is loaded: show only probes for that sample via
  `client.list_probe_positions(sample_id=self.current_sample_id)` and then
  fetch probe details for each unique probe_id
- When no sample is loaded: show all probes (current behavior)
- Add a "Show all / Show sample" toggle button

#### 4.4 Add probe name autocomplete

When typing a probe name, offer autocomplete from `DEFAULT_FLUOROPHORE_SPECTRA`
keys (models.py). Import the dict and populate a `QCompleter`.

---

### Task 5: New FRET Pairs tab

**Why**: No FRET pairs UI exists. The `flr_fret_forster_radius` table was
added by PRD-02 (schema.py:344) with columns: forster_radius_id, sample_id,
donor_probe_id, acceptor_probe_id, forster_radius, kappa_squared, n_refr,
overlap_integral, details. The user has no way to view, add, or edit these
records. Note: R15-3 found that `sample_id` is declared `INTEGER NOT NULL`
but `mfdb_sample.sample_id` is `TEXT PRIMARY KEY` — this type mismatch must
be fixed (schema.py:344) before this tab works correctly.

#### 5.1 Create `fret_pairs_tab()` method

**File**: `tool.py` — add after `positions_tab()` (line 1528).

Create a `QWidget` with:
- Filter bar: sample_id combo (populated from `client.list_samples()`) +
  "🔄 Refresh" button
- `QTableWidget` with columns:
  `["id", "sample", "donor", "acceptor", "R₀ (nm)", "κ²", "n", "overlap_integral", "details"]`
- Donor/acceptor columns show probe names resolved from probe_ids

#### 5.2 Add CRUD buttons

- "➕ New pair" → dialog:
  - sample_id (combo or auto-fill from current sample)
  - donor_probe (combo from probes for this sample)
  - acceptor_probe (combo from probes for this sample)
  - R₀ (nm), κ² (default 0.6667), n (default 1.33)
  - overlap_integral (optional)
- "✏️ Edit" → inline edit on double-click
- "🗑 Delete" → `client.delete_fret_pair(forster_radius_id)`
- "♻️ Recompute R₀" → (nice-to-have) calls `compute_forster_radius()` with
  the donor/acceptor probe spectra and updates

#### 5.3 Wire into `setup_ui()`

**File**: `tool.py:744` (in `setup_ui()`)

Add the FRET pairs tab to the tab creation list, between "Label positions"
and "Metadata" tabs. Register in `_all_items_sources` for the All Items
master table.

#### 5.4 Create `fill_fret_pairs()` method

Calls `client.list_fret_pairs(sample_id)`. For each pair, resolve
`donor_probe_id` and `acceptor_probe_id` to probe names for display.

---

### Task 6: Overhaul the Label Positions tab

**Why**: `positions_tab()` (`tool.py:1507-1528`) shows 9 columns but is
missing the flrCIF position fields added by PRD-02: atom_id, mutation_flag,
modification_flag, auth_name (schema.py, `flr_poly_probe_position` table,
line 243). These columns exist in the database but are not displayed.
`fill_positions()` (`tool.py:2259-2278`) reads from `sample_probes` data
but does not include these fields. The tab is entirely read-only.

#### 6.1 Expand table columns

**File**: `tool.py:1513-1525`

Current columns:
```python
["sample_probe_id", "sample", "probe_id", "probe", "entity", "chain", "residue", "type", "description"]
```

New columns:
```python
["sample_probe_id", "sample", "probe_id", "probe", "entity", "chain",
 "residue", "atom_id", "mutation", "modification", "auth_name", "type", "description"]
```

#### 6.2 Update `fill_positions()` to include new fields

**File**: `tool.py:2259-2278`

Add to the values list:
```python
mapping.get("atom_id", ""),
mapping.get("mutation_flag", ""),
mapping.get("modification_flag", ""),
mapping.get("auth_name", ""),
```

This requires the backend to return these fields — the new
`list_probe_positions_handler` (Task 1) must join `flr_poly_probe_position`
to include them.

#### 6.3 Add control buttons

Add button row above table:
- "🔄 Refresh" → reloads positions for current sample
- "➕ New position" → dialog:
  - probe (combo from probes list), entity (combo from entities list)
  - seq_id (int), comp_id (text), asym_id (text, default "A")
  - atom_id (text), mutation_flag (checkbox), modification_flag (checkbox)
  - auth_name (text), fluorophore_type (combo: intrinsic, extrinsic, unspecified)
- "✏️ Edit" → double-click to modify
- "🗑 Delete" → removes the sample_probe record

#### 6.4 Make table editable

Set edit triggers to `DoubleClicked`. When a cell is edited, collect the row
and call `client.save_probe()` + position update via the backend.

---

### Task 7: Full-description preview panel

**Why**: `get_sample_full_description()` returns a rich nested dict with
all entities, probes (with optical properties and positions), FRET pairs,
condition, and key-values. There is no way to view this output in the GUI.
The All Items tab (`tool.py:1079-1130`) has a master table with type/id/label/
details columns and double-click navigation, but no preview panel for the
selected item's full structure.

#### 7.1 Add preview panel to All Items tab

**File**: `tool.py:1079-1130` (`all_items_tab()`)

Wrap the existing table in a `QSplitter(Qt.Horizontal)`:
- Left: existing `all_items_table` (keep as-is)
- Right: preview panel widget with:
  - `QPlainTextEdit` (read-only, monospace font) for JSON display
  - "📋 Copy JSON" button → copy to clipboard
  - "✅ Validate" button → calls `validate_sample_export` for sample items

#### 7.2 Wire selection to preview

Connect `all_items_table.currentItemChanged` signal to a method that:
1. Gets the selected item's type and id from `UserRole` payload
2. If type == "sample": calls `client.get_sample_full_description(id)` and
   displays as formatted JSON in the preview panel
3. If type == other: calls the appropriate get handler and displays raw dict

#### 7.3 Standalone Full Description viewer (alternative)

If the All Items splitter approach is too cramped, add a "Full Description"
tab to the main tab bar (via `setup_ui()`). This tab would have:
- Sample selector (combo from `client.list_samples()`)
- Full JSON view with tree structure
- Validate + Copy buttons
- This overlaps with Task 2.6 — decide which location is primary

---

### Task 8: Metadata tab enhancements

**Why**: The Metadata tab (`tool.py:1559-1580`) has a `MetadataEditor` widget
that shows key-value pairs per sample. It works for manual entry but has no
autocomplete for PDBx/flrCIF standard keys (from `suggest_pdbx_keys()` in
`sample_manager.py`) and no validation of values against the mmCIF dictionary.

#### 8.1 Add PDBx key autocomplete

**File**: `tool.py:1559-1580` (or in the `MetadataEditor` widget class)

When the user types in a key field, call `client.suggest_pdbx_keys(prefix)`
(new handler from Task 1) and populate a `QCompleter` with the results.

Implementation:
- In `MetadataEditor`, override the key cell delegate to use a `QLineEdit`
  with a `QCompleter`
- The completer should fire on 3+ characters typed
- Show the full PDBx key path (e.g., `_flr_sample.sample_id`) as completion

#### 8.2 Add value validation on save

**File**: `tool.py` → `save_sample_metadata()` method

Before saving, for each key-value pair:
1. Call `client.validate_pdbx_value(key, value)` (new handler)
2. If invalid, show a warning icon next to the row and collect all warnings
3. Show a summary dialog: "2 of 15 values failed validation" with details
4. Allow the user to save anyway (warnings, not errors)

#### 8.3 Group metadata by category

In `fill_metadata()`, sort key-values by prefix:
- `flr.*` → "FLR Standard" group header
- `pdbx.*` → "PDBx" group header
- `chisurf.*` → "ChiSurf" group header
- Other → "Custom" group header

Use `QTableWidget` section headers or row-spanning cells to visually separate
groups.

---

### Task 9: Import/Export tab enhancements

**Why**: The Import/Export tab (`tool.py:1530-1557`) has Import, Export Sample,
and Export Table buttons with a preview text area. It doesn't show what was
created on import, doesn't validate before export, and doesn't preview the
CIF output.

#### 9.1 Show import summary

**File**: `tool.py` → `import_file()` method

After `client.import_file(path)` returns, the result dict should contain
counts of created records. Display in the `preview_edit` text area:

```
Import complete:
  Samples: 2 created
  Entities: 4 created
  Probes: 3 created
  Positions: 6 created
  FRET pairs: 1 created
  Conditions: 2 created
```

This may require updating `import_file_handler` in `services.py` to return
these counts (currently returns a generic summary dict).

#### 9.2 Pre-export validation

**File**: `tool.py` → `export_selected_sample()` method

Before calling `client.export_sample()`:
1. Call `client.validate_sample_export(sample_id)`
2. If warnings exist, show dialog:
   "The following issues were found:\n• warning 1\n• warning 2\n\nExport anyway?"
3. On "Yes" → proceed with export. On "No" → cancel.

#### 9.3 Add "Preview flrCIF" button (nice-to-have)

Add a button between Export and the preview area:
- "👁 Preview CIF" → calls export with a temp path, reads the CIF file,
  displays content in `preview_edit`
- User can review the CIF text before committing to a file save

## Implementation notes

### Backend architecture

The admin plugin currently uses a client/server split with RPC:
- GUI (`tool.py`) → `MFDBClient` (`client.py`) → RPC → handlers (`services.py`)
- Handlers open `MFDatabase` and call repository methods

For PRD-02b, the handlers should call `sample_manager` functions (not raw
SQL). This means:

```python
# OLD (services.py:798)
def save_sample_handler(sample, auth=None):
    db = _get_db()
    db.add_sample(sample["sample_id"], ...)  # raw repository call

# NEW
def create_structured_sample_handler(sample_data, auth=None):
    db = _get_db()
    request = SampleCreateRequest(**sample_data)
    sample_id = create_sample(db, request.to_sample_definition())
    return {"sample_id": sample_id, "sample": get_sample_full_description(db, sample_id)}
```

### GUI patterns

The existing admin GUI uses:
- `QtWidgets.QTableWidget` for all tables (not QTableView/model)
- `QtWidgets.QFormLayout` for detail forms
- `DockArea` for tab management
- `_install_table_context_menu` for check/uncheck/delete patterns
- Status updates via `self.status_label.setText(...)`

Follow these patterns. For new sub-panels in the Sample tab, use
`QSplitter` or `QTabWidget` to organize entities/probes/pairs/condition
sections without making the tab overwhelming.

### In-process mode

`MFDBClient` supports in-process mode (`inprocess=True`) which creates a
`ServiceDispatcher` directly — no ZMQ server needed. This is the primary
mode for local use. New handlers must be registered in `register_services()`
to work in both modes.

## Prerequisites (R15 fixes required before starting)

These open review findings from CODE_REVIEW.md must be fixed first, as they
affect the correctness of the data the admin GUI will display and edit:

| Finding | Issue | Blocking task |
|---------|-------|---------------|
| R15-1 | `add_entity()` accepts `sequence` but never stores it | Task 1.1 (entities.save), Task 3.2 |
| R15-2 | `asym_id="A"` default makes `not self.asym_id` always False, breaking legacy chain_id and DEFAULT_FLUOROPHORE_SPECTRA lookups | Task 2.3 (probe auto-fill), Task 4.4 |
| R15-3 | `flr_fret_forster_radius.sample_id` is `INTEGER` but `mfdb_sample.sample_id` is `TEXT` — FK mismatch | Task 5 (FRET pairs tab) |
| R15-6 | `add_fret_forster_radius()` ignores `forster_radius_id` param | Task 5.2 (edit pair) |
| R15-7 | `__init__.py` missing exports for new PRD-02 symbols | Task 1.1 (imports) |

## Definition of Done

### Must-have (MVP)

- [ ] **Task 1 complete**: All 14 new RPC handlers registered and working (1.1-1.4)
- [ ] **Task 1.3 complete**: `save_sample_handler` delegates to `create_sample()` for structured data
- [ ] **Task 2.8 complete**: `load_sample()` uses `get_sample_full_description()` output
- [ ] **Task 2.9 complete**: `collect_sample()` emits structured dict with entities/probes/fret_pairs
- [ ] **Task 3 complete**: Entities tab — view, add, edit, delete; sequence column populated
- [ ] **Task 4.1-4.3 complete**: Probes tab — expanded columns, sample-scoped, add/edit/delete
- [ ] **Task 5 complete**: FRET pairs tab — view, add, delete
- [ ] **Task 6 complete**: Label positions — all flrCIF fields visible and editable
- [ ] **Task 7 complete**: Full description preview visible for any selected sample
- [ ] **Task 8.1 complete**: Metadata tab has PDBx key autocomplete
- [ ] All new RPC handlers have tests (see test list below)

### Nice-to-have (post-MVP)

- [ ] Spectra plots (absorption/emission) per probe (Task 4.2 "🔬 Spectra")
- [ ] Auto-compute R₀ from spectral overlap in FRET pairs editor (Task 5.2 "♻️ Recompute R₀")
- [ ] Probe name combo auto-fills photophysical properties from DEFAULT_FLUOROPHORE_SPECTRA (Task 4.4)
- [ ] flrCIF export preview before writing file (Task 9.3)
- [ ] Import summary showing created record counts (Task 9.1)
- [ ] Condition form embedded in Sample tab as sub-tab (Task 2.5)
- [ ] Metadata grouped by category prefix (Task 8.3)
- [ ] Metadata value validation on save (Task 8.2)
- [ ] Keyboard shortcuts for common actions (Ctrl+N new, Ctrl+S save)

### Tests

Test file: `chisurf/plugins/core/mfdb_admin/test/test_admin_handlers.py`

```python
# --- Task 1 handler tests ---

def test_full_description_handler(db):
    """mfdb.samples.full_description returns nested dict with entities,
    probes (with optical_properties and positions), fret_pairs, condition,
    key_values."""

def test_validate_export_handler_incomplete(db):
    """mfdb.samples.validate_export returns warnings list for sample
    missing required fields (no probes, no condition)."""

def test_validate_export_handler_complete(db):
    """mfdb.samples.validate_export returns empty warnings for fully
    populated sample."""

def test_create_structured_handler(db):
    """mfdb.samples.create_structured creates sample with entities, probes,
    positions, and FRET pairs via SampleDefinition. Verify with
    get_sample_full_description."""

def test_create_structured_handler_auto_fills_spectra(db):
    """mfdb.samples.create_structured with known probe name auto-populates
    optical properties from DEFAULT_FLUOROPHORE_SPECTRA."""

def test_entity_list_handler(db):
    """mfdb.entities.list returns entities scoped to sample_id when provided,
    all non-deleted when sample_id is None."""

def test_entity_save_handler_with_sequence(db):
    """mfdb.entities.save persists entity with sequence in entity_poly_seq
    table (verifies R15-1 fix)."""

def test_entity_delete_handler(db):
    """mfdb.entities.delete soft-deletes entity."""

def test_probe_save_handler_all_fields(db):
    """mfdb.probes.save persists all chemical fields: probe_origin,
    probe_link_type, reactive_probe_flag, reactive_probe_name, center_atom."""

def test_probe_optical_properties_save_handler(db):
    """mfdb.probes.optical_properties.save creates/updates abs_max, em_max,
    QY, ext_coeff for a probe."""

def test_probe_positions_list_handler(db):
    """mfdb.probes.positions.list returns positions with atom_id,
    mutation_flag, modification_flag, auth_name fields."""

def test_fret_pair_crud_handlers(db):
    """mfdb.fret_pairs.{list,save,delete} round-trip: create pair with
    donor/acceptor/R0/kappa/n, list by sample, delete by id."""

def test_pdbx_suggest_handler(db):
    """mfdb.pdbx.suggest_keys returns matching keys for prefix '_flr'."""

def test_pdbx_validate_handler(db):
    """mfdb.pdbx.validate_value returns valid=True for conforming values,
    valid=False with message for violations."""

# --- Task 1.3 backward compat test ---

def test_save_sample_handler_legacy_compat(db):
    """save_sample_handler still works with flat legacy dict (no probes/
    entities/fret_pairs keys) — backward compatibility."""

def test_save_sample_handler_structured(db):
    """save_sample_handler with structured dict (probes/entities/fret_pairs)
    delegates to create_sample() and persists all fields."""
```
