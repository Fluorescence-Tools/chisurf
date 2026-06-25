# PRD-35: Light Path Optical Presets live in MFDB

## Status

Proposed — the light path simulator currently stores optical path presets as
raw JSON files on disk (`~/.chisurf/presets/lightpath_optical/`). This PRD
defines the migration into MFDB. The JSON-on-disk code path is the approved
interim; this document captures the target state and phases.

## Goal

Make every optical path preset — the complete graph (laser lines, dichroics,
splitters, bandpass filters, detector QE curves, sample dyes with QY/EC) — a
first-class, queryable, provenance-tracked MFDB entity. Once stored, a preset
can be:

- Shared across users on the same machine (single object store)
- Versioned and linked to the study/experiment/sample that used it
- Queried: "which experiments used this dichroic / this dye pair?"
- Reused from the detector wizard or easy mode without filesystem access
- Cleaned up via the existing object-store refcount mechanism

## Non-goals

- **Replacing the Full Simulator node graph** — the `GraphDef` dict (nodes +
  edges) remains the canonical serialization. The MFDB entity stores the graph
  payload, not a decomposed optical model.
- **Structuring every component as a separate row** — PRD-08 already defines a
  detailed optical channel schema (`mfdb_optical_channel`) for instrument
  settings. This PRD is about *presets* (named, reusable configurations), not
  about breaking each component into relational columns. The two schemas are
  complementary: a preset's graph can be *exported* into the PRD-08 instrument
  schema when the preset is *applied* to a measurement.
- **Decomposing the graph into SQL** — the graph stays opaque to SQL queries
  (`mfdb_object` blob). Metadata tags enable queryability.

## Context: What Exists Today

| Component | Storage | Notes |
|-----------|---------|-------|
| Optical path presets | `~/.chisurf/presets/lightpath_optical/*.json` | Full `GraphDef` dicts saved by the Full Simulator |
| Easy-mode last config | `~/.chisurf/settings/lightpath_easy_last.json` | Preset name + params |
| Probe catalogue (spectra) | MFDB (`get_probes_info()`) | Already in MFDB; the light path reads from it |
| Object store | `~/.chisurf/objects/{md5[0:2]}/{md5[2:4]}/{md5}` | Content-addressed blob store, shared across users |

The interim JSON path is clean and functional. The only missing piece is
MFDB integration.

## Proposed Schema

### New table: `mfdb_optical_preset`

| Column | Type | Description |
|--------|------|-------------|
| `preset_id` | INTEGER PK | Auto-increment |
| `name` | TEXT NOT NULL UNIQUE | Human-readable name, e.g. "4ch-smFRET 488/561/640" |
| `description` | TEXT | Free-text notes |
| `graph_object_uuid` | TEXT NOT NULL | UUID of the `mfdb_object` row holding the `GraphDef` JSON |
| `metadata` | TEXT (JSON) | Key-value tags: `laser_lines`, `dye_probes`, `detector_count`, `has_polarizer`, etc. |
| `owner_user_id` | INTEGER FK → mfdb_user | Who created/saved this preset |
| `created_at` | TEXT (ISO-8601) | |
| `updated_at` | TEXT (ISO-8601) | |
| `superseded_by` | INTEGER FK → mfdb_optical_preset.preset_id | NULL = active; non-NULL = replaced |

The graph payload is stored in the existing object store via
`mfdb.objects.put`. The `graph_object_uuid` column holds the object UUID that
resolves to the MD5-addressed blob.

### Metadata tags (stored in the `metadata` JSON column)

These enable filtered queries without parsing the graph blob:

| Tag | Example | Purpose |
|-----|---------|---------|
| `laser_lines` | `["488", "561", "640"]` | Find presets compatible with a given laser box |
| `dye_probe_ids` | `[10, 11, 15]` | Find presets that use these dyes |
| `emission_splitters` | `2` | Number of cascaded dichroic/polarizer stages |
| `detector_count` | `4` | Number of detector channels |
| `has_polarizer` | `true` | Whether the optical path includes a polarizer splitter |
| `excitation_dichroic_probe_id` | `123` | Which dichroic reflects the laser |
| `tags` | `["smFRET", "PIE", "ALEX"]` | Arbitrary user tags |

### No new table for cached simulation results

Cached crosstalk matrices and Förster radii are ephemeral — they change when
the dye profile changes. They stay in the auto-saved JSON
(`lightpath_easy_last.json`) or can be stored as a separate
`mfdb_artifact` linked to the preset + dye-profile combination if needed.

## How It Connects to Existing Entities

```
mfdb_optical_preset ──graph_object_uuid──▶ mfdb_object (content-addressed GraphDef)
        │
        ├──owner_user_id──────────────────▶ mfdb_user
        │
        └── (used_by) ───────────────────▶ mfdb_setup / flr_inst_setting
                                           (via a future link column or a
                                            join table: mfdb_optical_preset_use)
```

The `mfdb_optical_preset_use` join table (future) records which preset was
active when a measurement was taken, creating the provenance chain:

```sql
CREATE TABLE mfdb_optical_preset_use (
    id INTEGER PRIMARY KEY,
    preset_id INTEGER NOT NULL REFERENCES mfdb_optical_preset(preset_id),
    experiment_id INTEGER REFERENCES flr_experiment(experiment_id),
    setup_id INTEGER REFERENCES mfdb_setup(setup_id),
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

## RPC Endpoints

### `mfdb.optical_presets.list` → `list[{id, name, description, metadata}]`

List all active (non-superseded) presets. Supports optional tag filters:
`laser_lines`, `dye_probe_ids`, `detector_count`.

### `mfdb.optical_presets.get` → `{id, name, description, graph, metadata, ...}`

Fetch a preset by ID or name. Returns the full `GraphDef` dict from the
object store.

### `mfdb.optical_presets.put` → `{preset_id}`

Save a preset. Accepts `{name, description, graph (dict), metadata (dict)}`.
Stores the graph in the object store (deduplicated by MD5), creates/updates
the `mfdb_optical_preset` row.

### `mfdb.optical_presets.delete`

Soft-delete (set `superseded_by` to the next version or a tombstone).

### `mfdb.optical_presets.list_tags` → `list[str]`

Return the union of all values for a given metadata key across active presets
(e.g., all distinct `laser_lines` values). Powers autocomplete in the easy-mode
combo box.

## Migration Path (Phased)

### Phase 0 (current — done)

JSON files on disk. Full Simulator saves to `~/.chisurf/presets/lightpath_optical/`.
Easy mode reads from there.

### Phase 1 — MFDB write path (next)

- Backend: add `mfdb.optical_presets.put` RPC handler. Writes graph to object
  store, creates `mfdb_optical_preset` row, extracts metadata tags from the
  graph.
- Full Simulator: "Save as Optical Path Preset…" calls
  `mfdb.optical_presets.put` instead of (or in addition to) writing a JSON file.
- Easy mode: populate the path combo from `mfdb.optical_presets.list` instead
  of scanning the filesystem.
- CLI: `imp-tricks lightpath ...` can read/write presets from MFDB.

### Phase 2 — Provenance (future)

- Add `mfdb_optical_preset_use` join table.
- When a detector wizard setup is saved, link to the active preset.
- Expose in the experiment browser: "show me all measurements that used the
  4-channel FRET preset".

### Phase 3 — Offline fallback (future)

- When MFDB is not available, fall back to the JSON-on-disk path
  transparently. The easy mode tries MFDB first, falls back to `glob()`.

## Backward Compatibility

- The JSON path is not removed. Old presets on disk are readable regardless
  of MFDB state.
- When MFDB is connected, users can import on-disk JSON presets into MFDB
  via a "Migrate…" button in the Full Simulator or easy mode.

## Open Questions

1. Should `mfdb_optical_preset_use` be a separate PRD (linked to the
   instrument/setup overhaul in PRD-08), or rolled into this one?
2. Should the easy-mode last-config auto-save also go to MFDB (as a "most
   recently used" preference on the user row)?
3. Do we need a permission model for presets (read-only shared presets vs.
   user-private)? Current JSON presets are world-readable on a multi-user
   machine.
