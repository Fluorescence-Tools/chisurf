# PRD-43: Align GUI Operation History with MFDB Provenance

> **Scope:** Inspection of the current history mechanism + incremental alignment
> plan.  Phase 1 is shippable now (no new dependencies); Phases 3–4 ride
> PRD-21/27 and are deferred until the architecture track lands.

## ⚑ IMPLEMENTED (2026-06-27) — history is now a projection over MFDB

The original Phase-1 ("store history as an opaque `history.jsonl` blob") was
**superseded** by a stronger design: **MFDB is the durable source of truth and the
in-memory `OperationHistory` is a projection over it.**  Delivered in four staged,
headless-tested increments (all degrade to a no-op without a database):

1. **Headless replay** — the checkpoint/delta merge moved out of the Qt layer
   into Qt-free `chisurf/history/projection.py` (`DomainState` +
   `build_target_state`); `main_helper.py` is a thin adapter.
2. **Vocabulary alignment** — the controlled action vocabulary + the coarse
   `action→operation_type` map are declared **in the dictionary**
   (`_mfdb_event_log.action_type` enumeration + `_item_enumeration.detail`) and
   read at runtime; `canonical()` (pure separator syntax) replaces the fragile
   `resolve_name` munging.  Fixed a latent model-state replay bug.
3. **Bounded snapshots** — `OperationHistory._checkpoints` is capped + evicts
   (keep earliest + most-recent N); evicted snapshots are rebuildable by replay.
4. **MFDB persistence as projection** — new `mfdb_event_log` table (authored in
   `mfdb_flr_ext.dic`, materialized by `reconcile_schema`).  `record()` dual-writes
   best-effort; `list_events(source="mfdb")` reads the durable log;
   `chisurf/core/mfdb/event_log.py` is the append-only writer/reader.
   `archive_project_to_mfdb` stamps events with `project_id`;
   `restore_project_from_artifacts` returns them under `extra.history_events`
   (the existing `.csp` consumer seam) — **this subsumes the Phase-1 goal**: MFDB
   save/restore no longer loses history.

**Still deferred (Phase 2/3):** auto-creating `mfdb_operation` rows from coarse GUI
events inside the dispatcher (the `operation_id` column + dic mapping are ready;
the linking belongs with the PRD-11/16 operation-creation channels).

## ⚑ UNDO/REDO via the History Browser (2026-06-27) — now functional

Making the projection record again exposed that the **interactive undo/redo**
replay path (history-browser cursor → `_on_history_cursor_changed` →
`build_target_state` → `sync_domain_entities` + `_apply_*`) had **never actually
run**: nothing recorded, so it was never exercised. Three stacked, latent bugs,
each hidden by the previous one, are now fixed (commits `d409c8c2`, `9f205d89`):

1. **Recording was dead** — `chisurf.history` (the singleton attribute) was shadowed
   by the `chisurf/history/` *subpackage*: once any `from chisurf.history import …`
   ran, `cs.history` resolved to the module, so `record_action` / the dispatcher
   silently no-op'd. Fixed by housing the singleton in
   `chisurf/history/__init__.py` (`get_history()` + a delegating module
   `__getattr__`); `chisurf/__init__.py` returns that instance. The collision is now
   benign whether `cs.history` resolves to the module or the instance.
2. **Undo crashed** — `replay.py` had no module-level `import chisurf as cs`, so
   `sync_domain_entities` raised `NameError` on its first line, swallowed by a bare
   `except` in the cursor handler → undo appeared to "do nothing". Fixed; the
   handler now *logs* replay-apply failures instead of swallowing them.
3. **Redo didn't re-create fits** — `sync_domain_entities` keyed its `creation_map`
   for `fit.add` on `target_uid`, while `reconstruct_navigation_state` reads the fit
   UID from `source_uid`; the lookup always missed. Fixed (key on `source_uid`).

**UID remapping (redo of UID-keyed state).** Re-creating an entity on redo yields a
*fresh* UID, but `model_state` is keyed by the recorded (old) fit-group UID.
`sync_domain_entities` now returns a `{old_uid: new_uid}` map for re-created fits;
the cursor handler rewrites `model_state` keys through it before applying.
Parameter and fit-range state are **name-keyed** (fit-group name is stable) and need
no remap, so they already restore.

### Remaining undo/redo work (tracked)

- **Local-fit UID remap.** Only the top-level fit-group UID is remapped; the
  `local_fit_uid` keys *inside* `model_state` are not, so deep multi-local-fit
  *model-component* redo may be partial.
- **Dataset UID remap.** Only re-created fits are captured in the remap; re-created
  datasets also get fresh UIDs (datasets mostly resolve by name elsewhere, so this
  is usually harmless — verify).
- **Dataset re-add fidelity.** Redo re-adds datasets by dispatching `dataset.add`
  with the recorded payload (`experiment_reader` forced to `None`); reload fidelity
  for all reader types is unverified.
- **Duplicate recording.** Every operation records *twice* with inconsistent vocab:
  `dataset_add` / `fit_add` (from `core_fit._record_history`, rich payload with
  UIDs) **and** `dataset.add` / `fit.add` (from the dispatcher, payload without
  UIDs). Reconstruction tolerates it (canonicalized + reads the rich one) but it is
  noisy and ambiguous — consolidate to a single record path per operation.
- **Multi-entity remap ordering.** The fit remap assumes the last-added fit
  corresponds to the missing UID being replayed; correct for the common single-fit
  case, best-effort for batch re-creation.
- **GUI-window refresh.** When a fit closes/opens during replay, confirm the fit
  windows / model layout refresh (the apply mutates `cs.fits`; widget sync is the
  GUI adapter's job).

**Status:** recording ✓, undo ✓, redo re-creates fits + restores name-keyed state +
remaps model-state fit groups ✓. Not yet exhaustively tested in the live GUI; the
items above are the known edges.

The sections below are the original analysis that motivated this work.

## Goal

Preserve the interactive client-side undo/redo session history
(`OperationHistory`) across database save/restore cycles, and incrementally
align the GUI event stream with the backend's provenance model.

Today, saving a project to a local `.csp` archive preserves the full action
history. Saving to MFDB (Project Browser) discards it entirely — the user
loses their undo stack and the scientific exploration trail.

## Current State Analysis

### 1. Client-Side GUI History (`OperationHistory`)

Implemented in [chisurf/history/core.py](file:///Users/tpeulen/dev/chisurf/chisurf/history/core.py);
navigated via `HistoryBrowserWidget`.

**Action dispatch loop:**
GUI and CLI operations are decorated with `@action`
([_decorator.py](file:///Users/tpeulen/dev/chisurf/chisurf/core/actions/_decorator.py)).
When executed via the `ActionDispatcher`
([_infra.py](file:///Users/tpeulen/dev/chisurf/chisurf/core/actions/_infra.py)),
each action is recorded to the global `cs.history` singleton.

**In-memory event log:**
Each event is a JSON-serializable dict:

```json
{
  "event_id": "uuid",
  "timestamp": "ISO-8601-UTC",
  "action_type": "dataset.add | parameter.value | fit.add | …",
  "source_uid": "optional-uuid",
  "target_uid": "optional-uuid",
  "payload": { … },
  "summary": "human-readable"
}
```

**Checkpoint snapshots:**
Every *N* events (default 50), `capture_domain_snapshot`
([replay.py](file:///Users/tpeulen/dev/chisurf/chisurf/history/replay.py))
dumps a full domain state: navigation state, parameter values, fit ranges,
setup config, model component states.  Snapshots are kept in-memory in a dict
keyed by event index.

**Undo/redo via replay:**
When the cursor moves backward
([main_helper.py:1676](file:///Users/tpeulen/dev/chisurf/chisurf/gui/main_helper.py#L1676)):

1. `get_events_from_checkpoint()` finds the nearest checkpoint ≤ target index.
2. The checkpoint snapshot is applied to restore baseline state.
3. Delta events are replayed through `reconstruct_parameter_state`,
   `reconstruct_model_state`, etc. to reach the exact target point.

This replay path is tightly coupled to the GUI widget tree — `main_helper.py`
directly manipulates combo boxes, selectors, and fit windows.

### 2. File-Based Persistence (`.csp` archives)

In [core_fit.py](file:///Users/tpeulen/dev/chisurf/chisurf/macros/core_fit.py):

- **Save:** `_write_history_snapshot_to_archive()` serializes `cs.history` to
  `history.jsonl` inside the zip archive.
- **Load:** `load_project_payload()` reads `history.jsonl` from the extracted
  archive and calls `cs.history.load_events(events, replace=True)`.

Result: full round-trip — history survives save/load.

### 3. Database Persistence (MFDB Project Browser)

In [project_archiver.py](file:///Users/tpeulen/dev/chisurf/chisurf/core/mfdb/project_archiver.py):

- **Save (`archive_project_to_mfdb`):** decomposes the project into relational
  artifacts (datasets, fits, parameters, edges).  **The `OperationHistory` is
  not touched** — no history artifact is created.
- **Restore (`restore_project_from_artifacts`):** returns datasets, fits,
  chinet sessions, parameters, and dependency edges.  **No history events** are
  included in the restored payload.

Result: history is silently lost on every database round-trip.

---

## Design Gap Summary

| Concern | `.csp` archive | MFDB database |
|---------|----------------|---------------|
| History preserved? | ✅ yes (`history.jsonl` in zip) | ❌ no (not stored) |
| Undo/redo works after restore? | ✅ yes | ❌ no (empty history) |
| Checkpoints preserved? | ❌ no (re-captured at runtime) | ❌ no |

**Additional mismatches:**

1. **Divergent event vocabularies:** GUI uses `dataset.add`, `parameter.value`,
   `fit.add`; MFDB uses `measurement_import`, `local_fit`, `derived_from`.
   PRD-27 introduces `mfdb_state_transition` as a separate append-only log.
   There is no mapping between the two vocabularies.
2. **Checkpoint size:** snapshots store the full domain state as in-memory
   dicts.  For large projects these grow unbounded — no eviction, no
   compression, no streaming to disk.

---

## Proposed Alignment (Phased)

### Phase 1 — History Artifact (shippable now)

**Dependencies:** PRD-03 object store + artifact registration (already done).

**Goal:** Stop losing history on MFDB save/restore.

**Mechanism:**

1. In `archive_project_to_mfdb()`, serialize `cs.history.list_events()` to a
   JSONL byte string.
2. `db.put_object(data=jsonl_bytes, …)` → content-addressed blob.
3. `db.register_artifact(artifact_kind="project_history", role="history_log")`.
4. Create a `project_contains` edge from the version operation to this artifact.
5. In `restore_project_from_artifacts()`, detect `project_history` artifacts,
   read the JSONL stream, and return it in the payload under
   `extra.history_events`.
6. In `load_project_payload()` (the consumer), load history events into
   `cs.history` when the `extra.history_events` key is present — the same path
   already exists for `.csp` archives.

**Not in scope for Phase 1:** checkpoints are not persisted (they are cheap to
re-create at runtime); the event vocabulary is not changed.

### Phase 2 — Vocabulary & ID Alignment (after PRD-11/16 transformer contract)

**Dependencies:** PRD-11/16 (operation-node abstraction + transformer contract).

Ensure GUI events carry database entity IDs where they overlap:

- `dataset.add` payloads include `artifact_id` when the dataset was registered.
- `fit.run.finish` payloads include `operation_id` of the MFDB operation.
- Document a mapping table between client-side `action_type` names and backend
  `operation_type` / `mfdb_state_transition` event types.

### Phase 3 — Transactional GUI↔DB Operations (after PRD-21/27)

**Dependencies:** PRD-21 (lineage API + event model), PRD-27 (append-only core).

Heavy GUI operations (fit runs, dataset imports) write an MFDB operation record
in real-time rather than only at project-archive time.  Lightweight operations
(parameter slider moves) remain client-only and are reconciled on save.

### Phase 4 — Unified Replay (long-term, after PRD-22 pipeline engine)

**Dependencies:** PRD-22 (pipeline/workflow engine), PRD-27 branches.

Replace client-side checkpoint snapshots with database version pointers.
Moving the history cursor checks out a database branch head; replay uses
PRD-21 replayable compute specs instead of custom Python reconstruction
functions.  This is the end-state where GUI history and DB provenance are the
same thing — speculative and deferred until the architecture track is stable.

---

## Tasks

1. **[Phase 1]** Update `archive_project_to_mfdb()` to serialize and store the
   history log as a `project_history` artifact.
2. **[Phase 1]** Update `restore_project_from_artifacts()` to extract
   `project_history` artifacts and return them in the payload.
3. **[Phase 1]** Add a round-trip test: archive a project with *N* history
   events → restore → verify all *N* events are present in `cs.history`.
4. **[Phase 2]** Document the action-type ↔ operation-type vocabulary mapping.
5. **[Phase 2]** Enrich GUI event payloads with MFDB entity IDs where available.

---

## Definition of Done

- [ ] Project save/restore via MFDB (Project Browser) preserves the full
      undo/redo history stack.
- [ ] History is stored as a content-addressed `project_history` artifact —
      no new database tables required.
- [ ] Round-trip test passes: archive with history → restore → history intact.

## Definition of Clean

History persistence uses the existing object store and artifact registration
APIs (no parallel storage path).  No GUI logic in the archiver.  History events
are opaque JSONL blobs to the database — the database does not parse or index
individual events.

## Relationship

- **PRD-01** (fix roundtrip): PRD-43 Phase 1 closes the remaining gap — `.csp`
  round-trips already work; MFDB round-trips do not.
- **PRD-03** (result registry / object store): Phase 1 uses the object store +
  artifact registration infrastructure that PRD-03 delivered.
- **PRD-21** (lineage API + event model): Phase 3 aligns GUI events with the
  backend event bus.
- **PRD-27** (append-only provenance core): Phase 4 folds GUI history into
  the append-only log as the shared source of truth.
