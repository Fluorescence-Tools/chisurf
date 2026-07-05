---
type: Subsystem
title: Operation History
description: Append-only recording of user/scripted actions with headless replay, projected durably into the MFDB provenance store.
resource: chisurf/history/
tags: [history, core, provenance]
timestamp: '2026-07-05T00:00:00Z'
---

# Scope

`chisurf/history/` records every traceable state change as an ordered event log
and can reconstruct the scientific state at any cursor position (undo/replay),
independently of Qt.

| Module | Role |
| --- | --- |
| `core.py` | `OperationHistory` — append-only log, subscribers, checkpoints |
| `projection.py` | `DomainState`, `build_target_state` — pure replay projection |
| `replay.py` | `reconstruct_*_state` / `capture_domain_snapshot` deltas |

The package name doubles as the process-wide singleton `cs.history` (a shim in
`__init__.py` reconciles the collision).

# Recording

State changes call `record_action(...)`
(`chisurf/core/actions/_infra.py`), which forwards to `cs.history.record(...)`.
So macros, the [action layer](/architecture/action-layer.md), and GUI clicks all
produce identical events `{action_type, summary, payload, source_uid,
target_uid}`. `suppress_recording()` guards re-entrant replay; the log
auto-compacts past ~5000 events.

# Replay / projection

- Periodic **checkpoints** (`DEFAULT_CHECKPOINT_INTERVAL = 50`, capped at 20
  in-memory snapshots) are full-domain deep copies; evicted ones are rebuildable
  from the log.
- `build_target_state(checkpoint, events_to_replay)` seeds from the nearest
  snapshot and merges deltas from `replay.py`
  (`reconstruct_navigation_state`, `reconstruct_parameter_state`,
  `reconstruct_fit_range_state`, `reconstruct_setup_state`,
  `reconstruct_model_state`) into a plain-dict `DomainState`.
- Projection is deliberately **GUI-free** so it is headless-testable and reusable
  server-side; applying the state to live widgets is the GUI adapter's job
  (`chisurf/gui/main_helper.py`).

# Durable projection into MFDB ([PRD-43](/prds/prd-43.md))

The in-memory list is authoritative for the live session; each event is also
**best-effort** persisted to the [MFDB](/architecture/mfdb.md) durable event log:

- `OperationHistory._persist_event` → `chisurf.core.mfdb.event_log.append_event`
  (into `mfdb_event_log`). This is a no-op without a database, so history works
  identically offline (local mode).
- `list_events(source="mfdb")` reads the durable log back; `source="memory"`
  (default) returns the live view. `action_type` values use the dotted MFDB
  vocabulary, normalized in `replay.py`.

Per [PRD-43](/prds/prd-43.md) the GUI history browser is a **projection over
MFDB** rather than a parallel store, aligning session history with the
provenance DAG.

See also: [overview](/overview.md), [Core](/subsystems/core.md),
[macros & CLI](/subsystems/macros-cli.md), [server](/architecture/server.md).
