# PRD-10 bug — processed datasets never appear in the browser

Reported: "I processed a dataset and expected the processed dataset to appear in
the MFDB dataset load; it does not."

## Root cause: registration fails **silently**, so nothing is stored

Reproduced directly (register a raw + a `processed_data` under the resolved active
user, then browse). Two independent failures, both **swallowed** by best-effort
registration (`register_result` logs a warning and returns `""`), so the data
vanishes with no user-visible error:

### Bug A — `operation_type="microtime_shift"` is not in the vocabulary

```
register_result failed (kind=processed_data):
  Unknown extensible vocabulary value 'microtime_shift' for field 'operation_type'
```

`OPERATION_TYPES` (`chisurf/core/mfdb/models.py:37`) is a fixed tuple validated by
`validate_extensible_vocab`. The PRD-09 shifter registers with
`operation_type="microtime_shift"`, which is **not** in that tuple → `record_operation`
raises → the whole `register_result` transaction rolls back → the processed
artifact is never stored → it cannot appear in the browser. (This is the same
class of issue hit earlier with `g_factor_processing`, which was worked around by
using `"calibration"`.)

**Fix:** add `microtime_shift` (and any other new operation types) to
`OPERATION_TYPES`, or make `validate_extensible_vocab` actually extensible (accept
values registered in the `mfdb_vocabulary` table) and have the shifter register
its type. Prefer adding the value to the canonical tuple.

### Bug B — `created_by_user_id` FK fails when the active user has no row

```
register_artifact: FOREIGN KEY constraint failed
```

PRD-10 added `created_by_user_id TEXT REFERENCES flr_sample_users(user_id)` to
`mfdb_artifact`, and `register_result`/`register_raw_measurement` stamp
`_resolve_active_user_id()`. That resolver returns
`cs_settings["mfdb"]["default_user_id"]` — on this machine, **"tpeulen"** — but the
schema only bootstraps `user_default` and `guest` rows (`schema.py:1734,1760`).
There is no `flr_sample_users` row for `tpeulen`, so the FK fails and the **entire**
registration (raw and processed) rolls back.

Repro: with no `tpeulen` row, both raw and processed registration fail; after
inserting a `tpeulen` row, raw registration succeeds and appears in
`browse_datasets(scope="own")`.

**Fix:** ensure the resolved active user exists in `flr_sample_users` before
stamping (bootstrap/`INSERT OR IGNORE` the active user, or fall back to
`user_default` when the configured `default_user_id` has no row). A configured
`default_user_id` that points at a non-existent user must not break all
registration.

### Bug C (meta) — best-effort registration hides real bugs

Both A and B are *real* errors (vocab rejection, FK violation), not the
"MFDB-unavailable" condition the best-effort path is meant to tolerate. Swallowing
them as a debug/warning log and returning `""` caused **silent data loss** — the
user saw a successful-looking process but nothing was stored. Per the Definition
of Clean ("best-effort MFDB, but fail loud on real bugs"), distinguish:

- MFDB unavailable / no DB → soft warn (current behavior is fine), vs.
- a registration error when a DB *is* present (FK, vocab, integrity) → surface it
  (visible warning / raise in non-GUI callers), so it cannot silently drop data.

## Verification (after fixes)

- Registering `processed_data` with `operation_type="microtime_shift"` returns a
  non-empty id and appears in `browse_datasets(scope="own", owner_id=<active>)`.
- Works when `default_user_id` is a user that wasn't pre-seeded (auto-bootstrapped).
- A registration error with a present DB is surfaced, not swallowed.

## Note

Add a regression test that registers a `processed_data` artifact via the real
`register_result` (not a hand-inserted row) under a configured non-default user,
and asserts it appears in `browse_datasets` — this would have caught both bugs.
