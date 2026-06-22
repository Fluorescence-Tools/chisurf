# PRD-04 Sidequest D (Time-Versioned Setup Calibration) — Code Review

Review of the dated-calibration implementation. Source: `PRD-04-burst-pipeline.md`
"Sidequest D: Time-Versioned Setup Calibration".

Reviewed (2026-06-21):

- `chisurf/core/mfdb/schema.py` (v35 table + migration backfill)
- `chisurf/core/mfdb/data/mfdb_flr_ext.dic` (`mfdb_setup_calibration` category)
- `chisurf/core/mfdb/repository.py` (`add_setup_calibration`,
  `list_setup_calibration_dates`, `get_setup_calibration`, `save_setup`)
- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_channel_definition.py`
  (date combobox)
- `test/fio/test_setup_calibration_history.py`

Tests: 17 passed (calibration history + prerequisites + detector setups).

## Verdict

**Mostly solid, but two real defects must be fixed before it behaves as
intended.** The table, dict declaration, generated DDL, gate coverage,
append-only API, latest-per-channel resolution, migration backfill, and the date
combobox are all present and tested. However (1) `save_setup` appends a
calibration snapshot on **every** save with no change detection, so ordinary
structural re-saves pollute the calibration history, and (2) the date combobox
resolves the setup id **without the active user**, so it is empty/mismatched for
per-user setups.

## What's correct (keep)

- `mfdb_setup_calibration` is `.dic`-declared, generated from the dictionary,
  added to `_SETUP_CATEGORIES`; the total-coverage gate passes (v35,
  `schema.py:3758`).
- `add_setup_calibration` is genuinely append-only and refreshes the
  `mfdb_setup_detector_channel` cache row (`repository.py:1242`).
- `list_setup_calibration_dates` returns distinct `calibrated_at` desc;
  `get_setup_calibration(setup, date|None)` resolves a specific snapshot or the
  latest-per-channel via `MAX(calibrated_at) GROUP BY channel_name` (correct,
  since ISO timestamps sort lexically).
- v35 migration backfills one snapshot per existing channel using
  `updated_at`/`created_at` as `calibrated_at` (`schema.py` v35 block).
- The Detector wizard has a `calibration_combo` next to the setup combo;
  `_on_calibration_changed` loads the snapshot's factors into the table.

## 🔴 Finding 1 — `save_setup` appends a snapshot on every save (history pollution)

`save_setup` inserts a new `mfdb_setup_calibration` row for each detector channel
whenever `g_factor`/`l1`/`l2` is present, with **no change detection**
(`repository.py:4202`):

```python
# Append calibration snapshot when calibration data is present
if g_factor is not None or l1 is not None or l2 is not None:
    self.conn.execute("INSERT INTO mfdb_setup_calibration ... VALUES (... now ...)")
```

Because the Detector wizard's Save (`_on_save` → `save_detector_setups` →
`save_setups` → `save_setup`) runs on *any* edit, a user who saves the setup
twice — e.g. after tweaking a PIE window — creates **two snapshots per channel
with identical factors but different timestamps**. The date combobox then fills
with bogus "calibration dates" that do not correspond to real calibration
changes. This contradicts D2: "structural edits and calibration updates are
distinct paths … never overwrite," and replaces overwrite with the opposite
problem — spurious appends.

The test codifies the wrong behavior: `test_save_setup_creates_calibration_snapshots`
saves once and asserts `len(snapshots) == 2` (2 channels), so it never exercises
a re-save and would not catch the duplication.

Fix (pick one):
- **Dedup-on-change (recommended):** append from `save_setup` only when
  `(g_factor, l1, l2, g_factor_calibration_id)` differ from the latest snapshot
  for that channel (compare against `get_setup_calibration(setup, None)`).
- **Or** stop appending from `save_setup` entirely and route calibration through
  `add_setup_calibration` only, from explicit calibration actions (the G-factor
  apply path, or a dedicated "record calibration" action).
- Add a test that saves the same setup twice with unchanged factors and asserts
  the snapshot count does **not** grow.

## 🟠 Finding 2 — date combobox ignores the active user (per-user mismatch)

`_populate_calibration_combo` and `_on_calibration_changed` resolve the setup id
without a user:

```python
setup_id = setup_id_for_name(setup_name)            # tttr_channel_definition.py:1003
setup_id = setup_id_for_name(self.current_setup_name)  # :1036
```

But the save path namespaces by the active user
(`setup_id_for_name(name, user_id, prefix)`), so for a logged-in user the
snapshots live under `tttr_detector_setup:<user>:<slug>` while the combobox
queries `tttr_detector_setup:<slug>` (global). Result: the date picker shows only
"Latest" and loading a date finds nothing. It works only in the no-login/global
case (which is why the tests, run headless without login, pass).

Fix: pass the active user (`_resolve_active_user_id()`) to `setup_id_for_name` in
both call sites, matching the save path.

## 🟡 Minor

1. `get_setup_calibration` / `list_setup_calibration_dates` do not filter
   `deleted_at IS NULL`. Harmless while the table is append-only, but add the
   filter for consistency if soft-delete is ever used.
2. `add_setup_calibration` always overwrites the detector-channel cache, even when
   the new snapshot's `calibrated_at` is older than the current latest
   (backdated insert would clobber the cache with stale factors). Update the cache
   only when the new snapshot is the newest for that channel.
3. D6 is implemented: the burst pipeline resolves and records `calibrated_at`
   (`burst_selection/api/mfdb.py:62` `_resolve_calibrated_at`, threaded into the
   registered metadata at `:283,372,474`). Good — note it inherits Finding 2's
   user-scoping concern via `list_setup_calibration_dates(setup_id)`, so verify
   the setup_id it resolves is user-correct.

## Required before "complete"

- [ ] Finding 1: stop `save_setup` from appending a snapshot on unchanged
      factors (dedup-on-change), and add a re-save test asserting no growth.
- [ ] Finding 2: resolve `setup_id_for_name` with the active user in both
      calibration-combo call sites.
- [ ] Confirm D6 (analysis reproducibility metadata) is implemented.
