# PRD-04 Sidequest B addendum 2 — Code Review (MFDB-default save, remove legacy JSON)

Reviewed (2026-06-21):

- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_setup_utils.py`
  (`save_setups`, `migrate_json_to_mfdb`)
- `chisurf/core/fluorescence/fcs/channel_setups.py` (`load_fcs_channel_setups`)
- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_detector_setups.py`
  (`load_detector_setups`)
- `chisurf/plugins/fcs/fcs_channel_preset/__init__.py` (`_on_save`)
- `test/fio/test_fcs_setup.py`

Tests: 53/53 across the setup/calibration/jordi/gui suites.

## Verdict

Most of the addendum is implemented cleanly — default save goes MFDB-only with no
JSON side-write, the "Saved to MFDB" message is fixed, the legacy file is removed
in the load context, and the detector path shares the same machinery. **But the
no-MFDB behavior is a hard fail, which contradicts the locked decision ("soft
fallback — give warning").** That needs a decision/fix.

## 🔴 Finding 1 — no-MFDB save hard-fails instead of soft fallback + warning

Last turn you chose **"soft fallback — give warning"**, and the PRD says: *"if MFDB
cannot be opened, Save still succeeds by writing the JSON fallback, but the user
is warned … The save must not fail or lose data."*

The implementation does the opposite:

- `save_setups()` — when `file_path is None` (default save) and MFDB is
  unavailable/raises, it returns `False` with **no JSON written**
  (`tttr_setup_utils.py`: `if file_path is None: return False`).
- `_on_save()` — on `ok == False` shows a **critical "Error"** dialog ("Could not
  save … Check that the database is available.") and the user's edits are **lost**.

So a user with no/broken MFDB cannot save at all — the opposite of a soft,
data-preserving fallback. No test covers the no-MFDB path, so this isn't caught;
`test_default_fcs_save_uses_mfdb_no_json` only exercises the MFDB-available happy
path.

This is a genuine conflict between two of your statements:
- "soft fallback — give warning" (the decision I encoded in the PRD), vs.
- this implementation's "returns False … no JSON fallback for default saves"
  (hard fail).

**Decide which you want:**
- (A) Soft fallback (matches the current PRD): on no-MFDB, write the JSON fallback
  and show a *warning* ("MFDB unavailable; saved to a local JSON file, not stored
  in the database / not assigned to a user"). `_on_save` shows a warning, not a
  critical error. Add a test for it.
- (B) Hard fail (matches this implementation): keep return `False` + error, and I
  will update the PRD to say no-MFDB is a hard fail (drop the soft-fallback rule).

Recommendation: (A), because it preserves the user's work and is what was agreed;
the warning still makes the degraded state obvious.

## 🟡 Finding 2 — legacy file deleted even when migration imported nothing

`load_fcs_channel_setups` (and `load_detector_setups`) call
`migrate_json_to_mfdb(...)` then unconditionally `if path.exists(): path.unlink()`.
But `migrate_json_to_mfdb` is per-user idempotent — it imports **only when the
active user has no setups yet**. So if the active user already has setups, the
migration imports nothing, yet the load still deletes the legacy JSON.

- Normal lifecycle (file migrated on first load, then deleted) is fine.
- Edge case: a user who already has MFDB setups *and* a legacy JSON containing
  *different* setups loses that JSON unimported. Unlikely in the real lifecycle,
  but the delete is not actually gated on "a verified import happened this call,"
  despite the comment claiming it is.

Fix: gate the unlink on the migration having actually imported (e.g. have
`migrate_json_to_mfdb` return whether it imported, and only unlink on True), so
the file is removed exactly when its contents are safely in MFDB.

## Good (keep)

- Default save is MFDB-only; no JSON side-file at the canonical path
  (`save_setups` returns False rather than writing JSON when `file_path is None`).
- `_on_save` message now says "Saved … to MFDB", not the JSON path (the reported
  defect).
- Deletion correctly moved out of `migrate_json_to_mfdb` into the load context so
  direct multi-user test calls don't destroy the shared file prematurely.
- Explicit `file_path` export still round-trips JSON (export-only path preserved).
- Detector and FCS share the same machinery; no duplicate logic.

## Required before "complete"

- [ ] Finding 1: reconcile no-MFDB behavior with the chosen policy (recommend the
      soft fallback + warning per the PRD) and add a no-MFDB test.
- [ ] Finding 2: delete the legacy JSON only when the migration actually imported
      it this call.
