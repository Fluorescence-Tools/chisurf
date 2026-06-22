# PRD-04 Sidequest — Code Review (Per-User Setups + Public Sharing)

Review of the sidequest implementation (per-user detector-setup migration,
ownership, public sharing). Source: `PRD-04-burst-pipeline.md` Sidequest §,
`PRD-04-sidequest-punch-list.md`.

Reviewed (2026-06-20):

- `chisurf/core/mfdb/schema.py` (v32 migration, `mfdb_setup` columns)
- `chisurf/core/mfdb/data/mfdb_flr_ext.dic` (`created_by_user_id`, `is_public`)
- `chisurf/core/mfdb/repository.py` (`save_setup`)
- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_detector_setups.py`
- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_channel_definition.py`
- `test/fio/test_setup_user_migration.py`

Tests: `test_setup_user_migration.py` (8) + `test_setup_prerequisites.py` +
`test_burst_pipeline_mfdb.py` + `test_detector_setups.py` → **all pass (29)**.

## Verdict

The sidequest is implemented well and matches the locked decisions: owner column
(not ACL), user-namespaced ids, `is_public` flag + owner-only checkbox, per-user
idempotent migration, own+public read scoping, and the dict-validation gate
extended to the new columns. **One Medium issue:** the `is_public` *default* is
inverted across layers — public by default at the storage layer, private by
default at the GUI layer — which can silently leak a user's setups as public via
any non-GUI caller. No blockers.

## Task status

### ✅ S1 — Ownership + visibility columns
- `created_by_user_id TEXT REFERENCES flr_sample_users(user_id)` and `is_public`
  on `mfdb_setup`, declared in the `.dic` with `_chisurf_schema` bridges, added by
  v32 `ALTER TABLE` (`schema.py:3667-3688`). `save_setup()` takes both params.
- Dict-validation gate (`test_setup_prerequisites.py:161-192`) iterates
  `mfdb_setup` and asserts no unmapped items — the two new items map cleanly.

### ✅ S2 — User-scoped ids
- `setup_id_for_name(name, user_id="")` →
  `tttr_detector_setup:<user_slug>:<name_slug>` when a user is given; ownerless
  setups stay globally resolvable. Test: `test_same_name_different_users_no_collision`.

### ✅ S3 — Per-user idempotent migration
- `_migrate_json_setups_to_mfdb(db, path, user_id=...)` imports only if that user
  has no setups yet; default user resolved via `_resolve_active_user_id()`. Tests:
  `test_per_user_migration_idempotent`, `test_different_user_gets_separate_migration`,
  `test_default_user_migration_and_load`.

### ✅ S4 — Own + public scoping
- `_load_mfdb_detector_setups()` includes a row when `owner is None` (builtin),
  `is_public`, or `owner == user_id`. Tests:
  `test_load_excludes_other_users_private_setups`,
  `test_load_includes_shared_and_public_setups`.

### ✅ S4a — "Make public" checkbox
- `public_checkbox` in the wizard, default unchecked, enabled only for the owner
  (`tttr_channel_definition.py:151-160, 853-870`).

### ✅ S5 — JSON export
- `save_detector_setups()` / `load_detector_setups()` retain JSON round-trip;
  ownership/`is_public` applied on import, not baked into the file. Tests:
  `test_save_no_login_still_works`, default-user path.

## 🟠 Medium — `is_public` default is inverted across layers

Three layers disagree on the default visibility of a setup:

- **Column:** `is_public INTEGER DEFAULT 1` (`schema.py:869`) — public.
- **`MFDatabase.save_setup`:** `0 if is_public is False else 1`
  (`repository.py:3989`) — `None` → **public**.
- **GUI wrapper `_save_setup_row`:** `if is_public is None: is_public = False`
  with comment "all setups default to private" (`tttr_detector_setups.py:192-193`)
  — private.

The GUI path is correct (new setups private, opt-in via checkbox), but any other
caller of `db.save_setup(...)` that omits `is_public` gets a **public** setup —
the opposite of the feature's intent ("setups should be user-specific"). This is
a latent privacy inversion that the GUI-only tests don't catch.

Recommendation: make private the default at **every** layer to match the intent
and the GUI wrapper:
- Column `is_public INTEGER DEFAULT 0`.
- `save_setup`: `1 if is_public is True else 0` (i.e. `None` → private).
- Keep the v32 migration backfilling **pre-existing** rows as desired for
  back-compat — but note those rows are also `created_by_user_id IS NULL`, and
  the loader already treats ownerless setups as shared regardless of `is_public`,
  so the `DEFAULT 1` is largely redundant for back-compat and mainly creates the
  inversion risk for new owned rows.

## 🟡 Minor

1. **Docstring contradicts behavior.** `save_detector_setups()` says
   "`is_public` … True (default) = visible to all users"
   (`tttr_detector_setups.py:400-401`), but the code defaults to private. Fix the
   docstring (and align with the Medium fix above).
2. **Ownerless + private is still shared.** The loader includes any row with
   `owner is None` unconditionally, so a row with `is_public = 0` and
   `created_by_user_id = NULL` is shown to everyone. This is fine if "ownerless ==
   builtin == shared" is the intended invariant — worth a one-line comment
   stating it so a future reader doesn't read it as a leak.
3. **`mfdb_setup` is the grandfathered hand-written table**, so the two new
   columns are hand-added to its `CREATE` and also declared in the `.dic`. That is
   consistent with the prerequisite's "existing tables grandfathered" rule (only
   the child tables are generated), but the FK/owner column now lives in two
   hand-maintained spots (CREATE + v32 ALTER) — acceptable, just keep them in
   sync on the next bump.

## Required before merge

- [x] Medium: `is_public` now defaults private at every layer — column
      `DEFAULT 0` (fresh-DB `schema.py:869` and v32 `ALTER`), `save_setup`
      coercion `1 if is_public is True else 0` (`repository.py:3989`), and the
      `save_detector_setups` docstring corrected. The v32 migration carries a
      comment noting pre-existing rows are ownerless (so a private default does
      not hide previously-visible setups). 33 setup/sidequest tests pass.
- [x] Minor 2: the "ownerless == shared" invariant is documented inline at
      `tttr_detector_setups.py:231-232`.

Optional follow-up: add a regression test asserting `db.save_setup(...)` without
`is_public` yields `is_public = 0`, to lock the non-GUI default.

## Good (keep)

- Owner-only checkbox enable, user-namespaced ids, per-user idempotent migration,
  own+public scoping, JSON kept owner-agnostic, and the dict-validation gate
  extended to cover the new columns — all correct and tested.
