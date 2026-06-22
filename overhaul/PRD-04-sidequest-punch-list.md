# PRD-04 Sidequest — Coder Punch-List

Per-user detector-setup migration, ownership, and public sharing. Builds on the
merged PRD-04 prerequisite (structured setups, dict-generated DDL). Source:
`PRD-04-burst-pipeline.md` → "Sidequest: Per-User Detector-Setup Migration And
Ownership".

## Locked decisions

- **Ownership = owner column, not ACL.** A `created_by_user_id` column on
  `mfdb_setup` is the entire ownership mechanism. Do **not** use
  `mfdb_object_acl` / `create_default_acl_for_object`.
- **Visibility = `is_public` flag.** Users see their own setups **plus** all
  public ones. A "Make public" checkbox in the Detector Wizard toggles it
  (owner-only).
- **Setup ids are user-scoped** to stop same-named setups colliding. Recommended:
  `tttr_detector_setup:<user_slug>:<name_slug>`.
- **JSON stays a first-class export/import format** (portability, `.csp` project
  files). MFDB is the authoritative store; the JSON serializer is owner-agnostic.
- **Governing rule (from prerequisite):** the `.dic` dictates the schema — new
  columns are declared in `mfdb_flr_ext.dic` with `_chisurf_schema` bridges and
  generated/validated, never hardcoded. Total-coverage `validate_mapping()` gate
  must stay green.

## S1 — Ownership + visibility columns

- [ ] Declare `created_by_user_id` (`code`/TEXT, FK
      `flr_sample_users(user_id)`) and `is_public` (`int`, default 0) as `.dic`
      items on category `mfdb_setup`, each with `_chisurf_schema.table_name` /
      `_chisurf_schema.column_name` (and `_chisurf_schema.foreign_key` for the
      FK).
- [ ] Add both columns to `mfdb_setup` via `ALTER TABLE` under a new
      `SCHEMA_VERSION` bump (match the existing migration / `MigrationReport`
      style). Do not hand-write a second CREATE; the column set stays
      dictionary-sourced.
- [ ] Extend `MFDatabase.save_setup()` with `created_by_user_id` and `is_public`
      params; persist them.
- Done when: total-coverage gate passes with both new items mapping to live
  columns.

## S2 — User-scoped setup ids

- [ ] Change `setup_id_for_name()` to namespace by user:
      `tttr_detector_setup:<user_slug>:<name_slug>`.
- [ ] Thread the active user through every caller: `tttr_detector_setups.py`,
      `gui/tool.py`, and burst `api/mfdb.py` setup resolution.
- [ ] Builtin / ownerless setups stay globally resolvable (treated as public).
- Done when: two users with a setup named the same do not share a `setup_id`.

## S3 — Per-user auto-migration (idempotent)

- [ ] Resolve the active user from the existing login/session flow (GUI
      `current_user_id`); fall back to a defined default/local user when nobody is
      logged in. Ensure that default user is a real `flr_sample_users` row so the
      FK holds.
- [ ] Rewrite `_migrate_json_setups_to_mfdb()` to import `detector_setups.json`
      stamped with the active user **only if that user has no setups yet**.
      Re-running for the same user adds nothing; running for a different user
      never overwrites another user's setups.
- [ ] Do not delete `detector_setups.json` after import.
- Done when: per-user idempotent; single-user/no-login installs still migrate.

## S4 — Scope reads/writes (own + public)

- [ ] `load_detector_setups()` query: `created_by_user_id = <active_user> OR
      is_public = 1`.
- [ ] `save_detector_setups()` / Detector Wizard write under the active user's
      ownership + namespaced id, carrying `is_public`.
- [ ] Owner-only edits: a non-owner opening a public setup gets read-only /
      save-as-a-copy (saving creates a setup owned by them). Never mutate another
      user's row.
- [ ] mfdb-admin Setups view (prerequisite Task P4) shows `Owner` and `Public`
      columns.

## S4a — "Make public" checkbox

File: `tttr_channel_definition.py` (Detector Wizard) + `tttr_detector_setups.py`.

- [ ] Add a **"Public (visible to all users)"** checkbox to the setup save UI;
      map its state to `is_public` on save.
- [ ] On load, reflect the setup's current `is_public`; disable the checkbox (and
      edit controls) when the active user is not the owner.
- [ ] New setups default to private (`is_public = 0`).

## S5 — JSON export stays lossless

- [ ] `save_detector_setups()` / `load_detector_setups()` still round-trip the
      JSON shape against an explicit `file_path`.
- [ ] MFDB<->JSON conversion (`_setup_row_data()` + save path) drops no
      detector / window / `tttr_reading` fields.
- [ ] JSON serializer is **not** coupled to ownership — exported setups are
      owner-agnostic; ownership/`is_public` are applied on import under the active
      user.

## Tests (`test/fio/test_setup_user_migration.py`)

- [ ] Two users importing different JSON keep distinct, non-colliding setups.
- [ ] Per-user migration is idempotent (re-run adds nothing).
- [ ] `load_detector_setups()` for user A excludes user B's **private** setups but
      includes user B's **public** ones.
- [ ] A logged-in save records the owner; toggling public flips `is_public` and
      changes other users' visibility.
- [ ] No-login / default-user path still migrates and loads.
- [ ] JSON round-trip lossless; imported copy owned by the active user.

## Verify

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." \
  /Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest -p no:cov -o addopts='' \
  test/fio/test_setup_prerequisites.py \
  test/fio/test_burst_pipeline_mfdb.py \
  test/fio/test_setup_user_migration.py
```

- [ ] Prerequisite + dict-validation gate stay green (new owner/`is_public`
      columns map cleanly).
- [ ] Burst `setup_id` resolution still works with user-namespaced ids.

## Watch-outs

- A burst run uses the active user's setup — confirm `api/mfdb.py` resolves
  user-namespaced ids.
- The default/no-login user must exist in `flr_sample_users` (create if needed)
  for the FK.
- Keep the JSON serializer owner-free so `.csp`/exported setups stay portable.
