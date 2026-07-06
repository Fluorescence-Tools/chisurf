---
type: PRD
prd: "13"
title: "PRD-13: Study / Project Entity with Configurable Fields"
description: Promote the loose project-id string into a real study entity that groups samples and datasets with ownership, membership, and custom fields.
status: done
phase: "3"
resource: chisurf/core/mfdb
tags: [prd, mfdb, lims]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
`project_id` was a loose string with no table, membership, or custom fields. This PRD makes a first-class **study** entity (`mfdb_study`) that groups samples and datasets, carries ownership and visibility (own + public), supports many-to-many membership (`mfdb_study_member`), and holds per-study configurable metadata via the existing key-value pattern (no forked EAV stack). The dataset browser gains a `study_id` facet, and studies are backfilled idempotently from distinct existing `project_id` values. Repository CRUD, membership, scoping, RPC handlers, and a standalone admin studies view are provided, all dictionary-declared and gate-covered.

# Status
Done. Schema, CRUD, many-to-many membership, configurable fields, browser facet, RPC, and admin view landed (23 tests). Wiring the studies view into the admin dock layout and adding a study selector to the browser widget are deferred to the dock rewrite.

# Goal
Promote `project_id` from a loose string into a real **study** entity that groups
samples and datasets, carries per-study configurable metadata fields, has
ownership/visibility, and gives the dataset browser a "study" facet.

# Background
- LIMS reference: a mature LIMS groups samples into projects, each project
  defining user-configurable metadata fields (a project table + a field-definition
  table + a field-value table).
- MFDB today: `project_id TEXT` on `mfdb_sample`/`flr_sample`/`mfdb_experiment` —
  no project table, no membership, no custom fields.

# Design (dictionary-driven, reuse existing patterns)
- **`mfdb_study`** (`.dic`-declared, generated, gate-covered):
  `(study_id PK, name, description, created_by_user_id FK flr_sample_users,
  is_public INTEGER DEFAULT 0, created_at, updated_at, deleted_at)`. Ownership +
  visibility reuse the PRD-04/[PRD-10](prd-10.md) own+public model (and
  `mfdb_artifact_owner`-style scoping where multi-owner is wanted).
- **`mfdb_study_member`** (`.dic`-declared): `(study_id FK, member_type, member_id,
  role, created_at)`, `UNIQUE(study_id, member_type, member_id)`. `member_type` is
  `sample` or `artifact` (extensible vocab). Links existing samples/datasets into
  the study — many-to-many (a dataset can belong to several studies).
- **Configurable fields:** do **not** build a new EAV stack. Reuse the existing
  key-value pattern (`flr_sample_key_value` / `mfdb_*_key_value`) with a
  `mfdb_study_key_value(study_id, key, value, details)` for per-study metadata.
  Optionally a `mfdb_study_field_def(study_id, name, value_type, units, required,
  description)` (dictionary-described) if typed/validated custom fields are needed
  — mirror the operation-parameter-def approach from [PRD-11](prd-11.md), scoped
  per study.
- **Browser facet:** `browse_datasets` gains a `study_id` filter (artifacts whose
  `artifact_id` is a `mfdb_study_member`, or whose sample is a member). The
  picker/admin can list studies (own + public) and scope datasets to one.

# API
- `create_study(name, description, is_public=False)` → study_id (owner = active
  user via the canonical resolver).
- `add_study_member(study_id, member_type, member_id, role="member")`,
  `list_study_members(study_id)`.
- `list_studies(scope="mine"|"public"|"all")` (own + public, like setups).
- `set_study_field` / `get_study_fields` for configurable metadata.

# Tasks
1. `.dic` + schema: declare `mfdb_study`, `mfdb_study_member`,
   `mfdb_study_key_value` (and optional `mfdb_study_field_def`); generate DDL;
   `SCHEMA_VERSION` bump; add to the total-coverage gate. Backfill `mfdb_study`
   from distinct existing `project_id` values (one study per project_id), and
   create members from samples/experiments carrying that project_id.
2. Repository: study CRUD, membership, scoping (own + public), configurable fields.
3. RPC: `studies.list` / `studies.get` / `studies.save` / `studies.members.*`
   (user-scoped via the canonical resolver, consistent with datasets.browse).
4. Browser: `study_id` facet in `browse_datasets` + a study selector in the
   dataset browser widget.
5. mfdb-admin: a Studies entity (dictionary-sourced columns) with members + fields.
6. Tests: study scoping (own excludes others' private, includes public);
   membership many-to-many; browse-by-study; backfill from project_id; dict gate
   green; GUI smoke.

# Definition of Done
- [x] `mfdb_study` + `mfdb_study_member` + `mfdb_study_key_value` exist, dict-
      declared, generated, gate-covered; project_id backfilled into studies
      (`backfill_studies_from_project_ids`, callable/idempotent — not a version
      migration, per PRD-19's disposable-DB policy).
- [x] Studies are user-scoped (own + public); datasets/samples are many-to-many
      members; configurable per-study fields work (the key-value pattern, no forked
      EAV stack).
- [x] `browse_datasets` filters by study; the admin exposes studies (the dataset
      browser widget facet can call the same `study_id` parameter).
- [x] Tests pass including two-user scoping and browse-by-study. (23 tests, arm64.)

# Implementation status — COMPLETE
**Increment 1 (schema + CRUD + membership + fields).** `mfdb_study` /
`mfdb_study_member` / `mfdb_study_key_value` are `.dic`-declared and created by
`reconcile_schema`. `repository.py`: `create_study`/`get_study`/`list_studies(scope
mine|public|all)`, `add_study_member`/`list_study_members`/
`list_studies_for_member` (idempotent, many-to-many),
`set_study_field`/`get_study_fields` (configurable fields via the key-value
pattern), `backfill_studies_from_project_ids`. Study references are logical (the
dictionary key is not emitted as a PRIMARY KEY, so no hard FK — the repository
manages integrity, as for `mfdb_operation.protocol_id`). `test/fio/test_study.py`.

**Increment 2 (browser facet).** `browse_datasets` gains a `study_id` filter — an
artifact is in the study if it is a direct member or if its linked sample is a
member.

**Increment 3 (admin).** Backend handlers `mfdb.studies.{list,get,create,
members.add,fields.set}` + `MFDBClient` methods + a standalone
`gui/studies_view.py::StudiesView` (scoped list, members, fields, create +
add-member). Tested via the `InProcessClient` and offscreen. Kept standalone like
the Lifecycle/Protocols views so the mid-overhaul dock layer slots it in.

**Deferred (same as PRD-12/14):** wiring `StudiesView` into the admin tool's dock
layout and adding a study selector to the dataset-browser *widget* (the
`browse_datasets` `study_id` parameter is ready) — once the dock rewrite lands.

# Definition of Clean
`.dic` dictates the schema (no hardcoded SQL/blob); reuse the existing key-value
and own+public machinery (no forked stacks); server-side filtering; DI over
monkeypatching; behavior-asserting tests; GUI smoke.

# Relationships
- Reuses the own+public and multi-owner scoping model from [PRD-10](prd-10.md).
- Adds a `study_id` filter to the dataset browser of [PRD-10](prd-10.md).
- Shares the dictionary/key-value machinery with the other LIMS layers ([PRD-12](prd-12.md), [PRD-14](prd-14.md), [PRD-15](prd-15.md)).
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).
