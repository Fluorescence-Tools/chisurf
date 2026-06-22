# PRD-13: Study / Project Entity with Configurable Fields (LIMS P2)

## Goal

Promote `project_id` from a loose string into a real **study** entity that groups
samples and datasets, carries per-study configurable metadata fields, has
ownership/visibility, and gives the dataset browser a "study" facet.

## Background

- LIMS reference: iSkyLIMS `SampleProjects` + `SampleProjectsFields` +
  `SampleProjectsFieldsValue` — samples grouped into projects, each project
  defining user-configurable metadata fields.
- MFDB today: `project_id TEXT` on `mfdb_sample`/`flr_sample`/`mfdb_experiment` —
  no project table, no membership, no custom fields.
- See `overhaul/MFDB-LIMS-diagnosis.md` (P2).

## Design (dictionary-driven, reuse existing patterns)

- **`mfdb_study`** (`.dic`-declared, generated, gate-covered):
  `(study_id PK, name, description, created_by_user_id FK flr_sample_users,
   is_public INTEGER DEFAULT 0, created_at, updated_at, deleted_at)`. Ownership +
  visibility reuse the PRD-04/10 own+public model (and `mfdb_artifact_owner`-style
  scoping where multi-owner is wanted).
- **`mfdb_study_member`** (`.dic`-declared): `(study_id FK, member_type, member_id,
   role, created_at)`, `UNIQUE(study_id, member_type, member_id)`. `member_type`
  is `sample` or `artifact` (extensible vocab). Links existing samples/datasets
  into the study — many-to-many (a dataset can belong to several studies).
- **Configurable fields:** do **not** build a new EAV stack. Reuse the existing
  key-value pattern (`flr_sample_key_value` / `mfdb_*_key_value`) with a
  `mfdb_study_key_value(study_id, key, value, details)` for per-study metadata.
  Optionally a `mfdb_study_field_def(study_id, name, value_type, units, required,
  description)` (dictionary-described) if typed/validated custom fields are needed
  — mirror the operation-parameter-def approach from PRD-11, scoped per study.
- **Browser facet:** `browse_datasets` gains a `study_id` filter (artifacts whose
  `artifact_id` is a `mfdb_study_member`, or whose sample is a member). The
  picker/admin can list studies (own + public) and scope datasets to one.

## API

- `create_study(name, description, is_public=False)` → study_id (owner = active
  user via the canonical resolver).
- `add_study_member(study_id, member_type, member_id, role="member")`,
  `list_study_members(study_id)`.
- `list_studies(scope="mine"|"public"|"all")` (own + public, like setups).
- `set_study_field` / `get_study_fields` for configurable metadata.

## Tasks

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

## Definition of Done

- [ ] `mfdb_study` + `mfdb_study_member` (+ key-value/field-def) exist, dict-
      declared, generated, gate-covered; project_id backfilled into studies.
- [ ] Studies are user-scoped (own + public); datasets/samples are many-to-many
      members; configurable per-study fields work.
- [ ] `browse_datasets` filters by study; the browser and admin expose studies.
- [ ] Tests pass including two-user scoping and browse-by-study.

## Definition of Clean

`.dic` dictates the schema (no hardcoded SQL/blob); reuse the existing key-value
and own+public machinery (no forked stacks); server-side filtering; DI over
monkeypatching; behavior-asserting tests; GUI smoke.
