---
type: PRD
prd: "10"
title: "PRD-10: MFDB Dataset Browser Widget"
description: A reusable Qt widget to pick a registered MFDB dataset, with scope/visibility, server-side filtering, co-ownership, and file groups.
status: in-progress
phase: "0"
resource: chisurf/gui/widgets/mfdb/dataset_browser.py
tags: [prd, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
A reusable Qt widget lets users pick a registered MFDB dataset instead of hunting for a file on disk. It lists the active user's datasets (with Mine/Public/All scope), offers debounced server-side search and structured filters with pagination, and returns a typed `DatasetSelection` any plugin can resolve to a local path via an RPC. Object-store dedup is reconciled with a many-to-many ownership model (`mfdb_artifact_owner`) so co-owners all see a shared dataset under "Mine." Multi-file datasets are meant to register as one dataset via a dictionary-declared file-group membership table (`mfdb_artifact_member`), which is not yet implemented. The microtime shifter is the first consumer and end-to-end acceptance case. A companion bug note captures silent registration failures (unknown operation-type vocabulary; a missing active-user row breaking the owner foreign key) that caused processed datasets to never appear.

# Status
In-progress (re-verified against code 2026-07-05). Landed: the reusable browser + picker dialog, `datasets.browse`/`datasets.open` RPC, the co-ownership model (`mfdb_artifact_owner`) with Mine/Public/All scope and pagination, and the shifter integration. **Open:** the multi-file **file-group** membership schema (`mfdb_artifact_member`) is absent from the tree, so multi-file datasets are not yet registered/opened as one — this deliverable remains reopened.

# Goal
A reusable Qt widget that lets a user pick a **registered MFDB dataset** instead of
hunting for a file on disk. It lists the active user's datasets (and, when toggled,
public ones), with a fast filter/query, and returns a typed selection any plugin
can load. The Microtime Shifter is the first consumer and the acceptance case.

# Background — reuse, do not reinvent
Existing infrastructure: `repository.py` (`list_samples()`, `search_samples(query,
limit)`, `list_artifacts(...)`; sample↔user via `measured_by_user_id`); `api.py`
(`list_samples()`, `list_artifacts(...)`); mfdb_admin `backend/services.py` RPC
handlers (`samples.list/search/get`, `artifacts.list`, all taking an `auth`
principal); the object store (`mfdb_object`: `object_uuid`, `content_md5`,
`original_filename`, `storage_path`; `mfdb_artifact`: `artifact_id`,
`artifact_kind`, `data_format`, `object_uuid`, `file_path`); and the ownership /
visibility pattern from the setup work (`created_by_user_id` + `is_public`, own +
public + builtin scoping). There is **no** reusable dataset/sample picker widget
today — this PRD adds one.

# Design
## 1. Widget surface (reusable)
New shared module `chisurf/gui/widgets/mfdb/dataset_browser.py`:
- `MfdbDatasetBrowser(QWidget)` — embeddable panel.
- `MfdbDatasetPickerDialog(QDialog)` — wraps the panel with OK/Cancel and a static
  convenience `pick_dataset(parent=None, kinds=None, formats=None, scope="mine")
  -> DatasetSelection | None` (returns the chosen dataset, or `None` on cancel).
  Consumers depend only on this small surface — not on MFDB internals.

## 2. Selection result (the contract)
```python
@dataclass
class DatasetSelection:
    artifact_id: str
    object_uuid: str | None
    kind: str               # artifact_kind
    data_format: str | None
    display_name: str
    sample_id: str | None
    sample_name: str | None
    metadata: dict
```
Bytes/handle resolution is **not** the widget's job: a consumer calls a
`datasets.open` RPC (or object-store helper) that materializes the object to a
local readable path and returns it. The widget only selects.

## 3. Scope / visibility toggle
A segmented control: **Mine** (default, datasets owned by the active user) ·
**Public** (datasets flagged public, any owner) · **All** (mine + public).
Visibility derives from owner + a public flag, reusing the own+public model.

**Decision (locked): owner/visibility on the artifact.** Add `created_by_user_id`
and `is_public` (default 0) to `mfdb_artifact`, dictionary-declared in
`mfdb_flr_ext.dic` with `_chisurf_schema` bridges, generated via the
dictionary-driven DDL path, gate-covered, `SCHEMA_VERSION` bump, backfilled.
Rationale: datasets exist without a sample (the shifter registers raw TTTR with no
sample), so artifact-level ownership is robust and keeps the browse query simple
and directly testable. `register_raw_measurement` / `register_result` stamp the
active user. Sample grouping still works via the artifact's sample/experiment link
when present. No blob.

## 4. Filter / query
A debounced **search box** doing server-side query across sample name, artifact
kind, data format, original filename, and date (the `search_samples` style,
extended to datasets). **Structured filters:** kind (multi-select, pre-seeded from
the consumer's `kinds`), data format, and a date range; the consumer's
`kinds`/`formats` pre-filter and lock the relevant facets. Filtering is
**server-side (RPC) with pagination**, so a large store stays responsive.

## 5. Layout
Two-pane with scope toggle + search bar on top: left, a filterable **sample list**
under the current scope; right, **datasets** under the selected sample (columns:
name/filename, kind, format, date, size, owner when scope ≠ Mine). Double-click or
select + OK returns the `DatasetSelection`. A **flat mode** ("All datasets") drops
sample grouping for a single filterable dataset table. Empty states: "No datasets
match" / "MFDB not connected".

## 6. Backend / RPC
Add a single unified browse RPC so the widget makes one call per page:
```
datasets.browse(scope, query, kinds, formats, sample_id, limit, offset, auth)
  -> { ok, datasets: [DatasetSelection-shaped dicts], total, samples: [{id,name,count}] }
```
User-scoped via `auth` (mirror `list_samples_handler`); paginated
(`limit`/`offset` + `total`) — never load the whole store at once.
`datasets.open(artifact_id, auth) -> { ok, local_path }` materializes object bytes
to a readable path. Do not duplicate query SQL — extend `repository.list_artifacts`
/ `search_samples` with the scope/kind/format/date filters and call them from the
handler.

## 7. Connectivity
If MFDB is not connected, the browser shows a disabled "MFDB not connected" state
and `pick_dataset` returns `None`; consumers fall back to their file-load path.

# Multiple owners — a dataset can be co-owned (general, all plugins)
**Problem.** Ownership is a single column (`mfdb_artifact.created_by_user_id`)
stamped by whoever registered the content first. Object-store content dedup means a
second user who loads/processes the *same* file does not create their own artifact
— the md5 lookup reuses the first user's artifact. So under a second user's **Mine**
scope it does not appear, contradicting the reality that a dataset is often used by
several people.

**Decision (locked): ownership is many-to-many.** A dictionary-declared join table
**`mfdb_artifact_owner`** `(artifact_id FK, user_id FK -> flr_sample_users(user_id),
role TEXT DEFAULT 'owner', created_at)`, `UNIQUE(artifact_id, user_id)`,
declared with `_chisurf_schema` bridges, generated, gate-covered, `SCHEMA_VERSION`
bump. `mfdb_artifact.created_by_user_id` stays the **original creator**
(back-compat/provenance); the join table is the authoritative owner set. Migration
backfills one row per existing artifact from `created_by_user_id`. Registration
`INSERT OR IGNORE`s the active user as a co-owner whenever content is registered or
reused by md5, so later users of the same content become co-owners of the one
artifact (no duplicate). Browse `mine` scope matches artifacts where the user is in
`mfdb_artifact_owner` (or, for legacy rows, equals `created_by_user_id`); every
co-owner sees the dataset under Mine. This is general — raw inputs and processed
outputs, every plugin. Helpers: `add_artifact_owner`, `list_artifact_owners`; the
browser may show an `Owners` count when scope ≠ Mine. This is *ownership*, distinct
from the existing ACL permission grants — keep them separate.

# File groups — a dataset is often a set of files (general, all plugins)
Whenever a plugin loads more than one file, those files are **one dataset** (a file
group). Many formats are acquired/processed as a group (some detector/acquisition
formats split channels/positions across files), and any multi-file load registers
as one experimental dataset, not N independent artifacts. The grouping lives in
core/MFDB and is shared by every plugin; the browser shows one entry per group, and
loading it hands the consumer the whole set.

**Model (dictionary-declared, no blob).** A **group artifact** (`mfdb_artifact`,
e.g. `artifact_kind="raw_measurement"` with `storage_mode="local_directory"` /
`"managed_archive"`, or a dedicated `data_format`) represents the dataset. A new
`.dic`-declared child table **`mfdb_artifact_member`** lists member files
`(artifact_id FK, object_uuid FK, filename, role, ordinal, audit columns)`; each
member is content-addressed (dedup per file). Declared with `_chisurf_schema`
bridges, generated, gate-covered, `SCHEMA_VERSION` bump. A single-file dataset is a
group with one member — the browser treats both uniformly.

**Registration.** Add `register_raw_measurement_group(files: list[str], ...)` (and
a grouped `register_result` variant) that stores each file (dedup per file), records
`mfdb_artifact_member` rows under one group artifact, with `derived_from`
provenance at the group level. Detect groups by explicit caller intent (the loader
passes a list) — do not guess from filenames in core; grouping policy belongs to
the GUI/loader.

**Browser + consumers.** `browse_datasets` returns the group as one row with a
member count/list on demand. `datasets.open` for a group materializes **all**
members to a local directory and returns it (or primary file + sidecars);
single-file datasets return the one path. `DatasetSelection` gains `member_count`
and a way to fetch member paths.

# Tasks
1. **Repository/query:** extend `list_artifacts` (or add `browse_datasets`) with
   scope, `kinds`, `formats`, date range, `sample_id`, pagination; resolve
   ownership/visibility per §3.
2. **RPC:** add `datasets.browse` and `datasets.open` handlers (user-scoped);
   register in the mfdb_admin/service dispatcher.
3. **Widget:** `MfdbDatasetBrowser` + `MfdbDatasetPickerDialog.pick_dataset(...)`;
   scope toggle, debounced search, structured filters, two-pane + flat mode,
   pagination, connectivity/empty states. Talks to MFDB **only via RPC**.
4. **Ownership/visibility schema:** dictionary-declared + generated + gate-covered,
   no blob.
5. **Microtime Shifter integration (test case):** the shifter's Load gains a "From
   MFDB…" action calling `pick_dataset(kinds=["raw_measurement"], formats=[TTTR
   formats])`; on selection it resolves the path via `datasets.open` and loads it
   through the existing pure API. The file dialog / drag-drop load remains for
   unregistered files.
6. **Tests** + a GUI construction smoke test for the browser.

# Microtime Shifter as the acceptance case
1. Register a TTTR file (raw_measurement) under user A. 2. Open the shifter → "From
MFDB…" → the browser lists A's dataset (Mine, filtered to TTTR raw measurements).
3. Select it → the shifter loads via `datasets.open` and shifts as normal. 4. User
B does not see A's dataset under Mine; sees it under Public only if it is public.

# Definition of Clean
Widget→MFDB via RPC only (query logic lives once in the repository); no blobs (new
ownership/visibility columns dictionary-declared, generated, gate-covered);
server-side filtering + pagination (never `SELECT *` the store into the client);
correct user scoping (Mine excludes others' private; Public shows public regardless
of owner; two-user test); DI over monkeypatching (in-process RPC client / explicit
`db_path`); GUI construction smoke test; behavior-asserting tests; reuse
mfdb_admin services (no forked query stack).

# Tests
Repository/handler: `datasets.browse` returns only the user's datasets under Mine,
includes public under Public/All; `kinds`/`formats`/date filters narrow; pagination
returns stable non-overlapping pages with correct `total`. Two-user scoping.
`datasets.open` materializes a readable path. Shifter integration round-trip (DI'd
RPC client, temp DB). GUI construction smoke test.

# Definition of Done
- Reusable browser + `pick_dataset(...)` with scope toggle, debounced query,
  structured filters, two-pane + flat mode, pagination, connectivity/empty states.
- `datasets.browse` / `datasets.open` RPC, user-scoped, paginated; query logic in
  the repository (no duplication, no client-side full scans).
- Ownership/visibility resolved per §3, dictionary-declared; gate stays green.
- Microtime Shifter loads a registered dataset end to end; the two-user scoping
  case passes.
- Widget→MFDB via RPC only; DI tests; GUI smoke test; behavior-asserting tests; all
  green together with the dict gate.

# Known bug — processed datasets never appear in the browser
Reported: "I processed a dataset and expected it to appear in the MFDB dataset
load; it does not." Root cause: registration fails **silently**, so nothing is
stored (`register_result` logs a warning and returns `""`). Two independent
failures, both swallowed by best-effort registration:

- **Bug A — `operation_type="microtime_shift"` is not in the vocabulary.**
  `OPERATION_TYPES` (`chisurf/core/mfdb/models.py`) is a fixed tuple validated by
  `validate_extensible_vocab`. The shifter registers with an unknown
  `operation_type` → `record_operation` raises → the whole `register_result`
  transaction rolls back → the processed artifact is never stored → never appears.
  (Same class as an earlier `g_factor_processing` issue worked around with
  `"calibration"`.) **Fix:** add `microtime_shift` (and other new operation types)
  to `OPERATION_TYPES`, or make `validate_extensible_vocab` genuinely extensible
  (accept values registered in the `mfdb_vocabulary` table). Prefer adding to the
  canonical tuple.
- **Bug B — `created_by_user_id` FK fails when the active user has no row.** The
  new artifact owner column references `flr_sample_users(user_id)`, and
  registration stamps `_resolve_active_user_id()` which returns the configured
  `default_user_id` (e.g. "tpeulen"). The schema only bootstraps `user_default`
  and `guest` rows, so a configured user with no `flr_sample_users` row fails the
  FK and rolls back **all** registration (raw and processed). **Fix:** ensure the
  resolved active user exists before stamping (bootstrap/`INSERT OR IGNORE`, or
  fall back to `user_default` when the configured id has no row). A configured
  `default_user_id` pointing at a non-existent user must not break all
  registration.
- **Bug C (meta) — best-effort registration hides real bugs.** A vocab rejection
  or FK violation is a *real* error, not the "MFDB-unavailable" condition
  best-effort is meant to tolerate; swallowing it as a warning and returning `""`
  caused **silent data loss**. Distinguish: MFDB unavailable / no DB → soft warn
  (fine); a registration error when a DB *is* present (FK, vocab, integrity) →
  surface it (visible warning / raise in non-GUI callers).

Add a regression test that registers a `processed_data` artifact via the real
`register_result` under a configured non-default user and asserts it appears in
`browse_datasets` — this would have caught both bugs.

# Relationships
- Consumed first by [PRD-09](prd-09.md) (microtime shifter).
- Ownership/visibility model reused by the study entity in [PRD-13](prd-13.md).
- Gains a study facet from [PRD-13](prd-13.md) (`browse_datasets` `study_id` filter).
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).
