# PRD-10: MFDB Dataset Browser — reusable data-load widget

## Goal

A reusable Qt widget that lets a user pick a **registered MFDB dataset** instead
of hunting for a file on disk. It lists the active user's datasets (and, when
toggled, public ones), with a fast filter/query, and returns a typed selection
that any plugin can load. The Microtime Shifter (PRD-09) is the first consumer
and the acceptance test case.

## Background — reuse, do not reinvent

Existing infrastructure (read first):

- `chisurf/core/mfdb/repository.py` — `list_samples()`, `search_samples(query,
  limit)`, `list_artifacts(...)`; sample↔user via `measured_by_user_id`.
- `chisurf/core/mfdb/api.py` — `list_samples()`, `list_artifacts(...)`.
- `chisurf/plugins/core/mfdb_admin/backend/services.py` — RPC handlers
  `samples.list`, `samples.search`, `samples.get`, `artifacts.list`, all taking an
  `auth` principal for user scoping.
- Object store: `mfdb_object` (`object_uuid`, `content_md5`, `original_filename`,
  `storage_path`); `mfdb_artifact` (`artifact_id`, `artifact_kind`,
  `data_format`, `object_uuid`, `file_path`).
- Ownership/visibility pattern from PRD-04: `created_by_user_id` + `is_public`
  (own + public + builtin scoping in the setup loaders).

There is **no** reusable dataset/sample picker widget today — this PRD adds one.

## Design

### 1. Widget surface (reusable)

New shared module `chisurf/gui/widgets/mfdb/dataset_browser.py`:

- `MfdbDatasetBrowser(QWidget)` — embeddable panel (the browser proper).
- `MfdbDatasetPickerDialog(QDialog)` — wraps the panel with OK/Cancel and a
  static convenience:

  ```python
  MfdbDatasetPickerDialog.pick_dataset(
      parent=None,
      kinds: list[str] | None = None,      # e.g. ["raw_measurement"]
      formats: list[str] | None = None,    # e.g. ["ptu","ht3","spc"]
      scope: str = "mine",                 # "mine" | "public" | "all"
  ) -> DatasetSelection | None
  ```

  Returns the chosen dataset, or `None` on cancel. Consumers depend only on this
  small surface — not on MFDB internals.

### 2. Selection result (the contract)

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
    # resolution: a consumer calls resolve_local_path() (below) to get bytes
```

Bytes/handle resolution is **not** the widget's job: a consumer calls a
`datasets.open` RPC (or object-store helper) that materializes the object to a
local readable path and returns it. The widget only selects.

### 3. Scope / visibility toggle

A segmented control at the top: **Mine** (default) · **Public** · **All**.

- **Mine** — datasets owned by the active user.
- **Public** — datasets flagged public (any owner).
- **All** — mine + public (i.e. everything visible to the user).

Visibility derives from the dataset's owner + a public flag. Reuse the PRD-04
own+public model.

**Decision (locked): owner/visibility on the artifact (option b).** Add
`created_by_user_id` and `is_public` (default 0) to `mfdb_artifact`,
dictionary-declared in `mfdb_flr_ext.dic` with `_chisurf_schema` bridges,
generated via the PRD-04 Task-P1a path, gate-covered, `SCHEMA_VERSION` bump,
backfilled. Rationale: datasets exist without a sample (the shifter registers raw
TTTR with no sample), so artifact-level ownership is robust and keeps the browse
query simple and directly testable. `register_raw_measurement` / `register_result`
stamp the active user. Sample grouping still works via the artifact's
sample/experiment link when present; ungrouped otherwise. No blob.

### 4. Filter / query

- A **search box** (debounced) doing server-side query across: sample name,
  artifact kind, data format, original filename, and date — matching the
  `search_samples` style, extended to datasets.
- **Structured filters:** kind (multi-select, pre-seeded from the consumer's
  `kinds`), data format, and a date range. The consumer's `kinds`/`formats`
  pre-filter and lock the relevant facets (e.g. the shifter only shows TTTR raw
  measurements).
- Filtering is **server-side** (RPC) with pagination, so a large store stays
  responsive; the client only refines the current page.

### 5. Layout

Two-pane, with the scope toggle + search bar across the top:

- **Left:** filterable **sample list** (samples visible under the current scope).
- **Right:** **datasets** registered under the selected sample — a table with
  columns: name/filename, kind, format, date, size, (owner when scope ≠ Mine).
- Double-click a dataset (or select + OK) returns the `DatasetSelection`.
- Provide a **flat mode** toggle ("All datasets") that drops the sample grouping
  and shows a single filterable dataset table — for quick search when the user
  does not care about the sample.
- Empty states: "No datasets match" / "MFDB not connected" (see §7).

### 6. Backend / RPC

Reuse existing handlers where possible; add a single unified browse RPC so the
widget makes one call per page:

```
datasets.browse(scope, query, kinds, formats, sample_id, limit, offset, auth)
  -> { "ok": true, "datasets": [DatasetSelection-shaped dicts],
       "total": int, "samples": [{id, name, count}] }
```

- User-scoped via the `auth` principal (mirror `list_samples_handler`).
- Pagination (`limit`/`offset` + `total`) — never load the whole store at once.
- `datasets.open(artifact_id, auth) -> { ok, local_path }` materializes the object
  bytes to a readable path for the consumer.
- Do not duplicate query SQL — extend `repository.list_artifacts` /
  `search_samples` with the scope/kind/format/date filters and call them from the
  handler.

### 7. Connectivity

- If MFDB is not connected/available, the browser shows a disabled "MFDB not
  connected" state and `pick_dataset` returns `None`; consumers fall back to their
  file-load path (consistent with PRD-09's connectivity-aware design).

## Multiple owners — a dataset can be co-owned (general, all plugins)

**Current behavior (the problem).** Ownership is a single column
(`mfdb_artifact.created_by_user_id`) stamped by whoever registered the content
first. Object-store content dedup means a second user who loads/processes the
*same* file does **not** create their own artifact — `_lookup_or_register_raw`
(and any md5 lookup) reuses the existing artifact owned by the first user. So the
dataset is owned solely by the first registrant; under a second user's **Mine**
scope it does not appear (they'd see it only under Public/All). That contradicts
the reality that the same experimental dataset is often used/processed by several
people.

**Decision (locked): ownership is many-to-many.** A dataset (artifact) can have
multiple owners; content dedup keeps one artifact per content while the owner set
grows.

- Add a dictionary-declared join table **`mfdb_artifact_owner`**:
  `(artifact_id FK -> mfdb_artifact, user_id FK -> flr_sample_users(user_id),
  role TEXT DEFAULT 'owner', created_at)`, `UNIQUE(artifact_id, user_id)`.
  Declared in `mfdb_flr_ext.dic` with `_chisurf_schema` bridges, generated,
  gate-covered, `SCHEMA_VERSION` bump.
- Keep `mfdb_artifact.created_by_user_id` as the **original creator** (back-compat
  and provenance); the join table is the authoritative *owner set*. Migration
  backfills one `mfdb_artifact_owner` row per existing artifact from its
  `created_by_user_id`.
- **Registration adds the active user as a co-owner.** Whenever a user registers
  or reuses an artifact by content (the md5-lookup path), `INSERT OR IGNORE` the
  active user into `mfdb_artifact_owner`. First registrant creates the artifact +
  becomes owner #1; later users registering the same content become co-owners of
  the same artifact (no duplicate artifact).
- **Browse scope unions the owner set.** `browse_datasets` `own`/`mine` scope
  matches artifacts where the user is in `mfdb_artifact_owner` (or, for legacy
  rows, equals `created_by_user_id`). So every co-owner sees the dataset under
  Mine. `public`/`all` unchanged.
- This is general: it applies to raw inputs *and* processed outputs, for every
  plugin — not shifter-specific. (A derived/processed artifact is owned by whoever
  produced it; if two users independently produce byte-identical output, dedup
  yields one artifact co-owned by both.)
- Helper APIs: `add_artifact_owner(artifact_id, user_id)`,
  `list_artifact_owners(artifact_id)`; the browser may show an `Owners` column /
  count when scope ≠ Mine.

Relationship to ACL: this is *ownership* (who the dataset belongs to / sees it
under Mine), distinct from the existing `mfdb_acl_entry` permission grants. Keep
them separate; do not overload ACL for ownership here.

## File groups — a dataset is often a set of files (general, all plugins)

**This is a general, core concept, not browser-specific: whenever a plugin loads
more than one file, those files are always one dataset (a file group).** Many
formats are acquired/processed as a group (e.g. Becker & Hickl `.spc` measurements
are typically a set of files; some setups split channels/positions across files),
and any multi-file load — the Microtime Shifter included — registers as **one
experimental dataset**, not N independent artifacts. The grouping lives in
core/MFDB and is shared by every plugin (via the registration API below); the
browser shows one entry per group, and loading it hands the consumer the whole
set.

### Model (dictionary-declared, no blob)

- A **group artifact** (`mfdb_artifact`, e.g. `artifact_kind = "raw_measurement"`
  with `storage_mode = "local_directory"` / `"managed_archive"`, or a dedicated
  `data_format` marking a group) represents the dataset.
- A new `.dic`-declared child table **`mfdb_artifact_member`** lists the member
  files: `(artifact_id FK, object_uuid FK, filename, role, ordinal,
  created_at/updated_at/deleted_at)`. Each member file is content-addressed in the
  object store (dedup applies per file). Declared in `mfdb_flr_ext.dic` with
  `_chisurf_schema` bridges, generated, gate-covered, `SCHEMA_VERSION` bump.
- A single-file dataset is just a group with one member (or no member row + the
  artifact's own object) — the browser treats both uniformly.

### Registration

- Add a `register_raw_measurement_group(files: list[str], ...)` (and a grouped
  variant of `register_result`) that stores each file in the object store
  (dedup per file) and records the `mfdb_artifact_member` rows under one group
  artifact, with `derived_from` provenance at the group level.
- Detect/accept groups by explicit caller intent (the loader passes a list) — do
  not guess from filenames in the core; grouping policy (e.g. "all `.spc` in a
  folder") belongs to the GUI/loader, not the registry.

### Browser + consumers

- `browse_datasets` returns the **group** as one dataset row, with a member count
  / member list available on demand.
- `datasets.open` for a group materializes **all** member files to a local
  directory and returns the directory (or the primary file + sidecars) so the
  consumer can load the set. Single-file datasets return the one path.
- The picker's `DatasetSelection` gains `member_count` and a way to fetch member
  paths.

This makes "process a `.spc` group → one dataset in the browser" work, and keeps
per-file object-store dedup.

## Tasks

1. **Repository/query:** extend `list_artifacts` (or add `browse_datasets`) with
   scope (own/public/all), `kinds`, `formats`, date range, `sample_id`, and
   pagination; resolve ownership/visibility per the §3 decision.
2. **RPC:** add `datasets.browse` and `datasets.open` handlers (user-scoped via
   `auth`); register them in the mfdb_admin/service dispatcher.
3. **Widget:** `MfdbDatasetBrowser` + `MfdbDatasetPickerDialog.pick_dataset(...)`
   in `chisurf/gui/widgets/mfdb/dataset_browser.py`; scope toggle, debounced
   search, structured filters, two-pane + flat mode, pagination controls,
   connectivity/empty states. The widget talks to MFDB **only via RPC**.
4. **Ownership/visibility schema** (if §3 option (b) is chosen, or to add
   `is_public` to samples): dictionary-declared + generated + gate-covered. No
   blob.
5. **Microtime Shifter integration (test case):** PRD-09's Load gains a "From
   MFDB…" action that calls
   `MfdbDatasetPickerDialog.pick_dataset(kinds=["raw_measurement"],
   formats=[supported TTTR formats])`; on selection it resolves the local path via
   `datasets.open` and loads it through the existing pure API. The file dialog /
   drag-drop load remains for unregistered files (which then register, per PRD-09).
6. **Tests** (see below) + a GUI construction smoke test for the browser.

## Microtime Shifter as the acceptance case

The shifter must demonstrate the full loop end to end:

1. Register a TTTR file in MFDB (raw_measurement) under user A.
2. Open the shifter → "From MFDB…" → the browser lists user A's dataset (Mine
   scope), filtered to TTTR raw measurements.
3. Select it → the shifter loads it via `datasets.open` and shifts as normal.
4. A second user B does not see A's dataset under "Mine"; sees it under "Public"
   only if it is public.

## Definition of Clean (same bar as PRD-09)

1. Widget talks to MFDB **only via RPC**; no direct repository/DB calls in the Qt
   layer. Query logic lives once in the repository, reused by the handler.
2. No blobs: any new ownership/visibility column is dictionary-declared,
   generated, and covered by the total-coverage `validate_mapping()` gate.
3. **Server-side filtering + pagination** — never `SELECT *` the whole store into
   the client.
4. Correct user scoping: Mine excludes other users' private datasets; Public
   shows public regardless of owner; verified by a two-user test.
5. **DI over monkeypatching** in tests (inject an in-process RPC client / explicit
   `db_path`), not `monkeypatch.setattr` of internals.
6. GUI **construction smoke test** for the browser (mirror
   `test/gui/test_detector_wizard_page.py`); every Qt symbol imported.
7. Tests assert behavior (scope filtering, query matches, pagination boundaries,
   the shifter round-trip), not mere existence.
8. Reuse `mfdb_admin` services/patterns; do not fork a second sample/artifact
   query stack.

## Tests

- Repository/handler: `datasets.browse` returns only the user's datasets under
  Mine; includes public under Public/All; `kinds`/`formats`/date filters narrow
  correctly; pagination returns stable, non-overlapping pages with correct
  `total`.
- Two-user scoping: user B cannot see user A's private dataset; can see A's public
  one.
- `datasets.open` materializes a readable local path for a registered object.
- Shifter integration: pick-from-MFDB → load → shift round-trip (DI'd RPC client,
  temp DB).
- GUI construction smoke test for `MfdbDatasetBrowser`.

## Definition of Done

- [ ] Reusable `MfdbDatasetBrowser` + `MfdbDatasetPickerDialog.pick_dataset(...)`
      with scope toggle (Mine/Public/All), debounced query, structured filters,
      two-pane + flat mode, pagination, connectivity/empty states.
- [ ] `datasets.browse` / `datasets.open` RPC, user-scoped, paginated; query logic
      lives in the repository (no duplication, no client-side full scans).
- [ ] Ownership/visibility resolved per §3, dictionary-declared if a new column is
      added; gate stays green.
- [ ] Microtime Shifter loads a registered dataset via the widget end to end; the
      two-user scoping case passes.
- [ ] Widget→MFDB via RPC only; DI tests; GUI smoke test; behavior-asserting
      tests; all green together with the dict gate.
