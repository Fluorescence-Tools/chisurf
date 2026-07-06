---
type: PRD
prd: "15"
title: "PRD-15: Lightweight Reagent / Consumable Inventory"
description: Track consumables (dye lots, buffers, filters, kits) with lot/expiry and link them to operations, setups, and samples for reproducibility.
status: done
phase: "4"
resource: chisurf/core/mfdb
tags: [prd, mfdb, lims]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
This PRD adds a minimal consumable inventory so a result ties to exactly what was used — which dye lot, which filter set, which buffer. `mfdb_reagent_lot` records lots with vendor, catalog/lot number, concentration, and expiry, and an orthogonal many-to-many `mfdb_reagent_usage` links a lot to an operation, setup, or sample without adding columns to those tables. A fluorophore lot may optionally reference a known probe, connecting inventory to the fluorophore database. The repository API covers lot CRUD, usage links, and expiry queries (including a QC "expired lots" helper), and an admin view lists/filters lots and shows per-lot usage. It is the lowest-priority LIMS layer, intended for shared-MFDB or strict-reproducibility use.

# Status
Done. Dictionary-declared lot and usage tables, the reagents API, and an mfdb-admin view landed (headless + view tests). Reworked to drop the schema-version bump and hard foreign keys per the disposable-DB policy.

# Goal
Track consumables — fluorophore lots, buffers, optical filters, kits — with
lot/expiry, and reference them from operations/setups, so a result ties to exactly
what was used. Reproducibility: "which dye lot, which filter set, which buffer."

# Background
- LIMS reference: a mature LIMS tracks commercial kits, per-user lots, library
  kits, index kits, and instruments-in-lab — kits/lots/expiry tracked and tied to
  runs.
- MFDB today: instruments and setups are modelled, but consumables (dye lots,
  buffers, filters) are not inventoried. Lower priority for a single-lab tool;
  valuable as MFDB is shared and for strict reproducibility.

# Design (dictionary-driven, minimal)
- **`mfdb_reagent_lot`** (`.dic`-declared, generated, gate-covered):
  `(lot_id PK, kind, name, vendor, catalog_no, lot_number, concentration,
  concentration_units, received_at, opened_at, expiry, created_by_user_id, details,
  created_at, updated_at, deleted_at)`. `kind` ∈ {`fluorophore`, `buffer`,
  `optical_filter`, `kit`, `other`} (extensible vocab). Index on `(kind, expiry)`.
- **Usage links (many-to-many):** `mfdb_reagent_usage` (`.dic`-declared):
  `(id PK, lot_id FK mfdb_reagent_lot, target_type, target_id, role, created_at)`,
  where `target_type` ∈ {`operation`, `setup`, `sample`} (extensible vocab). Links
  a lot to an operation (what was used in a measurement/processing step), a setup
  (e.g. installed filter set), or a sample (e.g. labeling dye lot). Keeps the
  reagent record orthogonal to the rest of the schema — no columns added to
  operation/setup/sample.
- **Optional probe link:** where a fluorophore lot corresponds to a known probe,
  allow `mfdb_reagent_lot` → `probes(probe_id)` via a nullable FK, connecting the
  inventory to the existing fluorophore database ([PRD-06](prd-06.md)).

# API
- `add_reagent_lot(kind, name, lot_number, expiry=None, ...)` → lot_id.
- `link_reagent(lot_id, target_type, target_id, role="used")`,
  `list_reagents_for(target_type, target_id)`, `list_lots(kind=None,
  include_expired=False)`.
- Expiry helper: `expired_lots(as_of=now)` for QC/warnings.

# Tasks
1. `.dic` + schema: declare `mfdb_reagent_lot` + `mfdb_reagent_usage` (+ kind /
   target_type vocab); generate DDL; `SCHEMA_VERSION` bump; add to the gate.
2. Repository: lot CRUD, usage links, expiry queries.
3. Optional wiring: let setups record installed filter/laser lots, and let
   operations/registration optionally record reagent usage (best-effort).
4. mfdb-admin: a Reagent Lots entity (dictionary-sourced columns) + a usage view
   ("what used this lot" / "what lots did this operation use").
5. Tests: lot CRUD; usage many-to-many; expiry query; dict gate green; GUI smoke.

# Status — complete (headless core + mfdb-admin view)
`mfdb_reagent_lot` + `mfdb_reagent_usage` are dictionary-declared in
`mfdb_flr_ext.dic` (auto-created via `reconcile_schema`; audit columns
auto-appended). The API lives in `chisurf/core/mfdb/reagents.py`: `add_reagent_lot`
(validates `kind`), `link_reagent` (idempotent, validates `target_type`),
`list_reagents_for` (the "what was used" query), `list_lots` (kind filter, excludes
expired by default), `expired_lots` (QC, `as_of`-aware). Tests:
`test/fio/test_reagents.py` (6).

Reworked from the original sketch: no `SCHEMA_VERSION` bump (the version chain was
removed in PRD-19) and no hard `usage → lot` FK (the code key isn't emitted as a
PK; the repository manages integrity — the same convention as `mfdb_study_member`).

# Definition of Done
- [x] `mfdb_reagent_lot` + `mfdb_reagent_usage` exist, dict-declared, generated;
      lots carry lot_number/expiry.
- [x] Lots link to operations/setups/samples (many-to-many) without adding columns
      to those tables; expiry is queryable.
- [x] Reproducibility query "what was used" works (`list_reagents_for`).
- [x] Headless tests pass.
- [x] mfdb-admin lists lots, filters by kind, toggles expired, shows per-lot
      detail, and creates lots — `gui/reagents_view.py` (`ReagentLotsView`) over
      `mfdb.reagents.*` handlers + `MFDBClient` methods. Headless-tested
      (`test_reagent_handlers.py` 6, `test_reagents_view.py` 6) and
      screenshot-verified offscreen.

# Definition of Clean
`.dic` dictates the schema (no hardcoded SQL/blob); orthogonal linking table (no
schema pollution of operation/setup/sample); reuse extensible vocab for
kind/target_type; behavior-asserting tests; DI over monkeypatching; GUI smoke.

# Scope note
Lowest priority of the LIMS layers P1–P4. Implement when MFDB is shared across
users or when strict consumable reproducibility is required; the schema is
intentionally minimal and can grow (storage location, quantity tracking) if it
becomes a real inventory-management need.

# Relationships
- May link fluorophore lots to probe records from [PRD-06](prd-06.md).
- Links to operations (via [PRD-11](prd-11.md) nodes), setups, and samples orthogonally.
- Shares the extensible-vocabulary and own+public machinery with the other LIMS layers ([PRD-12](prd-12.md), [PRD-13](prd-13.md), [PRD-14](prd-14.md)).
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).
