# PRD-15: Lightweight Reagent / Consumable Inventory (LIMS P4)

## Goal

Track consumables — fluorophore lots, buffers, optical filters, kits — with
lot/expiry, and reference them from operations/setups, so a result ties to exactly
what was used. Reproducibility: "which dye lot, which filter set, which buffer."

## Background

- LIMS reference: iSkyLIMS `CommercialKits`, `UserLotCommercialKits`,
  `LibraryKit`, `CollectionIndexKit`, `SequencerInLab` — kits/lots/expiry tracked
  and tied to runs.
- MFDB today: instruments and setups are modelled, but consumables (dye lots,
  buffers, filters) are not inventoried. Lower priority for a single-lab tool;
  valuable as MFDB is shared and for strict reproducibility.
- See `overhaul/MFDB-LIMS-diagnosis.md` (P4).

## Design (dictionary-driven, minimal)

- **`mfdb_reagent_lot`** (`.dic`-declared, generated, gate-covered):
  `(lot_id PK, kind, name, vendor, catalog_no, lot_number, concentration,
   concentration_units, received_at, opened_at, expiry, created_by_user_id,
   details, created_at, updated_at, deleted_at)`. `kind` ∈ {`fluorophore`,
   `buffer`, `optical_filter`, `kit`, `other`} (extensible vocab). Index on
   `(kind, expiry)`.
- **Usage links (many-to-many):** `mfdb_reagent_usage` (`.dic`-declared):
  `(id PK, lot_id FK mfdb_reagent_lot, target_type, target_id, role, created_at)`,
  where `target_type` ∈ {`operation`, `setup`, `sample`} (extensible vocab). Links
  a lot to an operation (what was used in a measurement/processing step), a setup
  (e.g. installed filter set), or a sample (e.g. labeling dye lot). Keeps the
  reagent record orthogonal to the rest of the schema — no columns added to
  operation/setup/sample.
- **Optional probe link:** where a fluorophore lot corresponds to a known probe,
  allow `mfdb_reagent_lot` → `probes(probe_id)` via a nullable FK, connecting the
  inventory to the existing fluorophore database.

## API

- `add_reagent_lot(kind, name, lot_number, expiry=None, ...)` → lot_id.
- `link_reagent(lot_id, target_type, target_id, role="used")`,
  `list_reagents_for(target_type, target_id)`, `list_lots(kind=None,
  include_expired=False)`.
- Expiry helper: `expired_lots(as_of=now)` for QC/warnings.

## Tasks

1. `.dic` + schema: declare `mfdb_reagent_lot` + `mfdb_reagent_usage` (+ kind /
   target_type vocab); generate DDL; `SCHEMA_VERSION` bump; add to the gate.
2. Repository: lot CRUD, usage links, expiry queries.
3. Optional wiring: let setups record installed filter/laser lots, and let
   operations/registration optionally record reagent usage (best-effort).
4. mfdb-admin: a Reagent Lots entity (dictionary-sourced columns) + a usage view
   ("what used this lot" / "what lots did this operation use").
5. Tests: lot CRUD; usage many-to-many; expiry query; dict gate green; GUI smoke.

## Definition of Done

- [ ] `mfdb_reagent_lot` + `mfdb_reagent_usage` exist, dict-declared, generated,
      gate-covered; lots carry lot_number/expiry.
- [ ] Lots link to operations/setups/samples (many-to-many) without adding
      columns to those tables; expiry is queryable.
- [ ] Admin lists lots and their usage; reproducibility query "what was used"
      works.
- [ ] Tests pass.

## Definition of Clean

`.dic` dictates the schema (no hardcoded SQL/blob); orthogonal linking table (no
schema pollution of operation/setup/sample); reuse extensible vocab for
kind/target_type; behavior-asserting tests; DI over monkeypatching; GUI smoke.

## Scope note

Lowest priority of P1–P4. Implement when MFDB is shared across users or when
strict consumable reproducibility is required; the schema is intentionally minimal
and can grow (storage location, quantity tracking) if it becomes a real
inventory-management need.
