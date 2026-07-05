---
type: PRD
prd: "15"
title: "PRD-15: Lightweight Reagent / Consumable Inventory"
description: Track consumables (dye lots, buffers, filters, kits) with lot/expiry and link them to operations, setups, and samples for reproducibility.
status: done
phase: "4"
resource: overhaul/PRD-15-reagent-inventory.md
tags: [prd, mfdb, lims]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
This PRD adds a minimal consumable inventory so a result ties to exactly what was used — which dye lot, which filter set, which buffer. `mfdb_reagent_lot` records lots with vendor, catalog/lot number, concentration, and expiry, and an orthogonal many-to-many `mfdb_reagent_usage` links a lot to an operation, setup, or sample without adding columns to those tables. A fluorophore lot may optionally reference a known probe, connecting inventory to the fluorophore database. The repository API covers lot CRUD, usage links, and expiry queries (including a QC "expired lots" helper), and an admin view lists/filters lots and shows per-lot usage. It is the lowest-priority LIMS layer, intended for shared-MFDB or strict-reproducibility use.

# Status
Done. Dictionary-declared lot and usage tables, the reagents API, and an mfdb-admin view landed (headless + view tests). Reworked to drop the schema-version bump and hard foreign keys per the disposable-DB policy.

# Relationships
- May link fluorophore lots to probe records from [PRD-06](prd-06.md).
- Links to operations (via [PRD-11](prd-11.md) nodes), setups, and samples orthogonally.
- Shares the extensible-vocabulary and own+public machinery with the other LIMS layers ([PRD-12](prd-12.md), [PRD-13](prd-13.md), [PRD-14](prd-14.md)).
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-15-reagent-inventory.md`
