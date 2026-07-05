---
type: PRD
prd: "26"
title: "PRD-26: Model-Driven Data Layer"
description: Extends dictionary-dictates-schema to dictionary-dictates-the-system — generating the repository/DAO, admin entity registry, RPC parameter validation, and API/schema docs from the same .dic.
status: in-progress
phase: "2"
resource: overhaul/PRD-26-model-driven-data-layer.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-26 extends "the dictionary dictates the schema" to "the dictionary dictates the system": a single generator over the parsed dictionary produces typed repository/DAO accessors, the admin entity registry, RPC parameter validation, and API/schema documentation, so the hand-maintained surface and the drift between schema, ORM, admin, and validation largely disappear. Generated DAO accessors use fully parameterised SQL with identifiers whitelisted against the live schema, closing the f-string injection surface, and apply soft-delete and scoping conventions automatically. Boundary validation is derived from the dictionary (types, units, bounds, required, repeatable), and a generated reference document stays in sync with the schema. Altering a field becomes a `.dic` edit that propagates to every layer.

# Status
In-progress (re-verified against code 2026-07-05). Landed: the `DictionaryDao` generator with parameterised, schema-whitelisted SQL, the dictionary-derived boundary validator, the docs generator, and admin registry derivation. **Open:** the DoD items "remove the hand-maintained copies" and "no f-string SQL" are unmet — ~33 f-string SQL statements remain in `repository.py` and the insert (`add_*`/`save_*`) CRUD family is still hand-written. A transparent residual, not a regression.

# Relationships
- Builds directly on [PRD-19](prd-19.md) (canonical schema plus reconcile generator) and [PRD-11](prd-11.md) (operation parameter schemas).
- Subsumes most hand-maintained admin/validation work from [PRD-02b](prd-02b.md) and per-plugin parameter handling.
- Supplies the dictionary-driven boundary validation used by [PRD-25](prd-25.md).
- Targets the [MFDB target](/specs/mfdb.md); see [MFDB (current)](/architecture/mfdb.md).

# Source
- Primary: `overhaul/PRD-26-model-driven-data-layer.md`
