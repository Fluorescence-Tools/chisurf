---
type: PRD
prd: "45"
title: "PRD-45: Chemical registry-number as a first-class chemical identity in MFDB"
description: Promotes the chemical registry number from an ad-hoc free-text property to a dictionary-defined, validated, indexed, cross-entity chemical identity surfaced across GUI, CLI, and RPC.
status: draft
phase: "unassigned"
resource: overhaul/PRD-45-cas-chemical-identity.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-45 makes a standard chemical registry-number identity (CAS) a general, first-class feature of MFDB rather than an ad-hoc optical property on probes. A chemical registry number uniquely identifies a substance, so it is the natural join key for dyes and compounds across the reference set, samples, reagents, and public fluorophore spectral databases — the chemistry analogue of the sequence/structure accessions handled by [PRD-39](prd-39.md). The design declares the identifier in the `.dic` schema authority, validates and normalizes it (including its check digit) on every write, gives it a real indexed column or normalized identity table, shares one identity across reference probes / sample probes / reagent lots, and surfaces it through AutoForm view schemes, list filters, the CLI, and the RPC API, with an optional offline-safe external resolver. A minimal property-bag foundation already exists and is forward-compatible.

# Status
Draft (unassigned phase, STATUS TABLE authoritative). Design only — implementation deferred; a minimal property-bag groundwork (alias canonicalization, whitespace-tolerant lookup, a register convenience param) is in place.

# Relationships
- Chemistry analogue of [PRD-39](prd-39.md), which gives sequence/structure accessions the same first-class treatment for proteins.
- Depends on the vendor-neutral dictionary namespace from [PRD-44](prd-44.md) (the identifier is a `_mfdb_schema`-mapped item).
- Follows the fail-loud policy of PRD-25 for malformed values.
- Extends the [MFDB (current)](/architecture/mfdb.md) toward the [MFDB target](/specs/mfdb.md); surfaced via [GUI & AutoForm](/subsystems/gui-autoform.md) and the [RPC target](/specs/rpc.md).

# Source
- Primary: `overhaul/PRD-45-cas-chemical-identity.md`
