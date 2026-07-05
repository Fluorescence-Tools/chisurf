---
type: PRD
prd: "41"
title: "PRD-41: FDB4ChemBio Access-Layer & Interoperability Strategy"
description: A design note fixing the architectural boundary for MFDB as a prototype public resource — the dictionary is the product, deposition and dissemination differ, and a future read-only GraphQL endpoint is generated from the dictionary.
status: draft
phase: "unassigned"
resource: overhaul/PRD-41-fdb4chembio-access-layer-strategy.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-41 is a strategy note, not an implementation PRD: it sets the architectural boundary that downstream MFDB implementation PRDs must respect and records a query-technology decision so it is not re-litigated. Its principles are that the durable, ownable asset is the dictionary-conformant data model (and CIF file), not any database engine or API; that the prototype clock (SQLite, in-process API plus ZMQ RPC) must not block the eventual worldwide-resource clock; and that deposition (authenticated, validated, permissioned writes) is a different problem from dissemination (public, cacheable, federatable reads). It approves GraphQL only as a future, read-only, dictionary-generated dissemination endpoint — one interface among several, kept out of the deposition path and explicitly not built during the prototype phase.

# Status
Draft (unassigned phase, STATUS TABLE authoritative). Strategy and the GraphQL decision are recorded; the prototype guardrails are tracked under existing PRDs.

# Relationships
- Names PRD-02a (mmCIF dictionary infrastructure), [PRD-37](prd-37.md) (transport/authz hardening of the write path), and [PRD-39](prd-39.md) (cross-references, CIF round-trip) as the critical-path work; an API technology choice is not.
- Made possible by [PRD-44](prd-44.md), which de-brands the dictionary extension namespace — a precondition for disseminating the dictionary as a product.
- Treats the `.dic` family as schema authority alongside PRD-19 and PRD-26.
- Constrains the [MFDB (current)](/architecture/mfdb.md), [MFDB target](/specs/mfdb.md), and [RPC target](/specs/rpc.md).

# Source
- Primary: `overhaul/PRD-41-fdb4chembio-access-layer-strategy.md`
