---
type: PRD
prd: "48"
title: "PRD-48: Provider-Agnostic ELN Integration"
description: A provider-agnostic ELN integration layer for MFDB with a single gateway abstraction and two concrete electronic-lab-notebook backends, supporting bidirectional deposit, import, and reconciliation.
status: draft
phase: "unassigned"
resource: overhaul/PRD-48-eln-integration.md
tags: [prd, mfdb, eln]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
MFDB is the prototype implementation for a planned public fluorescence databank, so it must interoperate with the community's electronic lab notebook (ELN) platforms rather than live as an island. This PRD specifies a provider-agnostic `ElnGateway` abstraction under `chisurf/core/mfdb/eln/`, with a neutral entity model (Record, Resource, Chemical, Instrument, Link, Scope) and two concrete adapters plus a declared capability set so callers degrade gracefully. Integration is bidirectional — push (deposit records + attachments + links), pull (import chemicals/resources/instruments with CAS/InChIKey dedup), and conflict reconciliation — keeping a clean split between deposition and DOI dissemination. The guiding invariant is one identity per real-world object; users and groups are matched, never provisioned, and credentials live in the OS store per PRD-37.

# Status
Draft / unassigned (STATUS TABLE authoritative). Design complete; phased 1–4 from prime-backend push through import, secondary backend, and reconciliation/dissemination/GUI.

# Relationships
- Extends the [MFDB (current)](/architecture/mfdb.md) toward the [MFDB target](/specs/mfdb.md); mirrors the project-archiver decomposition into operations + artifacts + edges.
- Keeps `api.py` transport-agnostic per the [RPC target](/specs/rpc.md).
- Related to fluorophore-identity and chemical-identity work; CLI-first per repo headless rule.

# Source
- Primary: `overhaul/PRD-48-eln-integration.md`
