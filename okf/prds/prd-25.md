---
type: PRD
prd: "25"
title: "PRD-25: Consistency Hardening + Correctness Primitives"
description: A set of cross-cutting correctness changes — uniform fail-loud errors, one RPC envelope, a single sample read path, caching, N+1 removal, dead-code removal — plus typed IDs, first-class units, and boundary validation.
status: in-progress
phase: "1"
resource: overhaul/PRD-25-consistency-hardening.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-25 bundles cross-cutting correctness and consistency changes that each remove a latent bug class. The hardening items apply a uniform fail-loud error policy (warn only for a missing database, surface genuine FK/vocab/integrity/transport errors), define one typed RPC envelope with a single client-side unwrap, provide a single canonical sample read path, cache the parsed dictionary and generated DDL, remove N+1 queries from list endpoints, and drop confirmed-dead legacy paths. The correctness primitives add typed IDs (distinct value-object types for artifact, sample, user, operation, and setup ids), a first-class units/quantities layer validated against the dictionary, and validation moved to the RPC/registration boundary rather than deep inside a transaction.

# Status
In-progress (phase 1, STATUS TABLE authoritative). Items are independent and land in any order, interleaved with other PRDs.

# Relationships
- Cross-cutting; the error-policy and RPC-envelope items reinforce [PRD-18](prd-18.md).
- The sample-table item finishes the reconciliation completed by [PRD-19](prd-19.md) (which subsumes it) and depends on the flrCIF-canonical fix.
- Boundary validation is dictionary-driven via [PRD-26](prd-26.md), reusing the operation parameter schemas of [PRD-11](prd-11.md).
- Targets the [MFDB target](/specs/mfdb.md) and the [RPC target](/specs/rpc.md).

# Source
- Primary: `overhaul/PRD-25-consistency-hardening.md`
