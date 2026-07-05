---
type: PRD
prd: "17"
title: "PRD-17: Canonical Identity / Session Context"
description: Resolves the active user and target database once at the boundary into a single SessionContext threaded explicitly through registration, browse, and ownership code.
status: in-progress
phase: "1"
resource: overhaul/PRD-17-identity-session-context.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-17 eliminates divergent identity resolution — where registration stamped one user while read handlers resolved another — that caused ownership and visibility bugs (for example, "Mine" returning zero results). It introduces a single `SessionContext` dataclass carrying `user_id`, database handle, admin flag, groups, and auth principal, constructed once per entry point (GUI launch or RPC dispatch). All MFDB read and write APIs take the context explicitly, and the scattered per-module current-user resolvers are removed in favor of one canonical resolver used by both reads and writes.

# Status
In-progress (re-verified against code 2026-07-05). The canonical current-user resolver landed and the "Mine returns 0" ownership/visibility bug is fixed (registration owner and browse scope agree). **Open:** the "one `SessionContext`, constructed once per entry point and threaded explicitly" design is not realized — `resolve_session()` has no call sites and ~10 modules still resolve identity independently, so the DoD ("no module re-resolves identity on its own") is unmet. Corroborated by assessment [SV-03](/specs/assessment.md#sv-03).

# Relationships
- Subsumes the per-handler anonymous-fallback patches added in [PRD-10](prd-10.md).
- Pairs with [PRD-18](prd-18.md) — the SessionContext is the unit that gets dependency-injected.
- The injected identity/config becomes a prerequisite for [PRD-24](prd-24.md) package extraction.
- Targets the [MFDB target](/specs/mfdb.md); reduces reliance on [runtime globals](/architecture/runtime-globals.md).

# Source
- Primary: `overhaul/PRD-17-identity-session-context.md`
