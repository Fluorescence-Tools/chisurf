---
type: PRD
prd: "37"
title: "PRD-37: Network Deployment Security — transport auth + RPC authn/authz"
description: Adds encrypted, endpoint-authenticated transport plus per-call authentication, authorization, and scoped event broadcast so the server may bind beyond loopback.
status: draft
phase: "unassigned"
resource: overhaul/PRD-37-network-deployment-security.md
tags: [prd, rpc]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-37 is the security gate that must be satisfied before the ChiSurf server can bind to anything other than loopback. Today the transport is plaintext ZMQ REQ/REP + PUB/SUB with no encryption, no endpoint authentication, no per-call authorization, and an unscoped event broadcast; safety rests entirely on a fail-closed loopback-only guard. The PRD requires ZeroMQ CURVE/ZAP transport security, a login handshake issuing signed expiring session tokens verified at the dispatcher, per-method capability/role authorization with default-deny, and per-subscription scoping of the event broadcast with payload hygiene. It reuses existing building blocks (the session-token credential store, principal resolution, and the per-service auth parameter) rather than reinventing them.

# Status
Draft (unassigned phase, STATUS TABLE authoritative). Requirements and task breakdown specified; the loopback guard remains the safe default until all three layers land.

# Relationships
- Gates any future networked deployment; the loopback guard cannot be relaxed until transport security, authn, and authz are all active.
- Builds on canonical identity / `resolve_active_user_id` and the `flr_sample_users` table, and reuses the existing session-token credential store.
- Consumes the event model for audit and constrains how those events may cross the process boundary.
- Hardens the write path referenced by [PRD-41](prd-41.md)'s deposition strategy.
- Concerns the [server](/architecture/server.md), [RPC target](/specs/rpc.md), and [MFDB (current)](/architecture/mfdb.md).

# Source
- Primary: `overhaul/PRD-37-network-deployment-security.md`
