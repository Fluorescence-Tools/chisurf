---
type: PRD
prd: "37"
title: "PRD-37: Network Deployment Security — transport auth + RPC authn/authz"
description: Adds encrypted, endpoint-authenticated transport plus per-call authentication, authorization, and scoped event broadcast so the server may bind beyond loopback.
status: draft
phase: "unassigned"
resource: chisurf/server/
tags: [prd, rpc]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-37 is the security gate that must be satisfied before the ChiSurf server can bind to anything other than loopback. Today the transport is plaintext ZeroMQ REQ/REP + PUB/SUB with no encryption, no endpoint authentication, no per-call authorization, and an unscoped event broadcast; safety rests entirely on a fail-closed loopback-only guard. The PRD requires ZeroMQ CURVE/ZAP transport security, a login handshake issuing signed expiring session tokens verified at the dispatcher, per-method capability/role authorization with default-deny, and per-subscription scoping of the event broadcast with payload hygiene. It reuses existing building blocks (the session-token credential store, principal resolution, and the per-service auth parameter) rather than reinventing them.

# Status
Draft (unassigned phase, STATUS TABLE authoritative). Requirements and task breakdown specified; the loopback guard remains the safe default until all three layers land.

# Goal
Before the ChiSurf server may bind to anything other than loopback, the RPC transport must be **encrypted and endpoint-authenticated**, every command must be **authenticated and authorized** per call, and the event broadcast must be **scoped** so it cannot leak data to unauthorized subscribers. Today the server is safe *only* because it is loopback-only; this PRD is the gate that must be satisfied to lift that restriction.

# Current posture (as built)

- **Transport:** ZeroMQ REQ/REP (commands, default `:8765`) + PUB/SUB (event broadcast, default `:8766`), bound to `127.0.0.1`.
- **Fail-closed guard (good, keep):** `chisurf/server/transport/zmq.py` refuses any non-loopback `host` — *"Non-loopback host requires CURVE/ZAP transport security which is not yet implemented."* So the server cannot accidentally be exposed.
- **On the wire:** plaintext, **no encryption, no endpoint authentication**.
- **Per-call:** the dispatcher does **not** authenticate or authorize callers. `resolve_active_user_id(auth, conn)` (`chisurf/core/mfdb/session.py`) *resolves* a principal from an `auth` payload but nothing *verifies* that payload or checks whether the principal may call the requested method. `dispatcher.py`'s `auth`/`password`/`token` handling is **log redaction only**.
- **Events:** `chisurf/server/app.py` bridges `"*"` (every server event) to the PUB socket — any subscriber receives every event, unscoped.
- **In-process MFDB event bus** (`chisurf/core/mfdb/events.py`, PRD-21): in-process only, **not** bridged to the network. It must stay that way (or be reviewed under this PRD) before any network exposure — see "Event scoping" below.

Trust boundary today = "any local process/user on this machine."

# Existing building blocks to reuse (don't reinvent)

- **Session tokens:** `chisurf/core/mfdb/credentials.py` already stores/loads/rotates per-`(host, port, user)` session tokens via the OS credential store.
- **Principal resolution:** `resolve_active_user_id(auth, conn)` + the `flr_sample_users` table (`is_admin`, `password_hash`, `allow_passwordless_login`) from PRD-17.
- **Per-service `auth` param:** services already accept an `auth` payload threaded by the dispatcher — the hook where verification/authorization belongs.

# Requirements

## Transport security (authn + confidentiality + integrity)

- Implement **ZeroMQ CURVE** (Curve25519) with a **ZAP** handler for endpoint authentication on both the REQ/REP and PUB/SUB sockets; server keypair + authorized client public keys (allow-list), or a CA-style trust root.
- Key management: generated/rotatable server keys, client key enrollment, secure storage (reuse the credential store). No keys in the repo or logs.
- Only after this lands may the loopback guard be relaxed — and even then, non-loopback bind should require an explicit, audited opt-in.

## Authentication (who is calling)

- A **login handshake** that verifies credentials (password/passwordless per `flr_sample_users`) and issues a signed, expiring **session token**; clients send it on every command.
- The dispatcher **verifies** the token before dispatch (signature + expiry + binding to the connection/endpoint), rejecting unauthenticated calls — fail-closed. `resolve_active_user_id` consumes the *verified* principal, never a client-asserted one.
- Token lifecycle: expiry, refresh, revocation, rotation (reuse `credentials.py`).

## Authorization (what they may do)

- Per-method **capability/role checks** at the dispatcher (e.g. read vs. write vs. admin), tied to the resolved principal and `flr_sample_users.is_admin` / roles.
- Default-deny for unknown/unscoped methods; deny mutating MFDB calls to read-only principals.

## Event scoping (broadcast hygiene)

- Replace the unconditional `"*"` → PUB bridge with **per-subscription authorization** and **topic/visibility filtering**, so a subscriber only receives events it is entitled to (e.g. owner/visibility from PRD-17).
- **Payload hygiene:** no secrets/tokens/PII in event payloads. Treat any network-broadcast payload as readable by every authorized subscriber.
- The in-process MFDB event bus (PRD-21) **must not** be bridged to the network PUB socket without passing it through this scoping + payload review first.

## Operational

- Rate limiting / connection caps on the command socket; audit-log authn failures and authz denials (feed the PRD-21 event model / audit subscriber).
- Keep the **fail-closed** default: loopback-only unless transport security + authn + authz are all active.

# Tasks

1. CURVE/ZAP transport security on both sockets + key management; gate non-loopback bind behind it.
2. Login handshake → signed expiring session token; dispatcher-level token verification (fail-closed); wire verified principal into `resolve_active_user_id`.
3. Per-method authorization (capabilities/roles); default-deny.
4. Scope the event broadcast (per-subscription authz + payload hygiene); review any MFDB-event bridge.
5. Rate limiting + authn/authz audit logging.
6. Tests: rejected unauthenticated call; rejected unauthorized (under-privileged) call; tampered/expired token rejected; encrypted-transport round trip; a subscriber receives only authorized events; non-loopback bind refused unless security active.

# Definition of Done

- [ ] Non-loopback bind is possible **only** with CURVE/ZAP + authn + authz active; otherwise fail-closed (the existing guard).
- [ ] Every command is authenticated (verified session token) and authorized (per-method capability) before dispatch; unauthenticated/unauthorized calls are rejected and audited.
- [ ] Event broadcast is scoped per subscriber; no secrets in payloads; the in-process MFDB bus is not bridged to the network without review.

# Definition of Clean

The transport is encrypted and endpoint-authenticated; the dispatcher is the single authn/authz chokepoint (no handler trusts a client-asserted principal); broadcasts are least-privilege; loopback-only remains the safe default; behavior-asserting tests for reject-unauthenticated, reject-unauthorized, tamper/expiry, and scoped broadcast.

# Relationships
- Gates any future networked deployment; the loopback guard cannot be relaxed until transport security, authn, and authz are all active.
- Builds on canonical identity / `resolve_active_user_id` and the `flr_sample_users` table (PRD-17), and reuses the existing session-token credential store.
- Consumes the PRD-21 event model for audit and constrains how those events may cross the process boundary.
- Independent of the data-model PRDs; required before the server's loopback guard is ever relaxed.
- Hardens the write path referenced by [PRD-41](prd-41.md)'s deposition strategy.
- Concerns the [server](/architecture/server.md), [RPC target](/specs/rpc.md), and [MFDB (current)](/architecture/mfdb.md).
