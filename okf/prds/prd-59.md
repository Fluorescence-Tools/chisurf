---
type: PRD
prd: "59"
title: "PRD-59: Pluggable MFDB Authentication (local / LDAP)"
description: A pluggable authentication layer for MFDB with local-password and LDAP/Active-Directory providers behind one interface, resolving to the existing Principal/session, with JIT provisioning and directory-group mapping.
status: in-progress
phase: "landed: local + LDAP providers, CLI, hardening; no further providers planned"
resource: modules/mfdb/src/mfdb/security/
tags: [prd, mfdb, auth, security, ldap]
timestamp: '2026-07-08T00:00:00Z'
---

# Summary
MFDB is the prototype for an institute-hosted public fluorescence databank; its users are
researchers at institutions, so authentication should ride on the institutional **LDAP / Active
Directory** while a **local** password provider stays available for the bootstrap admin and
offline/standalone use. Previously MFDB auth was 100% local and hardcoded. This PRD introduces a
small **pluggable `AuthProvider`** layer in `security/`: providers turn a credential into a neutral
`AuthIdentity`, and a `login()` orchestrator maps it onto the existing MFDB `Principal`/session —
matching or JIT-provisioning the `flr_sample_users` row and mapping directory groups onto
`mfdb_group_member`. eLabFTW/ELN integration was explicitly dropped from scope (PRD-48 untouched).

# Goal / Motivation
Institutional deployments need directory-backed sign-in (one identity per person, central
credential/group management) without giving up MFDB's self-contained local login. The design must
keep MFDB standalone (optional deps stay lazy), be testable offline (no live directory in CI), and
preserve the existing token→`Principal`→ACL machinery unchanged.

# Design
A single provider interface, a configured default + always-available local fallback, and one
orchestrator that owns identity resolution and session minting.

## Provider interface (`security/auth_providers.py`)
- `AuthIdentity(provider, external_id, email, display_name, is_admin, groups, managed_groups, raw)`
  — the neutral result of a successful authentication. `managed_groups` is the universe of MFDB
  groups the provider authoritatively controls (`groups ⊆ managed_groups`), enabling directory
  reconciliation.
- `AuthProvider` `Protocol`: `authenticate(*, user_id, password) -> AuthIdentity | None` (returns
  `None` for bad credentials; raises only on misconfiguration).
- **Two providers, fixed dispatch.** `login.resolve_provider` returns `LocalAuthProvider` (default,
  always available) or `LdapAuthProvider` by name/config; any other name raises. No other providers
  are planned, so there is no open registration mechanism — the `AuthProvider` protocol is the seam
  if one is ever needed.
- `LocalAuthProvider` — verifies `flr_sample_users.password_hash` (PBKDF2-HMAC-SHA256, 100k) with the
  exact pre-existing admin / passwordless / no-hash rules, but **only for users homed on the local
  provider** (`auth_provider = 'local'`). This closes a critical hole: external (LDAP) users are
  provisioned without a local hash, and the no-hash branch would otherwise accept an *empty* password
  — logging anyone in as them without the directory. The legacy in-process `password_services.
  login_handler` carries the same guard.
- `LdapAuthProvider` — **search+bind**: service-account bind → search `user_filter` under `base_dn`
  → re-bind as the located user DN to verify the password → map `memberOf` to MFDB groups
  (`group_map`) and admin status (`admin_groups`). The `ldap3` dependency is optional and lazy
  (`_require_ldap3`; `[ldap]` extra); an injectable `connection_factory` allows fully-offline
  `ldap3` `MOCK_SYNC` testing. Hardened: LDAP-filter values escaped against injection, required-key
  validation (`base_dn`), connections always unbound (`try/finally`), optional TLS CA
  (`ca_cert` → `ldap3.Tls`), and directory-unreachable/search errors surfaced as a clear `AuthError`
  (never a crash) while a wrong user password is a plain `None`.

## Orchestrator (`security/login.py`)
`login(conn, *, provider, user_id, password, client_metadata, config, jit)`:
throttle check → `resolve_provider` (default from config; `local` always constructible) →
`authenticate` → `resolve_or_provision_user` → `sync_groups` → `create_session`; commits only on
success (prior behaviour preserved). `resolve_or_provision_user` matches by
`(auth_provider, external_id)` → `email` (linking the row) → JIT-creates a `flr_sample_users` row for
external providers (configurable); `local` returns its row unchanged. Email-matching only adopts a
**placeholder** local account (local-homed, no password) — a real local-credential or other-provider
account is never silently converted, so a directory email collision cannot hijack (e.g.) a local
admin. The RPC handler `mfdb.security.auth.login` (`admin/backend/auth_services.login_handler`)
delegates here.

**Directory-authoritative reconciliation** (`sync_identity`): on each external-provider login the
user's `is_admin` / `email` / `display_name` are refreshed from the identity, and the provider's
`managed_groups` are reconciled — mapped groups the identity carries are added, managed groups it no
longer carries are removed (admin membership follows `is_admin`). Locally-managed groups outside
`managed_groups` are never touched; `local` logins reconcile nothing. **Throttling correctness:**
failed attempts are committed (`_record_failure`) so `is_throttled` accumulates across the
per-request connections of the RPC path (previously they rolled back, silently disabling
brute-force throttling).

## Identity linkage & schema
`flr_sample_users` gains `auth_provider TEXT DEFAULT 'local'` and `external_id TEXT`, indexed by
`(auth_provider, external_id)` for reverse lookup. JIT users are created without a password hash and
attached to the built-in `users` group (admins also to `admins`), mirroring `bootstrap_auth_groups`.
The fixed-salt bootstrap-admin password hash was replaced with a per-user random salt.

## Config & secrets (`config.py`, `security/credentials.py`)
`configured_auth_config()` resolves provider selection + the LDAP block from a host-injected resolver
(`set_auth_config_resolver`, mirroring the PRD-24 default-user bridge) → `MFDB_AUTH_PROVIDER` /
`MFDB_LDAP_*` env → `None` (local). The LDAP service-account bind password is sourced from the OS
credential store (`store/load_ldap_bind_password`) or env — **never** settings JSON.

## Headless CLI (`admin/cli`)
`mfdb-admin auth login --user … [--password …] [--provider local|ldap]`, `auth whoami --token …`,
`auth status`. A standalone `mfdb-admin` console-script entry keeps the path headless-first (repo
rule), independent of ChiSurf's `csc` plugin mounting.

# Decisions
- Provider selection: one configured default + **Local always available**; per-login override.
  Enforced by a **local fallback** in `login()`: when a non-local default provider returns ``None``,
  the local provider is tried, so a locally-homed account (bootstrap admin) is never locked out by a
  directory being the default. Safe because the local provider rejects external-homed users — the
  fallback can only admit genuine local accounts.
- LDAP first login: **JIT auto-provision** (configurable), directory authoritative.
- Hashing: keep PBKDF2-SHA256/100k (no new hash dependency); fix the fixed-salt bootstrap admin.
- Library: `ldap3` (pure-Python, no C deps), optional + lazy.

# Verification
Offline tests only (no live directory/network): `tests/test_auth_providers.py` (Local rules,
`login()` session round-trip + throttle, JIT/link/email-fallback/match-only, group sync, stub-LDAP
end-to-end), `tests/test_ldap_auth.py` (`ldap3` `MOCK_SYNC`: attr+group mapping, wrong/empty
password, unknown user, service-bind-failure, filter-injection escaping, missing-dep error, `login()`
JIT end-to-end, env/resolver config), `tests/test_auth_cli.py` (headless `mfdb-admin auth`). The
existing `test_mfdb_auth.py` / `test_mfdb_user_management.py` stay green (Local behaviour preserved).

# Non-goals
- **No further auth providers** beyond local + LDAP (no OIDC/SAML/external-IdP). eLabFTW/ELN
  integration is out of scope (PRD-48 remains untouched).
- Argon2/bcrypt hash upgrade; per-provider password-reset flows.

# Relationships
Extends the [MFDB architecture](/architecture/mfdb.md) security layer; aligns with network-security
(PRD-37) and the standalone/default-user bridge (PRD-24); adjacent to the auth-threading gap
(INC-04) in [assessment](/specs/assessment.md).
