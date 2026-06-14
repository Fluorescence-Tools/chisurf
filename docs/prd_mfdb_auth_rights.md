# PRD: MFDB Authentication, Public Mode, Users, Groups, And Object Rights

Status date: 2026-06-14

Status: implementation-ready PRD

Owner: MFDB implementation agents

Review requirement: this PRD is intentionally explicit. A later reviewer should be
able to verify the implementation by checking every acceptance checkbox in this
file against code, tests, and runtime behavior.

## Summary

MFDB needs a hard-cut authorization architecture before production use. The
current development behavior trusts the active user setting and caller-supplied
`requester_id` fields. That is not acceptable for non-trusted networks or for
object privacy. This project replaces trusted caller identity with authenticated
sessions and Linux-like object rights.

MFDB access to non-public objects must require authentication. Public mode
remains, but it means anonymous read-only access to explicitly public objects.
Users can manage objects they own. Admins can manage everything. Objects use
owner, group, other, mode bits, and optional ACL entries. Breaking changes are
allowed and preferred over compatibility shims; update all dependents to the
new contract.

## Design Decisions

- [ ] MFDB is in development; breaking RPC, schema, GUI, and test changes are
  allowed.
- [ ] Do not preserve `requester_id` as an authority source.
- [ ] Do not keep unauthenticated writes.
- [ ] Do not add third-party dependencies.
- [ ] Use stdlib cryptography primitives plus existing PyZMQ capabilities.
- [ ] Enforce permissions in repository/service APIs and test those boundaries.
- [ ] Do not claim SQLite file-level access is a complete security boundary.
- [ ] Public mode means anonymous read-only access to explicitly public objects.
- [ ] New scientific/user-created objects are private by default.
- [ ] Curated catalog/reference data may be public-read/admin-write.

## Goals

- [ ] Require authentication for every non-public MFDB object read.
- [ ] Require authentication for every MFDB write, delete, share, ownership
  change, group change, and permission change.
- [ ] Keep anonymous public-read workflows for explicitly public data.
- [ ] Let object owners manage access to their own objects.
- [ ] Let admins manage all objects, users, groups, permissions, and sessions.
- [ ] Model rights using established POSIX concepts: owner, group, other, mode
  bits, and optional ACL entries.
- [ ] Make list/search/graph/export operations authorization-aware.
- [ ] Ensure private object metadata is not leaked by errors, list endpoints,
  graph traversals, or GUI tables.
- [ ] Update MFDB GUI/client, startup login, user editor, setup services,
  measurement services, Chinet integration, project archive/restore, and tests.

## Non-Goals

- [ ] Do not build a full web identity provider.
- [ ] Do not support OAuth, SAML, LDAP, or external identity integrations in
  this iteration.
- [ ] Do not allow anonymous mutation, even in development mode.
- [ ] Do not use passwords or session tokens as query parameters in file paths,
  URLs, object IDs, or logs.
- [ ] Do not preserve legacy `sample_database.*` authorization behavior if it
  conflicts with the new MFDB contract.
- [ ] Do not add a second object database or replace SQLite.
- [ ] Do not implement row-level SQLite security; application services remain
  the supported security boundary.

## Terms

- **Principal**: authenticated identity used for authorization. A principal is
  either anonymous, a user, or an admin user.
- **Session token**: opaque bearer token returned by login and supplied on
  subsequent protected RPC calls.
- **Object**: any MFDB-protected resource addressed by `(object_type,
  object_id)`.
- **Mode**: POSIX-style octal integer describing owner/group/other rights.
- **ACL**: explicit per-user or per-group allow/deny entries attached to an
  object.
- **Public object**: object with the other-read bit set, readable by anonymous
  callers.
- **Manage permission**: the `x` right. It permits chmod/chown/chgrp/share,
  ACL edits, and delete.

## Protected Object Inventory

Implement authorization for these object types first:

- [ ] `sample`: `flr_sample`
- [ ] `sample_condition`: `flr_sample_condition`
- [ ] `experiment`: `flr_experiment`
- [ ] `experiment_data`: `flr_experiment_data`
- [ ] `raw_data`: raw data references
- [ ] `processed_data`: processed data references/products
- [ ] `artifact`: `mfdb_artifact`
- [ ] `operation`: `mfdb_operation`
- [ ] `operation_artifact`: `mfdb_operation_artifact`
- [ ] `parameter`: `mfdb_parameter`
- [ ] `setup`: `mfdb_setup`
- [ ] `branch`: `mfdb_branch`
- [ ] `edge`: `mfdb_edge`
- [ ] `project_snapshot`: project snapshot artifacts
- [ ] `chinet_session`: Chinet session artifacts
- [ ] `user`: `flr_sample_users`
- [ ] `group`: `mfdb_group`

Reference/catalog object policy:

- [ ] `probe`, `vocabulary`, and `experiment_type` are public-read by default.
- [ ] Catalog writes require admin or explicit manage permission.
- [ ] Catalog reads must not expose private user/session/password metadata.

## Permission Model

Rights are POSIX-like:

- [ ] `r` means read visibility: list, get, search, export, graph visibility,
  restore payload visibility.
- [ ] `w` means mutation: create, update, record, link, status change, append
  metadata, write child objects.
- [ ] `x` means manage: chmod, chown, chgrp, share, edit ACL, delete, revoke,
  transfer.

Mode bits:

- [ ] Store mode as an integer representing POSIX-style octal bits.
- [ ] Default private mode for new scientific/user objects is `0o700`.
- [ ] Public-read mode is any mode with other-read set, such as `0o704` for
  owner-all/public-read.
- [ ] Group-readable mode is any mode with group-read set, such as `0o740`.
- [ ] Group-writable mode is any mode with group-write set, such as `0o760`.
- [ ] Other-write and other-manage modes are invalid for MFDB unless an admin
  explicitly enables them through a documented emergency/testing path.

Evaluation order:

- [ ] If principal is admin, allow.
- [ ] If principal is anonymous, allow only when required permission is `r` and
  object has other-read.
- [ ] If an explicit deny ACL matches the user, deny.
- [ ] If an explicit allow ACL matches the user, allow only the allowed bits.
- [ ] If an explicit deny ACL matches any group membership, deny.
- [ ] If one or more explicit allow group ACL entries match, allow the union of
  their allowed bits.
- [ ] If user owns the object, evaluate owner mode bits.
- [ ] If user is in the object's owning group, evaluate group mode bits.
- [ ] Otherwise evaluate other mode bits.
- [ ] If no rule grants the requested permission, deny.

Inheritance:

- [ ] A child object inherits ACL metadata from its nearest parent unless the
  creator supplies an explicit ACL policy.
- [ ] Sample conditions inherit from their sample when created through a sample
  workflow.
- [ ] Experiments inherit from their sample unless explicitly set.
- [ ] Experiment data, raw data, processed data, operations, operation-artifact
  links, parameters, and project/Chinet artifacts inherit from their experiment
  or operation parent.
- [ ] Edge visibility requires read access to both endpoint objects.
- [ ] Graph traversal must omit unreadable nodes and omit edges with unreadable
  endpoints.

## Schema Requirements

Add schema version after the current MFDB schema version.

### `mfdb_group`

- [ ] `group_id TEXT PRIMARY KEY`
- [ ] `display_name TEXT NOT NULL`
- [ ] `description TEXT`
- [ ] `is_builtin INTEGER DEFAULT 0`
- [ ] `created_by_user_id TEXT REFERENCES flr_sample_users(user_id)`
- [ ] `created_at TEXT DEFAULT CURRENT_TIMESTAMP`
- [ ] `updated_at TEXT DEFAULT CURRENT_TIMESTAMP`
- [ ] `deleted_at TEXT`

Built-in rows:

- [ ] `admins`: all admin users should be members.
- [ ] `users`: normal authenticated users should be members.
- [ ] `public`: conceptual anonymous/public group; do not use it as an auth
  bypass.

### `mfdb_group_member`

- [ ] `group_id TEXT NOT NULL REFERENCES mfdb_group(group_id)`
- [ ] `user_id TEXT NOT NULL REFERENCES flr_sample_users(user_id)`
- [ ] `role TEXT DEFAULT 'member'`
- [ ] `created_by_user_id TEXT REFERENCES flr_sample_users(user_id)`
- [ ] `created_at TEXT DEFAULT CURRENT_TIMESTAMP`
- [ ] `updated_at TEXT DEFAULT CURRENT_TIMESTAMP`
- [ ] `deleted_at TEXT`
- [ ] Primary key or unique index prevents duplicate active memberships.
- [ ] Group owners/managers are represented through `role='owner'` or
  `role='manager'`.

### `mfdb_object_acl`

- [ ] `object_type TEXT NOT NULL`
- [ ] `object_id TEXT NOT NULL`
- [ ] `owner_user_id TEXT NOT NULL REFERENCES flr_sample_users(user_id)`
- [ ] `owner_group_id TEXT REFERENCES mfdb_group(group_id)`
- [ ] `mode INTEGER NOT NULL DEFAULT 448` where `448 == 0o700`
- [ ] `inherits_from_type TEXT`
- [ ] `inherits_from_id TEXT`
- [ ] `created_at TEXT DEFAULT CURRENT_TIMESTAMP`
- [ ] `updated_at TEXT DEFAULT CURRENT_TIMESTAMP`
- [ ] `deleted_at TEXT`
- [ ] Unique active ACL row per `(object_type, object_id)`.
- [ ] Add index on owner user.
- [ ] Add index on owner group.
- [ ] Add index on object lookup.

### `mfdb_acl_entry`

- [ ] `entry_id INTEGER PRIMARY KEY AUTOINCREMENT`
- [ ] `object_type TEXT NOT NULL`
- [ ] `object_id TEXT NOT NULL`
- [ ] `subject_type TEXT NOT NULL CHECK(subject_type IN ('user', 'group'))`
- [ ] `subject_id TEXT NOT NULL`
- [ ] `effect TEXT NOT NULL CHECK(effect IN ('allow', 'deny'))`
- [ ] `permissions INTEGER NOT NULL`
- [ ] `created_by_user_id TEXT REFERENCES flr_sample_users(user_id)`
- [ ] `created_at TEXT DEFAULT CURRENT_TIMESTAMP`
- [ ] `updated_at TEXT DEFAULT CURRENT_TIMESTAMP`
- [ ] `deleted_at TEXT`
- [ ] Add index on object lookup.
- [ ] Add index on subject lookup.

### `mfdb_session`

- [ ] `session_id TEXT PRIMARY KEY`
- [ ] `user_id TEXT NOT NULL REFERENCES flr_sample_users(user_id)`
- [ ] `token_hash TEXT NOT NULL UNIQUE`
- [ ] `created_at TEXT DEFAULT CURRENT_TIMESTAMP`
- [ ] `expires_at TEXT NOT NULL`
- [ ] `last_used_at TEXT`
- [ ] `revoked_at TEXT`
- [ ] `client_host TEXT`
- [ ] `client_name TEXT`
- [ ] `client_metadata_json TEXT`
- [ ] Add index on `token_hash`.
- [ ] Add index on `expires_at`.
- [ ] Add index on `user_id`.

### `mfdb_auth_attempt`

- [ ] `attempt_id INTEGER PRIMARY KEY AUTOINCREMENT`
- [ ] `user_id TEXT`
- [ ] `client_host TEXT`
- [ ] `success INTEGER NOT NULL DEFAULT 0`
- [ ] `reason TEXT`
- [ ] `attempted_at TEXT DEFAULT CURRENT_TIMESTAMP`
- [ ] Add index on `(user_id, attempted_at)`.
- [ ] Add index on `(client_host, attempted_at)`.

### Bootstrap And Migration

- [ ] Fresh DB creates auth schema before service registration tests run.
- [ ] Existing dev DBs are migrated with deterministic defaults.
- [ ] Existing users are added to `users`.
- [ ] Existing admins are added to `admins`.
- [ ] Existing objects receive public-read ACLs only if the migration policy
  says so for dev data.
- [ ] New objects created after migration use private owner-only mode by
  default.
- [ ] `user_default` may exist for development/bootstrap, but it is not an
  authentication bypass and must not grant private access anonymously.

## Authentication Requirements

RPC methods:

- [ ] `mfdb.auth.login(user_id, password, client_metadata=None)`
- [ ] `mfdb.auth.logout(auth)`
- [ ] `mfdb.auth.me(auth)`
- [ ] `mfdb.auth.sessions.list(auth, user_id=None)`
- [ ] `mfdb.auth.sessions.revoke(auth, session_id)`

Login behavior:

- [ ] Login checks password hash for the target user.
- [ ] Empty password login is not allowed for admin users.
- [ ] If passwordless development users remain allowed, they must be
  configurable and local-loopback only.
- [ ] Successful login returns `{token, expires_at, user}`.
- [ ] Token is shown only once in the login response.
- [ ] Store only `sha256(token)` or stronger stdlib-derived hash.
- [ ] Session token generation uses `secrets.token_urlsafe`.
- [ ] Session expiry defaults to a finite duration, such as 12 hours.
- [ ] `last_used_at` updates on authenticated calls.
- [ ] Logout sets `revoked_at`.
- [ ] Expired/revoked/missing/unknown tokens are rejected.
- [ ] Failed login attempts are written to `mfdb_auth_attempt`.
- [ ] Add simple throttling after repeated failures by `(user_id, client_host)`.

Password behavior:

- [ ] Keep PBKDF2-SHA256 unless the implementation explicitly upgrades with
  stdlib-only code.
- [ ] Password hashes are never returned by any RPC endpoint.
- [ ] `has_password` may be returned, but only where user listing is permitted.
- [ ] Password/token/auth fields are redacted in logs and monitor callbacks.

## Authorization Module

Create `chisurf.core.mfdb.auth`.

Required public symbols:

- [ ] `Principal`
- [ ] `AnonymousPrincipal`
- [ ] `AuthenticatedPrincipal`
- [ ] `AuthError`
- [ ] `PermissionDenied`
- [ ] `authenticate_token(conn, token)`
- [ ] `principal_from_rpc_auth(conn, auth)`
- [ ] `require_authenticated(principal)`
- [ ] `can_access(conn, principal, object_type, object_id, permission)`
- [ ] `require_access(conn, principal, object_type, object_id, permission)`
- [ ] `grant_acl(conn, principal, object_type, object_id, subject_type,
  subject_id, permissions, effect='allow')`
- [ ] `revoke_acl(conn, principal, entry_id)`
- [ ] `chmod(conn, principal, object_type, object_id, mode)`
- [ ] `chown(conn, principal, object_type, object_id, owner_user_id)`
- [ ] `chgrp(conn, principal, object_type, object_id, owner_group_id)`
- [ ] `create_default_acl_for_object(conn, object_type, object_id, owner_user_id,
  owner_group_id=None, mode=0o700)`
- [ ] `inherit_acl_from_parent(conn, object_type, object_id, parent_type,
  parent_id, owner_user_id=None)`
- [ ] `filter_readable(conn, principal, object_type, rows, id_key)`

Behavior:

- [ ] `principal_from_rpc_auth` returns anonymous principal when auth is absent.
- [ ] `require_authenticated` rejects anonymous principals.
- [ ] `require_access` raises `PermissionDenied` with a generic message.
- [ ] Errors must not reveal whether a private object exists to unauthorized
  users.
- [ ] Admin bypass is centralized here, not duplicated across services.
- [ ] Permission evaluation is deterministic and covered by unit tests.

## RPC/API Contract

Every protected RPC method accepts:

```json
{
  "auth": {
    "token": "opaque-session-token"
  }
}
```

Rules:

- [ ] Protected reads require either read access or public-read status.
- [ ] Protected writes require authenticated write access.
- [ ] Protected manage actions require authenticated manage access.
- [ ] List/search endpoints filter unreadable objects from results.
- [ ] Get endpoints return a generic not-found/permission error for unreadable
  private objects.
- [ ] Create endpoints set ACL on the new object in the same transaction as the
  object write.
- [ ] Update endpoints check `w` before mutation.
- [ ] Delete endpoints check `x` before deletion.
- [ ] Link/edge endpoints check `w` on the link object and read/write as
  appropriate on endpoints.
- [ ] Remove or rewrite every `requester_id` check.
- [ ] Do not accept caller-supplied `operator_user_id` unless it matches the
  authenticated principal or the caller is admin.

## Repository And Service Architecture

Repository layer:

- [ ] Public repository entrypoints for protected operations accept
  `principal` or explicit auth context.
- [ ] Internal low-level helpers may skip auth only when they are not exposed as
  supported entrypoints and are covered by service-level tests.
- [ ] Object and ACL creation happen in the same transaction.
- [ ] Inherited ACL creation happens in the same transaction as child object
  creation.
- [ ] Audit log rows include the authenticated user id where available.

Service layer:

- [ ] Service handlers convert RPC `auth` payloads into principals.
- [ ] Service handlers pass principals to repository methods.
- [ ] Service handlers return structured authorization errors.
- [ ] Service handlers never trust `requester_id`.
- [ ] Service handlers never return password hashes or session token hashes.

Client layer:

- [ ] `MFDBClient.login(...)` stores the token.
- [ ] `MFDBClient.logout()` revokes and clears the token.
- [ ] `MFDBClient.me()` returns current authenticated user.
- [ ] `MFDBClient` injects `auth` automatically into protected calls.
- [ ] `MFDBClient` can make explicit anonymous public-read calls before login.
- [ ] Client code must not pass raw tokens into logs or UI tables.

## Group And User Management

Required RPC methods:

- [ ] `mfdb.groups.list(auth)`
- [ ] `mfdb.groups.get(auth, group_id)`
- [ ] `mfdb.groups.create(auth, group)`
- [ ] `mfdb.groups.update(auth, group)`
- [ ] `mfdb.groups.delete(auth, group_id)`
- [ ] `mfdb.groups.members.list(auth, group_id)`
- [ ] `mfdb.groups.members.add(auth, group_id, user_id, role='member')`
- [ ] `mfdb.groups.members.remove(auth, group_id, user_id)`
- [ ] `mfdb.permissions.get(auth, object_type, object_id)`
- [ ] `mfdb.permissions.chmod(auth, object_type, object_id, mode)`
- [ ] `mfdb.permissions.chown(auth, object_type, object_id, owner_user_id)`
- [ ] `mfdb.permissions.chgrp(auth, object_type, object_id, owner_group_id)`
- [ ] `mfdb.permissions.grant(auth, object_type, object_id, subject_type,
  subject_id, permissions, effect='allow')`
- [ ] `mfdb.permissions.revoke(auth, entry_id)`

Rules:

- [ ] Admin can manage all users and groups.
- [ ] Group owner/manager can manage memberships for that group.
- [ ] Normal users can read their own profile.
- [ ] Normal users cannot promote themselves to admin.
- [ ] Normal users cannot edit another user's private profile.
- [ ] User listing for anonymous login UI must be restricted to safe fields or
  replaced with manual user-id entry.
- [ ] Deleting a user with committed data should remain blocked unless admin
  explicitly transfers or archives ownership first.

## Public Mode

Anonymous behavior:

- [ ] Anonymous users can list/get only objects with other-read.
- [ ] Anonymous users cannot create objects.
- [ ] Anonymous users cannot update objects.
- [ ] Anonymous users cannot delete objects.
- [ ] Anonymous users cannot create links or edges.
- [ ] Anonymous users cannot modify permissions.
- [ ] Anonymous users cannot see private object IDs in list/search/graph/export.

Public sharing:

- [ ] "Make public" sets other-read on the object's mode or adds an equivalent
  public allow policy.
- [ ] "Make private" clears other-read and removes public allow entries.
- [ ] Public status does not duplicate objects.
- [ ] Public status should be visible in GUI lists.
- [ ] Public export includes only readable objects and readable graph edges.

## Secure Network Behavior

Loopback:

- [ ] Binding to `127.0.0.1`, `localhost`, or `::1` is allowed without ZMQ CURVE
  for local development.

Non-loopback:

- [ ] Binding to `0.0.0.0`, a LAN IP, or a public IP requires configured ZMQ
  CURVE security.
- [ ] Startup fails clearly when non-loopback networking is requested without
  transport security.
- [ ] Failure message must say how to switch back to loopback or configure ZMQ
  security.

Redaction:

- [ ] Redact `auth`.
- [ ] Redact `token`.
- [ ] Redact `password`.
- [ ] Redact `password_hash`.
- [ ] Redact session token hashes.
- [ ] Redaction applies to RPC logs, monitor callbacks, GUI error dialogs, and
  test failure helpers where practical.

## Dependent Code To Update

- [ ] `chisurf/plugins/core/mfdb_admin/gui/client.py`
- [ ] `chisurf/plugins/core/mfdb_admin/backend/password_services.py`
- [ ] `chisurf/plugins/core/mfdb_admin/backend/services.py`
- [ ] `chisurf/plugins/core/mfdb_admin/backend/measurement_services.py`
- [ ] `chisurf/plugins/core/mfdb_admin/backend/setup_services.py`
- [ ] `chisurf/plugins/core/mfdb_admin/gui/tool.py`
- [ ] `chisurf/plugins/core/user_editor/gui.py`
- [ ] startup login dialog in `chisurf/gui/__init__.py`
- [ ] `chisurf/core/mfdb/repository.py`
- [ ] `chisurf/core/mfdb/api.py`
- [ ] Chinet MFDB adapter
- [ ] project archive/restore code paths
- [ ] MFDB plugin manifest RPC method list
- [ ] sample database compatibility plugin or remove it if the hard cut makes it
  obsolete
- [ ] `test/fio/test_mfdb_user_management.py`
- [ ] MFDB/FDB provenance tests under `test/fio/`
- [ ] plugin tests that call MFDB services

## Implementation Order

Phase 1: PRD and schema

- [ ] Land this PRD.
- [ ] Add schema version and auth tables.
- [ ] Add indices and built-in group bootstrap.
- [ ] Add migration/backfill tests.

Phase 2: authorization core

- [ ] Add `chisurf.core.mfdb.auth`.
- [ ] Add principal classes and exceptions.
- [ ] Implement token authentication helpers.
- [ ] Implement permission evaluation.
- [ ] Implement ACL mutation helpers.
- [ ] Add focused unit tests for permission evaluation.

Phase 3: sessions and user/group RPC

- [ ] Replace login service with `mfdb.auth.login`.
- [ ] Add logout/me/session list/revoke.
- [ ] Add group management RPC.
- [ ] Add permission management RPC.
- [ ] Remove `requester_id` authorization checks.

Phase 4: protected MFDB objects

- [ ] Protect artifacts.
- [ ] Protect operations.
- [ ] Protect operation-artifact links.
- [ ] Protect parameters.
- [ ] Protect setups.
- [ ] Protect branches.
- [ ] Protect graph traversal/export.
- [ ] Ensure ACL writes share transactions with object writes.

Phase 5: protected scientific/domain services

- [ ] Protect samples.
- [ ] Protect sample conditions.
- [ ] Protect experiments.
- [ ] Protect experiment data.
- [ ] Protect raw data.
- [ ] Protect processed data.
- [ ] Protect project archive/restore.
- [ ] Protect Chinet sessions.

Phase 6: clients and GUI

- [ ] Update `MFDBClient` token handling.
- [ ] Update startup login.
- [ ] Update user editor.
- [ ] Update MFDB plugin user/group/permissions UI.
- [ ] Add public/private indicators.
- [ ] Add owner/group/mode controls for manageable objects.

Phase 7: security hardening

- [ ] Enforce non-loopback ZMQ security requirement.
- [ ] Add RPC redaction.
- [ ] Add auth/session log redaction.
- [ ] Add failed-login throttling.

Phase 8: final verification

- [ ] Run targeted auth tests.
- [ ] Run MFDB user/group tests.
- [ ] Run MFDB/FDB provenance tests.
- [ ] Run MFDB plugin tests.
- [ ] Run startup/login smoke test.
- [ ] Run full relevant pytest subset and record command output in the final
  implementation report.

## Acceptance Tests

Fresh schema:

- [ ] Fresh DB creates `mfdb_group`.
- [ ] Fresh DB creates `mfdb_group_member`.
- [ ] Fresh DB creates `mfdb_object_acl`.
- [ ] Fresh DB creates `mfdb_acl_entry`.
- [ ] Fresh DB creates `mfdb_session`.
- [ ] Fresh DB creates `mfdb_auth_attempt`.
- [ ] Fresh DB bootstraps `admins`, `users`, and `public`.
- [ ] Fresh DB adds admins to `admins`.
- [ ] Fresh DB adds normal users to `users`.

Authentication:

- [ ] Admin bootstrap can create the first admin securely.
- [ ] Login with correct password returns token once.
- [ ] Login with wrong password fails and records attempt.
- [ ] Repeated failed login attempts are throttled.
- [ ] Expired token fails.
- [ ] Revoked token fails.
- [ ] Forged token fails.
- [ ] Unknown token fails.
- [ ] Logout revokes token.
- [ ] `me` returns authenticated user for valid token.
- [ ] `me` rejects anonymous or invalid token.

Object defaults:

- [ ] New sample defaults private.
- [ ] New experiment defaults private or inherits private from sample.
- [ ] New artifact defaults private or inherits from parent.
- [ ] New operation defaults private or inherits from experiment.
- [ ] New parameter inherits from operation.
- [ ] New setup defaults private unless explicitly public/catalog.

Read behavior:

- [ ] Anonymous can read public object.
- [ ] Anonymous cannot read private object.
- [ ] Anonymous list excludes private object IDs.
- [ ] Owner can read private object.
- [ ] Group member can read group-readable object.
- [ ] Explicit user allow grants read.
- [ ] Explicit group allow grants read.
- [ ] Explicit deny blocks read.
- [ ] Admin can read every object.

Write behavior:

- [ ] Anonymous cannot write public object.
- [ ] Anonymous cannot write private object.
- [ ] Owner with write bit can update object.
- [ ] Owner without write bit cannot update object.
- [ ] Group member with group-write can update object.
- [ ] Explicit allow grants write.
- [ ] Explicit deny blocks write.
- [ ] Admin can write every object.

Manage behavior:

- [ ] Owner with manage bit can chmod object.
- [ ] Owner without manage bit cannot chmod object.
- [ ] Group manager with manage permission can grant ACL.
- [ ] Explicit deny blocks manage.
- [ ] Admin can chmod/chown/chgrp/delete every object.
- [ ] Non-admin cannot grant admin privileges.

List/search/graph/export:

- [ ] List endpoints filter unreadable objects.
- [ ] Search endpoints filter unreadable objects.
- [ ] Get endpoints do not leak private object existence.
- [ ] Graph traversal hides unreadable nodes.
- [ ] Graph traversal hides edges with unreadable endpoints.
- [ ] Export includes only readable objects for non-admin users.
- [ ] Project restore cannot load private artifacts without read access.
- [ ] Chinet session restore cannot load private artifacts without read access.

Network and logging:

- [ ] Loopback ZMQ starts without CURVE.
- [ ] Non-loopback ZMQ without CURVE fails startup.
- [ ] Non-loopback ZMQ with CURVE starts.
- [ ] RPC logs redact `auth`.
- [ ] RPC logs redact `password`.
- [ ] RPC logs redact `token`.
- [ ] RPC monitors receive redacted auth-sensitive payloads.

Dependency cleanup:

- [ ] No service uses `requester_id` for authorization.
- [ ] No GUI path assumes settings `default_user_id` is authenticated identity.
- [ ] No test relies on unauthenticated writes.
- [ ] No MFDB protected write succeeds without auth.
- [ ] `MFDBClient` injects auth automatically after login.

## Review Checklist

The reviewer should reject the implementation if any item below is true:

- [ ] A caller can pass a different user id and gain that user's permissions.
- [ ] A private object ID appears in an anonymous list/search/graph result.
- [ ] A write RPC succeeds without authentication.
- [ ] A delete or permission change succeeds without manage permission.
- [ ] A token or password appears in logs.
- [ ] Non-loopback ZMQ starts without transport security.
- [ ] Graph traversal returns unreadable endpoints.
- [ ] Project or Chinet restore bypasses artifact read checks.
- [ ] New scientific objects default public without an explicit request.
- [ ] Compatibility shims preserve insecure behavior.

## Final Implementation Report Requirements

When this PRD is implemented, the final report must include:

- [ ] Changed files grouped by subsystem.
- [ ] Schema version before and after.
- [ ] Migration/backfill policy actually implemented.
- [ ] RPC methods added, removed, and changed.
- [ ] Tests run with exact commands.
- [ ] Known remaining risks.
- [ ] Explicit statement whether any compatibility shim remains and why.
