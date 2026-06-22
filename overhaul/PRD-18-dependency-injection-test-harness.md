# PRD-18: Dependency Injection + Hermetic Test Harness (Architecture B)

## Goal

Stop resolving the database/dictionary/user via module-global functions
re-imported into many namespaces; pass them explicitly. Make the test suite
**hermetic** so no test can touch the real user database, and define an explicit
client interface tested against the *real* in-process client.

## Evidence (why)

- `resolve_database_path` is `from … import`-bound into `services`,
  `setup_services`, `api`, `database_resolver`. Patching one missed `api`, so the
  setup-handler tests read the **real** `~/.chisurf/...db` and polluted each other
  (a leaked `rpc_test_setup` row made them "pass"). Repros during development
  mutated the live DB.
- `MFDBClient` exposed `_call` but callers used `.call` → `AttributeError`
  swallowed → the dataset browser silently showed 0. Tests passed because they
  injected a **mock** client that had `.call`.

## Design

1. **Inject the DB/session** (PRD-17 `SessionContext`) into handlers and repository
   entry points instead of each calling `resolve_database_path()`. One resolution
   site at the boundary.
2. **Hermetic test harness:** an autouse `conftest` fixture (project-wide for
   `test/` and plugin tests) that points the settings dir + resolved DB + object
   store at a per-session temp location. No test can read or write the user DB.
   (Removes the whole class of cross-file pollution and protects user data.)
3. **Explicit client Protocol:** `MFDBClientBase`/`MFDBClient` declare `call`,
   `close` as the public contract; integration tests drive the **real** in-process
   client (not a mock) so interface drift (`.call` vs `_call`) fails loudly.
4. Where module-global resolution must stay (legacy), funnel it through one
   indirection that the harness overrides.

## Tasks

1. Project-wide autouse fixture redirecting settings/db/object-store to temp.
2. Add public `call` to `MFDBClient`; an integration test exercising the real
   client end-to-end (browse + open) — would have caught the `.call` bug.
3. Refactor handlers to take the session/db from PRD-17 rather than re-resolving.
4. Audit + remove `from … import resolve_database_path` namespace-binding that
   makes patching fragile; prefer `database_resolver.resolve_database_path()` calls
   or injected paths.
5. Tests: full-suite run is order-independent (no cross-file pollution); a guard
   test asserts the real user DB path is never opened during tests.

## Definition of Done

- [ ] Autouse harness: no test touches the real user DB/object store; suite is
      order-independent.
- [ ] `MFDBClient.call` is the public contract; an integration test uses the real
      client.
- [ ] Handlers take an injected session/db; fragile namespace-bound resolution
      removed.

## Definition of Clean

DI over monkeypatching; integration over mock-only; behavior-asserting tests;
no silent swallow of interface errors.

## Relationship

Carries PRD-17's `SessionContext` as the injected unit; the `MFDBClient.call`
fix from PRD-10 becomes the contracted interface here.
