# PRD-24: Extract MFDB into a Standalone Package (Architecture G)

## Goal

Move MFDB (schema, dictionary, generator, repository/API, server, mfdb-admin) out
of `chisurf` into an independent `modules/mfdb` package — like `chinet` — with a
stable public API and **no `chisurf` imports**. Forces a clean boundary, enables
reuse and independent testing.

## Background

`overhaul/TODO.md` already lists this as deferred cleanup: "Move MFDB out of
`chisurf` into `modules/` … own the schema, repository/API layer, MFDB server, and
mfdb-admin … after the current overhaul is basically finished so the extraction
can preserve stabilized interfaces instead of moving churn."

## When

**Last** — after the operation/transformer spine (PRD-11/16), the identity/DI
work (PRD-17/18), and the dict/migration cleanup (PRD-19) have stabilized the
interfaces. Extracting earlier just moves churn.

## Design

- `modules/mfdb/` owns: `schema` + `.dic` + generator, `repository`/`api`,
  `object_store`, `result_registry`, `auth`, `server`, and the admin backend. The
  GUI admin *frontend* may stay in chisurf (Qt) but talks only through the package's
  RPC/API.
- **No `chisurf` imports** in the package. Settings/paths it needs (db path, object
  store root, default user) are injected (PRD-17 `SessionContext` / explicit config),
  not read from `chisurf.core.settings`.
- A thin `chisurf` adapter wires chisurf settings/GUI to the package API.
- The package is independently installable and testable (its own hermetic test
  suite from PRD-18).

## Tasks

1. Inventory and cut the chisurf→mfdb dependency edges; replace
   `chisurf.core.settings` reads with injected config (depends on PRD-17/18).
2. Move the modules; keep import shims in `chisurf.core.mfdb` only as a transitional
   re-export, then delete.
3. Package metadata + standalone test run (uses the hermetic harness).
4. chisurf adapter: config/GUI ↔ package API.
5. CI: the package builds and tests on its own.

## Definition of Done

- [ ] `modules/mfdb` is a standalone package with no `chisurf` imports; injected
      config.
- [ ] chisurf uses it via a thin adapter; transitional shims removed.
- [ ] Package builds and its hermetic tests pass independently.

## Definition of Clean

No `chisurf` imports in the package; config injected not global; the hermetic test
harness (PRD-18) travels with the package; delete shims, don't alias.

## Relationship

Capstone of the architecture track. Requires PRD-17 (injected identity/config) and
PRD-18 (injected DB + hermetic tests) to be in place first; benefits from PRD-19
(self-contained dict-driven schema).
