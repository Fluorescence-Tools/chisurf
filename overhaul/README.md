# ChiSurf MFDB Overhaul

## Reading Order

1. **ARCHITECTURE.md** -- Target architecture. Read this first.
2. **PRD-01-fix-roundtrip.md** -- Make project save/load work. Do this first.
3. ~~**PRD-020-sqlalchemy-mfdb-mapping.md**~~ ✓ DONE -- SQLAlchemy MFDB ORM boundary.
4. ~~**PRD-02a-mmcif-dictionary-infrastructure.md**~~ ✓ DONE -- mmCIF/flrCIF dictionary parsing.
5. ~~**PRD-02-sample-tracking.md**~~ ✓ DONE -- Central sample registry.
6. ~~**PRD-02b-mfdb-admin-overhaul.md**~~ ✓ DONE -- Dictionary-driven mfdb-admin GUI overhaul.
7. ~~**PRD-02c-flrcif-alignment.md**~~ ✓ DONE -- flrCIF schema alignment. See `PRD-02_COMPLETION_REPORT.md`.
8. **PRD-03-result-registry.md** -- Simple API for plugins to register results in MFDB.
9. **PRD-04-burst-pipeline.md** -- Connect burst selection to MFDB.
10. **PRD-05-calibration-provenance.md** -- Track where calibration values come from.
11. **PRD-06-fluorophore-database.md** -- Expand dye catalog with real data.
12. **PRD-07-plugin-integration.md** -- Wire high-priority plugins to MFDB via result registry.
13. **PRD-08-optical-configuration.md** -- Structured optical path schema (excitation source → detector).

## Dependency Graph

```
PRD-01 (fix roundtrip)
  |
  v
PRD-020 ✓ (SQLAlchemy MFDB mapping)
  |
  v
PRD-02a ✓ (mmCIF dictionary infrastructure)
  |
  v
PRD-02 ✓ (sample tracking)
  |
PRD-02b ✓ (mfdb-admin overhaul)
  |
PRD-02c ✓ (flrCIF alignment)
  |
  v
PRD-03 (result registry)  <-- all later PRDs depend on this
  |
  +---> PRD-04 (burst pipeline)
  +---> PRD-05 (calibration)
  +---> PRD-06 (fluorophore DB)
  +---> PRD-07 (plugin integration)
  +---> PRD-08 (optical configuration)
```

## Existing Work -- Do Not Duplicate

Before starting, be aware of these existing files that overlap with this overhaul.
Read them to avoid reinventing the wheel, but do NOT treat them as authoritative --
the overhaul PRDs supersede them where they conflict.

| File | What It Contains | Relationship to Overhaul |
|------|-----------------|--------------------------|
| `AGENTS.md` (lines 185-213) | "Object Store & Provenance" docs for schema v27 | **Stale** -- schema is now v28. ExperimentReader provenance hooks described there already exist and work. PRD-03 should build on them, not replace them. |
| `OBJECT_STORE_IMPLEMENTATION.md` | Full implementation guide for content-addressed store + provenance | **Reference** -- the object store and reader provenance phases are done. PRD-03's result registry wraps this existing infrastructure. |
| `AGENT/ROADMAP.md` (lines 33-59) | "Full-Fidelity Project Save/Restore v3" at 60% complete | **Overlaps PRD-01** -- has its own phase plan (UID registry, linking correctness, save/load pipeline). PRD-01 supersedes this. |
| `AGENT/TODO.md` | MVC migration status (55/55 complete) | **Informational** -- MVC action controller is done. Plugins can use it. |
| `BUGS.md` bug #1 | "Save project → close project → open project causes bug" | **Directly relevant to PRD-01** -- this is one of the bugs PRD-01 fixes. |

## Ground Rules for Agents

- Breaking changes are allowed. Do not preserve backward compatibility with broken code.
- Delete dead code. Do not comment it out or alias it.
- The canonical tables are `mfdb_*`. The `fdb_*` and legacy FLR tables stay for now but
  new code must only use `mfdb_*` tables.
- Every function you write must have a test.
- Use dataclasses, not dicts, for structured data.
- No "TODO" or "FIXME" comments. Either fix it or don't touch it.
- Calibration values (R0, g-factor, gamma, etc.) may be "god given" -- entered by the
  user without any backing measurement. Always support both paths: (a) derived from a
  reference measurement with full provenance, and (b) entered manually as a fixed value.
  Use `method="user_provided"` for the latter.
