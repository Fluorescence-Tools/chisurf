# ChiSurf MFDB Overhaul

## Reading Order

1. **ARCHITECTURE.md** -- Target architecture. Read this first.
2. **PRD-01-fix-roundtrip.md** -- Make project save/load work. Do this first.
3. ~~**PRD-020-sqlalchemy-mfdb-mapping.md**~~ ✓ DONE -- SQLAlchemy MFDB ORM boundary.
4. ~~**PRD-02a-mmcif-dictionary-infrastructure.md**~~ ✓ DONE -- mmCIF/flrCIF dictionary parsing.
5. ~~**PRD-02-sample-tracking.md**~~ ✓ DONE -- Central sample registry.
6. ~~**PRD-02b-mfdb-admin-overhaul.md**~~ ✓ DONE -- Dictionary-driven mfdb-admin GUI overhaul.
7. ~~**PRD-02c-flrcif-alignment.md**~~ ✓ DONE -- flrCIF schema alignment. See `PRD-02_COMPLETION_REPORT.md`.
8. **PRD-03-result-registry.md** -- Simple API for plugins to register results in MFDB.  ◄ CURRENT (3/4)
9. **PRD-04-burst-pipeline.md** -- Connect burst selection to MFDB.  ◄ CURRENT (3/4)
10. **PRD-05-calibration-provenance.md** -- Track where calibration values come from.  *(g-factor part delivered by PRD-04 sidequest C/D; gamma/crosstalk/R0 remain)*
11. **PRD-06-fluorophore-database.md** -- Expand dye catalog with real data.
12. ~~**PRD-07-plugin-integration.md**~~ -- SUPERSEDED by PRD-16 (uniform transformer contract replaces ad-hoc per-plugin `register_result` calls).
13. **PRD-08-optical-configuration.md** -- Structured optical path schema (excitation source → detector).

### Added during the 03/04 line (largely built; not yet closed)

14. **PRD-09-microtime-shifter-plugin.md** -- Microtime Shifter as a workflow plugin + MFDB; first transformer.
15. **PRD-10-dataset-browser-widget.md** -- Reusable MFDB dataset browser + artifact ownership (multi-owner).

### Operation/transformer spine + LIMS layers (downstream of 03/04)

16. **PRD-11-data-operation-abstraction.md** -- Transformers as abstract operation nodes; `.dic`-typed parameters.
17. **PRD-16-transformer-contract.md** -- Uniform transformer plugin contract (ships with PRD-11; supersedes PRD-07).
18. **PRD-12-lifecycle-state-machine.md** -- LIMS P1: tracked lifecycles + transition history.
19. **PRD-13-study-project-entity.md** -- LIMS P2: study/project grouping + configurable fields.
20. **PRD-14-protocol-entity.md** -- LIMS P3: named, versioned procedures.
21. **PRD-15-reagent-inventory.md** -- LIMS P4: consumable lots/expiry (lowest priority).

### Architecture track (cross-cutting; de-risks and cleans up the above)

22. **PRD-17-identity-session-context.md** -- One canonical current-user/session context.
23. **PRD-18-dependency-injection-test-harness.md** -- Injected db/session + hermetic test harness.
24. **PRD-19-dict-vocab-declarative-migrations.md** -- Single canonical `.dic`-driven schema (folds ideas **I**+**K**); flrCIF authoritative, extended via the `.dic`; drop legacy + duplicate tables; versionless reconcile. *(Phase-1 foundation.)*
25. **PRD-21-lineage-api-event-model.md** -- Provenance/lineage query API + event bus (projects over PRD-27).
26. **PRD-22-pipeline-workflow-engine.md** -- Compose transformers into recorded pipelines.
27. **PRD-23-thin-widgets-view-api.md** -- View-only widgets; mandatory construction smoke tests.
28. **PRD-24-mfdb-package-extraction.md** -- Extract MFDB into `modules/mfdb` (capstone).
29. **PRD-25-consistency-hardening.md** -- Fail-loud, RPC envelope, caching, N+1, dead code + idea **N** (typed IDs, units, boundary validation).
30. **PRD-26-model-driven-data-layer.md** -- Idea **J**: the `.dic` generates DAO/repository, admin registry, RPC validation, and docs (on PRD-19).
31. **PRD-27-event-sourced-provenance-core.md** -- Idea **M**: append-only provenance/state core with branching. *(Phase-1 decision; PRD-12/21 project over it.)*
32. **PRD-28-ndxplorer-burst-integration.md** -- ndXplorer ↔ MFDB burst-selection round trip (send to / open from ndXplorer via the dataset picker). *(Phase-2 manual-test enabler.)*
33. **PRD-29-visual-burst-programming.md** -- visual burst programming workflow.
34. **PRD-30-cli-pipeline-tools.md** -- Unix pipe support for burst/TTTR CLI tools.
35. **PRD-31-ndxplorer-headless-cli.md** -- headless ndXplorer burst filtering + imaging CLI.
36. **PRD-32-acquisition-output-folder.md** -- setup-defined standard acquisition output folder.
37. **PRD-33-acquisition-mfdb-registration.md** -- direct acquisition registration into MFDB.
38. **PRD-38-model-view-spec-split.md** -- strict model/UI split: user-editable `<model>.view.json` drives auto-generated model editors. ◄ CURRENT (3/4: data spine + Lifetime pilot done; live wiring next)
39. **PRD-39-sequence-external-references.md** -- entity ↔ UniProt/PDB cross-references + engineered-mutation provenance (cysteine labeling) via the standard `struct_ref`/`struct_ref_seq`/`struct_ref_seq_dif` categories; live UniProt/SIFTS fetch + auto-diff. *(Builds on PRD-02.)*
40. **PRD-40-declarative-dataset-editors.md** -- a chisurf-native, guidata-like framework: declare a typed `DataSet` once → auto-generate its editor, replacing the ~5 ad-hoc `type→widget` mappers (settings/metadata/parameter/model editors). Generalises PRD-38's machinery out from under `models/`. *(Builds on PRD-38; complements PRD-23/26.)*

## ► Authoritative sequence: see **MASTER-ORDER.md**

`MASTER-ORDER.md` is the single ordered plan for all PRDs (phases, dependency
graph, parallelization). Supporting analysis: **MFDB-LIMS-diagnosis.md** (LIMS gap
behind PRD-12–15), **MFDB-architecture-ideas.md** (rationale behind PRD-17–27;
ideas A–H plus bold I–N, with I/K→19, J→26, M→27, N→25), and **ORANGE3-lessons.md**
(node/workflow + provenance patterns from `thirdparty/orange3`: `compute_value`
replayable lineage→21/27, typed ports→16/11, transformer-as-value→16/22,
DomainConversion + `migrate_settings`→19).

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
PRD-03 (result registry)  <-- CURRENT (3/4); all later PRDs depend on this
  |
  +---> PRD-04 (burst pipeline)  <-- CURRENT (3/4)
  |        +---> PRD-09 (microtime shifter)   first transformer
  |        +---> PRD-10 (dataset browser + ownership)
  +---> PRD-05 (calibration)  [g-factor done via PRD-04 sidequest C/D]
  +---> PRD-06 (fluorophore DB)
  +---> PRD-08 (optical configuration)
  |
  v
[Phase-1 foundations — do before the spine: PRD-18 tests, PRD-17 identity,
 PRD-19 canonical schema (I+K), PRD-27 append-only core (M); see MASTER-ORDER]
  |
  v
PRD-11 (operation-node abstraction) ──► PRD-16 (transformer contract; supersedes PRD-07)
  |                                    └► PRD-26 (model-driven layer J; on PRD-19+11)
  |                                         (refactor PRD-09 + PRD-04 burst as the 2 reference transformers)
  |     └──► PRD-14 (protocols; needs PRD-11 param schemas)
  ├──► PRD-12 (lifecycle state machine — LIMS P1)
  ├──► PRD-13 (study/project — LIMS P2)
  └──► PRD-15 (reagent inventory — LIMS P4)
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
