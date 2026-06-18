# PRD-020: SQLAlchemy MFDB Mapping

**Prerequisite for:** [PRD-02a: mmCIF Dictionary Infrastructure](PRD-02a-mmcif-dictionary-infrastructure.md), [PRD-02: Sample Tracking](PRD-02-sample-tracking.md)
**Priority:** High — establish a durable MFDB relationship layer before adding more schema-heavy sample and dictionary behavior.

## Goal

Introduce a deliberate SQLAlchemy mapping boundary for MFDB so relationship-heavy
features are expressed once, tested once, and reused by the existing repository
API. This PRD does **not** replace MFDB wholesale. It creates a bounded
SQLAlchemy slice for the tables that PRD-02a and PRD-02 will depend on most:
sample, probe, FRET pair, vocabulary, and dictionary metadata tables.

The immediate architectural driver is the R14 review finding: raw SQL currently
lets FRET pair data be globally scoped by probe IDs, which causes cross-sample
leaks and uniqueness failures. SQLAlchemy will not fix that schema bug by
itself, but a mapped relationship layer makes the correct sample-scoped model
harder to bypass.

## Non-goals

- Do not rewrite all of `MFDatabase` in one pass.
- Do not introduce a second, competing schema definition.
- Do not mix ad hoc SQLAlchemy sessions into GUI/plugin code.
- Do not change the public MFDB API unless a method is already broken.
- Do not migrate object-store blob logic in this PRD.

## Design constraints

- `chisurf/core/mfdb/schema.py` remains the canonical migration source during
  this phase.
- Existing callers continue using `MFDatabase` and high-level modules such as
  `sample_manager.py`.
- SQLAlchemy models live behind a small internal adapter; callers do not create
  sessions directly.
- All SQLAlchemy transactions must share the same sqlite database path and must
  not race the existing `sqlite3` connection inside a single logical operation.
- No table mapping may hide missing schema fields. If a relationship needs a
  foreign key or association table, add the schema change explicitly.

## Proposed package layout

```text
chisurf/core/mfdb/
  orm/
    __init__.py
    base.py              # DeclarativeBase, engine/session helpers
    models.py            # mapped classes for bounded MFDB slice
    sample_repository.py # SQLAlchemy-backed sample/probe operations
    sync.py              # schema/mapping consistency assertions
```

## Tables in scope

### Phase A — sample/probe core

- `mfdb_sample`
- `flr_sample`
- `flr_sample_condition`
- `flr_sample_probe`
- `flr_poly_probe_position`
- `entities`
- `entity_poly_seq`
- `probes`
- `chem_descriptors`
- `ihm_chemical_component_descriptor`
- `optical_properties`
- `spectra`
- `flr_fret_forster_radius`

### Phase B — vocabulary/dictionary support

- `mfdb_vocabulary`
- new dictionary cache tables only if PRD-02a chooses database-backed cache
  instead of JSON-only cache.

## Required schema correction before mapping

`flr_fret_forster_radius` must become sample-scoped before the ORM mapping is
considered complete.

Recommended minimal schema:

```sql
ALTER TABLE flr_fret_forster_radius ADD COLUMN sample_id TEXT REFERENCES flr_sample(sample_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_flr_forster_sample_pair
  ON flr_fret_forster_radius(sample_id, donor_probe_id, acceptor_probe_id)
  WHERE deleted_at IS NULL;
```

Preferred flrCIF-aligned schema:

```sql
ALTER TABLE flr_fret_forster_radius ADD COLUMN sample_id TEXT REFERENCES flr_sample(sample_id);
ALTER TABLE flr_fret_forster_radius ADD COLUMN donor_sample_probe_id INTEGER REFERENCES flr_sample_probe(sample_probe_id);
ALTER TABLE flr_fret_forster_radius ADD COLUMN acceptor_sample_probe_id INTEGER REFERENCES flr_sample_probe(sample_probe_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_flr_forster_sample_probe_pair
  ON flr_fret_forster_radius(sample_id, donor_sample_probe_id, acceptor_sample_probe_id)
  WHERE deleted_at IS NULL;
```

Use the preferred schema unless compatibility with existing seeded data makes it
too expensive.

## Tasks

### Task 1: Add SQLAlchemy as an explicit project dependency

Add SQLAlchemy to the project dependency configuration used by the active
development environments. Keep the version bounded conservatively.

Acceptance criteria:

- `import sqlalchemy` works in the arm64 development environment.
- Dependency metadata is updated in the project-owned dependency files.
- No GUI or plugin dependency imports SQLAlchemy directly.

### Task 2: Create ORM base and session adapter

Create `chisurf/core/mfdb/orm/base.py`.

Required API:

```python
def make_engine(db_path: str | Path):
    """Return a SQLAlchemy engine for an MFDB sqlite database."""

def session_scope(db_path: str | Path):
    """Yield a short-lived SQLAlchemy session with commit/rollback handling."""

def session_from_mfdatabase(db: "MFDatabase"):
    """Return a session bound to the same database path when safe."""
```

If binding safely to an existing `MFDatabase` instance is not possible, document
that SQLAlchemy-backed methods open short-lived sessions by path and must not be
used inside active `db._transaction()` blocks.

### Task 3: Map the bounded sample/probe model

Create mapped classes for the Phase A tables with explicit relationships:

- `MfdbSampleIndex`
- `FlrSample`
- `FlrSampleCondition`
- `FlrSampleProbe`
- `FlrPolyProbePosition`
- `Entity`
- `EntityPolySeq`
- `Probe`
- `OpticalProperty`
- `Spectrum`
- `FlrFretForsterRadius`

Acceptance criteria:

- Relationships express:
  `FlrSample -> sample_probes -> Probe`
  `FlrSampleProbe -> poly_probe_position -> Entity`
  `FlrSample -> condition`
  `FlrSample -> fret_pairs`
- `FlrFretForsterRadius` is scoped to one sample.
- ORM model metadata can be compared against the live sqlite schema without
  creating or mutating tables.

### Task 4: Build a SQLAlchemy-backed sample repository adapter

Create `chisurf/core/mfdb/orm/sample_repository.py`.

Required methods:

```python
def create_sample_graph(db: MFDatabase, definition: SampleDefinition) -> str:
    """Persist a full sample graph and return sample_id."""

def get_sample_graph(db: MFDatabase, sample_id: str) -> dict | None:
    """Return sample, entities, probes, positions, condition, and FRET pairs."""

def upsert_probe(db: MFDatabase, probe: ProbeDefinition) -> int:
    """Persist probe identity plus chemical descriptors and return probe_id."""
```

This adapter can initially be called only from tests. Once it is verified,
`sample_manager.create_sample()` and `get_sample_full_description()` can delegate
to it.

### Task 5: Preserve public API while moving implementation behind the adapter

Refactor existing sample-manager functions to call the SQLAlchemy-backed adapter
without changing their public signatures:

- `create_sample()`
- `get_sample_full_description()`
- `validate_sample_for_export()`

Keep the lightweight `mfdb_sample` index behavior intact.

### Task 6: Add schema/mapping consistency checks

Create tests that fail when mapped columns drift from `schema.py`.

Minimum checks:

- Every mapped column exists in the live sqlite schema.
- Every required relationship has a foreign key or explicit join condition.
- `flr_fret_forster_radius` contains sample scope.
- No ORM model calls `Base.metadata.create_all()` against production databases.

### Task 7: Regression tests for R14 findings

Add tests for:

1. Two samples with the same donor/acceptor dye names but different R0 values
   can both be created.
2. A sample sharing only one probe with another sample does not return the other
   sample's FRET pair.
3. Probe chemical fields persist to canonical probe/descriptor tables.
4. Request-layer custom probe names warn instead of raising.

## Definition of done

- SQLAlchemy dependency is declared and importable.
- ORM mapping exists for the bounded MFDB sample/probe slice.
- Existing public sample APIs still work.
- R14-1 and R14-4 are fixed by schema/API behavior, not only by metadata JSON.
- Focused sample tests pass without relying on global probe-pair uniqueness.
- PRD-02a can build dictionary validation on top of a stable MFDB relationship
  boundary.

## Risks

- **Dual transaction model:** sqlite3 and SQLAlchemy sessions can conflict if
  mixed inside the same operation. Mitigate by routing sample-graph operations
  through one adapter and documenting transaction boundaries.
- **Schema drift:** SQLAlchemy mappings can become a second schema source.
  Mitigate with schema/mapping consistency tests and by keeping `schema.py`
  canonical for migrations.
- **Scope creep:** Full MFDB ORM migration is tempting. Keep this PRD limited to
  sample/probe/FRET/vocabulary surfaces needed before PRD-02a and PRD-02.

