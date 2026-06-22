# Orange3 — architecture lessons for MFDB / chisurf

Source: `thirdparty/orange3` (Orange Data Mining). Orange is a mature, node/workflow-
based data-mining toolbox. It is **in-memory, single-process, numpy-matrix-centric,
single-user** — so we take its *concepts*, not its storage. But several of its core
abstractions are exactly what our operation/transformer/lineage PRDs are reaching for,
and it has solved them cleanly for ~15 years. Mapped to our PRDs below.

## The big one: provenance baked into the data model (`compute_value`)

Orange's most valuable idea. Every derived `Variable` (column) can carry a
`compute_value` — a serializable `Transformation` object that knows **how to recompute
this column from its source**. When a preprocessor transforms a table, it does:

```python
features = [var.copy(compute_value=impute.ReplaceUnknowns(var, value)) for var in ...]
```

So lineage is **intrinsic and recomputable**, not a side-table afterthought:
- A derived column *is* its derivation rule + source reference.
- Applying the same transform to new data "just works" (`data.transform(domain)`),
  because the target domain's variables carry their compute rules. → **reproducibility
  for free.**
- Transformations are picklable (note the `__getstate__`/`__setstate__` care).

**Lesson for us → PRD-21 (lineage) + PRD-27 (append-only provenance core).** Today our
provenance lives only in `mfdb_edge`/`mfdb_operation` side tables. Orange suggests the
*artifact itself* should also carry a serializable "how I was produced" spec
(operation type + parameters + source artifact ids). Then:
- An artifact is replayable/recomputable, not just traceable.
- PRD-27 "what-if" branches become "re-run this artifact's compute spec with one
  changed parameter" — a first-class, data-level operation.
- The side-table edges become a *projection/index* of the embedded specs, not the
  source of truth.

## Typed I/O signals = the transformer contract

Orange widgets declare ports as typed, named signals, matched by Python type:

```python
class Inputs:
    data = Input("Data", Orange.data.Table)
class Outputs:
    preprocessor = Output("Preprocessor", preprocess.Preprocess, dynamic=False)
    preprocessed_data = Output("Preprocessed Data", Orange.data.Table)
```

The canvas validates a connection only if output type ⊆ input type; a signal manager
routes values and re-invokes handlers on change (reactive). Ports have multiplicity
(single/multiple) and a `dynamic` flag.

**Lesson for us → PRD-16 (transformer contract) + PRD-11 (operation nodes).** Our
"role-indexed parameters / input-output kinds" should be **declared, typed, named
ports with multiplicity**, and connections validated by *kind* — exactly Orange's
model, which already matches chinet's reactive ports. Adopt: declarative port specs on
every transformer; validate wiring by kind; reject mismatches at the boundary
(reinforces PRD-25 N3).

## Transformers as first-class, serializable *values*

A `Preprocess` is an object, not a function: it can be an **Output** (`Output(
"Preprocessor", Preprocess)`), composed (`PreprocessorList`), stored, passed between
nodes, and applied later to any data. The transformation is data.

**Lesson for us → PRD-22 (pipeline engine) + PRD-16.** Model an operation/transformer
as a **serializable spec object** (identity + typed parameters), separate from its
execution. Then a pipeline is *data you can store, diff, share, and replay* — which is
precisely what PRD-22 wants and what makes PRD-27 branches meaningful. Our
`.dic`-defined parameters (PRD-11 option a) become the schema for that spec.

## `Domain` = a first-class typed schema object

Orange separates *data* (`Table`: X/Y/metas matrices) from *schema* (`Domain`: typed
`Variable`s). Variables are typed — `ContinuousVariable`, `DiscreteVariable`,
`StringVariable`, `TimeVariable` — carry a `.attributes` metadata dict, and are
deduplicated by identity. `DomainConversion` maps a table from one domain to another
(reorders/derives columns), enabling clean adaptation between schema versions.

**Lesson for us → PRD-26 (model-driven data layer) + PRD-25 N (typed IDs/units).**
- A typed schema object (our `.dic`-generated model) is the right shape; Orange
  validates the "typed columns, not bare strings" direction.
- `TimeVariable` and units-bearing variables echo PRD-25 N2 (first-class units).
- **`DomainConversion` is a clean pattern for the flrCIF codec / schema reconcile**:
  adapt records between an old and a target schema via an explicit conversion object,
  rather than ad-hoc migration SQL.

## `metas` channel — annotations that travel with data but aren't features

Orange tables have `metas`: columns that ride along with the data (ids, labels,
provenance) but are excluded from the analysis matrix. Clean separation of *payload*
vs *annotation*.

**Lesson for us.** Mirrors our metadata/provenance columns vs payload split — validates
keeping artifact metadata as a distinct, typed, queryable channel rather than a JSON
blob.

## Localized, data-only state migration (`settings_version` / `migrate_settings`)

Widgets version only their *stored state/params*, with small steppers:
```python
settings_version = 2
@classmethod
def migrate_settings(cls, settings, version): ...  # only transforms saved params
```
Structure is implicit in code; only serialized *values* are migrated.

**Lesson for us → PRD-19 K (versionless declarative schema).** This *validates* our
chosen direction: keep structure declarative (our `reconcile_schema`), and reserve
version-stamped migrations for **data/params only** (our "run-once data backfills").
Orange never hand-writes structural DDL migrations either.

## Workflows are saved, shareable documents (the scheme / `.ows`)

The canvas scheme (nodes + typed links) serializes to a workflow file you can save,
reload, and share. The graph *is* a document.

**Lesson for us → PRD-22 + PRD-27.** Make pipelines saveable/shareable documents, and
tie a saved pipeline to a PRD-27 branch so a shared workflow is reproducible against
recorded data.

## What NOT to copy

Orange is in-memory, single-process, numpy-centric, single-user, no persistence/DB,
no multi-owner/ACL, no content-addressed store. Our persistence, multi-owner, content-
addressing, and flrCIF-authoritative schema are out of its scope. Take the
*abstractions* (compute_value lineage, typed ports, transformer-as-value, Domain,
DomainConversion, metas), not the storage model.

## Priority recommendations

1. **`compute_value`-style embedded, replayable provenance** on artifacts — fold into
   **PRD-21/PRD-27**. Highest-leverage idea; turns lineage from "traceable" into
   "recomputable" and makes PRD-27 branches concrete.
2. **Typed, declared, kind-matched ports** for transformers — fold into **PRD-16/11**.
3. **Operation/transformer as a serializable spec value** — fold into **PRD-22/16**.
4. **`DomainConversion`-style explicit schema adaptation** — reference in **PRD-19**
   (reconcile) and the flrCIF codec.
5. Orange's `migrate_settings` *confirms* PRD-19 K (declarative structure, data-only
   version migrations) — cite as prior art.
