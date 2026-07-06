---
type: Reference
title: Visual node/workflow toolkit — architecture lessons
description: Durable architecture lessons drawn from an established visual node/workflow analysis toolkit, mapped to ChiSurf's provenance, transformer, and schema PRDs.
tags: [reference, architecture, mfdb]
timestamp: '2026-07-06T00:00:00Z'
---

# Reference

This concept captures the durable architecture lessons ChiSurf draws from an
**established, mature visual node/workflow analysis toolkit** — a node/canvas
data-mining environment that has, for well over a decade, solved several of the
abstractions ChiSurf's operation / transformer / lineage PRDs are reaching for.

That toolkit is deliberately in-memory, single-process, matrix-centric, and
single-user, with no persistence layer, no multi-owner ACL, and no
content-addressed store. ChiSurf takes its *concepts*, not its storage model:
`chisurf`'s persistence, multi-owner access, content-addressing, and
flrCIF-authoritative `mfdb` schema are all out of that toolkit's scope. What
follows are the abstractions worth adopting, each linked to the PRD it informs.

## Replayable lineage: provenance baked into the data model

The toolkit's highest-leverage idea. Every derived column can carry a
serializable *compute-value* — a transformation object that knows **how to
recompute this column from its source**. Lineage is therefore intrinsic and
recomputable, not a side-table afterthought: a derived value *is* its
derivation rule plus a source reference, applying the same transform to new
data "just works," and the transformation objects are serializable.

**Lesson for ChiSurf → [PRD-21](/prds/prd-21.md) (lineage) and
[PRD-27](/prds/prd-27.md) (append-only provenance / what-if branches).** Today
`mfdb` provenance lives only in the `mfdb_edge` / `mfdb_operation` side tables.
The compute-value pattern suggests the *artifact itself* should also carry a
serializable "how I was produced" spec (operation type + parameters + source
artifact ids). Then an artifact is replayable, not merely traceable;
"what-if" branches become "re-run this artifact's compute spec with one changed
parameter"; and the side-table edges become a projection/index over the
embedded specs rather than the source of truth.

## Typed ports: the transformer contract

Nodes declare their inputs and outputs as **typed, named ports**, matched by
type. The canvas validates a connection only when the output type is compatible
with the input type, and a signal manager routes values and re-invokes handlers
reactively. Ports carry multiplicity (single/multiple) and a dynamic flag.

**Lesson for ChiSurf → [PRD-16](/prds/prd-16.md) (transformer contract) and
[PRD-11](/prds/prd-11.md) (operation nodes).** ChiSurf's role-indexed
parameters and input/output kinds should be **declared, typed, named ports with
multiplicity**, with connections validated by kind — the same model already
matched by chinet's reactive ports. Adopt declarative port specs on every
transformer, validate wiring by kind, and reject mismatches at the boundary.

## Transformer-as-value: operations are serializable data

A transformation in the toolkit is an *object*, not a function: it can be a
node output, composed into a list, stored, passed between nodes, and applied
later to any compatible data. The transformation is data.

**Lesson for ChiSurf → [PRD-22](/prds/prd-22.md) (pipeline engine) and
[PRD-16](/prds/prd-16.md).** Model an operation/transformer as a **serializable
spec value** (identity + typed parameters) separate from its execution. A
pipeline then becomes data you can store, diff, share, and replay — exactly
what [PRD-22](/prds/prd-22.md) wants and what makes [PRD-27](/prds/prd-27.md)
branches meaningful. The `.dic`-defined parameters of [PRD-11](/prds/prd-11.md)
supply the schema for that spec.

## Typed schema object and explicit schema conversion

The toolkit separates *data* (the value matrices) from *schema* (a first-class
object of typed variables: continuous, discrete, string, time). Variables carry
a metadata dict and are deduplicated by identity, and an explicit
schema-conversion object maps a table from one schema to another
(reordering/deriving columns), enabling clean adaptation across schema versions.

**Lesson for ChiSurf → [PRD-26](/prds/prd-26.md) (model-driven data layer) and
[PRD-25](/prds/prd-25.md) (typed IDs / units).** A typed schema object — the
`.dic`-generated `mfdb` model — is the right shape; the toolkit validates the
"typed columns, not bare strings" direction, and its time/units-bearing
variables echo first-class units. Its explicit schema-conversion object is a
clean pattern for the flrCIF codec and schema reconcile: adapt records between
an old and a target schema through an explicit conversion object rather than
ad-hoc migration SQL.

## Annotation channel separate from payload

The toolkit's tables carry a *metas* channel: columns that ride along with the
data (ids, labels, provenance) but are excluded from the analysis matrix — a
clean separation of payload versus annotation. This mirrors and validates
`mfdb`'s split of metadata/provenance columns from payload: keep artifact
metadata as a distinct, typed, queryable channel rather than a JSON blob.

## Data-only settings migration

Nodes version only their *stored state/parameters*, with small stepwise
migrators; structure is implicit in code, and only serialized *values* are
migrated. The toolkit never hand-writes structural DDL migrations.

**Lesson for ChiSurf → [PRD-19](/prds/prd-19.md) (versionless declarative
schema).** This validates ChiSurf's chosen direction: keep structure
declarative (the `reconcile_schema` path) and reserve version-stamped
migrations for **data/params only** (run-once data backfills). Cite as prior
art for the declarative-structure + data-only-migration split.

## Workflows as saved, shareable documents

The node canvas serializes to a workflow file — nodes plus typed links — that
can be saved, reloaded, and shared. The graph *is* a document.

**Lesson for ChiSurf → [PRD-22](/prds/prd-22.md) and
[PRD-27](/prds/prd-27.md).** Make pipelines saveable, shareable documents, and
tie a saved pipeline to a [PRD-27](/prds/prd-27.md) branch so a shared workflow
is reproducible against recorded data.

## Priority ordering

1. Replayable, embedded provenance on artifacts — highest leverage; fold into
   [PRD-21](/prds/prd-21.md) / [PRD-27](/prds/prd-27.md).
2. Typed, declared, kind-matched ports for transformers — fold into
   [PRD-16](/prds/prd-16.md) / [PRD-11](/prds/prd-11.md).
3. Operation/transformer as a serializable spec value — fold into
   [PRD-22](/prds/prd-22.md) / [PRD-16](/prds/prd-16.md).
4. Explicit schema-adaptation objects — reference in
   [PRD-19](/prds/prd-19.md) (reconcile) and the flrCIF codec.
5. Data-only settings migration — confirms [PRD-19](/prds/prd-19.md) as prior
   art for declarative structure with data-only version migrations.
