---
type: Specification
title: MFDB — Metadata & Provenance Store — Target
description: The clean-architecture target for metadata and provenance — dictionary-generated schema, provenance DAG, one access layer.
resource: modules/mfdb/src/mfdb/
tags: [target, mfdb, metadata, provenance, schema]
timestamp: '2026-07-06T00:00:00Z'
---

> The clean-architecture target for metadata and provenance. Current shape: [MFDB architecture](/architecture/mfdb.md). Current-state gaps: [assessment](assessment.md).

## Purpose

MFDB is the vendored package for ChiSurf's memory of *what was done*: which samples and measurements
exist, how they were analyzed, and where every result came from. It records
metadata and provenance — not the heavy numeric arrays themselves, which it
references. It is the system of record behind projects and results; the
[Core](core.md) produces the science, and MFDB remembers its lineage.

## Design principles

- **Schema authored upstream, once.** The database structure derives from
  controlled dictionaries (the mmCIF/PDB-IHM family plus a local fluorescence
  extension). The dictionaries are the authority; the tables are generated from
  them. Structure is never hand-edited into existence in two places that must
  then be kept in sync.
- **Provenance is a graph, recorded as it happens.** Results are nodes produced
  by operations from inputs, forming a directed acyclic graph. Lineage is written
  when work is done, not reconstructed afterward by guesswork.
- **One access layer.** Application code reaches MFDB through a single
  transport-agnostic function API. That API is the contract; callers do not reach
  around it into raw SQL or a parallel object mapper.
- **Payloads are content-addressed.** Result artifacts are stored by the hash of
  their content, so identical results deduplicate and every reference is
  verifiable.
- **Access is checked at the boundary.** Authorization is decided in one place —
  the API — uniformly for every guarded entity, not scattered across the callers
  that happen to use it.
- **The schema is versioned and migratable.** A version number corresponds to an
  ordered chain of migrations, so any existing database can be upgraded
  deterministically.

## Target architecture

The data model is a provenance DAG over metadata:

```text
input artifact ─▶ operation ─▶ output artifact ─▶ operation ─▶ …
     (sample, measurement, parameters, result …)   (edges record how)
```

| Concept | Meaning |
|---------|---------|
| **Artifact** | A node: a sample, measurement, setup, parameter set, or result. Carries metadata; references (does not embed) bulk data. |
| **Operation** | A step that consumes input artifacts and produces output artifacts. |
| **Edge** | The recorded link between operations and artifacts — the lineage. |
| **Payload store** | Content-addressed storage for result blobs, referenced by artifacts. |
| **Vocabulary** | Controlled terms and entity types, sourced from the dictionaries. |

Layers, top to bottom:

- **API** — one transport-agnostic function surface; the sole entry point.
- **Domain model** — artifacts, operations, edges, lifecycle, and their rules.
- **Schema** — generated from the dictionaries; versioned with migrations.
- **Storage** — a relational store for metadata plus the content-addressed
  payload store for blobs.

State that lives here: metadata and lineage. State that does *not*: the live
analysis session (that is the service layer, [RPC & API](rpc.md)) and the
scientific objects themselves (that is [Core](core.md)). MFDB is imported as
`mfdb` and surfaced to
users through the metadata-admin plugin and through the RPC layer, never by
direct database access from the UI.

## Rules

1. The physical schema is generated from the dictionaries; canonical structure is
   NOT hand-edited. To change structure, change a dictionary and regenerate.
2. Every schema version corresponds to an ordered, replayable set of migrations;
   the version number is meaningful, not decorative.
3. Application code reaches MFDB only through the single function API — not raw
   SQL and not a competing object mapper.
4. Provenance is recorded as operations and edges at the time work is done; no
   result exists without recorded lineage.
5. Result payloads are stored by content hash and referenced by artifacts.
6. Authorization is enforced once, at the API boundary, uniformly for every
   guarded entity kind.
7. Writes that must be atomic (an operation with its artifacts and edges) happen
   in a single transaction — never a partial graph.

## Steering notes

Today MFDB is a historic mess: the core tables are hand-written (and duplicated
in two definitions that must be manually synced) while only some tables are truly
dictionary-generated; the version number is a stamp with no migration chain
behind it; three parallel access styles coexist in one oversized module;
authorization is enforced in only a handful of functions with ACL coverage for
few entity kinds; and some writes can leave a partial provenance graph. The
target is dictionary-generated structure end to end, a real migration chain, one
access layer, boundary-enforced auth, and atomic provenance writes. The backlog
is the `DATA-02`, `DATA-03`, `DATA-04`, `INC-03`, `INC-04`, and `INC-05` findings
in [assessment](assessment.md).
