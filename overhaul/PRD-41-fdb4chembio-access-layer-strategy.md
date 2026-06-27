# PRD-41: FDB4ChemBio Access-Layer & Interoperability Strategy

> **Type:** Design note / strategy (not an implementation PRD). It sets the
> architectural boundary that downstream implementation PRDs must respect, and
> records the GraphQL decision so it is not re-litigated.

## Context

MFDB (`chisurf/core/mfdb/`) is today an in-process, SQLite-backed store
(`SCHEMA_VERSION = 39`, ~40 tables) consumed by ChiSurf plugins via a flat
function API (`api.py`) and remotely via the ZeroMQ RPC server
(`chisurf/server/`, `ChisurfClient`). Its canonical tables are **generated from
mmCIF dictionaries** (`schema_from_dictionary.py` → `CREATE TABLE`/`ALTER` from
`data/*.dic`), which descend from the wwPDB / PDB-IHM family:
`mmcif_std`, `pdbx_v50`, `ihm_ext`, `ihm_flr_ext`, plus our own `mfdb_flr_ext`.

**MFDB is a prototype for FDB4ChemBio** — intended to become a central,
worldwide public resource for fluorescence data. This note answers: *what is
the durable core of that resource, and what role (if any) does GraphQL play?*

A question came up — "could GraphQL be useful, since we use `.dic` files to pin
down the DB?" The two are orthogonal (GraphQL is a query/transport facade; the
`.dic` files are storage semantics), but the question forces the right strategic
framing, captured below.

## Principle 1 — The dictionary is the product

For a worldwide resource, the durable, ownable asset is the **dictionary-conformant
data model**, not any database engine or API:

- The wwPDB ecosystem's standard *is* the PDBx/mmCIF dictionary. Multiple
  independent implementations (RCSB, PDBe, PDBj) consume the same dictionary.
  Our `mfdb_flr_ext.dic` / `ihm_flr_ext.dic` is the candidate to become "the
  worldwide fluorescence standard."
- The canonical **archived artifact is the dictionary-conformant CIF file**, not
  a database row. SQLite today (probably Postgres + object storage later) is a
  swappable serving detail underneath.

**Implication:** the highest-leverage investment now is the dictionary, its
validation, and full CIF import/export round-trip — *not* picking an API
technology. PRD-39 (`struct_ref` cross-refs to UniProt/PDB) and PRD-02a (mmCIF
dictionary infrastructure) are on the critical path; an API technology choice is
not.

## Principle 2 — Two clocks: prototype vs. resource

| | Prototype (now) | Worldwide resource (later) |
|---|---|---|
| Store | SQLite, single-node | Postgres + object store, read replicas |
| Access | in-process `api.py` + ZMQ RPC | public dissemination APIs + deposition pipeline |
| Contract | `.dic` drives DDL | `.dic` drives DDL **and** generated read APIs |
| Artifact | DB rows + CIF export | dictionary-conformant CIF is canonical |

Do not build the resource-tier API now. Do make the prototype *not block it*.

## Principle 3 — Deposition and dissemination are different problems

- **Deposition (writes):** authenticated, transactional, dictionary-validated,
  permissioned. This is the existing `register_artifact` / `record_operation` /
  ACL path (PRD-37 hardens its transport/authz). A worldwide write path looks
  like wwPDB's OneDep — a validation pipeline, **not** a generic mutation API.
- **Dissemination (reads):** public, cacheable, federatable, many heterogeneous
  clients. This is where flexible query technologies (incl. GraphQL) earn their
  keep.

**Keep GraphQL (and any generic mutation API) out of deposition.** It would only
reshape the validated write path and tempt dilution of validation.

## The GraphQL decision

**Decision: GraphQL is approved as a *future*, *read-only*, *dictionary-generated*
dissemination endpoint — one interface among several — and is explicitly *not*
built during the prototype phase and *not* the central component.**

Rationale:

- **Ecosystem-conformant, not exotic.** RCSB PDB ships a public GraphQL API
  alongside REST, search, and bulk download, generated from the same mmCIF data
  model. Our dictionaries are from that family, so a dictionary-generated GraphQL
  read endpoint matches client expectations and aids federation.
- **Our data is graph-shaped.** The provenance DAG
  (`mfdb_operation → mfdb_artifact → mfdb_edge`; `graph_upstream`/`downstream`/
  `traverse_canonical_graph`) plus `struct_ref` cross-refs are exactly the
  nested, federatable reads GraphQL serves well.
- **But it is one interface among several**, never the foundation:

| Need | Right tool | GraphQL fit |
|---|---|---|
| Bulk download / archival mirror | CIF files over FTP/S3 | ✗ wrong tool |
| Search / discovery | dedicated search index/API | ⚠ poor |
| Flexible nested reads, federation | **GraphQL** (dictionary-generated) | ✓ strong |
| Simple record fetch by ID | REST | ✓ also fine |
| Deposition (writes) | validated pipeline + auth (`api.py`/RPC/ACL) | ✗ keep out |

When GraphQL is built, generate its SDL **from the dictionary** (same source that
already drives DDL via `schema_from_dictionary.py`) so the query schema cannot
drift from storage. See appendix.

## What the prototype must do now (to not block the resource)

1. **Do not add GraphQL.** Nothing in the prototype needs it.
2. **Keep the access layer transport-agnostic.** `api.py` stays a clean domain
   layer with no ZMQ/HTTP/GraphQL assumptions, so GraphQL *and* REST *and* bulk
   export can be added later without a rewrite. Preserve the current discipline:
   `mfdb_admin` calls `api.py` in-process; `ChisurfClient` calls the same surface
   over ZMQ. "The API is just functions."
3. **Treat `mfdb_flr_ext.dic` as a published spec:** version it, validate deposits
   against it, guarantee CIF export/import round-trips (ties to PRD-02a / PRD-39).
4. **Do not conflate the MFDB SQLite schema with the resource's serving store.**
   A worldwide read tier outgrows SQLite; dissemination APIs sit above whatever
   store we migrate to. Keep that boundary clean now.

## Non-goals

- Choosing the resource-tier serving store (Postgres vs. other) — deferred.
- Building REST or GraphQL endpoints in this phase.
- Designing the public deposition/validation pipeline (its own future PRD,
  analogous to OneDep).

## Appendix — dictionary → GraphQL SDL (sketch, for when we build it)

The mmCIF dictionary already encodes everything a GraphQL schema needs, so the
read API should be generated, not hand-written:

| mmCIF dictionary concept | GraphQL SDL |
|---|---|
| category (e.g. `flr_experiment`) | `type FlrExperiment { … }` |
| item / column + type | field + scalar (`String`/`Float`/`Int`) |
| `key_item` (PK) | `id: ID!` |
| parent–child item link (`_item_linked`) | edge field → related `type` |
| controlled vocabulary (enumeration) | GraphQL `enum` |
| provenance edges (`mfdb_edge`) | `upstream`/`downstream` connection fields |

This reuses the same `MmcifDictionary` model that `schema_from_dictionary.py`
already walks for DDL; a parallel emitter produces SDL + resolvers backed by
`api.py`. Read-only; writes remain on the deposition path.

## Status

- [x] Strategy recorded; GraphQL decision made (future read-only, dictionary-generated).
- [ ] Prototype guardrails (transport-agnostic `api.py`, dictionary-as-spec) tracked
      under their existing PRDs (PRD-02a, PRD-37, PRD-39).
