---
type: PRD
prd: "11"
title: "PRD-11: Transformers as Abstract Data-Operation Nodes"
description: Model every data-manipulation step as a uniform MFDB operation node with typed, dictionary-declared parameters and input/output ports.
status: done
phase: "2"
resource: chisurf/core/mfdb
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Data-manipulation steps realized as plugins (burst selection, microtime shifter, background correction, correlation) are mapped into MFDB as a single abstract data-operation node — the data-side analog of a chinet computation node — so every processing step is traceable end to end. Each transformer registers uniformly: typed data inputs and outputs (operation-artifact ports and `derived_from` edges) plus a parameter set whose schema is declared in the `.dic` dictionary (`mfdb_operation_parameter_def`) and validated on registration. The key new piece is per-operation-type parameter typing (name, type, units, bounds, description, repeatable flag), an operation-type registry linking types to their input/output kinds and schemas, and role-indexed parameter rows for variable-arity parameters — retiring bespoke per-transformer tables such as the microtime-shift table. A visual node/workflow toolkit is cited as prior art for typed I/O signals and replayable recompute rules.

# Status
Done. The uniform operation-node contract, dictionary-declared parameter schemas with validation, role-indexed parameters, and the operation-type registry are in place.

# Goal
Map **transformers** — data-manipulation steps realized as plugins (Burst
Selection, Microtime Shifter, background correction, correlation, …) — into MFDB as
a single **abstract data-operation node**, the data-side analog of a chinet node,
so every processing step is traceable end to end:

```text
sample (generated/defined)
  -> measurement operation   [setup + setup parameters]     -> raw dataset
  -> processing operation    [operation type + parameters]  -> processed dataset
  -> processing operation    [operation type + parameters]  -> ...
```

Every transformer registers the same way: typed data inputs, typed data outputs,
and a parameter set whose schema is declared in the `.dic`. The abstraction is
uniform; per-transformer parameters are dictionary-defined (typed, described,
validated, admin-visible). A transformer plugin is therefore just an
`operation_type` with a declared parameter schema and declared input/output
artifact kinds — nothing transformer-specific leaks into the MFDB schema.

# chinet ↔ MFDB correspondence
chinet (`modules/chinet/chinet/{node,port}.py`) models computation as a graph of
**Nodes** with typed input/output **Ports** and a callback turning inputs into
outputs. MFDB already has the isomorphic data-side structures:

| chinet (fit-level) | MFDB (data-level) | Status |
|---|---|---|
| `Node` + callback / `evaluate()` | `mfdb_operation` (`operation_type`, `status`, `software_*`, `setup_id`) | exists |
| input / output ports | `mfdb_operation_artifact` (`direction`, `role`, `ordinal`) | exists |
| `Port` (value, bounds, fixed, error, units) | `mfdb_parameter` (`value`, bounds, `parameter_type`, `standard_error`, CI, `units`) | exists |
| port links between nodes | `mfdb_edge` (`derived_from`) | exists |
| **typed port schema (value_type, name)** | — parameters are a free `name`/`value` bag | **missing** |

The dataflow graph is already in MFDB; what is missing is the **typing of the
ports/parameters per operation type**. A visual node/workflow data-analysis toolkit
cited as prior art independently arrives at the same model — typed, named I/O
signals matched by type at connection time, plus derived data carrying a
serializable *recompute rule* — confirming the two moves here: (1) type the
ports/parameters per operation type, and (2) capture each operation as a replayable
spec (owned by the replayable-compute-spec PRD) so the node is reproducible.

# The gaps
1. **No per-operation-type parameter schema in the `.dic`.** `mfdb_parameter`
   stores arbitrary `(name, value)` rows; there is no dictionary declaration of
   which parameters an `operation_type` takes, their types, units, descriptions,
   defaults, or bounds. Parameters are unvalidated, self-undescribed, and the
   GUI/admin cannot render them.
2. **The abstraction is not uniform.** Some plugins register operations with
   parameters (burst selection), but the microtime shifter stores per-channel
   shifts in a bespoke `mfdb_microtime_shift` table — two ways to record the same
   kind of thing.
3. **No operation-type registry** tying `operation_type` → its parameter schema, so
   nothing says "a `burst_selection` operation consumes a raw_measurement and
   produces a burst_table, parameterised by min_photons, time_window, …".

# Design
## 1. The operation-node contract (uniform)
Every data manipulation registers through one path recording node + ports +
parameters. The result registry already provides most of it (`register_result`
records an operation, input/output `mfdb_operation_artifact` links, and
`mfdb_parameter` rows). Formalize it as the *only* way operations are recorded:

```text
register_operation(
    operation_type,                 # the node's processing kind
    inputs:  [artifact_id, ...],    # input ports  (direction='input')
    outputs: [artifact_id, ...],    # output ports (direction='output')
    parameters: {name: value|{value,error,bounds,units,fixed}},
    setup_id=..., software=..., status=...,
)
```

Inputs/outputs become `mfdb_operation_artifact` rows (ports) and `derived_from`
edges (output ← input). Parameters become `mfdb_parameter` rows validated against
the operation type's `.dic` schema. No bespoke per-transformer tables:
`mfdb_microtime_shift` is retired and its per-channel shifts become repeated
`shift` parameter rows (role = channel).

## 2. Operation-type parameter schemas in the `.dic` (the typed ports)
The core new piece. Declare, in `mfdb_flr_ext.dic`, the parameter schema for each
operation type — the data-side analog of chinet's typed ports. Recommended: a
`.dic`-declared table **`mfdb_operation_parameter_def`** `(operation_type, name,
value_type, units, default, lower_bound, upper_bound, required, description)`,
seeded from the dictionary (like `mfdb_vocabulary`), one row per (operation_type,
parameter); `operation_type` reuses the existing extensible vocabulary. Each type's
parameters are authored once (name, type, units, bounds, description, and whether
**repeatable**), e.g. `microtime_shift`: `global_shift` (int, micro-time channels)
plus a repeatable `shift` (int, one per detector channel); `burst_selection`:
`min_photons` (int), `time_window` (float, ms), `photon_window` (int),
`count_rate_n_ph_max` (int), … On registration each `mfdb_parameter` row is checked
against `mfdb_operation_parameter_def` for that type — unknown parameters and
missing required ones are rejected; a total-coverage gate asserts every declared
operation-parameter item maps to the table. This makes operation parameters typed,
described, queryable, admin-renderable, and UI-derivable (the same
`FieldSpec`-from-dictionary derivation used for setups) while the operation model
stays abstract.

## 3. Operation-type registry
Extend the `operation_type` vocabulary so each value carries (in the `.dic`) its
category (measurement vs processing vs analysis), expected input/output artifact
kinds, and a link to its parameter schema — the single source for "what operations
exist and what they consume/produce/parameterise", driving admin, validation, and
future node-based workflow runners.

## 4. The traceable chain
With the above, full lineage is one uniform graph: **sample** → `measured_sample`
edge; **measurement** = an operation (`measurement_import`) with a `setup_id` and
the setup's parameters → raw dataset; **processing** = operations (`microtime_shift`,
`burst_selection`, …) with declared parameters → processed datasets, `derived_from`
their inputs. Each step's parameters are dictionary-typed, each artifact is
content-addressed, every edge recorded — re-running, auditing, and "what produced
this and how" all fall out of the graph.

# Variable-arity parameters — decision (locked): role-indexed rows
Per-channel microtime shifts, per-pair settings, and similar variable-arity
parameters are recorded as **repeated `mfdb_parameter` rows distinguished by
`role`** (the data-side analog of chinet's multiple ports), **not** bespoke child
tables. The shared `mfdb_parameter` table has no uniqueness on `name`, so multiple
rows per operation are allowed. Add a dictionary-declared `role TEXT` column
(generated via the `.dic` path, `SCHEMA_VERSION` bump) with
`UNIQUE(operation_id, name, role)` so repeatable parameters have a typed index. The
`.dic` definition marks a parameter `repeatable` (and, where meaningful, names the
role domain, e.g. "detector_channel"); validation allows N rows for a repeatable
parameter and exactly one for a scalar. `mfdb_microtime_shift` is retired: its
per-channel shifts become repeated `shift` rows (role = channel). Adding a new
transformer never adds a table — only `.dic` parameter definitions.

# Tasks
1. **`.dic` operation-parameter schemas + `role` column** — declare
   `mfdb_operation_parameter_def` + per-operation-type parameter items (start with
   `microtime_shift`, `burst_selection`) including a `repeatable` flag; add the
   `role` column to `mfdb_parameter` with `UNIQUE(operation_id, name, role)`
   (`SCHEMA_VERSION` bump, generated DDL); seed from the dictionary; add to the
   total-coverage gate.
2. **Validation** — validate `mfdb_parameter` rows against the declared schema in
   the registration path: reject unknown and missing-required parameters; allow N
   rows only where `repeatable`. Best-effort warn in GUI flows, strict in tests.
3. **Uniform `register_operation`** — formalize the operation-node contract over
   the result registry; route transformer plugins through it. Retire
   `mfdb_microtime_shift`: migrate its rows to repeated `shift` parameter rows,
   then drop reads of the old table.
4. **Operation-type registry** — `operation_type` gains category + input/output
   kinds + parameter-schema link in the `.dic`.
5. **Admin/UI derivation** — render operation parameters from the dictionary (reuse
   the setup `FieldSpec` derivation); show the operation graph (inputs → operation →
   outputs) in mfdb-admin.
6. **Tests** — schema/validation gate; a worked chain (sample → measurement →
   microtime_shift → burst_selection) registers and is fully queryable; unknown
   parameter rejected; UI smoke.

# Definition of Clean
- `.dic` dictates the schema: operation-parameter definitions are dictionary-
  declared, generated, gate-covered — no hardcoded SQL, no blob.
- One uniform operation-node path; no per-operation bespoke tables as the
  authoritative store.
- Validation rejects undeclared parameters (typed ports), surfaced not swallowed on
  real errors; best-effort only for MFDB-unavailable.
- Behavior-asserting tests (a real chain registers and is queryable), DI over
  monkeypatching, GUI smoke for any admin view.

# Relationships
- Ships together with [PRD-16](prd-16.md), its plugin-side transformer-contract counterpart; together they supersede [PRD-07](prd-07.md).
- Reused by [PRD-14](prd-14.md) (protocols reference `operation_type` parameter schemas).
- Provides the provenance spine the lifecycle work ([PRD-12](prd-12.md)) hangs off.
- Subsumes the parameter side of the burst and microtime-shift ([PRD-09](prd-09.md)) work.
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).
