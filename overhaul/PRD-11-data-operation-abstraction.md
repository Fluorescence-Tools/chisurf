# PRD-11: Transformers as Abstract Data-Operation Nodes in MFDB (chinet-style, dictionary-driven)

## Goal

Map **transformers** — data-manipulation steps realized as plugins (e.g. Burst
Selection, Microtime Shifter, background correction, correlation) — into MFDB as a
single **abstract data-operation node** abstraction, the data-side analog of a
chinet node, so every processing step is traceable end to end:

```text
sample (generated/defined)
  -> measurement operation   [setup + setup parameters]   -> raw dataset
  -> processing operation     [operation type + parameters] -> processed dataset
  -> processing operation     [operation type + parameters] -> ...
```

Every transformer registers the same way: **typed data inputs, typed data
outputs, and a parameter set whose schema is declared in `.dic`**. The
abstraction is uniform across all transformers; the per-transformer parameters
are dictionary-defined (typed, described, validated, admin-visible). A transformer
plugin is therefore just an `operation_type` with a declared parameter schema and
declared input/output artifact kinds — nothing transformer-specific leaks into the
MFDB schema.

## chinet ↔ MFDB correspondence (read first)

chinet (`modules/chinet/chinet/{node,port}.py`) models computation as a graph of
**Nodes** with typed input/output **Ports** and a callback that turns inputs into
outputs. MFDB already has the isomorphic data-side structures:

| chinet (fit-level abstraction) | MFDB (data-level) | Status |
|---|---|---|
| `Node` + callback / `evaluate()` | `mfdb_operation` (`operation_type`, `status`, `software_*`, `setup_id`) | exists |
| input / output **ports** | `mfdb_operation_artifact` (`direction` = `input`/`output`, `role`, `ordinal`) | exists |
| `Port` (value, bounds, fixed, error, units) | `mfdb_parameter` (`value`, `lower/upper_bound`, `parameter_type`, `standard_error`, CI, `units`) | exists |
| port links between nodes | `mfdb_edge` (`derived_from`) | exists |
| **typed port schema (value_type, name)** | — parameters are a free `name`/`value` bag | **missing** |

So the dataflow graph is already in MFDB. What is missing is the **typing of the
ports/parameters per operation type** — chinet ports are typed; MFDB operation
parameters are not declared anywhere.

> **Prior art (Orange3).** Orange independently arrives at the same model: typed,
> named I/O signals matched by type at connection time, plus derived data that
> carries a serializable *recompute rule* (`compute_value`). It confirms the two
> moves here — (1) **type the ports/parameters** per operation type, and (2) capture
> each operation as a **replayable spec** (PRD-21) so the node is reproducible, not
> just recorded. See `overhaul/ORANGE3-lessons.md`.

## The gaps

1. **No per-operation-type parameter schema in the `.dic`.** `mfdb_parameter`
   stores arbitrary `(name, value)` rows. There is no dictionary declaration of
   which parameters a given `operation_type` takes, their types, units,
   descriptions, defaults, or bounds. (Confirmed: no operation-parameter schema
   in `mfdb_flr_ext.dic`.) Parameters are therefore unvalidated, self-undescribed,
   and the GUI/admin cannot render them.
2. **The abstraction is not uniform.** Some plugins register operations with
   parameters (burst selection), but the microtime shifter stores per-channel
   shifts in a bespoke `mfdb_microtime_shift` table instead of the generic
   node/parameter model — two ways to record the same kind of thing.
3. **No operation-type registry tying `operation_type` → its parameter schema**,
   so there is no single place that says "a `burst_selection` operation consumes
   a raw_measurement and produces a burst_table, parameterised by min_photons,
   time_window, …".

## Design

### 1. The operation-node contract (uniform)

Every data manipulation registers through one path that records the node + ports
+ parameters. PRD-03 already provides most of it (`register_result` records an
operation, input/output `mfdb_operation_artifact` links, and `mfdb_parameter`
rows). Formalize it as the contract and make it the *only* way operations are
recorded:

```text
register_operation(
    operation_type,                 # the node's processing kind
    inputs:  [artifact_id, ...],    # input ports  (direction='input')
    outputs: [artifact_id, ...],    # output ports (direction='output')
    parameters: {name: value|{value,error,bounds,units,fixed}},
    setup_id=..., software=..., status=...,
)
```

- Inputs/outputs become `mfdb_operation_artifact` rows (the ports) and
  `derived_from` edges (output ← input).
- Parameters become `mfdb_parameter` rows, **validated against the operation
  type's `.dic` schema** (Section 2).
- No bespoke per-transformer tables: `mfdb_microtime_shift` is retired and its
  per-channel shifts become repeated `shift` parameter rows (role = channel). The
  authoritative record for every transformer is the operation + its parameter
  rows (Variable-arity decision below).

### 2. Operation-type parameter schemas in the `.dic` (the typed ports)

This is the core new piece. Declare, **in `mfdb_flr_ext.dic`**, the parameter
schema for each operation type — the data-side analog of chinet's typed ports.

Recommended representation (dictionary-driven, generated, gate-covered):

- A `.dic`-declared table **`mfdb_operation_parameter_def`**:
  `(operation_type, name, value_type, units, default, lower_bound, upper_bound,
   required, description)`, seeded from the dictionary (like `mfdb_vocabulary`),
  one row per (operation_type, parameter). The `operation_type` values reuse the
  existing extensible `operation_type` vocabulary.
- Each operation type's parameters are authored once in the `.dic` (name, type,
  units, bounds, description, and whether the parameter is **repeatable**), e.g.:
  - `microtime_shift`: `global_shift` (int, micro-time channels), plus a
    **repeatable** `shift` (int) — one row per detector channel.
  - `burst_selection`: `min_photons` (int), `time_window` (float, ms),
    `photon_window` (int), `count_rate_n_ph_max` (int), …
- Validation: when an operation registers, each `mfdb_parameter` row is checked
  against `mfdb_operation_parameter_def` for that `operation_type` — unknown
  parameters and missing required ones are rejected (mirrors
  `DictionarySchemaMap.validate_mapping` for setups). A total-coverage gate
  asserts every declared operation-parameter item maps to the table.

This makes operation parameters **typed, described, queryable, admin-renderable,
and UI-derivable** (the same `FieldSpec`-from-dictionary derivation used for
setups), while the operation model stays abstract.

### 3. Operation-type registry

Extend the `operation_type` vocabulary so each value carries (in the `.dic`):
its category (measurement vs processing vs analysis), expected input/output
artifact kinds, and a link to its parameter schema. This is the single source for
"what operations exist and what they consume/produce/parameterise" — drives
admin, validation, and future node-based workflow runners.

### 4. The traceable chain (what it buys)

With the above, the full lineage is one uniform graph of operation nodes:

- **sample** (`mfdb_sample`/`flr_sample`) → `measured_sample` edge.
- **measurement** = an operation (type `measurement_import`) with a `setup_id`
  and the setup's parameters → produces the raw dataset.
- **processing** = operations (type `microtime_shift`, `burst_selection`, …) with
  declared parameters → produce processed datasets, `derived_from` their inputs.
- Each step's parameters are dictionary-typed; each artifact is content-addressed;
  every edge is recorded. Re-running, auditing, and "what produced this and how"
  all fall out of the graph.

## Tasks

1. **`.dic` operation-parameter schemas + `role` column** — declare
   `mfdb_operation_parameter_def` category + per-operation-type parameter items
   (start with `microtime_shift`, `burst_selection`), including a `repeatable`
   flag; add a dictionary-declared `role` column to `mfdb_parameter` with
   `UNIQUE(operation_id, name, role)` (`SCHEMA_VERSION` bump, generated DDL).
   Generate/seed from the dictionary; add to the total-coverage gate.
2. **Validation** — validate `mfdb_parameter` rows against the declared schema in
   the registration path: reject unknown parameters and missing required ones;
   allow N rows only where the definition is `repeatable` (scalar → exactly one).
   Best-effort warn in GUI flows, strict in tests.
3. **Uniform `register_operation`** — formalize the operation-node contract over
   PRD-03; route transformer plugins through it. **Retire `mfdb_microtime_shift`**:
   migrate its rows to repeated `shift` parameter rows (role = channel), then drop
   reads of the old table. No transformer keeps a bespoke table.
4. **Operation-type registry** — operation_type vocabulary gains category +
   input/output kinds + parameter-schema link in the `.dic`.
5. **Admin/UI derivation** — render operation parameters from the dictionary
   (reuse the setup `FieldSpec` derivation); show the operation graph
   (inputs → operation → outputs) in mfdb-admin.
6. **Tests** — schema/validation gate; a worked chain (sample → measurement →
   microtime_shift → burst_selection) registers and is fully queryable; unknown
   parameter rejected; UI smoke.

## Variable-arity parameters — decision (locked): option (a), role-indexed rows

Per-channel microtime shifts, per-pair settings and similar variable-arity
parameters are recorded as **repeated `mfdb_parameter` rows distinguished by
`role`** (the data-side analog of chinet's multiple ports) — **not** as bespoke
per-transformer child tables.

- The shared `mfdb_parameter` table already has no uniqueness on `name`, so
  multiple rows per operation are allowed. Use the existing `metadata_json` /
  `mapping_json`, or the parameter `name` itself, to carry the role/index — e.g.
  one `shift` row per detector channel with the channel in a `role` field.
- Add a `role TEXT` column to `mfdb_parameter` (dictionary-declared, generated via
  the `.dic` path, `SCHEMA_VERSION` bump) so repeatable parameters have a typed
  index, with `UNIQUE(operation_id, name, role)`.
- The `.dic` parameter definition marks a parameter `repeatable` (and, where the
  index is meaningful, names the role domain, e.g. "detector_channel"). Validation
  allows N rows for a repeatable parameter and exactly one for a scalar.
- **`mfdb_microtime_shift` is retired**: per-channel shifts become repeated
  `shift` parameter rows (role = channel). No transformer gets its own table; the
  authoritative record for every transformer is the operation + its parameter
  rows. This keeps the model uniform — adding a new transformer never adds a
  table, only `.dic` parameter definitions.

## Definition of Clean (carried over)

- `.dic` dictates the schema: operation-parameter definitions are dictionary-
  declared, generated, gate-covered — no hardcoded SQL, no blob.
- One uniform operation-node path; no per-operation bespoke tables as the
  authoritative store.
- Validation rejects undeclared parameters (typed ports), surfaced not swallowed
  on real errors; best-effort only for MFDB-unavailable.
- Behavior-asserting tests (a real chain registers and is queryable), DI over
  monkeypatching, GUI smoke for any admin view.

## Relationship to other PRDs

- **PRD-16** (Transformer Contract) is the plugin-side counterpart: PRD-11 is the
  MFDB data model (operation nodes + `.dic` parameter schemas); PRD-16 is the
  uniform contract every transformer plugin must obey to register into it. They
  ship together.
- Builds on PRD-03 (result registry: operations, operation_artifact, parameters).
- Subsumes the parameter side of PRD-04 burst registration and PRD-09 microtime
  shift (their parameters become `.dic`-declared operation parameters).
- Complements PRD-11's sibling **LIMS diagnosis** (`MFDB-LIMS-diagnosis.md`):
  this is the "protocol/operation" formalization (P3) plus the provenance spine
  that the lifecycle/state-machine work (P1) hangs off.
