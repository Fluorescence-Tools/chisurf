# PRD-14: Protocol Entity — Named, Versioned Procedures (LIMS P3)

## Goal

Add a **protocol** abstraction: a named, versioned procedure (measurement or
processing) with a declared parameter schema. Operations reference the protocol +
version they ran, formalizing reproducibility ("re-run protocol X v3 on sample
Y"). This generalizes the setup/calibration work and pairs with the operation-node
abstraction (PRD-11).

## Background

- LIMS reference: iSkyLIMS `ProtocolType` → `Protocols` → `ProtocolParameters`;
  values recorded per execution. A protocol names a procedure and defines the
  parameters it requires.
- MFDB today: setups describe instrument config (dictionary-declared, versioned
  calibration via `mfdb_setup_calibration`), and operations carry parameters, but
  there is no first-class "protocol" — no named, versioned procedure with a
  declared parameter schema that operations reference.
- See `overhaul/MFDB-LIMS-diagnosis.md` (P3) and `PRD-11` (operation nodes).

## Design (dictionary-driven; generalizes setup + operation-param schema)

- **`mfdb_protocol`** (`.dic`-declared, generated, gate-covered):
  `(protocol_id PK, name, version INTEGER DEFAULT 1, category, description,
   operation_type, setup_id FK mfdb_setup, created_by_user_id, is_public,
   created_at, updated_at, deleted_at)`, `UNIQUE(name, version)`.
  `category` ∈ {`measurement`, `processing`, `analysis`} (extensible vocab).
  `operation_type` links the protocol to the operation kind it realizes (PRD-11),
  and `setup_id` to the instrument config for measurement protocols.
- **Parameter schema:** reuse PRD-11's `mfdb_operation_parameter_def` keyed by the
  protocol's `operation_type` (or a `mfdb_protocol_parameter_def(protocol_id,
  name, value_type, units, default, lower/upper_bound, required, repeatable,
  description)` if a protocol pins parameters/defaults beyond the operation type).
  Dictionary-declared, generated, gate-covered.
- **Operations reference the protocol:** add `protocol_id` + `protocol_version`
  (dictionary-declared columns) to `mfdb_operation`, so every recorded operation
  names the exact procedure + version it executed. The operation's
  `mfdb_parameter` rows are validated against the protocol/operation parameter
  schema (PRD-11 validation).
- **Versioning:** a protocol is append-only by version (like calibration
  snapshots) — editing a protocol creates a new version; operations keep the
  version they ran. This makes "what changed between runs" and "reproduce exactly"
  first-class.

## API

- `create_protocol(name, category, operation_type, parameters_schema=...,
  setup_id=None, is_public=False)` → (protocol_id, version); a new version on edit.
- `get_protocol(protocol_id, version="latest")`, `list_protocols(scope=...)`.
- `register_operation(..., protocol_id=..., protocol_version=...)` (extends the
  PRD-11 contract) records the protocol reference + validates parameters against
  its schema.

## Tasks

1. `.dic` + schema: declare `mfdb_protocol` (+ optional
   `mfdb_protocol_parameter_def`); add `protocol_id`/`protocol_version` columns to
   `mfdb_operation`; generate DDL; `SCHEMA_VERSION` bump; add to the gate.
2. Repository: protocol CRUD with versioning (append-only), scoping (own+public),
   parameter-schema resolution.
3. Wire `register_operation`/registration to accept and record the protocol
   reference and validate parameters against it.
4. mfdb-admin: a Protocols entity (dictionary-sourced columns) with version
   history + parameter schema; show, on an operation, which protocol/version ran.
5. Tests: protocol create/version (append-only); operation records protocol ref;
   parameters validated against the protocol schema (unknown rejected); reproduce
   a run by protocol+version; dict gate green; GUI smoke.

## Definition of Done

- [ ] `mfdb_protocol` exists (dict-declared, generated, gate-covered), versioned
      append-only, scoped own+public, linked to operation_type/setup.
- [ ] Operations reference `protocol_id` + `protocol_version`; their parameters
      validate against the protocol/operation parameter schema.
- [ ] Admin shows protocols + versions and per-operation protocol provenance.
- [ ] Tests pass including versioning and parameter validation.

## Definition of Clean

`.dic` dictates the schema (no hardcoded SQL/blob); reuse PRD-11 operation-param
schemas and the own+public/versioning patterns (no forked stacks); validation
surfaced not swallowed on real errors; behavior-asserting tests; DI over
monkeypatching; GUI smoke.

## Relationship

Builds on PRD-11 (operation nodes + parameter schemas) and the PRD-04 setup/
calibration work; provides the "how was it measured/processed" record that the
PRD-12 lifecycle and PRD-13 study layers reference.
