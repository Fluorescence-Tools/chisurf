---
type: PRD
prd: "16"
title: "PRD-16: General Transformer Contract"
description: Defines one uniform contract every data-transformer plugin must obey — typed ports, dictionary-declared parameters, a pure transform, and uniform provenance registration.
status: done
phase: "2"
resource: chisurf/core/transform
tags: [prd, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-16 defines a single transformer abstraction that every data-transformer plugin (burst selection, microtime shifter, background correction, correlation, and others) must satisfy, replacing ad-hoc per-plugin structure. A conformant transformer declares typed input and output ports, reads its parameter schema from the `.dic` dictionary rather than hardcoding it, exposes a pure `transform` function free of Qt and database imports, and registers uniformly as an operation node. Invocations are serializable spec values, which makes them recordable, replayable, and composable. A registry enables discovery, and a parametrised conformance test gates any new transformer.

# Status
Done (phase 2, STATUS TABLE authoritative). The reference transformers (burst selection, microtime shifter) conform, and the conformance test gates new transformers.

# Goal
Define **one transformer abstraction** that every data-transformer plugin (Burst
Selection, Microtime Shifter, background correction, correlation, …) must obey, so
they stop being individual ad-hoc pieces and behave identically: declared typed
**inputs**, declared typed **outputs**, a `.dic`-declared **parameter schema**, a
**pure transform**, and **uniform MFDB registration** as an operation node. A
transformer becomes a conformant, discoverable unit — not a bespoke plugin. This is
the plugin-side counterpart of PRD-11 (which maps transformers to MFDB operation
nodes): PRD-11 defines *how the data graph records a transformer*; PRD-16 defines
*what a transformer is and the rules it must follow*.

# Problem
The workflow plugins emerged by copying Burst Selection's layout, but nothing
**enforces** it. Each re-invents request/result models, parameter handling, MFDB
registration, RPC wiring, and error behaviour. Consequences: inconsistent parameter
recording (some use `mfdb_parameter`, the shifter used a bespoke table),
inconsistent ownership/validation, no discovery, no way to compose them, and
per-plugin bugs (the shifter MFDB round-trip failures). There is no single contract
a transformer must satisfy.

# The transformer contract
A transformer is an `operation_type` (PRD-11) plus a small, **mandatory** Python
contract — a base/protocol in core, e.g. `chisurf/core/transform/transformer.py`:

```python
class Transformer(Protocol):
    transformer_id: str          # stable id (e.g. "microtime_shifter")
    operation_type: str          # PRD-11 operation vocabulary value
    version: str                 # contract/impl version

    input_spec:  list[PortSpec]  # typed input ports  (kinds/formats, arity)
    output_spec: list[PortSpec]  # typed output ports (kinds)
    # parameter schema is NOT defined in code — it is read from the .dic
    # (PRD-11 mfdb_operation_parameter_def for this operation_type)

    def transform(self, inputs: TransformInputs,
                  parameters: dict) -> TransformResult: ...   # PURE: no Qt, no DB
```

A visual node/workflow data-analysis toolkit cited as prior art declares typed,
named I/O signals and validates a connection only if the output type matches the
input port, with a manager routing values reactively — the same shape as `PortSpec`
(and chinet's typed ports). Adopt its rules: declared, named, typed ports with
arity; validate wiring by kind at the boundary; reject mismatches. It also treats a
transformer as a first-class serializable value that can be an output, composed,
stored, and replayed (see rule 7 and the pipeline-composition PRD).

Rules every transformer must obey:
1. **Typed ports.** `input_spec`/`output_spec` declare artifact kinds + formats and
   arity (the data-side ports). The contract validates that what a caller passes
   matches the declared ports.
2. **Parameters are `.dic`-declared, not code-defined.** A transformer's parameters
   come from PRD-11's `mfdb_operation_parameter_def` for its `operation_type`
   (typed, units, bounds, required, repeatable). The transformer does not hardcode a
   parameter list; it reads/validates against the dictionary. No JSON blob.
3. **Pure `transform`.** Computation is a pure function over inputs + parameters →
   outputs, with no Qt and no DB imports — directly unit-testable (the Burst/Shift
   `api/*.py` pattern, mandated).
4. **Uniform MFDB registration.** Persistence goes through the single PRD-11
   `register_operation(operation_type, inputs, outputs, parameters)` path (built on
   the result registry, which stamps ownership and **raises on real errors**). A
   transformer never hand-writes MFDB rows and never keeps a bespoke table.
5. **Best-effort archival, fail-loud on bugs.** MFDB-unavailable → warn, the
   transform still works; a real registration error (FK/vocab/validation) is
   surfaced, not swallowed.
6. **Standard layering.** `api/` (models, contract, pure transform, mfdb),
   `backend/services.py` (RPC handlers on the dispatcher), `cli/`, `gui/`
   (RPC-client + dockable tool). The GUI talks to the API only via RPC.
7. **Transformer invocation is a serializable spec value.** A bound invocation —
   `(transformer_id/operation_type, parameters, input artifact ids)` — is a plain
   serializable object, separable from execution. This is what `register_operation`
   records, what the replayable-compute-spec PRD stores as an artifact's replayable
   spec, and what the pipeline-composition PRD composes into pipelines. A
   transformer never needs a bespoke serialization; the spec is its `.dic`-typed
   parameters + ports.

# Transformer registry / discovery
Transformers self-register so the system can enumerate them: a
`register_transformer(transformer)` / `TRANSFORMER_REGISTRY` (loaded from plugin
manifests) exposing `transformer_id`, `operation_type`, `input_spec`,
`output_spec`, and the parameter-schema reference. Enables a uniform "available
transformers" list (with I/O kinds and dictionary parameters), validation that each
declared `operation_type` has a `.dic` parameter schema, and a foundation for
node-based composition (chain transformers: output kinds of one feed input kinds of
the next — the chinet-style data-level graph).

# Conformance
- **Refactor the reference transformers** (Burst Selection, Microtime Shifter) onto
  the contract; fold the shifter's `mfdb_microtime_shift` into role-indexed
  parameters (PRD-11).
- **Conformance test** parametrised over the registry: every transformer (1) has a
  `.dic` parameter schema for its `operation_type`; (2) declares input/output
  ports; (3) exposes a pure `transform` with no Qt/DB imports; (4) registers via the
  uniform path (a synthetic run produces an operation with the right input/output
  links + parameter rows). A new transformer cannot merge without passing it.

# Tasks
1. Define `Transformer` protocol + `PortSpec`/`TransformInputs`/`TransformResult`
   in `chisurf/core/transform/`.
2. Implement `register_operation` (PRD-11) and route the contract's registration
   through it.
3. Transformer registry + manifest loading; validation that each transformer's
   `operation_type` has a `.dic` parameter schema.
4. Refactor Burst Selection + Microtime Shifter to conform (pure transform, typed
   ports, dictionary parameters, uniform registration); retire bespoke tables.
5. Conformance test parametrised over the registry; per-transformer behaviour tests
   keep passing.
6. Docs: a short "how to write a transformer" guide pointing at the contract.

# Definition of Done
- A single `Transformer` contract exists; transformers declare typed ports and read
  their parameter schema from the `.dic` (no code-defined params, no blob).
- All transformers register uniformly via PRD-11 `register_operation`; no bespoke
  per-transformer tables; ownership/validation/error behaviour uniform.
- A registry enumerates transformers (id, operation_type, I/O kinds, params).
- Burst Selection + Microtime Shifter conform; the conformance test passes and gates
  new transformers.

# Definition of Clean
Layer purity (pure transform, GUI→RPC only); `.dic` dictates parameters (PRD-11), no
blobs, no hardcoded SQL; uniform registration (no bespoke tables); best-effort MFDB
but fail-loud on real bugs; DI over monkeypatching; behavior-asserting + conformance
tests; GUI construction smoke for each tool.

# Relationships
- Plugin-side counterpart of [PRD-11](prd-11.md), which maps transformers to MFDB operation nodes and defines the parameter schema storage.
- Reference transformers derive from [PRD-04](prd-04.md) and [PRD-09](prd-09.md).
- A transformer execution may reference a protocol/version per [PRD-14](prd-14.md).
- Serializable invocations feed [PRD-21](prd-21.md) (replayable compute spec) and [PRD-22](prd-22.md) (pipeline composition).
- GUI half enforced by [PRD-23](prd-23.md).
- Aligns with the [Plugins target](/specs/plugins.md) and [Core target](/specs/core.md); registration routes through the [action layer](/architecture/action-layer.md).
