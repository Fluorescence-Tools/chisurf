# PRD-22: Workflow / Pipeline Engine on the Transformer Contract (Architecture E)

## Goal

Let users **compose transformers into pipelines** — a node-based dataflow graph at
the data level (the chinet model applied to data operations) — that executes and is
recorded in MFDB as a chain of operations. Reproducible, shareable analysis
workflows.

## Background

Once PRD-11 (operation nodes) + PRD-16 (transformer contract) land, every
transformer declares typed input/output artifact kinds and a `.dic` parameter
schema. That is exactly what a dataflow engine needs to wire nodes: an output kind
of one transformer can feed an input kind of the next. chinet already does this at
the *fit* level (nodes/ports); this does it at the *data* level.

> **Prior art (Orange3).** Orange's canvas *is* this: a node graph of transformers
> wired by typed signals, validated by type at connection time, and **serialized to a
> shareable workflow document** (`.ows`). Two lessons to adopt (see
> `overhaul/ORANGE3-lessons.md`): (1) a pipeline is a **saveable/shareable document**,
> not just a runtime object; (2) each node is a serializable transformer-invocation
> *value* (PRD-16 rule 7), so the whole graph is data you can store, diff, and replay.
> Tie a saved pipeline to a **PRD-27 branch** so a shared workflow is reproducible
> against recorded data and "what-if" variants are branch operations.

## Design

- **Pipeline = ordered/graph of transformer invocations.** A
  `mfdb_pipeline(pipeline_id, name, version, definition_json|structured, owner,
  is_public, …)` declares nodes (transformer + bound parameters) and edges
  (output port → input port). Dictionary-declared where structured.
- **Type-checked composition:** an edge is valid only if the producer's output
  kind/format matches the consumer's declared input port (PRD-16 `input_spec`).
  Validate at definition time.
- **Execution:** a runner topologically evaluates the graph, calling each
  transformer's pure `transform` (PRD-16) and registering each step via
  `register_operation` (PRD-11) — so a pipeline run is a recorded chain of
  operations with full provenance; re-running is reproducible.
- **Reuse the lineage/event model (PRD-21):** the runner can be event-driven
  (a node fires when its inputs are available) and the resulting graph is queryable
  via the lineage API.
- **GUI (later):** a node-based workflow editor; out of scope for the first cut
  (headless/scripted pipelines first).

## Tasks

1. Pipeline definition model (`.dic`-declared where structured) + type-checked
   composition against transformer port specs.
2. A headless runner: topological execution, per-node `register_operation`, status
   tracking (reuse the operation status / PRD-12 lifecycle).
3. Persist pipeline definitions (own + public) and pipeline *runs* (link the chain
   of operations to a `pipeline_run` for grouping).
4. Tests: a 2–3 node pipeline (raw → microtime_shift → burst_selection) validates,
   executes, and produces a recorded, queryable operation chain; an invalid edge
   (kind mismatch) is rejected.
5. (Deferred) node-based GUI editor.

## Definition of Done

- [ ] Pipelines compose conformant transformers with type-checked edges.
- [ ] A headless runner executes a pipeline and records each step as an operation
      (full provenance); runs are reproducible and queryable.
- [ ] Invalid compositions are rejected at definition time.

## Definition of Clean

Built only on the PRD-16 contract + PRD-11 registration (no transformer-specific
glue); `.dic`-declared definitions, no blob for structured parts; behavior-asserting
tests over a real pipeline; fail-loud on invalid composition.

## Relationship

The endgame of PRD-11/16. Consumes PRD-21 (lineage/events) and PRD-12 (run status).
This is the chisurf-side analog of chinet's node graph, at the data-operation
level.
