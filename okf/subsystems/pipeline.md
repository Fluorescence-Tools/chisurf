---
type: Subsystem
title: Pipelines
description: Typed DAGs of transformer invocations, validated before execution and persisted through MFDB provenance.
resource: chisurf/core/pipeline/
tags: [core, pipeline, provenance, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Pipeline model

`chisurf/core/pipeline/` defines headless, typed data-processing pipelines. A
pipeline is a directed acyclic graph of transformer invocations:

| Object | Meaning |
| --- | --- |
| `PipelineNode` | One operation type plus bound parameters. |
| `PipelineEdge` | A typed wire from one node output port to another node input port. |
| `Pipeline` | Named, versioned graph of nodes and edges. |

`validate_pipeline` resolves every node's `operation_type` through the
transformer registry, checks that edges reference declared ports, verifies
producer/consumer port-kind compatibility, and rejects cycles before anything
runs.

# Execution

`runner.run_pipeline` topologically evaluates a validated pipeline. Each node is
converted to an MFDB `ComputeSpec` and executed by the registered replay
executor for its operation type. The result is a `PipelineRun` containing node
output artifact ids and the ordered operation ids that form the provenance
chain.

# Persistence

`store.py` saves pipeline definitions into structured MFDB tables
(`mfdb_pipeline`, `mfdb_pipeline_node`, `mfdb_pipeline_edge`) and records runs
through `mfdb_pipeline_run` / `mfdb_pipeline_run_operation`. The graph structure
is stored structurally, not as one opaque blob; only per-node parameter bags are
JSON.

See also [MFDB](/architecture/mfdb.md), [data IO](/subsystems/data-io.md), and
[operation history](/subsystems/history.md).
