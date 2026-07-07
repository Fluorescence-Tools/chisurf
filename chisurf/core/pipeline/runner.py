"""Headless pipeline runner (PRD-22).

Topologically evaluates a validated :class:`~chisurf.core.pipeline.model.Pipeline`
and runs each node. Execution reuses the PRD-21 *replay-executor* seam: a node is a
:class:`~chisurf.core.mfdb.provenance.compute_spec.ComputeSpec` (operation_type + parameters +
source artifact ids), run by the ``operation_type``'s registered executor, which
materializes inputs, runs the pure PRD-16 ``transform``, and registers each output
as a recorded operation. So a pipeline run is a chain of recorded operations with
full provenance — queryable through the lineage API and reproducible.

No transformer-specific glue lives here: the runner is built only on the transformer
registry (for type-checked composition, in ``model``) and the replay-executor
registry (for execution).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from chisurf.core.mfdb.provenance.compute_spec import (
    ComputeSpec,
    NoReplayExecutorError,
    get_compute_spec,
    get_replay_executor,
)
from chisurf.core.pipeline.model import Pipeline, topological_order, validate_pipeline


@dataclass
class PipelineRun:
    """The outcome of running a pipeline: what each node produced, and the chain.

    ``node_outputs`` maps a node name to the artifact id(s) it produced;
    ``operation_ids`` is the recorded operation chain in execution order (the
    provenance handle for lineage queries).
    """

    pipeline_name: str
    node_outputs: dict[str, list[str]] = field(default_factory=dict)
    operation_ids: list[str] = field(default_factory=list)
    #: Set once the run is persisted (see ``pipeline.store.record_pipeline_run``).
    pipeline_run_id: str = ""

    def outputs(self, node_name: str) -> list[str]:
        return self.node_outputs.get(node_name, [])

    @property
    def final_artifacts(self) -> list[str]:
        """Artifacts produced by sink nodes (those flattened across all outputs)."""
        return [aid for ids in self.node_outputs.values() for aid in ids]


def run_pipeline(
    pipeline: Pipeline,
    inputs: Mapping[str, list[str]],
    db: Any,
    *,
    validate: bool = True,
) -> PipelineRun:
    """Execute ``pipeline`` and return its :class:`PipelineRun`.

    Parameters
    ----------
    pipeline:
        The composition to run.
    inputs:
        Maps a *source* node name (a node with no incoming edge) to the external
        input artifact ids it consumes (e.g. the raw measurement id).
    db:
        An MFDB handle passed to each replay executor.
    validate:
        Re-validate composition (type-checked edges, acyclicity) before running.

    Each node, in topological order, gathers its source artifact ids (external inputs
    for source nodes plus the outputs of upstream nodes), is run by its registered
    replay executor, and its produced artifact id(s) and recorded operation are
    captured. Raises :class:`~chisurf.core.mfdb.provenance.compute_spec.NoReplayExecutorError`
    if a node's ``operation_type`` has no registered executor.
    """
    if validate:
        validate_pipeline(pipeline)

    run = PipelineRun(pipeline_name=pipeline.name)
    for node in topological_order(pipeline):
        source_ids: list[str] = list(inputs.get(node.name, []))
        for edge in pipeline.incoming(node.name):
            source_ids.extend(run.node_outputs.get(edge.source, []))

        executor = get_replay_executor(node.operation_type)
        if executor is None:
            raise NoReplayExecutorError(
                f"node {node.name!r}: no replay executor registered for "
                f"operation_type {node.operation_type!r}"
            )
        spec = ComputeSpec(
            operation_type=node.operation_type,
            parameters=dict(node.parameters),
            source_artifact_ids=tuple(source_ids),
        )
        new_artifact_id = executor(spec, db)
        run.node_outputs[node.name] = [new_artifact_id]

        produced_spec = get_compute_spec(db, new_artifact_id)
        if produced_spec is not None and produced_spec.operation_id:
            run.operation_ids.append(produced_spec.operation_id)
    return run
