"""Embedded, replayable compute specs for derived artifacts (PRD-21 Task 2).

Prior art (Orange3 `compute_value`): provenance is most useful when it is not just
*traceable* but *replayable* — a derived value carries enough to recompute itself.

Here the compute spec is **not a new serialization format**: it is the PRD-11
operation node that produced an artifact, addressed as a unit — its ``operation_type``,
its role-indexed parameters (`mfdb_parameter`), and its source artifact ids
(`mfdb_operation_artifact` inputs). From that:

- ``recompute(artifact_id)`` re-runs the spec; ``replay(artifact_id, overrides)`` is
  the "what-if" (re-run with one parameter changed) — the mechanism behind PRD-27
  branches.
- Execution is pluggable: an ``operation_type`` registers a replay executor (a
  transformer knows how to run itself). Extraction needs no executor and is always
  available.

>>> spec = get_compute_spec(db, shifted_artifact_id)
>>> spec.operation_type, spec.source_artifact_ids
('microtime_shift', ('raw-…',))
>>> what_if = spec.with_overrides({"global_shift": 9})
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ComputeSpec:
    """A derived artifact's producing operation captured as a replayable value."""

    operation_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source_artifact_ids: tuple[str, ...] = ()
    #: The operation that produced the artifact this spec was read from; empty for a
    #: derived "what-if" spec (it has not been recorded as an operation yet).
    operation_id: str = ""

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ComputeSpec":
        """Return a new spec with ``overrides`` merged over the parameters.

        The result has no ``operation_id`` (it is a hypothetical, not yet recorded).
        This is the "what-if" used by replay and PRD-27 branching.
        """
        params = dict(self.parameters)
        params.update(overrides)
        return replace(self, parameters=params, operation_id="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_type": self.operation_type,
            "parameters": dict(self.parameters),
            "source_artifact_ids": list(self.source_artifact_ids),
            "operation_id": self.operation_id,
        }


def _reconstruct_parameters(rows: list[Any]) -> dict[str, Any]:
    """Rebuild a register_operation-shaped parameter mapping from stored rows.

    A single role-less row → a scalar value; rows carrying roles → a role-indexed
    list of ``{"value", "role"}`` (the repeatable form register_operation accepts).
    """
    grouped: dict[str, list[tuple[Any, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["name"], []).append((row["value"], row["role"]))
    params: dict[str, Any] = {}
    for name, values in grouped.items():
        if len(values) == 1 and values[0][1] in (None, ""):
            params[name] = values[0][0]
        else:
            params[name] = [{"value": v, "role": r} for v, r in values]
    return params


def get_compute_spec(db: Any, artifact_id: str) -> ComputeSpec | None:
    """Return the replayable :class:`ComputeSpec` for ``artifact_id``, or ``None``.

    ``None`` when the artifact has no producing operation (a root/imported artifact
    with nothing to recompute).
    """
    # No raw SQL here — reuse the centralized MFDatabase query layer (which
    # itself goes through the dictionary-driven DAO) and the schema-whitelisted
    # DAO for the one reverse lookup that has no dedicated method.
    output_links = db.dao.list(
        "mfdb_operation_artifact",
        filters={"artifact_id": artifact_id, "direction": "output"},
        order_by="ordinal",
    )
    if not output_links:
        return None
    operation_id = output_links[0]["operation_id"]

    operation = db.get_operation(operation_id) or {}
    operation_type = operation.get("operation_type") or ""

    sources: list[str] = []
    for row in db.get_operation_artifacts(operation_id, direction="input"):
        aid = row["artifact_id"]
        if aid not in sources:  # DISTINCT, order-preserving
            sources.append(aid)

    param_rows = db.get_parameters(operation_id=operation_id)

    return ComputeSpec(
        operation_type=operation_type,
        parameters=_reconstruct_parameters(param_rows),
        source_artifact_ids=tuple(sources),
        operation_id=operation_id,
    )


# -- replay executor registry (the pluggable execution seam) -----------------

#: ``operation_type`` → executor ``(spec, db) -> new_artifact_id``. A transformer
#: registers how to run itself; extraction/what-if work without one.
ReplayExecutor = Callable[[ComputeSpec, Any], str]
_EXECUTORS: dict[str, ReplayExecutor] = {}


class NoReplayExecutorError(RuntimeError):
    """No replay executor is registered for an operation type."""


def register_replay_executor(operation_type: str, executor: ReplayExecutor) -> ReplayExecutor:
    """Register how to (re)run ``operation_type``; returns the executor."""
    _EXECUTORS[operation_type] = executor
    return executor


def unregister_replay_executor(operation_type: str) -> None:
    _EXECUTORS.pop(operation_type, None)


def get_replay_executor(operation_type: str) -> ReplayExecutor | None:
    return _EXECUTORS.get(operation_type)


def _execute(spec: ComputeSpec, db: Any) -> str:
    executor = _EXECUTORS.get(spec.operation_type)
    if executor is None:
        raise NoReplayExecutorError(
            f"no replay executor registered for operation_type {spec.operation_type!r}"
        )
    return executor(spec, db)


def recompute(db: Any, artifact_id: str) -> str:
    """Re-run an artifact's compute spec; returns the new artifact id."""
    spec = get_compute_spec(db, artifact_id)
    if spec is None:
        raise ValueError(
            f"artifact {artifact_id!r} has no producing operation; nothing to recompute"
        )
    return _execute(spec, db)


def replay(
    db: Any, artifact_id: str, parameter_overrides: Mapping[str, Any] | None = None
) -> str:
    """Re-run an artifact's spec with parameter overrides ("what-if").

    Returns the new artifact id. With no overrides this equals :func:`recompute`.
    """
    spec = get_compute_spec(db, artifact_id)
    if spec is None:
        raise ValueError(
            f"artifact {artifact_id!r} has no producing operation; nothing to replay"
        )
    if parameter_overrides:
        spec = spec.with_overrides(parameter_overrides)
    return _execute(spec, db)
