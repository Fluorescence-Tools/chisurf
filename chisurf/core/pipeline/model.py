"""Pipeline definition + type-checked composition (PRD-22).

A pipeline is a directed acyclic graph of **transformer invocations** at the data
level (the chinet node/port model applied to data operations). A node binds a
transformer's ``operation_type`` (PRD-11/16) to parameters; an edge wires a
producer's output port to a consumer's input port.

This module is pure: it knows nothing about MFDB, files, or Qt. It resolves a
node's ``operation_type`` to its registered transformer (PRD-16) purely to read the
declared port kinds, and validates every edge against them — so an invalid
composition fails *at definition time*, before anything runs. Execution lives in
:mod:`chisurf.core.pipeline.runner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chisurf.core.transform import PortSpec, Transformer, get_transformer_for_operation


@dataclass(frozen=True)
class PipelineNode:
    """One transformer invocation: an ``operation_type`` bound to parameters.

    ``name`` is the node's unique handle within a pipeline (edges reference it).
    ``parameters`` are the bound transformer parameters (PRD-11 schema); they are
    validated against the operation type's ``.dic`` schema when the node runs, not
    here.
    """

    name: str
    operation_type: str
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineEdge:
    """A typed wire: ``source.source_port`` feeds ``target.target_port``."""

    source: str
    source_port: str
    target: str
    target_port: str


@dataclass(frozen=True)
class Pipeline:
    """A named, versioned graph of transformer invocations."""

    name: str
    nodes: tuple[PipelineNode, ...] = ()
    edges: tuple[PipelineEdge, ...] = ()
    version: str = "1.0"

    def node(self, name: str) -> PipelineNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"no node named {name!r} in pipeline {self.name!r}")

    def incoming(self, node_name: str) -> list[PipelineEdge]:
        """Edges feeding ``node_name`` (its upstream wires)."""
        return [e for e in self.edges if e.target == node_name]

    def source_nodes(self) -> list[PipelineNode]:
        """Nodes with no incoming edge — they take external inputs at run time."""
        fed = {e.target for e in self.edges}
        return [n for n in self.nodes if n.name not in fed]


class PipelineValidationError(ValueError):
    """A pipeline is not a valid, type-checked composition."""


def _resolve(node: PipelineNode, resolver) -> Transformer:
    transformer = resolver(node.operation_type)
    if transformer is None:
        raise PipelineValidationError(
            f"node {node.name!r}: no registered transformer for operation_type "
            f"{node.operation_type!r}"
        )
    return transformer


def _port(specs: list[PortSpec], name: str) -> PortSpec | None:
    for spec in specs:
        if spec.name == name:
            return spec
    return None


def validate_pipeline(pipeline: Pipeline, *, resolver=get_transformer_for_operation) -> None:
    """Validate a pipeline as a type-checked DAG; raise on the first violation.

    Checks, in order: unique node names; every ``operation_type`` resolves to a
    registered transformer; every edge references existing nodes and declared ports;
    each edge is **type-compatible** (the producer's output-port kinds intersect the
    consumer's input-port kinds — PRD-16); and the graph is acyclic.
    """
    seen: set[str] = set()
    for node in pipeline.nodes:
        if node.name in seen:
            raise PipelineValidationError(f"duplicate node name {node.name!r}")
        seen.add(node.name)
        _resolve(node, resolver)  # every node's operation_type must be registered

    for edge in pipeline.edges:
        if edge.source not in seen:
            raise PipelineValidationError(f"edge from unknown node {edge.source!r}")
        if edge.target not in seen:
            raise PipelineValidationError(f"edge to unknown node {edge.target!r}")

        src = _resolve(pipeline.node(edge.source), resolver)
        dst = _resolve(pipeline.node(edge.target), resolver)
        out_port = _port(list(src.output_spec), edge.source_port)
        if out_port is None:
            raise PipelineValidationError(
                f"node {edge.source!r} ({src.operation_type}) has no output port "
                f"{edge.source_port!r}"
            )
        in_port = _port(list(dst.input_spec), edge.target_port)
        if in_port is None:
            raise PipelineValidationError(
                f"node {edge.target!r} ({dst.operation_type}) has no input port "
                f"{edge.target_port!r}"
            )
        if not set(out_port.kinds) & set(in_port.kinds):
            raise PipelineValidationError(
                f"incompatible edge {edge.source}.{edge.source_port} -> "
                f"{edge.target}.{edge.target_port}: producer kinds "
                f"{out_port.kinds} do not match consumer kinds {in_port.kinds}"
            )

    # acyclicity: topological_order raises on a cycle.
    topological_order(pipeline)


def topological_order(pipeline: Pipeline) -> list[PipelineNode]:
    """Return the nodes in a valid execution order (Kahn's algorithm).

    Raises :class:`PipelineValidationError` if the graph contains a cycle.
    """
    indegree = {n.name: 0 for n in pipeline.nodes}
    for edge in pipeline.edges:
        if edge.target in indegree:
            indegree[edge.target] += 1
    # stable order: queue the ready nodes in declaration order.
    ready = [n.name for n in pipeline.nodes if indegree[n.name] == 0]
    order: list[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for edge in pipeline.edges:
            if edge.source == name and edge.target in indegree:
                indegree[edge.target] -= 1
                if indegree[edge.target] == 0:
                    ready.append(edge.target)
    if len(order) != len(pipeline.nodes):
        cyclic = sorted(set(indegree) - set(order))
        raise PipelineValidationError(f"pipeline has a cycle among nodes {cyclic}")
    return [pipeline.node(name) for name in order]
