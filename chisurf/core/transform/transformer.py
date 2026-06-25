"""The transformer contract types, registry, and conformance check (PRD-16).

A transformer is an ``operation_type`` (PRD-11) plus a small mandatory contract:
typed input/output **ports**, parameters declared in the ``.dic`` (not code), a
**pure** ``transform`` (no Qt, no DB), and uniform MFDB registration via the PRD-11
operation-node path. This module defines the contract, a self-registering registry,
and a conformance check that gates new transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class PortSpec:
    """A typed data port — the data-side analog of a chinet port.

    Declares the artifact kinds (and optional formats) a transformer consumes or
    produces on this port, and its arity. Connections/inputs are validated against
    the declared kinds at the boundary.
    """

    name: str
    kinds: tuple[str, ...]
    formats: tuple[str, ...] = ()
    arity: str = "one"  # "one" | "many"
    required: bool = True

    def accepts_kind(self, kind: str) -> bool:
        return kind in self.kinds

    def accepts_format(self, fmt: str | None) -> bool:
        return not self.formats or (fmt or "") in self.formats


@dataclass
class TransformInputs:
    """Inputs handed to ``Transformer.transform``.

    ``artifacts`` maps an input port name to the resolved payload(s)/path(s) the
    transformer reads. Kept deliberately plain so ``transform`` stays pure.
    """

    artifacts: Mapping[str, Any] = field(default_factory=dict)
    files: tuple[str, ...] = ()


@dataclass
class TransformResult:
    """Outputs of a pure ``transform`` — payloads keyed by output port name."""

    outputs: Mapping[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class Transformer(Protocol):
    """The contract every data-transformer plugin must satisfy.

    Attributes
    ----------
    transformer_id : str
        Stable id (e.g. ``"microtime_shifter"``).
    operation_type : str
        PRD-11 operation vocabulary value; its parameter schema is declared in the
        ``.dic`` (``mfdb_operation_parameter_def``), never in code.
    version : str
        Contract/implementation version.
    input_spec, output_spec : list[PortSpec]
        Declared typed data ports.
    """

    transformer_id: str
    operation_type: str
    version: str
    input_spec: list[PortSpec]
    output_spec: list[PortSpec]

    def transform(self, inputs: TransformInputs, parameters: dict) -> TransformResult:
        """Pure transform: inputs + parameters -> outputs. No Qt, no DB."""
        ...


class TransformerConformanceError(ValueError):
    """A transformer does not satisfy the PRD-16 contract."""


_REGISTRY: dict[str, Transformer] = {}


def register_transformer(transformer: Transformer) -> Transformer:
    """Register a transformer so the system can enumerate it (idempotent by id)."""
    _REGISTRY[transformer.transformer_id] = transformer
    return transformer


def get_transformer(transformer_id: str) -> Transformer | None:
    return _REGISTRY.get(transformer_id)


def list_transformers() -> list[Transformer]:
    return sorted(_REGISTRY.values(), key=lambda t: t.transformer_id)


def get_transformer_for_operation(operation_type: str) -> Transformer | None:
    """Return a registered transformer whose ``operation_type`` matches, or ``None``.

    The pipeline engine (PRD-22) resolves a node's ``operation_type`` to the
    transformer that declares its typed ports, so edges can be type-checked against
    ``input_spec``/``output_spec``. If several transformers share an operation type,
    the lowest ``transformer_id`` wins (stable, deterministic).
    """
    for transformer in list_transformers():
        if transformer.operation_type == operation_type:
            return transformer
    return None


def check_transformer_conformance(transformer: Transformer, conn: Any = None) -> None:
    """Validate a transformer against the PRD-16 contract.

    Checks the required attributes, that it declares input/output ports, and that
    its ``operation_type`` has a ``.dic`` parameter schema (when a DB connection is
    supplied, via PRD-11 ``mfdb_operation_parameter_def``). Raises
    :class:`TransformerConformanceError` on any violation.
    """
    for attr in ("transformer_id", "operation_type", "version"):
        if not getattr(transformer, attr, None):
            raise TransformerConformanceError(f"transformer missing {attr!r}")
    if not getattr(transformer, "input_spec", None):
        raise TransformerConformanceError(
            f"{transformer.transformer_id}: no input_spec declared"
        )
    if not getattr(transformer, "output_spec", None):
        raise TransformerConformanceError(
            f"{transformer.transformer_id}: no output_spec declared"
        )
    for spec in list(transformer.input_spec) + list(transformer.output_spec):
        if not isinstance(spec, PortSpec) or not spec.kinds:
            raise TransformerConformanceError(
                f"{transformer.transformer_id}: invalid port spec {spec!r}"
            )
    if not callable(getattr(transformer, "transform", None)):
        raise TransformerConformanceError(
            f"{transformer.transformer_id}: transform is not callable"
        )
    if conn is not None:
        from chisurf.core.mfdb.operation_parameters import get_operation_parameter_defs

        if not get_operation_parameter_defs(conn, transformer.operation_type):
            raise TransformerConformanceError(
                f"{transformer.transformer_id}: operation_type "
                f"{transformer.operation_type!r} has no .dic parameter schema "
                "(mfdb_operation_parameter_def)"
            )
