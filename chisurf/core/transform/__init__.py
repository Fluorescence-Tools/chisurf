"""General transformer contract (PRD-16).

One abstraction every data-transformer plugin (Burst Selection, Microtime Shifter,
background correction, correlation, …) obeys: declared typed input/output ports, a
`.dic`-declared parameter schema (PRD-11 ``mfdb_operation_parameter_def``), a pure
``transform``, and uniform MFDB registration as an operation node. Mirrors chinet's
typed node/ports on the data side.
"""

from chisurf.core.transform.transformer import (
    PortSpec,
    TransformInputs,
    TransformResult,
    Transformer,
    TransformerConformanceError,
    check_transformer_conformance,
    get_transformer,
    list_transformers,
    register_transformer,
)

__all__ = [
    "PortSpec",
    "TransformInputs",
    "TransformResult",
    "Transformer",
    "TransformerConformanceError",
    "check_transformer_conformance",
    "get_transformer",
    "list_transformers",
    "register_transformer",
]
