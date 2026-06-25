"""Workflow / pipeline engine on the transformer contract (PRD-22).

Compose conformant transformers (PRD-16) into a type-checked dataflow graph that
executes as a recorded, reproducible chain of operations (PRD-11) — the data-level
analog of chinet's typed node graph. Definition + type-checking live in
:mod:`~chisurf.core.pipeline.model`; headless execution in
:mod:`~chisurf.core.pipeline.runner`.
"""

from chisurf.core.pipeline.model import (
    Pipeline,
    PipelineEdge,
    PipelineNode,
    PipelineValidationError,
    topological_order,
    validate_pipeline,
)
from chisurf.core.pipeline.runner import PipelineRun, run_pipeline

__all__ = [
    "Pipeline",
    "PipelineEdge",
    "PipelineNode",
    "PipelineValidationError",
    "PipelineRun",
    "run_pipeline",
    "topological_order",
    "validate_pipeline",
]
