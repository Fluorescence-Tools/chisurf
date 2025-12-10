from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from qtpy import QtWidgets


@dataclass
class PortSpec:
    """Specification of a node port used for layout and logic."""
    name: str
    is_output: bool
    # Optional type information and constraints. ``port_type`` is a short
    # abbreviation such as "sF"/"sI"/"vF"/"vI" (scalar/vector int/float).
    # ``fixed`` marks the port as fixed (non-variable), and ``min_value`` /
    # ``max_value`` allow bounded ranges for numeric ports. All fields are
    # optional so existing graphs that only specify names continue to work.
    port_type: str = ""
    fixed: bool = False
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class NodeModel:
    """Model for a node: title, port specs and optional widget factory.

    The view (`NodeGraphicsItem`) is responsible only for painting, geometry
    and interaction; it reads from this model and (optionally) calls the
    `content_factory` once to obtain an embedded QWidget.
    """

    title: str
    inputs: List[PortSpec]
    outputs: List[PortSpec]
    node_type: str = "generic"
    config: Dict[str, Any] = field(default_factory=dict)
    content_factory: Optional[Callable[[], QtWidgets.QWidget]] = None
