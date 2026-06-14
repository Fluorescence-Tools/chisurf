"""Node type registry for extensible node definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    pass


from .model import PortSpec


@dataclass
class NodeType:
    """Descriptor for a node type in the registry."""
    id: str
    title: str
    inputs: List[PortSpec]
    outputs: List[PortSpec]
    category: str = "General"
    factory: Optional[Callable] = None
    default_config: Optional[Dict] = None
    width: float = 190.0

    # Workflow metadata extensions (optional)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    config_schema: Optional[Dict] = None
    runtime: Optional[str] = None
    operation: Optional[str] = None
    executor: Optional[Callable] = None

    def __post_init__(self):
        if self.default_config is None:
            self.default_config = {}


class NodeRegistry:
    """Registry for node types, allowing dynamic registration."""

    def __init__(self):
        self._types: Dict[str, NodeType] = {}

    def register(self, node_type: NodeType):
        """Register a node type."""
        if node_type.id in self._types:
            raise ValueError(f"Node type '{node_type.id}' already registered")
        self._types[node_type.id] = node_type

    def unregister(self, type_id: str) -> None:
        """Unregister a node type by ID."""
        if type_id in self._types:
            del self._types[type_id]

    def replace(self, node_type: NodeType) -> None:
        """Replace or register a node type."""
        self._types[node_type.id] = node_type

    def by_category(self) -> Dict[str, List[NodeType]]:
        """Group registered node types by their category."""
        grouped: Dict[str, List[NodeType]] = {}
        for nt in self._types.values():
            grouped.setdefault(nt.category, []).append(nt)
        return grouped

    def workflow_types(self) -> List[NodeType]:
        """Return nodes that define a runtime or executor."""
        return [
            nt for nt in self._types.values()
            if nt.runtime is not None or nt.executor is not None
        ]

    def get(self, type_id: str) -> Optional[NodeType]:
        """Get a node type by ID."""
        return self._types.get(type_id)

    def all_types(self) -> Dict[str, NodeType]:
        """Get all registered node types."""
        return self._types.copy()

    def available_ids(self) -> List[str]:
        """Get list of available node type IDs."""
        return list(self._types.keys())


# Global registry instance
registry = NodeRegistry()
