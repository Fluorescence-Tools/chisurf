"""Node type registry for extensible node definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
from qtpy import QtWidgets

from .model import PortSpec

if TYPE_CHECKING:
    pass


@dataclass
class NodeType:
    """Descriptor for a node type in the registry."""
    id: str
    title: str
    inputs: List[PortSpec]
    outputs: List[PortSpec]
    category: str = "General"
    factory: Optional[Callable[[Dict], QtWidgets.QWidget]] = None
    default_config: Dict = None

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
