"""Headless graph data model — no Qt dependency."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PortDef:
    name: str
    is_output: bool
    port_type: str = "spectral"

    @staticmethod
    def from_dict(d: dict) -> "PortDef":
        return PortDef(
            name=d["name"],
            is_output=d.get("is_output", False),
            port_type=d.get("port_type", "spectral")
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_output": self.is_output,
            "port_type": self.port_type
        }


@dataclass
class NodeDef:
    id: str
    node_type: str
    title: str
    inputs: List[PortDef] = field(default_factory=list)
    outputs: List[PortDef] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0

    @staticmethod
    def from_dict(d: dict) -> "NodeDef":
        return NodeDef(
            id=str(d["id"]),
            node_type=d["type"],
            title=d["title"],
            inputs=[PortDef.from_dict(p) if isinstance(p, dict) else PortDef(p, False) for p in d.get("inputs", [])],
            outputs=[PortDef.from_dict(p) if isinstance(p, dict) else PortDef(p, True) for p in d.get("outputs", [])],
            config=d.get("config", {}).copy(),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0)
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.node_type,
            "title": self.title,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "config": self.config.copy(),
            "x": self.x,
            "y": self.y
        }


@dataclass
class EdgeDef:
    source: str       # node id
    source_port: int
    target: str
    target_port: int

    @staticmethod
    def from_dict(d: dict) -> "EdgeDef":
        return EdgeDef(
            source=str(d["source"]),
            source_port=d["source_port"],
            target=str(d["target"]),
            target_port=d["target_port"]
        )

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_port": self.source_port,
            "target": self.target,
            "target_port": self.target_port
        }


@dataclass
class GraphDef:
    nodes: List[NodeDef] = field(default_factory=list)
    edges: List[EdgeDef] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "GraphDef":
        """Parse NodeScene.to_dict() output."""
        return GraphDef(
            nodes=[NodeDef.from_dict(n) for n in d.get("nodes", [])],
            edges=[EdgeDef.from_dict(e) for e in d.get("edges", [])]
        )

    def to_dict(self) -> dict:
        """Produce NodeScene.from_dict()-compatible output."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges]
        }
