"""Headless graph data model — no Qt dependency."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


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
            port_type=d.get("type", d.get("port_type", "spectral")),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.port_type,
        }

    def to_entry(self) -> Any:
        if self.port_type == "spectral" or not self.port_type:
            return self.name
        return {
            "name": self.name,
            "type": self.port_type
        }


@dataclass
class NodeDef:
    id: str
    node_type: str
    title: str
    inputs: List[PortDef] = field(default_factory=list)
    outputs: List[PortDef] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0])
    collapsed: bool = False
    z: float = 1.0

    @property
    def x(self) -> float:
        return self.pos[0] if len(self.pos) == 2 else 0.0

    @property
    def y(self) -> float:
        return self.pos[1] if len(self.pos) == 2 else 0.0

    @staticmethod
    def from_dict(d: dict) -> "NodeDef":
        pos = d.get("pos")
        if not isinstance(pos, list) or len(pos) != 2:
            pos = [float(d.get("x", 0.0)), float(d.get("y", 0.0))]
        return NodeDef(
            id=str(d["id"]),
            node_type=d.get("type", "generic"),
            title=d.get("title", ""),
            inputs=[PortDef.from_dict(p) if isinstance(p, dict) else PortDef(p, False) for p in d.get("inputs", [])],
            outputs=[PortDef.from_dict(p) if isinstance(p, dict) else PortDef(p, True) for p in d.get("outputs", [])],
            config=d.get("config", {}).copy(),
            pos=pos,
            collapsed=d.get("collapsed", False),
            z=d.get("z", 1.0),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.node_type,
            "title": self.title,
            "inputs": [p.to_entry() for p in self.inputs],
            "outputs": [p.to_entry() for p in self.outputs],
            "config": self.config.copy(),
            "pos": self.pos,
            "collapsed": self.collapsed,
            "z": self.z,
        }


@dataclass
class EdgeDef:
    source: str       # node id
    source_port: int
    target: str
    target_port: int
    config: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "EdgeDef":
        return EdgeDef(
            source=str(d["source"]),
            source_port=d["source_port"],
            target=str(d["target"]),
            target_port=d["target_port"],
            config=d.get("config", {}).copy(),
        )

    def to_dict(self) -> dict:
        d = {
            "source": self.source,
            "source_port": self.source_port,
            "target": self.target,
            "target_port": self.target_port,
        }
        if self.config:
            d["config"] = self.config.copy()
        return d


@dataclass
class GraphDef:
    nodes: List[NodeDef] = field(default_factory=list)
    edges: List[EdgeDef] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    @staticmethod
    def from_dict(d: dict) -> "GraphDef":
        """Parse NodeScene.to_dict() output."""
        return GraphDef(
            nodes=[NodeDef.from_dict(n) for n in d.get("nodes", [])],
            edges=[EdgeDef.from_dict(e) for e in d.get("edges", [])],
            meta=d.get("meta", {}).copy(),
            version=d.get("version", 1),
        )

    @classmethod
    def from_scene_dict(cls, data: dict) -> "GraphDef":
        return cls.from_dict(data)

    def to_scene_dict(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict:
        """Produce NodeScene.from_dict()-compatible output."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "version": self.version,
            "meta": self.meta.copy(),
        }

    def incoming_edges(self, node_id: str) -> List[EdgeDef]:
        return [e for e in self.edges if e.target == node_id]

    def outgoing_edges(self, node_id: str) -> List[EdgeDef]:
        return [e for e in self.edges if e.source == node_id]

    def validate_acyclic(self) -> bool:
        try:
            self.topological_node_ids()
            return True
        except ValueError:
            return False

    def topological_node_ids(self) -> List[str]:
        # Simple topological sort using Kahn's algorithm
        adj = {n.id: [] for n in self.nodes}
        in_degree = {n.id: 0 for n in self.nodes}

        for edge in self.edges:
            # Skip malformed edges
            if edge.source not in adj or edge.target not in adj:
                continue
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        # Sort queue to be deterministic
        queue.sort()

        result = []
        visited_count = 0

        while queue:
            u = queue.pop(0)
            result.append(u)
            visited_count += 1

            neighbors = sorted(adj[u])
            for v in neighbors:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if visited_count != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result
