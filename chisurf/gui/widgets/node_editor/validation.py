"""Custom exceptions and validation helpers for node editor graphs."""

from typing import Dict, List, Any


class NodeGraphValidationError(ValueError):
    """Raised when a graph JSON fails validation."""

    def __init__(self, message: str, path: str = ""):
        super().__init__(f"{message} at {path}" if path else message)
        self.path = path


def validate_graph_dict(data: Dict[str, Any]) -> None:
    """Validate a graph dictionary against the JSON schema v1.

    Raises NodeGraphValidationError if invalid.
    """
    if not isinstance(data, dict):
        raise NodeGraphValidationError("Root must be an object", "")

    # Check top-level keys
    if "nodes" not in data:
        raise NodeGraphValidationError("Missing required field 'nodes'", "nodes")
    if "edges" not in data:
        raise NodeGraphValidationError("Missing required field 'edges'", "edges")
    nodes = data["nodes"]
    edges = data["edges"]
    version = data.get("version", 1)

    if not isinstance(nodes, list):
        raise NodeGraphValidationError("nodes must be an array", "nodes")
    if not isinstance(edges, list):
        raise NodeGraphValidationError("edges must be an array", "edges")
    if not isinstance(version, int) or version != 1:
        raise NodeGraphValidationError("version must be 1", "version")

    # Track node IDs
    node_ids = set()
    for i, node in enumerate(nodes):
        path = f"nodes[{i}]"
        if not isinstance(node, dict):
            raise NodeGraphValidationError("Node must be an object", path)
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise NodeGraphValidationError("id must be a non-empty string", f"{path}.id")
        if node_id in node_ids:
            raise NodeGraphValidationError(f"Duplicate node id '{node_id}'", f"{path}.id")
        node_ids.add(node_id)

        # Required fields
        required = ["type", "title", "inputs", "outputs", "config", "pos", "collapsed"]
        for req in required:
            if req not in node:
                raise NodeGraphValidationError(f"Missing required field '{req}'", f"{path}.{req}")

        # Type checks
        if not isinstance(node["type"], str):
            raise NodeGraphValidationError("type must be a string", f"{path}.type")
        if not isinstance(node["title"], str):
            raise NodeGraphValidationError("title must be a string", f"{path}.title")
        if not isinstance(node["inputs"], list):
            raise NodeGraphValidationError("inputs must be an array", f"{path}.inputs")
        if not isinstance(node["outputs"], list):
            raise NodeGraphValidationError("outputs must be an array", f"{path}.outputs")
        if not isinstance(node["config"], dict):
            raise NodeGraphValidationError("config must be an object", f"{path}.config")
        if not isinstance(node["pos"], list) or len(node["pos"]) != 2 or not all(isinstance(x, (int, float)) for x in node["pos"]):
            raise NodeGraphValidationError("pos must be [x, y] floats", f"{path}.pos")
        if not isinstance(node["collapsed"], bool):
            raise NodeGraphValidationError("collapsed must be a boolean", f"{path}.collapsed")

        # Check inputs/outputs format
        for port_type in ["inputs", "outputs"]:
            ports = node[port_type]
            for j, port in enumerate(ports):
                port_path = f"{path}.{port_type}[{j}]"
                if isinstance(port, str):
                    continue  # Simple format
                elif isinstance(port, dict):
                    if "name" not in port or not isinstance(port["name"], str):
                        raise NodeGraphValidationError("Port name must be a string", f"{port_path}.name")
                else:
                    raise NodeGraphValidationError("Port must be string or object", port_path)

    # Validate edges
    for i, edge in enumerate(edges):
        path = f"edges[{i}]"
        if not isinstance(edge, dict):
            raise NodeGraphValidationError("Edge must be an object", path)

        required_edge = ["source", "source_port", "target", "target_port"]
        for req in required_edge:
            if req not in edge:
                raise NodeGraphValidationError(f"Missing required field '{req}'", f"{path}.{req}")

        source_id = edge["source"]
        target_id = edge["target"]
        source_port = edge["source_port"]
        target_port = edge["target_port"]

        if not isinstance(source_id, str) or source_id not in node_ids:
            raise NodeGraphValidationError(f"Invalid source node '{source_id}'", f"{path}.source")
        if not isinstance(target_id, str) or target_id not in node_ids:
            raise NodeGraphValidationError(f"Invalid target node '{target_id}'", f"{path}.target")
        if not isinstance(source_port, int) or source_port < 0:
            raise NodeGraphValidationError("source_port must be non-negative integer", f"{path}.source_port")
        if not isinstance(target_port, int) or target_port < 0:
            raise NodeGraphValidationError("target_port must be non-negative integer", f"{path}.target_port")
