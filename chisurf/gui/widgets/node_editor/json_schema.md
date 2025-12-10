# Node Editor JSON Schema v1

## Overview

The node editor uses a JSON-based format to save and load graph states. This document defines version 1 of the schema.

## Top-level Structure

The JSON is an object with the following keys:

- `nodes` (array of objects): List of node definitions.
- `edges` (array of objects): List of edge definitions.
- `version` (integer, optional): Schema version (currently 1). Defaults to 1 if omitted.
- `meta` (object, optional): Free-form metadata for plugin-specific settings.

Example:

```json
{
  "version": 1,
  "meta": {"plugin_version": "1.0"},
  "nodes": [...],
  "edges": [...]
}
```

## Node Schema

Each node in the `nodes` array is an object with:

- `id` (string, required): Unique identifier for the node within the graph.
- `type` (string, required): Node type identifier (e.g., "constant", "binary_op").
- `title` (string, required): Human-readable title displayed on the node.
- `inputs` (array of strings or objects): Input port names. Can be:
  - Simple array of strings: `["Value 1", "Value 2"]`
  - Array of objects: `[{"name": "Value 1", "is_output": false}]` (for future extension).
- `outputs` (array of strings or objects): Output port names, same format as inputs but with `"is_output": true`.
- `config` (object): Free-form configuration for the node (must be JSON-serializable).
- `pos` (array of two floats): Scene position `[x, y]`.
- `collapsed` (boolean): Whether the node starts collapsed.

Example:

```json
{
  "id": "n0",
  "type": "constant",
  "title": "Constant",
  "inputs": [],
  "outputs": ["Value"],
  "config": {"value": 100.0, "label": "Value"},
  "pos": [-260, -80],
  "collapsed": false
}
```

## Edge Schema

Each edge in the `edges` array is an object with:

- `source` (string): ID of the source node.
- `source_port` (integer): Index into the source node's `outputs` array.
- `target` (string): ID of the target node.
- `target_port` (integer): Index into the target node's `inputs` array.

Example:

```json
{
  "source": "n0",
  "source_port": 0,
  "target": "n1",
  "target_port": 0
}
```

## Validation Rules

- Node IDs must be unique across all nodes.
- Edge source/target IDs must refer to existing nodes.
- Port indices must be valid for the referenced node's inputs/outputs.
- All required fields must be present.

## Future Versions

- Version 2 may add support for nested graphs or advanced port types.
- Backward compatibility will be maintained where possible.
