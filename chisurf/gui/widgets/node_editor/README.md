# Node Editor

A self-contained node-based editor for creating and editing computational graphs, built with PyQt5.

## Features

- **Visual Node Editing**: Drag-and-drop interface for creating nodes and connections.
- **JSON Serialization**: Save and load graphs as JSON files with validation.
- **DAG Checks**: Automatic detection of cycles with optional enforcement.
- **Customizable Nodes**: Define node types with custom widgets and logic.
- **Theming**: Dark theme with configurable colors and metrics.

## Installation

This is part of the chisurf project. No additional installation required beyond PyQt5 and optional networkx for advanced features.

## Usage

### Basic Editor

```python
from PyQt5.QtWidgets import QApplication
from node_editor.editor import NodeEditorWidget

app = QApplication([])
editor = NodeEditorWidget()
editor.show()
app.exec()
```

### Loading/Saving Graphs

```python
# Load from JSON string
editor.load_graph_from_json(json_str)

# Save to JSON string
json_str = editor.to_json()

# Load/save files
editor.load_graph_from_file("my_graph.json")
editor.save_graph_to_file("my_graph.json")
```

### Custom Node Types

Define nodes by subclassing and registering:

```python
from node_editor.model import NodeModel, PortSpec

class MyNode(NodeModel):
    def __init__(self):
        super().__init__(
            title="My Node",
            inputs=[PortSpec("Input", False)],
            outputs=[PortSpec("Output", True)],
            node_type="my_node",
            config={"value": 1.0}
        )

# Register in editor
editor.available_node_types = {"My Node": MyNode}
```

## JSON Schema

Graphs are saved in a structured JSON format (see `json_schema.md` for details).

- `nodes`: Array of node definitions with id, type, config, position.
- `edges`: Array of connections between node ports.

## DAG Validation

- Edges are validated to prevent invalid connections.
- Optional `enforce_acyclic` mode prevents cycle creation.
- Cycle detection uses networkx for accurate graph analysis.

## Testing

Run tests with:

```bash
cd node_editor
python -m pytest tests/
```

## Architecture

- **Model Layer**: `model.py` (NodeModel, PortSpec) - pure data and logic.
- **View/Controller Layer**: `node_item.py`, `scene.py`, `view.py` - Qt graphics and interaction.
- **Registry Layer**: `registry.py` - extensible node type definitions.
- **Editor Layer**: `editor.py` - high-level widget with UI panels.

Serialization flows through the model/scene layer, ensuring consistency.

## Node Registry

Register custom node types for extensibility:

```python
from registry import registry, NodeType
from model import PortSpec

registry.register(NodeType(
    id="my_node",
    title="My Custom Node",
    inputs=[PortSpec("Input", False)],
    outputs=[PortSpec("Output", True)],
    factory=my_widget_factory,
    default_config={"param": 1}
))
```

## JSON Schema

Graphs are saved in a versioned JSON format (see `json_schema.md`).

## DAG Validation

- Automatic cycle detection and highlighting.
- Optional `enforce_acyclic` mode prevents invalid connections.
- Uses NetworkX for robust graph analysis.
