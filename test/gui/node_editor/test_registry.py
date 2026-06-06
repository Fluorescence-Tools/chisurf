"""Unit tests for node type registry."""

import pytest

from chisurf.gui.widgets.node_editor.registry import registry, NodeType
from chisurf.gui.widgets.node_editor.model import PortSpec


def test_registry_register_and_get():
    """Test registering and retrieving node types."""
    # Clear registry for test
    registry._types.clear()

    node_type = NodeType(
        id="test_node",
        title="Test Node",
        inputs=[PortSpec("In", False)],
        outputs=[PortSpec("Out", True)],
        factory=lambda cfg: None,
        default_config={"value": 1}
    )

    registry.register(node_type)
    retrieved = registry.get("test_node")
    assert retrieved == node_type

    assert "test_node" in registry.available_ids()
    assert registry.all_types() == {"test_node": node_type}


def test_registry_duplicate_id():
    """Test that registering duplicate ID raises error."""
    registry._types.clear()

    node_type1 = NodeType(
        id="dup",
        title="First",
        inputs=[],
        outputs=[],
        factory=None
    )
    node_type2 = NodeType(
        id="dup",
        title="Second",
        inputs=[],
        outputs=[],
        factory=None
    )

    registry.register(node_type1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(node_type2)


def test_registry_get_missing():
    """Test getting non-existent node type."""
    registry._types.clear()
    assert registry.get("missing") is None


def test_builtin_nodes_registered():
    """Test that built-in nodes are registered."""
    # Assuming editor.py registers them on import
    from chisurf.gui.widgets.node_editor.editor import NodeEditorWidget  # noqa: F401

    # Check that built-in types are available
    ids = registry.available_ids()
    assert "constant" in ids
    assert "binary_op" in ids
    assert "output" in ids
    assert "controls" in ids

    # Check their properties
    constant = registry.get("constant")
    assert constant.title == "Constant"
    assert len(constant.outputs) == 1
    assert constant.outputs[0].name == "Value"
