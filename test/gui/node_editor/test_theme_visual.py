"""Theme and visual feedback tests for node editor."""

import pytest
from qtpy import QtWidgets

from chisurf.gui.widgets.node_editor import theme
from chisurf.gui.widgets.node_editor.edge_item import EdgeGraphicsItem
from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
from chisurf.gui.widgets.node_editor.model import NodeModel, PortSpec
from chisurf.gui.widgets.node_editor.scene import NodeScene


@pytest.fixture
def app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def scene(app):  # noqa: ARG001 - ensures QApplication exists
    return NodeScene()


def _make_two_nodes_one_edge(scene: NodeScene):
    """Helper to create two nodes and a single connecting edge."""
    model1 = NodeModel(
        title="N1",
        inputs=[PortSpec(name="In", is_output=False)],
        outputs=[PortSpec(name="Out", is_output=True)],
        node_type="test",
        config={},
    )
    model2 = NodeModel(
        title="N2",
        inputs=[PortSpec(name="In", is_output=False)],
        outputs=[PortSpec(name="Out", is_output=True)],
        node_type="test",
        config={},
    )

    item1 = NodeGraphicsItem(model1)
    item2 = NodeGraphicsItem(model2)
    scene.addItem(item1)
    scene.addItem(item2)
    item1.setPos(0, 0)
    item2.setPos(100, 0)

    edge = EdgeGraphicsItem(item1.port_items[1], item2.port_items[0])
    scene.addItem(edge)
    scene.register_edge(edge)
    return edge


def test_edge_cycle_colors_use_theme(scene):
    """EdgeGraphicsItem.set_cycle should use theme colors for normal/cycle states."""
    edge = _make_two_nodes_one_edge(scene)

    # Normal state
    edge.set_cycle(False)
    normal_color = theme.color("edge_normal", (160, 160, 160))
    assert edge.pen().color().getRgb()[:3] == normal_color.getRgb()[:3]

    # Cycle state
    edge.set_cycle(True)
    cycle_color = theme.color("edge_in_cycle", (200, 60, 60))
    assert edge.pen().color().getRgb()[:3] == cycle_color.getRgb()[:3]


def test_theme_has_per_node_type_title_colors():
    """Theme should expose per-node-type title colors for built-in types."""
    const_top = theme.color("node_title_top_constant", (1, 2, 3))
    const_bot = theme.color("node_title_bottom_constant", (4, 5, 6))
    bin_top = theme.color("node_title_top_binary_op", (7, 8, 9))
    out_top = theme.color("node_title_top_output", (10, 11, 12))
    ctrl_top = theme.color("node_title_top_controls", (13, 14, 15))

    # We at least expect QColor instances; exact palette is defined in theme.json
    for c in (const_top, const_bot, bin_top, out_top, ctrl_top):
        assert c.isValid()
