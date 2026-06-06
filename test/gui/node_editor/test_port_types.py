"""Unit tests for port type checking in node editor."""

import pytest
from qtpy import QtWidgets, QtCore

from chisurf.gui.widgets.node_editor.scene import NodeScene
from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
from chisurf.gui.widgets.node_editor.model import NodeModel, PortSpec


@pytest.fixture
def app():
    """Ensure a QApplication exists for the duration of the tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def scene(app):  # noqa: ARG001 - app fixture ensures QApplication exists
    """Create a NodeScene for testing."""
    return NodeScene()


def test_validate_connection_types(scene):
    """Test the validate_connection logic with different port types."""
    # Port types: spectral, dye_data, number, any
    
    # Matching types
    p1 = PortSpec("P1", is_output=True, port_type="spectral")
    p2 = PortSpec("P2", is_output=False, port_type="spectral")
    
    # Helper to mock items
    class MockPort:
        def __init__(self, spec):
            self.spec = spec
            
    assert scene.validate_connection(MockPort(p1), MockPort(p2)) is True
    
    # Mismatching types
    p3 = PortSpec("P3", is_output=False, port_type="dye_data")
    assert scene.validate_connection(MockPort(p1), MockPort(p3)) is False
    
    # 'any' type connects to anything
    p4 = PortSpec("P4", is_output=True, port_type="any")
    assert scene.validate_connection(MockPort(p4), MockPort(p3)) is True
    assert scene.validate_connection(MockPort(p4), MockPort(p2)) is True
    
    # Same direction (both outputs)
    p5 = PortSpec("P5", is_output=True, port_type="spectral")
    assert scene.validate_connection(MockPort(p1), MockPort(p5)) is False


def test_mouse_release_prevents_mismatch(scene):
    """Test that mouse release on an incompatible port cancels the connection."""
    # Create two nodes with mismatching ports
    m1 = NodeModel(
        title="N1", 
        inputs=[], 
        outputs=[PortSpec("O", True, port_type="spectral")], 
        node_type="t", 
        config={}
    )
    m2 = NodeModel(
        title="N2", 
        inputs=[PortSpec("I", False, port_type="dye_data")], 
        outputs=[], 
        node_type="t", 
        config={}
    )
    
    i1 = NodeGraphicsItem(m1)
    i2 = NodeGraphicsItem(m2)
    scene.addItem(i1)
    scene.addItem(i2)
    
    # i1 output is at index 0 (no inputs)
    # i2 input is at index 0
    p1_out = i1.port_items[0]
    p2_in = i2.port_items[0]
    
    # Start edge creation
    event_press = QtWidgets.QGraphicsSceneMouseEvent(QtCore.QEvent.GraphicsSceneMousePress)
    event_press.setScenePos(p1_out.scene_pos())
    event_press.setButton(QtCore.Qt.LeftButton)
    scene.mousePressEvent(event_press)
    
    assert scene._current_edge is not None
    
    # Release on mismatching port
    event_release = QtWidgets.QGraphicsSceneMouseEvent(QtCore.QEvent.GraphicsSceneMouseRelease)
    event_release.setScenePos(p2_in.scene_pos())
    event_release.setButton(QtCore.Qt.LeftButton)
    scene.mouseReleaseEvent(event_release)
    
    # Edge should be removed because of type mismatch
    assert len(scene.edges) == 0
    assert scene._current_edge is None


def test_mouse_release_allows_match(scene):
    """Test that mouse release on a matching port completes the connection."""
    m1 = NodeModel(
        title="N1", 
        inputs=[], 
        outputs=[PortSpec("O", True, port_type="spectral")], 
        node_type="t", 
        config={}
    )
    m2 = NodeModel(
        title="N2", 
        inputs=[PortSpec("I", False, port_type="spectral")], 
        outputs=[], 
        node_type="t", 
        config={}
    )
    
    i1 = NodeGraphicsItem(m1)
    i2 = NodeGraphicsItem(m2)
    scene.addItem(i1)
    scene.addItem(i2)
    
    p1_out = i1.port_items[0]
    p2_in = i2.port_items[0]
    
    # Start edge creation
    event_press = QtWidgets.QGraphicsSceneMouseEvent(QtCore.QEvent.GraphicsSceneMousePress)
    event_press.setScenePos(p1_out.scene_pos())
    event_press.setButton(QtCore.Qt.LeftButton)
    scene.mousePressEvent(event_press)
    
    # Release on matching port
    event_release = QtWidgets.QGraphicsSceneMouseEvent(QtCore.QEvent.GraphicsSceneMouseRelease)
    event_release.setScenePos(p2_in.scene_pos())
    event_release.setButton(QtCore.Qt.LeftButton)
    scene.mouseReleaseEvent(event_release)
    
    # Edge should be created
    assert len(scene.edges) == 1
    assert scene._current_edge is None
