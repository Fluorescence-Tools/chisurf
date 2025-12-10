"""Unit tests for node editor JSON validation."""

import pytest

from chisurf.gui.widgets.node_editor.validation import (
    NodeGraphValidationError,
    validate_graph_dict,
)


def test_valid_minimal_graph():
    """Test a minimal valid graph."""
    data = {
        "nodes": [
            {
                "id": "n0",
                "type": "constant",
                "title": "Constant",
                "inputs": [],
                "outputs": ["Value"],
                "config": {},
                "pos": [0.0, 0.0],
                "collapsed": False
            }
        ],
        "edges": []
    }
    validate_graph_dict(data)  # Should not raise


def test_valid_graph_with_edge():
    """Test a valid graph with nodes and an edge."""
    data = {
        "nodes": [
            {
                "id": "n0",
                "type": "constant",
                "title": "Constant",
                "inputs": [],
                "outputs": ["Value"],
                "config": {},
                "pos": [0.0, 0.0],
                "collapsed": False
            },
            {
                "id": "n1",
                "type": "output",
                "title": "Output",
                "inputs": ["Value"],
                "outputs": [],
                "config": {},
                "pos": [100.0, 0.0],
                "collapsed": False
            }
        ],
        "edges": [
            {"source": "n0", "source_port": 0, "target": "n1", "target_port": 0}
        ]
    }
    validate_graph_dict(data)  # Should not raise


def test_invalid_root_not_dict():
    """Test that root must be a dict."""
    with pytest.raises(NodeGraphValidationError, match="Root must be an object"):
        validate_graph_dict([])


def test_invalid_missing_nodes():
    """Test that nodes must be present."""
    data = {"edges": []}
    with pytest.raises(NodeGraphValidationError, match="Missing required field 'nodes'"):
        validate_graph_dict(data)


def test_invalid_duplicate_node_id():
    """Test that node IDs must be unique."""
    data = {
        "nodes": [
            {
                "id": "n0",
                "type": "constant",
                "title": "Constant",
                "inputs": [],
                "outputs": ["Value"],
                "config": {},
                "pos": [0.0, 0.0],
                "collapsed": False
            },
            {
                "id": "n0",  # Duplicate
                "type": "output",
                "title": "Output",
                "inputs": ["Value"],
                "outputs": [],
                "config": {},
                "pos": [100.0, 0.0],
                "collapsed": False
            }
        ],
        "edges": []
    }
    with pytest.raises(NodeGraphValidationError, match="Duplicate node id"):
        validate_graph_dict(data)


def test_invalid_missing_node_field():
    """Test missing required node field."""
    data = {
        "nodes": [
            {
                "id": "n0",
                "type": "constant",
                # Missing "title"
                "inputs": [],
                "outputs": ["Value"],
                "config": {},
                "pos": [0.0, 0.0],
                "collapsed": False
            }
        ],
        "edges": []
    }
    with pytest.raises(NodeGraphValidationError, match="Missing required field 'title'"):
        validate_graph_dict(data)


def test_invalid_edge_invalid_source():
    """Test invalid edge source."""
    data = {
        "nodes": [
            {
                "id": "n0",
                "type": "constant",
                "title": "Constant",
                "inputs": [],
                "outputs": ["Value"],
                "config": {},
                "pos": [0.0, 0.0],
                "collapsed": False
            }
        ],
        "edges": [
            {"source": "nonexistent", "source_port": 0, "target": "n0", "target_port": 0}
        ]
    }
    with pytest.raises(NodeGraphValidationError, match="Invalid source node"):
        validate_graph_dict(data)


def test_invalid_edge_negative_port():
    """Test negative port index."""
    data = {
        "nodes": [
            {
                "id": "n0",
                "type": "constant",
                "title": "Constant",
                "inputs": [],
                "outputs": ["Value"],
                "config": {},
                "pos": [0.0, 0.0],
                "collapsed": False
            }
        ],
        "edges": [
            {"source": "n0", "source_port": -1, "target": "n0", "target_port": 0}
        ]
    }
    with pytest.raises(NodeGraphValidationError, match="source_port must be non-negative integer"):
        validate_graph_dict(data)
