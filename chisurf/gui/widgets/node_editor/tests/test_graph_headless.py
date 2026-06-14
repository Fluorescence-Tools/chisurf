import pytest

from chisurf.gui.widgets.node_editor.graph import EdgeDef, GraphDef, NodeDef, PortDef


def test_headless_graph_validation():
    # Construct a simple DAG headlessly
    n1 = NodeDef(id="n1", node_type="test", title="Node 1", inputs=[], outputs=[PortDef("Out", True)])
    n2 = NodeDef(id="n2", node_type="test", title="Node 2", inputs=[PortDef("In", False)], outputs=[PortDef("Out", True)])
    n3 = NodeDef(id="n3", node_type="test", title="Node 3", inputs=[PortDef("In", False)], outputs=[])

    e1 = EdgeDef(source="n1", source_port=0, target="n2", target_port=0)
    e2 = EdgeDef(source="n2", source_port=0, target="n3", target_port=0)

    graph = GraphDef(nodes=[n1, n2, n3], edges=[e1, e2])

    assert graph.validate_acyclic() is True
    assert graph.topological_node_ids() == ["n1", "n2", "n3"]
    assert len(graph.incoming_edges("n2")) == 1
    assert len(graph.outgoing_edges("n2")) == 1
    assert graph.incoming_edges("n2")[0].source == "n1"
    assert graph.outgoing_edges("n2")[0].target == "n3"

def test_headless_graph_cycle():
    n1 = NodeDef(id="n1", node_type="test", title="Node 1", inputs=[PortDef("In", False)], outputs=[PortDef("Out", True)])
    n2 = NodeDef(id="n2", node_type="test", title="Node 2", inputs=[PortDef("In", False)], outputs=[PortDef("Out", True)])

    e1 = EdgeDef(source="n1", source_port=0, target="n2", target_port=0)
    e2 = EdgeDef(source="n2", source_port=0, target="n1", target_port=0)

    graph = GraphDef(nodes=[n1, n2], edges=[e1, e2])

    assert graph.validate_acyclic() is False
    with pytest.raises(ValueError, match="Graph contains a cycle"):
        graph.topological_node_ids()

def test_graph_def_round_trip():
    data = {
        "nodes": [
            {
                "id": "1",
                "type": "constant",
                "title": "Constant",
                "inputs": [],
                "outputs": ["Value"],
                "config": {"value": 5.0},
                "pos": [10.0, 20.0],
                "collapsed": False,
                "z": 1.0
            }
        ],
        "edges": [],
        "version": 1,
        "meta": {"purpose": "workflow"}
    }

    graph = GraphDef.from_scene_dict(data)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "1"
    assert graph.nodes[0].pos == [10.0, 20.0]
    assert graph.nodes[0].config["value"] == 5.0
    assert graph.meta["purpose"] == "workflow"

    out = graph.to_scene_dict()
    assert out["nodes"][0]["inputs"] == []
    assert out["nodes"][0]["outputs"] == ["Value"]
    assert out["meta"]["purpose"] == "workflow"
