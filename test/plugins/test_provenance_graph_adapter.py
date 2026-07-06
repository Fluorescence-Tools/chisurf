from chisurf.gui.widgets.node_editor.graph import GraphDef
from chisurf.gui.widgets.node_editor.validation import validate_graph_dict
from mfdb.admin.gui.provenance_graph import (
    layout_nodes,
    mfdb_graph_to_node_editor_graph,
    node_key,
    record_title,
    relation_color,
)


def test_empty_mfdb_graph():
    out = mfdb_graph_to_node_editor_graph({})
    assert out["version"] == 1
    assert out["nodes"] == []
    assert out["edges"] == []
    assert out["meta"]["purpose"] == "provenance_view"

def test_raw_to_process_to_product():
    graph = {
        "nodes": [
            {"node_type": "raw_data", "node_id": "raw_1", "data_type": "ptu"},
            {"node_type": "processing_run", "node_id": "proc_1", "type": "Burst Selection"},
            {"node_type": "processed_data", "node_id": "prod_1", "product_type": "bur"}
        ],
        "edges": [
            {
                "source_node_type": "raw_data", "source_node_id": "raw_1",
                "target_node_type": "processing_run", "target_node_id": "proc_1",
                "relationship_type": "input_to"
            },
            {
                "source_node_type": "processing_run", "source_node_id": "proc_1",
                "target_node_type": "processed_data", "target_node_id": "prod_1",
                "relationship_type": "produced"
            }
        ]
    }

    out = mfdb_graph_to_node_editor_graph(graph)
    assert len(out["nodes"]) == 3
    assert len(out["edges"]) == 2

    # Assert stable node keys
    expected_keys = {
        node_key("raw_data", "raw_1"),
        node_key("processing_run", "proc_1"),
        node_key("processed_data", "prod_1")
    }
    node_ids = {n["id"] for n in out["nodes"]}
    assert node_ids == expected_keys

    # Every converted node has exactly one input and one output
    for n in out["nodes"]:
        assert len(n["inputs"]) == 1
        assert len(n["outputs"]) == 1

    # Every converted edge uses source_port=1 and target_port=0
    for e in out["edges"]:
        assert e["source_port"] == 1
        assert e["target_port"] == 0

    # Color check
    assert out["edges"][0]["config"]["color"] == [70, 120, 200]  # input_to: blue
    assert out["edges"][1]["config"]["color"] == [70, 180, 100]  # produced: green

def test_edge_only_dependency_graph_synthesizes_endpoint_nodes():
    graph = {
        "edges": [
            {
                "source_node_type": "raw_data", "source_node_id": "raw_1",
                "target_node_type": "processing_run", "target_node_id": "proc_1",
                "relationship_type": "input_to"
            }
        ]
    }

    out = mfdb_graph_to_node_editor_graph(graph)

    assert len(out["nodes"]) == 2
    assert len(out["edges"]) == 1
    assert {n["id"] for n in out["nodes"]} == {
        node_key("raw_data", "raw_1"),
        node_key("processing_run", "proc_1"),
    }


def test_missing_nodes_edges_skipped():
    graph = {
        "nodes": [
            {"node_type": "raw_data", "node_id": "raw_1"}
        ],
        "edges": [
            {
                "source_node_type": "raw_data", "source_node_id": "raw_1",
                "target_node_type": "processing_run", "target_node_id": "missing_proc",
                "relationship_type": "input_to"
            }
        ]
    }
    out = mfdb_graph_to_node_editor_graph(graph)
    assert len(out["nodes"]) == 1
    assert len(out["edges"]) == 0


def test_converted_graph_validates_with_node_editor_schema():
    graph = {
        "nodes": [
            {"node_type": "raw_data", "node_id": "raw_1", "data_type": "ptu"},
            {"node_type": "processing_run", "node_id": "proc_1", "type": "Burst Selection"},
            {"node_type": "processed_data", "node_id": "prod_1", "product_type": "bur"},
        ],
        "edges": [
            {
                "source_node_type": "raw_data", "source_node_id": "raw_1",
                "target_node_type": "processing_run", "target_node_id": "proc_1",
                "relationship_type": "input_to",
            },
            {
                "source_node_type": "processing_run", "source_node_id": "proc_1",
                "target_node_type": "processed_data", "target_node_id": "prod_1",
                "relationship_type": "produced",
            },
        ],
    }

    out = mfdb_graph_to_node_editor_graph(graph)

    validate_graph_dict(out)
    graph_def = GraphDef.from_scene_dict(out)
    assert graph_def.topological_node_ids() == [
        node_key("raw_data", "raw_1"),
        node_key("processing_run", "proc_1"),
        node_key("processed_data", "prod_1"),
    ]


def test_layout_is_deterministic_by_level_kind_title_id():
    nodes = [
        {"id": "n2", "type": "mfdb_record", "title": "Product: bur", "config": {"node_type": "processed_data"}},
        {"id": "n1", "type": "mfdb_record", "title": "Raw: ptu", "config": {"node_type": "raw_data"}},
    ]

    positions = layout_nodes(nodes, [])

    assert positions["n2"] == (0.0, 0.0)
    assert positions["n1"] == (0.0, 130.0)


def test_relationship_colors():
    assert relation_color("input_to") == [70, 120, 200]
    assert relation_color("produced") == [70, 180, 100]
    assert relation_color("parameter_of") == [200, 180, 70]
    assert relation_color("derived_from") == [120, 120, 120]
    assert relation_color("unknown") == [180, 180, 180]


def test_record_title_uses_raw_file_suffix():
    assert record_title({"node_type": "raw_data", "node_id": "raw_1", "file_path": "/tmp/input.ptu"}) == "Raw: ptu"
