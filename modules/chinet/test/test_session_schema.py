from __future__ import annotations

import json

import numpy as np
import pytest

import chinet as cn
from chisurf.core.parameter import Parameter


def _connected_session() -> cn.Session:
    source = cn.Node(name="source")
    out = cn.Port(1.25, is_output=True, name="out")
    source.add_output_port("out", out)

    follower = cn.Node(name="follower")
    inp = cn.Port(
        0.0,
        is_bounded=True,
        lb=0.1,
        ub=10.0,
        name="in",
    )
    follower.add_input_port("in", inp)
    inp.link = out

    session = cn.Session({"source": source, "follower": follower})
    return session


def test_port_value_bounds_fixed_and_link_round_trip_json_document() -> None:
    cn.DB.clear()
    port = cn.Port(
        1.25,
        fixed=True,
        is_bounded=True,
        lb=0.1,
        ub=10.0,
        name="tau",
    )
    link = cn.Port(2.5, name="link")
    port.link = link

    doc = json.loads(port.get_json())
    assert doc["value"] == 2.5
    assert doc["fixed"] is True
    assert doc["bounds"] == [0.1, 10.0]
    assert doc["link"] == link.oid

    restored = cn.Port(oid=port.oid)
    restored.set_document(doc)
    assert restored.value == 2.5
    assert restored.fixed is True
    assert restored.bounds == (0.1, 10.0)


def test_node_python_callback_creates_ports_and_evaluates() -> None:
    cn.DB.clear()

    def multiply(x: float = 2.0, y: float = 3.0) -> dict[str, float]:
        return {"out": x * y}

    node = cn.Node(callback_function=multiply, name="multiply")
    assert set(node.inputs) == {"x", "y"}
    assert set(node.outputs) == {"out"}

    node.inputs["x"].value = 4.0
    node.inputs["y"].value = 5.0
    node.evaluate()
    assert node.outputs["out"].value == 20.0


def test_session_to_dict_includes_session_nodes_ports_and_links() -> None:
    cn.DB.clear()
    session = _connected_session()

    payload = session.to_dict()
    assert payload["session"]["type"] == "session"
    assert set(payload["session"]["nodes"]) == {"source", "follower"}
    assert payload["objects"]


def test_session_save_load_jsonl_remains_backward_compatible(tmp_path) -> None:
    cn.DB.clear()
    session = _connected_session()
    path = tmp_path / "session.jsonl"

    session.save(str(path))
    loaded = cn.Session.load(str(path))

    assert loaded is not None
    assert loaded.oid == session.oid
    assert set(loaded.nodes) == {"source", "follower"}
    follower_in = loaded.nodes["follower"].inputs["in"]
    assert follower_in.link.oid == loaded.nodes["source"].outputs["out"].oid


def test_chiurf_parameter_wrapping_chinet_port_preserves_state() -> None:
    cn.DB.clear()
    port = cn.Port(1.0, is_bounded=True, lb=0.0, ub=5.0)
    master = Parameter(name="master", value=1.0, port=port)
    follower = Parameter(name="follower", value=0.0, port=cn.Port(0.0), link=master)

    master.bounds = (0.0, 5.0)
    master.bounds_on = True
    master.fixed = True
    assert master.value == 1.0
    assert master.bounds == (0.0, 5.0)
    assert master.fixed is True
    assert follower.value == 1.0
    assert follower.is_linked is True


def test_session_schema_round_trip_preserves_ids_values_bounds_and_link() -> None:
    cn.DB.clear()
    session = _connected_session()
    source_out = session.nodes["source"].outputs["out"]
    follower_in = session.nodes["follower"].inputs["in"]

    payload = session.to_schema()
    assert payload["schema_name"] == "chinet.session.v1"
    assert payload["schema_version"] == 1
    assert payload["session_id"] == session.oid
    assert len(payload["nodes"]) == 2
    assert len(payload["ports"]) == 2
    assert payload["links"][0]["source_port_id"] == source_out.oid
    assert payload["links"][0]["target_port_id"] == follower_in.oid

    restored = cn.Session.from_schema(payload)
    restored_out = restored.nodes["source"].outputs["out"]
    restored_in = restored.nodes["follower"].inputs["in"]
    assert restored.oid == session.oid
    assert restored_out.oid == source_out.oid
    assert restored_in.oid == follower_in.oid
    assert restored_out.value == 1.25
    assert restored_in.link.oid == restored_out.oid
    assert restored_in.bounds == (0.1, 10.0)


def test_session_schema_round_trips_vector_and_scalar_ports() -> None:
    cn.DB.clear()
    node = cn.Node(name="vector_node")
    scalar = cn.Port(1.0, name="scalar")
    vector = cn.Port([1.0, 2.0, 3.0], name="vector")
    node.add_input_port("scalar", scalar)
    node.add_input_port("vector", vector)
    session = cn.Session({"vector_node": node})

    restored = cn.Session.from_schema(session.to_schema())
    assert restored.nodes["vector_node"].inputs["scalar"].value == 1.0
    assert np.allclose(restored.nodes["vector_node"].inputs["vector"].value, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_name": "chinet.session.v9", "schema_version": 9},
        {"schema_name": "chinet.session.v1", "schema_version": 2},
        {"schema_name": "chinet.session.v1", "schema_version": 1},
    ],
)
def test_session_schema_rejects_malformed_or_unsupported_payload(payload) -> None:
    cn.DB.clear()
    with pytest.raises(ValueError):
        cn.Session.from_schema(payload)
