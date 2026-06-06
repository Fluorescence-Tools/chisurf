from __future__ import annotations

import threading
import time
from chisurf.server.transport.zmq import ZmqServer, ZmqClient
from test.server.helpers import find_free_port


def test_zmq_round_trip():
    cmd_port = find_free_port()
    pub_port = find_free_port()

    def handler(method, params):
        return {"ok": True, "value": params.get("x", 0) + params.get("y", 0)}

    server = ZmqServer(handler=handler, cmd_port=cmd_port, pub_port=pub_port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    client = ZmqClient(cmd_port=cmd_port)
    result = client.call("test", {"x": 3, "y": 4})

    assert "result" in result
    inner = result["result"]
    assert inner["ok"]
    assert inner["value"] == 7

    server.stop()
    client.close()


def test_zmq_request_id_increments():
    """Each call gets a unique request id."""
    cmd_port = find_free_port()
    pub_port = find_free_port()

    def handler(method, params):
        return {"ok": True}

    server = ZmqServer(handler=handler, cmd_port=cmd_port, pub_port=pub_port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    client = ZmqClient(cmd_port=cmd_port)
    r1 = client.call("test", {})
    r2 = client.call("test", {})
    r3 = client.call("test", {})

    assert r1["id"] == 1
    assert r2["id"] == 2
    assert r3["id"] == 3

    server.stop()
    client.close()


def test_zmq_invalid_request_no_method():
    """Missing method field returns Invalid Request error."""
    cmd_port = find_free_port()
    pub_port = find_free_port()

    def handler(method, params):
        return {"ok": True}

    server = ZmqServer(handler=handler, cmd_port=cmd_port, pub_port=pub_port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{cmd_port}")
    sock.send_json({"jsonrpc": "2.0"})
    reply = sock.recv_json()
    assert reply["error"]["code"] == -32600
    assert reply["error"]["message"] == "Invalid Request"
    sock.close()
    ctx.term()
    server.stop()


def test_zmq_invalid_request_bad_type():
    """Non-dict message returns Invalid Request error."""
    cmd_port = find_free_port()
    pub_port = find_free_port()

    def handler(method, params):
        return {"ok": True}

    server = ZmqServer(handler=handler, cmd_port=cmd_port, pub_port=pub_port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{cmd_port}")
    sock.send_string("not a json dict")
    reply = sock.recv_json()
    assert reply["error"]["code"] == -32700
    assert "Parse" in reply["error"]["message"]
    sock.close()
    ctx.term()
    server.stop()


def test_zmq_timeout():
    """Client returns timeout error if server does not respond."""
    import zmq
    cmd_port = find_free_port()

    def handler(method, params):
        import time
        time.sleep(10)

    server = ZmqServer(handler=handler, cmd_port=cmd_port, pub_port=find_free_port())
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    client = ZmqClient(cmd_port=cmd_port, timeout_ms=200)
    result = client.call("test", {})
    assert "error" in result
    assert "timeout" in result["error"].lower()

    server.stop()
    client.close()


def test_zmq_parse_error_handled():
    """Missing jsonrpc field is handled (lenient — treated as success)."""
    cmd_port = find_free_port()
    pub_port = find_free_port()

    def handler(method, params):
        return {"ok": True, "method": method}

    server = ZmqServer(handler=handler, cmd_port=cmd_port, pub_port=pub_port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{cmd_port}")
    sock.send_json({"method": "meta.ping"})
    reply = sock.recv_json()
    assert "result" in reply
    assert reply["result"].get("ok") is True
    sock.close()
    ctx.term()
    server.stop()
