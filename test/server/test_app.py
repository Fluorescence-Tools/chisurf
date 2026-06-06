from __future__ import annotations

import threading
import time

from chisurf.core.api._client import ChisurfClient
from chisurf.server.app import ChiSurfServer
from test.server.helpers import find_free_port


def test_server_app_list_methods():
    cmd_port = find_free_port()
    pub_port = find_free_port()

    server = ChiSurfServer(cmd_port=cmd_port, pub_port=pub_port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    methods = client.list_methods()
    assert "list_datasets" in methods
    assert "list_fits" in methods
    assert "list_methods" in methods
    client.close()
    server.stop()


def test_server_app_list_datasets():
    cmd_port = find_free_port()
    pub_port = find_free_port()

    server = ChiSurfServer(cmd_port=cmd_port, pub_port=pub_port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    datasets = client.list_datasets()
    assert isinstance(datasets, list)
    client.close()
    server.stop()


def test_server_app_job_manager():
    server = ChiSurfServer()
    assert server.job_manager is not None
    assert server.dispatcher is not None
    assert server.event_bus is not None
