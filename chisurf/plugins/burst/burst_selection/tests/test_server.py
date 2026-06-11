"""Tests for Burst Selection server methods."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

from chisurf.plugins.burst.burst_selection.api.models import (
    AnalysisSettings,
    BurstDetectionSettings,
    DeltaMacroTimeFilterSettings,
    PhotonFilterSettings,
)
from chisurf.plugins.burst.burst_selection.api.selection import analyze_file
from chisurf.plugins.burst.burst_selection.server import methods as server_module
from chisurf.plugins.burst.burst_selection.server.services import register_burst_selection_services

DATA_FILE = Path(__file__).resolve().parent / "data" / "bh_spc132_sm_dna" / "m000.spc"


def real_data_settings() -> AnalysisSettings:
    """Return deterministic settings for the bundled BH SPC example."""
    settings = AnalysisSettings()
    settings.photon_filter = PhotonFilterSettings(
        channels=[0, 1, 8, 9],
        filter_active=False,
        delta_macro_time_filter=DeltaMacroTimeFilterSettings(dT_min=0.0),
    )
    settings.burst_detection = BurstDetectionSettings(min_photons=20, photon_window=10, time_window=1e-3)
    return settings


def real_data_settings_json() -> dict[str, Any]:
    """Return JSON-compatible settings for ZMQ analysis requests."""
    from chisurf.plugins.burst.burst_selection.api.serialization import to_jsonable

    return to_jsonable(real_data_settings())


def _free_port() -> int:
    """Return a currently free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_analyze_files_server_method_returns_service_shape(tmp_path: Path) -> None:
    """Server analysis method should return the standard service result shape."""
    response = server_module.analyze_files([str(DATA_FILE)], output_dir=str(tmp_path))
    assert response["ok"] is True
    assert response["result"]["files"] == [str(DATA_FILE)]
    assert response["result"]["metadata"]["n_photons"] == 174438


def test_inspect_bur_server_method_returns_service_shape(tmp_path: Path) -> None:
    """Server inspect method should summarize a real generated .bur file."""
    result = analyze_file(DATA_FILE, settings=real_data_settings(), output_dir=tmp_path)
    response = server_module.inspect_bur(result.output_paths["bur"])
    assert response["ok"] is True
    assert response["result"]["n_rows"] == 9533


def test_register_with_service_dispatcher() -> None:
    """Registered Burst Selection handlers should be callable through ServiceDispatcher."""
    from chisurf.server.dispatcher import ServiceDispatcher
    from chisurf.server.session import SessionState

    dispatcher = ServiceDispatcher(SessionState())
    register_burst_selection_services(dispatcher)
    assert dispatcher.has_method("burst_selection.inspect_bur")
    assert set(dispatcher.list_methods()) >= {
        "burst_selection.analyze_files",
        "burst_selection.fit_gmm_from_bur",
        "burst_selection.inspect_bur",
    }


def test_zmq_client_can_call_analyze_and_fit_gmm(tmp_path: Path) -> None:
    """Burst Selection ZMQ client should analyze real data and fit GMM from .bur."""
    from chisurf.plugins.burst.burst_selection.server.client import BurstSelectionClient
    from chisurf.server.transport.zmq import ZmqServer

    def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Dispatch Burst Selection server methods for the ZMQ test."""
        if method == "burst_selection.analyze_files":
            return server_module.analyze_files(**params)
        if method == "burst_selection.fit_gmm_from_bur":
            return server_module.fit_gmm_from_bur(**params)
        raise ValueError(method)

    cmd_port = _free_port()
    pub_port = _free_port()
    server = ZmqServer(handler=handler, cmd_port=cmd_port, pub_port=pub_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.1)
        client = BurstSelectionClient(cmd_port=cmd_port, pub_port=pub_port, timeout_ms=5000)
        try:
            analysis = client.analyze_files(
                [str(DATA_FILE)],
                output_dir=str(tmp_path),
                settings=real_data_settings_json(),
            )
            bur_path = analysis["output_paths"]["bur"]
            fit = client.fit_gmm_from_bur(
                bur_path,
                settings={"covariance_type": "spherical", "n_init": 1},
            )
        finally:
            client.close()
    finally:
        server.stop()
        thread.join(timeout=2)

    assert analysis["metadata"]["n_photons"] == 174438
    assert fit["gmm"]["n_components"] == 1
    assert len(fit["gmm"]["labels"]) == 9533


def test_zmq_client_can_call_inspect_bur(tmp_path: Path) -> None:
    """Burst Selection ZMQ client should call the inspect service."""
    from chisurf.plugins.burst.burst_selection.server.client import BurstSelectionClient
    from chisurf.server.transport.zmq import ZmqServer

    result = analyze_file(DATA_FILE, settings=real_data_settings(), output_dir=tmp_path)
    bur_path = Path(result.output_paths["bur"])
    cmd_port = _free_port()
    pub_port = _free_port()
    server = ZmqServer(
        handler=lambda method, params: server_module.inspect_bur(params["path"]),
        cmd_port=cmd_port,
        pub_port=pub_port,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.1)
        client = BurstSelectionClient(cmd_port=cmd_port, pub_port=pub_port, timeout_ms=2000)
        try:
            response = client.inspect_bur(str(bur_path))
        finally:
            client.close()
    finally:
        server.stop()
        thread.join(timeout=2)

    assert response["n_rows"] == 9533
