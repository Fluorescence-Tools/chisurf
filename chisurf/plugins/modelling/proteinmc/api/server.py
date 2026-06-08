"""ZMQ JSON-RPC server for ProteinMC.

Exposes ``run``, ``stop``, ``status``, and ``get_result`` methods over
a ZMQ REQ/REP socket and broadcasts progress on a PUB socket.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from chisurf.plugins.modelling.proteinmc.model import (
    ProteinMCProgress,
    ProteinMCRunner,
)
from chisurf.server.transport.zmq import ZmqServer

_log = logging.getLogger(__name__)


class ProteinMCApiServer:
    """ZMQ server that wraps ProteinMCRunner for remote access.

    Parameters
    ----------
    cmd_port : int
        TCP port for REQ/REP commands.
    pub_port : int
        TCP port for PUB progress broadcasts.
    host : str
        Bind address.
    """

    def __init__(
        self,
        cmd_port: int = 9765,
        pub_port: int = 9766,
        host: str = "127.0.0.1",
    ) -> None:
        self._host = host
        self._cmd_port = cmd_port
        self._pub_port = pub_port
        self._runner: ProteinMCRunner | None = None
        self._thread: threading.Thread | None = None
        self._zmq: ZmqServer | None = None

    def serve_forever(self) -> None:
        """Start the ZMQ event loop (blocking)."""
        self._zmq = ZmqServer(
            handler=self._handle_request,
            cmd_port=self._cmd_port,
            pub_port=self._pub_port,
            host=self._host,
        )
        _log.info(
            "ProteinMC API listening on tcp://%s:%s (cmd) / tcp://%s:%s (pub)",
            self._host, self._cmd_port, self._host, self._pub_port,
        )
        self._zmq.serve_forever()

    def start(self) -> None:
        """Run the server in a background daemon thread."""
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the server."""
        if self._zmq is not None:
            self._zmq._running = False
        if self._runner is not None:
            self._runner.stop()
        self._zmq = None
        self._runner = None

    def _broadcast(self, topic: str, payload: dict[str, Any]) -> None:
        if self._zmq is not None:
            self._zmq.broadcast_event(topic, payload)

    def _handle_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "run":
            return self._run(**params)
        elif method == "stop":
            return self._cmd_stop()
        elif method == "status":
            return self._status()
        elif method == "get_result":
            return self._get_result()
        else:
            return {"ok": False, "error": f"Unknown method: {method}"}

    # --- RPC methods -------------------------------------------------------

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        """Start ProteinMC in a background thread."""
        if self._runner is not None:
            return {"ok": False, "error": "A run is already in progress"}

        structure_source = kwargs.pop("structure_source", None)
        if structure_source is None:
            return {"ok": False, "error": "Missing required parameter: structure_source"}

        def progress_callback(p: ProteinMCProgress) -> None:
            self._broadcast("proteinmc.progress", {
                "frame_index": p.frame_index,
                "target_frames": p.target_frames,
                "iteration": p.iteration,
                "accepted": p.accepted,
                "rejected": p.rejected,
                "energy": p.energy,
                "labeling_energy": p.labeling_energy,
            })

        self._result: dict[str, Any] | None = None
        self._error: str | None = None

        def run_in_thread() -> None:
            try:
                runner = ProteinMCRunner(
                    structure_source,
                    progress_callback=progress_callback,
                    **kwargs,
                )
                self._runner = runner
                result = runner.run()
                self._result = {
                    "output_file": result.output_file,
                    "n_frames": result.n_frames,
                    "accepted": result.accepted,
                    "rejected": result.rejected,
                    "energies": result.energies,
                    "labeling_energies": result.labeling_energies,
                }
                self._broadcast("proteinmc.finished", self._result)
            except Exception as exc:
                self._error = str(exc)
                self._broadcast("proteinmc.failed", {"error": self._error})
            finally:
                self._runner = None

        threading.Thread(target=run_in_thread, daemon=True).start()
        return {"ok": True, "message": "ProteinMC run started"}

    def _cmd_stop(self) -> dict[str, Any]:
        if self._runner is not None:
            self._runner.stop()
            return {"ok": True, "message": "Stop requested"}
        return {"ok": False, "error": "No run in progress"}

    def _status(self) -> dict[str, Any]:
        if self._runner is not None:
            return {"ok": True, "running": True}
        if self._result is not None:
            return {"ok": True, "running": False, "finished": True}
        if self._error is not None:
            return {"ok": True, "running": False, "failed": True, "error": self._error}
        return {"ok": True, "running": False}

    def _get_result(self) -> dict[str, Any]:
        if self._result is not None:
            return {"ok": True, "result": self._result}
        if self._error is not None:
            return {"ok": False, "error": self._error}
        return {"ok": False, "error": "No result available yet"}
