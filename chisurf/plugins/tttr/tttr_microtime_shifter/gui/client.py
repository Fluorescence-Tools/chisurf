"""PluginClient wrapper for Micro-time Shifter.

Provides typed convenience methods and shields the GUI from direct API imports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from chisurf.core.plugin.client import InProcessClient


logger = logging.getLogger(__name__)


class MicrotimeShifterClient:
    """Client for Micro-time Shifter backend services.

    Wraps a PluginClient (typically InProcessClient for local use, or
    ZmqClient for remote/server mode). GUI code should never import
    ``..api.*`` directly when going through this client.
    """

    def __init__(self, client: Any = None):
        """Create a MicrotimeShifterClient.

        Parameters
        ----------
        client : PluginClient, optional
            A ``PluginClient`` instance. If ``None``, creates a local
            ``InProcessClient`` with microtime_shifter services registered.

        """
        if client is not None:
            self._client = client
        else:
            self._client = self._make_local_client()

    # ── public API ─────────────────────────────────────────────────

    def apply(
        self,
        file_paths: list[Path],
        global_shift: int = 0,
        channel_shifts: dict[int, int] | None = None,
        filetype: str | None = None,
        output_dir: str | Path | None = None,
        mfdb: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply micro-time shifts to files.

        Parameters
        ----------
        file_paths : list of Path
            TTTR file paths.
        global_shift : int
            Global micro-time shift.
        channel_shifts : dict, optional
            Per-channel shifts.
        filetype : str, optional
            Explicit file type.
        output_dir : str or Path, optional
            Output directory.
        mfdb : dict, optional
            MFDB archival context.

        Returns
        -------
        dict
            Shift result dict.

        Raises
        ------
        RuntimeError
            If the RPC call returned an error.

        """
        params: dict[str, Any] = {
            "files": [str(p) for p in file_paths],
            "global_shift": global_shift,
        }
        if channel_shifts:
            params["channel_shifts"] = {str(k): v for k, v in channel_shifts.items()}
        if filetype is not None:
            params["filetype"] = filetype
        if output_dir is not None:
            params["output_dir"] = str(output_dir)
        if mfdb is not None:
            params["mfdb"] = mfdb
        svc_result = self._client.call(METHOD_APPLY, params)
        if not svc_result.get("ok", True):
            err_msg = svc_result.get("error", "unknown error")
            raise RuntimeError(f"{METHOD_APPLY} failed: {err_msg}")
        return svc_result.get("result", {})

    def load_metadata(self, path: Path) -> dict[str, Any]:
        """Load metadata for a TTTR file.

        Parameters
        ----------
        path : Path
            TTTR file path.

        Returns
        -------
        dict
            Metadata with ``routing_channels``, ``n_mt``, ``n_photons``.

        """
        result = self._client.call(
            METHOD_LOAD_METADATA,
            {"path": str(path)},
        )
        return result.get("result", {})

    def identify(self, path: Path) -> dict[str, Any]:
        """Look up a file in the MFDB object store.

        Parameters
        ----------
        path : Path
            TTTR file path.

        Returns
        -------
        dict
            Identification result with ``found``, ``artifact_id``, ``md5``.

        """
        result = self._client.call(
            METHOD_IDENTIFY,
            {"path": str(path)},
        )
        return result.get("result", {})

    def histogram(
        self,
        path: Path | list[Path],
        global_shift: int = 0,
        channel_shifts: dict[int, int] | None = None,
        filetype: str | None = None,
    ) -> dict[str, Any]:
        """Load histogram data for GUI preview.

        Parameters
        ----------
        path : Path or list of Path
            TTTR file path(s).
        global_shift : int
            Global micro-time shift.
        channel_shifts : dict, optional
            Per-channel shifts.
        filetype : str, optional
            Explicit file type.

        Returns
        -------
        dict
            Histogram data with ``n_mt``, ``routing_channels``, and
            ``histograms`` keys.

        """
        if isinstance(path, list):
            path_val: Any = [str(p) for p in path]
        else:
            path_val = str(path)
        params: dict[str, Any] = {
            "path": path_val,
            "global_shift": global_shift,
        }
        if channel_shifts:
            params["channel_shifts"] = {str(k): v for k, v in channel_shifts.items()}
        if filetype is not None:
            params["filetype"] = filetype
        result = self._client.call(METHOD_HISTOGRAM, params)
        if not result.get("ok", True):
            err_msg = result.get("error", "unknown error")
            raise RuntimeError(f"{METHOD_HISTOGRAM} failed: {err_msg}")
        return result.get("result", {})

    def describe_contract(self) -> dict[str, Any]:
        """Return the workflow contract descriptor."""
        result = self._client.call(METHOD_DESCRIBE_CONTRACT, {})
        return result.get("result", {})

    # ── internals ──────────────────────────────────────────────────

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with microtime_shifter services."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from chisurf.plugins.tttr.tttr_microtime_shifter.backend.services import (
            register_services,
        )
        register_services(dispatcher)
        return InProcessClient(dispatcher)


# Import method constants from contract
from ..api.contract import (
    METHOD_APPLY,
    METHOD_DESCRIBE_CONTRACT,
    METHOD_HISTOGRAM,
    METHOD_IDENTIFY,
    METHOD_LOAD_METADATA,
)
