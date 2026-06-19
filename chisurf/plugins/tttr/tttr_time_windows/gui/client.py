"""PluginClient wrapper for Time Window Bins.

Provides typed convenience methods and shields the GUI from direct API imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chisurf import logging
from chisurf.core.plugin.client import InProcessClient


class TimeWindowClient:
    """Client for Time Window Bins backend services.

    Wraps a PluginClient (typically InProcessClient for local use, or
    ZmqClient for remote/server mode).
    """

    def __init__(self, client: Any = None) -> None:
        """Create a Time Window Bins client.

        Parameters
        ----------
        client : PluginClient, optional
            A ``PluginClient`` instance. If ``None``, creates a local
            ``InProcessClient`` with the time-window services registered.
        """
        if client is not None:
            self._client = client
        else:
            self._client = self._make_local_client()

    def analyze_files(
        self,
        file_paths: list[Path],
        time_window_ms: float = 10.0,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Split TTTR files into time-window BIDs.

        Parameters
        ----------
        file_paths : list of Path
            TTTR file paths.
        time_window_ms : float
            Time window duration in milliseconds.
        output_dir : str or Path, optional
            Output directory for ``.bst`` files.

        Returns
        -------
        dict
            Analysis result with ``files``, ``n_windows``, ``output_paths``,
            ``metadata`` keys.

        Raises
        ------
        RuntimeError
            If the RPC call returned an error.
        """
        params: dict[str, Any] = {
            "files": [str(p) for p in file_paths],
            "time_window_ms": time_window_ms,
        }
        if output_dir is not None:
            params["output_dir"] = str(output_dir)
        svc_result = self._client.call(
            "tttr_time_windows.jobs.analyze_files",
            params,
        )
        if not svc_result.get("ok", True):
            err_msg = svc_result.get("error", "unknown error")
            raise RuntimeError(
                f"tttr_time_windows.jobs.analyze_files failed: {err_msg}"
            )
        return svc_result.get("result", {})

    def load_preview(
        self,
        path: Path,
        time_window_ms: float,
    ) -> dict[str, Any]:
        """Load TTTR data for preview plotting.

        Parameters
        ----------
        path : Path
            TTTR file path.
        time_window_ms : float
            Time window in milliseconds for the intensity trace.

        Returns
        -------
        dict
            Preview data with ``tttr``, ``counts``, ``time_axis`` keys.
            Returns empty dict on failure.
        """
        try:
            import tttrlib
            import numpy as np

            tttr = tttrlib.TTTR(str(path))
            time_window_s = time_window_ms / 1000.0
            counts = tttr.get_intensity_trace(time_window_s)
            if counts is None:
                return {}
            time_axis = np.arange(len(counts), dtype=float) * time_window_s
            return {
                "tttr": tttr,
                "counts": np.asarray(counts, dtype=float),
                "time_axis": time_axis,
            }
        except Exception:
            logging.exception("Failed to load preview for %s", path)
            return {}

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with time-window services."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from chisurf.plugins.tttr.tttr_time_windows.backend.services import (
            register_services,
        )

        register_services(dispatcher)
        return InProcessClient(dispatcher)
