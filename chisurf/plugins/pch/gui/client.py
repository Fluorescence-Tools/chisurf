"""PluginClient wrapper for PCH backend services.

Provides typed convenience methods and shields the GUI from direct
backend/API imports.  Supports both in-process and ZMQ communication.
"""

from __future__ import annotations

import logging
from typing import Any

from chisurf.core.plugin.client import InProcessClient

logger = logging.getLogger(__name__)


class PCHClient:
    """Client for PCH backend services.

    Wraps a ``PluginClient`` (``InProcessClient`` by default, or
    ``ZmqClient`` for remote/server mode).

    Parameters
    ----------
    client : PluginClient, optional
        Pre-configured client instance.  If ``None``, creates a local
        ``InProcessClient``.
    host : str, optional
        ZMQ server host (ignored if *client* is provided).
    cmd_port : int, optional
        ZMQ command port (ignored if *client* is provided).
    pub_port : int, optional
        ZMQ pub port (ignored if *client* is provided).

    """

    def __init__(
        self,
        client: Any = None,
        host: str = "127.0.0.1",
        cmd_port: int = 8765,
        pub_port: int = 8766,
    ):
        if client is not None:
            self._client = client
        else:
            self._client = self._make_local_client()
        self._host = host
        self._cmd_port = cmd_port
        self._pub_port = pub_port

    # ── public API ─────────────────────────────────────────────────

    def load_tttr(self, filename: str) -> dict[str, Any]:
        """Load a TTTR file and return its metadata.

        Parameters
        ----------
        filename : str
            Path to a TTTR file.

        Returns
        -------
        dict
            Metadata dict (n_photons, routing_channels, resolutions, …).

        Raises
        ------
        RuntimeError
            On server error.

        """
        result = self._client.call("pch.load_tttr", {"filename": filename})
        if not result.get("ok", True):
            raise RuntimeError(result.get("error", "load_tttr failed"))
        return result.get("result", {})

    def compute(
        self,
        filename: str,
        channels: list[int] | None = None,
        bin_time_us: float = 100.0,
        micro_time_min: int = 0,
        micro_time_max: int = 65535,
    ) -> dict[str, Any]:
        """Compute a PCH histogram from a TTTR file.

        Parameters
        ----------
        filename : str
            Path to a TTTR file.
        channels : list of int, optional
            Routing channels to include.  *None* uses ``[0, 2]``.
        bin_time_us : float
            Macro-time bin width in microseconds.
        micro_time_min, micro_time_max : int
            Micro-time acceptance window.

        Returns
        -------
        dict
            Result dict (k_vals, p_exp, hist_counts, trace_t, …).

        Raises
        ------
        RuntimeError
            On server error.

        """
        params: dict[str, Any] = {
            "filename": filename,
            "bin_time_us": bin_time_us,
            "micro_time_min": micro_time_min,
            "micro_time_max": micro_time_max,
        }
        if channels is not None:
            params["channels"] = channels
        result = self._client.call("pch.compute", params)
        if not result.get("ok", True):
            raise RuntimeError(result.get("error", "compute failed"))
        return result.get("result", {})

    def fit(
        self,
        k_vals: list[float],
        p_exp: list[float],
        hist_counts: list[int] | None = None,
        total_bins: int | None = None,
        n_components: int = 1,
        initial_epsilons: list[float] | None = None,
        initial_Ns: list[float] | None = None,
        fit_low: int = 0,
        fit_high: int | None = None,
    ) -> dict[str, Any]:
        """Fit a multi-species PCH model.

        Parameters
        ----------
        k_vals : list of float
            Photon-count axis values.
        p_exp : list of float
            Experimental P(k) values.
        hist_counts : list of int, optional
            Raw histogram counts (used for χ² computation).
        total_bins : int, optional
            Total number of macro-time bins.
        n_components : int
            Number of species (1‑10).
        initial_epsilons, initial_Ns : list of float, optional
            Starting values for fit parameters.
        fit_low, fit_high : int
            k-range for fitting.

        Returns
        -------
        dict
            Fit result (epsilons, avg_Ns, fractions, chi2, p_fit, …).

        Raises
        ------
        RuntimeError
            On server error.

        """
        params: dict[str, Any] = {
            "k_vals": k_vals,
            "p_exp": p_exp,
            "n_components": n_components,
            "fit_low": fit_low,
        }
        if hist_counts is not None:
            params["hist_counts"] = hist_counts
        if total_bins is not None:
            params["total_bins"] = total_bins
        if initial_epsilons is not None:
            params["initial_epsilons"] = initial_epsilons
        if initial_Ns is not None:
            params["initial_Ns"] = initial_Ns
        if fit_high is not None:
            params["fit_high"] = fit_high
        result = self._client.call("pch.fit", params)
        if not result.get("ok", True):
            raise RuntimeError(result.get("error", "fit failed"))
        return result.get("result", {})

    # ── ZMQ remote client factory ──────────────────────────────────

    def make_remote_client(self) -> Any:
        """Create a ZMQ client connected to *host*:*cmd_port*.

        Returns
        -------
        ZmqClient
            Connected ZMQ JSON-RPC client.

        """
        from chisurf.server.transport.zmq import ZmqClient

        zmq_client = ZmqClient(
            host=self._host,
            cmd_port=self._cmd_port,
            pub_port=self._pub_port,
        )
        self._client = zmq_client
        return zmq_client

    # ── internals ──────────────────────────────────────────────────

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with PCH services."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from chisurf.plugins.pch.backend.services import register_services

        register_services(dispatcher)
        return InProcessClient(dispatcher)
