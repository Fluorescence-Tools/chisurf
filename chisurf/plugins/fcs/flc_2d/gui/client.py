"""Client wrapper for 2D-FLCS backend RPC services."""

from __future__ import annotations

from typing import Any

import numpy as np

from chisurf.core.plugin.client import InProcessClient

from ..backend import services


class FlcClient:
    """Typed convenience client for the ``flc2d.*`` service namespace."""

    def __init__(self, client: Any | None = None) -> None:
        """Initialize the client.

        Parameters
        ----------
        client : object, optional
            RPC client implementing ``call(method, params)``. If omitted, an
            in-process dispatcher is created for tests and local GUI use.
        """
        self._client = client if client is not None else self._make_local_client()

    def load_tttr(
        self,
        path: str,
        routing_channels: list[int] | None = None,
        include_arrays: bool = True,
    ) -> dict[str, Any]:
        """Load TTTR metadata and optionally photon arrays via RPC."""
        return self._call(
            services.METHOD_LOAD_TTTR,
            {
                "path": str(path),
                "routing_channels": routing_channels,
                "include_arrays": bool(include_arrays),
            },
        )

    def correlate(
        self,
        macro_times: Any,
        micro_times: Any,
        dT: float,
        ddT: float,
        tMin: float,
        tMax: float,
        logt_imax: int = 100,
        max_bins: int | None = None,
        build_lin: bool = True,
    ) -> dict[str, Any]:
        """Build a 2D-FDC matrix via ``flc2d.correlate``."""
        return self._call(
            services.METHOD_CORRELATE,
            {
                "macro_times": np.asarray(macro_times, dtype=np.int64).tolist(),
                "micro_times": np.asarray(micro_times, dtype=np.int64).tolist(),
                "dT": float(dT),
                "ddT": float(ddT),
                "tMin": float(tMin),
                "tMax": float(tMax),
                "logt_imax": int(logt_imax),
                "max_bins": None if max_bins is None else int(max_bins),
                "build_lin": bool(build_lin),
            },
        )

    def fit(
        self,
        matrix: Any,
        time_axis_ns: Any,
        mode: str = "tikhonov",
        tau_range: tuple[float, float] | list[float] | None = None,
        n_components: int = 24,
        irf: Any | None = None,
        irf_time_ns: Any | None = None,
        reg: float | None = None,
        max_bins: int | None = None,
    ) -> dict[str, Any]:
        """Invert a 2D-FDC matrix via ``flc2d.fit``."""
        return self._call(
            services.METHOD_FIT,
            {
                "matrix": np.asarray(matrix, dtype=float).tolist(),
                "time_axis_ns": np.asarray(time_axis_ns, dtype=float).tolist(),
                "mode": str(mode),
                "tau_range": list(tau_range) if tau_range is not None else None,
                "n_components": int(n_components),
                "irf": None if irf is None else np.asarray(irf, dtype=float).tolist(),
                "irf_time_ns": None
                if irf_time_ns is None
                else np.asarray(irf_time_ns, dtype=float).tolist(),
                "reg": reg,
                "max_bins": None if max_bins is None else int(max_bins),
            },
        )

    def lifetime_spectrum(
        self,
        micro_times: Any,
        n_microtime_bins: int,
        micro_time_resolution_ns: float,
        tau_range: tuple[float, float] | list[float] | None = None,
        n_components: int = 40,
        irf: Any | None = None,
        irf_time_ns: Any | None = None,
        method: str = "nnls",
        reg: float | None = None,
    ) -> dict[str, Any]:
        """Resolve a 1D lifetime distribution via RPC."""
        return self._call(
            services.METHOD_LIFETIME,
            {
                "micro_times": np.asarray(micro_times, dtype=np.int64).tolist(),
                "n_microtime_bins": int(n_microtime_bins),
                "micro_time_resolution_ns": float(micro_time_resolution_ns),
                "tau_range": list(tau_range) if tau_range is not None else None,
                "n_components": int(n_components),
                "irf": None if irf is None else np.asarray(irf, dtype=float).tolist(),
                "irf_time_ns": None
                if irf_time_ns is None
                else np.asarray(irf_time_ns, dtype=float).tolist(),
                "method": str(method),
                "reg": reg,
            },
        )

    def lifetime_lcurve(
        self,
        micro_times: Any,
        n_microtime_bins: int,
        micro_time_resolution_ns: float,
        tau_range: tuple[float, float] | list[float] | None = None,
        n_components: int = 40,
        irf: Any | None = None,
        irf_time_ns: Any | None = None,
        method: str = "nnls",
    ) -> dict[str, Any]:
        """Compute L-curve diagnostics via RPC."""
        return self._call(
            services.METHOD_LCURVE,
            {
                "micro_times": np.asarray(micro_times, dtype=np.int64).tolist(),
                "n_microtime_bins": int(n_microtime_bins),
                "micro_time_resolution_ns": float(micro_time_resolution_ns),
                "tau_range": list(tau_range) if tau_range is not None else None,
                "n_components": int(n_components),
                "irf": None if irf is None else np.asarray(irf, dtype=float).tolist(),
                "irf_time_ns": None
                if irf_time_ns is None
                else np.asarray(irf_time_ns, dtype=float).tolist(),
                "method": str(method),
            },
        )

    def describe_contract(self) -> dict[str, Any]:
        """Return the backend contract descriptor."""
        return self._call(services.METHOD_DESCRIBE, {})

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call an RPC method and unwrap service errors."""
        result = self._client.call(method, params)
        if not result.get("ok", True):
            raise RuntimeError(result.get("error", f"{method} failed"))
        return result.get("result", {})

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create an in-process RPC client for local GUI/test use."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        services.register_services(dispatcher)
        return InProcessClient(dispatcher)
