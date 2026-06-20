"""PluginClient wrapper for Jordi G-Factor.

GUI code uses this client instead of importing calculations directly.
"""

from __future__ import annotations

from typing import Any

from chisurf.core.plugin.client import InProcessClient


import logging

logger = logging.getLogger(__name__)


class JordiGFactorClient:
    """Client for Jordi G-Factor backend services."""

    def __init__(self, client: Any = None):
        if client is not None:
            self._client = client
        else:
            self._client = self._make_local_client()

    def put_object(self, path: str) -> dict[str, Any]:
        """Upload a file to the MFDB object store via ZMQ RPC."""
        logger.debug("JordiGFactorClient: put_object path=%s", path)
        result = self._client.call("mfdb.objects.put", {"path": path})
        if isinstance(result, dict) and not result.get("ok", True):
            raise RuntimeError(result.get("error", "Unknown error in mfdb.objects.put RPC call"))
        res = result.get("result", result)
        logger.info("JordiGFactorClient: put_object succeeded: %s", res)
        return res


    def calculate(
        self,
        parallel_data: list[float],
        perpendicular_data: list[float],
        region_bounds: list[float] | tuple[float, float],
        decay_shift: float = 0.0,
        use_bg: bool = False,
        bg_region_bounds: list[float] | tuple[float, float] | None = None,
        flip: bool = False,
    ) -> dict[str, Any]:
        """Perform G-factor tail-matching and background subtraction calculation."""
        result = self._client.call(
            "jordi_g_factor.calculate",
            {
                "parallel_data": list(parallel_data),
                "perpendicular_data": list(perpendicular_data),
                "region_bounds": list(region_bounds),
                "decay_shift": decay_shift,
                "use_bg": use_bg,
                "bg_region_bounds": list(bg_region_bounds) if bg_region_bounds is not None else None,
                "flip": flip,
            }
        )
        if isinstance(result, dict) and not result.get("ok", True):
            raise RuntimeError(result.get("error", "Unknown error in G-factor calculate RPC call"))
        return result.get("result", result)

    def perrin_steady_state(self, tau_ns: float, rho_ns: float, r0: float = 0.38) -> float:
        """Calculate steady-state Perrin anisotropy."""
        result = self._client.call(
            "jordi_g_factor.perrin_steady_state",
            {"tau_ns": tau_ns, "rho_ns": rho_ns, "r0": r0}
        )
        if isinstance(result, dict) and not result.get("ok", True):
            raise RuntimeError(result.get("error", "Unknown error in G-factor Perrin RPC call"))
        res = result.get("result", result)
        return float(res.get("r_steady_state", float('nan')))

    def solve_linked_l(self, sp: float, ss: float, g_factor: float, r_target: float) -> float:
        """Solve for the linked l1=l2 mixing parameter."""
        result = self._client.call(
            "jordi_g_factor.solve_linked_l",
            {"sp": sp, "ss": ss, "g_factor": g_factor, "r_target": r_target}
        )
        if isinstance(result, dict) and not result.get("ok", True):
            raise RuntimeError(result.get("error", "Unknown error in G-factor solve_linked_l RPC call"))
        res = result.get("result", result)
    def archive_g_factor(
        self,
        file_path: str,
        parameters: dict[str, Any],
        active_user: str | None = None,
    ) -> dict[str, Any]:
        """Archive a G-factor calculation and reference decay in MFDB."""
        logger.debug("JordiGFactorClient: archive_g_factor file_path=%s", file_path)
        try:
            result = self._client.call(
                "jordi_g_factor.archive_g_factor",
                {
                    "file_path": file_path,
                    "parameters": parameters,
                    "active_user": active_user,
                }
            )
            if isinstance(result, dict) and not result.get("ok", True):
                logger.warning("JordiGFactorClient: archive_g_factor failed: %s", result.get("error"))
                return {"ok": False, "error": result.get("error"), "calibration_id": ""}
            return result.get("result", result)
        except Exception as e:
            logger.warning("JordiGFactorClient: archive_g_factor RPC failed: %s", e)
            return {"ok": False, "error": str(e), "calibration_id": ""}

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with plugin services."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from chisurf.plugins.jordi_g_factor.backend.services import (
            register_services,
        )
        register_services(dispatcher)
        return InProcessClient(dispatcher)
