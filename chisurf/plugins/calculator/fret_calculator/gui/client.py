"""PluginClient wrapper for the FRET Calculator.

GUI code should use this client instead of importing ``..api`` or ``..core``
directly.
"""

from __future__ import annotations

from typing import Any

from chisurf.core.plugin.client import InProcessClient


class FretCalculatorClient:
    """Client for the FRET Calculator backend services."""

    def __init__(self, client: Any = None):
        if client is not None:
            self._client = client
        else:
            self._client = self._make_local_client()

    def compute_fret(
        self,
        R: float,
        R0: float,
        tau0: float,
        kappa2: float = 0.667,
        sigma: float = 0.0,
        distribution: str = "gaussian",
    ) -> dict[str, Any]:
        """Compute FRET parameters from distance.

        Returns
        -------
        dict
            ``{"ok": True, "result": {...}}`` or ``{"ok": False, "error": ...}``
        """
        return self._client.call(
            "fret_calculator.fret.compute",
            {
                "R": R,
                "R0": R0,
                "tau0": tau0,
                "kappa2": kappa2,
                "sigma": sigma,
                "distribution": distribution,
            },
        )

    def compute_fret_from_efficiency(
        self,
        E: float,
        R0: float,
        tau0: float,
        kappa2: float = 0.667,
    ) -> dict[str, Any]:
        """Compute FRET parameters from efficiency."""
        return self._client.call(
            "fret_calculator.fret.compute_from_efficiency",
            {"E": E, "R0": R0, "tau0": tau0, "kappa2": kappa2},
        )

    def compute_fret_from_lifetime(
        self,
        tau_DA: float,
        R0: float,
        tau0: float,
        kappa2: float = 0.667,
    ) -> dict[str, Any]:
        """Compute FRET parameters from lifetime."""
        return self._client.call(
            "fret_calculator.fret.compute_from_lifetime",
            {"tau_DA": tau_DA, "R0": R0, "tau0": tau0, "kappa2": kappa2},
        )

    def compute_fret_from_rate(
        self,
        kFRET: float,
        R0: float,
        tau0: float,
        kappa2: float = 0.667,
    ) -> dict[str, Any]:
        """Compute FRET parameters from rate constant."""
        return self._client.call(
            "fret_calculator.fret.compute_from_rate",
            {"kFRET": kFRET, "R0": R0, "tau0": tau0, "kappa2": kappa2},
        )

    def compute_homo_fret(
        self,
        t_RM: float,
        rho: float,
        tau0: float,
        R0: float,
    ) -> dict[str, Any]:
        """Compute homoFRET exchange rate and effective distance."""
        return self._client.call(
            "fret_calculator.homo.compute",
            {"t_RM": t_RM, "rho": rho, "tau0": tau0, "R0": R0},
        )

    def homo_backmap(
        self,
        R_DA: float,
        R0: float,
        tau0: float,
        rho: float,
    ) -> dict[str, Any]:
        """Back-map homoFRET distance to anisotropy relaxation time."""
        return self._client.call(
            "fret_calculator.homo.backmap",
            {"R_DA": R_DA, "R0": R0, "tau0": tau0, "rho": rho},
        )

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with fret_calculator services."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from chisurf.plugins.calculator.fret_calculator.backend.services import (
            register_services,
        )
        register_services(dispatcher)
        return InProcessClient(dispatcher)
