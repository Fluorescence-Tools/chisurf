"""PluginClient wrapper for Burst Selection.

Provides typed convenience methods and shields the GUI from direct API imports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from chisurf.core.plugin.client import InProcessClient


logger = logging.getLogger(__name__)


class BurstSelectionClient:
    """Client for Burst Selection backend services.

    Wraps a PluginClient (typically InProcessClient for local use, or ZmqClient
    for remote/server mode). GUI code should never import ``..api.*`` directly
    when going through this client.
    """

    def __init__(self, client: Any = None):
        """Create a BurstSelection client.

        Parameters
        ----------
        client : PluginClient, optional
            A ``PluginClient`` instance. If ``None``, creates a local
            ``InProcessClient`` with a ServiceDispatcher that has the
            burst_selection services registered.
        """
        if client is not None:
            self._client = client
        else:
            self._client = self._make_local_client()

    # ── public API ─────────────────────────────────────────────────

    def analyze_files(
        self,
        file_paths: list[Path],
        settings: dict[str, Any] | None = None,
        windows: dict[str, list[int]] | None = None,
        detectors: dict[str, dict[str, Any]] | None = None,
        filetype: str | None = None,
        output_dir: str | Path | None = None,
        legacy_output: bool = False,
        legacy_output_folder_name: str | None = None,
        selected_setup: str | None = None,
        legacy_parameters: dict[str, Any] | None = None,
        mfdb: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run burst selection analysis over files.

        Parameters
        ----------
        file_paths : list of Path
            TTTR file paths.
        settings : dict, optional
            Analysis settings as JSON-compatible dict.
        windows : dict, optional
            PIE window definitions (e.g. ``{"prompt": [0, 2048]}``).
        detectors : dict, optional
            Detector definitions with channel and microtime config.
        filetype : str, optional
            Explicit TTTR container/file type from the selected detector setup.
        output_dir : str or Path, optional
            Directory for generated output files.
        legacy_output : bool
            If ``True``, ask the API to write the legacy burstwise layout.
        legacy_output_folder_name : str, optional
            Legacy output folder name.
        selected_setup : str, optional
            Detector setup name stored in legacy metadata.
        legacy_parameters : dict, optional
            Additional legacy metadata fields.
        mfdb : dict, optional
            MFDB archival context.

        Returns
        -------
        dict
            Analysis result with keys: ``files``, ``dataframes``,
            ``metadata``, ``output_paths``.

        Raises
        ------
        RuntimeError
            If the RPC call returned an error, with the error message.

        """
        params: dict[str, Any] = {
            "files": [str(p) for p in file_paths],
        }
        if settings is not None:
            params["settings"] = settings
        if windows is not None:
            params["windows"] = windows
        if detectors is not None:
            params["detectors"] = detectors
        if filetype is not None:
            params["filetype"] = filetype
        if output_dir is not None:
            params["output_dir"] = str(output_dir)
        if legacy_output:
            params["legacy_output"] = True
        if legacy_output_folder_name is not None:
            params["legacy_output_folder_name"] = legacy_output_folder_name
        if selected_setup is not None:
            params["selected_setup"] = selected_setup
        if legacy_parameters is not None:
            params["legacy_parameters"] = legacy_parameters
        if mfdb is not None:
            params["mfdb"] = mfdb
        svc_result = self._client.call(
            "burst_selection.jobs.analyze_files", params
        )
        if not svc_result.get("ok", True):
            err_msg = svc_result.get("error", "unknown error")
            raise RuntimeError(
                f"burst_selection.jobs.analyze_files failed: {err_msg}"
            )
        return svc_result.get("result", {})

    def save_bur(self, dataframe: pd.DataFrame, path: Path) -> None:
        """Save a burst DataFrame as a .bur file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Burst data to save.
        path : Path
            Output path.

        """
        from ..api.io import write_bur
        write_bur(dataframe, path)

    def inspect_bur(self, path: Path) -> dict[str, Any]:
        """Inspect a .bur file.

        Parameters
        ----------
        path : Path
            Path to a .bur file.

        Returns
        -------
        dict
            Inspection result with ``path``, ``n_rows``, ``columns``,
            ``summary`` keys.

        """
        result = self._client.call(
            "burst_selection.results.inspect_bur",
            {"path": str(path)},
        )
        return result.get("result", {})

    def fit_gmm(
        self,
        path: Path,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fit a GMM to features extracted from a .bur file.

        Parameters
        ----------
        path : Path
            Path to a .bur file.
        settings : dict, optional
            GMM settings.

        Returns
        -------
        dict
            GMM fit result.

        """
        params: dict[str, Any] = {"path": str(path)}
        if settings is not None:
            params["settings"] = settings
        result = self._client.call("burst_selection.gmm.fit", params)
        return result.get("result", {})

    def describe_contract(self) -> dict[str, Any]:
        """Return the Burst Selection workflow contract."""
        result = self._client.call("burst_selection.contract.describe", {})
        return result.get("result", {})

    def load_diagnostics(
        self,
        path: Path,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Load diagnostic plot data for a TTTR file.

        This executes the full analysis pipeline locally (not through RPC)
        because TTTR objects are not JSON-serializable and the plotting code
        needs the real TTTR object.

        Parameters
        ----------
        path : Path
            TTTR file path.
        settings : dict
            Analysis settings as JSON-compatible dict.

        Returns
        -------
        dict
            Diagnostic data with ``tttr`` (the TTTR object), ``selected``
            (bool array), ``start_stop`` (Nx2 int array), ``settings``
            (AnalysisSettings) keys.
            Returns empty dict on failure.

        """
        try:
            from ..api.io import load_tttr
            from ..api.models import AnalysisSettings
            from ..api.selection import apply_photon_filters, find_bursts
            from ..api.serialization import settings_from_dict

            analysis_settings = settings_from_dict(settings) if settings else AnalysisSettings()
            if analysis_settings.photon_filter.microtime_ranges is None:
                analysis_settings.photon_filter.microtime_ranges = []
            tttr = load_tttr(str(path))
            selected = apply_photon_filters(
                tttr,
                analysis_settings.photon_filter,
                burst_detection=analysis_settings.burst_detection,
            )
            start_stop = find_bursts(selected)
            return {
                "tttr": tttr,
                "selected": selected,
                "start_stop": start_stop,
                "settings": analysis_settings,
            }
        except Exception:
            logger.exception("Failed to load burst-selection diagnostics for %s", path)
            return {}

    # ── internals ──────────────────────────────────────────────────

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with burst_selection services."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from chisurf.plugins.burst.burst_selection.backend.services import (
            register_services,
        )
        register_services(dispatcher)
        return InProcessClient(dispatcher)
