"""ServiceDispatcher-compatible RPC handlers for Burst Selection.

These are thin adapters that accept JSON-compatible params, delegate to
the ``api/`` and ``core/`` layers, and return JSON-safe results.
"""

from __future__ import annotations

from typing import Any

from ..api.features import extract_features, fit_gmm
from ..api.io import load_tttr
from ..api.contract import (
    METHOD_ANALYZE_FILES,
    METHOD_DESCRIBE_CONTRACT,
    METHOD_FIT_GMM,
    METHOD_INSPECT_BUR,
    METHOD_LOAD_DIAGNOSTICS,
    analysis_request_from_payload,
    contract_descriptor,
    service_success,
)
from ..api.models import AnalysisSettings
from ..api.selection import analyze_request
from ..api.serialization import settings_from_dict
from ..api.stats import summarize_dataframes


def register_services(dispatcher: Any) -> None:
    """Register Burst Selection RPC handlers with a ServiceDispatcher.

    Parameters
    ----------
    dispatcher : ServiceDispatcher
        The server's service dispatcher.

    Registers both new dotted names and legacy snake_case aliases for
    backward compatibility during migration.

    """
    # New dotted names (canonical)
    dispatcher.register(
        METHOD_ANALYZE_FILES,
        lambda params: analyze_files_handler(**params),
    )
    dispatcher.register(
        METHOD_INSPECT_BUR,
        lambda params: inspect_bur_handler(**params),
    )
    dispatcher.register(
        METHOD_FIT_GMM,
        lambda params: fit_gmm_handler(**params),
    )
    dispatcher.register(
        METHOD_LOAD_DIAGNOSTICS,
        lambda params: diagnostics_handler(**params),
    )
    dispatcher.register(
        METHOD_DESCRIBE_CONTRACT,
        lambda params: contract_handler(**(params or {})),
    )
    # Legacy aliases (backward compat, will be removed after migration)
    dispatcher.register(
        "burst_selection.analyze_files",
        lambda params: analyze_files_handler(**params),
    )
    dispatcher.register(
        "burst_selection.inspect_bur",
        lambda params: inspect_bur_handler(**params),
    )
    dispatcher.register(
        "burst_selection.fit_gmm_from_bur",
        lambda params: fit_gmm_handler(**params),
    )


def list_methods() -> dict[str, str]:
    """Return the Burst Selection RPC method catalogue (includes legacy aliases)."""
    return {
        # New dotted names (canonical)
        METHOD_ANALYZE_FILES: "Run burst selection analysis over TTTR files.",
        METHOD_INSPECT_BUR: "Inspect a saved ChiSurf .bur file.",
        METHOD_FIT_GMM: "Fit a GMM to features extracted from a .bur file.",
        METHOD_LOAD_DIAGNOSTICS: "Run photon filtering and burst finding for diagnostic plots.",
        METHOD_DESCRIBE_CONTRACT: "Return the Burst Selection workflow contract.",
        # Legacy aliases
        "burst_selection.analyze_files": "[legacy] Run burst selection analysis over TTTR files.",
        "burst_selection.inspect_bur": "[legacy] Inspect a saved ChiSurf .bur file.",
        "burst_selection.fit_gmm_from_bur": "[legacy] Fit a GMM to features extracted from a .bur file.",
    }


def analyze_files_handler(
    files: list[str],
    filetype: str | None = None,
    windows: dict[str, list[int]] | None = None,
    detectors: dict[str, dict[str, Any]] | None = None,
    settings: dict[str, Any] | None = None,
    output_dir: str | None = None,
    legacy_output: bool = False,
    legacy_output_folder_name: str | None = None,
    selected_setup: str | None = None,
    legacy_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Burst Selection analysis over TTTR files.

    Parameters
    ----------
    files : list of str
        TTTR file paths.
    filetype : str, optional
        Explicit TTTR file type.
    windows : dict, optional
        PIE window definitions.
    detectors : dict, optional
        Detector definitions.
    settings : dict, optional
        JSON-compatible analysis settings.
    output_dir : str, optional
        Output directory for generated ``.bur`` files.
    legacy_output : bool
        If ``True``, write the legacy burstwise folder layout.
    legacy_output_folder_name : str, optional
        Legacy output folder name. If omitted, the API derives it from settings.
    selected_setup : str, optional
        Detector setup name stored in the legacy Info metadata.
    legacy_parameters : dict, optional
        Additional legacy Info metadata.

    Returns
    -------
    dict
        JSON-serializable ServiceResult.

    """
    try:
        request = analysis_request_from_payload(
            {
                "files": files,
                "filetype": filetype,
                "windows": windows or {},
                "detectors": detectors or {},
                "settings": settings or {},
                "output_dir": output_dir,
                "legacy_output": legacy_output,
                "legacy_output_folder_name": legacy_output_folder_name,
                "selected_setup": selected_setup,
                "legacy_parameters": legacy_parameters or {},
            }
        )
        result = analyze_request(request)
        return service_success(result)
    except Exception as exc:
        from chisurf.server.services import OPERATION_FAILED, service_error

        return service_error(str(exc), error_code=OPERATION_FAILED)


def contract_handler() -> dict[str, Any]:
    """Return the Burst Selection workflow contract descriptor."""
    return service_success(contract_descriptor())


def inspect_bur_handler(path: str) -> dict[str, Any]:
    """Inspect a ``.bur`` file and return summary statistics.

    Parameters
    ----------
    path : str
        Path to a ``.bur`` file.

    Returns
    -------
    dict
        JSON-serializable ServiceResult.

    """
    try:
        import pandas as pd

        df = pd.read_csv(path, sep="\t")
        return {
            "ok": True,
            "result": {
                "path": path,
                "n_rows": int(len(df)),
                "columns": list(df.columns),
                "summary": summarize_dataframes([df]),
            },
        }
    except Exception as exc:
        from chisurf.server.services import OPERATION_FAILED, service_error

        return service_error(str(exc), error_code=OPERATION_FAILED)


def fit_gmm_handler(
    path: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a GMM to features extracted from a ``.bur`` file.

    Parameters
    ----------
    path : str
        Path to a ``.bur`` file.
    settings : dict, optional
        GMM settings.

    Returns
    -------
    dict
        JSON-serializable ServiceResult.

    """
    try:
        import pandas as pd

        df = pd.read_csv(path, sep="\t")
        gmm_settings = settings_from_dict({"gmm": settings or {}}).gmm
        features = extract_features([df])
        return {
            "ok": True,
            "result": {
                "path": path,
                "features": features.to_dict(orient="records"),
                "gmm": fit_gmm(features, gmm_settings),
            },
        }
    except Exception as exc:
        from chisurf.server.services import OPERATION_FAILED, service_error

        return service_error(str(exc), error_code=OPERATION_FAILED)


def diagnostics_handler(
    path: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run photon filtering and burst finding for diagnostic plots.

    Parameters
    ----------
    path : str
        TTTR file path.
    settings : dict, optional
        JSON-compatible analysis settings.

    Returns
    -------
    dict
        JSON-serializable ServiceResult with ``selected`` (list of int indices)
        and ``start_stop`` (list of ``[start, stop]`` pairs).

    """
    try:
        import numpy as np

        analysis_settings = (
            settings_from_dict(settings) if settings else AnalysisSettings()
        )
        tttr = load_tttr(path)
        from ..api.selection import apply_photon_filters, find_bursts

        selected = apply_photon_filters(
            tttr,
            analysis_settings.photon_filter,
            burst_detection=analysis_settings.burst_detection,
        )
        start_stop = find_bursts(selected)
        selected_indices = np.flatnonzero(selected).tolist()
        start_stop_list = start_stop.tolist() if start_stop.size > 0 else []
        return {
            "ok": True,
            "result": {
                "path": path,
                "selected": selected_indices,
                "start_stop": start_stop_list,
                "n_selected": int(np.count_nonzero(selected)),
                "n_photons": int(len(tttr)),
            },
        }
    except Exception as exc:
        from chisurf.server.services import OPERATION_FAILED, service_error

        return service_error(str(exc), error_code=OPERATION_FAILED)
