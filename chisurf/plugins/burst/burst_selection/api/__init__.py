"""Burst Selection API package.

The API package is the canonical non-GUI integration surface.  GUI, CLI and
RPC layers normalize their inputs to :class:`AnalysisRequest` and return
JSON-compatible :class:`AnalysisResult` payloads.
"""

from .contract import (
    CONTRACT_VERSION,
    METHOD_ANALYZE_FILES,
    METHOD_DESCRIBE_CONTRACT,
    METHOD_FIT_GMM,
    METHOD_INSPECT_BUR,
    METHOD_LOAD_DIAGNOSTICS,
    PLUGIN_ID,
    analysis_request_from_payload,
    analysis_request_to_payload,
    analysis_result_to_payload,
    contract_descriptor,
    service_success,
)
from .features import extract_features, fit_gmm
from .io import (
    get_unique_folder_path,
    load_tttr,
    read_bur,
    write_bur,
    write_hdf5,
    zip_output_folder,
)
from .models import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisSettings,
    BurstDetectionSettings,
    CountRateFilterSettings,
    DeltaMacroTimeFilterSettings,
    GMMSettings,
    PhotonFilterSettings,
)
from .selection import (
    analyze_file,
    analyze_request,
    apply_photon_filters,
    find_bursts,
    summarize_bursts,
)
from .serialization import from_jsonable, settings_from_dict, to_jsonable
from .stats import summarize_dataframes

__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisSettings",
    "BurstDetectionSettings",
    "CONTRACT_VERSION",
    "CountRateFilterSettings",
    "DeltaMacroTimeFilterSettings",
    "GMMSettings",
    "METHOD_ANALYZE_FILES",
    "METHOD_DESCRIBE_CONTRACT",
    "METHOD_FIT_GMM",
    "METHOD_INSPECT_BUR",
    "METHOD_LOAD_DIAGNOSTICS",
    "PLUGIN_ID",
    "analysis_request_from_payload",
    "analysis_request_to_payload",
    "analysis_result_to_payload",
    "PhotonFilterSettings",
    "analyze_file",
    "analyze_request",
    "apply_photon_filters",
    "contract_descriptor",
    "extract_features",
    "find_bursts",
    "fit_gmm",
    "from_jsonable",
    "get_unique_folder_path",
    "load_tttr",
    "read_bur",
    "settings_from_dict",
    "service_success",
    "summarize_bursts",
    "summarize_dataframes",
    "to_jsonable",
    "write_bur",
    "write_hdf5",
    "zip_output_folder",
]
