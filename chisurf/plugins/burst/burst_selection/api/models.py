"""Data models for the Burst Selection API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class BurstFilterMode(str, Enum):
    """Supported photon/burst filter modes."""

    COUNT_RATE = "count_rate"
    BURST = "burst"


@dataclass
class CountRateFilterSettings:
    """Count-rate photon filter settings."""

    n_ph_max: int = 60
    time_window: float = 1e-3
    invert: bool = False


@dataclass
class DeltaMacroTimeFilterSettings:
    """Delta macro-time filter settings."""

    dT_min: float = 1e-4
    dT_max: float = 0.15
    dT_min_active: bool = True
    dT_max_active: bool = True


@dataclass
class PhotonFilterSettings:
    """Photon pre-filter settings used before burst detection."""

    channels: list[int] = field(default_factory=list)
    microtime_ranges: list[tuple[int, int]] = field(default_factory=list)
    filter_active: bool = True
    used_filter: BurstFilterMode = BurstFilterMode.COUNT_RATE
    count_rate_filter: CountRateFilterSettings = field(default_factory=CountRateFilterSettings)
    delta_macro_time_filter: DeltaMacroTimeFilterSettings = field(default_factory=DeltaMacroTimeFilterSettings)
    invert_filter: bool = False
    max_gap: int = 4
    use_gap_fill: bool = True

    def __post_init__(self) -> None:
        """Normalize unset optional collections from GUI/RPC callers."""
        if self.channels is None:
            self.channels = []
        if self.microtime_ranges is None:
            self.microtime_ranges = []


@dataclass
class BurstDetectionSettings:
    """Burst search settings."""

    min_photons: int = 60
    photon_window: int = 10
    time_window: float = 1e-3


@dataclass
class GMMSettings:
    """Gaussian mixture model fitting settings."""

    covariance_type: str = "full"
    random_state: int = 42
    max_iter: int = 300
    n_init: int = 10
    tol: float = 1e-3
    max_components: int = 10
    reg_covar: float = 1e-6
    auto_components: bool = False


@dataclass
class AnalysisSettings:
    """Complete burst-selection analysis settings.

    Attributes
    ----------
    photon_filter : PhotonFilterSettings
        Photon filtering settings applied before burst detection.
    burst_detection : BurstDetectionSettings
        Burst search parameters.
    gmm : GMMSettings
        Optional Gaussian mixture model settings.
    output_formats : list of str
        Output formats to write. Supported values are ``"bur"`` and
        ``"hdf5"``.
    zip_output : bool
        Whether legacy output folders should be zipped after processing.
    remove_folder : bool
        Whether the unzipped legacy output folder should be removed after
        zipping.

    """

    photon_filter: PhotonFilterSettings = field(default_factory=PhotonFilterSettings)
    burst_detection: BurstDetectionSettings = field(default_factory=BurstDetectionSettings)
    gmm: GMMSettings = field(default_factory=GMMSettings)
    output_formats: list[str] = field(default_factory=lambda: ["bur"])
    zip_output: bool = False
    remove_folder: bool = False


@dataclass
class AnalysisRequest:
    """Workflow input object for burst-selection analysis.

    This is the canonical input passed from GUI, CLI, RPC and future node-based
    workflow runners into the pure API layer.

    Attributes
    ----------
    files : list of str
        TTTR files to analyze.
    filetype : str, optional
        Explicit tttrlib file type. ``None`` lets tttrlib infer the type.
    windows : dict
        PIE/microtime window definitions, for example
        ``{"prompt": (0, 2048)}``.
    detectors : dict
        Detector setup definitions used for burst summary columns.
    settings : AnalysisSettings
        Analysis and output settings.
    output_dir : str, optional
        Directory for direct output files.
    legacy_output : bool
        If ``True``, write the legacy ``burstwise...`` output folder layout.
    legacy_output_folder_name : str, optional
        Override for the legacy output folder name.
    selected_setup : str, optional
        Detector setup label stored in output metadata.
    legacy_parameters : dict
        Additional metadata written to legacy ``Info`` files.

    """

    files: list[str]
    filetype: str | None = None
    windows: dict[str, tuple[int, int]] = field(default_factory=dict)
    detectors: dict[str, dict[str, Any]] = field(default_factory=dict)
    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    output_dir: str | None = None
    legacy_output: bool = False
    legacy_output_folder_name: str | None = None
    selected_setup: str | None = None
    legacy_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Workflow output object returned by burst-selection analysis.

    Attributes
    ----------
    files : list of str
        Files included in the analysis.
    dataframes : dict
        Per-file burst rows as JSON-compatible records.
    feature_dataframe : dict, optional
        Optional combined feature table.
    gmm_fit : dict, optional
        Optional GMM result.
    output_paths : dict
        Paths written by the analysis, keyed by output role.
    metadata : dict
        Counts and additional run metadata.

    """

    files: list[str]
    dataframes: dict[str, dict[str, Any]] = field(default_factory=dict)
    feature_dataframe: dict[str, Any] | None = None
    gmm_fit: dict[str, Any] | None = None
    output_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass to a dictionary with tuple-safe serialization."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_to_dict(val) for key, val in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): dataclass_to_dict(val) for key, val in value.items()}
    if isinstance(value, Enum):
        return value.value
    return value
