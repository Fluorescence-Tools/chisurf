"""Burst Selection as a conformant transformer (PRD-16 / PRD-11).

Adapts the plugin's pure ``analyze_request`` to the general transformer contract:
declared typed ports, an ``operation_type`` ("burst_selection") whose parameter
schema lives in the `.dic` (``mfdb_operation_parameter_def``), and a pure
``transform``. No Qt, no DB. The flat declared parameters are mapped back onto the
nested ``AnalysisSettings`` (the inverse of ``extract_burst_parameters``).
"""

from __future__ import annotations

from typing import Any

from chisurf.core.transform import PortSpec, TransformInputs, TransformResult, register_transformer
from chisurf.plugins.burst.burst_selection.api.models import (
    AnalysisRequest,
    AnalysisSettings,
    BurstDetectionSettings,
    CountRateFilterSettings,
    DeltaMacroTimeFilterSettings,
    GMMSettings,
    PhotonFilterSettings,
)
from chisurf.plugins.burst.burst_selection.api.selection import analyze_request

#: Vocabulary value shared with mfdb_operation_parameter_def / register_operation.
OPERATION_TYPE = "burst_selection"


def _channels_from_parameters(parameters: dict) -> list[int]:
    """Build the ``channels`` stream mask from the role-indexed ``channels`` param.

    Accepts the register_operation/compute-spec shape (a list of ``{"value", "role"}``
    entries, ordered by role) or a plain ``list``/iterable of channel numbers. Empty
    when unset (the photon stream is then unmasked, the prior behaviour).
    """
    channels = parameters.get("channels")
    if not isinstance(channels, (list, tuple)):
        return []
    entries: list[tuple[int, int]] = []
    plain: list[int] = []
    for i, entry in enumerate(channels):
        if isinstance(entry, dict) and "value" in entry:
            role = entry.get("role")
            order = int(role) if role not in (None, "") else i
            entries.append((order, int(entry["value"])))
        else:
            plain.append(int(entry))
    if entries:
        return [v for _, v in sorted(entries)]
    return plain


def settings_from_parameters(parameters: dict) -> AnalysisSettings:
    """Build nested ``AnalysisSettings`` from the flat declared parameters.

    The inverse of ``api.mfdb.extract_burst_parameters``; only the declared
    operation-parameter names are consumed. The channel stream mask and the GMM
    determinism fields are part of the reproducible compute spec, so they round-trip
    here too (so a replay reproduces the same photon selection and clustering).
    """
    p = parameters or {}
    gmm = GMMSettings(max_components=int(p.get("gmm_max_components", 10)))
    if p.get("gmm_covariance_type"):
        gmm.covariance_type = str(p["gmm_covariance_type"])
    if p.get("gmm_random_state") is not None:
        gmm.random_state = int(p["gmm_random_state"])
    if p.get("gmm_max_iter") is not None:
        gmm.max_iter = int(p["gmm_max_iter"])
    if p.get("gmm_n_init") is not None:
        gmm.n_init = int(p["gmm_n_init"])
    return AnalysisSettings(
        photon_filter=PhotonFilterSettings(
            channels=_channels_from_parameters(p),
            filter_active=bool(p.get("filter_active", True)),
            count_rate_filter=CountRateFilterSettings(
                n_ph_max=int(p.get("count_rate_n_ph_max", 60)),
                time_window=float(p.get("count_rate_time_window", 1e-3)),
            ),
            delta_macro_time_filter=DeltaMacroTimeFilterSettings(
                dT_min=float(p.get("delta_macro_time_min", 1e-4)),
                dT_max=float(p.get("delta_macro_time_max", 0.15)),
            ),
        ),
        burst_detection=BurstDetectionSettings(
            min_photons=int(p.get("min_photons", 60)),
            photon_window=int(p.get("photon_window", 10)),
            time_window=float(p.get("time_window", 1e-3)),
        ),
        gmm=gmm,
    )


class BurstSelectionTransformer:
    """Conformant transformer over the ``burst_selection`` operation type."""

    transformer_id = "burst_selection"
    operation_type = OPERATION_TYPE
    version = "1.0"
    input_spec = [
        PortSpec(
            # A TTTR file, whether freshly imported (``raw_measurement``) or already
            # micro-time shifted (``processed_data``) — both are valid burst inputs,
            # which lets the canonical raw -> microtime_shift -> burst_selection
            # pipeline (PRD-22) type-check.
            name="raw",
            kinds=("raw_measurement", "processed_data"),
            formats=("ptu", "spc", "ht3", "hdf", "h5"),
        )
    ]
    output_spec = [PortSpec(name="burst_table", kinds=("burst_table", "processed_data"))]

    def transform(self, inputs: TransformInputs, parameters: dict) -> TransformResult:
        """Pure transform: run burst selection over the input files. No Qt, no DB."""
        request = AnalysisRequest(
            files=list(inputs.files),
            settings=settings_from_parameters(parameters),
            filetype=parameters.get("filetype"),
            output_dir=parameters.get("output_dir"),
        )
        result = analyze_request(request)
        outputs: dict[str, Any] = dict(getattr(result, "output_paths", {}) or {})
        warnings = list(getattr(result, "warnings", []) or [])
        return TransformResult(outputs={"burst_table": outputs}, warnings=warnings)


#: Registered instance for discovery / conformance enumeration.
BURST_SELECTION_TRANSFORMER = register_transformer(BurstSelectionTransformer())
