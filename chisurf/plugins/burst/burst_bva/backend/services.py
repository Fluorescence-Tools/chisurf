"""ServiceDispatcher-compatible RPC handlers for BVA."""

from __future__ import annotations

import pathlib
from typing import Any

from ..api.contract import (
    METHOD_COMPUTE_BVA,
    METHOD_DESCRIBE_CONTRACT,
    contract_descriptor,
    service_success,
)
from ..api.models import BvaSettings
from ..api.serialization import to_jsonable


def register_services(dispatcher: Any) -> None:
    """Register BVA RPC handlers with a ServiceDispatcher."""
    dispatcher.register(
        METHOD_COMPUTE_BVA,
        lambda params: compute_bva_handler(**params),
    )
    dispatcher.register(
        METHOD_DESCRIBE_CONTRACT,
        lambda params: contract_handler(**(params or {})),
    )


def list_methods() -> dict[str, str]:
    return {
        METHOD_COMPUTE_BVA: "Run Burst Variance Analysis over burst data.",
        METHOD_DESCRIBE_CONTRACT: "Return the BVA workflow contract.",
    }


def compute_bva_handler(
    files: list[str] | None = None,
    analysis_folder: str | None = None,
    pattern: str = "bi4_bur",
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run BVA analysis."""
    try:
        from ..core.computation import (
            read_burst_analysis,
            compute_bva,
            write_bv4_analysis,
            compute_static_bva_line,
        )
        import pandas as pd
        import numpy as np
        import json

        if analysis_folder is None and files:
            analysis_folder = str(pathlib.Path(files[0]).parent)

        af = pathlib.Path(analysis_folder) if analysis_folder else None
        bva_settings = BvaSettings(**(settings or {}))

        df, tttrs = read_burst_analysis(af, bva_settings.file_type, pattern=pattern)
        df_v = compute_bva(
            df, tttrs,
            donor_channels=bva_settings.donor_channels,
            donor_micro_time_ranges=bva_settings.donor_micro_time_ranges,
            acceptor_channels=bva_settings.acceptor_channels,
            acceptor_micro_time_ranges=bva_settings.acceptor_micro_time_ranges,
            minimum_window_length=bva_settings.minimum_window_length,
            number_of_photons_per_slice=bva_settings.number_of_photons_per_slice,
        )

        df_selected = df_v[df_v["Proximity Ratio Std"] > 0.0]

        write_bv4_analysis(df_v, str(af))

        bv4_folder = af / "bv4" if af else pathlib.Path("bv4")
        bv4_folder.mkdir(parents=True, exist_ok=True)
        settings_path = bv4_folder / "bva_settings.json"
        with open(settings_path, "w") as f:
            json.dump(to_jsonable(bva_settings), f, indent=4)

        return service_success({
            "n_bursts_total": int(len(df_v)),
            "n_bursts_valid": int(len(df_selected)),
            "output_paths": {"bv4": str(bv4_folder), "settings": str(settings_path)},
            "files": files or [],
        })

    except Exception as exc:
        from chisurf.server.services import OPERATION_FAILED, service_error
        return service_error(str(exc), error_code=OPERATION_FAILED)


def contract_handler() -> dict[str, Any]:
    return service_success(contract_descriptor())
