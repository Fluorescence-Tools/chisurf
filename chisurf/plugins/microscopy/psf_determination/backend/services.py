"""RPC service registration for psf_determination."""

from __future__ import annotations

import logging
from typing import Any

from ..api.contract import (
    METHOD_FIT,
    METHOD_CONTRACT,
    contract_descriptor,
    service_success,
    service_error,
)

logger = logging.getLogger(__name__)


def register_services(dispatcher: Any) -> None:
    """Register all psf_determination RPC handlers with *dispatcher*."""
    dispatcher.register(METHOD_FIT, _handle_fit)
    dispatcher.register(METHOD_CONTRACT, _handle_contract)


def _handle_fit(params: dict[str, Any]) -> dict[str, Any]:
    """Detect beads and fit 3-D Gaussian PSF to all of them."""
    try:
        stack_path = params["stack_path"]
        pixel_nm = float(params.get("pixel_size_nm", 100.0))
        z_step_nm = float(params.get("z_step_nm", 200.0))
        roi_xy = int(params.get("roi_xy", 15))
        roi_z = int(params.get("roi_z", 15))
        pixels_per_frame = int(params.get("pixels_per_frame", 20))
        min_distance = float(params.get("min_distance", 5.0))

        try:
            import imageio.v2 as imageio  # type: ignore[import]
        except ImportError:
            import imageio  # type: ignore[import, no-redef]
        import numpy as np

        data = imageio.imread(stack_path)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr[..., 0][np.newaxis, ...]

        from ..api.psf import detect_beads, fit_all_beads

        beads = detect_beads(arr, roi_xy=roi_xy, roi_z=roi_z,
                             pixels_per_frame=pixels_per_frame, min_distance=min_distance)
        results = fit_all_beads(arr, beads, roi_xy, roi_z, pixel_nm, z_step_nm)

        return service_success({
            "n_beads": len(beads),
            "fits": results,
        })
    except Exception as exc:
        logger.exception("psf_determination.fit.run failed")
        return service_error(exc)


def _handle_contract(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the RPC contract descriptor."""
    return service_success(contract_descriptor())
