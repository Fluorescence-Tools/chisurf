"""RPC service registration for img_pixel_mle."""

from __future__ import annotations

import logging
from typing import Any

from ..api.contract import (
    METHOD_ANALYZE,
    METHOD_CONTRACT,
    contract_descriptor,
    service_success,
    service_error,
)

logger = logging.getLogger(__name__)


def register_services(dispatcher: Any) -> None:
    """Register all img_pixel_mle RPC handlers with *dispatcher*."""
    dispatcher.register(METHOD_ANALYZE, _handle_analyze)
    dispatcher.register(METHOD_CONTRACT, _handle_contract)


def _handle_analyze(params: dict[str, Any]) -> dict[str, Any]:
    """Run pixel-wise MLE analysis (headless, no Qt)."""
    try:
        from ..api.models import PixelMleRequest, PixelMleSettings

        settings_raw = params.get("settings") or {}
        # Build settings only from known fields
        known = {f.name for f in PixelMleSettings.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        settings = PixelMleSettings(**{k: v for k, v in settings_raw.items() if k in known})
        request = PixelMleRequest(
            files=params["files"],
            irf_file=params["irf_file"],
            output_dir=params.get("output_dir", ""),
            settings=settings,
        )
        # The actual analysis runs through the existing LifetimeMleAnalysisWizard
        # in headless mode.  For now we return a stub; a full headless path
        # would delegate to the wizard's processing method without the GUI.
        logger.info("img_pixel_mle.analyze.run: %d file(s)", len(request.files))
        return service_success({
            "processed_files": [],
            "output_paths": [],
            "warnings": ["Headless pixel-wise MLE not yet implemented; use the GUI."],
        })
    except Exception as exc:
        logger.exception("img_pixel_mle.analyze.run failed")
        return service_error(exc)


def _handle_contract(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the RPC contract descriptor."""
    return service_success(contract_descriptor())
