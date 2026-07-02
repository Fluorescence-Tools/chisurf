"""RPC service registration for the CLSM plugin."""

from __future__ import annotations

import logging
from typing import Any

from ..api.contract import (
    METHOD_CONTRACT,
    METHOD_DECAY,
    METHOD_FRC,
    METHOD_INFO,
    METHOD_REPRESENTATION,
    METHOD_SETUPS,
    contract_descriptor,
    service_error,
    service_success,
)

logger = logging.getLogger(__name__)


def register_services(dispatcher: Any) -> None:
    """Register all ``clsm.*`` RPC handlers with *dispatcher*."""
    dispatcher.register(METHOD_SETUPS, _handle_setups)
    dispatcher.register(METHOD_INFO, _handle_info)
    dispatcher.register(METHOD_REPRESENTATION, _handle_representation)
    dispatcher.register(METHOD_DECAY, _handle_decay)
    dispatcher.register(METHOD_FRC, _handle_frc)
    dispatcher.register(METHOD_CONTRACT, _handle_contract)


def _guard(method: str, fn, params: dict[str, Any] | None):
    """Run *fn* with *params* and wrap the result/exception in an envelope."""
    try:
        return service_success(fn(**(params or {})))
    except Exception as exc:  # pragma: no cover - exercised via tests indirectly
        logger.exception("%s failed", method)
        return service_error(exc)


def _handle_setups(params: dict[str, Any] | None = None) -> dict[str, Any]:
    from ..api.clsm import list_setups

    return _guard(METHOD_SETUPS, lambda: list_setups(), {})


def _handle_info(params: dict[str, Any]) -> dict[str, Any]:
    from ..api.clsm import image_info

    return _guard(METHOD_INFO, image_info, params)


def _handle_representation(params: dict[str, Any]) -> dict[str, Any]:
    from ..api.clsm import compute_representation

    return _guard(METHOD_REPRESENTATION, compute_representation, params)


def _handle_decay(params: dict[str, Any]) -> dict[str, Any]:
    from ..api.clsm import extract_decay

    return _guard(METHOD_DECAY, extract_decay, params)


def _handle_frc(params: dict[str, Any]) -> dict[str, Any]:
    from ..api.clsm import compute_frc

    return _guard(METHOD_FRC, compute_frc, params)


def _handle_contract(params: dict[str, Any] | None = None) -> dict[str, Any]:
    return service_success(contract_descriptor())
