from __future__ import annotations

from typing import Any, Optional

from chisurf.server.services import (
    ServiceResult,
    service_error,
    NOT_FOUND,
    OPERATION_FAILED,
    _resolve_fit,
)
from chisurf.server.session import SessionState


def model_finalize(
    state: SessionState,
    model_name: Optional[str] = None,
    fit_index: int = 0,
    fit_uid: Optional[str] = None,
) -> ServiceResult:
    """Finalize a model on either a specific fit or by model name."""
    fit, _ = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    try:
        model = getattr(fit, "model", None)
        if model is None:
            return service_error("fit has no model", error_code=OPERATION_FAILED)
        if hasattr(model, "finalize"):
            model.finalize()
            return {"ok": True}
        return service_error("model has no finalize method", error_code=OPERATION_FAILED)
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def model_set_parse_function(
    state: SessionState,
    parse_function: str,
    model_name: Optional[str] = None,
    fit_index: int = 0,
    fit_uid: Optional[str] = None,
) -> ServiceResult:
    """Set the parse function on a model.

    The parse function is a string that defines how model expressions
    are parsed. It is set on the model's ``parse_function`` attribute.
    """
    fit, _ = _resolve_fit(state, fit_index, fit_uid)
    if fit is None:
        return service_error("fit not found", error_code=NOT_FOUND)
    try:
        model = getattr(fit, "model", None)
        if model is None:
            return service_error("fit has no model", error_code=OPERATION_FAILED)
        setattr(model, "parse_function", parse_function)
        return {"ok": True}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)