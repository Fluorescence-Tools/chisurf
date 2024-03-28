"""Collection of macros / functions that control parsed models."""
import chisurf
import chisurf.experiments


def change_model(function_str: str, fit_idx: int = None) -> None:
    cs = chisurf.cs
    if fit_idx is None:
        fit = cs.current_fit
    else:
        fit = chisurf.fits[fit_idx]
    for f in fit:
        f.model.func = f"{function_str}"
