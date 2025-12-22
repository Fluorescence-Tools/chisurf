from __future__ import annotations

import chisurf
import chisurf.data
import chisurf.experiments


def set_linearization(
        idx: int = None,
        curve_name: str = None,
        fit: 'chisurf.fitting.fit.FitGroup' = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit

    lin_table = fit.model.corrections.lin_select.datasets[idx]
    for f in fit[fit.selected_fit_index:]:
        f.model.corrections.lintable = chisurf.data.DataCurve(
            x=lin_table.x,
            y=lin_table.y
        )
        f.model.corrections.correct_dnl = True

    lin_name = curve_name
    for f in fit[fit.selected_fit_index:]:
        f.model.corrections.lineEdit.setText(lin_name)
        f.model.corrections.checkBox.setChecked(True)
    fit.update()


def normalize_amplitudes(
        name: str,
        normalize: bool,
        fit: chisurf.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    for f in fit:
        try:
            target = getattr(f.model, name)
        except AttributeError:
            continue
        try:
            setattr(target, "normalize_amplitudes", normalize)
        except Exception:
            continue
        try:
            f.model.update()
        except Exception:
            continue


def absolute_amplitudes(
        name: str,
        use_absolute_amplitudes: bool,
        fit: chisurf.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    for f in fit:
        try:
            target = getattr(f.model, name)
        except AttributeError:
            continue
        try:
            setattr(target, "absolute_amplitudes", use_absolute_amplitudes)
        except Exception:
            continue
        try:
            f.model.update()
        except Exception:
            continue


def remove_component(
        name: str,
        fit: chisurf.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    for f in fit:
        try:
            target = getattr(f.model, name)
        except AttributeError:
            continue
        pop = getattr(target, "pop", None)
        if not callable(pop):
            continue
        try:
            pop()
        except Exception:
            continue
        try:
            f.model.update()
        except Exception:
            continue


def change_irf(
        dataset_idx: int,
        irf_name: str,
        fit: chisurf.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit

    irf = fit.model.convolve.irf_select.datasets[dataset_idx]
    for f in fit[fit.selected_fit_index:]:
        f.model.convolve._irf = chisurf.data.DataCurve(x=irf.x, y=irf.y)
    fit.update()
    for f in fit[fit.selected_fit_index:]:
        f.model.convolve.lineEdit.setText(irf_name)


def add_component(
        name: str,
        fit: chisurf.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    for f in fit:
        try:
            target = getattr(f.model, name)
        except AttributeError:
            continue
        append = getattr(target, "append", None)
        if not callable(append):
            continue
        try:
            append()
        except TypeError:
            # Fallback for append signatures that expect amplitude/lifetime
            try:
                append(amplitude=1.0, lifetime=4.0)
            except Exception:
                continue
        except Exception:
            continue
        try:
            f.model.update()
        except Exception:
            continue


