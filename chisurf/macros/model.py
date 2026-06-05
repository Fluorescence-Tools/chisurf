from __future__ import annotations

import typing

import chisurf
import chisurf.core.data
import chisurf.core.experiments


def set_linearization(
        idx: int = None,
        curve_name: str = None,
        fit: 'chisurf.core.fitting.fit.FitGroup' = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit

    if fit is None or idx is None:
        return

    try:
        lin_table = fit.model.corrections.lin_select.datasets[idx]
    except Exception:
        return

    for f in fit[fit.selected_fit_index:]:
        f.model.corrections.lintable = chisurf.core.data.DataCurve(
            x=lin_table.x,
            y=lin_table.y
        )
        f.model.corrections.correct_dnl = True

    lin_name = curve_name
    for f in fit[fit.selected_fit_index:]:
        f.model.corrections.lineEdit.setText(str(lin_name or ""))
        f.model.corrections.checkBox.setChecked(True)
    fit.update()


def unload_lintable(
        fit: 'chisurf.core.fitting.fit.FitGroup' = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit

    if fit is None:
        return

    for f in fit[fit.selected_fit_index:]:
        try:
            f.model.corrections.unload_lintable()
        except Exception:
            pass
    fit.update()


def set_correction(
        correction_type: str,
        value: typing.Any,
        fit: 'chisurf.core.fitting.fit.FitGroup' = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit

    if fit is None:
        return

    for f in fit[fit.selected_fit_index:]:
        try:
            setattr(f.model.corrections, correction_type, value)
        except Exception:
            pass
    fit.update()


def normalize_amplitudes(
        normalize: bool = True,
        name: str = "amplitudes",
        fit: chisurf.core.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    if fit is None:
        return
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
        use_absolute_amplitudes: bool = True,
        name: str = "amplitudes",
        fit: chisurf.core.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    if fit is None:
        return
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
        fit: chisurf.core.fitting.fit.FitGroup = None
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
        fit: chisurf.core.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit

    irf_curve = None

    try:
        selector_datasets = list(getattr(fit.model.convolve.irf_select, "datasets", []) or [])
    except Exception:
        selector_datasets = []

    if 0 <= int(dataset_idx) < len(selector_datasets):
        irf_curve = selector_datasets[int(dataset_idx)]

    if irf_curve is None:
        try:
            imported = list(getattr(chisurf, "imported_datasets", []) or [])
        except Exception:
            imported = []

        name = str(irf_name or "").strip()
        if name:
            basename = name.replace("\\", "/").split("/")[-1]
            for ds in imported:
                ds_name = str(getattr(ds, "name", "") or "")
                if ds_name == name or ds_name.endswith(name) or ds_name.endswith(basename):
                    irf_curve = ds
                    break

        if irf_curve is None and 0 <= int(dataset_idx) < len(imported):
            irf_curve = imported[int(dataset_idx)]

    if irf_curve is None:
        return

    for f in fit[fit.selected_fit_index:]:
        f.model.convolve._irf = chisurf.core.data.DataCurve(x=irf_curve.x, y=irf_curve.y)

    fit.update()
    for f in fit[fit.selected_fit_index:]:
        f.model.convolve.lineEdit.setText(str(irf_name or getattr(irf_curve, "name", "")))


def unload_irf(
        fit: chisurf.core.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit

    if fit is None:
        return

    for f in fit[fit.selected_fit_index:]:
        try:
            f.model.convolve.unload_irf()
        except Exception:
            pass
        try:
            f.model.convolve.lineEdit.setText("")
        except Exception:
            pass
    try:
        fit.update()
    except Exception:
        pass


def unload_background_curve(
        fit: 'chisurf.core.fitting.fit.FitGroup' = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit

    if fit is None:
        return

    for f in fit[fit.selected_fit_index:]:
        try:
            f.model.nuisance.unload_background_curve()
        except Exception:
            pass
    fit.update()


def update_model(
        fit: 'chisurf.core.fitting.fit.FitGroup' = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit

    if fit is None:
        return

    fit.update()


def add_component(
        name: str,
        fit: chisurf.core.fitting.fit.FitGroup = None
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


def remove_local_fit(
        row: int,
        fit: 'chisurf.core.fitting.fit.FitGroup' = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    if fit is None:
        return
    try:
        fit.remove_local_fit(row)
    except Exception:
        pass


def clear_local_fits(
        fit: 'chisurf.core.fitting.fit.FitGroup' = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    if fit is None:
        return
    try:
        fit.clear_local_fits()
    except Exception:
        pass


def append_global_parameter(
        parameter_name: str,
        fit: 'chisurf.core.fitting.fit.FitGroup' = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    if fit is None:
        return
    try:
        fit.append_global_parameter(parameter_name)
    except Exception:
        pass


def append_fit(
        fit_index: int,
        fit: 'chisurf.core.fitting.fit.FitGroup' = None
) -> None:
    # Local import to ensure symbol resolution in static analyzers and at runtime
    import chisurf as _cs
    if fit is None:
        cs = _cs.cs
        fit = cs.current_fit
    if fit is None:
        return
    try:
        target_fit = _cs.fits[fit_index]
        _cs.logging.info(
            f"macros.model.append_fit: requested fit_index={fit_index}; receiver fit obj type={type(fit).__name__}"
        )
        # Prefer model-level append when available (GlobalFitModel.append_fit expects a Fit)
        model_obj = getattr(fit, "model", None)
        used_path = None
        if hasattr(model_obj, "append_fit") and callable(getattr(model_obj, "append_fit", None)):
            used_path = "fit.model.append_fit"
            _cs.logging.info(
                f"macros.model.append_fit: using {used_path}; target_fit type={type(target_fit).__name__}, name={getattr(target_fit, 'name', None)}"
            )
            model_obj.append_fit(target_fit)
        elif hasattr(fit, "append_fit") and callable(getattr(fit, "append_fit", None)):
            used_path = "fit.append_fit"
            _cs.logging.info(
                f"macros.model.append_fit: using {used_path}; target_fit type={type(target_fit).__name__}, name={getattr(target_fit, 'name', None)}"
            )
            fit.append_fit(target_fit)
        else:
            _cs.logging.warning(
                "macros.model.append_fit: neither fit.model.append_fit nor fit.append_fit is available; no-op"
            )
        if used_path is not None:
            recv = model_obj if used_path.startswith("fit.model") else fit
            _cs.logging.info(
                f"macros.model.append_fit: appended via {used_path} to receiver={type(recv).__name__}"
            )
    except Exception as e:
        try:
            _cs.logging.exception("macros.model.append_fit: exception while appending")
        except Exception:
            pass


