import chisurf
import chisurf.data
import chisurf.fitting
import chisurf.experiments


def set_linearization(
        idx: int = None,
        curve_name: str = None,
        fit: chisurf.fitting.fit.FitGroup = None
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
        exec(f"f.model.{name}.normalize_amplitudes = {normalize}")
        f.model.update()


def absolute_amplitudes(
        name: str,
        use_absolute_amplitudes: bool,
        fit: chisurf.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    for f in fit:
        exec(f"f.model.{name}.absolute_amplitudes = {use_absolute_amplitudes}")
        f.model.update()

def remove_component(
        name: str,
        fit: chisurf.fitting.fit.FitGroup = None
) -> None:
    if fit is None:
        cs = chisurf.cs
        fit = cs.current_fit
    for f in fit:
        eval(f"f.model.{name}.pop()")
        f.model.update()


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
        eval(f"f.model.{name}.append()")
        f.model.update()


