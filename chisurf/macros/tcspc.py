import chisurf
import chisurf.experiments


def set_linearization(
        idx: int = None,
        curve_name: str = None,
):
    cs = chisurf.cs
    lin_table = cs.current_fit.model.corrections.lin_select.datasets[idx]
    for f in cs.current_fit[cs.current_fit.selected_fit_index:]:
        f.model.corrections.lintable = chisurf.experiments.data.DataCurve(
            x=lin_table.x,
            y=lin_table.y
        )
        f.model.corrections.correct_dnl = True

    lin_name = curve_name
    for f in cs.current_fit[cs.current_fit.selected_fit_index:]:
        f.model.corrections.lineEdit.setText(lin_name)
        f.model.corrections.checkBox.setChecked(True)
    cs.current_fit.update()


def normalize_lifetime_amplitudes(
        normalize: bool
) -> None:
    cs = chisurf.cs
    cs.current_fit.models.lifetimes.normalize_amplitudes = normalize
    cs.current_fit.update()


def remove_lifetime(
        name: str
) -> None:
    cs = chisurf.cs
    for f in cs.current_fit:
        eval("f.model.%s.pop()" % name)
        f.model.update()


def change_irf(
        dataset_idx: int,
        irf_name: str
) -> None:
    cs = chisurf.cs
    irf = cs.current_fit.model.convolve.irf_select.datasets[dataset_idx]
    for f in cs.current_fit[cs.current_fit.selected_fit_index:]:
        f.model.convolve._irf = chisurf.experiments.data.DataCurve(
            x=irf.x,
            y=irf.y
        )
    cs.current_fit.update()
    current_fit = chisurf.cs.current_fit
    for f in current_fit[current_fit.selected_fit_index:]:
        f.model.convolve.lineEdit.setText(irf_name)


def add_lifetime(
        name: str
) -> None:
    cs = chisurf.cs
    for f in cs.current_fit:
        eval("f.model.%s.append()" % name)
        f.model.update()


def absolute_amplitudes(
        use_absolute_amplitudes: bool
) -> None:
    cs = chisurf.cs
    cs.current_fit.models.lifetimes.absolute_amplitudes = use_absolute_amplitudes
    cs.current_fit.update()
