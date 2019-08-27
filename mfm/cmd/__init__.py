import mfm
import mfm.experiments.data
from PyQt5.QtWidgets import QMainWindow


def change_irf(
        dataset_idx: int,
        irf_name: str
) -> None:
    cs = mfm.cs
    irf = cs.current_fit.model.convolve.irf_select.datasets[dataset_idx]
    for f in cs.current_fit[cs.current_fit._selected_fit:]:
        f.model.convolve._irf = mfm.experiments.data.DataCurve(x=irf.x, y=irf.y)
    cs.current_fit.update()
    current_fit = mfm.cs.current_fit
    for f in current_fit[current_fit._selected_fit:]:
        f.model.convolve.lineEdit.setText(irf_name)


def add_lifetime(
        name: str
) -> None:
    cs = mfm.cs
    for f in cs.current_fit:
        eval("f.model.%s.append()" % name)
        f.model.update()


def remove_lifetime(
        name: str
) -> None:
    cs = mfm.cs
    for f in cs.current_fit:
        eval("f.model.%s.pop()" % name)
        f.model.update()


def normalize_lifetime_amplitudes(
        normalize: bool
) -> None:
    cs = mfm.cs
    cs.current_fit.models.lifetimes.normalize_amplitudes = normalize
    cs.current_fit.update()


def absolute_amplitudes(
        use_absolute_amplitudes: bool
) -> None:
    cs = mfm.cs
    cs.current_fit.models.lifetimes.absolute_amplitudes = use_absolute_amplitudes
    cs.current_fit.update()

