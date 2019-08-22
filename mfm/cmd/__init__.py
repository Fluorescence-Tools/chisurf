import mfm.experiments.data
from PyQt5.QtWidgets import QMainWindow


def change_irf(
        dataset_idx: int,
        cs: QMainWindow
):
    irf = cs.current_fit.model.convolve.irf_select.datasets[dataset_idx]
    for f in cs.current_fit[cs.current_fit._selected_fit:]:
       f.model.convolve._irf = mfm.experiments.data.DataCurve(x=irf.x, y=irf.y)
    cs.current_fit.update()

