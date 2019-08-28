from __future__ import annotations
from typing import List

import mfm
import mfm.experiments.data
from PyQt5.QtWidgets import QMainWindow


def add_fit(
        dataset_indices: List[int] = None,
        model_name: str = None,
        **kwargs
):
    cs = mfm.cs
    if dataset_indices is None:
        dataset_indices = [cs.dataset_selector.selected_curve_index]
    datasets = [cs.dataset_selector.datasets[i] for i in dataset_indices]

    if model_name is None:
        model_name = cs.current_model_name

    model_names = datasets[0].experiment.model_names
    model_class = datasets[0].experiment.model_classes[0]
    for model_idx, mn in enumerate(model_names):
        if mn == model_name:
            model_class = datasets[0].experiment.model_classes[model_idx]
            break

    for data_set in datasets:
        if data_set.experiment is datasets[0].experiment:
            if not isinstance(data_set, mfm.experiments.data.DataGroup):
                data_set = mfm.experiments.data.ExperimentDataCurveGroup(data_set)
            fit = mfm.fitting.fit.FitGroup(data=data_set, model_class=model_class)
            mfm.fits.append(fit)
            fit.model.find_parameters()
            fit_control_widget = mfm.fitting.widgets.FittingControllerWidget(fit)

            cs.modelLayout.addWidget(fit_control_widget)
            for f in fit:
                cs.modelLayout.addWidget(f.model)
            fit_window = mfm.fitting.widgets.FitSubWindow(fit,
                                                          control_layout=cs.plotOptionsLayout,
                                                          fit_widget=fit_control_widget)
            fit_window = cs.mdiarea.addSubWindow(fit_window)
            mfm.fit_windows.append(fit_window)
            fit_window.show()


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

