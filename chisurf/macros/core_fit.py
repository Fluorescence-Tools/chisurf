from __future__ import annotations

import os

import docx
from docx.shared import Inches

from chisurf import typing
from chisurf import logging

import chisurf.base
import chisurf.data
import chisurf.fitting
import chisurf.gui
import chisurf.gui.widgets


def add_fit(
        dataset_indices: typing.List[int] = None,
        model_name: str = None,
        model_kw: typing.Dict = None
):
    cs = chisurf.cs
    # Process inputs of macro and replace None
    # with more sensible values that are read
    # from the GUI
    if dataset_indices is None:
        dataset_indices = [cs.dataset_selector.selected_curve_index]
    if model_name is None:
        model_name = cs.current_model_name

    # Do nothing of no dataset is selected
    if len(dataset_indices) == 0:
        return

    # create a list of data sets to which a fit with
    # a particular model is added
    data_sets = [cs.dataset_selector.datasets[i] for i in dataset_indices]

    model_names = data_sets[0].experiment.model_names
    model_class = data_sets[0].experiment.model_classes[0]
    for model_idx, mn in enumerate(model_names):
        if mn == model_name:
            model_class = data_sets[0].experiment.model_classes[model_idx]
            break

    for data_set in data_sets:
        if data_set.experiment is data_sets[0].experiment:
            # Make sure the data set is a DataGroup
            if not isinstance(data_set, chisurf.data.DataGroup):
                data_group = chisurf.data.ExperimentDataCurveGroup([data_set])
            else:
                data_group = data_set

            # Create the fit
            fit_group = chisurf.fitting.fit.FitGroup(
                data=data_group,
                model_class=model_class,
                model_kw=model_kw
            )
            chisurf.fits.append(fit_group)

            fit_control_widget = chisurf.gui.widgets.fitting.FittingControllerWidget(
                fit=fit_group
            )
            cs.modelLayout.addWidget(fit_control_widget)
            for fit in fit_group:
                cs.modelLayout.addWidget(fit.model)

            fit_window = chisurf.gui.widgets.fitting.FitSubWindow(
                fit=fit_group,
                control_layout=cs.plotOptionsLayout,
                fit_widget=fit_control_widget
            )

            fit_window.setWindowTitle(fit.name)
            fit_window = cs.mdiarea.addSubWindow(fit_window)
            chisurf.gui.fit_windows.append(fit_window)
            cs.current_fit = fit_group
            fit_control_widget.onAutoFitRange()
            fit_window.show()
    cs.update()


def save_fit(target_path: str = None, use_complex_name: bool = False, fit_window=None):
    cs = chisurf.cs
    if fit_window is None:
        fit_window = cs.mdiarea.currentSubWindow()

    fit = fit_window.fit
    fit_control_widget = fit_window.fit_widget
    fit_group = fit_control_widget.fit

    if target_path is None:
        target_path = chisurf.working_path
    if use_complex_name:
        save_name = chisurf.base.clean_string(fit.name)
    else:
        save_name = os.path.basename(fit.data.name)
    filename = os.path.join(target_path, save_name)
    #fit.save(filename, 'json', save_curves=False)
    fit.save(filename, 'csv', save_curves=True)

    # Create word document
    document = docx.Document()
    document.add_heading(cs.current_fit.name, 0)
    if os.path.isdir(target_path):
        _ = document.add_heading('Fit-Results', level=1)

        for i, f in enumerate(fit):
            fit_control_widget.selected_fit = i
            fit_name = os.path.basename(fit.data.name)[0]
            model = f.model
            document.add_paragraph(
                text=fit_name,
                style='ListNumber'
            )
            for png_name, source in zip(
                    [save_name + '_screenshot_fit.png', save_name + '_screenshot_model.png'],
                    [fit_window, model]
            ):
                png_filename = os.path.join(target_path, png_name)
                source.grab().save(png_filename)
                document.add_picture(
                    os.path.join(target_path, png_filename),
                    width=Inches(2.0)
                )

        document.add_heading(
            text='Summary',
            level=1
        )

        p = document.add_paragraph(text='Parameters which are fitted are given in ')
        p.add_run('bold').bold = True
        p.add_run(', linked parameters in ')
        p.add_run('italic.').italic = True
        p.add_run(' fixed parameters are plain name. ')
        n_fits = len(fit_group.grouped_fits)
        table = document.add_table(rows=1, cols=n_fits + 1)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Fit-Nbr"
        for i, fit in enumerate(fit_group):
            hdr_cells[i + 1].text = str(i + 1)
            model = fit.model
            pk = list(model.parameters_all_dict.keys())
            pk.sort()
            for k in pk:
                row_cells = table.add_row().cells
                row_cells[0].text = str(k)
                for i, fit in enumerate(fit_group):
                    paragraph = row_cells[i + 1].paragraphs[0]
                    run = paragraph.add_run(
                        text='{:.3f}'.format(model.parameters_all_dict[k].value)
                    )
                    if model.parameters_all_dict[k].fixed:
                        continue
                    else:
                        if model.parameters_all_dict[k].link is not None:
                            run.italic = True
                        else:
                            run.bold = True

        row_cells = table.add_row().cells
        row_cells[0].text = str("Chi2r")

        for i, fit in enumerate(fit_group):
            paragraph = row_cells[i + 1].paragraphs[0]
            run = paragraph.add_run('{:.4f}'.format(fit.chi2r))
        tr = save_name
        document.save(os.path.join(target_path, tr + '.docx'))
    else:
        chisurf.logging.warning('The target folder %s does not exist', target_path)


def load_fit_result(
        fit_index: int,
        filename: str
) -> bool:
    if os.path.isfile(filename):
        chisurf.fits[fit_index].model.load(filename)
        chisurf.fits[fit_index].update()
        return True
    else:
        return False


def save_fits(target_path: str, use_complex_name: bool = False):
    if os.path.isdir(target_path):
        for fit_window in chisurf.gui.fit_windows:
            fit = fit_window.fit

            # Skip global fits
            if isinstance(fit.data.setup, chisurf.experiments.globalfit.GlobalFitSetup):
                continue

            if use_complex_name:
                save_name = chisurf.base.clean_string(fit.name)
            else:
                save_name = os.path.basename(fit.data.name)

            fit_name = fit.name
            p2 = target_path + '/' + save_name
            os.mkdir(p2)
            save_fit(target_path=p2, fit_window=fit_window)


def close_fit(idx: int = None):
    cs = chisurf.cs
    if idx is None:
        sub_window = cs.mdiarea.currentSubWindow()
        for i, w in enumerate(chisurf.gui.fit_windows):
            if w is sub_window:
                idx = i
    # chisurf.gui.widgets.clear_layout(cs.modelLayout)
    # chisurf.gui.widgets.clear_layout(cs.plotOptionsLayout)
    chisurf.fits.pop(idx)
    sub_window = chisurf.gui.fit_windows.pop(idx)
    sub_window.close()

    cs.update()


def link_fit_group(
        fitting_parameter_name: str,
        csi: int = 0
) -> None:
    """
    This macro links the parameters with a name
    specified by fitting_parameter_name within
    a FitGroup

    :param fitting_parameter_name:
    :param csi:
    :return:
    """
    cs = chisurf.cs
    if csi == 2:
        current_fit = cs.current_fit
        parameter = current_fit.model.parameters_all_dict[fitting_parameter_name]
        for f in cs.current_fit:
            try:
                p = f.model.parameters_all_dict[fitting_parameter_name]
                if p is not parameter:
                    p.link = parameter
            except KeyError:
                chisurf.logging.warning(f"The fit {f.name} has no parameter {fitting_parameter_name}")
    if csi == 0:
        for f in cs.current_fit:
            try:
                p = f.model.parameters_all_dict[fitting_parameter_name]
                p.link = None
            except KeyError:
                pass


def change_selected_fit_of_group(
    selected_fit: int
) -> None:
    """
    Changes the currently selected fit

    :param selected_fit:
    :return:
    """
    cs = chisurf.cs
    cs.current_fit.model.hide()
    cs.current_fit.selected_fit = selected_fit
    cs.current_fit.update()
    cs.current_fit.model.show()

