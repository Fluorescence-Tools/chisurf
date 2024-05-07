from __future__ import annotations


import chisurf.base
import chisurf.data
import chisurf.fitting
import chisurf.gui
from chisurf import typing


def group_datasets(dataset_indices: typing.List[int]) -> None:
    selected_data = [
        chisurf.imported_datasets[i] for i in dataset_indices
    ]
    if isinstance(
            selected_data[0],
            chisurf.data.DataCurve
    ):
        # TODO: check for double names!!!
        dg = chisurf.data.ExperimentDataCurveGroup(
            selected_data,
            name="Data-Group"
        )
    else:
        dg = chisurf.data.ExperimentDataGroup(
            selected_data,
            name="Data-Group"
        )
    dn = list()
    for d in chisurf.imported_datasets:
        if d not in dg:
            dn.append(d)
    dn.append(dg)
    chisurf.imported_datasets = dn


def remove_datasets(dataset_indices: typing.List[int]) -> None:
    if not isinstance(dataset_indices, list):
        dataset_indices = [dataset_indices]

    imported_datasets = list()
    for i, d in enumerate(chisurf.imported_datasets):
        if d.name == 'Global Dataset':
            imported_datasets.append(d)
            continue
        if i not in dataset_indices:
            imported_datasets.append(d)
        else:
            fw = list()
            for fit_window in chisurf.gui.fit_windows:
                if fit_window.fit.data is d:
                    fit_window.close_confirm = False
                    fit_window.close()
                else:
                    fw.append(fit_window)
            chisurf.gui.fit_windows = fw

    chisurf.imported_datasets = imported_datasets


def add_dataset(
        expriment_reader: chisurf.experiments.reader.ExperimentReader = None,
        dataset: chisurf.base.Data = None,
        **kwargs
) -> None:
    cs = chisurf.cs

    # Handle file name argument from prompt
    # if multiple filenames (list) cleanup list else
    # use only one filename (old behaviour)
    filename = kwargs.get('filename', None)
    if filename is not None:
        filename = filename.split('|')
        if len(filename) == 1:
            filename = filename[0]
    kwargs['filename'] = filename

    if expriment_reader is None:
        expriment_reader = cs.current_experiment_reader
    if dataset is None:
        dataset = expriment_reader.get_data(**kwargs)

    if isinstance(dataset, chisurf.data.ExperimentDataGroup):
        dataset_group = dataset
    else:
        dataset_group = chisurf.data.ExperimentDataCurveGroup(dataset)

    if len(dataset_group) == 1:
        chisurf.imported_datasets.append(dataset_group[0])
    else:
        chisurf.imported_datasets.append(dataset_group)

    cs.update()

