from __future__ import annotations
from chisurf import typing
from chisurf.core.actions._decorator import action

@action("dataset.add", schema={"experiment_reader": None})
def add_dataset(experiment_reader, **kwargs):
    """Load a dataset using the given experiment reader."""
    from chisurf.macros import core_data
    return core_data.add_dataset(experiment_reader=experiment_reader, _from_controller=True, **kwargs)

@action("dataset.remove", schema={"dataset_indices": list})
def remove_datasets(dataset_indices: typing.List[int]):
    """Remove datasets by index."""
    from chisurf.macros import core_data
    return core_data.remove_datasets(dataset_indices=list(dataset_indices or []), _from_controller=True)

@action("dataset.group", schema={"dataset_indices": list})
def group_datasets(dataset_indices: typing.List[int]):
    """Group datasets into a curve group."""
    from chisurf.macros import core_data
    return core_data.group_datasets(dataset_indices=list(dataset_indices or []), _from_controller=True)

@action("dataset.ungroup", schema={"dataset_indices": list})
def ungroup_datasets(dataset_indices: typing.List[int]):
    """Ungroup a dataset group."""
    from chisurf.macros import core_data
    return core_data.ungroup_datasets(dataset_indices=list(dataset_indices or []), _from_controller=True)


@action("dataset.restore_global_fit")
def restore_global_fit_dataset():
    """Restore the reserved Global-fit dataset if missing."""
    from chisurf.macros import core_data
    return core_data.restore_global_fit_dataset(_from_controller=True)
