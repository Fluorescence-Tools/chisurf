import time
from chisurf import typing
from chisurf.runtime.action_decorator import action


@action("fit.add", schema={"dataset_indices": list})
def add_fit(dataset_indices: typing.List[int], model_name: typing.Optional[str] = None, model_kw: typing.Optional[typing.Dict[str, typing.Any]] = None):
    """Add a new fit group."""
    from chisurf.macros import core_fit
    kwargs: typing.Dict[str, typing.Any] = {"dataset_indices": list(dataset_indices or [])}
    if isinstance(model_kw, dict):
        kwargs["model_kw"] = model_kw
    if model_name is not None:
        kwargs["model_name"] = str(model_name)
    return core_fit.add_fit(**kwargs)


@action("fit.load")
def load_fit(filename: str):
    """Load a fit project from a JSON file."""
    from chisurf.macros import core_fit
    return core_fit.load_fit_project(filename)


@action("fit.save")
def save_fit(target_path: str):
    """Save the current fit project."""
    from chisurf.macros import core_fit
    return core_fit.save_fit(target_path=target_path)


@action("fit.save_all")
def save_all_fits(target_path: str):
    """Save all fit projects to a directory."""
    from chisurf.macros import core_fit
    return core_fit.save_fits(target_path=target_path)


@action("fit.add.start")
def add_fit_start():
    """Signify start of adding a fit."""
    return {}


@action("fit.close_all")
def close_all_fits():
    """Close all fit windows."""
    from chisurf.macros import core_fit
    return core_fit.close_all_fits()


@action("fit.close", schema={"idx": int})
def close_fit(idx: int):
    """Close a specific fit window."""
    from chisurf.macros import core_fit
    return core_fit.close_fit(idx=int(idx))


@action("fit.data_set", debounce_ms=200)
def set_fit_data(fit_index: int, dataset_index: int):
    """Set data for a fit."""
    import chisurf
    fit_idx = int(fit_index)
    ds_idx = int(dataset_index)
    if ds_idx < 0:
        ds_idx = len(chisurf.imported_datasets) - 1
    if ds_idx < 0:
        return None
    fit_obj = chisurf.fits[fit_idx]
    dataset_obj = chisurf.imported_datasets[ds_idx]
    fit_obj.data = dataset_obj
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("fit.run.start")
def run_fit_start(fit_name: str):
    """Signify start of a fit run."""
    return {}


@action("fit.run.finish")
def run_fit_finish(fit_name: str):
    """Signify finish of a fit run."""
    return {}


@action("fit.run.abort")
def run_fit_abort(fit_name: str):
    """Signify abortion of a fit run."""
    return {}


@action("fit.run.execute", replayable=False, side_effect_class="execution")
def run_fit_execute(fit_controller: typing.Any):
    """Execute the fit controller implementation."""
    if hasattr(fit_controller, "_run_fit_impl"):
        fit_controller._run_fit_impl()
    return {}


@action("fit.group_link")
def link_fit_group(fit_indices: typing.List[int]):
    """Link a group of fits."""
    return {}


@action("fit.group_unlink")
def unlink_fit_group(fit_indices: typing.List[int]):
    """Unlink a group of fits."""
    return {}


@action("fit.group_link_toggle", debounce_ms=200)
def toggle_fit_group_link(fit_indices: typing.List[int]):
    """Toggle linking of a group of fits."""
    return {}


@action("fit.update", debounce_ms=200)
def update_fit(fit_index: int = 0):
    """Update a fit's state."""
    import chisurf
    fit_obj = chisurf.fits[int(fit_index)]
    fit_obj.update()
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("fit.mask_set", replayable=False, debounce_ms=200)
def set_fit_mask(fit_index: int, mask: typing.Any):
    """Set the fitting mask."""
    import chisurf
    fit_obj = chisurf.fits[int(fit_index)]
    fit_obj.mask = mask
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("fit.range.set", schema={"fit_index": int, "xmin": int, "xmax": int}, debounce_ms=60)
def set_fit_range(fit_index: int, xmin: int, xmax: int):
    """Set the fit range for a fit group."""
    import chisurf
    try:
        fit_obj = chisurf.fits[int(fit_index)]
    except (IndexError, AttributeError):
        return None
    fit_obj.fit_range = (int(xmin), int(xmax))
    fit_obj.update()
    
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("fit.set_dataset", schema={"fit_index": int, "dataset_index": int})
def set_fit_dataset(fit_index: int, dataset_index: int = -1):
    """Assign a dataset to a fit."""
    import chisurf
    fit_idx = int(fit_index)
    ds_idx = int(dataset_index)
    if ds_idx < 0:
        ds_idx = len(chisurf.imported_datasets) - 1
    if ds_idx < 0:
        return None

    fit_obj = chisurf.fits[fit_idx]
    dataset_obj = chisurf.imported_datasets[ds_idx]
    fit_obj.data = dataset_obj
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}


@action("fit.run", schema={"fit_index": int}, replayable=False, side_effect_class="execution")
def run_fit(fit_index: int):
    """Execute the fit optimization."""
    import chisurf
    fit_idx = int(fit_index)
    fit_obj = chisurf.fits[fit_idx]
    fit_obj.run()
    return {"source_uid": str(getattr(fit_obj, "unique_identifier", ""))}
