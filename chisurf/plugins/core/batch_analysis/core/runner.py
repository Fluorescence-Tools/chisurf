"""Headless batch-fit runner for the Batch-Analysis plugin.

Runs one *template* fit over a list of already-loaded datasets and/or files,
snapshotting the template's parameters and restoring them before every run so
each item starts from the same initial guess, then collecting the per-parameter
results. This module is deliberately Qt-free: the GUI wizard and the CLI both
drive :func:`run_batch`, and the pure helpers (queue building, parameter
snapshot/restore, CSV/DOCX/ZIP export) are unit-tested without a running Qt
application.

The template fit is chosen (and pre-optimised) by the user in ChiSurf; the
runner only reads its parameter values/fixed flags and re-runs it against each
item. Fit execution is delegated through two injectable seams so tests can pass
fakes:

* ``fit_client`` — an object exposing ``get_fit_objects()``,
  ``set_parameter_value(...)`` and ``set_parameter_fixed(...)`` (in production,
  :func:`chisurf.gui.widgets.fitting.fitting_client.get_fitting_client`).
* ``dispatch`` — the action dispatcher used to load data / assign datasets /
  run fits (in production, ``chisurf.core.actions.dispatch``).
"""

from __future__ import annotations

import csv
import dataclasses
import os
import pathlib
import shutil
import typing

#: Column order of the consolidated results CSV / DOCX table.
FIELDNAMES: tuple[str, ...] = ("Run", "Filename", "Parameter", "Fixed", "Value", "Chi2r")


@dataclasses.dataclass
class BatchItem:
    """One unit of work in the batch queue.

    Parameters
    ----------
    kind : str
        Either ``"dataset"`` (an already-loaded ChiSurf dataset) or ``"file"``
        (a path to load before fitting).
    value : object
        The dataset object (``kind == "dataset"``) or the file path string
        (``kind == "file"``).
    name : str
        Human-readable label used in the results table and output filenames.
    """

    kind: str
    value: typing.Any
    name: str


@dataclasses.dataclass
class BatchResults:
    """Collected results of a batch run.

    Attributes
    ----------
    rows : list of dict
        One row per (item, parameter) with the :data:`FIELDNAMES` columns plus a
        hidden ``GroupKey`` used to group rows/screenshots by item.
    file_order : list of str
        The item display names in processing order (used to order the report).
    screenshot_map : dict
        Maps an item's group key to a screenshot path captured by the GUI.
    """

    rows: list[dict] = dataclasses.field(default_factory=list)
    file_order: list[str] = dataclasses.field(default_factory=list)
    screenshot_map: dict[str, str] = dataclasses.field(default_factory=dict)

    def write_csv(self, path: str) -> None:
        """Write the consolidated results to *path* as CSV."""
        write_csv(self.rows, path)


# ── pure helpers ────────────────────────────────────────────────────────────
def sanitize_filename(name: str) -> str:
    """Return a filename-safe stem of *name* (no directory, no extension).

    Parameters
    ----------
    name : str
        Original filename or label.

    Returns
    -------
    str
        Sanitized base name, or ``"file"`` when *name* reduces to nothing.
    """
    base = pathlib.Path(name).stem
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in base)
    return safe or "file"


def norm_key(file_path: str) -> str:
    """Return a stable, absolute key for grouping results/screenshots."""
    try:
        return str(pathlib.Path(file_path).resolve())
    except Exception:
        return os.path.abspath(file_path)


def build_queue(
    datasets: typing.Sequence[typing.Any],
    files: typing.Sequence[str],
) -> list[BatchItem]:
    """Build the processing queue: loaded datasets first, then files.

    Parameters
    ----------
    datasets : sequence
        Already-loaded dataset objects (each read for a ``name``/``filename``).
    files : sequence of str
        File paths to load and fit.

    Returns
    -------
    list of BatchItem
        The ordered queue.
    """
    items: list[BatchItem] = []
    for ds in datasets:
        name = (
            getattr(ds, "name", None)
            or getattr(ds, "filename", None)
            or f"Dataset {len(items) + 1}"
        )
        items.append(BatchItem(kind="dataset", value=ds, name=str(name)))
    for fpath in files:
        items.append(BatchItem(kind="file", value=str(fpath), name=str(fpath)))
    return items


def datasets_have_mixed_types(datasets: typing.Sequence[typing.Any]) -> bool:
    """Return ``True`` when *datasets* span more than one experiment class.

    Batch analysis applies a single template fit, so mixing experiment types
    (e.g. TCSPC and FCS) is rejected upstream.
    """
    if len(datasets) <= 1:
        return False
    classes = set()
    for ds in datasets:
        try:
            classes.add(type(getattr(ds, "experiment", None)))
        except Exception:
            classes.add(type(None))
    return len(classes) > 1


def snapshot_parameters(fit: typing.Any) -> dict[str, tuple[float, bool]]:
    """Return ``{name: (value, fixed)}`` for every parameter of *fit*'s model."""
    return {p.name: (p.value, p.fixed) for p in fit.model.parameters_all}


def restore_parameters(
    fit_client: typing.Any,
    fit_index: int,
    fit: typing.Any,
    snapshot: typing.Mapping[str, tuple[float, bool]],
) -> None:
    """Restore the template *snapshot* onto *fit* through *fit_client*.

    Parameters
    ----------
    fit_client : object
        Provides ``set_parameter_value`` / ``set_parameter_fixed``.
    fit_index : int
        Index of the fit in the fitting client.
    fit : object
        The fit whose model parameters are being reset.
    snapshot : mapping
        The ``{name: (value, fixed)}`` mapping from :func:`snapshot_parameters`.
    """
    for param in fit.model.parameters_all:
        if param.name not in snapshot:
            continue
        value, fixed = snapshot[param.name]
        fit_client.set_parameter_value(
            parameter_name=str(param.name), value=value, fit_index=fit_index
        )
        fit_client.set_parameter_fixed(
            parameter_name=str(param.name), fixed=fixed, fit_index=fit_index
        )


def collect_rows(
    fit: typing.Any,
    run_index: int,
    name: str,
    group_key: str,
) -> list[dict]:
    """Return the result rows for one completed run of *fit*."""
    rows = []
    for param in fit.model.parameters_all:
        rows.append(
            {
                "Run": str(run_index),
                "Filename": name,
                "GroupKey": group_key,
                "Parameter": param.name,
                "Fixed": "Yes" if param.fixed else "No",
                "Value": param.value,
                "Chi2r": fit.chi2r,
            }
        )
    return rows


def write_csv(rows: typing.Sequence[dict], path: str) -> None:
    """Write *rows* to *path* as CSV using :data:`FIELDNAMES` (extra keys dropped)."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_docx(
    rows: typing.Sequence[dict],
    path: str,
    file_order: typing.Sequence[str],
    screenshot_map: typing.Mapping[str, str] | None = None,
    *,
    csv_name: str = "",
) -> tuple[bool, str]:
    """Write a DOCX report with per-item screenshots and a results table.

    Parameters
    ----------
    rows : sequence of dict
        Result rows (as produced by :func:`collect_rows`).
    path : str
        Destination ``.docx`` path.
    file_order : sequence of str
        Item display names, in report order.
    screenshot_map : mapping, optional
        Maps an item's group key to a screenshot image path.
    csv_name : str, optional
        Basename of the companion CSV, referenced in the report header.

    Returns
    -------
    (bool, str)
        ``(True, path)`` on success, otherwise ``(False, reason)``.
    """
    try:
        from docx import Document
        from docx.shared import Inches
    except Exception as exc:  # pragma: no cover - optional dependency
        return False, f"python-docx not available: {exc}"

    screenshot_map = screenshot_map or {}
    grouped: dict[str, list] = {}
    for row in rows:
        key = row.get("GroupKey", norm_key(row.get("Filename", "")))
        grouped.setdefault(key, []).append(row)

    doc = Document()
    doc.add_heading("Batch Fit Results", level=0)
    if csv_name:
        doc.add_paragraph(f"CSV: {csv_name}")

    for idx, name in enumerate(file_order, start=1):
        key = norm_key(name)
        doc.add_heading(f"{idx}. {os.path.basename(name)}", level=1)
        img = screenshot_map.get(key, "")
        if img and os.path.exists(img):
            try:
                doc.add_picture(img, width=Inches(6))
            except Exception as exc:
                doc.add_paragraph(f"[Could not add image: {exc}]")

    consolidated = [r for name in file_order for r in grouped.get(norm_key(name), [])]
    if consolidated:
        table = doc.add_table(rows=1, cols=6)
        for cell, text in zip(
            table.rows[0].cells, ("Filename", "Parameter", "Fixed", "Value", "Chi2r", "Run")
        ):
            cell.text = text
        for r in consolidated:
            cells = table.add_row().cells
            cells[0].text = str(r.get("Filename", ""))
            cells[1].text = str(r.get("Parameter", ""))
            cells[2].text = str(r.get("Fixed", ""))
            cells[3].text = str(r.get("Value", ""))
            cells[4].text = str(r.get("Chi2r", ""))
            cells[5].text = str(r.get("Run", ""))
    else:
        doc.add_paragraph("No parameters found.")

    try:
        doc.save(path)
        return True, path
    except Exception as exc:
        return False, f"Could not save DOCX: {exc}"


def zip_directory(source_dir: str, zip_base: str) -> str | None:
    """Zip *source_dir* into ``<zip_base>.zip`` and return the archive path (or None)."""
    try:
        return shutil.make_archive(zip_base, "zip", root_dir=source_dir)
    except Exception:
        return None


# ── orchestration ───────────────────────────────────────────────────────────
def run_batch(
    fit_index: int,
    items: typing.Sequence[BatchItem],
    *,
    fit_client: typing.Any = None,
    dispatch: typing.Callable[..., typing.Any] | None = None,
    imported_datasets: typing.Sequence[typing.Any] | None = None,
    on_progress: typing.Callable[[int, int, str], None] | None = None,
    on_run_complete: typing.Callable[[BatchItem, int, str], str | None] | None = None,
    fit_export_dir: str | None = None,
) -> BatchResults:
    """Run the template fit (``fit_index``) over every item and collect results.

    Before each run the template parameters are restored, so every item starts
    from the same initial guess. After each run the parameters and reduced χ²
    are read back into :class:`BatchResults`.

    Parameters
    ----------
    fit_index : int
        Index of the template fit in *fit_client*.
    items : sequence of BatchItem
        The processing queue (see :func:`build_queue`).
    fit_client : object, optional
        Fitting client; defaults to
        :func:`chisurf.gui.widgets.fitting.fitting_client.get_fitting_client`.
    dispatch : callable, optional
        Action dispatcher; defaults to ``chisurf.core.actions.dispatch``.
    imported_datasets : sequence, optional
        The list used to resolve a dataset's index for ``fit.set_dataset``;
        defaults to ``chisurf.imported_datasets``.
    on_progress : callable, optional
        Called ``(index, total, name)`` (1-based) after each item finishes.
    on_run_complete : callable, optional
        Called ``(item, run_index, group_key)`` after each fit; may return a
        screenshot path, which is stored in ``results.screenshot_map``.
    fit_export_dir : str, optional
        Directory to write per-run numeric fit exports (``fit.save``). When
        ``None`` no per-run export is written.

    Returns
    -------
    BatchResults
        The collected rows, processing order and screenshot map.
    """
    if fit_client is None:
        from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client

        fit_client = get_fitting_client()
    if dispatch is None:
        import chisurf as cs

        dispatch = cs.core.actions.dispatch
    if imported_datasets is None:
        import chisurf as cs

        imported_datasets = getattr(cs, "imported_datasets", [])

    fit_objects = fit_client.get_fit_objects()
    fit = fit_objects[fit_index] if 0 <= fit_index < len(fit_objects) else None
    if fit is None:
        raise IndexError(f"fit_index {fit_index} out of range ({len(fit_objects)} fits)")

    snapshot = snapshot_parameters(fit)
    results = BatchResults()
    total = len(items)

    for i, item in enumerate(items, start=1):
        key = norm_key(item.name)
        restore_parameters(fit_client, fit_index, fit, snapshot)

        if item.kind == "dataset":
            ds = item.value
            ds_idx = imported_datasets.index(ds) if ds in imported_datasets else -1
            dispatch(
                name="fit.set_dataset",
                payload={"fit_index": int(fit_index), "dataset_index": int(ds_idx)},
            )
            dispatch(name="fit.run", payload={"fit_index": int(fit_index)})
        else:
            file_path = pathlib.Path(item.value).as_posix()
            dispatch(
                name="dataset.add",
                payload={"filename": file_path, "experiment_reader": None},
            )
            dispatch(
                name="fit.set_dataset",
                payload={"fit_index": int(fit_index), "dataset_index": -1},
            )
            dispatch(name="fit.run", payload={"fit_index": int(fit_index)})

        if fit_export_dir:
            try:
                base = os.path.join(fit_export_dir, f"{i:03d}_{sanitize_filename(item.name)}")
                fit.save(base, "csv", save_curves=True)
            except Exception:
                pass

        if on_run_complete is not None:
            try:
                img = on_run_complete(item, i, key)
                if img:
                    results.screenshot_map[key] = img
            except Exception:
                pass

        results.rows.extend(collect_rows(fit, i, item.name, key))
        results.file_order.append(item.name)
        if on_progress is not None:
            on_progress(i, total, item.name)

    return results
