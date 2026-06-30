"""GUI front-end for :mod:`chisurf.core.fio.staging`.

:func:`load_with_progress` runs a (potentially slow) file load off the GUI
thread behind an :class:`~chisurf.gui.widgets.progress.EnhancedProgressDialog`
that shows copy percent, transfer speed (MB/s) and ETA while the file is being
staged from slow/network storage, then an indeterminate "Parsing…" bar while
``tttrlib`` reads the local copy. The call is *synchronous from the caller's
point of view* -- it spins a local event loop and returns the loaded object --
so existing call sites that do ``obj = tttrlib.TTTR(path)`` only need to swap
the one line. Returns ``None`` when the user cancels.

All heavy lifting lives in the Qt-free core module; this file only marshals the
core's ``progress_cb`` onto the GUI thread and manages the dialog.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable

from chisurf.core.fio import staging
from chisurf.gui import QtCore, QtWidgets, run_on_gui_thread
from chisurf.gui.widgets.progress import EnhancedProgressDialog, Worker

__all__ = ["load_with_progress", "make_lazy_stage_dialog"]


def _format_label(done: int, total: int, mbps: float, eta: float | None) -> str:
    eta_txt = f" · ETA {int(eta)}s" if eta and eta > 0 else ""
    if total > 0:
        return (
            f"Copying … {done / 1e6:.0f} / {total / 1e6:.0f} MB "
            f"· {staging.format_rate(mbps)}{eta_txt}"
        )
    return f"Copying … {done / 1e6:.0f} MB · {staging.format_rate(mbps)}"


def load_with_progress(
    parent: QtWidgets.QWidget | None,
    loader: Callable[[str], object],
    src_path,
    *,
    title: str = "Loading file",
    label: str | None = None,
):
    """Load *src_path* via ``loader`` off the GUI thread with a progress dialog.

    Parameters
    ----------
    parent : QWidget or None
        Dialog parent.
    loader : callable
        ``loader(local_path: str) -> object``. Called with the local (possibly
        staged) path once any copy has finished; its return value is returned
        from this function. Typically ``lambda p: tttrlib.TTTR(p)`` or a variant
        passing a reading routine.
    src_path : str or pathlib.Path
        The source file to load. Staged locally first if it is on slow storage.
    title, label
        Dialog window title and initial label (defaults to the source path).

    Returns
    -------
    object or None
        The object returned by ``loader``, or ``None`` if the user cancelled.

    Raises
    ------
    Exception
        Re-raises any exception raised by staging or ``loader`` (other than a
        user cancellation, which yields ``None``).
    """
    src_path = str(src_path)

    app = QtWidgets.QApplication.instance()
    if app is None:
        # Headless / no event loop: stage + load synchronously, no dialog.
        with staging.staged_source(src_path) as local:
            return loader(str(local))

    dlg = EnhancedProgressDialog(title, label or src_path, 0, 100, parent)
    cancelled = threading.Event()
    dlg.canceled.connect(cancelled.set)

    def _apply_progress(done, total, mbps, eta):
        try:
            pct = int(100 * done / total) if total > 0 else 0
            dlg.update_progress(pct, _format_label(done, total, mbps, eta))
        except RuntimeError:
            pass  # dialog already destroyed

    def _set_parsing():
        try:
            dlg.setRange(0, 0)  # indeterminate / busy
            dlg.update_text("Parsing …")
        except RuntimeError:
            pass

    outcome: dict = {}

    def _work():
        def on_progress(done, total, mbps, eta):
            run_on_gui_thread(_apply_progress, done, total, mbps, eta)

        try:
            local, was_staged = staging.stage_path_if_slow(
                src_path, progress_cb=on_progress, cancel_cb=cancelled.is_set
            )
        except staging.StagingCancelled:
            outcome["cancelled"] = True
            return
        except BaseException as exc:  # noqa: BLE001 - surfaced to caller below
            outcome["error"] = exc
            return

        try:
            run_on_gui_thread(_set_parsing)
            outcome["value"] = loader(str(local))
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc
        finally:
            if was_staged:
                shutil.rmtree(local.parent, ignore_errors=True)

    loop = QtCore.QEventLoop()
    worker = Worker(_work)
    worker.signals.finished.connect(loop.quit)

    pool = QtCore.QThreadPool.globalInstance()
    dlg.show()
    pool.start(worker)
    loop.exec_()

    dlg.finish(auto_close=True, close_delay_ms=0)

    if "error" in outcome:
        raise outcome["error"]
    if outcome.get("cancelled"):
        return None
    return outcome.get("value")


def make_lazy_stage_dialog(parent: QtWidgets.QWidget | None = None, title: str = "Loading data"):
    """Build ``(progress_cb, cancel_cb, finish)`` driving a lazy progress dialog.

    Intended for the central reader path (``ExperimentReader.set_stage_callbacks``)
    where the read runs *on the GUI thread*. ``progress_cb`` therefore updates
    the dialog directly (``update_progress`` pumps ``processEvents``), keeping
    the UI responsive while the pure-Python staging copy loop runs. The dialog
    is created only on the first progress callback, so fast/local loads that are
    never staged show no dialog at all.

    Returns
    -------
    (progress_cb, cancel_cb, finish)
        ``progress_cb(done, total, mbps, eta)`` and ``cancel_cb() -> bool`` are
        passed to :meth:`ExperimentReader.set_stage_callbacks`; ``finish()``
        closes the dialog and must be called when loading is done (in a
        ``finally``).
    """
    state: dict = {"dlg": None}

    def progress_cb(done, total, mbps, eta):
        dlg = state["dlg"]
        if dlg is None:
            dlg = state["dlg"] = EnhancedProgressDialog(title, "Copying …", 0, 100, parent)
            dlg.show()
        try:
            pct = int(100 * done / total) if total > 0 else 0
            dlg.update_progress(pct, _format_label(done, total, mbps, eta))
        except RuntimeError:
            pass

    def cancel_cb() -> bool:
        dlg = state["dlg"]
        try:
            return bool(dlg is not None and dlg.wasCanceled())
        except RuntimeError:
            return False

    def finish():
        dlg = state["dlg"]
        if dlg is not None:
            try:
                dlg.finish(auto_close=True, close_delay_ms=0)
            except RuntimeError:
                pass
            state["dlg"] = None

    return progress_cb, cancel_cb, finish
