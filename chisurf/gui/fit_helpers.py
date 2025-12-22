from __future__ import annotations

import traceback

import chisurf
from chisurf.gui import QtCore


def add_fits_for_datasets(window, data_idx, model_name: str):
    """Add fits for the provided dataset indices, mirroring onAddFit."""
    if not data_idx:
        return

    indices = list(data_idx)

    def _create_next_fit():
        if not indices:
            return
        idx = indices.pop(0)
        try:
            chisurf.macros.core_fit.add_fit(
                dataset_indices=[idx],
                model_name=model_name,
            )
        except Exception as e:
            msg = f"Add fit failed for dataset index {idx} with model '{model_name}': {e}"
            try:
                chisurf.logging.error(msg)
                chisurf.logging.error(traceback.format_exc())
            except Exception:
                pass
            try:
                window.status.showMessage(msg, 10000)
            except Exception:
                pass
        if indices:
            QtCore.QTimer.singleShot(0, _create_next_fit)

    _create_next_fit()
