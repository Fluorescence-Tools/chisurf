"""Standalone launcher for the Batch-Analysis wizard (``csg_batch_analysis``)."""

from __future__ import annotations


def main() -> None:
    """Launch the batch-analysis wizard as a standalone Qt application."""
    from qtpy import QtWidgets

    from .gui.tool import BatchProcessingWizard

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = BatchProcessingWizard()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
