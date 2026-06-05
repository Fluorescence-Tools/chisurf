from __future__ import annotations

from qtpy import QtWidgets

import chisurf
import chisurf.core.fitting
import chisurf.gui.plots

from chisurf.gui.widgets.models.model_widget import ModelWidget


MolViewPlot = getattr(chisurf.gui.plots, "MolViewPlot", None)


class ProteinMCModelWidget(ModelWidget):
    name = "ProteinMC"

    try:
        plot_classes = [
            (chisurf.gui.plots.ProteinMCPlot, {}),
        ]
        if MolViewPlot is not None:
            plot_classes.append((MolViewPlot, {}))
    except Exception:
        plot_classes = []

    def __init__(
            self,
            fit: "chisurf.core.fitting.fit.Fit",
            *args,
            **kwargs
    ):
        """Initialize the ProteinMC model widget.

        Parameters
        ----------
        fit : chisurf.core.fitting.fit.Fit
            Fit object this model is attached to.
        """
        super().__init__(fit=fit, *args, **kwargs)

        self.structure = getattr(self.fit, "data", None)
        try:
            chisurf.logging.info(
                "ProteinMCModelWidget.__init__: fit=%s data=%s type=%s",
                getattr(self.fit, "name", "unknown"),
                getattr(self.structure, "name", getattr(self.structure, "filename", "unknown")),
                self.structure.__class__.__name__ if self.structure is not None else "None",
            )
        except Exception:
            pass

        self.rmsd = []
        self.drmsd = []
        self.energy = []
        self.chi2r = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label = QtWidgets.QLabel("ProteinMC structure model")
        label.setWordWrap(True)
        layout.addWidget(label)

        self.setLayout(layout)
        self.layout = layout

    def update_model(self, **kwargs):
        """Update the model state by logging current tracking data lengths."""
        try:
            chisurf.logging.info(
                "ProteinMCModelWidget.update_model: len(rmsd)=%d len(drmsd)=%d len(energy)=%d len(chi2r)=%d",
                len(self.rmsd),
                len(self.drmsd),
                len(self.energy),
                len(self.chi2r),
            )
        except Exception:
            pass
        return

    def update_widgets(self) -> None:
        """Refresh GUI widgets and log the update."""
        try:
            chisurf.logging.info("ProteinMCModelWidget.update_widgets: called")
        except Exception:
            pass
        super().update_widgets()

    def update(self) -> None:
        """Perform a full update of the model, widgets, and tracking data."""
        try:
            chisurf.logging.info(
                "ProteinMCModelWidget.update: before super.update len(rmsd)=%d len(drmsd)=%d len(energy)=%d len(chi2r)=%d",
                len(self.rmsd),
                len(self.drmsd),
                len(self.energy),
                len(self.chi2r),
            )
        except Exception:
            pass
        super().update()
        try:
            chisurf.logging.info(
                "ProteinMCModelWidget.update: after super.update len(rmsd)=%d len(drmsd)=%d len(energy)=%d len(chi2r)=%d",
                len(self.rmsd),
                len(self.drmsd),
                len(self.energy),
                len(self.chi2r),
            )
        except Exception:
            pass
