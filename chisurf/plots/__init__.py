import chisurf.settings
import pyqtgraph as pg

pg.setConfigOptions(
    **chisurf.settings.cs_settings['gui']['plot']['pyqtgraph_config']
)

import chisurf.plots.global_fit
import chisurf.plots.global_tcspc
from chisurf.plots.distribution import DistributionPlot
from chisurf.plots.fitinfo import *
from chisurf.plots.lineplot import *
from chisurf.plots.parameter_scan import ParameterScanPlot
from chisurf.plots.plotbase import *
from chisurf.plots.surfaceplot import SurfacePlot
from chisurf.plots.wr_plot import ResidualPlot
from chisurf.plots.table_plot import FitTablePlot
from chisurf.plots.residual_image import Residual2DPlot


def __getattr__(name: str):
    """Lazy-load heavy plotting modules on first access."""
    if name == "molview":
        from chisurf.plots import molview
        return molview
    if name == "proteinMC":
        from chisurf.plots import proteinMC
        return proteinMC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

