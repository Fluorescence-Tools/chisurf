import chisurf.settings
import pyqtgraph as pg

pg.setConfigOptions(
    **chisurf.settings.cs_settings['gui']['plot']['pyqtgraph_config']
)

import chisurf.plots.global_fit
#import chisurf.plots.global_tcspc
#import chisurf.plots.av_plot
from chisurf.plots.molview import *
from chisurf.plots.distribution import DistributionPlot
from chisurf.plots.fitinfo import *
from chisurf.plots.lineplot import *
from chisurf.plots.parameter_scan import ParameterScanPlot
from chisurf.plots.plotbase import *
from chisurf.plots.proteinMC import *
from chisurf.plots.wr_plot import ResidualPlot
