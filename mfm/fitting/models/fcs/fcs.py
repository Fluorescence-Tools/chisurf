import os
from PyQt5 import QtGui

import mfm
from mfm import plots
from mfm.fitting.models.parse import ParseModelWidget


class ParseFCSWidget(ParseModelWidget):

    """
    FCS
    """

    plot_classes = [(plots.LinePlot, {'d_scalex': 'log',
                                                   'd_scaley': 'lin',
                                                   'r_scalex': 'log',
                                                   'r_scaley': 'lin',
                                                   'x_label': 't [ms]',
                                                   'y_label': 'G(t)'
                                                   }),
        (plots.FitInfo, {}), (plots.ParameterScanPlot, {})
    ]

    def __init__(self, fit, **kwargs):
        self.icon = QtGui.QIcon(":/icons/icons/FCS.png")
        fn = os.path.join(mfm.package_directory, 'settings/models.yaml')
        ParseModelWidget.__init__(self, fit, model_file=fn, **kwargs)


