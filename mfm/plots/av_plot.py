import numpy as np
from qtpy import  QtWidgets
from pyqtgraph.dockarea import DockArea, Dock
import pyqtgraph.opengl as gl
from matplotlib import cm

import mfm
import mfm.math
import mfm.fluorescence
from mfm.plots import plotbase


class AvPlotControl(
    QtWidgets.QWidget
):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)


class AvPlot(plotbase.Plot):
    """
    Started off as a plotting class to display TCSPC-data displaying the IRF, the experimental data, the residuals
    and the autocorrelation of the residuals. Now it is also used also for fcs-data.

    In case the models is a :py:class:`~experiment.models.tcspc.LifetimeModel` it takes the irf and displays it:

        irf = fit.models.convolve.irf
        irf_y = irf.y

    The models data and the weighted residuals are taken directly from the fit:

        model_x, model_y = fit[:]
        wres_y = fit.weighted_residuals

    """

    name = "Accessible Volume"

    def __init__(
            self,
            fit,
            *args,
            **kwargs
    ):
        super().__init__(
            fit=fit,
            *args,
            **kwargs
        )
        # plot control dialog
        self.pltControl = AvPlotControl(self, **kwargs)
        self.layout = QtWidgets.QVBoxLayout(self)

        area = DockArea()
        self.layout.addWidget(area)

        hide_title = mfm.settings.gui['plot']['hideTitle']
        d1 = Dock("quenching", size=(300, 300), hideTitle=hide_title)
        d2 = Dock("diffusion", size=(300, 300), hideTitle=hide_title)
        d3 = Dock("equilibrium", size=(300, 300), hideTitle=hide_title)

        area.addDock(d1, 'top')
        area.addDock(d2, 'right', d1)
        area.addDock(d3, 'bottom')

        self.quenching_widget = gl.GLViewWidget()
        self.quenching_widget.opts['distance'] = 100

        d1.addWidget(self.quenching_widget)

    def update_all(self, only_fit_range=False, *args, **kwargs):
        fit = self.fit

        pdb_filename = './test/data/modelling/pdb_files/hGBP1_closed.pdb'
        residue_number = 18
        atom_name = 'CB'
        free_diffusion = 8.0
        atomic_slow_factor = 0.9
        contact_distance = 4.5
        av = mfm.fluorescence.fps.ACV(pdb_filename,
                                      radius1=5.0,
                                      linker_length=21.5,
                                      residue_seq_number=residue_number,
                                      atom_name=atom_name,
                                      diffusion_coefficients=(free_diffusion, atomic_slow_factor)
                                      )


        alpha = 0.1
        data = av.rate_map
        min_v, max_v = mfm.math.datatools.minmax(data.flatten(), ignore_zero=True)
        d1 = cm.jet((data - min_v) / (max_v - min_v)) * 255
        d1 = d1.astype(dtype=np.ubyte)

        d2 = np.zeros_like(d1)
        nx, ny, nz = data.shape
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    d2[ix, iy, iz, 0] = d1[ix, iy, iz, 0]
                    d2[ix, iy, iz, 1] = d1[ix, iy, iz, 1]
                    d2[ix, iy, iz, 2] = d1[ix, iy, iz, 2]
                    d2[ix, iy, iz, 3] = 255 * alpha if data[ix, iy, iz] > 0.0 else 0


        #d2 = av.points
        v = gl.GLVolumeItem(d2)
        v.translate(-nx/2, -ny/2, -nz/2)
        self.quenching_widget.addItem(v)

        ax = gl.GLAxisItem()
        self.quenching_widget.addItem(ax)
        self.quenching_widget.show()
