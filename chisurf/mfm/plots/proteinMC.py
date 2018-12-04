from PyQt5 import QtCore, QtGui, QtWidgets
#from guiqwt.plot import CurveDialog
#from guiqwt.builder import make
import numpy as np

from mfm.plots.plotbase import Plot
from PyQt5 import Qt, QtCore, QtGui, QtWidgets, uic
import numpy as np
import mfm
from mfm.plots import plotbase
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
pyqtgraph_settings = mfm.cs_settings['gui']['plot']["pyqtgraph"]
for setting in mfm.cs_settings['gui']['plot']['pyqtgraph']:
    pg.setConfigOption(setting, mfm.cs_settings['gui']['plot']['pyqtgraph'][setting])
colors = mfm.cs_settings['gui']['plot']['colors']
color_scheme = mfm.colors
import matplotlib.colors as mpl_colors


class ProteinMCPlot(Plot):

    name = "Trajectory-Plot"

    def __init__(self, fit):
        mfm.plots.Plot.__init__(self, fit)

        self.trajectory = fit.model
        self.source = fit.model

        self.layout = QtWidgets.QVBoxLayout(self)
        area = DockArea()
        self.layout.addWidget(area)
        hide_title = mfm.cs_settings['gui']['plot']['hideTitle']
        d1 = Dock("RMSD")
        d2 = Dock("dRMSD")
        d3 = Dock("Energy")
        d4 = Dock("FRET")

        p1 = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])
        p2 = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])
        p3 = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])
        p4 = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])

        d1.addWidget(p1)
        d2.addWidget(p2)
        d3.addWidget(p3)
        d4.addWidget(p4)

        area.addDock(d1, 'top')
        area.addDock(d2, 'right', d1)
        area.addDock(d3, 'bottom')
        area.addDock(d4, 'right', d3)

        # RMSD - Curves
        self.rmsd_plot = p1.getPlotItem()
        self.drmsd_plot = p2.getPlotItem()
        self.energy_plot = p3.getPlotItem()
        self.fret_plot = p4.getPlotItem()

        lw = mfm.cs_settings['gui']['plot']['line_width']
        self.rmsd_curve = self.rmsd_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['irf'], width=lw), name='rmsd')
        self.drmsd_curve = self.drmsd_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['data'], width=lw), name='drmsd')
        self.energy_curve = self.energy_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['model'], width=lw), name='energy')
        self.fret_curve = self.fret_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['model'], width=lw), name='fret')

    def update_all(self, *args, **kwargs):

        rmsd = np.array(self.trajectory.rmsd)
        drmsd = np.array(self.trajectory.drmsd)
        energy = np.array(self.trajectory.energy)
        energy_fret = np.array(self.trajectory.chi2r)
        x = list(range(len(rmsd)))

        self.rmsd_curve.setData(x=x, y=rmsd)
        self.drmsd_curve.setData(x=x, y=drmsd)
        self.energy_curve.setData(x=x, y=energy)
        self.fret_curve.setData(x=x, y=energy_fret)



# class ProteinMCPlot_Old(Plot):
#
#     name = "Trajectory-Plot"
#
#     def __init__(self, fit):
#         Plot.__init__(self, fit)
#         self.layout = QtGui.QVBoxLayout(self)
#
#         self.trajectory = fit.model
#         self.source = fit.model
#
#         # RMSD - Curves
#         top_left = QtGui.QFrame(self)
#         top_left.setFrameShape(QtGui.QFrame.StyledPanel)
#         l = QtGui.QVBoxLayout(top_left)
#
#         top_right = QtGui.QFrame(self)
#         top_right.setFrameShape(QtGui.QFrame.StyledPanel)
#         r = QtGui.QVBoxLayout(top_right)
#
#         splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
#         splitter.addWidget(top_left)
#         splitter.addWidget(top_right)
#
#         self.layout.addWidget(splitter)
#
#         win = CurveDialog()
#         self.rmsd_plot = win.get_plot()
#         self.rmsd_plot.set_titles(ylabel='RMSD')
#         self.rmsd_curve = make.curve([],  [], color="m", linewidth=1)
#         self.rmsd_plot.add_item(self.rmsd_curve)
#         l.addWidget(self.rmsd_plot)
#
#         win = CurveDialog()
#         self.drmsd_plot = win.get_plot()
#         self.drmsd_plot.set_titles(ylabel='dRMSD')
#         self.drmsd_curve = make.curve([],  [], color="r", linewidth=1)
#         self.drmsd_plot.add_item(self.drmsd_curve)
#         r.addWidget(self.drmsd_plot)
#
#         # Energy - Curves
#         top_left = QtGui.QFrame(self)
#         top_left.setFrameShape(QtGui.QFrame.StyledPanel)
#         l = QtGui.QVBoxLayout(top_left)
#
#         top_right = QtGui.QFrame(self)
#         top_right.setFrameShape(QtGui.QFrame.StyledPanel)
#         r = QtGui.QVBoxLayout(top_right)
#
#         splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
#         splitter.addWidget(top_left)
#         splitter.addWidget(top_right)
#
#         self.layout.addWidget(splitter)
#
#         win = CurveDialog()
#         self.fret_plot = win.get_plot()
#         self.fret_plot.set_titles(ylabel='FRET-Energy')
#         self.fret_curve = make.curve([],  [], color="m", linewidth=1)
#         self.fret_plot.add_item(self.fret_curve)
#         l.addWidget(self.fret_plot)
#
#         win = CurveDialog()
#         self.energy_plot = win.get_plot()
#         self.energy_plot.set_titles(ylabel='System-Energy')
#         self.energy_curve = make.curve([],  [], color="g", linewidth=1)
#         self.energy_plot.add_item(self.energy_curve)
#         r.addWidget(self.energy_plot)
#
#     def update_all(self, *args, **kwargs):
#
#         rmsd = np.array(self.trajectory.rmsd)
#         drmsd = np.array(self.trajectory.drmsd)
#         energy = np.array(self.trajectory.energy)
#         energy_fret = np.array(self.trajectory.chi2r)
#         x = list(range(len(rmsd)))
#
#         self.rmsd_curve.set_data(x, rmsd)
#         self.drmsd_curve.set_data(x, drmsd)
#         self.energy_curve.set_data(x, energy)
#         self.fret_curve.set_data(x, energy_fret)
#
#         self.energy_plot.do_autoscale()
#         self.fret_plot.do_autoscale()
#         self.rmsd_plot.do_autoscale()
#         self.drmsd_plot.do_autoscale()
