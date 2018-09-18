"""
Created on 13 June 2016
@author: Thomas Peulen
"""

# After updating of icon run:
# pyrcc4 rescource.qrc -o rescource_rc.py

from __future__ import absolute_import
#import demandimport; demandimport.enable()

import os
import sys
import sip
sip.setapi('QDate', 2)
sip.setapi('QDateTime', 2)
sip.setapi('QString', 2)
sip.setapi('QTextStream', 2)
sip.setapi('QTime', 2)
sip.setapi('QUrl', 2)
sip.setapi('QVariant', 2)

import numpy as np
from PyQt4 import QtGui, QtCore, uic
from slugify import slugify
import mfm


class Main(QtGui.QMainWindow):

    @property
    def current_dataset(self):
        return self.curve_selector.selected_dataset

    @current_dataset.setter
    def current_dataset(self, v):
        self.curve_selector.selected_curve_index = v

    @property
    def current_model_class(self):
        return self.current_dataset.experiment.model_classes[self.comboBox_Model.currentIndex()]

    @property
    def current_experiment(self):
        return mfm.experiment[self.comboBox_experimentSelect.currentIndex()]

    @property
    def fit_idx(self):
        subwindow = self.mdiarea.currentSubWindow()
        if subwindow is not None:
            for fit_idx, f in enumerate(mfm.fits):
                if f == subwindow.fit:
                    return fit_idx
        else:
            return -1

    @current_experiment.setter
    def current_experiment(self, name):
        i = self.comboBox_experimentSelect.currentIndex()
        j = i
        for j, e in enumerate(mfm.experiment):
            if e.name == name:
                break
        if i != j:
            self.comboBox_experimentSelect.setCurrentIndex(j)

    @property
    def current_setup(self):
        return self.current_experiment.get_setups()[self.comboBox_setupSelect.currentIndex()]

    @current_setup.setter
    def current_setup(self, name):
        i = self.comboBox_setupSelect.currentIndex()
        j = i
        setups = self.current_experiment.get_setups()
        for j, s in enumerate(setups):
            if s.name == name:
                break
        if j != i:
            self.comboBox_setupSelect.setCurrentIndex(j)

    @property
    def current_model_name(self):
        return str(self.comboBox_Model.currentText())

    def closeEvent(self, event):
        if mfm.settings['gui']['confirm_close_program']:
            reply = QtGui.QMessageBox.question(self,
                                               'Message',
                                               "Are you sure to quit?",
                                               QtGui.QMessageBox.Yes,
                                               QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def subWindowActivated(self):
        subwindow = self.mdiarea.currentSubWindow()
        if subwindow is not None:
            for fit_idx, f in enumerate(mfm.fits):
                if f == subwindow.fit:
                    mfm.run("cs.current_fit = mfm.fits[%s]" % self.fit_idx)
                    break
            self.current_fit_widget = subwindow.fit_widget
            fit_name = self.current_fit.name
            window_title = mfm.__name__ + "(" + mfm.__version__ + "): " + fit_name

            self.setWindowTitle(window_title)

            mfm.widgets.hide_items_in_layout(self.modelLayout)
            mfm.widgets.hide_items_in_layout(self.plotOptionsLayout)
            self.current_fit.model.update()
            self.current_fit.model.show()
            self.current_fit_widget.show()
            subwindow.current_plt_ctrl.show()

    def onRunMacro(self):
        filename = mfm.widgets.open_file("Python macro", file_type="Python file (*.py)")
        mfm.run("mfm.console.run_macro(filename='%s')" % filename)

    def onTileWindows(self):
        self.mdiarea.setViewMode(QtGui.QMdiArea.SubWindowView)
        self.mdiarea.tileSubWindows()

    def onTabWindows(self):
        self.mdiarea.setViewMode(QtGui.QMdiArea.TabbedView)

    def onCascadeWindows(self):
        self.mdiarea.setViewMode(QtGui.QMdiArea.SubWindowView)
        self.mdiarea.cascadeSubWindows ()

    def onCurrentDatasetChanged(self):
        #mfm.run("cs.current_dataset = %s" % self.curve_selector.selected_curve_index)
        self.comboBox_Model.clear()
        ds = self.current_dataset
        if mfm.data_sets:
            model_names = ds.experiment.model_names
            self.comboBox_Model.addItems(model_names)

    def onAddFit(self):
        idxs = [r.row() for r in self.curve_selector.selectedIndexes()]
        mfm.run("cs.add_fit(model_name='%s', dataset_idx=%s)" % (self.current_model_name, idxs))
        #mfm.run("cs.current_fit = mfm.fits[-1]")

    def add_fit(self, **kwargs):
        dataset_idx = kwargs.get('dataset_idx', [self.curve_selector.selected_curve_index])
        datasets = [self.curve_selector.datasets[i] for i in dataset_idx]
        model_name = kwargs.get('model_name', self.current_model_name)

        model_names = datasets[0].experiment.model_names
        model_class = datasets[0].experiment.model_classes[0]
        for model_idx, mn in enumerate(model_names):
            if mn == model_name:
                model_class = datasets[0].experiment.model_classes[model_idx]
                break

        for data_set in datasets:
            if data_set.experiment is datasets[0].experiment:
                if not isinstance(data_set, mfm.curve.DataGroup):
                    data_set = mfm.curve.ExperimentDataCurveGroup(data_set)

                fit = mfm.fitting.FitGroup(data=data_set, model_class=model_class)
                mfm.fits.append(fit)
                fit_control_widget = mfm.fitting.FittingControllerWidget(fit)

                self.modelLayout.addWidget(fit_control_widget)
                for f in fit:
                    self.modelLayout.addWidget(f.model)
                fit_window = mfm.fitting.FitSubWindow(fit,
                                                      control_layout=self.plotOptionsLayout,
                                                      parent=self.mdiarea,
                                                      fit_widget=fit_control_widget)
                fit_window = self.mdiarea.addSubWindow(fit_window)
                fit_window.resize(mfm.settings['gui']['fit_windows_size_x'], mfm.settings['gui']['fit_windows_size_y'])
                fit_window.setOption(QtGui.QMdiSubWindow.RubberBandResize, mfm.settings['gui']['RubberBandResize'])
                fit_window.setOption(QtGui.QMdiSubWindow.RubberBandMove, mfm.settings['gui']['RubberBandMove'])
                try:
                    fit_window.setWindowIcon(fit.model.icon)
                except AttributeError:
                    fit_window.setWindowIcon(QtGui.QIcon(":/icons/icons/list-add.png"))
                mfm.fit_windows.append(fit_window)
                fit_window.show()
                #self.current_fit = fit
                fit.update()
                #mfm.fits.append(fit)

    def onExperimentChanged(self):
        experiment_name = self.comboBox_experimentSelect.currentText()
        mfm.run("cs.current_experiment = '%s'" % experiment_name)
        # Add setups for selected experiment
        self.comboBox_setupSelect.blockSignals(True)
        self.comboBox_setupSelect.clear()
        self.comboBox_setupSelect.addItems(self.current_experiment.setup_names)
        self.comboBox_setupSelect.blockSignals(False)
        self.onSetupChanged()

    def onLoadFitResults(self, **kwargs):
        filename = mfm.widgets.open_file(file_type="*.json", description="Load results into fit-model", **kwargs)
        mfm.run("mfm.fits[%s].model.load('%s')" % (self.fit_idx, filename))
        mfm.run("mfm.fits[%s].update()" % (self.fit_idx))

    def onSetupChanged(self):
        mfm.widgets.hide_items_in_layout(self.verticalLayout_5)
        setup_name = self.comboBox_setupSelect.currentText()
        mfm.run("cs.current_setup = '%s'" % setup_name)
        self.verticalLayout_5.addWidget(self.current_setup)
        self.current_setup.show()

    def onCloseAllFits(self):
        for sub_window in mfm.fit_windows:
            sub_window.widget().close_confirm = False
            sub_window.close()
        mfm.fits = []
        mfm.fit_windows = []

    def close_fit(self, idx=None):
        if idx is None:
            sub_window = self.mdiarea.currentSubWindow()
            for i, w in enumerate(mfm.fit_windows):
                if w is sub_window:
                    idx = i
        mfm.fits.pop(idx)
        sub_window = mfm.fit_windows.pop(idx)
        sub_window.close_confirm = False
        mfm.widgets.hide_items_in_layout(self.modelLayout)
        mfm.widgets.hide_items_in_layout(self.plotOptionsLayout)
        sub_window.close()

    def add_dataset(self, **kwargs):
        setup = kwargs.get('setup', self.current_setup)
        experiment = kwargs.get('experiment', self.current_experiment)
        dataset = kwargs.pop('dataset', None)

        if dataset is None:
            dataset = setup.get_data(**kwargs)
        if not isinstance(dataset, mfm.curve.ExperimentDataGroup):
            dataset_group = mfm.curve.ExperimentDataCurveGroup(dataset)
        else:
            dataset_group = dataset

        # Include Data in Database
        for d in dataset_group:
            if isinstance(d, mfm.curve.ExperimentalData):
                mfm.data_sets.append(d)
                d.experiment = experiment
            mfm.db_session.add(d)
            mfm.db_session.commit()

        self.curve_selector.update()

    def remove_dataset(self, idx):
        idx = [idx] if not isinstance(idx, list) else idx

        l = list()
        for i, d in enumerate(mfm.data_sets):
            if d.name == 'Global-fit':
                l.append(d)
                continue
            if i not in idx:
                l.append(d)
            else:
                fw = list()
                for fit_window in mfm.fit_windows:
                    if fit_window.widget().fit.data is d:
                        fit_window.widget().close_confirm = False
                        fit_window.close()
                    else:
                        fw.append(fit_window)
                mfm.fit_windows = fw

        mfm.data_sets = l

    def save_fits(self, **kwargs):
        path = kwargs.get('path', mfm.widgets.get_directory(**kwargs))
        for i, fw in enumerate(mfm.fit_windows):
            fit_name = fw.fit_widget.fit.name
            path_name = slugify(unicode(fit_name))
            p2 = path + '//' + path_name
            os.mkdir(p2)
            self.save_fit(directory=p2)

    def save_fit(self, **kwargs):
        directory = kwargs.pop('directory', None)
        if directory is None:
            mfm.working_path = mfm.widgets.get_directory(**kwargs)
        else:
            mfm.working_path = directory
        mfm.console.run_macro('./macro/save_fit.py')

    # def update(self, *__args):
    #     QtGui.QMainWindow.update(self, *__args)
    #     for f in mfm.fits:
    #         f.update()

    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)
        uic.loadUi("mfm/ui/mainwindow.ui", self)
        self.setCentralWidget(self.mdiarea)

        ##########################################################
        #      Fluorescence widgets                              #
        #      (Commented widgets don't work at the moment       #
        ##########################################################
        self.lifetime_calc = mfm.tools.FRETCalculator()
        self.connect(self.actionCalculator, QtCore.SIGNAL('triggered()'), self.lifetime_calc.show)

        self.kappa2_dist = mfm.tools.kappa2dist.Kappa2Dist()
        self.connect(self.actionKappa2_Distribution, QtCore.SIGNAL('triggered()'), self.kappa2_dist.show)

        #self.decay_generator = mfm.tools.dye_diffusion.TransientDecayGenerator()
        #self.connect(self.actionDye_Diffusion, QtCore.SIGNAL('triggered()'), self.decay_generator.show)

        #self.fret_lines = mfm.tools.fret_lines.FRETLineGeneratorWidget()
        #self.connect(self.actionFRET_Lines, QtCore.SIGNAL('triggered()'), self.fret_lines.show)

        #self.decay_fret_generator = mfm.fluorescence.dye_diffusion.TransientFRETDecayGenerator()

        ##########################################################
        #      TTTR-widgets                                      #
        ##########################################################
        self.tttr_convert = mfm.tools.TTTRConvert()
        self.connect(self.actionConvert, QtCore.SIGNAL('triggered()'), self.tttr_convert.show)

        self.tttr_correlate = mfm.tools.CorrelateTTTR()
        self.connect(self.actionCorrelate, QtCore.SIGNAL('triggered()'), self.tttr_correlate.show)

        self.tttr_histogram = mfm.tools.HistogramTTTR()
        self.connect(self.actionGenerate_decay, QtCore.SIGNAL('triggered()'), self.tttr_histogram.show)

        ##########################################################
        #      TTTR-widgets                                      #
        ##########################################################
        self.hdf2pdb = mfm.tools.MDConverter()
        self.connect(self.actionTrajectory_converter, QtCore.SIGNAL('triggered()'), self.hdf2pdb.show)

        self.trajectory_rot_trans = mfm.tools.RotateTranslateTrajectoryWidget()
        self.connect(self.actionRotate_Translate_trajectory, QtCore.SIGNAL('triggered()'), self.trajectory_rot_trans.show)

        self.calculate_potential = mfm.tools.PotentialEnergyWidget()
        self.connect(self.actionCalculate_Potential, QtCore.SIGNAL('triggered()'), self.calculate_potential.show)

        self.pdb2label = mfm.tools.PDB2Label()
        self.connect(self.actionPDB2Label, QtCore.SIGNAL('triggered()'), self.pdb2label.show)

        self.structure2transfer = mfm.tools.Structure2Transfer()
        self.connect(self.actionStructure2Transfer, QtCore.SIGNAL('triggered()'), self.structure2transfer.show)

        self.join_trajectories = mfm.tools.JoinTrajectoriesWidget()
        self.connect(self.actionJoin_trajectories, QtCore.SIGNAL('triggered()'), self.join_trajectories.show)

        self.traj_save_topol = mfm.tools.SaveTopology()
        self.connect(self.actionSave_topology, QtCore.SIGNAL('triggered()'), self.traj_save_topol.show)

        self.remove_clashes = mfm.tools.RemoveClashedFrames()
        self.connect(self.actionRemove_clashes, QtCore.SIGNAL('triggered()'), self.remove_clashes.show)

        self.align_trajectory = mfm.tools.AlignTrajectoryWidget()
        self.connect(self.actionAlign_trajectory, QtCore.SIGNAL('triggered()'), self.align_trajectory.show)

        #self.update_widget = mfm.widgets.downloader.UpdateDialog()
        #self.connect(self.actionUpdate, QtCore.SIGNAL('triggered()'), self.update_widget.show)

        self.f_test = mfm.tools.FTestWidget()
        self.connect(self.actionF_Test, QtCore.SIGNAL('triggered()'), self.f_test.show)

        #self.configuration = mfm.widgets.configuration_editor.ParameterEditor(json_file=mfm.settings_file, target=mfm)
        #self.connect(self.actionSettings, QtCore.SIGNAL('triggered()'), self.configuration.show)

        ##########################################################
        #      Help and About widgets                            #
        ##########################################################
        self.about = uic.loadUi("mfm/ui/about.ui")
        self.about.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.about.hide()
        self.connect(self.actionHelp_2, QtCore.SIGNAL('triggered()'), self.about.show)
        self.connect(self.actionAbout, QtCore.SIGNAL('triggered()'), self.about.show)

        ##########################################################
        #      IPython console                                   #
        #      Push varibels to console and add it to            #
        #      user interface                                    #
        ##########################################################
        if not mfm.settings['gui']['show_console']:
            self.dockWidget_console.hide()
        self.verticalLayout_4.addWidget(mfm.console)
        mfm.console.pushVariables({'cs': self})
        mfm.console.pushVariables({'mfm': mfm})
        mfm.console.pushVariables({'np': np})
        mfm.console.pushVariables({'cl': mfm.pyopencl})
        mfm.console.pushVariables({'os': os})
        mfm.console.pushVariables({'QtCore': QtCore})
        mfm.console.pushVariables({'QtGui': QtGui})
        mfm.run = mfm.console.execute
        # ZMQ test
        #mfm.run = mfm.socket.send_string
        mfm.run(str(mfm.settings['gui']['console']['init']))

        ##########################################################
        #      Record and run recorded macros                    #
        ##########################################################
        self.connect(self.actionRecord, QtCore.SIGNAL('triggered()'), mfm.console.start_recording)
        self.connect(self.actionStop, QtCore.SIGNAL('triggered()'), mfm.console.save_macro)
        self.connect(self.actionRun, QtCore.SIGNAL('triggered()'), self.onRunMacro)

        ##########################################################
        #      Arrange Docks and window positions                #
        #      Window-controls tile, stack etc.                  #
        ##########################################################
        self.tabifyDockWidget(self.dockWidget_load, self.dockWidget_fit)
        self.tabifyDockWidget(self.dockWidget_fit, self.dockWidget_plot)
        #self.tabifyDockWidget(self.dockWidget_plot, self.dockWidget_console)
        self.modelLayout.setAlignment(QtCore.Qt.AlignTop)
        self.plotOptionsLayout.setAlignment(QtCore.Qt.AlignTop)
        self.dockWidget_load.raise_()

        self.connect(self.actionTile_windows, QtCore.SIGNAL('triggered()'), self.onTileWindows)
        self.connect(self.actionTab_windows, QtCore.SIGNAL('triggered()'), self.onTabWindows)
        self.connect(self.actionCascade, QtCore.SIGNAL('triggered()'), self.onCascadeWindows)
        self.mdiarea.subWindowActivated.connect(self.subWindowActivated)
        self.groupBox_4.setChecked(False)

        ##########################################################
        #    Connect changes in User-interface to actions like:  #
        #    Loading dataset, changing setups, models, etc.     #
        ##########################################################
        self.connect(self.actionSetupChanged, QtCore.SIGNAL('triggered()'), self.onSetupChanged)
        self.connect(self.actionExperimentChanged, QtCore.SIGNAL('triggered()'), self.onExperimentChanged)
        self.connect(self.actionChange_current_dataset, QtCore.SIGNAL('triggered()'), self.onCurrentDatasetChanged)
        self.connect(self.actionAdd_fit, QtCore.SIGNAL('triggered()'), self.onAddFit)
        self.connect(self.actionSaveAllFits, QtCore.SIGNAL('triggered()'), self.save_fits)
        self.connect(self.actionSaveCurrentFit, QtCore.SIGNAL('triggered()'), self.save_fit)
        self.connect(self.actionClose_Fit, QtCore.SIGNAL('triggered()'), self.close_fit)
        self.connect(self.actionClose_all_fits, QtCore.SIGNAL('triggered()'), self.onCloseAllFits)
        self.connect(self.actionLoad_Data, QtCore.SIGNAL('triggered()'), self.add_dataset)
        self.connect(self.actionLoad_result_in_current_fit, QtCore.SIGNAL('triggered()'), self.onLoadFitResults)

        self.curve_selector = mfm.widgets.CurveSelector(click_close=False, curve_types='all',
                                                        change_event=self.onCurrentDatasetChanged,
                                                        drag_enabled=True)
        self.verticalLayout_8.addWidget(self.curve_selector)

        ##########################################################
        #      Initialize Experiments and Setups                 #
        #      (Commented widgets don't work at the moment       #
        ##########################################################
        tcspc = mfm.experiments.Experiment('TCSPC')
        tcspc_setups = [mfm.experiments.TCSPCSetupWidget(name="CSV/PQ/IBH", **mfm.settings['tcspc']),
                        mfm.experiments.TCSPCSetupSDTWidget()]
        tcspc.add_setups(tcspc_setups)
        tcspc.add_models(mfm.fitting.models.tcspc.models)
        mfm.experiment.append(tcspc)

        fcs = mfm.experiments.Experiment('FCS')
        fcs.add_setup(mfm.experiments.FCSKristine(experiment=fcs))
        fcs.add_setup(mfm.experiments.FCSCsv(experiment=fcs))
        fcs.add_models(mfm.fitting.models.fcs.models)
        mfm.experiment.append(fcs)

        stopped_flow = mfm.experiments.Experiment('Stopped-Flow')
        stopped_flow.add_setup(
        mfm.io.widgets.CSVFileWidget(name='CSV', weight_calculation=np.sqrt, skiprows=0, use_header=True))
        stopped_flow.add_model(mfm.fitting.models.stopped_flow.ParseStoppedFlowWidget)
        #stopped_flow.add_model(mfm.fitting.models.stopped_flow.ReactionWidget)
        mfm.experiment.append(stopped_flow)

        modelling = mfm.experiments.Experiment('Modelling')
        #modelling.addSetup(e.setups.modelling.LoadStructureFolder())
        modelling.add_setup(mfm.experiments.modelling.LoadStructure(name='PDB'))
        modelling.add_model(mfm.fitting.models.structure.proteinMC.ProteinMonteCarloControlWidget)
        mfm.experiment.append(modelling)

        global_fit = mfm.experiments.Experiment('Global')
        global_setup = mfm.experiments.GlobalFitSetup(name='Global-Fit', experiment=global_fit)
        global_fit.add_model(mfm.fitting.models.GlobalFitModelWidget)
        global_fit.add_model(mfm.fitting.models.tcspc.et.EtModelFreeWidget)
        global_fit.add_setup(global_setup)
        mfm.experiment.append(global_fit)

        self.experiment_names = [b.name for b in mfm.experiment if b.name is not 'Global']
        self.comboBox_experimentSelect.addItems(self.experiment_names)

        self.current_fit = None
        self.add_dataset(experiment=global_fit, setup=global_setup)  # Add Global-Dataset by default


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    app.setStyleSheet(mfm.style_sheet)
    #app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))
    #app.setStyle(QtGui.QStyleFactory.create("Fusion"))
    mfm.console = mfm.widgets.QIPythonWidget()
    mfm.console.set_default_style('linux')
    import pkg_resources
    color_theme = 'monokai'  # specify color theme
    ss = open(pkg_resources.resource_filename("jupyter_qtconsole_colorschemes", "{}.css".format(color_theme))).read()
    mfm.console.style_sheet = mfm.style_sheet
    mfm.console.syntax_style = color_theme
    win = Main(parent=None)
    win.show()
    sys.exit(app.exec_())
    mfm.console.run_macro(sys.argv[1])
