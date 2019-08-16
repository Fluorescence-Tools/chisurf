from __future__ import annotations

import os
import sys
import slugify
import numpy as np
import sip
sip.setapi('QDate', 2)
sip.setapi('QDateTime', 2)
sip.setapi('QString', 2)
sip.setapi('QTextStream', 2)
sip.setapi('QTime', 2)
sip.setapi('QUrl', 2)
sip.setapi('QVariant', 2)
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QApplication
import mfm.ui.resource
import tools


class Main(QMainWindow):

    @property
    def current_dataset(self):
        return self.dataset_selector.selected_dataset

    @current_dataset.setter
    def current_dataset(self, v):
        self.dataset_selector.selected_curve_index = v

    @property
    def current_model_class(self):
        return self.current_dataset.experiment.model_classes[self.comboBox_Model.currentIndex()]

    @property
    def fit_idx(self):
        subwindow = self.mdiarea.currentSubWindow()
        if subwindow is not None:
            for fit_idx, f in enumerate(mfm.fits):
                if f == subwindow.fit:
                    return fit_idx
        else:
            return None

    @property
    def current_experiment_idx(self):
        return self.comboBox_experimentSelect.currentIndex()

    @current_experiment_idx.setter
    def current_experiment_idx(self, v):
        self.comboBox_experimentSelect.setCurrentIndex(v)

    @property
    def current_experiment(self):
        return mfm.experiment[self.current_experiment_idx]

    @current_experiment.setter
    def current_experiment(self, name):
        p = self.comboBox_experimentSelect.currentIndex()
        n = p
        for n, e in enumerate(mfm.experiment):
            if e.name == name:
                break
        if p != n:
            self.current_experiment_idx = n

    @property
    def current_setup_idx(self):
        return self.comboBox_setupSelect.currentIndex()

    @current_setup_idx.setter
    def current_setup_idx(self, v):
        self.comboBox_setupSelect.setCurrentIndex(v)

    @property
    def current_setup(self):
        return self.current_experiment.setups[self.current_experiment_idx]

    @current_setup.setter
    def current_setup(self, name):
        i = self.current_setup_idx
        j = i
        setups = self.current_experiment.get_setups()
        for j, s in enumerate(setups):
            if s.name == name:
                break
        if j != i:
            self.current_setup_idx = j

    @property
    def current_model_name(self):
        return str(self.comboBox_Model.currentText())

    def closeEvent(self, event):
        if mfm.cs_settings['gui']['confirm_close_program']:
            reply = QtWidgets.QMessageBox.question(self,
                                               'Message',
                                               "Are you sure to quit?",
                                               QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
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
                    if self.current_fit is not mfm.fits[self.fit_idx]:
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
        filename = mfm.widgets.get_filename("Python macro", file_type="Python file (*.py)")
        mfm.run("mfm.console.run_macro(filename='%s')" % filename)

    def onTileWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        self.mdiarea.tileSubWindows()

    def onTabWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.TabbedView)

    def onCascadeWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        self.mdiarea.cascadeSubWindows ()

    def onCurrentDatasetChanged(self):
        #mfm.run("cs.current_dataset = %s" % self.curve_selector.selected_curve_index)
        self.comboBox_Model.clear()
        ds = self.current_dataset
        if mfm.data_sets:
            model_names = ds.experiment.get_model_names()
            self.comboBox_Model.addItems(model_names)

    def onAddFit(self):
        idxs = [r.row() for r in self.dataset_selector.selectedIndexes()]
        mfm.run("cs.add_fit(model_name='%s', dataset_idx=%s)" % (self.current_model_name, idxs))

    def add_fit(self, **kwargs):
        dataset_idx = kwargs.get('dataset_idx', [self.dataset_selector.selected_curve_index])
        datasets = [self.dataset_selector.datasets[i] for i in dataset_idx]
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
                fit.model.find_parameters()
                fit_control_widget = mfm.fitting.FittingControllerWidget(fit)

                self.modelLayout.addWidget(fit_control_widget)
                for f in fit:
                    self.modelLayout.addWidget(f.model)
                fit_window = mfm.fitting.FitSubWindow(fit,
                                                      control_layout=self.plotOptionsLayout,
                                                      fit_widget=fit_control_widget)
                fit_window = self.mdiarea.addSubWindow(fit_window)
                mfm.fit_windows.append(fit_window)
                fit_window.show()

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
        filename = mfm.widgets.get_filename(file_type="*.json", description="Load results into fit-model", **kwargs)
        mfm.run("mfm.fits[%s].model.load('%s')" % (self.fit_idx, filename))
        mfm.run("mfm.fits[%s].update()" % (self.fit_idx))

    def onSetupChanged(self):
        mfm.widgets.hide_items_in_layout(self.verticalLayout_5)
        setup_name = self.comboBox_setupSelect.currentText()
        mfm.run("cs.current_setup =cs.current_setup = '%s'" % setup_name)
        self.verticalLayout_5.addWidget(self.current_setup)
        self.current_setup.show()

    def onCloseAllFits(self):
        for sub_window in mfm.fit_windows:
            sub_window.widget().close_confirm = False
            sub_window.close()
        mfm.fits = []
        mfm.fit_windows = []

    def close_fit(self, idx=None):
        """
        
        :param idx: 
        :return:
        
        >>> print(1)
        1
        """
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

    def group_datasets(self, dataset_numbers):
        """
        :param dataset_numbers: list of datasets
        :return: 
        """
        #selected_data = mfm.data_sets[dataset_numbers]
        selected_data = [mfm.data_sets[i] for i in dataset_numbers]
        if isinstance(selected_data[0], mfm.curve.DataCurve):
            # TODO: check for double names!!!
            dg = mfm.curve.ExperimentDataCurveGroup(selected_data, name="Data-Group")
        else:
            dg = mfm.curve.ExperimentDataGroup(selected_data, name="Data-Group")
        dn = list()
        for d in mfm.data_sets:
            if d not in dg:
                dn.append(d)
        dn.append(dg)
        mfm.data_sets = dn

    def add_dataset(self, **kwargs):
        setup = kwargs.get('setup', self.current_setup)
        dataset = kwargs.pop('dataset', None)
        if dataset is None:
            dataset = setup.get_data(**kwargs)

        dataset_group = dataset if \
            isinstance(dataset, mfm.curve.ExperimentDataGroup) else \
            mfm.curve.ExperimentDataCurveGroup(dataset)
        if len(dataset_group) == 1:
            mfm.data_sets.append(dataset_group[0])
        else:
            mfm.data_sets.append(dataset_group)
        self.dataset_selector.update()

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
                    if fit_window.fit.data is d:
                        fit_window.close_confirm = False
                        fit_window.close()
                    else:
                        fw.append(fit_window)
                mfm.fit_windows = fw

        mfm.data_sets = l

    def save_fits(self, **kwargs):
        path = kwargs.get('path', mfm.widgets.get_directory(**kwargs))
        cf = self.fit_idx
        for fit in mfm.fits:
            fit_name = fit.name
            path_name = slugify.slugify(str(fit_name))
            p2 = path + '//' + path_name
            os.mkdir(p2)
            self.current_fit = fit
            self.save_fit(directory=p2)
        self.current_fit = mfm.fits[cf]

    def save_fit(self, **kwargs):
        directory = kwargs.pop('directory', None)
        if directory is None:
            mfm.working_path = mfm.widgets.get_directory(**kwargs)
        else:
            mfm.working_path = directory
        mfm.console.run_macro('./macro/save_fit.py')

    def __init__(self, *args, **kwargs):
        import mfm.experiments

        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        uic.loadUi("mfm/ui/mainwindow.ui", self)
        self.setCentralWidget(self.mdiarea)

        ##########################################################
        #      Fluorescence widgets                              #
        #      (Commented widgets don't work at the moment       #
        ##########################################################
        self.lifetime_calc = mfm.tools.FRETCalculator()
        self.actionCalculator.triggered.connect(self.lifetime_calc.show)

        #self.kappa2_dist = mfm.tools.kappa2dist.Kappa2Dist()
        #self.connect(self.actionKappa2_Distribution, QtCore.SIGNAL('triggered()'), self.kappa2_dist.show)

        #self.decay_generator = mfm.tools.dye_diffusion.TransientDecayGenerator()
        #self.connect(self.actionDye_Diffusion, QtCore.SIGNAL('triggered()'), self.decay_generator.show)

        #self.fret_lines = mfm.tools.fret_lines.FRETLineGeneratorWidget()
        #self.connect(self.actionFRET_Lines, QtCore.SIGNAL('triggered()'), self.fret_lines.show)

        #self.decay_fret_generator = mfm.fluorescence.dye_diffusion.TransientFRETDecayGenerator()

        ##########################################################
        #      TTTR-widgets                                      #
        ##########################################################
        self.tttr_convert = mfm.tools.TTTRConvert()
        self.actionConvert.triggered.connect(self.tttr_convert.show)

        self.tttr_correlate = mfm.tools.CorrelateTTTR()
        self.actionCorrelate.triggered.connect(self.tttr_correlate.show)

        self.tttr_histogram = mfm.tools.HistogramTTTR()
        self.actionGenerate_decay.triggered.connect(self.tttr_histogram.show)

        ##########################################################
        #      TTTR-widgets                                      #
        ##########################################################
        #self.hdf2pdb = mfm.tools.MDConverter()
        #self.actionTrajectory_converter.triggered.connect(self.hdf2pdb.show)

        #self.trajectory_rot_trans = mfm.tools.RotateTranslateTrajectoryWidget()
        #self.actionRotate_Translate_trajectory.triggered.connect(self.trajectory_rot_trans.show)

        #self.calculate_potential = mfm.tools.PotentialEnergyWidget()
        #self.actionCalculate_Potential.triggered.connect(self.calculate_potential.show)

        self.pdb2label = mfm.tools.PDB2Label()
        self.actionPDB2Label.triggered.connect(self.pdb2label.show)

        self.structure2transfer = tools.traj2fret.gui.Structure2Transfer()
        self.actionStructure2Transfer.triggered.connect(self.structure2transfer.show)

        #self.join_trajectories = mfm.tools.JoinTrajectoriesWidget()
        #self.actionJoin_trajectories.triggered.connect(self.join_trajectories.show)

        #self.traj_save_topol = mfm.tools.SaveTopology()
        #self.actionSave_topology.triggered.connect(self.traj_save_topol.show)

        #self.remove_clashes = mfm.tools.RemoveClashedFrames()
        #self.actionRemove_clashes.triggered.connect(self.remove_clashes.show)

        #self.align_trajectory = mfm.tools.AlignTrajectoryWidget()
        #self.actionAlign_trajectory.triggered.connect(self.align_trajectory.show)

        #self.update_widget = mfm.widgets.downloader.UpdateDialog()
        #self.connect(self.actionUpdate, QtCore.SIGNAL('triggered()'), self.update_widget.show)

        self.f_test = mfm.tools.FTestWidget()
        self.actionF_Test.triggered.connect(self.f_test.show)

        self.configuration = mfm.widgets.CodeEditor(
            filename=mfm.settings.settings_file,
            language='YAML',
            can_load=False
        )
        self.actionSettings.triggered.connect(self.configuration.show)

        ##########################################################
        #      Help and About widgets                            #
        ##########################################################
        self.about = uic.loadUi("mfm/ui/about.ui")
        self.about.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.about.hide()
        self.actionHelp_2.triggered.connect(self.about.show)
        self.actionAbout.triggered.connect(self.about.show)

        ##########################################################
        #      IPython console                                   #
        #      Push variables to console and add it to           #
        #      user interface                                    #
        ##########################################################
        self.dockWidgetScriptEdit.setVisible(mfm.cs_settings['gui']['show_macro_edit'])
        self.dockWidget_console.setVisible(mfm.cs_settings['gui']['show_console'])

        self.verticalLayout_4.addWidget(mfm.console)
        mfm.console.pushVariables({'cs': self})
        mfm.console.pushVariables({'mfm': mfm})
        mfm.console.pushVariables({'np': np})
        #mfm.console.pushVariables({'cl': mfm.pyopencl})
        mfm.console.pushVariables({'os': os})
        mfm.console.pushVariables({'QtCore': QtCore})
        mfm.console.pushVariables({'QtGui': QtGui})
        mfm.run = mfm.console.execute
        mfm.run(str(mfm.cs_settings['gui']['console']['init']))

        ##########################################################
        #      Record and run recorded macros                    #
        ##########################################################
        self.actionRecord.triggered.connect(mfm.console.start_recording)
        self.actionStop.triggered.connect(mfm.console.save_macro)
        self.actionRun.triggered.connect(self.onRunMacro)

        ##########################################################
        #      Arrange Docks and window positions                #
        #      Window-controls tile, stack etc.                  #
        ##########################################################
        self.tabifyDockWidget(self.dockWidgetDatasets, self.dockWidgetAnalysis)
        self.tabifyDockWidget(self.dockWidgetAnalysis, self.dockWidgetPlot)
        self.tabifyDockWidget(self.dockWidgetPlot, self.dockWidgetFits)
        #self.tabifyDockWidget(self.dockWidgetPlot, self.dockWidgetScriptEdit)
        self.editor = mfm.widgets.CodeEditor()
        self.verticalLayout_10.addWidget(self.editor)

        self.modelLayout.setAlignment(QtCore.Qt.AlignTop)
        self.plotOptionsLayout.setAlignment(QtCore.Qt.AlignTop)
        self.dockWidgetDatasets.raise_()

        self.actionTile_windows.triggered.connect(self.onTileWindows)
        self.actionTab_windows.triggered.connect(self.onTabWindows)
        self.actionCascade.triggered.connect(self.onCascadeWindows)
        self.mdiarea.subWindowActivated.connect(self.subWindowActivated)
        #self.groupBox_4.setChecked(False)

        ##########################################################
        #    Connect changes in User-interface to actions like:  #
        #    Loading dataset, changing setups, models, etc.     #
        ##########################################################
        self.actionSetupChanged.triggered.connect(self.onSetupChanged)
        self.actionExperimentChanged.triggered.connect(self.onExperimentChanged)
        self.actionChange_current_dataset.triggered.connect(self.onCurrentDatasetChanged)
        self.actionAdd_fit.triggered.connect(self.onAddFit)
        self.actionSaveAllFits.triggered.connect(self.save_fits)
        self.actionSaveCurrentFit.triggered.connect(self.save_fit)
        self.actionClose_Fit.triggered.connect(self.close_fit)
        self.actionClose_all_fits.triggered.connect(self.onCloseAllFits)
        self.actionLoad_Data.triggered.connect(self.add_dataset)
        self.actionLoad_result_in_current_fit.triggered.connect(self.onLoadFitResults)

        self.dataset_selector = mfm.widgets.CurveSelector(click_close=False, curve_types='all',
                                                          change_event=self.onCurrentDatasetChanged,
                                                          drag_enabled=True)
        self.verticalLayout_8.addWidget(self.dataset_selector)

        ##########################################################
        #      Initialize Experiments and Setups                 #
        #      (Commented widgets don't work at the moment       #
        ##########################################################
        tcspc = mfm.experiments.Experiment('TCSPC')
        tcspc.add_setups(mfm.experiments.tcspc_setups)
        tcspc.add_models(mfm.fitting.models.tcspc.models)
        mfm.experiment.append(tcspc)

        fcs = mfm.experiments.Experiment('FCS')
        fcs.add_setups(mfm.experiments.fcs_setups)
        fcs.add_models(mfm.fitting.models.fcs.models)
        mfm.experiment.append(fcs)

        global_fit = mfm.experiments.Experiment('Global')
        global_setup = mfm.experiments.GlobalFitSetup(name='Global-Fit', experiment=global_fit)
        global_fit.add_model(mfm.fitting.models.GlobalFitModelWidget)
        global_fit.add_setup(global_setup)
        mfm.experiment.append(global_fit)

        self.experiment_names = [b.name for b in mfm.experiment if b.name is not 'Global']
        self.comboBox_experimentSelect.addItems(self.experiment_names)

        self.current_fit = None
        self.add_dataset(experiment=global_fit, setup=global_setup)  # Add Global-Dataset by default


if __name__ == "__main__":
    app = QApplication(sys.argv)
    import mfm
    import mfm.widgets
    import mfm.tools
    mfm.console = mfm.widgets.QIPythonWidget()

    # See: https://github.com/ipython/ipykernel/issues/370
    # this should be fixed newer
    def _abort_queues(kernel):
        pass
    mfm.console.kernel_manager.kernel._abort_queues = _abort_queues

    win = Main(parent=None)
    mfm.console.history_widget = win.plainTextEditHistory
    mfm.cs = win

    style_sheet = open(mfm.settings.style_sheet_file, 'r').read()
    app.setStyleSheet(style_sheet)

    win.show()
    sys.exit(app.exec_())
