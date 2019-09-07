from __future__ import annotations

import os
import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QApplication
import mfm.cmd


class Main(QMainWindow):

    @property
    def current_dataset(self):
        return self.dataset_selector.selected_dataset

    @current_dataset.setter
    def current_dataset(
            self,
            dataset_index: int
    ):
        self.dataset_selector.selected_curve_index = dataset_index

    @property
    def current_model_class(self):
        return self.current_dataset.experiment.model_classes[
            self.comboBox_Model.currentIndex()
        ]

    @property
    def fit_idx(self) -> int:
        subwindow = self.mdiarea.currentSubWindow()
        if subwindow is not None:
            for fit_idx, f in enumerate(mfm.fits):
                if f == subwindow.fit:
                    return fit_idx
        else:
            return None

    @property
    def current_experiment_idx(self) -> int:
        return self.comboBox_experimentSelect.currentIndex()

    @current_experiment_idx.setter
    def current_experiment_idx(
            self,
            v: int
    ):
        self.comboBox_experimentSelect.setCurrentIndex(v)

    @property
    def current_experiment(self):
        return mfm.experiment[self.current_experiment_idx]

    @current_experiment.setter
    def current_experiment(
            self,
            name: str
    ) -> None:
        p = self.comboBox_experimentSelect.currentIndex()
        n = p
        for n, e in enumerate(mfm.experiment):
            if e.name == name:
                break
        if p != n:
            self.current_experiment_idx = n

    @property
    def current_setup_idx(self) -> int:
        return self.comboBox_setupSelect.currentIndex()

    @current_setup_idx.setter
    def current_setup_idx(
            self,
            v: int
    ):
        self.comboBox_setupSelect.setCurrentIndex(v)

    @property
    def current_setup(self):
        return self.current_experiment.setups[self.current_experiment_idx]

    @current_setup.setter
    def current_setup(
            self,
            name: str
    ):
        i = self.current_setup_idx
        j = i
        setups = self.current_experiment.get_setups()
        for j, s in enumerate(setups):
            if s.name == name:
                break
        if j != i:
            self.current_setup_idx = j

    @property
    def current_model_name(self) -> str:
        return str(self.comboBox_Model.currentText())

    def closeEvent(self, event):
        if mfm.settings.cs_settings['gui']['confirm_close_program']:
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
            for f in mfm.fits:
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
        filename = mfm.widgets.get_filename("Python macros", file_type="Python file (*.py)")
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
        if mfm.imported_datasets:
            model_names = ds.experiment.get_model_names()
            self.comboBox_Model.addItems(model_names)

    def onAddFit(self):
        mfm.run(
            "mfm.cmd.add_fit(model_name='%s', dataset_indices=%s)" %
            (self.current_model_name, [r.row() for r in self.dataset_selector.selectedIndexes()])
        )

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
        filename = mfm.widgets.get_filename(file_type="*.json", description="Load results into fit-models", **kwargs)
        mfm.run("mfm.cmd.load_fit_result(%s, %s)" % (self.fit_idx, filename))

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

    def onAddDataset(self):
        mfm.cmd.add_dataset(self.current_setup)

    def onSaveFits(self, **kwargs):
        path = kwargs.get('path', mfm.widgets.get_directory(**kwargs))
        mfm.cmd.save_fits(path)

    def onSaveFit(self, **kwargs):
        directory = kwargs.pop('directory', None)
        if directory is None:
            mfm.working_path = mfm.widgets.get_directory(**kwargs)
        else:
            mfm.working_path = directory
        mfm.console.run('mfm.cmd.save_fit()')

    def init_widgets(self):
        #self.decay_generator = mfm.tools.dye_diffusion.TransientDecayGenerator()
        #self.connect(self.actionDye_Diffusion, QtCore.SIGNAL('triggered()'), self.decay_generator.show)

        #self.fret_lines = mfm.tools.fret_lines.FRETLineGeneratorWidget()
        #self.connect(self.actionFRET_Lines, QtCore.SIGNAL('triggered()'), self.fret_lines.show)

        #self.decay_fret_generator = mfm.fluorescence.dye_diffusion.TransientFRETDecayGenerator()


        ##########################################################
        #      Fluorescence widgets                              #
        #      (Commented widgets don't work at the moment       #
        ##########################################################
        self.lifetime_calc = mfm.tools.fret_calculator.tau2r.FRETCalculator()
        self.actionCalculator.triggered.connect(self.lifetime_calc.show)

        self.kappa2_dist = mfm.tools.kappa2_distribution.kappa2dist.Kappa2Dist()
        self.actionKappa2_Distribution.triggered.connect(self.kappa2_dist.show)

        ##########################################################
        #      TTTR-widgets                                      #
        ##########################################################
        self.tttr_convert = mfm.tools.TTTRConvert()
        self.actionConvert.triggered.connect(self.tttr_convert.show)

        self.tttr_correlate = mfm.tools.CorrelateTTTR()
        self.actionCorrelate.triggered.connect(self.tttr_correlate.show)

        self.tttr_histogram = mfm.tools.tttr.decay_histogram.HistogramTTTR()
        self.actionGenerate_decay.triggered.connect(self.tttr_histogram.show)

        ##########################################################
        #      TTTR-widgets                                      #
        ##########################################################
        self.hdf2pdb = mfm.tools.modelling.trajectory.MDConverter()
        self.actionTrajectory_converter.triggered.connect(self.hdf2pdb.show)

        self.trajectory_rot_trans = mfm.tools.modelling.trajectory.RotateTranslateTrajectoryWidget()
        self.actionRotate_Translate_trajectory.triggered.connect(self.trajectory_rot_trans.show)

        #self.calculate_potential = mfm.tools.PotentialEnergyWidget()
        #self.actionCalculate_Potential.triggered.connect(self.calculate_potential.show)

        self.pdb2label = mfm.tools.PDB2Label()
        self.actionPDB2Label.triggered.connect(self.pdb2label.show)

        self.structure2transfer = mfm.tools.traj2fret.gui.Structure2Transfer()
        self.actionStructure2Transfer.triggered.connect(self.structure2transfer.show)

        self.join_trajectories = mfm.tools.modelling.trajectory.JoinTrajectoriesWidget()
        self.actionJoin_trajectories.triggered.connect(self.join_trajectories.show)

        self.traj_save_topol = mfm.tools.modelling.trajectory.SaveTopology()
        self.actionSave_topology.triggered.connect(self.traj_save_topol.show)

        #self.remove_clashes = mfm.tools.RemoveClashedFrames()
        #self.actionRemove_clashes.triggered.connect(self.remove_clashes.show)

        self.align_trajectory = mfm.tools.modelling.trajectory.AlignTrajectoryWidget()
        self.actionAlign_trajectory.triggered.connect(self.align_trajectory.show)

        #self.update_widget = mfm.widgets.downloader.UpdateDialog()
        #self.connect(self.actionUpdate, QtCore.SIGNAL('triggered()'), self.update_widget.show)

        self.f_test = mfm.tools.FTestWidget()
        self.actionF_Test.triggered.connect(self.f_test.show)

    def __init__(self, *args, **kwargs):
        import mfm.experiments

        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        uic.loadUi("mfm/ui/mainwindow.ui", self)
        self.setCentralWidget(self.mdiarea)

        self.init_widgets()

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
        self.dockWidgetScriptEdit.setVisible(mfm.settings.cs_settings['gui']['show_macro_edit'])
        self.dockWidget_console.setVisible(mfm.settings.cs_settings['gui']['show_console'])

        self.verticalLayout_4.addWidget(mfm.console)
        mfm.console.pushVariables({'cs': self})
        mfm.console.pushVariables({'mfm': mfm})
        mfm.console.pushVariables({'np': np})
        mfm.console.pushVariables({'os': os})
        mfm.console.pushVariables({'QtCore': QtCore})
        mfm.console.pushVariables({'QtGui': QtGui})
        mfm.run = mfm.console.execute
        mfm.run(str(mfm.settings.cs_settings['gui']['console']['init']))

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
        self.tabifyDockWidget(self.dockWidgetPlot, self.dockWidgetScriptEdit)
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
        self.actionSaveAllFits.triggered.connect(self.onSaveFits)
        self.actionSaveCurrentFit.triggered.connect(self.onSaveFit)
        self.actionClose_Fit.triggered.connect(mfm.cmd.close_fit)
        self.actionClose_all_fits.triggered.connect(self.onCloseAllFits)
        self.actionLoad_Data.triggered.connect(self.onAddDataset)
        self.actionLoad_result_in_current_fit.triggered.connect(self.onLoadFitResults)

        self.dataset_selector = mfm.widgets.CurveSelector(click_close=False, curve_types='all',
                                                          change_event=self.onCurrentDatasetChanged,
                                                          drag_enabled=True)
        self.verticalLayout_8.addWidget(self.dataset_selector)

    def init_setups(self):
        ##########################################################
        #      Initialize Experiments and Setups                 #
        #      (Commented widgets don't work at the moment       #
        ##########################################################
        # This needs to move to the QtApplication or it needs to be
        # independent as new Widgets can only be created once a QApplication has been created

        structure_setups = [
            mfm.experiments.modelling.PDBLoad()
        ]

        tcspc = mfm.experiments.experiment.Experiment('TCSPC')
        tcspc.add_setups(
            [
                mfm.experiments.tcspc.TCSPCSetupWidget(name="CSV/PQ/IBH", **mfm.settings.cs_settings['tcspc_csv']),
                mfm.experiments.tcspc.TCSPCSetupSDTWidget(),
                mfm.experiments.tcspc.TCSPCSetupDummyWidget()
            ]
        )
        tcspc.add_models(
            models=[
                mfm.models.tcspc.lifetime.LifetimeModelWidget,
                mfm.models.tcspc.fret.FRETrateModelWidget,
                mfm.models.tcspc.fret.GaussianModelWidget,
                mfm.models.tcspc.pddem.PDDEMModelWidget,
                mfm.models.tcspc.fret.WormLikeChainModelWidget
            ]
        )
        mfm.experiment.append(tcspc)

        fcs = mfm.experiments.experiment.Experiment('FCS')
        fcs.add_setups(
            [
                mfm.experiments.fcs.FCSKristine(experiment=mfm.experiments.fcs),
                mfm.experiments.fcs.FCS(experiment=mfm.experiments.fcs)
            ]
        )
        fcs.add_models(
            models=[
                mfm.models.fcs.ParseFCSWidget
            ]
        )
        mfm.experiment.append(fcs)

        global_fit = mfm.experiments.experiment.Experiment('Global')
        global_setup = mfm.experiments.globalfit.GlobalFitSetup(name='Global-Fit', experiment=global_fit)
        #global_fit.add_model(mfm.fitting.model.GlobalFitModelWidget)
        global_fit.add_setup(global_setup)
        mfm.experiment.append(global_fit)

        self.experiment_names = [b.name for b in mfm.experiment if b.name is not 'Global']
        self.comboBox_experimentSelect.addItems(self.experiment_names)

        self.current_fit = None
        mfm.cmd.add_dataset(setup=global_setup)
        #self.onAddDataset(experiment=global_fit, setup=global_setup)  # Add Global-Dataset by default


if __name__ == "__main__":
    app = QApplication(sys.argv)
    import mfm
    import mfm.widgets
    import mfm.tools
    import mfm.ui.resource

    mfm.console = mfm.widgets.QIPythonWidget()

    win = Main(parent=None)
    mfm.console.history_widget = win.plainTextEditHistory
    mfm.cs = win
    win.init_setups()

    with open(mfm.settings.style_sheet_file, 'r') as fp:
        style_sheet = fp.read()
        app.setStyleSheet(style_sheet)

    win.show()
    sys.exit(app.exec_())
