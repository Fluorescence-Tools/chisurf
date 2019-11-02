from __future__ import annotations
import utils
import os
import sys

import webbrowser
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets, uic

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)

import chisurf
import chisurf.decorators
import chisurf.base
import chisurf.macros
import chisurf.tools
import chisurf.experiments.widgets
import chisurf.experiments.tcspc.controller
import chisurf.widgets
import chisurf.models
import chisurf.settings.ui.resource

from chisurf import fitting
from chisurf import experiments


class Main(QtWidgets.QMainWindow):

    @property
    def current_dataset(
            self
    ) -> chisurf.base.Data:
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
    def fit_idx(
            self
    ) -> int:
        subwindow = self.mdiarea.currentSubWindow()
        if subwindow is not None:
            for fit_idx, f in enumerate(chisurf.fits):
                if f == subwindow.fit:
                    return fit_idx

    @property
    def current_experiment_idx(
            self
    ) -> int:
        return self.comboBox_experimentSelect.currentIndex()

    @current_experiment_idx.setter
    def current_experiment_idx(
            self,
            v: int
    ):
        self.comboBox_experimentSelect.setCurrentIndex(v)

    @property
    def current_experiment(
            self
    ) -> experiments.experiment.Experiment:
        return chisurf.experiment[
            self.current_experiment_idx
        ]

    @current_experiment.setter
    def current_experiment(
            self,
            name: str
    ) -> None:
        p = self.comboBox_experimentSelect.currentIndex()
        n = p
        for n, e in enumerate(chisurf.experiment):
            if e.name == name:
                break
        if p != n:
            self.current_experiment_idx = n

    @property
    def current_setup_idx(
            self
    ) -> int:
        return self.comboBox_setupSelect.currentIndex()

    @current_setup_idx.setter
    def current_setup_idx(
            self,
            v: int
    ):
        self.comboBox_setupSelect.setCurrentIndex(v)

    @property
    def current_setup_name(self):
        return str(
            self.comboBox_setupSelect.currentText()
        )

    @property
    def current_setup(
            self
    ) -> experiments.reader.ExperimentReader:
        current_setup = self.current_experiment.readers[
            self.current_setup_idx
        ]
        return current_setup

    @property
    def current_experiment_reader(self):
        if isinstance(
            self.current_setup,
            experiments.reader.ExperimentReader
        ):
            return self.current_setup
        elif isinstance(
                self.current_setup,
                experiments.reader.ExperimentReaderController
        ):
            return self.current_setup.experiment_reader

    @current_setup.setter
    def current_setup(
            self,
            name: str
    ) -> None:
        i = self.current_setup_idx
        j = i
        for j, s in enumerate(
                self.current_experiment.readers
        ):
            if s.name == name:
                break
        if j != i:
            self.current_setup_idx = j

    @property
    def current_model_name(
            self
    ) -> str:
        return str(
            self.comboBox_Model.currentText()
        )

    @property
    def current_fit(
            self
    ) -> fitting.fit.FitGroup:
        return self._current_fit

    @current_fit.setter
    def current_fit(
            self,
            v: fitting.fit.FitGroup
    ) -> None:
        self._current_fit = v

    def closeEvent(
            self,
            event: QtGui.QCloseEvent
    ):
        if chisurf.settings.gui['confirm_close_program']:
            reply = chisurf.widgets.widgets.MyMessageBox.question(
                self,
                'Message',
                "Are you sure to quit?",
                QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def subWindowActivated(self):
        sub_window = self.mdiarea.currentSubWindow()
        if sub_window is not None:
            for f in chisurf.fits:
                if f == sub_window.fit:
                    if self.current_fit is not chisurf.fits[self.fit_idx]:
                        chisurf.run("cs.current_fit = chisurf.fits[%s]" % self.fit_idx)
                        break

            self.current_fit_widget = sub_window.fit_widget
            window_title = chisurf.__name__ + "(" + chisurf.__version__ + "): " + self.current_fit.name

            self.setWindowTitle(window_title)

            chisurf.widgets.hide_items_in_layout(self.modelLayout)
            chisurf.widgets.hide_items_in_layout(self.plotOptionsLayout)
            self.current_fit.model.update()
            self.current_fit.model.show()
            self.current_fit_widget.show()
            sub_window.current_plt_ctrl.show()

    def onRunMacro(self):
        filename = chisurf.widgets.get_filename(
            "Python macros",
            file_type="Python file (*.py)"
        )
        chisurf.run("chisurf.console.run_macro(filename='%s')" % filename)

    def onTileWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        self.mdiarea.tileSubWindows()

    def onTabWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.TabbedView)

    def onCascadeWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        self.mdiarea.cascadeSubWindows()

    def onCurrentDatasetChanged(self):
        self.comboBox_Model.clear()
        ds = self.current_dataset
        if chisurf.imported_datasets:
            try:
                model_names = ds.experiment.get_model_names()
                self.comboBox_Model.addItems(model_names)
            except AttributeError:
                pass

    def onAddFit(self):
        chisurf.run(
            "chisurf.macros.add_fit(model_name='%s', dataset_indices=%s)" %
            (
                self.current_model_name,
                [r.row() for r in self.dataset_selector.selectedIndexes()]
            )
        )

    def onExperimentChanged(self):
        experiment_name = self.comboBox_experimentSelect.currentText()
        chisurf.run("cs.current_experiment = '%s'" % experiment_name)
        # Add setups for selected experiment
        self.comboBox_setupSelect.blockSignals(True)
        self.comboBox_setupSelect.clear()
        self.comboBox_setupSelect.addItems(
            self.current_experiment.reader_names
        )
        self.comboBox_setupSelect.blockSignals(False)
        self.onSetupChanged()

    def onLoadFitResults(
            self,
            **kwargs
    ):
        filename = chisurf.widgets.get_filename(
            file_type="*.json",
            description="Load results into fit-models",
            **kwargs
        )
        chisurf.run(
            "chisurf.macros.load_fit_result(%s, %s)" % (
                self.fit_idx,
                filename
            )
        )

    def onSetupChanged(self):
        chisurf.widgets.hide_items_in_layout(
            self.layout_experiment_reader
        )
        chisurf.run(
            "cs.current_setup = '%s'" % self.current_setup_name
        )
        try:
            self.layout_experiment_reader.addWidget(
                self.current_setup
            )
            self.current_setup.show()
        except TypeError:
            self.layout_experiment_reader.addWidget(
                self.current_setup.controller
            )
            self.current_setup.controller.show()

    def onCloseAllFits(self):
        for sub_window in chisurf.fit_windows:
            sub_window.widget().close_confirm = False
            sub_window.close()

        chisurf.fits.clear()
        chisurf.fit_windows.clear()

    def onAddDataset(self):
        try:
            filename = self.current_setup.controller.filename
        except AttributeError:
            filename = None
        chisurf.run(
            'chisurf.macros.add_dataset(filename="%s")' % filename
        )

    def onSaveFits(
            self,
            path: str = None,
            **kwargs
    ):
        if path is None:
            path = chisurf.widgets.get_directory(**kwargs)
        chisurf.macros.save_fits(path)

    def onSaveFit(
            self,
            directory: str = None,
            **kwargs
    ):
        if directory is None:
            chisurf.working_path = chisurf.widgets.get_directory(**kwargs)
        chisurf.working_path = directory
        chisurf.console.run('chisurf.macros.save_fit()')

    def onOpenHelp(
            self
    ):
        webbrowser.open_new(chisurf.settings.url)

    def init_widgets(self):
        # self.decay_generator = chisurf.tools.decay_generator.TransientDecayGenerator()
        # self.connect(self.actionDye_Diffusion, QtCore.SIGNAL('triggered()'), self.decay_generator.show)

        #self.fret_lines = chisurf.tools.fret_lines.FRETLineGeneratorWidget()
        #self.connect(self.actionFRET_Lines, QtCore.SIGNAL('triggered()'), self.fret_lines.show)

        #self.decay_fret_generator = chisurf.fluorescence.dye_diffusion.TransientFRETDecayGenerator()

        ##########################################################
        #      Fluorescence widgets                              #
        #      (Commented widgets don't work at the moment       #
        ##########################################################
        self.lifetime_calc = chisurf.tools.fret_calculator.tau2r.FRETCalculator()
        self.actionCalculator.triggered.connect(self.lifetime_calc.show)

        self.kappa2_dist = chisurf.tools.kappa2_distribution.kappa2dist.Kappa2Dist()
        self.actionKappa2_Distribution.triggered.connect(self.kappa2_dist.show)

        ##########################################################
        #      TTTR-widgets                                      #
        ##########################################################
        self.tttr_convert = chisurf.tools.tttr.convert.TTTRConvert()
        self.actionConvert.triggered.connect(self.tttr_convert.show)

        self.tttr_correlate = chisurf.tools.tttr.correlate.CorrelateTTTR()
        self.actionCorrelate.triggered.connect(self.tttr_correlate.show)

        self.tttr_histogram = chisurf.tools.tttr.decay_histogram.HistogramTTTR()
        self.actionGenerate_decay.triggered.connect(self.tttr_histogram.show)

        ##########################################################
        #      TTTR-widgets                                      #
        ##########################################################
        self.hdf2pdb = chisurf.tools.modelling.trajectory.MDConverter()
        self.actionTrajectory_converter.triggered.connect(self.hdf2pdb.show)

        self.trajectory_rot_trans = chisurf.tools.modelling.trajectory.RotateTranslateTrajectoryWidget()
        self.actionRotate_Translate_trajectory.triggered.connect(self.trajectory_rot_trans.show)

        self.calculate_potential = chisurf.tools.modelling.potential_enery.PotentialEnergyWidget()
        self.actionCalculate_Potential.triggered.connect(self.calculate_potential.show)

        self.pdb2label = chisurf.tools.fps_json.label_structure.LabelStructure()
        self.actionPDB2Label.triggered.connect(self.pdb2label.show)

        self.structure2transfer = chisurf.tools.traj2fret.gui.Structure2Transfer()
        self.actionStructure2Transfer.triggered.connect(self.structure2transfer.show)

        self.join_trajectories = chisurf.tools.modelling.trajectory.JoinTrajectoriesWidget()
        self.actionJoin_trajectories.triggered.connect(self.join_trajectories.show)

        self.traj_save_topol = chisurf.tools.modelling.trajectory.SaveTopology()
        self.actionSave_topology.triggered.connect(self.traj_save_topol.show)

        self.remove_clashes = chisurf.tools.modelling.remove_clashed_frames.RemoveClashedFrames()
        self.actionRemove_clashes.triggered.connect(self.remove_clashes.show)

        self.align_trajectory = chisurf.tools.modelling.trajectory.AlignTrajectoryWidget()
        self.actionAlign_trajectory.triggered.connect(self.align_trajectory.show)

        #self.update_widget = chisurf.widgets.downloader.UpdateDialog()
        #self.connect(self.actionUpdate, QtCore.SIGNAL('triggered()'), self.update_widget.show)

        self.f_test = chisurf.tools.f_test.f_calculator.FTestWidget()
        self.actionF_Test.triggered.connect(self.f_test.show)

    def init_console(self):
        self.verticalLayout_4.addWidget(chisurf.console)
        chisurf.console.pushVariables({'cs': self})
        chisurf.console.pushVariables({'chisurf': chisurf})
        chisurf.console.pushVariables({'np': np})
        chisurf.console.pushVariables({'os': os})
        chisurf.console.pushVariables({'QtCore': QtCore})
        chisurf.console.pushVariables({'QtGui': QtGui})
        chisurf.run = chisurf.console.execute
        chisurf.run(str(chisurf.settings.gui['console']['init']))

    def init_setups(self):
        ##########################################################
        #       Structure                                        #
        ##########################################################
        structure = experiments.experiment.Experiment('Modelling')
        structure.add_readers(
            [
                (
                    experiments.modelling.LoadStructure(
                        experiment=structure
                    ),
                    None
                )
            ]
        )
        structure.add_model_classes(
            [
                chisurf.models.tcspc.widgets.LifetimeModelWidget
            ]
        )
        chisurf.experiment.append(structure)

        ##########################################################
        #       TCSPC                                            #
        ##########################################################
        tcspc = experiments.experiment.Experiment('TCSPC')
        tcspc.add_readers(
            [
                (
                    experiments.tcspc.TCSPCReader(
                        experiment=tcspc
                    ),
                    experiments.tcspc.controller.TCSPCReaderControlWidget(
                        name='CSV/PQ/IBH',
                        **chisurf.settings.cs_settings['tcspc_csv'],
                    )
                ),
                (
                    experiments.tcspc.bh_sdt.TCSPCSetupSDTWidget(
                        experiment=tcspc
                    ),
                    None
                ),
                (
                    experiments.tcspc.dummy.TCSPCSetupDummy(
                        experiment=tcspc
                    ),
                    experiments.tcspc.controller.TCSPCSetupDummyWidget()
                )
            ]
        )
        tcspc.add_model_classes(
            models=[
                chisurf.models.tcspc.widgets.LifetimeModelWidget,
                chisurf.models.tcspc.widgets.FRETrateModelWidget,
                chisurf.models.tcspc.widgets.GaussianModelWidget,
                chisurf.models.tcspc.widgets.PDDEMModelWidget,
                chisurf.models.tcspc.widgets.WormLikeChainModelWidget
            ]
        )
        chisurf.experiment.append(tcspc)

        ##########################################################
        #       FCS                                              #
        ##########################################################
        fcs = experiments.experiment.Experiment('FCS')
        fcs.add_readers(
            [
                (
                    experiments.fcs.FCS(
                        name='FCS-CSV',
                        experiment=fcs,
                        experiment_reader='csv'
                    ),
                    chisurf.experiments.widgets.FCSController(
                        file_type='All files (*.*)'
                    )
                ),
                (
                    experiments.fcs.FCS(
                        name='Kristine',
                        experiment=fcs,
                        experiment_reader='Kristine'
                    ),
                    chisurf.experiments.widgets.FCSController(
                        file_type='Kristine files (*.cor)'
                    )
                ),
                (
                    experiments.fcs.FCS(
                        name='China-mat',
                        experiment=fcs,
                        experiment_reader='China-mat'
                    ),
                    chisurf.experiments.widgets.FCSController(
                        file_type='Kristine files (*.mat)'
                    )
                ),
                (
                    experiments.fcs.FCS(
                        name='Zeiss Confocor3',
                        experiment=fcs,
                        experiment_reader='confocor3'
                    ),
                    chisurf.experiments.widgets.FCSController(
                        file_type='Confocor3 files (*.fcs)'
                    )
                ),
                (
                    experiments.fcs.FCS(
                        name='ALV',
                        experiment=fcs,
                        experiment_reader='ALV'
                    ),
                    chisurf.experiments.widgets.FCSController(
                        file_type='Kristine files (*.asc)'
                    )
                )
            ]
        )
        fcs.add_model_classes(
            models=[
                chisurf.models.fcs.fcs.ParseFCSWidget
            ]
        )
        chisurf.experiment.append(fcs)

        ##########################################################
        #       Global datasets                                  #
        ##########################################################
        global_fit = experiments.experiment.Experiment('Global')
        global_setup = experiments.globalfit.GlobalFitSetup(
            name='Global-Fit',
            experiment=global_fit
        )
        global_fit.add_model_classes(
            models=[
                chisurf.models.global_model.GlobalFitModelWidget
            ]
        )
        global_fit.add_reader(global_setup)
        chisurf.experiment.append(global_fit)

        chisurf.macros.add_dataset(
            setup=global_setup,
            experiment=global_fit
        )

        ##########################################################
        #       Update UI                                        #
        ##########################################################
        self.experiment_names = [
            b.name for b in chisurf.experiment if b.name is not 'Global'
        ]
        self.comboBox_experimentSelect.addItems(
            self.experiment_names
        )

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        uic.loadUi(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "gui.ui"
            ),
            self
        )

        self.current_fit_widget = None
        self._current_fit = None
        self.experiment_names = list()

        self.setCentralWidget(self.mdiarea)
        self.init_widgets()
        self.configuration = chisurf.tools.text_editor.CodeEditor(
            filename=chisurf.settings.chisurf_settings_file,
            language='YAML',
            can_load=False
        )
        self.actionSettings.triggered.connect(self.configuration.show)

        ##########################################################
        #      Help and About widgets                            #
        ##########################################################
        self.about = uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "about.ui"
            )
        )
        self.about.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.about.hide()
        self.actionAbout.triggered.connect(self.about.show)
        self.actionHelp_2.triggered.connect(self.onOpenHelp)

        ##########################################################
        #      IPython console                                   #
        #      Push variables to console and add it to           #
        #      user interface                                    #
        ##########################################################
        self.dockWidgetScriptEdit.setVisible(chisurf.settings.gui['show_macro_edit'])
        self.dockWidget_console.setVisible(chisurf.settings.gui['show_console'])
        self.init_console()

        ##########################################################
        #      Record and run recorded macros                    #
        ##########################################################
        self.actionRecord.triggered.connect(chisurf.console.start_recording)
        self.actionStop.triggered.connect(chisurf.console.save_macro)
        self.actionRun.triggered.connect(self.onRunMacro)

        ##########################################################
        #      Arrange Docks and window positions                #
        #      Window-controls tile, stack etc.                  #
        ##########################################################
        self.tabifyDockWidget(self.dockWidgetReadData, self.dockWidgetDatasets)
        self.tabifyDockWidget(self.dockWidgetDatasets, self.dockWidgetAnalysis)
        self.tabifyDockWidget(self.dockWidgetAnalysis, self.dockWidgetPlot)
        self.tabifyDockWidget(self.dockWidgetPlot, self.dockWidgetScriptEdit)
        self.tabifyDockWidget(self.dockWidgetDatasets, self.dockWidgetHistory)
        self.editor = chisurf.tools.text_editor.CodeEditor()
        self.verticalLayout_10.addWidget(self.editor)

        self.modelLayout.setAlignment(QtCore.Qt.AlignTop)
        self.plotOptionsLayout.setAlignment(QtCore.Qt.AlignTop)
        self.dockWidgetReadData.raise_()

        self.actionTile_windows.triggered.connect(self.onTileWindows)
        self.actionTab_windows.triggered.connect(self.onTabWindows)
        self.actionCascade.triggered.connect(self.onCascadeWindows)
        self.mdiarea.subWindowActivated.connect(self.subWindowActivated)

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
        self.actionClose_Fit.triggered.connect(chisurf.macros.close_fit)
        self.actionClose_all_fits.triggered.connect(self.onCloseAllFits)
        self.actionLoad_Data.triggered.connect(self.onAddDataset)
        self.actionLoad_result_in_current_fit.triggered.connect(self.onLoadFitResults)

        self.dataset_selector = experiments.widgets.ExperimentalDataSelector(
            click_close=False,
            curve_types='all',
            change_event=self.onCurrentDatasetChanged,
            drag_enabled=True
        )
        self.verticalLayout_8.addWidget(self.dataset_selector)


def gui():
    app = QtWidgets.QApplication(sys.argv)
    chisurf.console = chisurf.widgets.QIPythonWidget()
    win = Main()
    chisurf.console.history_widget = win.plainTextEditHistory
    chisurf.cs = win
    win.init_setups()
    app.setStyleSheet(
      open(
          os.path.join(
              os.path.dirname(
                  __file__
              ),
              './settings/gui_styles/dark.css'
          ),
          mode='r'
      ).read()
    )

    win.show()
    return app


if __name__ == "__main__":
    app = gui()
    sys.exit(app.exec_())
