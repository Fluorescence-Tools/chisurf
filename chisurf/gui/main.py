from __future__ import annotations

import os

import pathlib
import webbrowser

import chisurf.gui
import chisurf.macros.core_fit
from chisurf import typing

import numpy as np
from chisurf.gui import QtWidgets, QtGui, QtCore, uic

import chisurf
import chisurf.decorators
import chisurf.base
import chisurf.fio
import chisurf.experiments
import chisurf.macros

import chisurf.gui.tools
import chisurf.gui.widgets
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.experiments.modelling

import chisurf.models
import chisurf.plugins
import chisurf.fitting
import chisurf.gui.resources


class Main(QtWidgets.QMainWindow):
    """

    Attributes
    ----------
    current_dataset : chisurf.base.Data
        The dataset that is currently selected in the ChiSurf GUI. This
        dataset corresponds to the analysis window selected by the user in
        the UI.
    current_model_class : chisurf.model.Model
        The model used in the analysis (fit) of the currently selected analysis
        windows.
    fit_idx : int
        The index of the currently selected fit in the fit list chisurf.fits
        The current fit index corresponds to the currently selected fit window
        in the list of all fits of the fit.
    current_experiment_idx : int
        The index of the experiment type currently selected in the UI out of
        the list all supported experiments. This corresponds to the index of
        the UI combo box used to select the experiment.
    current_experiment : chisurf.experiments.Experiment
        The experiment currently selected in the GUI.
    current_setup_idx : int
        The index of the setup currently selected in the GUI.
    current_setup_name : str
        The name of the setup currently selected in the GUI.
    current_setup : chisurf.experiments.reader.ExperimentReader
        The current experiment setup / experiment reader selecetd in the GUI
    experiment_names : list
        A list containing the names of the experiments.

    """

    _current_dataset: chisurf.base.Data = None
    experiment_names: typing.List[str] = list()

    @property
    def current_dataset(self) -> chisurf.base.Data:
        return self._current_dataset

    @current_dataset.setter
    def current_dataset(self, dataset_index: int):
        self.dataset_selector.selected_curve_index = dataset_index

    @property
    def current_model_class(self):
        return self._current_model_class

    @property
    def fit_idx(self) -> int:
        return self._fit_idx

    @property
    def current_experiment_idx(self) -> int:
        return self._current_experiment_idx

    @current_experiment_idx.setter
    def current_experiment_idx(self, v: int):
        self.set_current_experiment_idx(v)

    @property
    def current_experiment(self) -> chisurf.experiments.Experiment:
        return chisurf.experiment[self.comboBox_experimentSelect.currentText()]

    @current_experiment.setter
    def current_experiment(self, name: str) -> None:
        combo = self.comboBox_experimentSelect
        # find the row in the combo whose text matches your experiment name
        idx = combo.findText(name)
        if idx == -1:
            raise ValueError(f"Experiment “{name}” not found in comboBox")
        # if it’s different from the current index, update both the combo and your internal idx
        if combo.currentIndex() != idx:
            combo.setCurrentIndex(idx)
            self._current_experiment_idx = idx

    @property
    def current_setup_idx(self) -> int:
        return self._current_setup_idx

    @current_setup_idx.setter
    def current_setup_idx(self, v: int):
        self.set_current_experiment_idx(v)

    @property
    def current_setup_name(self):
        return self.current_setup.name

    @property
    def current_setup(self) -> chisurf.experiments.reader.ExperimentReader:
        current_setup = self.current_experiment.readers[
            self.current_setup_idx
        ]
        return current_setup

    @current_setup.setter
    def current_setup(self, name: str) -> None:
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
    def current_experiment_reader(self):
        if isinstance(
            self.current_setup,
            chisurf.experiments.reader.ExperimentReader
        ):
            return self.current_setup
        elif isinstance(
                self.current_setup,
                chisurf.experiments.reader.ExperimentReaderController
        ):
            return self.current_setup.experiment_reader

    @property
    def current_model_name(self) -> str:
        return self.current_model_class.name

    @property
    def current_fit(self) -> chisurf.fitting.fit.FitGroup:
        return self._current_fit

    @current_fit.setter
    def current_fit(self, v: chisurf.fitting.fit.FitGroup) -> None:
        self._current_fit = v

    def set_current_experiment_idx(self, v):
        self.comboBox_experimentSelect.setCurrentIndex(v)

    def closeEvent(self, event: QtGui.QCloseEvent):
        if chisurf.settings.gui['confirm_close_program']:
            reply = chisurf.gui.widgets.general.MyMessageBox.question(
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
            for fit_idx, f in enumerate(chisurf.fits):
                if f == sub_window.fit:
                    if self.current_fit is not chisurf.fits[fit_idx]:
                        chisurf.run(f"cs.current_fit = chisurf.fits[{fit_idx}]")
                        self._fit_idx = fit_idx
                        break

            self.current_fit_widget = sub_window.fit_widget

            window_title = chisurf.__name__ + "(" + chisurf.__version__ + "): " + self.current_fit.name
            self.setWindowTitle(window_title)

            chisurf.gui.widgets.hide_items_in_layout(self.modelLayout)
            chisurf.gui.widgets.hide_items_in_layout(self.plotOptionsLayout)
            self.current_fit.model.show()
            self.current_fit_widget.show()
            sub_window.current_plot_controller.show()

    def onRunMacro(
            self,
            filename: pathlib.Path = None,
            executor: str = 'console',
            globals=None, locals=None
    ):
        chisurf.logging.info(f"Running macro: {filename}")
        if filename is None:
            filename = chisurf.gui.widgets.get_filename(
                "Python macros",
                file_type="Python file (*.py)"
            )
        if executor == 'console':
            filename_str = filename.as_posix()
            chisurf.run(f"chisurf.console.run_macro(filename='{filename_str}')")
        elif executor == 'exec':
            if globals is None:
                globals = {"__name__": "__main__"}
            globals.update({"__file__": filename})
            with open(filename, 'rb') as file:
                exec(compile(file.read(), filename, 'exec'), globals, locals)

    def onTileWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        self.mdiarea.tileSubWindows()

    def onTabWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.TabbedView)
        self.mdiarea.setTabsClosable(True)
        self.mdiarea.setTabsMovable(True)

    def onCascadeWindows(self):
        self.mdiarea.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        self.mdiarea.cascadeSubWindows()

    def onCurrentDatasetChanged(self):
        self._current_dataset = self.dataset_selector.selected_dataset
        self.comboBox_Model.clear()
        ds = self.current_dataset
        if chisurf.imported_datasets:
            # Get all model names from the experiment
            all_model_names = ds.experiment.get_model_names()

            # Get the list of disabled models from settings
            disabled_models = chisurf.settings.cs_settings.get('plugins', {}).get('disabled_models', [])

            # Filter out disabled models
            model_names = [name for name in all_model_names if name not in disabled_models]

            # Add only enabled models to the combobox
            self.comboBox_Model.addItems(model_names)

    def onCurrentModelChanged(self):
        model_idx = self.comboBox_Model.currentIndex()
        if model_idx >= 0:  # Make sure a valid model is selected
            # Get the selected model name from the combobox
            selected_model_name = self.comboBox_Model.currentText()

            # Find the corresponding model class in the experiment's model classes
            for model_class in self.current_dataset.experiment.model_classes:
                if model_class.name == selected_model_name:
                    self._current_model_class = model_class
                    break

    def onAddFit(self, *args, data_idx: typing.List[int] = None):
        if data_idx is None:
            data_idx = [r.row() for r in self.dataset_selector.selectedIndexes()]
        chisurf.run(f"chisurf.macros.add_fit(model_name='{self.current_model_name}', dataset_indices={data_idx})")

    def onExperimentChanged(self):
        experiment_name = self.comboBox_experimentSelect.currentText()
        chisurf.run(f"cs.current_experiment = '{experiment_name}'")

        # Add setups for selected experiment
        self.comboBox_setupSelect.blockSignals(True)
        self.comboBox_setupSelect.clear()
        self.comboBox_setupSelect.addItems(
            self.current_experiment.reader_names
        )
        self.comboBox_setupSelect.blockSignals(False)
        self._current_experiment_idx = self.comboBox_experimentSelect.currentIndex()
        self.onSetupChanged()

    def onLoadFitResults(self, **kwargs):
        filename = chisurf.gui.widgets.get_filename(
            file_type="*.json",
            description="Load results into fit-models",
            **kwargs
        )
        chisurf.run(f"chisurf.macros.load_fit_result({self.fit_idx}, {filename})")

    def set_current_setup_idx(self, v: int):
        self.comboBox_setupSelect.setCurrentIndex(v)
        self._current_setup_idx = v

    def onSetupChanged(self):
        chisurf.gui.widgets.hide_items_in_layout(
            self.layout_experiment_reader
        )
        chisurf.run(f"cs.current_setup = '{self.current_setup_name}'")
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
        self._current_setup_idx = self.comboBox_setupSelect.currentIndex()

    def onCloseAllFits(self):
        for sub_window in chisurf.gui.fit_windows:
            sub_window.widget().close_confirm = False
            sub_window.close()

        chisurf.fits.clear()
        chisurf.gui.fit_windows.clear()

    def onAddDataset(self):
        filename = self.current_setup.controller.get_filename()
        if isinstance(filename, list):
            l = [r"{}".format(pathlib.Path(f).as_posix()) for f in filename]
            s = '|'.join(l)
        elif isinstance(filename, pathlib.Path):
            s = r"{}".format(filename.as_posix())
        else:
            s = r"{}".format(filename)
        s = s.replace("\\", "/")
        chisurf.run(f'chisurf.macros.add_dataset(filename=r"{s}")')

    def onSaveFits(self, event: QtCore.QEvent = None):
        path, _ = chisurf.gui.widgets.get_directory()
        chisurf.working_path = path
        chisurf.run(f'chisurf.macros.save_fits(target_path=r"{path.as_posix()}")')

    def onSaveFit(self, event: QtCore.QEvent = None, **kwargs):
        path, _ = chisurf.gui.widgets.get_directory(**kwargs)
        chisurf.working_path = path
        chisurf.run(f'chisurf.macros.save_fit(target_path=r"{path.as_posix()}")')

    def onOpenHelp(self):
        webbrowser.open_new(chisurf.info.help_url)

    def onOpenUpdate(self):
        webbrowser.open_new(chisurf.info.help_url)


    def init_console(self):
        self.verticalLayout_4.addWidget(chisurf.console)
        chisurf.console.pushVariables({'cs': self})
        chisurf.console.pushVariables({'chisurf': chisurf})
        chisurf.console.pushVariables({'np': np})
        chisurf.console.pushVariables({'os': os})
        chisurf.console.pushVariables({'QtCore': QtCore})
        chisurf.console.pushVariables({'QtGui': QtGui})
        chisurf.console.set_default_style('linux')
        chisurf.run = chisurf.console.execute
        chisurf.run(str(chisurf.settings.gui['console_init']))

    def init_setups(self):

        ##########################################################
        #       TCSPC                                            #
        ##########################################################
        # tcspc = chisurf.experiments.Experiment('TCSPC')
        # fcs = chisurf.experiments.experiment.Experiment('FCS')
        # structure = chisurf.experiments.Experiment('Modelling')
        # locals().update(chisurf.experiments.experiments)

        # Get disabled experiments list (for reference only, we'll load all experiments)
        disabled_experiments = chisurf.settings.cs_settings.get('plugins', {}).get('disabled_experiments', [])

        tcspc = chisurf.experiments.types['tcspc']
        # Always initialize the experiment, even if it's disabled
        tcspc.add_readers(
                [
                    (
                        chisurf.experiments.tcspc.TCSPCReader(
                            name="TXT/CSV",
                            experiment=tcspc
                        ),
                        chisurf.gui.widgets.experiments.tcspc.controller.TCSPCReaderControlWidget(
                            name='CSV/PQ/IBH',
                            **chisurf.settings.cs_settings['tcspc_csv'],
                        )
                    ),
                (
                    chisurf.experiments.tcspc.TCSPCTTTRReader(
                        name="TTTR-file",
                        experiment=tcspc
                    ),
                    chisurf.gui.widgets.experiments.tcspc.controller.TCSPCTTTRReaderControlWidget(
                        **chisurf.settings.cs_settings['tcspc_csv'],
                    )
                ),
                (
                    chisurf.gui.widgets.experiments.tcspc.bh_sdt.TCSPCSetupSDTWidget(
                        name="Becker-SDT",
                        experiment=tcspc
                    ),
                    None
                ),
                (
                    chisurf.experiments.tcspc.TCSPCSimulatorSetup(
                        name="Simulator",
                        experiment=tcspc
                    ),
                    chisurf.gui.widgets.experiments.tcspc.controller.TCSPCSimulatorSetupWidget()
                )
            ]
        )
        tcspc.add_model_classes(
            models=[
                chisurf.models.tcspc.widgets.LifetimeModelWidget,
                chisurf.models.tcspc.widgets.FRETrateModelWidget,
                chisurf.models.tcspc.widgets.GaussianModelWidget,
                chisurf.models.tcspc.widgets.PDDEMModelWidget,
                chisurf.models.tcspc.widgets.WormLikeChainModelWidget,
                chisurf.models.tcspc.widgets.ParseDecayModelWidget,
                chisurf.models.tcspc.widgets.LifetimeMixtureModelWidget
            ]
        )
        chisurf.experiment[tcspc.name] = tcspc

        ##########################################################
        #       Stopped flow                                     #
        ##########################################################
        stopped_flow = chisurf.experiments.types['stopped_flow']
        # Always initialize the experiment, even if it's disabled
        stopped_flow.add_readers(
            [
                (
                    chisurf.experiments.tcspc.TCSPCReader(
                        name="TXT/CSV",
                        experiment=stopped_flow
                    ),
                    chisurf.gui.widgets.experiments.tcspc.controller.TCSPCReaderControlWidget(
                        name='CSV/PQ/IBH',
                        **chisurf.settings.cs_settings['tcspc_csv'],
                    )
                ),
                (
                    chisurf.experiments.tcspc.TCSPCTTTRReader(
                        name="TTTR-file",
                        experiment=stopped_flow
                    ),
                    chisurf.gui.widgets.experiments.tcspc.controller.TCSPCTTTRReaderControlWidget(
                        **chisurf.settings.cs_settings['tcspc_csv'],
                    )
                )
            ]
        )
        stopped_flow.add_model_classes(
            models=[
                chisurf.models.stopped_flow.ParseStoppedFlowWidget,
                chisurf.models.stopped_flow.ReactionWidget
            ]
        )
        chisurf.experiment[stopped_flow.name] = stopped_flow

        ##########################################################
        #       Photon distribution analysis                     #
        ##########################################################
        pda = chisurf.experiments.types['pda']
        # Always initialize the experiment, even if it's disabled
        pda.add_readers(
            [
                (
                    chisurf.experiments.pda.PdaReader(
                        name="PTU/HT3/SPC",
                        micro_time_ranges=[(0, 16000), (0, 16000)],
                        channels=([0, 3], [1, 2]),
                        experiment=pda
                    ),
                    chisurf.gui.widgets.experiments.pda.controller.PdaTTTRWidget(
                        name="PTU/HT3/SPC"
                    )
                )
            ]
        )
        pda.add_model_classes(
            models=[
                chisurf.models.pda.widgets.PdaSimpleModelWidget
            ]
        )
        chisurf.experiment[pda.name] = pda

        ##########################################################
        #       FCS                                              #
        ##########################################################
        fcs = chisurf.experiments.types['fcs']
        # Always initialize the experiment, even if it's disabled
        fcs.add_readers(
            [
                (
                    chisurf.experiments.fcs.FCS(
                        name='Seidel Kristine',
                        experiment=fcs,
                        experiment_reader='kristine'
                    ),
                    chisurf.gui.widgets.experiments.widgets.FCSController(
                        file_type='Kristine files (*.cor)'
                    )
                ),
                (
                    chisurf.experiments.fcs.FCS(
                        name='FCS-CSV',
                        experiment=fcs,
                        experiment_reader='csv'
                    ),
                    chisurf.gui.widgets.experiments.widgets.FCSController(
                        file_type='All files (*.*)'
                    )
                ),
                (
                    chisurf.experiments.fcs.FCS(
                        name='Zeiss Confocor3',
                        experiment=fcs,
                        experiment_reader='confocor3'
                    ),
                    chisurf.gui.widgets.experiments.widgets.FCSController(
                        file_type='Confocor3 files (*.fcs)'
                    )
                ),
                (
                    chisurf.experiments.fcs.FCS(
                        name='China FCS',
                        experiment=fcs,
                        experiment_reader='china-mat'
                    ),
                    chisurf.gui.widgets.experiments.widgets.FCSController(
                        file_type='Kristine files (*.mat)'
                    )
                ),
                (
                    chisurf.experiments.fcs.FCS(
                        name='PyCorrFit',
                        experiment=fcs,
                        experiment_reader='pycorrfit'
                    ),
                    chisurf.gui.widgets.experiments.widgets.FCSController(
                        file_type='PyCorrFit csv files (*.csv)'
                    )
                ),
                (
                    chisurf.experiments.fcs.FCS(
                        name='ALV-Correlator',
                        experiment=fcs,
                        experiment_reader='alv'
                    ),
                    chisurf.gui.widgets.experiments.widgets.FCSController(
                        file_type='ALV-Correlator files (*.asc)'
                    )
                )
            ]
        )
        fcs.add_model_classes(
            models=[
                chisurf.models.fcs.fcs.ParseFCSWidget
            ]
        )
        chisurf.experiment[fcs.name] = fcs

        ##########################################################
        #       Structure                                        #
        ##########################################################
        structure = chisurf.experiments.types['structure']
        # Always initialize the experiment, even if it's disabled
        structure.add_readers(
            [
                (
                    chisurf.experiments.modelling.StructureReader(
                        experiment=structure
                    ),
                    chisurf.gui.widgets.experiments.modelling.StructureReaderController()
                )
            ]
        )

        structure.add_model_classes(
            [
                chisurf.models.tcspc.widgets.LifetimeModelWidget
            ]
        )
        chisurf.experiment[structure.name] = structure

        ##########################################################
        #       Global datasets                                  #
        ##########################################################
        global_fit = chisurf.experiments.experiment.Experiment(
            name='Global',
            hidden=True
        )
        # Always initialize the experiment, even if it's disabled
        global_setup = chisurf.experiments.globalfit.GlobalFitSetup(
            name='Global-Fit',
            experiment=global_fit
        )
        global_fit.add_model_classes(
            models=[
                chisurf.models.global_model.GlobalFitModelWidget,
                chisurf.models.global_model.ParameterTransformWidget
            ]
        )
        global_fit.add_reader(global_setup)
        chisurf.experiment[global_fit.name] = global_fit

        chisurf.macros.add_dataset(global_setup, name="Global Dataset")

        ##########################################################
        #       Update UI                                        #
        ##########################################################
        # Get the list of disabled experiments from settings
        disabled_experiments = chisurf.settings.cs_settings.get('plugins', {}).get('disabled_experiments', [])

        # Filter out hidden and disabled experiments
        self.experiment_names = [
            b.name for b in list(chisurf.experiment.values()) 
            if not b.hidden and b.name not in disabled_experiments
        ]
        self.comboBox_experimentSelect.addItems(
            self.experiment_names
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(pathlib.Path(__file__).parent / "gui.ui", self)

        self.current_fit_widget = None
        self._current_fit = None
        self._current_model_class = None
        self._current_experiment_idx = 0
        self._fit_idx = 0
        self._current_setup_idx = 0

        self.experiment_names = list()
        self.dataset_selector = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
            click_close=False,
            curve_types='all',
            change_event=self.onCurrentDatasetChanged,
            drag_enabled=True,
            experiment=None
        )

        # widget listing the existing fits
        self.fit_selector = chisurf.gui.widgets.fitting.ModelDataRepresentationSelector(parent=self)

        self.about = uic.loadUi(pathlib.Path(__file__).parent / "about.ui")

        # Setup status bar with progress bar and message
        self.status = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status)

        # Create a QWidget to hold the progress bar and message
        status_widget = QtWidgets.QWidget()
        status_layout = QtWidgets.QHBoxLayout(status_widget)

        # Set spacing and margins to zero
        status_layout.setSpacing(0)  # Set spacing between widgets to zero
        status_layout.setContentsMargins(0, 0, 0, 0)  # Set margins to zero

        # Create a progress bar
        self.progress_bar = QtWidgets.QProgressBar(self.status)
        self.progress_bar.setFixedWidth(150)  # Set a fixed width for the progress bar
        self.progress_bar.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_bar.setFixedHeight(15)  # Adjust the height as needed

        # Create a label for the status message
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setMaximumHeight(20)  # Set a maximum height for the status message

        # Add the progress bar and status message to the status layout
        status_layout.addWidget(self.status_label)
        status_layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))
        status_layout.addWidget(self.progress_bar)

        # Add the status widget to the status bar, aligning to the left
        self.status.addWidget(status_widget, 1)  # 1 gives the widget some stretch

    def update(self):
        super().update()
        self.fit_selector.update()
        self.dataset_selector.update()

    def arrange_widgets(self):
        # self.setCentralWidget(self.mdiarea)

        ##########################################################
        #      Help and About widgets                            #
        ##########################################################
        self.about.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.about.hide()

        ##########################################################
        #      IPython console                                   #
        #      Push variables to console and add it to           #
        #      user interface                                    #
        ##########################################################
        self.dockWidgetScriptEdit.setVisible(chisurf.settings.gui['show_macro_edit'])
        self.dockWidget_console.setVisible(chisurf.settings.gui['show_console'])
        self.init_console()

        ##########################################################
        #      Arrange Docks and window positions                #
        #      Window-controls tile, stack etc.                  #
        ##########################################################
        self.tabifyDockWidget(self.dockWidgetReadData, self.dockWidgetDatasets)
        self.tabifyDockWidget(self.dockWidgetDatasets, self.dockWidgetAnalysis)
        self.tabifyDockWidget(self.dockWidgetAnalysis, self.dockWidgetPlot)
        self.tabifyDockWidget(self.dockWidgetPlot, self.dockWidgetScriptEdit)
        self.tabifyDockWidget(self.dockWidgetDatasets, self.dockWidgetHistory)
        self.editor = chisurf.gui.tools.code_editor.CodeEditor()

        self.verticalLayout_10.addWidget(self.editor)

        # Add data selector widget
        self.verticalLayout_8.addWidget(self.dataset_selector)

        # Add fit selector widget
        self.verticalLayout_5.addWidget(self.fit_selector)

        self.modelLayout.setAlignment(QtCore.Qt.AlignTop)
        self.plotOptionsLayout.setAlignment(QtCore.Qt.AlignTop)
        self.dockWidgetReadData.raise_()

    def define_actions(self):
        ##########################################################
        # GUI ACTIONS
        ##########################################################
        self.actionTile_windows.triggered.connect(self.onTileWindows)
        self.actionTab_windows.triggered.connect(self.onTabWindows)
        self.actionCascade.triggered.connect(self.onCascadeWindows)
        self.mdiarea.subWindowActivated.connect(self.subWindowActivated)
        self.actionAbout.triggered.connect(self.about.show)
        self.actionHelp_2.triggered.connect(self.onOpenHelp)
        self.actionUpdate.triggered.connect(self.onOpenUpdate)

        ##########################################################
        #      Record and run recorded macros                    #
        ##########################################################
        self.actionRecord.triggered.connect(chisurf.console.start_recording)
        self.actionStop.triggered.connect(chisurf.console.save_macro)
        self.actionRun.triggered.connect(
            lambda: self.onRunMacro(filename=None, executor='console')
        )

        ##########################################################
        #    Connect changes in User-interface to actions like:  #
        #    Loading dataset, changing setups, models, etc.      #
        ##########################################################
        self.actionSetupChanged.triggered.connect(self.onSetupChanged)
        self.actionExperimentChanged.triggered.connect(self.onExperimentChanged)
        self.actionChange_current_dataset.triggered.connect(self.onCurrentDatasetChanged)
        self.comboBox_Model.currentIndexChanged.connect(self.onCurrentModelChanged)
        self.comboBox_experimentSelect.currentIndexChanged.connect(self.onExperimentChanged)
        self.comboBox_setupSelect.currentIndexChanged.connect(self.onSetupChanged)
        self.actionAdd_fit.triggered.connect(self.onAddFit)
        self.actionSaveAllFits.triggered.connect(self.onSaveFits)
        self.actionSaveCurrentFit.triggered.connect(self.onSaveFit)
        self.actionClose_Fit.triggered.connect(chisurf.macros.core_fit.close_fit)
        self.actionClose_all_fits.triggered.connect(self.onCloseAllFits)
        self.actionLoad_Data.triggered.connect(self.onAddDataset)
        self.actionLoad_result_in_current_fit.triggered.connect(self.onLoadFitResults)

    def load_tools(self):
        import chisurf
        import chisurf.gui
        import chisurf.gui.tools

        ##########################################################
        #      Fluorescence widgets                              #
        #      (Commented widgets don't work at the moment       #
        ##########################################################
        self.lifetime_calc = chisurf.gui.tools.fret.calculator.tau2r.FRETCalculator()
        self.actionCalculator.triggered.connect(self.lifetime_calc.show)

        self.kappa2_dist = chisurf.gui.tools.kappa2_distribution.k2dgui.Kappa2Dist()
        self.actionKappa2_Distribution.triggered.connect(self.kappa2_dist.show)

        self.f_test = chisurf.gui.tools.f_test.FTestWidget()
        self.actionF_Test.triggered.connect(self.f_test.show)

        ##########################################################
        #      Settings                                          #
        ##########################################################
        # Configuration editor
        self.configuration = chisurf.gui.tools.code_editor.CodeEditor(
            filename=chisurf.settings.chisurf_settings_file,
            language='YAML',
            can_load=False
        )
        self.actionSettings.triggered.connect(self.configuration.show)
        # Clear local settings, i.e., the settings file in the user folder
        self.actionClear_local_settings.triggered.connect(chisurf.settings.clear_settings_folder)

        ##########################################################
        #      Initialize                                        #
        ##########################################################
        self.onExperimentChanged()
