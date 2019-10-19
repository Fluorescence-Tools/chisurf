from __future__ import annotations

import sys
import json
import os
import tempfile

import numpy as np
from qtpy import QtWidgets, uic
from guiqwt.builder import make
from guiqwt.plot import CurveDialog
import qdarkstyle

import chisurf.mfm as mfm
import chisurf.fio.coordinates
import chisurf.fluorescence.fps.widgets
from chisurf.fluorescence.simulation.dye_diffusion import DyeDecay
from chisurf.plots.molview.MolView import MolQtWidget
from chisurf.structure.structure import Structure
from chisurf.widgets.pdb import PDBSelector


class TransientDecayGenerator(DyeDecay, QtWidgets.QWidget):

    name = "Decay Generator"

    @property
    def n_curves(self) -> int:
        return int(self.spinBox_4.value())

    @n_curves.setter
    def n_curves(
            self,
            v: int
    ):
        self.spinBox_4.setValue(v)

    @property
    def nTAC(self) -> int:
        return int(self.spinBox_2.value())

    @nTAC.setter
    def nTAC(
            self,
            v: int
    ):
        return self.spinBox_2.setValue(v)

    @property
    def dtTAC(self) -> float:
        return float(self.doubleSpinBox_18.value())

    @dtTAC.setter
    def dtTAC(
            self,
            v: float
    ):
        return self.doubleSpinBox_18.setValue(v)

    @property
    def decay_mode(self) -> str:
        if self.radioButton_3.isChecked():
            return 'photon'
        if self.radioButton_4.isChecked():
            return 'curve'

    @decay_mode.setter
    def decay_mode(
            self,
            v: str
    ):
        if v == 'photon':
            self.radioButton_4.setChecked(True)
            self.radioButton_3.setChecked(False)
        elif v == 'curve':
            self.radioButton_3.setChecked(True)
            self.radioButton_4.setChecked(False)

    @property
    def settings_file(self) -> str:
        return self._settings_file

    @settings_file.setter
    def settings_file(
            self,
            v: str
    ):
        return self.lineEdit_2.setText(v)

    @property
    def all_quencher_atoms(self) -> bool:
        return not bool(self.groupBox_5.isChecked())

    @all_quencher_atoms.setter
    def all_quencher_atoms(
            self,
            v: bool
    ):
        self.groupBox_5.setChecked(not v)

    @property
    def filename_prefix(self) -> str:
        return str(self.lineEdit_5.text())

    @property
    def skip_frame(self) -> int:
        return int(self.spinBox_3.value())

    @property
    def n_frames(self) -> int:
        return int(self.spinBox.value())

    @property
    def nBins(self) -> int:
        return int(self.spinBox_2.value())

    @property
    def n_photons(self) -> int:
        return int(self.doubleSpinBox_11.value() * 1e6)

    @n_photons.setter
    def n_photons(
            self,
            v: int
    ):
        return self.doubleSpinBox_11.setValue(v / 1e6)

    @property
    def critical_distance(self) -> float:
        return self.dye_parameter.critical_distance

    @critical_distance.setter
    def critical_distance(
            self,
            v: float
    ):
        self.dye_parameter.critical_distance = v

    @property
    def t_max(self) -> float:
        """
        simulation time in nano-seconds
        """
        return float(self.doubleSpinBox_6.value()) * 1000.0

    @t_max.setter
    def t_max(
            self,
            v: float
    ):
        self.onSimulationTimeChanged()
        self.doubleSpinBox_6.setValue(float(v / 1000.0))

    @property
    def t_step(self) -> float:
        """
        time-step in picoseconds
        """
        return float(self.doubleSpinBox_7.value()) / 1000.0

    @t_step.setter
    def t_step(
            self,
            v: float
    ):
        self.onSimulationTimeChanged()
        self.doubleSpinBox_7.setValue(float(v * 1000.0))

    @property
    def attachment_chain(self) -> str:
        return self.pdb_selector.chain_id

    @attachment_chain.setter
    def attachment_chain(
            self,
            v: str
    ):
        self.pdb_selector.chain_id = v

    @property
    def attachment_residue(self) -> int:
        return self.pdb_selector.residue_id

    @attachment_residue.setter
    def attachment_residue(
            self,
            v: int
    ):
        self.pdb_selector.residue_id = v

    @property
    def attachment_atom_name(self) -> str:
        return self.pdb_selector.atom_name

    @attachment_atom_name.setter
    def attachment_atom_name(
            self,
            v: str
    ):
        self.pdb_selector.atom_name = v

    def __init__(
            self,
            *args,
            **kwargs
    ):
        fn = os.path.join(mfm.package_directory, './settings/sample.json')
        dye_diffusion_settings_file = kwargs.get('dye_diffusion_settings_file', fn)
        self.verbose = kwargs.get('verbose', mfm.verbose)
        settings = json.load(
            chisurf.fio.zipped.open_maybe_zipped(
                filename=dye_diffusion_settings_file,
                mode='r'
            )
        )

        super().__init__(*args, **settings)

        QtWidgets.QWidget.__init__(self)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "dye_diffusion2.ui"
            ),
            self
        )

        self.pdb_selector = PDBSelector()
        self.verticalLayout_10.addWidget(self.pdb_selector)
        self._settings_file = None
        self.settings_file = dye_diffusion_settings_file

        self.protein_quenching = chisurf.fluorescence.fps.widgets.ProteinQuenchingWidget(
            k_quench_protein=kwargs.get('k_quench_protein', 5.0),
        )
        self.verticalLayout_11.addWidget(self.protein_quenching)
        self.dye_parameter = chisurf.fluorescence.fps.dynamic.DyeParameterWidget(**kwargs)
        self.verticalLayout_14.addWidget(self.dye_parameter)

        self.sticking = chisurf.fluorescence.fps.dynamic.StickingParameterWidget()
        self.verticalLayout_13.addWidget(self.sticking)

        # # User-interface
        self.actionLoad_PDB.triggered.connect(self.onLoadPDB)
        self.actionLoad_settings.triggered.connect(self.onLoadSettings)
        self.actionSave_AV.triggered.connect(self.onSaveAV)
        self.actionLoad_settings.triggered.connect(self.onLoadSettings)
        self.actionUpdate_all.triggered.connect(self.update_model)
        self.actionSave_histogram.triggered.connect(self.onSaveHist)
        self.actionSave_trajectory.triggered.connect(self.onSaveTrajectory)

        self.doubleSpinBox_6.valueChanged.connect(self.onSimulationTimeChanged)
        self.doubleSpinBox_7.valueChanged.connect(self.onSimulationDtChanged)

        self.tmp_dir = tempfile.gettempdir()
        print("Temporary Directory: %s" % self.tmp_dir)

        ## Decay-Curve
        fd = CurveDialog(edit=False, toolbar=True)
        self.plot_decay = fd.get_plot()
        self.hist_curve = make.curve([1], [1], color="r", linewidth=1)
        self.unquenched_curve = make.curve([1], [1], color="b", linewidth=1)
        self.plot_decay.add_item(self.hist_curve)
        self.plot_decay.add_item(self.unquenched_curve)
        self.plot_decay.set_scales('lin', 'log')
        self.verticalLayout_2.addWidget(fd)

        ## Diffusion-Trajectory-Curve
        options = dict(title="Trajectory", xlabel="time [ns]", ylabel=("|R-<R>|"))
        fd = CurveDialog(edit=False, toolbar=True, options=options)
        self.plot_diffusion = fd.get_plot()
        self.diffusion_curve = make.curve([1], [1], color="b", linewidth=1)
        self.plot_diffusion.add_item(self.diffusion_curve)
        self.plot_diffusion.set_scales('lin', 'lin')
        self.verticalLayout_6.addWidget(fd)

        options = dict(xlabel="corr. time [ns]", ylabel=("A.Corr.(|R-<R>|)"))
        fd = CurveDialog(edit=False, toolbar=True, options=options)
        self.plot_autocorr = fd.get_plot()
        self.diffusion_autocorrelation = make.curve([1], [1], color="r", linewidth=1)
        self.plot_autocorr.add_item(self.diffusion_autocorrelation)
        self.plot_autocorr.set_scales('log', 'lin')
        self.verticalLayout_6.addWidget(fd)

        ## Protein Structure
        self.molview = MolQtWidget(self, enableUi=False)
        self.verticalLayout_4.addWidget(self.molview)

        self.diff_file = None
        self.av_slow_file = None
        self.av_fast_file = None

        self.hide()

    def onSaveHist(self, verbose=False):
        verbose = self.verbose or verbose
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', '.')[0]
        self.save_histogram(filename=filename, verbose=verbose)

    def molview_highlight_quencher(self):
        quencher = self.quencher
        pymol = self.molview.pymol
        for res_name in quencher:
            pymol.cmd.do("hide lines, resn %s" % res_name)
            pymol.cmd.do("show sticks, resn %s" % res_name)
            pymol.cmd.do("color red, resn %s" % res_name)

    def onLoadSettings(self):
        print("onLoadSettings")
        print("Setting File: %s" % self.settings_file)

    def update_3D(self):
        self.molview.reset()
        if isinstance(self.structure, Structure):
            self.molview.openFile(self.structure.filename, frame=1, mode='cartoon', object_name='protein')
        self.molview.pymol.cmd.do("hide all")
        self.molview.pymol.cmd.do("show surface, protein")
        self.molview.pymol.cmd.do("color gray, protein")

        if self.diff_file is not None:
            self.molview.openFile(self.diff_file, frame=1, object_name='trajectory')
            self.molview.pymol.cmd.do("color green, trajectory")

        self.molview.pymol.cmd.orient()
        self.molview_highlight_quencher()

    def update_trajectory_curve(self):
        y = self.diffusion.distance_to_mean
        x = np.linspace(0, self.t_max, y.shape[0])
        self.diffusion_curve.set_data(x, y)
        self.diffusion_autocorrelation.set_data(x, chisurf.math.signal.autocorr(y))
        self.plot_autocorr.do_autoscale()
        self.plot_diffusion.do_autoscale()

    def update_decay_histogram(self):
        x, y = self.get_histogram(nbins=self.nBins)
        y[y < 0.001] = 0.001
        self.hist_curve.set_data(x, y + 1.0)
        yu = np.exp(-x / self.tau0) * y[1]
        self.unquenched_curve.set_data(x, yu + 1.0)
        self.plot_decay.do_autoscale()

    def update_model(self, verbose=False):
        DyeDecay.update_model(self)
        self.doubleSpinBox_16.setValue(self.quantum_yield)
        self.doubleSpinBox_15.setValue(self.collisions * 100.0)
        diff_file, av_slow_file, av_fast_file = self.onSaveAV(directory=self.tmp_dir)
        self.diff_file = diff_file
        self.av_slow_file = av_slow_file
        self.av_fast_file = av_fast_file
        self.update_3D()
        self.update_decay_histogram()
        self.update_trajectory_curve()

    def onSaveAV(
            self,
            directory: str = None,
            verbose: bool = True
    ):
        """
        Saves the accessible volumes to a directory. If no directory is provided a dialog-window
        is opned in which the user chooses the target-directory.
        """
        verbose = self.verbose or verbose
        if verbose:
            print("\nWriting AVs to directory")
            print("-------------------------")
        if directory is None:
            directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose directory', '.'))
        if verbose:
            print("Directory: %s" % directory)
            print("Filename-Prefix: %s" % str(self.filename_prefix))
        diff_file = os.path.join(directory, self.filename_prefix + self.diffusion.trajectory_suffix)
        if verbose:
            print("Saving trajectory...")
            print("Trajectory filename: %s" % diff_file)
            print("Saving every %i frame." % self.skip_frame)
        self.diffusion.save_trajectory(diff_file, self.skip_frame)

        av_slow_file = os.path.join(directory, self.filename_prefix + '_av_slow.xyz')
        if verbose:
            print("\nSaving slow AV...")
            print("Trajectory filename: %s" % av_slow_file)
        chisurf.fio.coordinates.write_xyz(av_slow_file, self.av.points_slow)

        av_fast_file = os.path.join(directory, self.filename_prefix + '_av_fast.xyz')
        if verbose:
            print("\nSaving slow AV...")
            print("Trajectory filename: %s" % av_fast_file)
        chisurf.fio.coordinates.write_xyz(av_fast_file, self.av.points_fast)
        return diff_file, av_slow_file, av_fast_file

    def onSimulationDtChanged(self):
        time_steps = self.t_max / self.t_step
        self.spinBox.setValue(int(time_steps))

    def onSimulationTimeChanged(self):
        time_steps = self.t_max / self.t_step
        self.spinBox.setValue(int(time_steps))

    def onLoadPDB(self):
        #pdb_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open PDB-File', '', 'PDB-files (*.pdb)'))
        filename = chisurf.widgets.get_filename('Open PDB-File', 'PDB-files (*.pdb)')
        self.lineEdit.setText(filename)
        self.structure = filename
        self.pdb_selector.atoms = self.structure.atoms
        self.tmp_dir = tempfile.gettempdir()
        self.update_3D()

    def onSaveTrajectory(self):
        filename = str(QtWidgets.QFileDialog.getSaveFileName(None, 'Open PDB-File', '', 'H5-FRET-files (*.fret.h5)'))[0]
        self.diffusion.save_simulation_results(filename)
        self.save_photons(filename)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = TransientDecayGenerator()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
