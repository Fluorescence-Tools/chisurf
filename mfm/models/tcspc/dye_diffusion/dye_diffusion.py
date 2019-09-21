from __future__ import annotations
from typing import Tuple

import gc
import json
import os
import tempfile

import numpy as np
from qtpy import QtWidgets, uic

import mfm.io.xyz
from mfm.fluorescence.fps.dynamic import DiffusionSimulation, Dye, Sticking, \
    ProteinQuenching
from mfm.fluorescence.fps.widgets import ProteinQuenchingWidget, DyeWidget, StickingWidget

import mfm
import mfm.fitting.fit
import mfm.models.tcspc.nusiance
import mfm.fitting.widgets
import mfm.fluorescence.fps as fps
import mfm.fluorescence.tcspc.convolve
import mfm.io
import mfm.math
import mfm.models.tcspc.widgets
import mfm.structure
import mfm.structure.structure
from mfm import plots
from mfm.curve import Curve
from mfm.models.model import Model
from mfm.fluorescence.fps import ACV
from mfm.fluorescence.simulation import photon
from mfm.structure.structure import Structure, get_coordinates_of_residues
from mfm.widgets.pdb import PDBSelector


class DyeDecay(Model, Curve):

    @property
    def sticky_mode(self):
        return self.sticking_parameter.sticky_mode

    @sticky_mode.setter
    def sticky_mode(self, v):
        self.sticking_parameter.sticky_mode = v

    @property
    def slow_fact(self):
        return self.sticking_parameter.slow_fact

    @slow_fact.setter
    def slow_fact(self, v):
        self.sticking_parameter.slow_fact = v

    @property
    def slow_radius(self):
        return self.sticking_parameter.slow_radius

    @slow_radius.setter
    def slow_radius(self, v):
        self.sticking_parameter.slow_radius = v

    @property
    def av_parameter(self):
        return self.dye_parameter.av_parameter

    @av_parameter.setter
    def av_parameter(self, d):
        self.dye_parameter.av_parameter = d

    @property
    def diffusion_coefficient(self):
        return self.dye_parameter.diffusion_coefficient

    @diffusion_coefficient.setter
    def diffusion_coefficient(self, v):
        self.dye_parameter.diffusion_coefficient = v

    @property
    def tau0(self):
        return self.dye_parameter.tau0

    @tau0.setter
    def tau0(self, v):
        self.dye_parameter.tau0 = v

    @property
    def n_curves(self):
        """
        The number of decay-curves used to calculate a smooth decay (number of samples of the simulation)
        """
        return self._n_curves

    @n_curves.setter
    def n_curves(self, v):
        self._n_curves = int(v)

    @property
    def x_values(self):
        """
        The x-values of the time-resolved fluorescence histogram
        """
        return np.arange(self.n_points) * self.convolve.dt

    @property
    def quantum_yield(self):
        """
        Relative quantum yield of the compared to the unquenched dye
        """
        x = self.x_values
        yq = sum(self.y_values)
        y = sum(np.exp(-x / self.tau0)) * max(self.y_values)
        if self.verbose:
            print("Quantum yield: %.4f" % float(yq / y))
        return float(yq / y)

    @property
    def diffusion(self):
        """
        Diffusion simulation of type :py:class:`.SimulateDiffusion`. If no trajectory is simulated yet it will
        be simulated after calling the attribute.
        """
        if self._diffusion is None:
            self._diffusion = self.simulate_diffusion_trajectory()
        return self._diffusion

    @property
    def slow_center(self):
        if self.sticking_parameter.sticky_mode == 'quencher':
            coordinates = get_coordinates_of_residues(self.structure.atoms, self.quencher)
            s = [np.vstack(coordinates[res_key]) for res_key in coordinates if len(coordinates[res_key]) > 0]
            coordinates = np.vstack(s)
        elif self.sticking_parameter.sticky_mode == 'surface':
            slow_atoms = np.where(self.structure.atoms['atom_name'] == 'CB')[0]
            coordinates = self.structure.atoms['coord'][slow_atoms]
        return coordinates

    @property
    def av(self):
        """
        The accessible volume used to calculate the decay. If this is not provided or set it is calculated at the
        first access using :py:meth:`.calc_av`
        """
        if not isinstance(self._av, fps.ACV):
            self._av = self.calc_av()
            self._av.calc_acv(save=self.save_av, slow_centers=self.slow_center,
                              slow_radius=self.slow_radius, verbose=self.verbose)
        return self._av

    @av.setter
    def av(self, v):
        self._av = v

    @property
    def photon_trace(self):
        """
        The photon trace of the simulation. If it is `None` (default after initialization) it is calculated using
        the method :py:meth:`.calc_photons`
        """
        if self._photon_trace is None:
            self.calc_photons()
        return self._photon_trace

    @property
    def collisions(self):
        """
        Relative number of collided frames (That are frames below the critical distance)
        """
        try:
            return float(self.diffusion.collided.sum()) / self.diffusion.collided.shape[0]
        except AttributeError:
            return 0.0

    @property
    def structure(self):
        """
        Sets the structure used to calculate the donor-decay the structure should be of the type :class:`mfm.structure.structure`
        """
        return self._structure

    @structure.setter
    def structure(self, v):
        self._structure = Structure(v)

    @property
    def decay_mode(self):
        """
        This defines the way the time-resolved fluorescence is calculated. This parameter should be either *photon* or
        *curve*. If it is *photon* a Poisson process is simulated if *curve* the quenching rate is integrated to
        obtain a smooth curve.
        """
        return self._decay_mode

    @decay_mode.setter
    def decay_mode(self, v):
        self._decay_mode = v

    @property
    def attachment_residue(self):
        return self.dye_parameter.attachment_residue

    @property
    def attachment_atom_name(self):
        return self.dye_parameter.attachment_atom_name

    @property
    def attachment_chain(self):
        return self.dye_parameter.attachment_chain

    @property
    def kQ(self):
        return self.protein_quenching.kQ_scale

    @kQ.setter
    def kQ(self, v):
        self.protein_quenching.kQ_scale = v

    @property
    def quencher(self):
        return self.protein_quenching.quencher

    @quencher.setter
    def quencher(self, v):
        self.protein_quenching.quencher = v

    @property
    def n_photons(self):
        return self._n_photons

    @n_photons.setter
    def n_photons(self, v):
        self._n_photons = float(v)

    @property
    def t_max(self):
        return self._t_max

    @t_max.setter
    def t_max(self, v):
        self._t_max = v

    @property
    def t_step(self):
        return self._t_step

    @t_step.setter
    def t_step(self, v):
        self._t_step = float(v)

    def __init__(self, tau0=4.2, k_quench_protein=5.0, n_photons=10e6,
                 attachment_residue=None, attachment_atom=None, attachment_chain=None, dg=0.5, sticky_mode='quencher',
                 save_av=True, diffusion_coefficient=7.1, slow_fact=0.1, critical_distance=7.0, output_file='out',
                 av_parameter={'linker_length': 20.0, 'linker_width': 0.5, 'radius1': 3.5},
                 quencher={'TRP': ['CB'], 'TYR': ['CB'], 'HIS': ['CB'], 'PRO': ['CB']}, t_max=500.0, t_step=0.05,
                 slow_radius=10.0, decay_mode='curve', n_curves=10000,
                 **kwargs):

        self.fit = kwargs.get('fit', None)
        Curve.__init__(self)
        Model.__init__(self, fit=self.fit)

        self.dye_parameter = Dye(tau0=tau0, diffusion_coefficient=diffusion_coefficient,
                                          critical_distance=critical_distance,
                                          av_length=av_parameter['linker_length'],
                                          av_width=av_parameter['linker_width'],
                                          av_radius=av_parameter['radius1'],
                                          attachment_residue=attachment_residue,
                                          attachment_atom=attachment_atom,
                                          attachment_chain=attachment_chain
        )

        self.sticking_parameter = Sticking(
            slow_radius=slow_radius,
            slow_fact=slow_fact,
            sticky_mode='surface'
        )

        self.protein_quenching = ProteinQuenching(
            k_quench_protein=k_quench_protein,
            quenching_amino_acids=quencher
        )

        self.av_parameter = av_parameter
        self.convolve = kwargs.get('convolve', mfm.models.tcspc.nusiance.Convolve(self.fit))
        self.generic = kwargs.get('generic', mfm.models.tcspc.nusiance.Generic())
        self.corrections = kwargs.get('corrections', mfm.models.tcspc.nusiance.Corrections(model=self, **kwargs))

        self.quencher = quencher
        self._n_curves = n_curves
        self._decay_mode = decay_mode
        self._diffusion = None
        self._av = None
        self._structure = mfm.structure.structure.Structure()
        self.tau0 = tau0
        self._n_photons = n_photons
        self._photon_trace = None
        self.dg = dg
        self.save_av = save_av
        self.output_file = output_file
        self._t_max = t_max
        self._t_step = t_step

    def __repr__(self):
        s = ""
        try:
            s += "t_max: %s" % self.t_max + "\n"
            s += "t_step: %s" % self.t_step + "\n"
            s += "n_photons: %s" % self.n_photons + "\n"
            s += "PDB\n"
            s += "---\n"
            s += "%s" % self._structure.filename
            s += "\nAV\n"
            s += "--\n"
            s += "%s" % self._av
        except AttributeError:
            s += ""
        return s

    def calc_av(self, **kwargs):
        """
        :param kwargs:
            verbose (bool)
        :return:
        """
        verbose = kwargs.get('verbose', self.verbose)
        attachment_residue = kwargs.get('attachment_residue', self.attachment_residue)
        attachment_atom = kwargs.get('attachment_atom', self.attachment_atom_name)
        chain = kwargs.get('chain', self.attachment_chain)
        save_av = kwargs.get('save_av', self.save_av)
        dg = kwargs.get('dg', self.dg)
        structure = kwargs.get('structure', self.structure)
        av_parameter = kwargs.get('av_parameter', self.av_parameter)
        av = ACV(structure, residue_seq_number=attachment_residue, atom_name=attachment_atom, chain_identifier=chain,
                 simulation_grid_resolution=dg, save_av=save_av, verbose=verbose, output_file=self.output_file,
                 **av_parameter)
        return av

    def simulate_diffusion_trajectory(self, **kwargs):
        """
        :param kwargs:
            verbose; all_quencher_atoms; quencher; av
        :return:
        """
        verbose = kwargs.get('verbose', self.verbose)
        av = kwargs.get('av', self.av)
        diffusion = DiffusionSimulation(av, dye=self.dye_parameter,
                                      verbose=verbose, quencher_definition=self.protein_quenching,
                                      sticking_parameter=self.sticking_parameter)
        diffusion.run(D=self.diffusion_coefficient, slow_fact=self.slow_fact, t_max=self.t_max, t_step=self.t_step)
        return diffusion

    def calc_photons(self, verbose=False, donor_traj=None, kQ=None):
        """
        `n_ph`: number of generated photons
        `pq`: quenching probability if distance of fluorphore is below the `critical_distance`
        """
        verbose = verbose or self.verbose
        n_photons = self.n_photons
        if kQ is None:
            kQ = self.kQ
        tau0 = self.tau0
        if donor_traj is None:
            donor_traj = self.diffusion
        is_collided = donor_traj.collided
        if verbose:
            print("")
            print("Simulating decay:")
            print("----------------")
            print("Number of excitation photons: %s" % n_photons)
            print("Number of frames    : %s" % is_collided.shape[0])
            print("Number of collisions: %s" % is_collided.sum())
            print("Quenching constant kQ[1/ns]: %s" % kQ)

        t_step = donor_traj.t_step
        # dts, phs = photon.simulate_photon_trace_kQ(n_ph=n_photons, collided=is_collided, kQ=kQ, t_step=t_step, tau0=tau0)
        #dts, phs = photon.simulate_photon_trace(n_ph=n_photons, collided=is_collided, kQ=kQ, t_step=t_step, tau0=tau0)
        kq_array = is_collided * kQ
        dts, phs = photon.simulate_photon_trace_rate(n_ph=n_photons, quench=kq_array, t_step=t_step, tau0=tau0)

        n_photons = phs.shape[0]
        n_f = phs.sum()
        if verbose or self.verbose:
            print("Number of absorbed photons: %s" % (n_photons))
            print("Number of fluorescent photons: %s" % (n_f))
            print("Quantum yield: %.2f" % (float(n_f) / n_photons))
        dt2 = dts.take(np.where(phs > 0)[0])
        self._photon_trace = dt2

    def get_histogram(
            self,
            irf = None,
            decay_mode: str = None,
            **kwargs
    ) -> Tuple[
        np.array,
        np.array
    ]:
        """
        Returns two arrays one is the time axis and one is the photon-counts

        :param kwargs: Takes *nbins* the number of histogram bins, *tac_range* the histogram tac_range, *decay_mode* the
        way the histogram gets generated: either *photon* or *curve*. If this parameter is *photon* the histogram
        will be the result of the simulation of a Poisson process. If it is *curve* the sum of smooth curves will be
        returned based on several integrated quenched rates.
        """
        nbins = kwargs.get('nbins', self.nTAC)
        r = kwargs.get('tac_range', (0, 50, 0.0141))

        if decay_mode is None:
            decay_mode = self.decay_mode

        if decay_mode == 'photon':
            dts = self.photon_trace
            y, x = np.histogram(dts, bins=np.arange(r[0], r[1] + r[2], self.dtTAC))
            x = x[:-1]
        elif decay_mode == 'curve':
            x = self.x_values
            y = self._curve_y
            if isinstance(irf, np.ndarray):
                y = np.convolve(irf, y, mode='full')[:len(x)]

        return x, y

    def save_histogram(
            self,
            filename: str = 'hist.txt',
            verbose: bool = False,
            nbins:int = 4096,
            tac_range: Tuple[int, int] = (0, 50)
    ):
        """
        Used to save the current histogram to a file.

        :param filename:
        :param verbose:
        :param nbins:
        :param tac_range:
        """
        verbose = verbose or self.verbose
        x, y = self.get_histogram(nbins, tac_range)
        mfm.io.ascii.save_xy(filename, x, y, verbose, header_string="time\tcount")

    def update_decay_curve(self):
        self._curve_y = np.zeros(self.nTAC, dtype=np.float64)
        photon.simulate_decay_kQ(
            self.n_curves, self._curve_y, self.dtTAC,
            self.diffusion.collided,
            kQ=self.protein_quenching.kQ_scale,
            t_step=self.t_step,
            tau0=self.tau0
        )

    def update_model(
            self,
            **kwargs
    ):
        verbose = kwargs.get('verbose', self.verbose)
        lintable = kwargs.get('lintable', self.corrections.lintable)

        gc.collect()
        if verbose:
            print("Updating simulation")
        self.av = self.calc_av(**kwargs)
        self.av.calc_acv(save=self.save_av, slow_radius=self.slow_radius, slow_centers=self.slow_center)

        self.diffusion.av = self.av
        self.diffusion.update(slow_fact=self.slow_fact, diffusion_coefficient=self.diffusion_coefficient,
                              verbose=verbose)
        if verbose:
            print("Determining quencher distances.")
        self.diffusion.critical_distance = self.critical_distance
        if verbose:
            print("Simulating decay")
        if self.decay_mode == 'photon':
            self.calc_photons(verbose=verbose)
        elif self.decay_mode == 'curve':
            self.update_decay_curve()
        x, y = self.get_histogram()
        decay = self.convolve.convolve(y, mode='full')

        # scale the decay
        self.convolve.scale(decay, self.fit.data, bg=self.generic.background)
        decay += self.generic.background
        decay[decay < 0.0] = 0.0

        if lintable is not None:
            decay *= lintable

        self._y_values = decay


class TransientDecayGenerator(QtWidgets.QWidget, DyeDecay):

    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
                                                  }
                    ),
                    (plots.SurfacePlot, {}),
                    (plots.MolView, {
                        'quencher': mfm.common.quencher_names
                    })
    ]

    name = "Dye-diffusion"

    @property
    def n_curves(
            self
    ) -> int:
        return int(self.spinBox_4.value())

    @n_curves.setter
    def n_curves(
            self,
            v: int
    ):
        self.spinBox_4.setValue(v)

    @property
    def nTAC(
            self
    ) -> int:
        return self.nBins

    @nTAC.setter
    def nTAC(
            self,
            v: int
    ):
        # TODO remove this option
        pass

    @property
    def dtTAC(
            self
    ) -> float:
        return self.convolve.dt

    @dtTAC.setter
    def dtTAC(
            self,
            v
    ) -> float:
        self.convolve.dt = v

    @property
    def decay_mode(
            self
    ) -> str:
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
    def settings_file(
            self
    ) -> str:
        return str(self.lineEdit_2.text())

    @settings_file.setter
    def settings_file(
            self,
            v: str
    ):
        return self.lineEdit_2.setText(v)

    @property
    def all_quencher_atoms(
            self
    ) -> bool:
        return not bool(self.groupBox_5.isChecked())

    @all_quencher_atoms.setter
    def all_quencher_atoms(
            self,
            v: bool
    ):
        self.groupBox_5.setChecked(not v)

    @property
    def filename_prefix(
            self
    ) -> str:
        return str(self.lineEdit_5.text())

    @property
    def skip_frame(
            self
    ) -> int:
        return int(self.spinBox_3.value())

    @property
    def n_frames(
            self
    ) -> int:
        return int(self.spinBox.value())

    @property
    def nBins(
            self
    ) -> int:
        nbin = self.fitting_widget.xmax - self.fitting_widget.xmin
        return int(nbin)

    @property
    def n_photons(
            self
    ) -> int:
        return int(self.doubleSpinBox_11.value() * 1e6)

    @n_photons.setter
    def n_photons(
            self,
            v: float
    ):
        self.doubleSpinBox_11.setValue(v / 1e6)

    @property
    def critical_distance(
            self
    ) -> float:
        return self.dye_parameter.critical_distance

    @critical_distance.setter
    def critical_distance(
            self,
            v: float
    ):
        self.dye_parameter.critical_distance = v

    @property
    def t_max(
            self
    ) -> float:
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
    def t_step(
            self
    ) -> float:
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
        return self.doubleSpinBox_7.setValue(float(v * 1000.0))

    @property
    def attachment_chain(
            self
    ) -> str:
        return self.pdb_selector.chain_id

    @attachment_chain.setter
    def attachment_chain(
            self,
            v: str
    ):
        self.pdb_selector.chain_id = v

    @property
    def attachment_residue(
            self
    ) -> int:
        return self.pdb_selector.residue_id

    @attachment_residue.setter
    def attachment_residue(
            self,
            v: int
    ):
        self.pdb_selector.residue_id = v

    @property
    def attachment_atom_name(
            self
    ) -> str:
        return self.pdb_selector.atom_name

    @attachment_atom_name.setter
    def attachment_atom_name(
            self,
            v: str
    ):
        self.pdb_selector.atom_name = v

    def __init__(
            self,
            fit: mfm.fitting.fit.FitGroup,
            **kwargs
    ):
        self.verbose = kwargs.get('verbose', mfm.verbose)
        generic = mfm.models.tcspc.widgets.GenericWidget(fit=fit, parent=self, model=self, **kwargs)
        convolve = mfm.models.tcspc.widgets.ConvolveWidget(fit=fit, model=self, dt=fit.data.dt, **kwargs)
        corrections = mfm.models.tcspc.widgets.CorrectionsWidget(fit, model=self, **kwargs)

        fn = os.path.join(mfm.package_directory, 'settings/sample.json')
        settings_file = kwargs.get('dye_diffusion_settings_file', fn)
        settings = json.load(open(settings_file))
        DyeDecay.__init__(self, fit=fit, convolve=convolve, generic=generic, corrections=corrections, **settings)

        if not kwargs.get('disable_fit', False):
            fitting_widget = mfm.fitting.widgets.FittingControllerWidget(fit, **kwargs)
        else:
            fitting_widget = QtWidgets.QLabel()

        self.generic = generic
        self.convolve = convolve
        self.fitting_widget = fitting_widget

        QtWidgets.QWidget.__init__(self)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "dye_diffusion3.ui"
            ),
            self
        )

        self.pdb_selector = PDBSelector()
        self.verticalLayout_10.addWidget(self.pdb_selector)
        self.settings_file = settings_file

        self.protein_quenching = ProteinQuenchingWidget(
            k_quench_protein=kwargs.get('k_quench_protein', 5.0),
        )
        self.verticalLayout_11.addWidget(self.protein_quenching)
        self.dye_parameter = DyeWidget(**kwargs)
        self.verticalLayout_14.addWidget(self.dye_parameter)

        self.verticalLayout.addWidget(fitting_widget)
        self.verticalLayout.addWidget(convolve)
        self.verticalLayout.addWidget(generic)
        self.verticalLayout.addWidget(corrections)

        self.sticking_parameter = StickingWidget()
        self.verticalLayout_13.addWidget(self.sticking_parameter)

        # # User-interface
        self.actionLoad_PDB.triggered.connect(self.onLoadPDB)
        self.actionLoad_settings.triggered.connect(self.onLoadSettings)
        self.actionSave_AV.triggered.connect(self.onSaveAV)
        self.actionLoad_settings.triggered.connect(self.onLoadSettings)
        self.actionUpdate_all.triggered.connect(self.update_model)
        self.actionSave_histogram.triggered.connect(self.onSaveHist)
        self.actionSave_trajectory.triggered.connect(self.onSaveTrajectory)

        self.doubleSpinBox_6.valueChanged[double].connect(self.onSimulationTimeChanged)
        self.doubleSpinBox_7.valueChanged[double].connect(self.onSimulationDtChanged)

        self.tmp_dir = tempfile.gettempdir()
        self.diff_file = None
        self.av_slow_file = None
        self.av_fast_file = None
        self.fitting_widget = mfm.fitting.widgets.FittingWidget(fit=self.fit)

        self.hide()

    @property
    def molview(self):
        return [p for p in self.plots if isinstance(p, mfm.plots.MolView)][0]

    def onSaveHist(self, verbose=False):
        verbose = self.verbose or verbose
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', '.')[0]
        self.save_histogram(filename=filename, verbose=verbose)

    def onLoadSettings(self):
        print("onLoadSettings")
        print("Setting File: %s" % self.settings_file)

    def update_3d(self):
        diff_file, av_slow_file, av_fast_file = self.onSaveAV(directory=self.tmp_dir)
        self.diff_file = diff_file
        self.av_slow_file = av_slow_file
        self.av_fast_file = av_fast_file

        self.molview.reset()
        if isinstance(self.structure, Structure):
            self.molview.pymolWidget.openFile(self.structure.filename, frame=1, mode='cartoon', object_name='protein')
        self.molview.pymol.cmd.do("hide all")
        self.molview.pymol.cmd.do("show surface, protein")
        self.molview.pymol.cmd.do("color gray, protein")

        if self.diff_file is not None:
            self.molview.pymolWidget.openFile(self.diff_file, frame=1, object_name='trajectory')
            self.molview.pymol.cmd.do("color green, trajectory")

        self.molview.pymol.cmd.orient()
        #self.molview.highlight_quencher(self.quencher)

    def update_plots(self):
        Model.update_plots(self)
        self.update_3d()

    def update_model(self, **kwargs):
        DyeDecay.update_model(self, **kwargs)
        self.doubleSpinBox_16.setValue(self.quantum_yield)
        self.doubleSpinBox_15.setValue(self.collisions * 100.0)

    def onSaveAV(self, directory=None, verbose=True):
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
        mfm.io.xyz.write_xyz(av_slow_file, self.av.points_acv)

        av_fast_file = os.path.join(directory, self.filename_prefix + '_av_fast.xyz')
        if verbose:
            print("\nSaving slow AV...")
            print("Trajectory filename: %s" % av_fast_file)
        mfm.io.xyz.write_xyz(av_fast_file, self.av.points_fast)
        return diff_file, av_slow_file, av_fast_file

    def onSimulationDtChanged(self):
        time_steps = self.t_max / self.t_step
        self.spinBox.setValue(int(time_steps))

    def onSimulationTimeChanged(self):
        time_steps = self.t_max / self.t_step
        self.spinBox.setValue(int(time_steps))

    def onLoadPDB(self):
        #pdb_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open PDB-File', '', 'PDB-files (*.pdb)'))
        filename = mfm.widgets.get_filename('Open PDB-File', 'PDB-files (*.pdb)')
        self.lineEdit.setText(filename)
        self.structure = filename
        self.pdb_selector.atoms = self.structure.atoms
        self.tmp_dir = tempfile.gettempdir()
        self.update_3D()

    def onSaveTrajectory(self):
        filename = str(QtWidgets.QFileDialog.getSaveFileName(None, 'Open PDB-File', '', 'H5-FRET-files (*.fret.h5)'))[0]
        self.diffusion.save_simulation_results(filename)


