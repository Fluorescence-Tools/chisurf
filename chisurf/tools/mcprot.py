from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
from progressbar import Bar, ETA, ProgressBar, RotatingMarker, Percentage

import chisurf.mfm as mfm
from chisurf.math.rand import weighted_choice, mc
from mfm.structure.protein import ProteinCentroid
from mfm.structure.trajectory import TrajectoryFile, Universe

verbose = False


class ProteinMCWorker(object):

    def __init__(
            self,
            structure: mfm.structure.protein.ProteinCentroid,
            output_traj_file: str = None,
            settings: dict = None,
            update_rmsd: bool = False,
            verbose: bool = verbose,
            **kwargs
    ):
        self.exiting = False
        self._config_filename = None
        self.verbose = verbose
        if settings is None:
            settings = mfm.settings.cs_settings['mc_settings']
        self.settings = settings

        self.output_traj_file = output_traj_file
        if self.output_traj_file is None:
            _, new_file = tempfile.mkstemp(
                suffix=".h5"
            )
            if self.verbose:
                print("input_pdb: %s" % structure.labeling_file)
                print("output_file: %s" % new_file)
            self.output_traj_file = new_file

        self.u1 = Universe()
        self.update_rmsd = update_rmsd
        self.structure = structure
        self.output_traj = TrajectoryFile(
            self.structure,
            mode='w',
            filename=self.output_traj_file
        )

    def load_config(
            self,
            filename: str
    ):
        if self.verbose:
            print("Load config: %s" % filename)
        self._config_filename = filename
        if os.path.isfile(filename):
            self.settings = json.load(open(filename))
            self.add_potentials(**self.settings)
        if self.verbose:
            print(self.settings)

    def add_potentials(
            self,
            potentials: dict = None,
            **kwargs
    ):
        for p in potentials:
            try:
                pot = potentials.potentialDict[p['name']](structure=self.structure, **p['settings'])
                pot.hide()
                self.add_potential(potential=pot, potential_weight=p['weight'])
            except TypeError:
                pot = potentials.potentialDict[p['name']].__bases__[0](structure=self.structure, **p['settings'])
                self.add_potential(potential=pot, potential_weight=p['weight'])

    @property
    def move_map(self):
        try:
            if self.settings['movemap'] is not None:
                return np.array(self.settings['movemap'])
            else:
                n_res = self.structure.n_residues
                return np.ones(n_res, dtype=np.float64)
        except KeyError:
            n_res = self.structure.n_residues
            return np.ones(n_res, dtype=np.float64)

    def append(self, xyz, energy, fret_energy, **kwargs):
        kwargs['energy_fret'] = fret_energy
        kwargs['update_rmsd'] = self.update_rmsd
        kwargs['verbose'] = self.verbose
        kwargs['energy'] = energy
        self.output_traj.append(xyz, **kwargs)

    def add_potential(
            self,
            potential,
            potential_weight: float = 1.0,
            **kwargs
    ):
        potential = kwargs.get('potential', potential)
        self.u1.addPotential(potential, potential_weight)

    def mc_1u(self, **kwargs):
        """

        :param structure:
        :param kwargs:
        :return:

         Example
         -------

        >>> structure = mfm.structure.protein.ProteinCentroid('./test/data/atomic_coordinates/pdb_files/eGFP-mCherry.pqr')
        >>> mcw = ProteinMCWorker(structure)
        >>> mcw.run(n_out=100, n_iter=10000)
        """
        settings = dict(self.settings, **kwargs)
        #print settings
        u1 = self.u1

        structure = self.structure
        widgets = ['Simulating: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]
        verbose = kwargs.get('verbose', self.verbose)
        structure.auto_update = False
        structure.update()
        move_map = np.array(self.move_map)
        n_res = int(structure.n_residues)
        scale = settings['scale']
        n_out = settings['pdb_nOut']
        n_iter = settings['n_iter']
        kt = settings['kt']
        n_write = settings['n_written']

        s10 = structure
        s10.auto_update = False

        cPhi = np.empty_like(s10.phi)
        cPsi = np.empty_like(s10.psi)
        cOmega = np.empty_like(s10.omega)
        cChi = np.empty_like(s10.chi)
        nChi = cChi.shape[0]

        coord_back = np.empty_like(s10.internal_coordinates)
        np.copyto(coord_back, s10.internal_coordinates)

        e10 = u1.getEnergy(s10)
        self.append(s10.xyz, e10, 0.0)
        pPhi = self.settings['pPhi']
        pPsi = self.settings['pPsi']
        pOmega = self.settings['pOmega']
        pChi = self.settings['pChi']

        if verbose:
            print("n_iter: %s" % n_iter)
            print("n_out: %s" % n_out)
            #print "output_file: %s" % self.output_traj_file
            print("nRes: %s" % n_res)
            print("kt: %s" % kt)
            print("move_map: %s" % move_map)
            print("scale: %s" % scale)
            print("Potentials (name, energy):")
            for e, pot in zip(u1.getEnergies(s10), u1.potentials):
                print(pot.name + ": " + str(e))
            print("--------------------------------------\n")

        pbar = ProgressBar(widgets=widgets, maxval=n_write + 1).start()
        n_accepted = 0
        n_rejected = 0
        n_written = 0
        i = 0
        while i < n_iter and n_written < n_write:
            i += 1
            # decide which angle to move
            move_phi, move_psi, move_ome, move_chi = np.random.ranf(4) < [pPhi, pPsi, pOmega, pChi]
            # decide which aa to move
            moving_aa = weighted_choice(move_map, 1)
            if move_phi:
                cPhi *= 0.0
                cPhi[moving_aa] += (np.random.ranf() - 0.5) * scale
                s10.phi = (s10.phi + cPhi)
            if move_psi:
                cPsi *= 0.0
                cPsi[moving_aa] += (np.random.ranf() - 0.5) * scale
                s10.psi = (s10.psi + cPsi)
            if move_ome:
                cOmega *= 0.0
                cOmega[moving_aa] += (np.random.ranf() - 0.5) * scale
                s10.omega = (s10.omega + cOmega)
            if move_chi:
                cChi *= 0.0
                cChi[moving_aa % nChi] += (np.random.ranf() - 0.5) * scale
                s10.chi = (s10.chi + cChi)

            # Monte-Carlo step
            s10.update()
            e11 = u1.getEnergy(s10)
            if mc(e10, e11, kt):
                e10 = e11
                n_accepted += 1
                if n_accepted % n_out == 0:
                    self.append(s10.xyz, e11, 0.0)
                    pbar.update(n_written)
                    n_written += 1
                np.copyto(coord_back, s10.internal_coordinates)
            else:
                n_rejected += 1
                np.copyto(s10.internal_coordinates, coord_back)
        pbar.finish()
        fraction_rejected = float(n_rejected) / (float(n_accepted) + float(n_rejected))
        print("Rejection-ratio [%%]: %.2f" % (fraction_rejected * 100.0))
        print("Simulation finished!")

    def run(
            self,
            **kwargs
    ):
        mc_mode = self.settings['mc_mode']

        if mc_mode == 'simple':
            self.mc_1u(**kwargs)
        elif mc_mode == 'av_mc':
            self.monteCarlo2U(**kwargs)


if __name__ == "__main__":
    """
    Example for ChiSurf

    >>> import subprocess
    >>> from subprocess import Popen, CREATE_NEW_CONSOLE
    >>> subprocess.Popen("python -m tools.mcprot .\sample_data\modelling\pdb_files\eGFP-mCherry.pqr 50000 100 test.h5", creationflags=CREATE_NEW_CONSOLE)
    >>> o = subprocess.check_output("python -m tools.mcprot .\sample_data\modelling\pdb_files\eGFP-mCherry.pqr 1000 100 test.h5", shell=True)
    """
    parser = argparse.ArgumentParser(description='Monte Carlo simulation.')
    parser.add_argument('pdb_file', metavar='file', type=str, help='The starting pdb-filename')
    parser.add_argument('setting_file', metavar='file', type=str, help='Setting file (JSON-file).', default=None)
    parser.add_argument('output_file', metavar='output_file', type=str, help='The output hdf-file.')
    parser.add_argument("-v", "--verbose", type=bool, default=False, help='The program displays more output if True')
    parser.add_argument("-s", "--scale", type=float, help='Scaling factor of ')

    args = parser.parse_args()
    kwargs = vars(args)
    verbose = args.verbose
    if verbose:
        print("--------------------------------------")

    structure = ProteinCentroid(args.pdb_file)
    mcw = ProteinMCWorker(structure, output_traj_file=args.output_file)
    if args.setting_file is not None:
        mcw.load_config(args.setting_file)
    mcw.run()

