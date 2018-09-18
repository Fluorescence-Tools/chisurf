import time
import os
import copy
import tempfile
import json

from PyQt4 import QtCore, QtGui, uic
import numpy as np

import mfm
from mfm.math.rand import mc
from mfm.fitting.models import Model
from mfm.structure.potential import potentials
from mfm.structure.trajectory import TrajectoryFile, Universe
import mfm.structure
import mfm.math.rand as mrand
from mfm import plots
import mfm.widgets

mc_setting = mfm.settings['mc_settings']


class X(QtCore.QObject):

    def emitMySignal(self, xyz, energy, fret_energy, elapsed):
        self.emit(QtCore.SIGNAL('newStructure'), xyz, energy, fret_energy, elapsed)


class ProteinMCWorker(QtCore.QThread):

    daemonIsRunning = False
    daemonStopSignal = False
    x = X()

    def __init__(self, parent, **kwargs):
        QtCore.QThread.__init__(self, parent)
        self.parent = parent
        self.exiting = False
        self.verbose = kwargs.get('verbose', mfm.verbose)

    def monteCarlo2U(self, **kwargs):
        verbose = kwargs.get('verbose', self.verbose)
        p = self.p
        p.structure.auto_update = False
        p.structure.update()
        moveMap = np.array(p.movemap)
        nRes = int(p.structure.n_residues)

        if verbose:
            print("monteCarlo2U")
            print("moveMap: %s" % moveMap)
            print("Universe(1)-Energy: %s" % p.u1.getEnergy(p.structure))
            print("Universe(2)-Energy: %s" % p.u2.getEnergy(p.structure))
            print("nRes: %s" % nRes)
            print("nOut: %s" % p.pdb_nOut)

        # Get paramters from parent object
        scale = float(p.scale)
        ns = int(p.number_of_moving_aa)
        s10 = p.structure
        s10.auto_update = False
        av_number_protein_mc = int(p.av_number_protein_mc)

        start = time.time()
        elapsed = 0.0
        nAccepted = 0
        do_av_steepest_descent = p.do_av_steepest_descent

        cPhi = np.empty_like(s10.phi)
        cPsi = np.empty_like(s10.psi)
        cOmega = np.empty_like(s10.omega)
        cChi = np.empty_like(s10.chi)
        nChi = cChi.shape[0]

        e20 = p.u2.getEnergy(s10)
        e10 = p.u1.getEnergy(s10)

        coord_back_1 = np.empty_like(s10.internal_coordinates)
        coord_back_2 = np.empty_like(s10.internal_coordinates)

        while not self.daemonStopSignal:
            elapsed = (time.time() - start)
            # save configuration before inner MC-loop
            np.copyto(coord_back_2, s10.internal_coordinates)

            # inner Monte-Carlo loop
            # run av_number_protein_mc monte carlo-steps
            for trialRes in range(av_number_protein_mc):
                # decide which aa to move
                moving_aa = mrand.weighted_choice(moveMap, n=ns)
                # decide which angle to move
                move_phi, move_psi, move_ome, move_chi = np.random.ranf(4) < [p.pPhi, p.pPsi, p.pOmega, p.pChi]
                # save coordinates
                np.copyto(coord_back_1, s10.internal_coordinates)
                # move aa
                if move_phi:
                    cPhi *= 0.0
                    cPhi[moving_aa] += np.random.normal(0.0, scale, ns) * s10.phi[moving_aa]
                    s10.phi = (s10.phi + cPhi)
                if move_psi:
                    cPsi *= 0.0
                    cPsi[moving_aa] += np.random.normal(0.0, scale, ns) * s10.psi[moving_aa]
                    s10.psi = (s10.psi + cPsi)
                if move_ome:
                    cOmega *= 0.0
                    cOmega[moving_aa] += np.random.normal(0.0, scale, ns) * s10.omega[moving_aa]
                    s10.omega = (s10.omega + cOmega)
                if move_chi:
                    cChi *= 0.0
                    cChi[moving_aa % nChi] += np.random.normal(0.0, scale, ns) * s10.chi[moving_aa % nChi]
                    s10.chi = (s10.chi + cChi)

                # Monte-Carlo step
                s10.update(min(moving_aa))
                e11 = p.u1.getEnergy(s10)
                if mc(e10, e11, p.kt):
                    e10 = e11
                    np.copyto(coord_back_1, s10.internal_coordinates)
                else:
                    np.copyto(s10.internal_coordinates, coord_back_1)

            s10.update()
            e21 = p.u2.getEnergy(s10)
            accept = e21 < e20 if do_av_steepest_descent else mc(e20, e21, p.ktAv)

            if accept:
                # AV-MC accepted
                e20 = e21
                nAccepted += 1
                if nAccepted % p.pdb_nOut == 0:
                    self.x.emitMySignal(s10.xyz, e10, e20, elapsed)
            else:
                # AV-MC not accepted return to stored coordinates
                np.copyto(s10.internal_coordinates, coord_back_2)

    def monteCarlo1U(self, verbose=True):
        verbose = verbose or self.verbose
        p = self.p
        p.auto_update = False

        p.structure.update()
        moveMap = np.array(p.movemap)
        nRes = int(p.structure.n_residues)
        if verbose:
            print("monteCarlo1U")
            print("moveMap: %s" % moveMap)
            print("Universe(1)-Energy: %s" % p.u1.getEnergy(p.structure))
            print("nRes: %s" % nRes)
            print("nOut: %s" % p.pdb_nOut)

        # Get paramters from parent object
        scale = float(p.scale)
        ns = int(p.number_of_moving_aa)
        s10 = p.structure
        s10.auto_update = False

        start = time.time()
        nAccepted = 0

        cPhi = np.empty_like(s10.phi)
        cPsi = np.empty_like(s10.psi)
        cOmega = np.empty_like(s10.omega)
        cChi = np.empty_like(s10.chi)
        nChi = cChi.shape[0]

        coord_back = np.empty_like(s10.internal_coordinates)
        np.copyto(coord_back, s10.internal_coordinates)

        e10 = p.u1.getEnergy(s10)
        self.x.emitMySignal(s10.xyz, e10, 0.0, 0.0)

        while not self.daemonStopSignal:
            elapsed = (time.time() - start)
            # decide which angle to move
            move_phi, move_psi, move_ome, move_chi = np.random.ranf(4) < [p.pPhi, p.pPsi, p.pOmega, p.pChi]
            # decide which aa to move
            moving_aa = mrand.weighted_choice(moveMap, n=ns)
            if move_phi:
                cPhi *= 0.0
                cPhi[moving_aa] += np.random.normal(0.0, scale, ns) * s10.phi[moving_aa]
                s10.phi = (s10.phi + cPhi)
            if move_psi:
                cPsi *= 0.0
                cPsi[moving_aa] += np.random.normal(0.0, scale, ns) * s10.psi[moving_aa]
                s10.psi = (s10.psi + cPsi)
            if move_ome:
                cOmega *= 0.0
                cOmega[moving_aa] += np.random.normal(0.0, scale, ns) * s10.omega[moving_aa]
                s10.omega = (s10.omega + cOmega)
            if move_chi:
                cChi *= 0.0
                cChi[moving_aa % nChi] += np.random.normal(0.0, scale, ns) * s10.chi[moving_aa % nChi]
                s10.chi = (s10.chi + cChi)

            # Monte-Carlo step
            s10.update(start_point=min(moving_aa))
            e11 = p.u1.getEnergy(s10)
            if mc(e10, e11, p.kt):
                e10 = e11
                nAccepted += 1
                if nAccepted % p.pdb_nOut == 0:
                    self.x.emitMySignal(s10.xyz, e11, 0.0, elapsed)
                np.copyto(coord_back, s10.internal_coordinates)
            else:
                np.copyto(s10.internal_coordinates, coord_back)

    def setDaemonStopSignal(self, bool):
        self.daemonStopSignal = bool

    def run(self):
        self.daemonIsRunning = True
        self.daemonStopSignal = False

        if self.parent.mc_mode == 'simple':
            self.monteCarlo1U()
        elif self.parent.mc_mode == 'av_mc':
            self.monteCarlo2U()


class ProteinMonteCarloControl(TrajectoryFile, Model):

    @property
    def movemap(self):
        return self._movemap

    @movemap.setter
    def movemap(self, v):
        print v
        if v is None:
            mm = copy.copy(self.structure.b_factors)
            mm += 1.0 if max(mm) == 0.0 else 0.0
            self._movemap = mm / max(mm)
        else:
            v = np.array(v)
            self._movemap = v
            self.structure.b_factors = v

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, v):
        v = str(v)
        TrajectoryFile.__init__(self, self.structure, mode='w', filename=v)
        self._filename = v

    @property
    def fps_file(self):
        return self.av.filename

    @fps_file.setter
    def fps_file(self, v):
        if v is None:
            self.mc_mode = 'simple'
        else:
            self.av.filename = v

    @property
    def config_filename(self):
        return self._config_filename

    @config_filename.setter
    def config_filename(self, v):
        self.load_config(v)

    def get_wres(self, *args, **kwargs):
        return np.ones(10)

    def __init__(self, fit, **kwargs):
        filename = kwargs.get('save_filename', tempfile.mktemp(".h5"))

        Model.__init__(self, fit=fit)
        structure = mfm.structure.ProteinCentroid(fit.data, make_coarse=True)
        TrajectoryFile.__init__(self, structure, mode='w')
        self.fit = fit

        self.ktAv = kwargs.get('ktAv', mc_setting['ktAv'])
        self.kt = kwargs.get('kt', mc_setting['kt'])
        self.scale = kwargs.get('scale', mc_setting['scale'])
        self.pPsi = kwargs.get('pPsi', mc_setting['pPsi'])
        self.pPhi = kwargs.get('pPhi', mc_setting['pPhi'])
        self.pChi = kwargs.get('pChi', mc_setting['pChi'])
        self.pOmega = kwargs.get('pOmega', mc_setting['pOmega'])
        self.pdb_nOut = kwargs.get('pdbOut', mc_setting['pdbOut'])
        self.av_number_protein_mc = kwargs.get('av_number_protein_mc', mc_setting['av_number_protein_mc'])
        self.number_of_moving_aa = kwargs.get('number_of_moving_aa', mc_setting['number_of_moving_aa'])
        self.update_rmsd = kwargs.get('update_rmsd', mc_setting['update_rmsd'])
        self.movemap = kwargs.get('movemap', None)

        self._config_filename = kwargs.get('config_file', None)
        if self._config_filename is not None:
            self.load_config(self._config_filename)

        self.do_av_steepest_descent = kwargs.get('do_av_steepest_descent', mc_setting['do_av_steepest_descent'])
        self.potential_weight = 1.0
        self.cluster_structures = kwargs.get('cluster_structures', mc_setting['cluster_structures'])
        self.av_filename = kwargs.get('av_filename', mc_setting['av_filename'])
        self.outputDir = os.path.dirname(filename)
        self.append_new_structures = kwargs.get('append_new_structures', mc_setting['append_new_structures'])
        self.fps_file = kwargs.get('fps_file', None)

        # initialize variables
        self.u1 = Universe()
        self.u2 = Universe()

    def get_config(self):
        parameter = dict()
        parameter['number_of_moving_aa'] = self.number_of_moving_aa
        parameter['save_filename'] = self.filename
        parameter['pPsi'] = self.pPsi
        parameter['pPhi'] = self.pPhi
        parameter['pOmega'] = self.pOmega
        parameter['pChi'] = self.pChi
        parameter['pdb_nOut'] = self.pdb_nOut
        parameter['av_number_protein_mc'] = self.av_number_protein_mc
        parameter['ktAv'] = self.ktAv
        parameter['kt'] = self.kt
        parameter['scale'] = self.scale
        parameter['mc_mode'] = self.mc_mode
        parameter['fps_file'] = self.fps_file
        parameter['movemap'] = list(self.movemap)
        parameter['do_av_steepest_descent'] = self.do_av_steepest_descent
        n = list()
        for i, p in enumerate(self.u1.potentials):
            d = {
                'name': p.name,
                'weight': self.u1.scaling[i]
            }
            n.append(d)
        parameter['potentials'] = n
        return parameter

    def clearPotentials(self):
        table = self.tableWidget
        self.u1.clearPotentials()
        for i in reversed(range(table.rowCount())):
            table.removeRow(i)

    def set_config(self, **kwargs):
        self.number_of_moving_aa = kwargs.get('number_of_moving_aa', 1)
        self.pPsi = kwargs.get('pPsi', 0.3)
        self.pPhi = kwargs.get('pPhi', 0.7)
        self.pOmega = kwargs.get('pOmega', 0.00)
        self.pChi = kwargs.get('pChi', 0.01)
        self.pdb_nOut = kwargs.get('pdb_nOut', 5)
        self.av_number_protein_mc = kwargs.get('av_number_protein_mc', 50)
        self.ktAv = kwargs.get('ktAv', 1.0)
        self.kt = kwargs.get('kt', 1.5)
        self.scale = kwargs.get('scale', 0.0025)
        self.mc_mode = kwargs.get('mc_mode', 'av_mc')
        self.fps_file = kwargs.get('fps_file', None)
        self.movemap = kwargs.get('movemap', None)
        self.do_av_steepest_descent = kwargs.get('do_av_steepest_descent', True)
        self.filename = kwargs.get('save_filename', 'out.h5')
        self.structure.b_factors = self.movemap
        self.clearPotentials()
        for p in kwargs.get('potentials'):
            pot = potentials.potentialDict[p['name']](structure=self.structure, parent=self)
            pot.hide()
            self.add_potential(potential=pot, potential_weight=p['weight'])

    def append(self, xyz, energy, fret_energy, elapsed_time, **kwargs):
        kwargs['energy_fret'] = fret_energy
        kwargs['update_rmsd'] = self.update_rmsd
        kwargs['verbose'] = self.verbose
        kwargs['energy'] = energy
        TrajectoryFile.append(self, xyz, **kwargs)

    def add_potential(self, potential, potential_weight):
        print "ProteinMonteCarloControl:add_potential"
        self.u1.addPotential(potential, potential_weight)

    def remove_potential(self, idx):
        self.u1.removePotential(idx)

    def run_simulation(self):
        self.proteinWorker.setDaemonStopSignal(False)
        if self.movemap is None:
           raise ValueError("Movemap not set not possible to perform MC-integration")
        self.proteinWorker.p = self
        self.proteinWorker.start()

    def save_config(self, filename):
        p = self.get_config()
        json.dump(p, open(filename, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
        self.config_filename = filename

    def load_config(self, filename):
        self._config_filename = filename
        if os.path.isfile(filename):
            p = json.load(open(filename))
            self.set_config(**p)


class ProteinMonteCarloControlWidget(QtGui.QWidget, ProteinMonteCarloControl):

    name = "ProteinMC"

    plot_classes = [(plots.ProteinMCPlot, {}),
                    #(plots.SurfacePlot, {}),
                    (plots.MolView, {
                        'enableUi': False,
                        'mode': 'coarse',
                        'sequence': False
                    })
    ]

    def __init__(self, fit, **kwargs):
        """
        Parameters
        ----------
        :param config_file: string, optional
            Filename containing the parameters for the simulation. If this is specified all other parameters
            are taken from the configuration file and the passed parameters are overwritten.
        :param fps_file: string, optional
            Filename of the fps-labeling file (JSON-File).
        :param movemap: array of floats
            If specified should have length of residues. Specifies probability that dihedral of a certain amino-acid
            is changed.
        :param fit:
        :param number_of_moving_aa: int
            The number of amino-acids move in one Monte-Carlo step
        :param save_filename: str, optional
            Filename the trajectory is saved to. The filename-ending should be '.h5'. If no filename is provided
            a temporary filename is generated and used.
        :param pPsi: float
            Probability of moving the Psi-angle
        :param pPhi: float
            Probability of moving the Phi-angle
        :param pOmega: float
            Probability of moving the Omega-angle
        :param pChi: float
            Probability of moving the Chi-angle
        :param pdbOut: int
            Only every pdbOut frame is written to the tajectory
        :param av_number_protein_mc: int
            Number of MC-steps (total number of accepted number) performed before an AV-Monte-Carlo-step (AV-MC)
            is performed.
        :param ktAv: float
            Specifies the 'temperature' of the AV-MC. Here Chi2 is taken as the energy of the AV.
        :param kt: float
            Specifies the temperature of the MC-step
        :param maxTime: number, optional
            Maximum time (in real-time/laboratory-time) the simulation is performed in seconds. If no value
            is provided the simulations runs for one hour.
        :param scale: float
            Magnitude of change of the Monte-Carlo step. The actual change in each MC-step is determined
            by taking a random number out of a normal-distribution of a width of 'scale'.
        """
        ProteinMonteCarloControl.__init__(self, fit, **kwargs)
        QtGui.QWidget.__init__(self)
        self.icon = QtGui.QIcon(":/icons/icons/Peptide_FRET.ico")
        uic.loadUi('./mfm/ui/proteinMC2.ui', self)
        self.proteinWorker = ProteinMCWorker(parent=self)

        self.av = mfm.fluorescence.fps.AvWidget(self)
        self.u2.addPotential(self.av)

        #p = potentials.potentialDict['H-Potential'](structure=self.structure, parent=self)
        #p.hide()
        #self.add_potential(potential=p)

        #p = potentials.potentialDict['Iso-UNRES'](structure=self.structure, parent=self)
        #p.hide()
        #self.add_potential(potential=p)

        self.update_ui()

        ## User-interface
        self.verticalLayout.addWidget(self.av)
        self.connect(self.actionAdd_potential, QtCore.SIGNAL('triggered()'), self.add_potential)
        self.connect(self.actionSet_output_filename, QtCore.SIGNAL('triggered()'), self.onSetOutputFilename)
        self.connect(self.actionRun_Simulation, QtCore.SIGNAL('triggered()'), self.run_simulation)
        self.connect(self.actionLoad_configuration_file, QtCore.SIGNAL('triggered()'), self.on_load_config)
        self.connect(self.actionSave_configuration_file, QtCore.SIGNAL('triggered()'), self.save_config)
        self.connect(self.actionStop_simulation, QtCore.SIGNAL('triggered()'), self.onMTStop)
        self.connect(self.actionSave_RMSD_table, QtCore.SIGNAL('triggered()'), self.onSaveSimulationTable)
        self.connect(self.actionCurrent_potential_changed, QtCore.SIGNAL('triggered()'), self.onSelectedPotentialChanged)
        self.connect(self.tableWidget, QtCore.SIGNAL("cellDoubleClicked (int, int)"), self.on_remove_potential)
        self.comboBox_2.addItems(list(potentials.potentialDict.keys()))
        self.comboBox_2.setCurrentIndex(1)
        self.comboBox_2.setCurrentIndex(0)

        mfm.c(self.checkBox.stateChanged, "cs.current_fit.model.update_rmsd=%s", self.checkBox.isChecked)
        mfm.c(self.radioButton_2.toggled, "cs.current_fit.model.do_av_steepest_descent=%s", self.radioButton_2.isChecked)
        mfm.c(self.spinBox_3.valueChanged, "cs.current_fit.model.av_number_protein_mc=%s", self.spinBox_3.value)
        mfm.c(self.checkBox_4.stateChanged, "cs.current_fit.model.append_new_structures=%s", self.checkBox_4.isChecked)
        mfm.c(self.spinBox.valueChanged, "cs.current_fit.model.number_of_moving_aa=%s", self.spinBox.value)

        self.groupBox_7.toggled.connect(lambda: mfm.run(
            "cs.current_fit.model.mc_mode = %s\n" % ('"av_mc"'
                                                                  if self.groupBox_7.isChecked()
                                                                  else '"simple"'))
        )
        self.doubleSpinBox_8.valueChanged.connect(lambda: mfm.run(
            "cs.current_fit.model.ktAv = %s\n" % self.doubleSpinBox_8.value())
        )
        self.doubleSpinBox.valueChanged.connect(lambda: mfm.run(
            "cs.current_fit.model.scale = %s\n" % (self.doubleSpinBox.value() / 1000.0) )
        )
        self.spinBox_2.valueChanged.connect(lambda: mfm.run(
            "cs.current_fit.model.pdb_nOut = %s\n" % self.spinBox_2.value())
        )
        self.doubleSpinBox_2.valueChanged.connect(lambda: mfm.run(
            "cs.current_fit.model.kt = %s\n" % self.doubleSpinBox_2.value())
        )
        self.doubleSpinBox_3.valueChanged.connect(lambda: mfm.run(
            "cs.current_fit.model.pPsi = %s\n" % self.doubleSpinBox_3.value())
        )
        self.doubleSpinBox_4.valueChanged.connect(lambda: mfm.run(
            "cs.current_fit.model.pPhi = %s\n" % self.doubleSpinBox_4.value())
        )
        self.doubleSpinBox_5.valueChanged.connect(lambda: mfm.run(
            "cs.current_fit.model.pOmega = %s\n" % self.doubleSpinBox_5.value())
        )
        self.doubleSpinBox_6.valueChanged.connect(lambda: mfm.run(
            "cs.current_fit.model.pChi = %s\n" % self.doubleSpinBox_6.value())
        )
        """
        self.groupBox_6.toggled.connect(lambda: mfm.run(
            "cs.current_fit.model.cluster_structures = %s\n" % self.groupBox_6.isChecked())
        )
        self.lineEdit.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.av_filename = %s\n" % self.lineEdit.text())
        )
        """

        ## QtThread
        self.connect(self.proteinWorker, QtCore.SIGNAL("finished()"), self.onSimulationStop)
        self.connect(self.proteinWorker, QtCore.SIGNAL("stopped()"), self.onSimulationStop)
        self.connect(self.proteinWorker.x, QtCore.SIGNAL("newStructure"), self.append)

    def update_ui(self):
        # Update user interface
        self.lineEdit_13.setText(self.filename)
        self.checkBox.setChecked(self.update_rmsd)
        self.radioButton_2.setChecked(self.do_av_steepest_descent)
        self.spinBox_3.setValue(self.av_number_protein_mc)
        if self.mc_mode == "av_mc":
            self.groupBox_7.setChecked(True)
        else:
            self.groupBox_7.setChecked(False)
        self.checkBox_4.setChecked(self.append_new_structures)
        self.spinBox.setValue(self.number_of_moving_aa)
        self.doubleSpinBox_8.setValue(self.ktAv)
        self.doubleSpinBox.setValue(self.scale * 1000.0)
        self.spinBox_2.setValue(self.pdb_nOut)
        self.doubleSpinBox_2.setValue(self.kt)
        self.doubleSpinBox_3.setValue(self.pPsi)
        self.doubleSpinBox_4.setValue(self.pPhi)
        self.doubleSpinBox_5.setValue(self.pOmega)
        self.doubleSpinBox_6.setValue(self.pChi)

    def onSetOutputFilename(self):
        filename = mfm.widgets.save_file("Save File", 'MC-trajectory output (*.h5)')
        mfm.run(
            "cs.current_fit.model.filename = '%s'\n" % filename
        )
        self.lineEdit_13.setText(filename)

    def save_config(self, filename=None):
        if filename is None:
            filename = mfm.widgets.save_file("Save File", 'MC-config files (*.mc.json)')
        ProteinMonteCarloControl.save_config(self, filename)

    def on_load_config(self):
        filename = mfm.widgets.open_file('Open File', 'MC-config files (*.mc.json)')
        mfm.run("cs.current_fit.model.load_config('%s')\n" % filename)

    def load_config(self, filename=None):
        self.lineEdit_2.setText(filename)
        ProteinMonteCarloControl.load_config(self, filename)
        #self.update_ui()

    def append(self, xyz, energy, fret_energy, elapsed_time):
        ProteinMonteCarloControl.append(self, xyz, energy, fret_energy, elapsed_time)

        table = self.tableWidget_3
        rows = table.rowCount()
        table.insertRow(rows)
        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, float(self.drmsd[-1]))
        table.setItem(rows, 0, tmp)
        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, float(energy))
        table.setItem(rows, 1, tmp)
        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, float(fret_energy))
        table.setItem(rows, 2, tmp)
        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, float(self.rmsd[-1]))
        table.setItem(rows, 3, tmp)

        # update plots
        if self.append_new_structures:
            #mfm.run("cs.current_fit.pymolPlot.append_structure(cs.current_fit.model.structure)")
            self.pymolPlot.append_structure(self.structure)
        self.fit.plots[0].update_all()

    def onSelectedPotentialChanged(self):
        layout = self.verticalLayout_2
        for i in range(layout.count()):
            layout.itemAt(i).widget().close()
        layout.addWidget(self.potential)

    @property
    def potential(self):
        return potentials.potentialDict[self.potential_name](structure=self.structure, parent=self)

    def add_potential(self, **kwargs):
        potential_weight = kwargs.get('potential_weight', self.potential_weight)
        potential = kwargs.get('potential', self.potential)
        ProteinMonteCarloControl.add_potential(self, potential=potential, potential_weight=potential_weight)

        table = self.tableWidget
        rc = table.rowCount()
        table.insertRow(rc)
        tmp = QtGui.QTableWidgetItem(str(potential.name))
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(rc, 0, tmp)
        tmp = QtGui.QTableWidgetItem(str(potential_weight))
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(rc, 1, tmp)
        table.resizeRowsToContents()

    def remove_potential(self, idx):
        table = self.tableWidget
        rc = table.rowCount()
        if rc >= 0:
            if idx < 0:
                idx = 0
            table.removeRow(idx)
        ProteinMonteCarloControl.remove_potential(self, idx)

    def on_remove_potential(self, idx=None):
        table = self.tableWidget
        if idx is None:
            idx = int(table.currentIndex().row())
        mfm.run("cs.current_fit.model.remove_potential(%s)\n" % idx)

    def onSaveSimulationTable(self):
        filename = str(QtGui.QFileDialog.getSaveFileName(self, 'Save structure table', '.txt'))
        fp = open(filename, 'w')
        s = "dRMSD\tEnergy\tFRET(chi2)\tRMSD(vsFirst)\n"
        fp.write(s)
        table = self.tableWidget_3
        for r in range(table.rowCount()):
            drmsd = table.item(r, 0).text()
            e = table.item(r, 1).text()
            chi2 = table.item(r, 2).text()
            rmsd = table.item(r, 3).text()
            s = "%s\t%s\t%s\t%s\n" % (drmsd, e, chi2, rmsd)
            fp.write(s)
        fp.close()

    def onSimulationStop(self):
        mfm.widgets.MyMessageBox('Simulation finished')

    def onMTStop(self):
        self.proteinWorker.setDaemonStopSignal(True)

    @property
    def plot(self):
        return [p for p in self.plots if isinstance(p, mfm.plots.ProteinMCPlot)][0]

    @property
    def pymolPlot(self):
        return [p for p in self.fit.plots if isinstance(p, mfm.plots.MolView)][0]

    @property
    def calculate_av(self):
        return self.groupBox_7.isChecked()

    @property
    def potential_number(self):
        return int(self.comboBox_2.currentIndex())

    @property
    def potential_name(self):
        return list(potentials.potentialDict.keys())[self.potential_number]

