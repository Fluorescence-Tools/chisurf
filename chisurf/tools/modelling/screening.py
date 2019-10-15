from __future__ import annotations

import os
import pickle

from qtpy import QtCore, QtWidgets
import numpy as np
from scipy.stats import f

import chisurf.decorators
import chisurf.structure.cluster
from chisurf.structure.potential import potentials
import chisurf.structure as slib
from chisurf.structure.trajectory import TrajectoryFile
import chisurf.widgets


class FPSScreenTrajectory(QtWidgets.QWidget):

    name = "Screening"
    modelID = 0

    @chisurf.decorators.init_with_ui(ui_filename="filterStructures.ui")
    def __init__(
            self,
            fit,
            parent,
            *args,
            **kwargs
    ):
        self.structure_table_data = []
        try:
            filenames = fit.data.filenames
            load_structures = False
        except AttributeError:
            filenames = [fit.data.labeling_file]
            load_structures = True
        TrajectoryFile.__init__(
            self,
            filenames=filenames,
            load_structures=load_structures
        )
        self.calc_drmsd()
        self.calc_rmsd()

        self.initWidgets()
        self.update_widgets()

        self.structures = fit.data.structures
        self.avPotential = potentials.AvWidget(self)
        self.parent = parent
        self.surface = self
        self.fit = fit

        self.clZ = None
        self.clDistances = None
        self.clAssignment = None

        self.comboBox_2.clear()
        atN = list(self.structures[0].atom_types)
        atN.sort()
        self.comboBox_2.addItems(atN + ['all'])

        # Connect signals
        self.pushButton_12.clicked.connect(self.onSaveStructureTable)
        self.pushButton_6.clicked.connect(self.onClusterClicked)
        self.pushButton_14.clicked.connect(self.onSaveClusterResults)
        self.pushButton_10.clicked.connect(self.onLoadClusterResults)
        self.pushButton_4.clicked.connect(self.onLoadLabeling)
        self.pushButton_8.clicked.connect(self.onScreeningStart)
        self.pushButton_15.clicked.connect(self.onClearClusters)
        self.pushButton_5.clicked.connect(self.onSaveClusterTable)
        self.pushButton_21.clicked.connect(self.onLoadTableOnStructures)
        self.tableWidget.cellClicked [int, int].connect(self.onStructureTableClicked)
        self.tableWidget_2.cellClicked [int, int].connect(self.onClusterTableClicked)
        self.radioButton.clicked.connect(self.onAlignAverage)

        # selection changed events
        self.radioButton_9.clicked.connect(self.onSelectionChanged)
        self.radioButton_7.clicked.connect(self.onSelectionChanged)
        self.radioButton_5.clicked.connect(self.onSelectionChanged)
        self.radioButton_6.clicked.connect(self.onSelectionChanged)


    @property
    def cluster(self):
        return np.zeros_like(self.rmsd)

    @property
    def chi2_weights(self):
        chis, dof = self.chi2r, self.dof
        chis = np.array(chis)
        fValues = chis / min(chis)
        p = f.pdf(fValues, dof, dof)
        return p

    @property
    def align_atom(self):
        return str(self.comboBox_2.currentText())

    @property
    def threshold(self):
        return self.doubleSpinBox_3.value()

    def onClearClusters(self):
        print("onClearClusters")
        self.tableWidget_2.setRowCount(0)
        self.clZ = None
        self.clDistances = None

    ######### OKAY #################

    def onAlignAverage(self):
        self.rmsd_ref_state = 'average'

    @property
    def rmsd_ref_state(self):
        return str(self.lineEdit.text())

    @property
    def chi2Max(self):
        return float(self.doubleSpinBox.value())

    @property
    def dof(self):
        return float(self.doubleSpinBox_2.value())

    @rmsd_ref_state.setter
    def rmsd_ref_state(self, v):
        self.lineEdit.setText("%s" % v)
        self.calc_rmsd()
        self.update_widgets()
        try:
            self.plot.update()
        except AttributeError:
            pass

    @property
    def pymolPlot(self):
        return [p for p in self.fit if isinstance(p, experiments.plots.MolView)][0]

    @property
    def plot(self):
        return [p for p in self.fit if isinstance(p, experiments.plots.ProteinMCPlot)][0]

    def onStructureTableClicked(self):
        row = int(self.tableWidget.currentRow())
        self.radioButton_2.setChecked(True)
        self.rmsd_ref_state = self.tableWidget.item(row, 0).data(0).toInt()[0]

    def update_widgets(self):
        rmsds = self.rmsd
        drmsd = self.drmsd
        chi2 = self.chi2r
        cluster = self.cluster
        for i, table_row in enumerate(self.structure_table_data):
            table_row[0].set_data(0, i)
            table_row[1].setText(self.filenames[i])
            table_row[2].set_data(0, drmsd[i])
            table_row[3].set_data(0, rmsds[i])
            table_row[4].set_data(0, float(chi2[i]))
            table_row[5].set_data(0, float(cluster[i]))

    def initWidgets(self):
        self.comboBox.addItems(slib.clusterCriteria)
        # structure-table
        nStructures = len(self)
        table = self.tableWidget
        table.setRowCount(nStructures)
        for i, filename in enumerate(self.filenames):
            table_row = []

            tmp = QtWidgets.QTableWidgetItem()
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(i, 0, tmp)
            table_row += [tmp]

            tmp = QtWidgets.QTableWidgetItem()
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(i, 1, tmp)
            table_row += [tmp]

            tmp = QtWidgets.QTableWidgetItem()
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(i, 2, tmp)
            table_row += [tmp]

            tmp = QtWidgets.QTableWidgetItem()
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(i, 3, tmp)
            table_row += [tmp]

            tmp = QtWidgets.QTableWidgetItem()
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(i, 4, tmp)
            table_row += [tmp]

            tmp = QtWidgets.QTableWidgetItem()
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(i, 5, tmp)
            table_row += [tmp]

            self.structure_table_data.append(table_row)

        table.resizeRowsToContents()

    def onSaveStructureTable(self):
        print("onSaveStructureTable")
        filename = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save structure table', '.txt'))[0]
        with chisurf.fio.zipped.open_maybe_zipped(
                filename=filename,
                mode='w'
        ) as fp:
            s = "SN\tchi2\trmsd\tf\tfilename\n"
            fp.write(s)
            for r in range(self.tableWidget.rowCount()):
                filename = self.tableWidget.item(r, 1).text()
                chi2 = self.tableWidget.item(r, 2).text()
                rmsd = self.tableWidget.item(r, 3).text()
                cl = self.tableWidget.item(r, 4).text()
                s = "%s\t%s\t%s\t%s\t%s\n" % (r, chi2, rmsd, cl, filename)
                fp.write(s)

    ######### OKAY #################

    def onClusterClicked(self):
        print("onClusterClicked")
        threshold = self.threshold
        idx = self.comboBox.currentIndex()
        criterion = str(slib.clusterCriteria[idx])

        print("Clustering of the structures...")
        Z = self.clZ
        if len(self.structures) > 0:
            file = os.path.abspath(self.structures[0].folder)
            dir = os.path.dirname(file)
        else:
            dir = None
        results = chisurf.structure.cluster.cluster(
            self.structures, criterion=criterion,
            threshold=threshold, Z=Z, distances=self.clDistances, directory=dir)
        self.clZ, self.clClusters, self.clAssignment, self.clDistances = results
        print(self.clAssignment)
        print(self.clAssignment.shape)
        self.updateClusterTable()

    def updateClusterTable(self):
        print("updateClusterTable")
        # Clean Cluster-Table
        table2 = self.tableWidget_2
        table2.setRowCount(0)
        minM = 1e12
        try:
            for r, clName in enumerate(self.clClusters):
                table2.setRowCount(r + 1)
                # cluster name
                cl = self.clClusters[clName]
                tmp = QtWidgets.QTableWidgetItem()
                tmp.setData(0, int(clName))
                table2.setItem(r, 0, tmp)
                # cluster-size
                tmp = QtWidgets.QTableWidgetItem()
                tmp.setData(0, int(len(cl)))
                table2.setItem(r, 1, tmp)
                if len(self.structures) > 0:
                    # SN of rep
                    idxOfRepresentativeStructure = slib.findRepresentative(self.structures, cl)
                    tmp = QtWidgets.QTableWidgetItem()
                    tmp.setData(0, idxOfRepresentativeStructure)
                    table2.setItem(r, 2, tmp)
                    # rmsd of representative
                    clRmsdRepresentative = float(self.rmsds[idxOfRepresentativeStructure])
                    tmp = QtWidgets.QTableWidgetItem()
                    tmp.setData(0, clRmsdRepresentative)
                    table2.setItem(r, 6, tmp)
                try:
                    # chi2 of rep
                    clChi2Representative = float(self.chi2s[idxOfRepresentativeStructure])
                    tmp = QtWidgets.QTableWidgetItem()
                    tmp.setData(0, clChi2Representative)
                    table2.setItem(r, 3, tmp)

                    # Mean cluster chi2
                    clChi2s = [float(self.chi2s[sn]) for sn in cl]
                    M = float(np.median(np.array(clChi2s)))
                    tmp = QtWidgets.QTableWidgetItem()
                    tmp.setData(0, M)
                    table2.setItem(r, 4, tmp)
                    # SD/stdev of cluster chi2
                    SD = float(np.array(clChi2s).std())
                    tmp = QtWidgets.QTableWidgetItem()
                    tmp.setData(0, SD)
                    table2.setItem(r, 5, tmp)
                except IndexError:
                    # Mean cluster chi2
                    tmp = QtWidgets.QTableWidgetItem('NA')
                    table2.setItem(r, 3, tmp)
                    tmp = QtWidgets.QTableWidgetItem('NA')
                    table2.setItem(r, 4, tmp)
                    tmp = QtWidgets.QTableWidgetItem('NA')
                    table2.setItem(r, 5, tmp)
                table2.resizeRowsToContents()
        except AttributeError:
            print("data not clustered yet.")

    def onClusterTableClicked(self):
        print("onClusterTableClicked")
        row = self.tableWidget_2.currentRow()
        idx = self.tableWidget_2.item(row, 2).data(0).toInt()[0]
        print("Using state %s as reference" % idx)
        self.pymolPlot.reset()
        self.pymolPlot.open_structure(self.structures[idx])

    def onSaveClusterResults(self, directory=None):
        print("onSaveClusterResults")
        if directory is None:
            directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        print("saving cluster results in: %s" % directory)
        np.save(directory + '/' + 'clLinkage.npy', self.clZ)
        pickle.dump(self.clClusters, open(directory + '/' + 'clDict.pkl', 'wb'))
        np.save(directory + '/' + 'clAssignment.npy', self.clAssignment)
        np.save(directory + '/' + 'clDistances.npy', self.clDistances)

    def onLoadClusterResults(self):
        print("onLoadClusterResults")
        directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.clZ = np.load(directory + '/' + 'clLinkage.npy')
        self.clAssignment = np.load(directory + '/' + 'clAssignment.npy')
        self.clClusters = pickle.load(open(directory + '/' + 'clDict.pkl', 'rb'))
        self.clDistances = np.load(directory + '/' + 'clDistances.npy')
        self.updateClusterTable()

    def onLoadLabeling(self):
        print("onLoadLabling")
        self.avPotential.show()
        self.avPotential.activateWindow()

    def onScreeningStart(self):
        chi2s = np.zeros(len(self.structures))
        for r in range(self.tableWidget.rowCount()):
            sn = self.tableWidget.item(r, 0).data(0).toInt()[0]
            filename = self.tableWidget.item(r, 1).text()
            structure = self.structures[sn]
            self.avPotential.structure = structure
            chi2 = self.avPotential.chi2_1
            chi2s[sn] = chi2
            print("%s: %s/%s : %.3f" % (filename, r + 1, len(self.structures), chi2))
        self.chi2r = chi2s
        self.update()

    def getCandidates(self, withWeights=False):
        print("getCandidates")
        # get chi2 and rmsds of selected structure
        chi2s = []
        rmsds = []
        candidates = []
        if self.radioButton_9.isChecked():
            print("using all structures")
            for r in range(self.tableWidget.rowCount()):
                chi2 = self.tableWidget.item(r, 2).data(0).toDouble()[0]
                if chi2 < self.chi2Max:
                    chi2s.append(chi2)
                    rmsds.append(self.tableWidget.item(r, 3).data(0).toDouble()[0])
                    candidates.append(self.tableWidget.item(r, 0).data(0).toInt()[0])
        elif self.radioButton_7.isChecked():
            print("using representatives of cluster.")
            for r in range(self.tableWidget_2.rowCount()):
                chi2 = self.tableWidget_2.item(r, 3).data(0).toDouble()[0]
                if chi2 < self.chi2Max:
                    chi2s.append(chi2)
                    rmsds.append(self.tableWidget_2.item(r, 6).data(0).toDouble()[0])
                    candidates.append(self.tableWidget_2.item(r, 2).data(0).toInt()[0])
        print("selected fit_index/Total fit_index.: %s/%s" % (len(candidates), len(chi2s)))
        weights = self.chi2_weights
        if not withWeights:
            print("Printing weights:")
            if self.radioButton_5.isChecked():
                print("chi2-weighted")
            elif self.radioButton_6.isChecked():
                print("unweighted")
                weights = np.ones(len(chi2s), dtype=np.float32)
        return candidates, chi2s, rmsds, weights

    def onSaveClusterTable(self):
        print("onSaveClusterTable")
        filename = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save cluster table', '.txt'))[0]
        with chisurf.fio.zipped.open_maybe_zipped(filename, 'w') as fp:
            s = "Cl\tsize\tSN\tchi2Rep\trmsd\tchi2M\tchi2SD\tfilename\n"
            fp.write(s)
            clKeys = list(self.clClusters.keys())
            for r in range(self.tableWidget_2.rowCount()):
                cl = self.tableWidget_2.item(r, 0).data(0).toInt()[0]
                clSize = self.tableWidget_2.item(r, 1).data(0).toInt()[0]
                sn = self.tableWidget_2.item(r, 2).data(0).toInt()[0]
                chi2 = self.tableWidget_2.item(r, 3).data(0).toDouble()[0]
                m = self.tableWidget_2.item(r, 4).data(0).toDouble()[0]
                sd = self.tableWidget_2.item(r, 5).data(0).toDouble()[0]
                rmsd = self.tableWidget_2.item(r, 6).data(0).toDouble()[0]
                filename = self.structures[sn].folder
                s = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cl, clSize, r, chi2, rmsd, m, sd, filename)
                fp.write(s)

    def onGetSelectionRepresentative(self):
        print("onGetSelectionRepresentative")
        candidates, chi2, rmsd, weights = self.getCandidates()
        selectedStructures = [self.structures[i] for i in candidates]
        print("calculating average structure of selection")
        average = slib.calcAverage(structures=selectedStructures, weights=weights)
        iMin, representativeStructure = slib.findBest(targetStructure=average, structures=selectedStructures)
        idxiMin = candidates[iMin]
        self.lineEdit_7.setText(str(idxiMin))

    def onLoadTableOnStructures(self):
        print("onLoadTableOnStructures")
        #filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Save cluster table', '', 'CSV data files (*.csv)'))
        filename = chisurf.widgets.get_filename('Save cluster table', 'CSV data files (*.csv)')

        if os.path.isfile(filename):
            self.onClearStructures()
            with chisurf.fio.zipped.open_maybe_zipped(filename, 'r') as fp:
                filenames = []
                lines = fp.readlines()
                for r, l in enumerate(lines[1:]):
                    self.tableWidget.setRowCount(r + 1)
                    s = l.split()
                    #Chi2
                    chi2 = float(s[1])
                    self.chi2s.append(chi2)
                    tmp = QtWidgets.QTableWidgetItem()
                    self.tableWidget.setItem(r, 2, tmp)
                    self.tableWidget.item(r, 2).set_data(0, chi2)
                    # Cluster
                    try:
                        cl = int(s[2])
                        tmp = QtWidgets.QTableWidgetItem()
                        tmp.setData(0, cl)
                    except ValueError:
                        tmp = QtWidgets.QTableWidgetItem('NA')
                    self.tableWidget.setItem(r, 4, tmp)
                    # Filenames
                    filenames.append(s[4])

    def onSelectionChanged(self):
        print("onSelectionChanged")
        print(self.fit.surface)
        self.fit.surface.setChis([self.chi2s])
        self.fit.surface.setParameter(np.vstack([self.chi2s, self.rmsds]))
        self.fit.plots['Chi2-RMSD'].update()



