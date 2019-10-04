import os

import numpy as np
from qtpy import  QtWidgets, uic, QtCore

import mfm
import mfm.io
from mfm.structure.structure import Structure
from mfm.structure.trajectory import TrajectoryFile


class PDBSelector(
    QtWidgets.QWidget
):
    """

    """

    def __init__(
            self,
            show_labels: bool = True,
            update=None
    ):
        """

        :param show_labels:
        :param update:
        """
        super(PDBSelector, self).__init__()
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "pdb_widget.ui"
            ),
            self
        )
        self._pdb = None
        self.comboBox.currentIndexChanged[int].connect(self.onChainChanged)
        self.comboBox_2.currentIndexChanged[int].connect(self.onResidueChanged)
        if not show_labels:
            self.label.hide()
            self.label_2.hide()
            self.label_3.hide()
        if update is not None:
            self.comboBox_2.currentIndexChanged[int].connect(update)
            self.comboBox_2.currentIndexChanged[int].connect(update)

    @property
    def atoms(self):
        return self._pdb

    @atoms.setter
    def atoms(
            self,
            v
    ):
        self._pdb = v
        self.update_chain()

    @property
    def chain_id(
            self
    ) -> str:
        return str(self.comboBox.currentText())

    @chain_id.setter
    def chain_id(
            self,
            v: str
    ):
        pass

    @property
    def residue_name(
            self
    ) -> str:
        try:
            return str(
                self.atoms[self.atom_number]['res_name']
            )
        except ValueError:
            return "NA"

    @property
    def residue_id(
            self
    ) -> int:
        try:
            return int(self.comboBox_2.currentText())
        except ValueError:
            return 0

    @residue_id.setter
    def residue_id(
            self,
            v: int
    ):
        pass

    @property
    def atom_name(
            self
    ) -> str:
        return str(self.comboBox_3.currentText())

    @atom_name.setter
    def atom_name(
            self,
            v: str
    ):
        pass

    @property
    def atom_number(
            self
    ) -> int:
        residue_key = self.residue_id
        atom_name = self.atom_name
        chain = self.chain_id

        w = mfm.io.coordinates.get_atom_index(
            self.atoms,
            chain,
            residue_key,
            atom_name,
            None
        )
        return w

    def onChainChanged(self):
        print("PDBSelector:onChainChanged")
        self.comboBox_2.clear()
        pdb = self._pdb
        chain = str(self.comboBox.currentText())
        atom_ids = np.where(pdb['chain'] == chain)[0]
        residue_ids = list(set(self.atoms['res_id'][atom_ids]))
        residue_ids_str = [str(x) for x in residue_ids]
        self.comboBox_2.addItems(residue_ids_str)

    def onResidueChanged(self):
        self.comboBox_3.clear()
        pdb = self.atoms
        chain = self.chain_id
        residue = self.residue_id
        print("onResidueChanged: %s" % residue)
        atom_ids = np.where((pdb['res_id'] == residue) & (pdb['chain'] == chain))[0]
        atom_names = [atom['atom_name'] for atom in pdb[atom_ids]]
        self.comboBox_3.addItems(atom_names)

    def update_chain(self):
        self.comboBox.clear()
        chain_ids = list(set(self.atoms['chain'][:]))
        self.comboBox.addItems(chain_ids)


class LoadThread(QtCore.QThread):

    #procDone = pyqtSignal(bool)
    #partDone = pyqtSignal(int)

    def run(self):
        nFiles = len(self.filenames)
        print('File loading started')
        print('#Files: %s' % nFiles)
        for i, fn in enumerate(self.filenames):
            f = self.read(fn, *self.read_parameter)
            self.target.append(f, *self.append_parameter)
            self.partDone.emit(float(i + 1) / nFiles * 100)
        #self.procDone.emit(True)
        print('reading finished')


class PDBFolderLoad(
    QtWidgets.QWidget
):
    """

    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        """

        :param parent:
        """
        super(PDBFolderLoad, self).__init__(
            *args,
            **kwargs
        )
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "proteinFolderLoad.ui"
            ),
            self
        )

        self.pushButton_12.clicked.connect(self.onLoadStructure)
        self.updatePBar(0)
        self.load_thread = LoadThread()
        self.load_thread.partDone.connect(self.updatePBar)
        self.load_thread.procDone.connect(self.fin)
        self.trajectory = TrajectoryFile()

    def fin(self):
        print("Loading of structures finished")
        self.lineEdit.setText(str(self.nAtoms))
        self.lineEdit_2.setText(str(self.nResidues))

    def updatePBar(self, val):
        self.progressBar.setValue(val)

    def onLoadStructure(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))

        self.folder = directory
        filenames = [os.path.join(directory, f) for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f))]
        filenames.sort()

        pdb_filenames = list()
        for filename in filenames:
            extension = os.path.splitext(filename)[1][1:]
            if filename.lower().endswith('.pdb') or extension.isdigit():
                pdb_filenames.append(filename)

        self.n_files = len(pdb_filenames)

        self.load_thread.read = Structure
        self.load_thread.read_parameter = [self.calc_internal, self.verbose]
        self.load_thread.append_parameter = [self.calc_rmsd]
        self.trajectory = TrajectoryFile(
            use_objects=self.use_objects,
            calc_internal=self.calc_internal,
            verbose=self.verbose
        )
        self.load_thread.filenames = pdb_filenames
        self.load_thread.target = self.trajectory
        self.load_thread.start()

    @property
    def calc_rmsd(
            self
    ) -> bool:
        return self.checkBox_4.isChecked()

    @property
    def n_files(
            self
    ) -> int:
        return int(self.lineEdit_3.text())

    @n_files.setter
    def n_files(
            self,
            v: int
    ):
        self.lineEdit_3.setText(str(v))

    @property
    def verbose(
            self
    ) -> bool:
        return bool(self.checkBox_3.isChecked())

    @property
    def nAtoms(
            self
    ) -> int:
        return self.trajectory[0].n_atoms

    @property
    def nResidues(
            self
    ) -> int:
        return self.trajectory[0].n_residues

    @property
    def folder(
            self
    ) -> str:
        return str(self.lineEdit_7.text())

    @folder.setter
    def folder(
            self,
            v: str
    ):
        self.lineEdit_7.setText(v)

    @property
    def use_objects(
            self
    ) -> bool:
        return bool(self.checkBox_2.isChecked())

    @property
    def calc_internal(
            self
    ) -> bool:
        return self.checkBox.isChecked()
