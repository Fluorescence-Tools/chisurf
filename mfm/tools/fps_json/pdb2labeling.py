"""
This module contain a small tool to generate JSON-labeling files
"""
from __future__ import annotations

import os
import json
from collections import OrderedDict

from PyQt5 import QtWidgets, uic

import mfm
import mfm.structure
import mfm.structure.structure


class PDB2Label(QtWidgets.QWidget):

    name = "PDB2Label"

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "fps_json_edit.ui"
            ),
            self
        )
        self.toolButton_4.clicked.connect(self.onLoadReferencePDB)
        self.pushButton_2.clicked.connect(self.onAddLabel)
        self.pushButton_3.clicked.connect(self.onAddDistance)
        self.actionSave.triggered.connect(self.onSaveLabelingFile)
        self.actionClear.triggered.connect(self.onClear)
        self.actionLoad.triggered.connect(self.onLoadJSON)
        self.actionReadTextEdit.triggered.connect(self.onReadTextEdit)

        self.comboBox.currentIndexChanged[int].connect(self.onSimulationTypeChanged)
        #self.listWidget.itemDoubleClicked[QListWidgetItem].connect(self.onLabelingListDoubleClicked)
        #self.listWidget_2.itemDoubleClicked[QListWidgetItem].connect(self.onDistanceListDoubleClicked)

        self.atom_select = mfm.widgets.pdb.PDBSelector()
        self.verticalLayout_3.addWidget(self.atom_select)
        self.av_properties = mfm.widgets.accessible_volume.AVProperties()
        self.verticalLayout_4.addWidget(self.av_properties)

        self.structure = None
        self.json_file = None
        self.positions = OrderedDict()
        self.distances = OrderedDict()

    def onReadTextEdit(self):
        s = str(self.textEdit_2.toPlainText())
        p = json.loads(s)
        self.distances = p["Distances"]
        self.positions = p["Positions"]
        self.onUpdateJSON()
        self.onUpdateInterface()

    def onLoadJSON(self):
        #self.json_file = str(QtGui.QFileDialog.getOpenFileName(self, 'Open JSON Labeling-File',
        #                                                       '.', 'JSON-Files (*.fps.json)'))
        filename = mfm.widgets.get_filename('Open JSON Labeling-File', 'JSON-Files (*.fps.json)')
        self.json_file = filename
        p = json.load(open(self.json_file, 'r'))
        self.distances = p["Distances"]
        self.positions = p["Positions"]
        self.onUpdateJSON()
        self.onUpdateInterface()

    def onLabelingListDoubleClicked(self, item):
        label_name = str(item.text())
        del self.positions[label_name]
        for dist in list(self.distances.values()):
            if dist['position1_name'] == label_name or dist['position1_name'] == label_name:
                del dist
        self.onUpdateInterface()
        self.onUpdateJSON()

    def onDistanceListDoubleClicked(self, item):
        distance_name = str(item.text())
        del self.distances[distance_name]
        self.onUpdateInterface()
        self.onUpdateJSON()

    def onSimulationTypeChanged(self):
        self.av_properties.av_type = self.simulation_type

    def onUpdateInterface(self):
        self.comboBox_2.clear()
        self.comboBox_4.clear()
        self.listWidget.clear()
        self.listWidget_2.clear()
        label_names = [label for label in list(self.positions.keys())]
        distance_names = [label for label in list(self.distances.keys())]
        self.listWidget.addItems(label_names)
        self.listWidget_2.addItems(distance_names)
        self.comboBox_2.addItems(label_names)
        self.comboBox_4.addItems(label_names)

    @property
    def distance_type(self) -> str:
        distance_type = str(self.comboBox_3.currentText())
        if distance_type == 'dRDA':
            return 'RDAMean'
        elif distance_type == 'dRDAE':
            return 'RDAMeanE'
        elif distance_type == 'dRmp':
            return 'Rmp'
        elif distance_type == 'pRDA':
            return 'pRDA'

    @property
    def label1(self) -> str:
        return str(self.comboBox_2.currentText())

    @property
    def label2(self) -> str:
        return str(self.comboBox_4.currentText())

    @property
    def distance(self) -> float:
        return float(self.doubleSpinBox.value())

    @property
    def forster_radius(self) -> float:
        return float(self.doubleSpinBox_4.value())

    @property
    def error_pos(self) -> float:
        return float(self.doubleSpinBox_2.value())

    @property
    def error_neg(self) -> float:
        return float(self.doubleSpinBox_3.value())

    @property
    def pdb_filename(self) -> str:
        return str(self.lineEdit.text())

    @pdb_filename.setter
    def pdb_filename(
            self,
            v: str
    ):
        self.lineEdit.setText(str(v))

    @property
    def simulation_type(self) -> str:
        return str(self.comboBox.currentText())

    @property
    def position_name(self) -> str:
        return str(self.lineEdit_2.text())

    def onLoadReferencePDB(self):
        #self.pdb_filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open PDB-File', '.pdb', 'PDB-Files (*.pdb)'))
        filename = mfm.widgets.get_filename('Open PDB-File', 'PDB-Files (*.pdb)')
        self.pdb_filename = filename
        self.structure = mfm.structure.structure.Structure(self.pdb_filename)
        self.atom_select.atoms = self.structure.atoms

    def onAddLabel(self):
        label = {
            "atom_name": str(self.atom_select.atom_name),
            "chain_identifier": str(self.atom_select.chain_id),
            "residue_seq_number": int(self.atom_select.residue_id),
            "residue_name": str(self.atom_select.residue_name),
            "attachment_atom_index": int(self.atom_select.atom_number),
            "simulation_type": str(self.simulation_type),
            "linker_length": float(self.av_properties.linker_length),
            "linker_width": float(self.av_properties.linker_width),
            "radius1": float(self.av_properties.radius_1),
            "radius2": float(self.av_properties.radius_2),
            "radius3": float(self.av_properties.radius_3),
            "simulation_grid_resolution": float(self.av_properties.resolution)
        }
        if self.position_name != '' and self.position_name not in list(self.positions.keys()):
            self.positions[self.position_name] = label
            self.onUpdateInterface()
            self.onUpdateJSON()

    def onAddDistance(self):
        distance = {
            "Forster_radius": self.forster_radius,
            "distance_type": self.distance_type,
            "position1_name": self.label1,
            "position2_name": self.label2,
        }

        if self.distance_type == "pRDA":
            fn = mfm.widgets.get_filename("DA-Distance distribution (1st column RDA, 2nd pRDA)")
            csv = mfm.io.ascii.Csv(filename=fn, skiprows=1)
            distance['rda'] = list(csv.data_x)
            distance['prda'] = list(csv.data_y)
        else:
            distance['distance'] = self.distance
            distance['error_neg'] = self.error_neg
            distance['error_pos'] = self.error_pos

        distance_name = self.label1+'_'+self.label2
        self.distances[distance_name] = distance
        self.onUpdateInterface()
        self.onUpdateJSON()

    def onSaveLabelingFile(
            self,
            json_file: str = None
    ):
        self.json_file = json_file if json_file is not None else self.json_file
        if self.json_file is None:
            #self.json_file = str(QtGui.QFileDialog.getOpenFileName(self, 'Open FPS-JSON File',
            #                                                          '.fps.json', 'JSON-Files (*.fps.json)'))
            filename = mfm.widgets.get_filename('Open FPS-JSON File', 'JSON-Files (*.fps.json)')
            self.json_file = filename

        p = json.load(open(self.json_file), sort_keys=True, indent=4, separators=(',', ': '))
        self.distances = p["Distances"]
        self.positions = p["Positions"]
        self.onUpdateJSON()
        self.onUpdateInterface()

    def onUpdateJSON(self):
        p = OrderedDict()
        p["Distances"] = self.distances
        p["Positions"] = self.positions
        s = json.dumps(p, sort_keys=True, indent=4, separators=(',', ': '))
        self.textEdit_2.clear()
        self.textEdit_2.setText(s)

    def onClear(self):
        self.positions = OrderedDict()
        self.distances = OrderedDict()
        self.onUpdateInterface()
        self.onUpdateJSON()
