"""
This module contain a small tool to generate JSON-labeling files
"""

import json
from collections import OrderedDict

from PyQt4 import QtCore, QtGui, uic

import mfm
import mfm.structure


class PDB2Label(QtGui.QWidget):

    name = "PDB2Label"

    def __init__(self):
        QtGui.QWidget.__init__(self)
        uic.loadUi('./mfm/ui/tools/fps_json_edit.ui', self)
        self.connect(self.toolButton_4, QtCore.SIGNAL("clicked()"), self.onLoadReferencePDB)
        self.connect(self.pushButton_2, QtCore.SIGNAL("clicked()"), self.onAddLabel)
        self.connect(self.pushButton_3, QtCore.SIGNAL("clicked()"), self.onAddDistance)
        self.connect(self.actionSave, QtCore.SIGNAL('triggered()'), self.onSaveLabelingFile)
        self.connect(self.actionClear, QtCore.SIGNAL('triggered()'), self.onClear)
        self.connect(self.actionLoad, QtCore.SIGNAL("triggered()"), self.onLoadJSON)
        self.connect(self.actionReadTextEdit, QtCore.SIGNAL("triggered()"), self.onReadTextEdit)

        self.connect(self.comboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.onSimulationTypeChanged)
        self.connect(self.listWidget, QtCore.SIGNAL("itemDoubleClicked(QListWidgetItem*)"),
                     self.onLabelingListDoubleClicked)
        self.connect(self.listWidget_2, QtCore.SIGNAL("itemDoubleClicked(QListWidgetItem*)"),
                     self.onDistanceListDoubleClicked)

        self.atom_select = mfm.widgets.PDBSelector()
        self.verticalLayout_3.addWidget(self.atom_select)
        self.av_properties = mfm.widgets.AVProperties()
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
        filename = mfm.widgets.open_file('Open JSON Labeling-File', 'JSON-Files (*.fps.json)')
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
    def distance_type(self):
        distance_type = str(self.comboBox_3.currentText())
        if distance_type == 'dRDA':
            return 'RDAMean'
        elif distance_type == 'dRDAE':
            return 'RDAMeanE'
        elif distance_type == 'dRmp':
            return 'Rmp'

    @property
    def label1(self):
        return str(self.comboBox_2.currentText())

    @property
    def label2(self):
        return str(self.comboBox_4.currentText())

    @property
    def distance(self):
        return float(self.doubleSpinBox.value())

    @property
    def forster_radius(self):
        return float(self.doubleSpinBox_4.value())

    @property
    def error_pos(self):
        return float(self.doubleSpinBox_2.value())

    @property
    def error_neg(self):
        return float(self.doubleSpinBox_3.value())

    @property
    def pdb_filename(self):
        return str(self.lineEdit.text())

    @pdb_filename.setter
    def pdb_filename(self, v):
        self.lineEdit.setText(str(v))

    @property
    def simulation_type(self):
        return str(self.comboBox.currentText())

    @property
    def position_name(self):
        return str(self.lineEdit_2.text())

    def onLoadReferencePDB(self):
        #self.pdb_filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open PDB-File', '.pdb', 'PDB-Files (*.pdb)'))
        filename = mfm.widgets.open_file('Open PDB-File', 'PDB-Files (*.pdb)')
        self.pdb_filename = filename
        self.structure = mfm.structure.Structure(self.pdb_filename)
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
            print "Add Label"
            self.positions[self.position_name] = label
            self.onUpdateInterface()
            self.onUpdateJSON()

    def onAddDistance(self):
        distance = {
            "Forster_radius": self.forster_radius,
            "distance": self.distance,
            "distance_type": self.distance_type,
            "error_neg": self.error_neg,
            "error_pos": self.error_pos,
            "position1_name": self.label1,
            "position2_name": self.label2
        }
        distance_name = self.label1+'_'+self.label2
        self.distances[distance_name] = distance
        self.onUpdateInterface()
        self.onUpdateJSON()

    def onSaveLabelingFile(self, json_file=None):
        self.json_file = json_file if json_file is not None else self.json_file
        if self.json_file is None:
            #self.json_file = str(QtGui.QFileDialog.getOpenFileName(self, 'Open FPS-JSON File',
            #                                                          '.fps.json', 'JSON-Files (*.fps.json)'))
            filename = mfm.widgets.open_file('Open FPS-JSON File', 'JSON-Files (*.fps.json)')
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
