import glob
import os

import mdtraj as md
from mdtraj.scripts import mdconvert
from qtpy import QtCore, QtGui, QtWidgets


import chisurf.core.decorators
import chisurf.gui.decorators
import chisurf.gui.widgets

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c



@persist_plugin_state("traj_convert")
class MDConverter(
    QtWidgets.QWidget
):

    name = "MC-Converter"

    @chisurf.gui.decorators.init_with_ui(
        ui_filename="convert_structures.ui"
    )
    def __init__(
            self,
            parent=None,
            *args,
            **kwargs
    ):
        self.toolButton.clicked.connect(self.onLoadHDFFile)
        self.toolButton_2.clicked.connect(self.onSelecteTargetDir)
        self.pushButton_3.clicked.connect(self.onConvert)
        self.actionOpen_Topology.triggered.connect(self.onOpenTopology)
        self.verbose = kwargs.get('verbose', chisurf.core.settings.cs_settings['verbose'])
        self._setup_layout()

    def _empty_icon(self):
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.transparent)
        return QtGui.QIcon(pixmap)

    def _setup_layout(self) -> None:
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.setMaximumSize(16777215, 16777215)
        self.groupBox.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum
        )
        self.groupBox_2.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum
        )
        self._log = QtWidgets.QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Log")
        self._log.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.verticalLayout.addWidget(self._log, 1)
        self.pushButton_3.setIcon(self._empty_icon())
        self.toolButton.setIcon(self._empty_icon())
        self.toolButton_2.setIcon(self._empty_icon())
        self.toolButton_3.setIcon(self._empty_icon())

    def _append_log(self, message: str) -> None:
        timestamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self._log.appendPlainText(f"[{timestamp}] {message}")

    def onOpenTopology(self):
        #self.topology_file = str(QtGui.QFileDialog.getOpenFileName(self, 'Open PDB-File', '.', 'PDB-File (*.pdb)'))
        filename = chisurf.gui.widgets.get_filename('Open PDB-File', 'PDB-File (*.pdb)')
        self.topology_file = filename

    def onLoadHDFFile(self):
        if not self.use_folder:
            #self.trajectory = str(QtGui.QFileDialog.getOpenFileName(self, 'Open HDF-File', '.', 'H5-File (*.h5)'))
            filename = chisurf.gui.widgets.get_filename('Open HDF-File', 'H5-File (*.h5)')
            self.trajectory = filename
        else:
            self.trajectory = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open PDB-Files', '.'))

    def onSelecteTargetDir(self):
        self.target_directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Target-Folder', '.'))

    def onConvert(self, **kwargs):
        verbose = kwargs.get('verbose', self.verbose)
        if verbose:
            print("Converting trajectory")
        self._append_log("Starting trajectory conversion")
        args = Object()
        args.topology = self.topology_file
        args.input = [self.trajectory] if not self.use_folder else glob.glob(os.path.join(self.trajectory, '*.pdb'))
        args.index = None
        args.chunk = 1000
        args.stride = self.stride

        if self.first_frame != 0 or self.last_frame != -1:
            args.index = slice(self.first_frame, self.last_frame, self.stride)
            args.stride = None
            args.chunk = None

        args.force = True
        args.atom_indices = None
        self._append_log(f"Input frames: {self.first_frame}:{self.last_frame} stride={self.stride}")

        if self.split:
            i = 0
            for chunk in md.iterload(
                    self.trajectory,
                    chunk=args.chunk,
                    top=self.topology_file
            ):
                for s in chunk:
                    try:
                        fn = os.path.join(
                            self.target_directory,
                            self.filename + '_%0*d' % (8, i) + self.ending
                        )
                        if verbose:
                            print(fn)
                        self._append_log(f"Saving frame {i}: {fn}")
                        s.save(fn)
                    except Exception as exc:
                        self._append_log(f"Frame {i} failed: {exc}")
                    i += 1
        else:
            args.output = os.path.join(
                self.target_directory,
                self.filename + self.ending
            )
            self._append_log(f"Output: {args.output}")
            mdconvert.main(args)
        chisurf.gui.widgets.general.MyMessageBox('Conversion done!')
        self._append_log("Conversion done")

    @property
    def first_frame(self):
        return int(self.spinBox_2.value())

    @property
    def last_frame(self):
        return int(self.spinBox_3.value())

    @property
    def stride(self):
        return int(self.spinBox.value())

    @property
    def topology_file(self):
        s = str(self.lineEdit_4.text())
        if os.path.isfile(s):
            return s
        else:
            return None

    @topology_file.setter
    def topology_file(self, v):
        return self.lineEdit_4.setText(v)

    @property
    def ending(self):
        return str(self.comboBox.currentText())

    @property
    def split(self):
        return bool(self.checkBox.isChecked())

    @property
    def filename(self):
        return str(self.lineEdit_3.text())

    @property
    def trajectory(self):
        return str(self.lineEdit.text())

    @trajectory.setter
    def trajectory(self, v):
        self.lineEdit.setText(str(v))

    @property
    def target_directory(self):
        return str(self.lineEdit_2.text())

    @target_directory.setter
    def target_directory(self, v):
        self.lineEdit_2.setText(str(v))

    @property
    def use_folder(self):
        return bool(self.radioButton.isChecked())


class Object(object):
    # This is only used to pass the arguments
    pass
