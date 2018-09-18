"""Qt-Widgets used throughout ChiSURF

"""
import os
import fnmatch
import pickle
import random
from datetime import datetime
from PyQt4.QtCore import pyqtSignal
from PyQt4 import QtGui, QtCore, uic

os.environ['PYZMQ_BACKEND'] = 'cython'
from qtconsole.qtconsoleapp import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

from IPython.lib import guisupport
import numpy as np
from mfm.structure import Structure
from mfm.structure.trajectory import TrajectoryFile
import mfm

DEFAULT_INSTANCE_ARGS = ['qtconsole','--pylab=inline', '--colors=linux']


class QIPythonWidget(RichJupyterWidget):

    def start_recording(self):
        self._macro = ""
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def run_macro(self, filename=None):
        if filename is None:
            filename = mfm.widgets.open_file("Python macro", file_type="Python file (*.py)")
        with open(filename, mode='r') as fp:
            text = fp.read()
            self.execute(text, hidden=True)

    def save_macro(self, filename=None):
        self.stop_recording()
        if filename is None:
            filename = mfm.widgets.save_file("Python macro", file_type="Python file (*.py)")
        with open(filename, mode='w') as fp:
            fp.write(self._macro)

    def execute(self, *args, **kwargs):
        hidden = kwargs.get('hidden', False)
        if not hidden:
            try:
                new_text = args[0] + '\n'
                with open(self.session_file, 'a+') as fp:
                    fp.write(new_text)
                if self.recording:
                    self._macro += new_text
            except IndexError:
                pass
        RichJupyterWidget.execute(self, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        RichJupyterWidget.__init__(self, *args, **kwargs)

        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel(show_banner=False)
        kernel_manager.kernel.gui = 'qt4'
        self.kernel_client = kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            guisupport.get_app_qt4().exit()

        self.exit_requested.connect(stop)
        self.width = kwargs.get('width', mfm.settings['gui']['console']['width'])
        self._macro = ""
        self.recording = False

        # save nevertheless every inputs into a session file
        filename = datetime.now().strftime('session_%H_%M_%d_%m_%Y.py')
        home = os.path.expanduser("~")
        path = os.path.abspath(os.path.join(home, './chisurf'))
        try:
            os.makedirs(path)
        except:
            print "no new path created"
        self.session_file = os.path.join(path, filename)

    def pushVariables(self, variableDict):
        """ Given a dictionary containing name / value pairs, push those variables to the IPython console widget """
        self.kernel_manager.kernel.shell.push(variableDict)

    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()

    def printText(self, text):
        """ Prints some plain text to the console """
        self._append_plain_text(text)

    def executeCommand(self, command):
        """ Execute a command in the frame of the console widget """
        self._execute(command, mfm.settings['show_commands'])


def widgets_in_layout(layout):
    """Return the widgets within a layout

    :param layout:
    :return:
    """
    return (layout.itemAt(i) for i in range(layout.count()))


def clear_layout(layout):
    """Used to clear all widgets in a layout
    :param layout:
    :return:
    """
    if layout != None:
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                clear_layout(child.layout())


def hide_items_in_layout(layout):
    """This function hides all items within a Qt-layout

    :param layout: Qt-Layout
    """
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if type(item) == QtGui.QWidgetItem:
            item.widget().hide()


def get_fortune(fortunepath='./mfm/ui/fortune/', min_length=0, max_length=100, attempts=1000, **kwargs):
    fortune_files = [os.path.splitext(pdat)[0] for pdat in os.listdir(fortunepath) if pdat.endswith(".pdat")]
    attempt = 0
    while True:
        fortune_file = os.path.join(fortunepath, random.choice(fortune_files))
        data = pickle.load(open(fortune_file+".pdat", "rb"))
        (start, length) = random.choice(data)
        print(random.choice(data))
        if length < min_length or (max_length is not None and length > max_length):
            attempt += 1
            if attempt > attempts:
                return ""
            continue
        ffh = open(fortune_file, 'rU')
        ffh.seek(start)
        fortunecookie = ffh.read(length)
        ffh.close()
        return fortunecookie


class AVProperties(QtGui.QWidget):

    def __init__(self, av_type="AV1"):
        QtGui.QWidget.__init__(self)
        uic.loadUi('./mfm/ui/av_property.ui', self)
        self._av_type = av_type
        self.av_type = av_type
        self.groupBox.hide()

    @property
    def av_type(self):
        return self._av_type

    @av_type.setter
    def av_type(self, v):
        self._av_type = v
        if v == 'AV1':
            self.label_4.setEnabled(False)
            self.label_5.setEnabled(False)
            self.doubleSpinBox_4.setEnabled(False)
            self.doubleSpinBox_5.setEnabled(False)
        if v == 'AV0':
            self.doubleSpinBox_4.setEnabled(False)
            self.doubleSpinBox_5.setEnabled(False)
        elif v == 'AV3':
            self.label_4.setEnabled(True)
            self.label_5.setEnabled(True)
            self.doubleSpinBox_4.setEnabled(True)
            self.doubleSpinBox_5.setEnabled(True)

    @property
    def linker_length(self):
        return float(self.doubleSpinBox.value())

    @linker_length.setter
    def linker_length(self, v):
        self.doubleSpinBox.setValue(v)

    @property
    def linker_width(self):
        return float(self.doubleSpinBox_2.value())

    @linker_width.setter
    def linker_width(self, v):
        self.doubleSpinBox_2.setValue(v)

    @property
    def radius_1(self):
        return float(self.doubleSpinBox_3.value())

    @radius_1.setter
    def radius_1(self, v):
        self.doubleSpinBox_3.setValue(v)

    @property
    def radius_2(self):
        return float(self.doubleSpinBox_4.value())

    @radius_2.setter
    def radius_2(self, v):
        self.doubleSpinBox_4.setValue(v)

    @property
    def radius_3(self):
        return float(self.doubleSpinBox_5.value())

    @radius_3.setter
    def radius_3(self, v):
        self.doubleSpinBox_5.setValue(v)

    @property
    def resolution(self):
        return float(self.doubleSpinBox_6.value())

    @resolution.setter
    def resolution(self, v):
        self.doubleSpinBox_6.setValue(v)

    @property
    def initial_linker_sphere(self):
        return float(self.doubleSpinBox_7.value())

    @initial_linker_sphere.setter
    def initial_linker_sphere(self, v):
        self.doubleSpinBox_7.setValue(v)

    @property
    def initial_linker_sphere_min(self):
        return float(self.doubleSpinBox_8.value())

    @initial_linker_sphere_min.setter
    def initial_linker_sphere_min(self, v):
        self.doubleSpinBox_8.setValue(v)

    @property
    def initial_linker_sphere_max(self):
        return float(self.doubleSpinBox_9.value())

    @initial_linker_sphere_max.setter
    def initial_linker_sphere_max(self, v):
        self.doubleSpinBox_9.setValue(v)


class PDBSelector(QtGui.QWidget):

    def __init__(self, show_labels=True, update=None):
        QtGui.QWidget.__init__(self)
        uic.loadUi('./mfm/ui/pdb_widget.ui', self)
        self._pdb = None
        self.connect(self.comboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.onChainChanged)
        self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), self.onResidueChanged)
        if not show_labels:
            self.label.hide()
            self.label_2.hide()
            self.label_3.hide()
        if update is not None:
            self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), update)
            self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), update)

    @property
    def atoms(self):
        return self._pdb

    @atoms.setter
    def atoms(self, v):
        self._pdb = v
        self.update_chain()

    @property
    def chain_id(self):
        return str(self.comboBox.currentText())

    @chain_id.setter
    def chain_id(self, v):
        pass

    @property
    def residue_name(self):
        try:
            return str(self.atoms[self.atom_number]['res_name'])
        except ValueError:
            return 0

    @property
    def residue_id(self):
        try:
            return int(self.comboBox_2.currentText())
        except ValueError:
            return 0

    @residue_id.setter
    def residue_id(self, v):
        pass

    @property
    def atom_name(self):
        return str(self.comboBox_3.currentText())

    @atom_name.setter
    def atom_name(self, v):
        pass

    @property
    def atom_number(self):
        residue_key = self.residue_id
        atom_name = self.atom_name
        chain = self.chain_id

        w = mfm.io.pdb.get_atom_index(self.atoms,
                                  chain,
                                  residue_key,
                                  atom_name,
                                  None)
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


class MyMessageBox(QtGui.QMessageBox):

    def __init__(self, label=None, info=None):
        QtGui.QMessageBox.__init__(self)
        self.Icon = 1
        self.setSizeGripEnabled(True)
        self.setIcon(QtGui.QMessageBox.Information)
        if label is not None:
            self.setWindowTitle(label)
        if info is not None:
            self.setDetailedText(info)
        if mfm.settings['fortune']:
            fortune = get_fortune(**mfm.settings['fortune'])
            self.setInformativeText(fortune)
            self.exec_()
            self.setMinimumWidth(450)
            self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        else:
            self.close()

    def event(self, e):
        result = QtGui.QMessageBox.event(self, e)

        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        self.setMaximumWidth(16777215)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        textEdit = self.findChild(QtGui.QTextEdit)
        if textEdit != None :
            textEdit.setMinimumHeight(0)
            textEdit.setMaximumHeight(16777215)
            textEdit.setMinimumWidth(0)
            textEdit.setMaximumWidth(16777215)
            textEdit.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        return result


class FileList(QtGui.QListWidget):

    @property
    def filenames(self):
        fn = list()
        for row in range(self.count()):
            item = self.item(row)
            fn.append(str(item.text()))
        return fn

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super(FileList, self).dragEnterEvent(event)

    def dragMoveEvent(self, event):
        super(FileList, self).dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                s = str(url.toLocalFile())
                url.setScheme("")
                if s.endswith(self.filename_ending) or \
                        s.endswith(self.filename_ending+'.gz') or \
                        s.endswith(self.filename_ending+'.bz2') or \
                        s.endswith(self.filename_ending+'.zip'):
                    self.addItem(s)
            event.acceptProposedAction()
        else:
            super(FileList, self).dropEvent(event)

    def __init__(self, **kwargs):
        QtGui.QListWidget.__init__(self)
        accept_drops = kwargs.get('drag_enabled', True)
        self.filename_ending = kwargs.get('filename_ending', '*')
        if accept_drops:
            self.setAcceptDrops(True)
            self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)

        self.drag_item = None
        self.drag_row = None
        self.setWindowIcon(QtGui.QIcon(":/icons/icons/list-add.png"))


class CurveSelector(QtGui.QTreeWidget):

    @property
    def curve_name(self):
        try:
            return self.selected_dataset.name
        except AttributeError:
            return "Noname"

    @property
    def datasets(self):
        data_curves = self.get_data_curves(curve_type=self.curve_type)
        if self.setup is not None:
            return [d for d in data_curves if isinstance(d.setup, self.setup)]
        else:
            return data_curves

    @property
    def selected_curve_index(self):
        if self.currentIndex().parent().isValid():
            return self.currentIndex().parent().row()
        else:
            return self.currentIndex().row()

    @selected_curve_index.setter
    def selected_curve_index(self, v):
        self.setCurrentItem(self.topLevelItem(v))

    @property
    def selected_dataset(self):
        return self.datasets[self.selected_curve_index]

    @property
    def selected_datasets(self):
        data_sets_idx = [r.row() for r in self.selectedIndexes()]
        return [self.datasets[i] for i in data_sets_idx]

    def onCurveChanged(self):
        if self.click_close:
            self.hide()
        self.change_event()

    def onChangeCurveName(self):
        # select current curve and change its name
        pass

    def onRemoveDataset(self):
        dataset_idx = [selected_index.row() for selected_index in self.selectedIndexes()]
        mfm.console.execute('cs.remove_dataset(%s)' % dataset_idx)
        self.update()

    def onSaveDataset(self):
        filename = mfm.widgets.save_file(file_type="*.csv")
        self.selected_dataset.save(filename)

    def onGroupDatasets(self):
        datasets = self.selected_datasets
        if isinstance(datasets[0], mfm.curve.DataCurve):
            # TODO: check for double names!!!
            dg = mfm.curve.ExperimentDataCurveGroup(datasets, name="Unnamed")
        else:
            dg = mfm.curve.ExperimentDataGroup(datasets, name="Unnamed")
        dn = list()
        for i, d in enumerate(mfm.data_sets):
            if d not in dg:
                dn.append(d)
        dn.append(dg)
        mfm.data_sets = dn
        self.update()

    def onUnGroupDatasets(self):
        dg = mfm.curve.ExperimentDataGroup(self.selected_datasets)[0]
        dn = list()
        for i, d in enumerate(mfm.data_sets):
            if d is not dg:
                dn.append(d)
            else:
                dn += dg
        mfm.data_sets = dn
        self.update()

    def contextMenuEvent(self, event):
        menu = QtGui.QMenu(self)
        menu.setTitle("Datasets")
        menu.addAction("Save").triggered.connect(self.onSaveDataset)
        menu.addAction("Remove").triggered.connect(self.onRemoveDataset)
        menu.addAction("Group").triggered.connect(self.onGroupDatasets)
        menu.addAction("Ungroup").triggered.connect(self.onUnGroupDatasets)
        menu.exec_(event.globalPos())

    def update(self, *args, **kwargs):
        QtGui.QTreeWidget.update(self, *args, **kwargs)
        try:
            window_title = self.fit.name
            self.setWindowTitle(window_title)
        except AttributeError:
            self.setWindowTitle("")

        self.clear()

        for d in self.datasets:
            fn = d.name
            item = QtGui.QTreeWidgetItem(self, [fn])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            if isinstance(d, mfm.curve.ExperimentDataGroup):
                for di in d:
                    i2 = QtGui.QTreeWidgetItem(item, [di.name])

    def dragMoveEvent(self, event):
        super(CurveSelector, self).dragMoveEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super(CurveSelector, self).dragEnterEvent(event)

    def startDrag(self, supportedActions):
        #self.drag_item = self.currentItem()
        #self.drag_row = self.row(self.drag_item)
        super(CurveSelector, self).startDrag(supportedActions)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [str(url.toLocalFile()) for url in event.mimeData().urls()]
            paths.sort()
            for path in paths:
                s = "cs.add_dataset(filename='%s')" % path
                mfm.run(s)
            event.acceptProposedAction()
        else:
            super(CurveSelector, self).dropEvent(event)
        self.update()

    def onItemChanged(self):
        if self.selected_datasets:
            ds = self.selected_datasets[0]
            ds.name = str(self.currentItem().text(0))

    def change_event(self):
        pass

    def show(self):
        self.update()
        QtGui.QTreeWidget.show(self)

    def __init__(self, **kwargs):
        QtGui.QTreeWidget.__init__(self)
        self.setWindowIcon(QtGui.QIcon(":/icons/icons/list-add.png"))
        #self.setWordWrap(True)
        self.setHeaderHidden(True)
        #self.setColumnCount(1)
        self.setAlternatingRowColors(True)

        if kwargs.get('drag_enabled', False):
            self.setAcceptDrops(True)
            self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
            #self.setDragDropMode(QtGui.QAbstractItemView.DropOnly)

        # http://python.6.x6.nabble.com/Drag-and-drop-editing-in-QListWidget-or-QListView-td1792540.html
        self.drag_item = None
        self.drag_row = None
        self.get_data_curves = kwargs.get('get_data_curves', mfm.get_data)
        self.change_event = kwargs.get('change_event', self.change_event)
        self.curve_type = kwargs.get('curve_types', 'experiment')
        self.click_close = kwargs.get('click_close', True)
        self.setup = kwargs.get('setup', None)
        self.fit = kwargs.get('fit', None)
        self.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)

        self.clicked.connect(self.onCurveChanged)
        self.itemChanged.connect(self.onItemChanged)


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


class PDBFolderLoad(QtGui.QWidget):

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        uic.loadUi("mfm/ui/proteinFolderLoad.ui", self)
        self.connect(self.pushButton_12, QtCore.SIGNAL("clicked()"), self.onLoadStructure)
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
        directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))

        self.folder = directory
        filenames = [os.path.join(directory, f) for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f))]
        filenames.sort()

        pdb_filenames = []
        for i, filename in enumerate(filenames):
            extension = os.path.splitext(filename)[1][1:]
            if filename.lower().endswith('.pdb') or extension.isdigit():
                pdb_filenames.append(filename)

        self.n_files = len(pdb_filenames)

        self.load_thread.read = Structure
        self.load_thread.read_parameter = [self.calc_internal, self.verbose]
        self.load_thread.append_parameter = [self.calc_rmsd]
        self.trajectory = TrajectoryFile(use_objects=self.use_objects, calc_internal=self.calc_internal,
                                     verbose=self.verbose)
        self.load_thread.filenames = pdb_filenames
        self.load_thread.target = self.trajectory
        self.load_thread.start()

    @property
    def calc_rmsd(self):
        return self.checkBox_4.isChecked()

    @property
    def n_files(self):
        return int(self.lineEdit_3.text())

    @n_files.setter
    def n_files(self, v):
        self.lineEdit_3.setText(str(v))

    @property
    def verbose(self):
        return bool(self.checkBox_3.isChecked())

    @property
    def nAtoms(self):
        return self.trajectory[0].n_atoms

    @property
    def nResidues(self):
        return self.trajectory[0].n_residues

    @property
    def folder(self):
        return str(self.lineEdit_7.text())

    @folder.setter
    def folder(self, v):
        self.lineEdit_7.setText(v)

    @property
    def use_objects(self):
        return bool(self.checkBox_2.isChecked())

    @property
    def calc_internal(self):
        return self.checkBox.isChecked()


def open_file(description='', file_type='All files (*.*)', working_path=None):
    """Open a file within a working path. If no path is specified the last
    path is used. After using this function the current working path of the
    running program (ChiSurf) is updated according to the folder of the opened
    file.

    :param working_path:
    :param description:
    :param file_type:
    :return:
    """
    if working_path is None:
        working_path = mfm.working_path
    filename = str(QtGui.QFileDialog.getOpenFileName(None, description, working_path, file_type))
    mfm.working_path = os.path.dirname(filename)
    return filename


def open_files(description='', file_type='All files (*.*)', working_path=None):
    """Open a file within a working path. If no path is specified the last
    path is used. After using this function the current working path of the
    running program (ChiSurf) is updated according to the folder of the opened
    file.

    :param working_path:
    :param description:
    :param file_type:
    :return:
    """
    if working_path is None:
        working_path = mfm.working_path
    filenames = QtGui.QFileDialog.getOpenFileNames(None, description, working_path, file_type)
    mfm.working_path = os.path.dirname(filenames[0])
    return filenames


def save_file(description='', file_type='All files (*.*)', working_path=None):
    """Same as open see above a file within a working path. If no path is specified the last
    path is used. After using this function the current working path of the
    running program (ChiSurf) is updated according to the folder of the opened
    file.

    :param working_path:
    :param description:
    :param file_type:
    :return:
    """
    if working_path is None:
        working_path = mfm.working_path
    filename = str(QtGui.QFileDialog.getSaveFileName(None, description, working_path, file_type))
    mfm.working_path = os.path.dirname(filename)
    return filename


def get_directory(**kwargs):
    """Opens a new window where you can choose a directory. The current working path
    is updated to this directory.

    It either returns the directory or the files within the directory (if get_files is True).
    The returned files can be filtered for the filename ending using the kwarg filename_ending.

    :return: directory str
    """
    fn_ending = kwargs.get('filename_ending', None)
    directory = kwargs.get('directory', mfm.working_path)
    get_files = kwargs.get('get_files', False)
    if isinstance(directory, str):
        directory = str(QtGui.QFileDialog.getExistingDirectory(None, "Select Directory", directory))
    else:
        directory = str(QtGui.QFileDialog.getExistingDirectory(None, "Select Directory"))
    mfm.working_path = directory
    if not get_files:
        return directory
    else:
        filenames = [directory + '/' + s for s in os.listdir(directory)]
        if fn_ending is not None:
            filenames = fnmatch.filter(filenames, fn_ending)
        return directory, filenames



