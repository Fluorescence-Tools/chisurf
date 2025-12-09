import os
import sys

from qtpy import QtGui, QtCore, uic, QtWidgets
from OpenGL.GL import *
from qtpy.QtOpenGL import *
from qtpy.QtCore import Qt

try:
    import pymol2
    _HAVE_PYMOL2 = True
except Exception:
    pymol2 = None
    _HAVE_PYMOL2 = False
import chisurf
from chisurf.plots.plotbase import Plot

try:
    # Older Qt bindings expose QGLWidget/QGLFormat via QtOpenGL; this may be missing
    from qtpy.QtOpenGL import QGLWidget, QGLFormat  # type: ignore[attr-defined]
    _HAVE_QGL = True
except Exception:
    _HAVE_QGL = False

    class QGLFormat:  # minimal stub used only when real GL format is unavailable
        def setStencil(self, *args, **kwargs):
            pass

        def setRgba(self, *args, **kwargs):
            pass

        def setDepth(self, *args, **kwargs):
            pass

        def setDoubleBuffer(self, *args, **kwargs):
            pass

    class QGLWidget(QtWidgets.QWidget):  # type: ignore[misc]
        def __init__(self, fmt=None, parent=None):
            # Ignore the format object in environments without real GL support
            super().__init__(parent)

try:
    if _HAVE_QGL:
        chisurf.logging.info("MolView: using real QGLWidget backend")
    else:
        chisurf.logging.warning("MolView: using stub QGLWidget backend; PyMOL display may be disabled")
except Exception:
    pass


# class EmittingStream(QtCore.QObject):
#
#     textWritten = QtCore.pyqtSignal(str)
#
#     def write(self, text):
#         self.textWritten.emit(str(text))
#

_PYMOL_MISSING_HINT = (
    "PyMOL is not available in this ChiSurf environment, so the 3D view is disabled.\n\n"
    "To enable the 3D viewer, install ChiSurf (which bundles PyMOL) via the ChiSurf "
    "Package Manager (Help → Updates and Packages → Package Manager) or through "
    "`conda install chisurf -c conda-forge -c tpeulen` on the command line."
)


class _PymolCmdStub:

    def __getattr__(self, name):
        if name == "count_states":
            def _count_states(*_, **__):
                return 0
            return _count_states

        def _noop(*_, **__):
            return None

        return _noop

    # Explicit helpers used in several places
    def do(self, *_, **__):
        return None

    def set(self, *_, **__):
        return None

    def hide(self, *_, **__):
        return None

    def show(self, *_, **__):
        return None

    def delete(self, *_, **__):
        return None

    def read_pdbstr(self, *_, **__):
        return None

    def load(self, *_, **__):
        return None

    def alter(self, *_, **__):
        return None

    def set_color(self, *_, **__):
        return None


class _PymolStub:

    def __init__(self):
        self.cmd = _PymolCmdStub()

    def start(self):
        return None

    def idle(self):
        return False

    def draw(self):
        return None

    def reshape(self, *_, **__):
        return None

    def drag(self, *_, **__):
        return None

    def button(self, *_, **__):
        return None

    def reinitialize(self):
        return None


class MolQtWidget(QGLWidget):
    """
    http://www.mail-archive.com/pymol-users@lists.sourceforge.net/msg09609.html
    maybe later use this...: http://www.plosone.org/article/info:doi/10.1371/journal.pone.0021931
    """
    _buttonMap = {Qt.LeftButton:0,
                  Qt.MidButton:1,
                  Qt.RightButton:2}

    def __init__(self, parent, enableUi=True, File="", play=False, sequence=False):
        #enableUi=True
        f = QGLFormat()
        f.setStencil(True)
        f.setRgba(True)
        f.setDepth(True)
        f.setDoubleBuffer(True)
        self.play = play
        self.nFrames = 0
        QGLWidget.__init__(self, f, parent=parent)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: black; border: 1px solid red;")

        if not _HAVE_PYMOL2:
            self._enableUi = False
            self.pymol = _PymolStub()
            self.cmd = self.pymol.cmd
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)
            icon = QtWidgets.QLabel("ℹ️", self)
            font = icon.font()
            font.setPointSize(font.pointSize() + 6)
            icon.setFont(font)
            icon.setAlignment(QtCore.Qt.AlignCenter)
            label = QtWidgets.QLabel(_PYMOL_MISSING_HINT, self)
            label.setWordWrap(True)
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addStretch(1)
            layout.addWidget(icon)
            layout.addSpacing(8)
            layout.addWidget(label)
            layout.addStretch(1)
            return

        self._enableUi = enableUi
        self.pymol = pymol2.PyMOL()# _pymolPool.getInstance()
        self.pymol.start()
        self.cmd = self.pymol.cmd
        try:
            # Ensure a dark background so a successfully initialized viewer is visible
            self.pymol.cmd.bg_color('black')
        except Exception:
            pass
        try:
            chisurf.logging.info(
                "MolQtWidget.__init__: PyMOL started (enableUi=%s, sequence=%s)",
                enableUi,
                sequence
            )
        except Exception:
            pass
        if hasattr(self.pymol, 'toPymolName'):
            self.toPymolName = self.pymol.toPymolName ### Attribute Error
        self._pymolProcess()

        if not self._enableUi:
            self.pymol.cmd.set("internal_gui", 0)
            self.pymol.cmd.set("internal_feedback", 1)
            self.pymol.cmd.button("double_left", "None", "None")
            self.pymol.cmd.button("single_right", "None", "None")

        self.pymol.cmd.set("internal_gui_mode", "0")
        self.pymol.cmd.set("internal_feedback", "0")
        if sequence:
            self.pymol.cmd.set("seq_view", "1")
        if File != "":
            self.nFrames += 1
            self.pymol.cmd.load(File, 'p', self.nFrames)

        self.pymol.reshape(self.width(),self.height())
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._pymolProcess)
        self.resizeGL(self.width(),self.height())
        #globalSettings.settingsChanged.connect(self._updateGlobalSettings)
        self._updateGlobalSettings()

    def openFile(self, File, frame=None, mode=None, verbose=True, object_name=None):
        if isinstance(File, str):
            if os.path.isfile(File):
                if self.play is True:
                    self.pymol.cmd.mplay()

                if frame is None:
                    self.nFrames += 1
                    frame = self.nFrames

                object_name = object_name if object_name is not None else os.path.basename(File)
                if verbose:
                    print("Pymol opening file: %s" % File)
                    print("Object name: %s" % object_name)
                self.pymol.cmd.load(File, object_name, frame)
                if self.nFrames == 1:
                    if mode is not None:
                        if mode == 'coarse':
                            self.pymol.cmd.hide('all')
                            self.pymol.cmd.do('show ribbon')
                            self.pymol.cmd.do('show spheres, name cb')
                        else:
                            self.pymol.cmd.hide('all')
                            self.pymol.cmd.show(mode)

                self.pymol.cmd.orient()
                #self.pymol.cmd.iterate_state()

    def __del__(self):
        pass

    def _updateGlobalSettings(self):
        #for k,v in globalSettings.settings.iteritems():
        #    self.pymol.cmd.set(k, v)
        #self.update()
        return

    def redoSizing(self):
        self.resizeGL(self.width(), self.height())

    def paintGL(self):
        glViewport(0, 0, self.width(), self.height())
        bottom = self.mapToGlobal(QtCore.QPoint(0,self.height())).y()
        #self.pymol.cmd.set("_stencil_parity", bottom & 0x1)
        self._doIdle()
        self.pymol.draw()

    def mouseMoveEvent(self, ev):
        self.pymol.drag(ev.x(), self.height()-ev.y(),0)
        self._pymolProcess()

    def mousePressEvent(self, ev):
        if not self._enableUi:
            self.pymol.cmd.button("double_left","None","None")
            self.pymol.cmd.button("single_right","None","None")
        self.pymol.button(self._buttonMap[ev.button()], 0, ev.x(),
                          self.height() - ev.y(),0)
        self._pymolProcess()

    def mouseReleaseEvent(self, ev):
        self.pymol.button(self._buttonMap[ev.button()], 1, ev.x(),
                          self.height()-ev.y(),0)
        self._pymolProcess()
        self._timer.start(0)

    def resizeGL(self, w, h):
        try:
            chisurf.logging.info("MolQtWidget.resizeGL: w=%d h=%d", w, h)
        except Exception:
            pass
        self.pymol.reshape(w,h, True)
        self._pymolProcess()

    def initializeGL(self):
        pass

    def _pymolProcess(self):
        self._doIdle()
        self.update()

    def _doIdle(self):
        if self.pymol.idle():
            self._timer.start(0)

    def strPDB(self, str):
        self.cmd.read_pdbstr(str)

    def reset(self):
        self.nFrames = 0
        self.pymol.cmd.reinitialize()


class ControlWidget(QtWidgets.QWidget):

    history = ['spectrum count, rainbow_rev, all, byres=1', 'intra_fit all']

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "molViewControlWidget.ui"
            ),
            self
        )

        self.parent = parent

        self.pushButton_4.clicked.connect(self.onReset)
        self.pushButton_6.clicked.connect(self.onIntrafit)
        self.pushButton_5.clicked.connect(self.onExeCommand)

        self.actionNext_frame.triggered.connect(self.onNextFrame)
        self.actionPrevious_frame.triggered.connect(self.onPreviousFrame)
        self.actionPause.triggered.connect(self.onStopPymol)
        self.actionPlay.triggered.connect(self.onPlayPymol)
        self.actionTo_last_frame.triggered.connect(self.onLastState)
        self.actionTo_first_frame.triggered.connect(self.onFirstState)
        self.actionShow_quencher.triggered.connect(self.onHighlightQuencher)

        self.lineEdit.returnPressed .connect(self.onExeCommand)
        self.radioButton.clicked.connect(self.onUpdateSpectrum)
        self.radioButton_2.clicked.connect(self.onUpdateSpectrum)
        self.actionCurrentStateChanged.triggered.connect(self.onCurrentStateChanged)

    def onPreviousFrame(self):
        s = self.current_state
        s -= 1
        s = self.n_states if s < 1 else s
        self.current_state = (s % self.n_states) + 1

    def onNextFrame(self):
        self.current_state = ((self.current_state + 1) % self.n_states) + 1

    def onCurrentStateChanged(self):
        self.parent.pymolWidget.pymol.cmd.do("set state, %i" % self.current_state)

    @property
    def current_state(self):
        return int(self.spinBox.value())

    @current_state.setter
    def current_state(self, v):
        self.spinBox.setValue(v)
        self.parent.pymolWidget.pymol.cmd.do("set state, %i" % v)
        self.onCurrentStateChanged()

    @property
    def n_states(self):
        return int(self.spinBox_2.value())

    @n_states.setter
    def n_states(self, v):
        self.spinBox_2.setValue(v)
        self.horizontalSlider.setMaximum(v)
        self.spinBox.setMaximum(v)

    @property
    def spectrum_type(self):
        if self.radioButton.isChecked():
            return 'movemap'
        elif self.radioButton_2.isChecked():
            return 'index'

    @property
    def pymol(self):
        return self.parent.pymolWidget.pymol

    def onIntrafit(self):
        self.pymol.cmd.do("intra_fit all")

    @property
    def first_frame(self):
        return int(self.spinBox_3.value())

    @property
    def last_frame(self):
        return int(self.spinBox_2.value())

    @property
    def step(self):
        return int(self.spinBox.value())

    def onLastState(self):
        self.parent.pymolWidget.pymol.cmd.do("set state, %i" % self.n_states)
        self.current_state = self.n_states

    def onFirstState(self):
        self.parent.pymolWidget.pymol.cmd.do("set state, %i" % 1)
        self.current_state = 1

    def onExeCommand(self):
        print("onExeCommand")
        #sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        c = str(self.lineEdit.text())
        print("%s" % c)
        self.parent.pymolWidget.pymol.cmd.do(c)
        self.lineEdit.clear()
        sys.stdout = sys.__stdout__

    def normalOutputWritten(self, text):
        """Append name to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.plainTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.plainTextEdit.setTextCursor(cursor)
        self.plainTextEdit.ensureCursorVisible()

    def onUpdateSpectrum(self):
        if self.spectrum_type == 'movemap':
            self.parent.pymolWidget.pymol.cmd.do("spectrum b")
        elif self.spectrum_type == 'index':
            self.parent.pymolWidget.pymol.cmd.do('spectrum count, rainbow_rev, all, byres=1')

    def onStopPymol(self):
        print("onStopPymol")
        self.parent.pymolWidget.pymol.cmd.mstop()

    def onPlayPymol(self):
        print("onPlayPymol")
        self.parent.pymolWidget.pymol.cmd.mplay()

    def onReset(self):
        self.plainTextEdit.clear()
        self.parent.reset()

    def onHighlightQuencher(self):
        quencher = self.parent.quencher
        pymol = self.parent.pymol
        for res_name in quencher:
            pymol.cmd.do("hide lines, resn %s" % res_name)
            pymol.cmd.do("show sticks, resn %s" % res_name)
            pymol.cmd.do("color red, resn %s" % res_name)


class MolView(Plot):

    @property
    def pymol(self):
        return self.pymolWidget.pymol

    def __init__(self, fit=None, enableUi=False, mode='cartoon', sequence=True, **kwargs):
        Plot.__init__(self, fit)
        self.quencher = kwargs.get('quencher', )
        self.name = kwargs.get('name', 'MolView')
        self.fit = fit
        self.mode = mode
        self.plot_controller = ControlWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.pymolWidget = MolQtWidget(self, play=False, sequence=sequence, enableUi=enableUi)
        self.layout.addWidget(self.pymolWidget)
        self.pymolWidget.show()

        structure = None
        try:
            model = getattr(fit, 'model', None)
            structure = getattr(model, 'structure', None)
            if structure is None:
                structure = getattr(fit, 'data', None)
        except Exception:
            structure = None

        try:
            chisurf.logging.info(
                "MolView.__init__: fit=%s structure=%s type=%s",
                getattr(fit, 'name', 'unknown'),
                getattr(structure, 'name', getattr(structure, 'filename', 'unknown')) if structure is not None else 'None',
                structure.__class__.__name__ if structure is not None else 'None',
            )
        except Exception:
            pass

        self.structure = structure
        if structure is not None:
            try:
                self.open_structure(structure)
            except Exception as e:
                try:
                    chisurf.logging.warning(f"MolView.__init__: open_structure failed: {e}")
                except Exception:
                    pass

    def open_file(self, filename, bfact=None):
        self.pymolWidget.openFile(filename, mode=self.mode)
        self.set_bfactor(bfact)

    def set_bfactor(self, bfact=None):

        #if bfact is not None:
        #    self.pymol.bfact = list(bfact)
        #    self.pymol.cmd.alter("all and n. CA", "b=bfact.pop(0)")
        #self.pymol.cmd.do("orient")

        if bfact is not None:
            cmd = self.pymol.cmd
            structure = self.structure
            index_min = min(set(structure.atoms['res_id']))
            counter = index_min
            for b in bfact:
                cmd.alter("p and resi %s and n. CA" % counter, "b=%s" % b)
                counter += 1

    def append_structure(self, structure):
        cmd = self.pymol.cmd
        cmd.read_pdbstr(str(structure), 'p')
        bfacts = structure.b_factors
        self.set_bfactor(bfacts)
        #obj = cmd.get_object_list('p')[0]

        #bfacts = structure.b_factors

        #print bfacts
        #index_min = min(set(structure.atoms['res_id']))
        #counter = index_min
        #for bfact in bfacts:
        #    cmd.alter("p and resi %s and n. CA" % counter, "b=%s" % bfact)
        #    counter += 1
        #cmd.spectrum("b", "rainbow", "p and n. CA ")
        #cmd.ramp_new("count", obj, [min(bfacts), max(bfacts)], "rainbow")
        #cmd.recolor()

        self.plot_controller.n_states = self.pymol.cmd.count_states(selection="(all)")

    def open_structure(self, structure, bfact=None, mode=None):
        mode = mode if mode is not None else self.mode
        self.pymol.cmd.read_pdbstr(str(structure), 'p')
        self.set_bfactor(bfact)
        #self.pymol.cmd.hide('all')
        #self.pymol.cmd.do('show lines')

        if mode is not None:
            if mode == 'coarse':
                self.pymol.cmd.hide('all')
                self.pymol.cmd.do('show ribbon')
                self.pymol.cmd.do('show spheres, name cb')
            else:
                self.pymol.cmd.hide('all')
                self.pymol.cmd.show(mode)

    def replace_structure(self, structure, bfact, stateNbr=0):
        print("MolView:replace_structure")
        self.open_structure(structure, stateNbr, mode=self.mode)
        self.set_bfactor(bfact)

    def setState(self, nbr):
        print("set state: % s" % nbr)
        self.pymol.cmd.set("state", nbr)

    def reset(self):
        self.pymolWidget.reset()

    def colorStates(self, rgbColors):
        print("colorStates") # TODO
        for i, r in enumerate(rgbColors):
            color = "[%s, %s, %s]" % (r[0] / 255., r[1] / 255., r[2] / 255.)
            self.pymol.cmd.set_color("tmpcolor", color)
            self.pymol.cmd.set("cartoon_color", "tmpcolor", "p", (i + 1))

