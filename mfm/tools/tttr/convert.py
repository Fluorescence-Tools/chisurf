from __future__ import annotations

import sys
import copy

from qtpy import QtWidgets
import qdarkstyle

import mfm
import mfm.decorators
import mfm.io.tttr
import mfm.widgets

filetypes = copy.copy(mfm.io.tttr.filetypes)
filetypes.pop('hdf')


class TTTRConvert(QtWidgets.QWidget):

    name = "tttr-convert"

    @property
    def filetype(self):
        return str(self.comboBox.currentText())

    @property
    def ending(self):
        return filetypes[self.filetype]['ending']

    @property
    def reading_routine(self):
        return filetypes[self.filetype]['read']

    @property
    def filenames(self):
        return self.filelist.filenames

    @mfm.decorators.init_with_ui(ui_filename="tttr_convert.ui")
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.comboBox.addItems(
            filetypes.keys()
        )
        self.hide()

        self.filelist = mfm.widgets.FileList(filename_ending=self.ending)
        self.actionClear_list.triggered.connect(self.filelist.clear)
        self.actionOpen_Target.triggered.connect(self.open_target)
        self.actionEnding_changed.triggered.connect(self.ending_changed)

        self.verticalLayout.addWidget(self.filelist)

    def ending_changed(self):
        self.filelist.filename_ending = self.ending

    def open_target(self, filename=None):
        if filename is None:
            filename = mfm.widgets.save_file(file_type="*.photon.h5")
        self.lineEdit.setText(filename)
        spc_files = self.filenames
        h5 = mfm.io.tttr.spc2hdf(spc_files, routine_name=self.filetype, filename=filename)
        h5.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = TTTRConvert()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
