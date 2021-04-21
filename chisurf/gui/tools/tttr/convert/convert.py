from __future__ import annotations

import sys
import copy

from qtpy import QtWidgets

import chisurf.decorators
import chisurf.fio.fluorescence.tttr
import chisurf.gui.decorators
import chisurf.gui.widgets

filetypes = copy.copy(chisurf.fio.fluorescence.tttr.filetypes)
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
        return self.file_list.filenames

    @chisurf.gui.decorators.init_with_ui(ui_filename="tttr_convert.ui")
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.comboBox.addItems(
            filetypes.keys()
        )
        self.hide()

        self.file_list = chisurf.gui.widgets.FileList(filename_ending=self.ending)
        self.actionClear_list.triggered.connect(self.file_list.clear)
        self.actionOpen_Target.triggered.connect(self.open_target)
        self.actionEnding_changed.triggered.connect(self.ending_changed)

        self.verticalLayout.addWidget(self.file_list)

    def ending_changed(self):
        self.file_list.filename_ending = self.ending

    def open_target(
            self
    ):
        filename = chisurf.gui.widgets.save_file(
            file_type="Photon-HDF (*.photon.h5)"
        )
        self.lineEdit.setText(filename)
        spc_files = self.filenames
        h5 = chisurf.fio.fluorescence.tttr.spc2hdf(
            spc_files,
            routine_name=self.filetype,
            filename=filename
        )
        h5.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = TTTRConvert()
    win.show()
    sys.exit(app.exec_())
