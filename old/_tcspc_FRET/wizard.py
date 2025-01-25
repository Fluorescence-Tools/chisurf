import sys
import pathlib

import chisurf.gui
import chisurf.gui.decorators
import chisurf.gui.tools
import chisurf.plugins.anisotropy
import chisurf.gui.tools.parameter_editor
from chisurf.gui import QtWidgets, QtGui

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree, parameterTypes as ptypes

class QIComboBox(QtWidgets.QComboBox):
    def __init__(self,parent=None):
        super(QIComboBox, self).__init__(parent)



class ChisurfWizard(QtWidgets.QWizard):

    @chisurf.gui.decorators.init_with_ui(
        "wizard.ui",
        path=pathlib.Path(chisurf.plugins.tcspc_Anisotropy.__file__).parent
    )
    def __init__(self, *args, **kwargs):
        self.addPage(Page1(self))
        self.addPage(Page2(self))

class Page1(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(Page1, self).__init__(parent)
        self.comboBox = QIComboBox(self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.comboBox)

        def a(x=5, y=6):
            QtWidgets.QMessageBox.information(None, 'Hello World', f'X is {x}, Y is {y}')

        # -----------
        # discussion is from here:
        params = Parameter.create(
            name='"a" parameters', type='group', children=[
            dict(name='x', type='int', value=5),
            dict(name='y', type='int', value=6)
        ]
        )
        tree = ParameterTree()
        tree.setParameters(params)
        layout.addWidget(tree)

        self.setLayout(layout)


class Page2(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(Page2, self).__init__(parent)
        self.label1 = QtWidgets.QLabel()
        self.label2 = QtWidgets.QLabel()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label1)
        layout.addWidget(self.label2)

        fn = pathlib.Path(__file__).parent / "anisotropy_corrections.json"
        fn_str = str(fn.as_posix())
        d = {}
        conf_edit = chisurf.gui.tools.parameter_editor.ParameterEditor(target=d, json_file=fn_str)

        layout.addWidget(conf_edit)
        self.setLayout(layout)

    def initializePage(self):
        self.label1.setText("Example text")
        self.label2.setText("Example text")


if __name__ == '__main__':
    wizard = ChisurfWizard()
    wizard.show()

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     win = TCSPCAnisotropyWizard()
#     win.show()
#     sys.exit(app.exec_())
