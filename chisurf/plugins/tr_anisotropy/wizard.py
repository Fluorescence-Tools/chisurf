import sys
import json
import os.path
import pathlib
import typing
import numpy as np

from chisurf.gui import QtWidgets, QtGui, QtCore

import chisurf.gui
import chisurf.gui.widgets
import chisurf.gui.decorators
import chisurf.gui.tools
import chisurf.plugins
import chisurf.gui.tools.parameter_editor

import chisurf.data
import chisurf.experiments
import chisurf.curve
import chisurf.fitting

import chisurf.macros

import pyqtgraph as pg


colors = ['b', 'r']

name = "Tools:Anisotropy-Wizard"


class ChisurfWizard(QtWidgets.QWizard):

    data: typing.Dict[str, chisurf.curve.Curve] = {
        'irf_vv': None,
        'irf_vh': None,
        'irf_vv_bg': None,
        'irf_vh_bg': None,
        'irf_vv_bg_norm': None,
        'irf_vh_bg_norm': None,
        'data_vv': None,
        'data_vh': None
    }

    plots: typing.Dict[str, pg.PlotItem] = {
        'irf_vv': None,
        'irf_vh': None,
        'irf_vv_bg': None,
        'irf_vh_bg': None,
        'irf_vv_bg_norm': None,
        'irf_vh_bg_norm': None
    }

    def readTableValues(self, table):
        rows = table.rowCount()
        cols = table.columnCount()
        data = []
        for row in range(rows):
            row_data = []
            for col in range(cols):
                item = table.item(row, col)
                if item is not None:
                    row_data.append(float(item.text()))
                else:
                    row_data.append(None)
            data.append(row_data)
        return data

    def writeTableValues(self, table: QtWidgets.QTableWidget, arr):
        table.setRowCount(0)
        for row in arr:
            rc = table.rowCount()
            table.insertRow(rc)
            for i, col in enumerate(row):
                table.setItem(rc, i, QtWidgets.QTableWidgetItem(f"{col: 0.2f}"))
                table.resizeRowsToContents()
            table.resizeRowsToContents()

    @property
    def rotation_spectrum(self):
        data = self.readTableValues(self.tableWidget_2)
        data = np.array(data, dtype=np.float64).flatten()
        return data

    @rotation_spectrum.setter
    def rotation_spectrum(self, v: typing.List[float]):
        self.writeTableValues(self.tableWidget_2, v)

    @property
    def lifetime_spectrum(self):
        data = self.readTableValues(self.tableWidget)
        data = np.array(data, dtype=np.float64).flatten()
        return data

    @lifetime_spectrum.setter
    def lifetime_spectrum(self, v: typing.List[float]):
        self.writeTableValues(self.tableWidget, v)

    def ready(self):
        if not self.wizardPageSelectData.isComplete():
            return False
        if not self.wizardPageComponents.isComplete():
            return False
        return True

    def data_files_setup(self):
        pairs = [
            ('irf', 'vv', self.lineEdit.text()),
            ('irf', 'vh', self.lineEdit_3.text()),
            ('data', 'vv', self.lineEdit_2.text()),
            ('data', 'vh', self.lineEdit_4.text()),
        ]
        for _, _, f in pairs:
            if not pathlib.Path(f).is_file():
                return False
        return True

    def load_data(self):
        pairs = [
            ('irf', 'vv', self.lineEdit.text()),
            ('irf', 'vh', self.lineEdit_3.text()),
            ('data', 'vv', self.lineEdit_2.text()),
            ('data', 'vh', self.lineEdit_4.text()),
        ]
        for v, suffix, filename_str in pairs:
            ts = v + "_" + suffix
            chisurf.run(f"cs.current_setup.polarization = '{suffix}'")
            name = os.path.splitext(filename_str)[0] + suffix
            expriment_reader = chisurf.cs.current_experiment_reader
            dataset = expriment_reader.get_data(filename=f"{filename_str}", name=f"{name}")
            dataset = dataset[0]
            n, _ = os.path.splitext(dataset.name)
            dataset.name = n + "_" + suffix
            self.data[ts] = dataset

    def update_plot(self):
        for pk in self.plots:
            p = self.plots[pk]
            c = self.data[pk]
            if p is not None and c is not None:
                x = np.arange(len(c.x))
                p.setData(x=x, y=c.y)

    def update_irfs(self):
        lb, ub = self.region.getRegion()
        lb, ub = int(lb), int(ub)
        bg_vv = self.data['irf_vv'].y[lb:ub].mean()
        bg_vh = self.data['irf_vh'].y[lb:ub].mean()

        vv = np.clip(self.data['irf_vv'].y - bg_vv, 0, None)
        vh = np.clip(self.data['irf_vh'].y - bg_vh, 0, None)
        s = (vv + vh).sum() / 2.0
        vv *= s / vv.sum()
        vh *= s / vh.sum()

        self.data['irf_vv_bg_norm'] = chisurf.data.DataCurve(
            x=self.data['irf_vv'].x, y=vv, ey=self.data['irf_vv'].ey,
            experiment=chisurf.cs.current_experiment,
            setup=chisurf.experiments.tcspc.TCSPCReader,
            name=os.path.splitext(self.data['irf_vv'].name)[0] + "_vv"
        )

        self.data['irf_vh_bg_norm'] = chisurf.data.DataCurve(
            x=self.data['irf_vh'].x, y=vh, ey=self.data['irf_vh'].ey,
            experiment=chisurf.cs.current_experiment,
            setup=chisurf.experiments.tcspc.TCSPCReader,
            name=os.path.splitext(self.data['irf_vh'].name)[0] + "_vh"
        )

        self.update_plot()

    def onRegionChanged(self):
        lb = self.spinBox.value()
        ub = self.spinBox_2.value()
        self.region.setRegion((lb, ub))

    def init_widgets(self):
        self.irf_bg_range_plot.setLogMode(x=False, y=True)
        self.verticalLayout_3.addWidget(self.irf_bg_range_plot)

        for i, pk in enumerate(self.plots):
            color = colors[i % len(colors)]
            plot_item = self.irf_bg_range_plot.getPlotItem()
            self.plots[pk] = plot_item.plot(x=[0.0], y=[0.0], pen=pg.mkPen(color, width=2))

        self.region.setRegion((0, 100))
        self.irf_bg_range_plot.addItem(self.region)
        def onRegionUpdate(evt):
            lb, ub = self.region.getRegion()
            lb = int(lb)
            ub = int(ub)
            self.spinBox.setValue(lb)
            self.spinBox_2.setValue(ub)
            self.update_irfs()

        self.region.sigRegionChangeFinished.connect(onRegionUpdate)

    def page_actions(self):
        if self.currentPage().title() == "Normalize instrument response functions":
            self.load_data()
            self.update_plot()
        elif self.currentPage().title() =="Add lifetime & rotational components":
            fn = pathlib.Path(__file__).parent / 'wizard.spk.json'
            self.onLoadLifetimes(None, filename=fn)

    def add_rotation(self):
        lt = float(self.doubleSpinBox_4.value())
        a = float(self.doubleSpinBox_3.value())

        table = self.tableWidget_2
        rc = table.rowCount()
        table.insertRow(rc)

        table.setItem(rc, 0, QtWidgets.QTableWidgetItem(f"{lt:.2f}"))
        table.setItem(rc, 1, QtWidgets.QTableWidgetItem(f"{a:.2f}"))
        table.resizeRowsToContents()

    def remove_rotation(self):
        table = self.tableWidget_2
        rc = table.rowCount()
        idx = int(table.currentIndex().row())
        if rc >= 0:
            if idx < 0:
                idx = 0
            table.removeRow(idx)

    def add_lifetime(self):
        a = float(self.doubleSpinBox.value())
        lt = float(self.doubleSpinBox_2.value())

        table = self.tableWidget
        rc = table.rowCount()
        table.insertRow(rc)

        table.setItem(rc, 0, QtWidgets.QTableWidgetItem(f"{lt:.2f}"))
        table.setItem(rc, 1, QtWidgets.QTableWidgetItem(f"{a:.2f}"))
        table.resizeRowsToContents()

    def remove_component(self):
        table = self.tableWidget
        rc = table.rowCount()
        idx = int(table.currentIndex().row())
        if rc >= 0:
            if idx < 0:
                idx = 0
            table.removeRow(idx)

    def create_fits(self):
        if not self.ready():
            chisurf.gui.widgets.MyMessageBox(
                "No fits created!",
                info="Parameters or data missing:\n",
                show_fortune=True
            )
            return

        n = len(chisurf.imported_datasets)

        # Create lifetime fit for added data sets
        ##########################################
        model_kw = dict()
        model_kw.update(self.correction_factors)

        chisurf.macros.core_fit.add_fit(
            model_name='Lifetime fit',
            dataset_indices=[n - 2, n - 1],
            model_kw=model_kw
        )
        self.fit_vv = chisurf.fits[-2]
        self.fit_vh = chisurf.fits[-1]

        # add lifetimes
        self.fit_vv.model.lifetimes.pop()
        self.fit_vh.model.lifetimes.pop()
        lt = self.lifetime_spectrum
        for i in range(0, len(lt), 2):
            amplitude = lt[i]
            lifetime = lt[i + 1]
            self.fit_vv.model.lifetimes.append(amplitude, lifetime)
            self.fit_vh.model.lifetimes.append(amplitude, lifetime)

        # add rotations
        self.fit_vv.model.anisotropy.radioButtonVV.setChecked(True)
        self.fit_vv.model.anisotropy.hide_roation_parameters()
        self.fit_vh.model.anisotropy.radioButtonVH.setChecked(True)
        self.fit_vh.model.anisotropy.hide_roation_parameters()

        self.fit_vv.model.anisotropy.remove_rotation()
        self.fit_vh.model.anisotropy.remove_rotation()
        rs = self.rotation_spectrum
        for i in range(0, len(rs), 2):
            amplitude = rs[i]
            lifetime = rs[i + 1]
            self.fit_vv.model.anisotropy.add_rotation(b=amplitude, rho=lifetime)
            self.fit_vh.model.anisotropy.add_rotation(b=amplitude, rho=lifetime)

        # Setup IRF
        self.fit_vv.model.convolve._irf = self.data['irf_vv_bg_norm']
        self.fit_vv.model.convolve.lineEdit.setText(self.data['irf_vv_bg_norm'].name)
        self.fit_vv.update()

        self.fit_vh.model.convolve._irf = self.data['irf_vh_bg_norm']
        self.fit_vh.model.convolve.lineEdit.setText(self.data['irf_vh_bg_norm'].name)
        self.fit_vh.update()

        # Create Global fit and add vv, vh fit
        #######################################
        chisurf.macros.core_fit.add_fit(model_name='Global fit', dataset_indices=[0])

        self.global_fit = chisurf.fits[-1]
        self.global_fit.model.append_fit(self.fit_vv)
        self.global_fit.model.append_fit(self.fit_vh)

        # Link VH parameters to VV
        #######################################
        # number of photons
        self.fit_vh.model.parameters_all_dict['n0'].link = self.fit_vv.model.parameters_all_dict['n0']
        self.fit_vv.model.parameters_all_dict['n0'].fixed = False
        self.fit_vh.model.parameters_all_dict['n0'].fixed = False

        self.fit_vv.model.parameters_all_dict['l1'].fixed = False
        self.fit_vv.model.parameters_all_dict['l1'].value = self.conf_edit.dict['l1']
        self.fit_vv.model.parameters_all_dict['l1'].fixed = True
        self.fit_vv.model.parameters_all_dict['l1'].controller.finalize()

        self.fit_vv.model.parameters_all_dict['l2'].fixed = False
        self.fit_vv.model.parameters_all_dict['l2'].value = self.conf_edit.dict['l2']
        self.fit_vv.model.parameters_all_dict['l2'].fixed = True
        self.fit_vv.model.parameters_all_dict['l2'].controller.finalize()

        self.fit_vv.model.parameters_all_dict['g'].fixed = False
        self.fit_vv.model.parameters_all_dict['g'].value = self.conf_edit.dict['g_factor']
        self.fit_vv.model.parameters_all_dict['g'].fixed = True
        self.fit_vv.model.parameters_all_dict['g'].controller.finalize()

        # rotation
        self.fit_vv.model.anisotropy.polarization_type = 'vv'
        self.fit_vh.model.anisotropy.polarization_type = 'vh'
        n_rotation = len(rs) // 2
        for i in range(1, n_rotation + 1):
            self.fit_vh.model.parameters_all_dict[f'rho({i})'].link = self.fit_vv.model.parameters_all_dict[f'rho({i})']
            self.fit_vh.model.parameters_all_dict[f'b({i})'].link = self.fit_vv.model.parameters_all_dict[f'b({i})']

        # lifetime
        n_lifetime = len(lt) // 2
        for i in range(1, n_lifetime + 1):
            self.fit_vh.model.parameters_all_dict[f'xL{i}'].link = self.fit_vv.model.parameters_all_dict[f'xL{i}']
            self.fit_vh.model.parameters_all_dict[f'tL{i}'].link = self.fit_vv.model.parameters_all_dict[f'tL{i}']

        self.fit_vv.model.parameters_all_dict['lb'].fixed = True
        self.fit_vh.model.parameters_all_dict['lb'].fixed = True

        self.fit_vv.model.parameters_all_dict['l1'].fixed = True
        self.fit_vv.model.parameters_all_dict['l2'].fixed = True
        self.fit_vv.model.parameters_all_dict['g'].fixed = True
        self.fit_vh.model.parameters_all_dict['l1'].link = self.fit_vv.model.parameters_all_dict['l1']
        self.fit_vh.model.parameters_all_dict['l2'].link = self.fit_vv.model.parameters_all_dict['l2']
        self.fit_vh.model.parameters_all_dict['g'].link = self.fit_vv.model.parameters_all_dict['g']

        self.fit_vv.update()
        self.fit_vh.update()

    def onFinish(self):
        # Add corrected data to data selector
        for k in ['irf_vv_bg_norm', 'irf_vh_bg_norm', 'data_vv', 'data_vh']:
            dg = self.data[k]
            if dg is not None:
                chisurf.imported_datasets.append(dg)
        chisurf.cs.dataset_selector.update()

        self.create_fits()

    def onLoadLifetimes(self, event=None, filename: pathlib.Path = None):
        if filename is None:
            filename = chisurf.gui.widgets.get_filename(
                'Lifetime/anisotropy spectrum',
                file_type='Lifetime/anisotropy spectrum (*.spk.json)',
                working_path=pathlib.Path(__file__).parent
            )
        with open(filename, 'r') as fp:
            self.lineEdit_5.setText(filename.as_posix())
            d = json.load(fp)
            self.lifetime_spectrum = d['lifetime_spectrum']
            self.rotation_spectrum = d['rotation_spectrum']
            self.wizardPageComponents.completeChanged.emit()

    def onSaveLifetimes(self, event):
        print("onSaveLifetimes")
        filename = chisurf.gui.widgets.save_file(
                'Lifetime/anisotropy spectrum',
                file_type='Lifetime/anisotropy spectrum (*.spk.json)',
                working_path=pathlib.Path(__file__).parent
        )

        d = {
            'lifetime_spectrum': [],
            'rotation_spectrum': [],
        }

        lt = self.lifetime_spectrum
        for i in range(len(lt) // 2):
            a = lt[2 * i + 0]
            l = lt[2 * i + 1]
            d['lifetime_spectrum'].append([a, l])

        rt = self.rotation_spectrum
        for i in range(len(rt) // 2):
            a = rt[2 * i + 0]
            l = rt[2 * i + 1]
            d['rotation_spectrum'].append([a, l])
        if pathlib.Path(filename).parent.is_dir():
            with open(filename, 'w+') as fp:
                json.dump(d, fp)
        self.activateWindow()
        self.raise_()

    def liferot_setup(self):
        if len(self.lifetime_spectrum) < 2:
            return False
        if len(self.rotation_spectrum) < 2:
            return False
        return True

    def connect_actions(self):
        self.button(QtWidgets.QWizard.NextButton).clicked.connect(self.page_actions)
        self.actionAdd_Rotation.triggered.connect(self.add_rotation)
        self.actionAdd_Lifetime.triggered.connect(self.add_lifetime)
        self.actionRemove_Lifetime.triggered.connect(self.remove_component)
        self.actionRemove_Rotation.triggered.connect(self.remove_rotation)
        self.actionRegionChanged.triggered.connect(self.onRegionChanged)

        self.button(QtWidgets.QWizard.FinishButton).clicked.connect(self.onFinish)

        self.actionLoad_Lifetimes.triggered.connect(self.onLoadLifetimes)
        self.actionSave_Lifetime.triggered.connect(self.onSaveLifetimes)

        # Define when page is complete
        self.wizardPageSelectData.isComplete = self.data_files_setup
        # enable drag to line edit
        chisurf.gui.decorators.lineEdit_dragFile_injector(
            self.lineEdit,
            call=self.wizardPageSelectData.completeChanged.emit
        )
        chisurf.gui.decorators.lineEdit_dragFile_injector(
            self.lineEdit_3,
            call=self.wizardPageSelectData.completeChanged.emit
        )
        chisurf.gui.decorators.lineEdit_dragFile_injector(
            self.lineEdit_2,
            call=self.wizardPageSelectData.completeChanged.emit
        )
        chisurf.gui.decorators.lineEdit_dragFile_injector(
            self.lineEdit_4,
            call=self.wizardPageSelectData.completeChanged.emit
        )

        # Lifetime & Rotation page
        self.wizardPageComponents.isComplete = self.liferot_setup
        self.actionAdd_Lifetime.triggered.connect(self.wizardPageComponents.completeChanged.emit)
        self.actionAdd_Rotation.triggered.connect(self.wizardPageComponents.completeChanged.emit)
        self.actionRemove_Lifetime.triggered.connect(self.wizardPageComponents.completeChanged.emit)
        self.actionRemove_Rotation.triggered.connect(self.wizardPageComponents.completeChanged.emit)

    @chisurf.gui.decorators.init_with_ui("tr_anisotropy/wizard.ui", path=chisurf.settings.plugin_path)
    def __init__(self, *args, **kwargs):
        self.irf_bg_range_plot = pg.PlotWidget()
        self.region = pg.LinearRegionItem()
        self.correction_factors = {}
        self.data_loaded = False

        self.fit_vv: chisurf.fitting.fit.Fit = None
        self.fit_vh: chisurf.fitting.fit.Fit = None
        self.global_fit: chisurf.fitting.fit.Fit = None

        fn = chisurf.settings.chisurf_settings_path / "anisotropy_corrections.json"
        print("anisotropy_corrections:", fn)
        self.conf_edit = chisurf.gui.tools.parameter_editor.ParameterEditor(
            target=self.correction_factors,
            json_file=fn
        )

        self.verticalLayout_2.addWidget(self.conf_edit)

        self.init_widgets()
        self.connect_actions()


if __name__ == "plugin":
    wizard = ChisurfWizard()
    wizard.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wizard = ChisurfWizard()
    wizard.show()
    sys.exit(app.exec_())
