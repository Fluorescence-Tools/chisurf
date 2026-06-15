import sys
import json
import os.path
import pathlib
import typing
import numpy as np
import shutil

from chisurf.gui import QtWidgets
from chisurf import logging

import chisurf.gui
import chisurf.gui.widgets
import chisurf.gui.decorators
import chisurf.plugins
import chisurf.gui.widgets.parameter_editor

import chisurf.core.data
import chisurf.core.experiments
import chisurf.core.curve
import chisurf.core.fitting
import chisurf.core.settings

import chisurf.macros
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client

import pyqtgraph as pg

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c



colors = ['b', 'r']

name = "Tools:Anisotropy-Wizard"


@persist_plugin_state("tr_anisotropy")
class ChisurfWizard(QtWidgets.QWizard):

    data: typing.Dict[str, chisurf.core.curve.Curve] = {
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
    
    @staticmethod
    def get_user_plugin_settings_path() -> pathlib.Path:
        """
        Get the path to the user settings directory for the tr_anisotropy plugin.
        Creates the directory if it doesn't exist.
        
        Returns:
            pathlib.Path: Path to the user settings directory for the plugin
        """
        # Get the user settings path
        user_plugin_path = chisurf.core.settings.chisurf_settings_path / "plugins" / "tr_anisotropy"
        # Create the directory if it doesn't exist
        user_plugin_path.mkdir(parents=True, exist_ok=True)
        return user_plugin_path
    
    @classmethod
    def get_spk_json_path(cls) -> pathlib.Path:
        """
        Get the path to the wizard.spk.json file in the user settings directory.
        If the file doesn't exist, copy it from the plugin directory.
        
        Returns:
            pathlib.Path: Path to the wizard.spk.json file
        """
        # Get the user plugin settings path
        user_plugin_path = cls.get_user_plugin_settings_path()
        # Define the path to the wizard.spk.json file
        spk_json_path = user_plugin_path / "wizard.spk.json"
        
        # If the file doesn't exist in the user settings directory, copy it from the plugin directory
        if not spk_json_path.exists():
            # Get the path to the plugin directory
            plugin_path = pathlib.Path(__file__).parent
            # Define the path to the original wizard.spk.json file
            original_spk_json_path = plugin_path / "wizard.spk.json"
            
            # Check if the original file exists
            if original_spk_json_path.exists():
                # Copy the file to the user settings directory
                shutil.copyfile(original_spk_json_path, spk_json_path)
            else:
                # Create a default file if the original doesn't exist
                default_content = {
                    "lifetime_spectrum": [[0.3, 1.8], [0.7, 4.1]],
                    "rotation_spectrum": [[0.28, 0.15], [0.1, 10.0]]
                }
                with open(spk_json_path, 'w') as f:
                    json.dump(default_content, f)
        
        return spk_json_path

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
        # Check if we're using Jordi format
        _cs_mod = getattr(chisurf, "cs", object())
        _current_setup = getattr(_cs_mod, "current_setup", None)
        is_jordi = getattr(_current_setup, "is_jordi", False)
        
        if is_jordi:
            # For Jordi format, we only need one file for IRF and one for data
            pairs = [
                ('irf', 'vv/vh', self.lineEdit.text()),  # IRF file contains both VV and VH
                ('data', 'vv/vh', self.lineEdit_2.text()),  # Data file contains both VV and VH
            ]
        else:
            # For regular format, we need separate files for VV and VH
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
        _cs_mod = getattr(chisurf, "cs", object())
        _current_setup = getattr(_cs_mod, "current_setup", None)
        is_jordi = getattr(_current_setup, "is_jordi", False)
        
        def _load_polarized(v, suffix, filename_str):
            ts = v + "_" + suffix
            name = os.path.splitext(filename_str)[0] + suffix
            if _current_setup is not None:
                _current_setup.polarization = suffix
            _reader = getattr(_current_setup, "experiment_reader", None)
            if _reader is None:
                _reader = getattr(_cs_mod, "current_experiment_reader", None)
            dataset = _reader.get_data(filename=f"{filename_str}", name=f"{name}")
            dataset = dataset[0]
            n, _ = os.path.splitext(dataset.name)
            dataset.name = n + "_" + suffix
            self.data[ts] = dataset
        
        if is_jordi:
            # For Jordi format, we need to load each file twice with different polarization parameters
            jordi_pairs = [
                ('irf', 'vv', self.lineEdit.text()),
                ('irf', 'vh', self.lineEdit.text()),
                ('data', 'vv', self.lineEdit_2.text()),
                ('data', 'vh', self.lineEdit_2.text()),
            ]
            
            for v, suffix, filename_str in jordi_pairs:
                _load_polarized(v, suffix, filename_str)
        else:
            # For regular format, we load each file separately
            pairs = [
                ('irf', 'vv', self.lineEdit.text()),
                ('irf', 'vh', self.lineEdit_3.text()),
                ('data', 'vv', self.lineEdit_2.text()),
                ('data', 'vh', self.lineEdit_4.text()),
            ]
            
            for v, suffix, filename_str in pairs:
                _load_polarized(v, suffix, filename_str)

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

        # Ensure required data is available
        if 'irf_vv' not in self.data or 'irf_vh' not in self.data:
            return
        if self.data['irf_vv'] is None or self.data['irf_vh'] is None:
            return

        y_vv = np.asarray(self.data['irf_vv'].y)
        y_vh = np.asarray(self.data['irf_vh'].y)
        n = min(len(y_vv), len(y_vh))
        if n == 0:
            return

        # Clamp bounds to valid range [0, n]
        lb = max(0, min(lb, n))
        ub = max(0, min(ub, n))

        # Avoid empty slice: if region invalid or empty, use 0.0 background to skip correction
        if ub <= lb:
            logging.warning(f"Background region is empty or invalid (lb={lb}, ub={ub}). Skipping background update.")
            return

        # Compute safe background means
        sl_vv = y_vv[lb:ub]
        sl_vh = y_vh[lb:ub]
        bg_vv = float(np.nanmean(sl_vv)) if sl_vv.size > 0 else 0.0
        bg_vh = float(np.nanmean(sl_vh)) if sl_vh.size > 0 else 0.0

        # Background subtract and clip to non-negative
        vv = np.clip(y_vv - bg_vv, 0, None)
        vh = np.clip(y_vh - bg_vh, 0, None)

        # Normalize intensities; guard against division by zero
        s = (vv + vh).sum() / 2.0
        vv_sum = vv.sum()
        vh_sum = vh.sum()
        if s > 0:
            if vv_sum > 0:
                vv = vv * (s / vv_sum)
            if vh_sum > 0:
                vh = vh * (s / vh_sum)

        _cs_mod = getattr(chisurf, "cs", object())
        _current_experiment = getattr(_cs_mod, "current_experiment", None)
        self.data['irf_vv_bg_norm'] = chisurf.core.data.DataCurve(
            x=self.data['irf_vv'].x, y=vv, ey=self.data['irf_vv'].ey,
            experiment=_current_experiment,
            setup=chisurf.core.experiments.tcspc.TCSPCReader,
            name=os.path.splitext(self.data['irf_vv'].name)[0] + "_vv"
        )

        self.data['irf_vh_bg_norm'] = chisurf.core.data.DataCurve(
            x=self.data['irf_vh'].x, y=vh, ey=self.data['irf_vh'].ey,
            experiment=_current_experiment,
            setup=chisurf.core.experiments.tcspc.TCSPCReader,
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

        plot_item = self.irf_bg_range_plot.getPlotItem()
        
        # Add legend to the plot
        plot_item.addLegend()
        
        # Create plots with specific styling for each type
        for pk in self.plots:
            if pk.endswith('_bg_norm'):
                # Background-corrected IRF should pop more - use brighter colors and thicker lines
                if pk.startswith('irf_vv'):
                    self.plots[pk] = plot_item.plot(x=[0.0], y=[0.0], pen=pg.mkPen('b', width=3), name="VV (corrected)")
                else:
                    self.plots[pk] = plot_item.plot(x=[0.0], y=[0.0], pen=pg.mkPen('r', width=3), name="VH (corrected)")
            else:
                # Non-corrected IRF with 60% alpha
                if pk.startswith('irf_vv'):
                    color = pg.mkColor('b')
                    color.setAlphaF(0.4)
                    self.plots[pk] = plot_item.plot(x=[0.0], y=[0.0], pen=pg.mkPen(color, width=2), name="VV (raw)")
                else:
                    color = pg.mkColor('r')
                    color.setAlphaF(0.4)
                    self.plots[pk] = plot_item.plot(x=[0.0], y=[0.0], pen=pg.mkPen(color, width=2), name="VH (raw)")

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

    def set_initial_region(self):
        """
        Set the initial background region to 30%-80% of the data range.
        This is called after the data is loaded and plotted.
        """
        # Check if we have data
        if 'irf_vv' in self.data and self.data['irf_vv'] is not None:
            # Get the data range (length of x-axis)
            data_range = len(self.data['irf_vv'].x)
            
            # Calculate 30% and 80% of the range
            lower_bound = int(data_range * 0.3)
            upper_bound = int(data_range * 0.8)
            
            # Set the region
            self.region.setRegion((lower_bound, upper_bound))
            
            # Update the spinbox values
            self.spinBox.setValue(lower_bound)
            self.spinBox_2.setValue(upper_bound)
            
            # Update the IRFs with the new region
            self.update_irfs()
            
            logging.info(f"Set initial background region to {lower_bound}-{upper_bound} (30%-80% of data range {data_range})")
    
    def page_actions(self):
        # Auto-save when changing pages if we're on the components page
        if self.currentPage().title() == "Lifetime and rotation components":
            # Only auto-save if we have valid data
            if self.liferot_setup():
                self.onSaveLifetimes(None)
                
        if self.currentPage().title() == "Normalize instrument response functions":
            self.load_data()
            self.update_plot()
            # Set the initial region after the data is loaded and plotted
            self.set_initial_region()

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

        _cs_mod = getattr(chisurf, "cs", object())
        n = len(getattr(_cs_mod, "imported_datasets", []))
        fc = get_fitting_client()

        # Create lifetime fit for added data sets
        ##########################################
        model_kw = dict()
        model_kw.update(self.correction_factors)

        chisurf.core.actions.dispatch(
            name="fit.add",
            payload={
                "model_name": "Lifetime fit",
                "dataset_indices": [n - 2, n - 1],
                "model_kw": model_kw,
            },
        )
        self.fit_vv = fc.get_fit_objects()[-2]
        self.fit_vh = fc.get_fit_objects()[-1]
        _vv_idx = len(fc.get_fit_objects()) - 2
        _vh_idx = len(fc.get_fit_objects()) - 1

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
        chisurf.core.actions.dispatch(
            name="fit.add",
            payload={
                "model_name": "Global fit",
                "dataset_indices": [0],
            },
        )

        self.global_fit = fc.get_fit_objects()[-1]
        self.global_fit.model.append_fit(self.fit_vv)
        self.global_fit.model.append_fit(self.fit_vh)

        # Link VH parameters to VV
        #######################################
        # number of photons
        fc.link_parameters(
            parameter_name='n0', target_parameter_name='n0',
            fit_index=_vh_idx, target_fit_index=_vv_idx,
        )
        fc.set_parameter_fixed(parameter_name='n0', fixed=False, fit_index=_vv_idx)
        fc.set_parameter_fixed(parameter_name='n0', fixed=False, fit_index=_vh_idx)

        fc.set_parameter_fixed(parameter_name='l1', fixed=False, fit_index=_vv_idx)
        fc.set_parameter_value(parameter_name='l1', value=self.conf_edit.dict['l1'], fit_index=_vv_idx)
        fc.set_parameter_fixed(parameter_name='l1', fixed=True, fit_index=_vv_idx)
        fc.update_fit(fit_index=_vv_idx)

        fc.set_parameter_fixed(parameter_name='l2', fixed=False, fit_index=_vv_idx)
        fc.set_parameter_value(parameter_name='l2', value=self.conf_edit.dict['l2'], fit_index=_vv_idx)
        fc.set_parameter_fixed(parameter_name='l2', fixed=True, fit_index=_vv_idx)
        fc.update_fit(fit_index=_vv_idx)

        fc.set_parameter_fixed(parameter_name='g', fixed=False, fit_index=_vv_idx)
        fc.set_parameter_value(parameter_name='g', value=self.conf_edit.dict['g_factor'], fit_index=_vv_idx)
        fc.set_parameter_fixed(parameter_name='g', fixed=True, fit_index=_vv_idx)
        fc.update_fit(fit_index=_vv_idx)

        # rotation
        self.fit_vv.model.anisotropy.polarization_type = 'vv'
        self.fit_vh.model.anisotropy.polarization_type = 'vh'
        n_rotation = len(rs) // 2
        for i in range(1, n_rotation + 1):
            fc.link_parameters(
                parameter_name=f'rho({i})', target_parameter_name=f'rho({i})',
                fit_index=_vh_idx, target_fit_index=_vv_idx,
            )
            fc.link_parameters(
                parameter_name=f'b({i})', target_parameter_name=f'b({i})',
                fit_index=_vh_idx, target_fit_index=_vv_idx,
            )

        # lifetime
        n_lifetime = len(lt) // 2
        for i in range(1, n_lifetime + 1):
            fc.link_parameters(
                parameter_name=f'xL{i}', target_parameter_name=f'xL{i}',
                fit_index=_vh_idx, target_fit_index=_vv_idx,
            )
            fc.link_parameters(
                parameter_name=f'tL{i}', target_parameter_name=f'tL{i}',
                fit_index=_vh_idx, target_fit_index=_vv_idx,
            )

        fc.set_parameter_fixed(parameter_name='lb', fixed=True, fit_index=_vv_idx)
        fc.set_parameter_fixed(parameter_name='lb', fixed=True, fit_index=_vh_idx)

        fc.set_parameter_fixed(parameter_name='l1', fixed=True, fit_index=_vv_idx)
        fc.set_parameter_fixed(parameter_name='l2', fixed=True, fit_index=_vv_idx)
        fc.set_parameter_fixed(parameter_name='g', fixed=True, fit_index=_vv_idx)
        fc.link_parameters(
            parameter_name='l1', target_parameter_name='l1',
            fit_index=_vh_idx, target_fit_index=_vv_idx,
        )
        fc.link_parameters(
            parameter_name='l2', target_parameter_name='l2',
            fit_index=_vh_idx, target_fit_index=_vv_idx,
        )
        fc.link_parameters(
            parameter_name='g', target_parameter_name='g',
            fit_index=_vh_idx, target_fit_index=_vv_idx,
        )

        self.fit_vv.update()
        self.fit_vh.update()

    def onFinish(self):
        _cs_mod = getattr(chisurf, "cs", object())
        _datasets = getattr(_cs_mod, "imported_datasets", None)
        if _datasets is not None:
            for k in ['irf_vv_bg_norm', 'irf_vh_bg_norm', 'data_vv', 'data_vh']:
                dg = self.data[k]
                if dg is not None:
                    _datasets.append(dg)
        _ds_selector = getattr(_cs_mod, "dataset_selector", None)
        if _ds_selector is not None:
            _ds_selector.update()

        self.create_fits()

    def onLoadLifetimes(self, event=None, filename: pathlib.Path = None):
        if filename is None:
            # Use the user plugin settings path as the working path
            user_plugin_path = self.get_user_plugin_settings_path()
            # Ensure the wizard.spk.json file exists in the user settings directory
            default_file = self.get_spk_json_path()
            
            filename = chisurf.gui.widgets.get_filename(
                'Lifetime/anisotropy spectrum',
                file_type='Lifetime/anisotropy spectrum (*.spk.json)',
                working_path=user_plugin_path
            )
        with open(filename, 'r') as fp:
            self.lineEdit_5.setText(filename.as_posix())
            d = json.load(fp)
            self.lifetime_spectrum = d['lifetime_spectrum']
            self.rotation_spectrum = d['rotation_spectrum']
            self.wizardPageComponents.completeChanged.emit()

    def onSaveLifetimes(self, event):
        logging.info("onSaveLifetimes")
        # Get the current file path from the lineEdit
        current_file = self.lineEdit_5.text()
        
        # If no current file is set, use the default path
        if not current_file:
            filename = self.get_spk_json_path()
        else:
            filename = pathlib.Path(current_file)

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
                
            # If the file was saved to a different location than the default,
            # copy it to the default location as well
            default_path = self.get_spk_json_path()
            if pathlib.Path(filename) != default_path:
                # Make a backup of the default file if it exists
                if default_path.exists():
                    backup_path = default_path.with_suffix('.backup.json')
                    shutil.copyfile(default_path, backup_path)
                # Copy the new file to the default location
                shutil.copyfile(filename, default_path)
                
        self.activateWindow()
        self.raise_()

    def liferot_setup(self):
        if len(self.lifetime_spectrum) < 2:
            return False
        if len(self.rotation_spectrum) < 2:
            return False
        return True

    def update_ui_for_jordi(self):
        """Update the UI based on the is_jordi flag."""
        _cs_mod = getattr(chisurf, "cs", object())
        _current_setup = getattr(_cs_mod, "current_setup", None)
        is_jordi = getattr(_current_setup, "is_jordi", False)
        
        if is_jordi:
            # For Jordi format, disable and hide the VH file input fields
            # and update the labels to indicate that one file contains both VV and VH
            self.lineEdit_3.setEnabled(False)
            self.lineEdit_4.setEnabled(False)
            self.lineEdit_3.setVisible(False)
            self.lineEdit_4.setVisible(False)
            
            # Also hide the labels for these fields
            for label in self.findChildren(QtWidgets.QLabel):
                if label.text() == "IRF VH:":
                    label.setVisible(False)
                elif label.text() == "Data VH:":
                    label.setVisible(False)
                elif label.text() == "IRF VV:":
                    label.setText("IRF (VV+VH):")
                elif label.text() == "Data VV:":
                    label.setText("Data (VV+VH):")
        else:
            # For regular format, ensure all fields are enabled and visible
            self.lineEdit_3.setEnabled(True)
            self.lineEdit_4.setEnabled(True)
            self.lineEdit_3.setVisible(True)
            self.lineEdit_4.setVisible(True)
            
            # Restore the original labels
            for label in self.findChildren(QtWidgets.QLabel):
                if label.text() == "IRF (VV+VH):":
                    label.setText("IRF VV:")
                elif label.text() == "Data (VV+VH):":
                    label.setText("Data VV:")
                elif label.text() == "IRF VH:":
                    label.setVisible(True)
                elif label.text() == "Data VH:":
                    label.setVisible(True)
    
    def connect_actions(self):
        self.button(QtWidgets.QWizard.NextButton).clicked.connect(self.page_actions)
        self.button(QtWidgets.QWizard.BackButton).clicked.connect(self.page_actions)
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

    @chisurf.gui.decorators.init_with_ui("fluorescence_decay/tr_anisotropy/wizard.ui", path=chisurf.core.settings.plugin_path)
    def __init__(self, *args, **kwargs):
        self.irf_bg_range_plot = pg.PlotWidget()
        self.region = pg.LinearRegionItem()
        self.correction_factors = {}
        self.data_loaded = False

        self.fit_vv: chisurf.core.fitting.fit.Fit = None
        self.fit_vh: chisurf.core.fitting.fit.Fit = None
        self.global_fit: chisurf.core.fitting.fit.Fit = None

        fn = chisurf.core.settings.chisurf_settings_path / "anisotropy_corrections.json"
        logging.info(f"anisotropy_corrections: {fn}")
        self.conf_edit = chisurf.gui.widgets.parameter_editor.ParameterEditor(
            target=self.correction_factors,
            json_file=fn
        )

        self.verticalLayout_2.addWidget(self.conf_edit)

        self.init_widgets()
        self.connect_actions()
        
        # Update the UI based on the is_jordi flag
        self.update_ui_for_jordi()
        
        # Copy default wizard.spk.json to user folder and load from there
        try:
            # Get the path to the plugin directory
            plugin_path = pathlib.Path(__file__).parent
            # Define the path to the original wizard.spk.json file
            original_spk_json_path = plugin_path / "wizard.spk.json"
            
            # Get the user plugin settings path
            user_plugin_path = self.get_user_plugin_settings_path()
            # Define the path to the user's wizard.spk.json file
            user_spk_json_path = user_plugin_path / "wizard.spk.json"
            
            # Copy the file to the user settings directory only if it doesn't already exist
            if original_spk_json_path.exists() and not user_spk_json_path.exists():
                shutil.copyfile(original_spk_json_path, user_spk_json_path)
                logging.info(f"Copied default settings to {user_spk_json_path}")
            
            # Load the settings from the user's file
            if user_spk_json_path.exists():
                self.onLoadLifetimes(filename=user_spk_json_path)
                logging.info(f"Loaded default settings from {user_spk_json_path}")
        except Exception as e:
            logging.info(f"Error loading default settings: {e}")


if __name__ == "plugin":
    wizard = ChisurfWizard()
    wizard.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wizard = ChisurfWizard()
    wizard.show()
    sys.exit(app.exec_())
