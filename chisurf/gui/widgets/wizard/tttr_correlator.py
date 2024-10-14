import os
import pathlib
import typing

import tttrlib
import json
import numpy as np

import pyqtgraph as pg

import chisurf.fio as io
import chisurf.gui.decorators
from chisurf.gui import QtGui, QtWidgets, QtCore, uic

colors = chisurf.settings.gui['plot']['colors']


class WizardTTTRCorrelator(QtWidgets.QWizardPage):

    @property
    def analysis_folder(self):
        return pathlib.Path(self.lineEdit_3.text())

    @property
    def output_path(self):
        return pathlib.Path(self.lineEdit_5.text())

    @property
    def channel_a(self) -> typing.List[int]:
        s = self.lineEdit.text()
        if len(s) > 0:
            return [int(x) for x in s.split(',')]
        elif isinstance(self.tttr, tttrlib.TTTR):
            return self.tttr.get_used_routing_channels()
        else:
            return []

    @property
    def channel_b(self) -> typing.List[int]:
        s = self.lineEdit_2.text()
        if len(s) > 0:
            return [int(x) for x in s.split(',')]
        elif isinstance(self.tttr, tttrlib.TTTR):
            return self.tttr.get_used_routing_channels()
        else:
            return []

    @property
    def filter_file(self):
        return self.lineEdit_4.text()

    @property
    def filter_enabled(self):
        return pathlib.Path(self.filter_file).exists()

    @property
    def correlation_nbins(self):
        return self.spinBox_2.value()

    @property
    def correlation_ncasc(self):
        return self.spinBox_3.value()

    @property
    def correlation_is_fine(self) -> bool:
        return bool(self.checkBox_2.isChecked())

    @property
    def correlation_nsplits(self):
        return self.spinBox.value()

    @property
    def target_path(self) -> pathlib.Path:
        return pathlib.Path(self.lineEdit_5.text())

    @property
    def microtime_range_a(self):
        s = str(self.lineEdit_6.text())
        return self.get_microtime_ranges(s)

    @property
    def microtime_range_b(self):
        s = str(self.lineEdit_7.text())
        return self.get_microtime_ranges(s)

    def get_microtime_ranges(self, s) -> typing.Optional[typing.List[typing.Tuple[int, int]]]:
        chisurf.logging.log(0, "WizardTTTRCorrelator::get_microtime_ranges")
        # Check if the input string is empty
        if not s:
            chisurf.logging.log(0, "::microtime_ranges: Warning - Input string is empty.")
            return None

        try:
            ranges = [tuple(map(int, item.split('-'))) for item in s.split(';')]

            # Check if each range has exactly two values
            if all(len(r) == 2 for r in ranges):
                return ranges
            else:
                chisurf.logging.log(1, "::microtime_ranges: Invalid format for microsecond ranges.")
                return None
        except (ValueError, TypeError):
            chisurf.logging.log(1, "::microtime_ranges: Invalid values in microsecond ranges.")
            return None

    def update_plots(self):
        chisurf.logging.log(0, 'WizardTTTRCorrelator::Updating plots')
        self.pw_fcs.clear()
        for i, cor in enumerate(self.correlations):
            pen = pg.mkPen(chisurf.settings.colors[i % len(chisurf.settings.colors)]['hex'], width=1)
            self.plot_item_fcs.plot(x=cor['x'], y=cor['y'], pen=pen)

    def read_tttrs(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::read_tttrs")
        fn = self.current_tttr_filename
        if fn:
            if pathlib.Path(fn).exists():
                n = len(self.settings['tttr_filenames'])
                self.spinBox_4.setMaximum(n - 1)
                self.comboBox.setEnabled(False)
                self.tttr = tttrlib.TTTR(fn, self.filetype)
                header = self.tttr.get_header()
                s = header.json
                d = json.loads(s)
                self.settings['header'] = d
                self.update_plots()

    def update_output_path(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::update_output_path")
        if len(self.channel_a) > 0 and len(self.channel_b) > 0:
            cha = ','.join([str(x) for x in self.channel_a])
            chb = ','.join([str(x) for x in self.channel_b])
            chs = cha + '-' + chb
        else:
            chs = 'All'
        s = pathlib.Path('cr5/') / f'{chs}'
        self.lineEdit_5.setText(s.as_posix())

    def update_parameter(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::update_parameter")
        self.settings['correlation']['is_fine'] = self.correlation_is_fine
        self.settings['correlation']['ncasc'] = self.correlation_ncasc
        self.settings['correlation']['nbins'] = self.correlation_nbins
        self.settings['correlation']['nsplits'] = self.correlation_nsplits
        self.settings['correlation']['channel_a'] = self.channel_a
        self.settings['correlation']['channel_b'] = self.channel_b
        self.settings['correlation']['filter'] = self.filter_file

        self.update_plots()
        self.update_output_path()

    def onClearFiles(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::onClearFiles")
        self.settings['tttr_filenames'].clear()
        self.comboBox.setEnabled(True)
        self.lineEdit.clear()
        self.tttr = None

    def split_array(self, tttr, n):
        chisurf.logging.log(0, "WizardTTTRCorrelator::split_array")
        chunk_size = len(tttr) // n
        chunks = [tttr[i * chunk_size: (i + 1) * chunk_size] for i in range(n)]
        return chunks

    def get_correlation_settings(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::Getting correlation settings")
        d = {
            "n_bins": self.correlation_nbins,
            "n_casc": self.correlation_ncasc,
            "make_fine": self.correlation_is_fine
        }
        chisurf.logging.log(0, "Correlation settings:", d)
        return d

    def save_correlations(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::saving correlations to files")
        output_folder = self.analysis_folder / self.output_path
        output_folder.mkdir(parents=True, exist_ok=True)
        for i, cor in enumerate(self.correlations):
            json_path = output_folder / f'chnk-{i:04}.json.gz'
            with io.open_maybe_zipped(json_path, 'w') as fp:
                json.dump(cor, fp)

    def correlate_data(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::Correlate data")
        n_chunks = self.correlation_nsplits

        ch1 = self.channel_a
        ch2 = self.channel_b
        chisurf.logging.log(0, "ch1", ch1)
        chisurf.logging.log(0, "ch2", ch2)
        chisurf.logging.log(0, "n_chunks", n_chunks)
        chisurf.logging.log(0, "self.tttr:", self.tttr)

        correlation_settings = self.get_correlation_settings()
        correlations = self.correlations
        correlations.clear()
        for i, tttr in enumerate(self.split_array(self.tttr, n_chunks)):
            t = tttr.macro_times

            # Select based on channels
            mask_a = tttrlib.TTTRMask()
            mask_b = tttrlib.TTTRMask()
            mask_a.select_channels(tttr, ch1, mask=True)
            mask_b.select_channels(tttr, ch2, mask=True)
            m_a = np.array(mask_a.get_mask(), dtype=bool)
            m_b = np.array(mask_b.get_mask(), dtype=bool)

            if self.microtime_range_a:
                mask_a = tttrlib.TTTRMask()
                mask_a.select_microtime_ranges(tttr, self.microtime_range_a)
                mask_a.flip()
                ma2 = np.array(mask_a.get_mask(), dtype=bool)
                m_a = np.logical_and(m_a, ma2)
            if self.microtime_range_b:
                mask_b = tttrlib.TTTRMask()
                mask_b.select_microtime_ranges(tttr, self.microtime_range_b)
                mask_b.flip()
                mb2 = np.array(mask_b.get_mask(), dtype=bool)
                m_b = np.logical_and(m_b, mb2)

            w1 = np.array(m_a, dtype=np.float64)
            w2 = np.array(m_b, dtype=np.float64)
            sw1, sw2 = sum(w1), sum(w2)
            dT = tttr.header.macro_time_resolution
            dur = tttr.macro_times[-1] * dT  # seconds
            if sw1 > 0.0 and sw2 > 0.0:
                correlator = tttrlib.Correlator(**correlation_settings)
                correlator.set_macrotimes(t, t)
                correlator.set_weights(w1, w2)
                x = correlator.x_axis * dT
                if self.correlation_is_fine:
                    n_microtime_channels = tttr.get_number_of_micro_time_channels()
                    mt = tttr.micro_times
                    correlator.set_microtimes(mt, mt, n_microtime_channels)
                    x *= tttr.header.micro_time_resolution
                d = {
                    'x': x.tolist(),
                    'y': correlator.correlation.tolist(),
                    'correlation_settings': correlation_settings,
                    'analysis_folder': self.analysis_folder.as_posix(),
                    'chunk': i,
                    'duration': dur,
                    'channel_a': {
                        'channels': ch1,
                        'microtime_range': self.microtime_range_a,
                        'counts': sw1
                    },
                    'channel_b': {
                        'channels': ch2,
                        'microtime_range': self.microtime_range_b,
                        'counts': sw2
                    }
                }
                correlations.append(d)
            else:
                chisurf.logging.log(1, "Warning: no photons to correlate with.")
        self.update_plots()
        self.save_correlations()

    def open_sl5(self, filename: str) -> tttrlib.TTTR:
        chisurf.logging.log(0, 'WizardTTTRCorrelator::open_sl5:', filename)
        data = dict()
        with io.open_maybe_zipped(filename) as fp:
            data.update(json.load(fp))
        tttr_filename = self.analysis_folder / pathlib.Path(data['filename'])
        tttr_filetype = data['filetype']
        f = chisurf.fio.decompress_numpy_array(data['filter'])
        idx = np.where(f > 0)[0]
        if tttr_filename.exists():
            chisurf.logging.log(0, 'tttr_filename: ', tttr_filename)
            chisurf.logging.log(0, 'tttr_filetype: ', tttr_filetype)
            tttr = tttrlib.TTTR(tttr_filename.as_posix(), tttr_filetype)
            tttr = tttr[idx]
            chisurf.logging.log(0, tttr.get_macro_times())
        return tttr

    def open_selections(self, filenames: typing.List[pathlib.Path]) -> tttrlib.TTTR:
        chisurf.logging.log(0, "WizardTTTRCorrelator::open_selections:", filenames)
        self.tttr = self.open_sl5(filenames[0])
        for filename in filenames:
            self.tttr.append(self.open_sl5(filename))
        chisurf.logging.log(0, "tttr", self.tttr)

    def open_analysis_folder(self, folder: pathlib.Path = None):
        chisurf.logging.log(0, "WizardTTTRCorrelator::open_analysis_folder")
        if folder is None:
            folder = self.analysis_folder / 'sl5'
        selected_files = sorted(list(folder.glob('*.json.gz')))
        chisurf.logging.log(0, 'Opening analysis folder')
        chisurf.logging.log(0, list(selected_files))
        self.open_selections(selected_files)

    @chisurf.gui.decorators.init_with_ui("tttr_correlator.ui")
    def __init__(self, *args, **kwargs):
        self.setTitle("Correlator")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sizePolicy)

        self.tttr: tttrlib.TTTR = None
        self.settings: dict = dict()
        self.settings['correlation'] = dict()
        self.correlations = list()

        self.textEdit.setVisible(False)
        chisurf.gui.decorators.lineEdit_dragFile_injector(self.lineEdit_3, call=self.open_analysis_folder)

        # Create plots
        self.pw_fcs = pg.plot()
        self.pw_fcs.resize(150, 150)
        self.plot_item_fcs = self.pw_fcs.getPlotItem()
        self.plot_item_fcs.setLogMode(True, False)
        self.verticalLayout_2.addWidget(self.pw_fcs)

        # Connect actions
        self.actionUpdate_ouput_path.triggered.connect(self.update_output_path)
        self.toolButton_3.clicked.connect(self.correlate_data)
        self.toolButton_4.clicked.connect(self.onClearFiles)

        self.update_parameter()

