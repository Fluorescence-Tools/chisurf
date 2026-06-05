from __future__ import annotations

import chisurf.core.settings
from chisurf.core.experiments.tcspc.reader import TCSPCReader
import chisurf.core.experiments.tcspc.reader as reader_module


def _make_reader(**kwargs) -> TCSPCReader:
    defaults = {
        'dt': 0.01,
        'rep_rate': 80.0,
        'fit_area': 0.9,
        'fit_start_fraction': 0.5,
        'fit_count_threshold': 8.0,
    }
    defaults.update(kwargs)
    return TCSPCReader(**defaults)


def test_reader_defaults_include_l1_l2(monkeypatch):
    monkeypatch.setattr(
        chisurf.core.settings,
        'anisotropy',
        {'g_factor': 1.234, 'l1': 0.012, 'l2': 0.034},
        raising=False,
    )

    reader = _make_reader()

    assert reader.g_factor == 1.234
    assert reader.l1 == 0.012
    assert reader.l2 == 0.034


def test_reader_passes_and_preserves_calibration_metadata(monkeypatch):
    captured = {}

    class _Curve:
        def __init__(self):
            self.meta_data = {}

    class _Group(list):
        def __init__(self):
            super().__init__([_Curve()])
            self.meta_data = {'g_factor': 9.99, 'anisotropy_calibration_source': 'file_metadata'}

    def _fake_read_tcspc_csv(**kwargs):
        captured.update(kwargs)
        return _Group()

    monkeypatch.setattr(reader_module.tcspc_io, 'read_tcspc_csv', _fake_read_tcspc_csv)

    reader = _make_reader(g_factor=1.1, l1=0.2, l2=0.3)
    result = reader.read(filename='dummy.csv', reading_routine='csv')

    assert captured['g_factor'] == 1.1
    assert captured['l1'] == 0.2
    assert captured['l2'] == 0.3

    # Existing values are preserved, missing ones are filled in.
    assert result.meta_data['g_factor'] == 9.99
    assert result.meta_data['l1'] == 0.2
    assert result.meta_data['l2'] == 0.3
    assert result.meta_data['anisotropy_calibration_source'] == 'file_metadata'

    curve_meta = result[0].meta_data
    assert curve_meta['g_factor'] == 9.99
    assert curve_meta['l1'] == 0.2
    assert curve_meta['l2'] == 0.3
    assert curve_meta['anisotropy_calibration_source'] == 'file_metadata'
