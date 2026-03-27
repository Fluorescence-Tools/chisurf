import unittest
import pathlib

import utils

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

from chisurf.macros.core_fit import (
    _apply_anisotropy_calibration_to_fit,
    _resolve_dataset_anisotropy_calibration,
)


class _Param:
    def __init__(self, value=0.0):
        self.value = float(value)


class _Anisotropy:
    def __init__(self):
        self._g = _Param(0.0)
        self._l1 = _Param(0.0)
        self._l2 = _Param(0.0)


class _Model:
    def __init__(self):
        self.anisotropy = _Anisotropy()


class _Fit:
    def __init__(self):
        self.model = _Model()


class _FitGroup:
    def __init__(self):
        self.model = _Model()
        self.grouped_fits = [_Fit(), _Fit()]


class _Reader:
    def __init__(self, g_factor=None, l1=None, l2=None):
        self.g_factor = g_factor
        self.l1 = l1
        self.l2 = l2


class _Curve:
    def __init__(self, meta_data=None, data_reader=None):
        self.meta_data = meta_data or {}
        self.data_reader = data_reader


class _Group(list):
    def __init__(self, curves, meta_data=None, data_reader=None):
        super().__init__(curves)
        self.meta_data = meta_data or {}
        self.data_reader = data_reader


class CoreFitAnisotropyCalibrationTests(unittest.TestCase):

    def test_resolve_dataset_anisotropy_calibration_uses_reader_and_metadata(self):
        # Reader provides g_factor and l1, metadata provides l2 fallback.
        reader = _Reader(g_factor=1.23, l1=0.11, l2=None)
        curve = _Curve(meta_data={'l2': 0.22}, data_reader=reader)
        group = _Group([curve], meta_data={}, data_reader=reader)

        calibration = _resolve_dataset_anisotropy_calibration(group)

        self.assertAlmostEqual(calibration['g_factor'], 1.23)
        self.assertAlmostEqual(calibration['l1'], 0.11)
        self.assertAlmostEqual(calibration['l2'], 0.22)

    def test_apply_anisotropy_calibration_updates_top_and_grouped_fits(self):
        fit_group = _FitGroup()

        _apply_anisotropy_calibration_to_fit(
            fit_group,
            {
                'g_factor': 1.45,
                'l1': 0.015,
                'l2': 0.025,
            }
        )

        # Top fit
        self.assertAlmostEqual(fit_group.model.anisotropy._g.value, 1.45)
        self.assertAlmostEqual(fit_group.model.anisotropy._l1.value, 0.015)
        self.assertAlmostEqual(fit_group.model.anisotropy._l2.value, 0.025)

        # Grouped fits
        for member in fit_group.grouped_fits:
            self.assertAlmostEqual(member.model.anisotropy._g.value, 1.45)
            self.assertAlmostEqual(member.model.anisotropy._l1.value, 0.015)
            self.assertAlmostEqual(member.model.anisotropy._l2.value, 0.025)

    def test_resolve_dataset_anisotropy_calibration_ignores_non_finite_values(self):
        # Non-finite reader values should be ignored to avoid feeding invalid
        # model kwargs into anisotropy parameter construction.
        reader = _Reader(g_factor=float('nan'), l1=float('inf'), l2=0.22)
        group = _Group([_Curve(meta_data={})], meta_data={}, data_reader=reader)

        calibration = _resolve_dataset_anisotropy_calibration(group)

        self.assertIsNone(calibration['g_factor'])
        self.assertIsNone(calibration['l1'])
        self.assertAlmostEqual(calibration['l2'], 0.22)

    def test_apply_anisotropy_calibration_ignores_non_finite_values(self):
        fit_group = _FitGroup()

        # Seed baseline values so we can assert they are unchanged.
        fit_group.model.anisotropy._g.value = 1.0
        fit_group.model.anisotropy._l1.value = 0.003
        fit_group.model.anisotropy._l2.value = 0.004

        _apply_anisotropy_calibration_to_fit(
            fit_group,
            {
                'g_factor': float('nan'),
                'l1': float('inf'),
                'l2': 0.02,
            }
        )

        self.assertAlmostEqual(fit_group.model.anisotropy._g.value, 1.0)
        self.assertAlmostEqual(fit_group.model.anisotropy._l1.value, 0.003)
        self.assertAlmostEqual(fit_group.model.anisotropy._l2.value, 0.02)


if __name__ == '__main__':
    unittest.main()
