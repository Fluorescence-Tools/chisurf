import utils
import os
import unittest
import tempfile
import argparse
import numpy as np

import chisurf.fio.fluorescence.fcs
import chisurf.cmd_tools.fcs_convert


class Tests(unittest.TestCase):

    def test_fcs_convert(self):
        test_files = [
            {
                'filename': './test/data/fcs/kristine/Kristine_with_error.cor',
                'reader_name': 'kristine',
            },
            {
                'filename': './test/data/fcs/kristine/Kristine_without_error.cor',
                'reader_name': 'kristine',
            }
        ]
        data_types = set([k['reader_name'] for k in test_files])
        for k in test_files:
            curve_ref = chisurf.fio.fluorescence.fcs.read_fcs(**k)
            for dt in data_types:
                _, output_filename = tempfile.mkstemp(
                    suffix='.tmp'
                )
                args = {
                        'input_filename': k['filename'],
                        'input_type': k['reader_name'],
                        'output_filename': output_filename,
                        'output_type': dt,
                        'skiprows': 0,
                        'use_header': False
                }
                chisurf.cmd_tools.fcs_convert.main(
                    args=argparse.Namespace(**args)
                )

                args = dict()
                args['filename'] = output_filename
                args['reader_name'] = dt
                curve = chisurf.fio.fluorescence.fcs.read_fcs(
                    **args
                )
                self.assertEqual(
                    np.allclose(
                        curve_ref.x,
                        curve.x
                    ),
                    True
                )
                self.assertEqual(
                    np.allclose(
                        curve_ref.y,
                        curve.y
                    ),
                    True
                )
