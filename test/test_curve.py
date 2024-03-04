import utils
import unittest
import pathlib

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

import numpy as np
import tempfile

import chisurf.curve


class Tests(unittest.TestCase):

    def test_curve_init(self):
        c1 = chisurf.curve.Curve()
        self.assertEqual(
            np.array_equal(c1.x, np.array([], dtype=np.float64)),
            True
        )
        self.assertEqual(
            np.array_equal(c1.y, np.array([], dtype=np.float64)),
            True
        )
        self.assertEqual(c1.name, 'Curve')

        c1 = chisurf.curve.Curve()
        x = np.linspace(0, 2. * np.pi, 100)
        y = np.sin(x)
        c1.x = x
        c1.y = y

        name = 'C2'
        c2 = chisurf.curve.Curve(x, y, name=name)
        self.assertEqual(
            np.array_equal(c1.x, c2.x),
            True
        )
        self.assertEqual(
            np.array_equal(c1.y, c2.y),
            True
        )
        self.assertEqual(
            c2.name,
            name
        )
        c3 = chisurf.curve.Curve()
        c3.from_json(
            json_string=c2.to_json()
        )

        c2_dict = c2.to_dict()
        c3_dict = c2.to_dict()
        for k in c2_dict:
            if isinstance(c2_dict[k], np.ndarray):
                self.assertEqual(
                    np.allclose(
                        c2_dict[k],
                        c3_dict[k]
                    ),
                    True
                )
            else:
                self.assertEqual(
                    c2_dict[k],
                    c3_dict[k]
                )

        # test no copy option
        x = np.linspace(0, 2.0 * np.pi, 20)
        y = np.sin(x)
        c4 = chisurf.curve.Curve(
            x=x,
            y=y,
            copy_array=False
        )
        c4.x[0] = 11
        self.assertEqual(
            x[0],
            11
        )

        # test curve shift
        c5 = c4 << 1.0
        self.assertEqual(
            type(c5),
            type(c4)
        )

        # test length
        self.assertEqual(
            len(c5),
            len(c5.y)
        )

        # test getitem
        x, y = c5[:5]
        self.assertEqual(
            np.allclose(
                y,
                c5.y[:5]
            ),
            True
        )
        self.assertEqual(
            np.allclose(
                x,
                c5.x[:5]
            ),
            True
        )

    def test_attributes(self):
        import scipy.stats
        import chisurf.math.signal

        x = np.linspace(0, 10, 100)
        y = scipy.stats.distributions.norm.pdf(
            x,
            loc=5,
            scale=2
        )
        c2 = chisurf.curve.Curve(x, y)
        self.assertEqual(
            c2.fwhm,
            chisurf.math.signal.calculate_fwhm(
                x_values=c2.x, y_values=c2.y
            )[0]
        )

        cdf = c2.cdf
        self.assertEqual(
            np.array_equal(
                cdf.y,
                np.cumsum(c2.y)
            ),
            True
        )

        self.assertEqual(
            np.array_equal(
                cdf.dx,
                np.diff(c2.x)
            ),
            True
        )

    def test_reading(self):
        _, filename = tempfile.mkstemp(
            suffix='.yaml'
        )

        x = np.linspace(0, 2. * np.pi, 100)
        y = np.sin(x)
        c1 = chisurf.curve.Curve(x, y)

        c1.save(
            filename=filename,
            file_type='yaml'
        )

        c2 = chisurf.curve.Curve()
        c2.load(
            filename=filename,
            file_type='yaml'
        )
        self.assertEqual(
            np.array_equal(c2.x, c1.x),
            True
        )
        self.assertEqual(
            np.array_equal(c2.y, c1.y),
            True
        )
        self.assertEqual(
            c2.unique_identifier,
            c1.unique_identifier
        )

    def test_normalize(self):
        import scipy.stats

        x = np.linspace(0, 10, 11)
        c1 = chisurf.curve.Curve(
            x,
            scipy.stats.distributions.norm.pdf(x, loc=2, scale=1)
        )
        factor = c1.normalize()
        self.assertAlmostEqual(
            max(c1.y),
            1.0
        )
        self.assertAlmostEqual(
            factor,
            0.3989422804014327
        )

        factor = c1.normalize(
            mode='sum'
        )
        self.assertAlmostEqual(
            sum(c1.y),
            1.0
        )

    def test_calc(self):
        import scipy.stats

        x = np.linspace(0, 10, 11)
        c1 = chisurf.curve.Curve(
            x,
            scipy.stats.distributions.norm.pdf(x, loc=5, scale=2)
        )
        c2 = chisurf.curve.Curve(
            x,
            scipy.stats.distributions.norm.pdf(x, loc=2, scale=1)
        )

        c3 = c1 + c2
        self.assertEqual(
            np.array_equal(
                c3.y,
                c1.y + c2.y
            ),
            True
        )

        c3 = c1 - c2
        self.assertEqual(
            np.array_equal(
                c3.y,
                c1.y - c2.y
            ),
            True
        )

        c3 = c1 * c2
        self.assertEqual(
            np.array_equal(
                c3.y,
                c1.y * c2.y
            ),
            True
        )

        c3 = c1 / c2
        self.assertEqual(
            np.array_equal(
                c3.y,
                c1.y / c2.y
            ),
            True
        )


if __name__ == '__main__':
    unittest.main()
