import utils
import os
import unittest
import pathlib

TOPDIR = pathlib.Path(__file__).parent.parent

utils.set_search_paths(TOPDIR)

import numpy as np

import chisurf.math.regularization as regularization


class Tests(unittest.TestCase):

    def test_csvd(self):
        """Test compact SVD function."""
        A = np.random.randn(10, 5)
        U, s, V = regularization.csvd(A)
        self.assertEqual(U.shape, (10, 5))
        self.assertEqual(s.shape, (5,))
        self.assertEqual(V.shape, (5, 5))

        # Check reconstruction
        A_reconstructed = U @ np.diag(s) @ V.T
        np.testing.assert_allclose(A, A_reconstructed, rtol=1e-10)

    def test_tikhonov(self):
        """Test Tikhonov regularization."""
        np.random.seed(42)  # For reproducibility
        A = np.random.randn(10, 5)
        x_true = np.random.randn(5)
        b = A @ x_true + 0.1 * np.random.randn(10)  # Add noise

        U, s, V = regularization.csvd(A)
        x_reg, rho, eta = regularization.tikhonov(U, s, V, b, 0.1)

        self.assertEqual(x_reg.shape, (5,))
        self.assertIsInstance(rho, (float, np.ndarray))
        self.assertIsInstance(eta, (float, np.ndarray))

        # Check that regularization reduces norm
        x_unreg = np.linalg.pinv(A) @ b
        self.assertLess(np.linalg.norm(x_reg), np.linalg.norm(x_unreg))

    def test_tsvd(self):
        """Test Truncated SVD regularization."""
        np.random.seed(42)
        A = np.random.randn(10, 5)
        x_true = np.random.randn(5)
        b = A @ x_true + 0.1 * np.random.randn(10)

        U, s, V = regularization.csvd(A)
        x_reg, rho, eta = regularization.tsvd(U, s, V, b, 3)

        self.assertEqual(x_reg.shape, (5,))
        self.assertIsInstance(rho, float)
        self.assertIsInstance(eta, float)

    def test_gcv(self):
        """Test Generalized Cross-Validation."""
        np.random.seed(42)
        A = np.random.randn(10, 5)
        b = np.random.randn(10)

        U, s, _ = regularization.csvd(A)
        reg_min, G, reg_param = regularization.gcv(U, s, b)

        self.assertIsInstance(reg_min, float)
        self.assertEqual(len(G), len(reg_param))

    def test_l_curve(self):
        """Test L-curve computation."""
        np.random.seed(42)
        A = np.random.randn(10, 5)
        b = np.random.randn(10)

        U, s, _ = regularization.csvd(A)
        rho, eta, k = regularization.l_curve(U, s, b)

        self.assertEqual(len(rho), len(eta))
        self.assertIsInstance(k, (int, type(None)))

    def test_corner(self):
        """Test corner detection."""
        # Create synthetic L-curve data
        rho = np.logspace(1, -1, 10)
        eta = np.logspace(1, -1, 10)[::-1]

        k = regularization.corner(rho, eta)
        self.assertIsInstance(k, (int, type(None)))


if __name__ == '__main__':
    unittest.main()
