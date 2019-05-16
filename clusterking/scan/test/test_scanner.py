#!/usr/bin/env python3

# std
import unittest
from pathlib import Path
import tempfile

# 3rd
import numpy as np

# ours
from clusterking.util.testing import MyTestCase
from clusterking.scan.scanner import Scanner
from clusterking.data.data import Data


def func_zero(coeffs):
    return 0.


def func_identity(coeffs):
    return coeffs


def func_zero_bins(coeffs, x):
    return coeffs


class TestScanner(MyTestCase):

    def setUp(self):
        # We also want to test writing, to check that there are e.g. no
        # JSON serialization problems.
        self.tmpdir = tempfile.TemporaryDirectory()

    def cleanUp(self):
        self.tmpdir.cleanup()

    def test_set_spoints_grid(self):
        s = Scanner()
        s.set_spoints_grid({'c': [1, 2], 'a': [3], 'b': [1j, 1+1j]})
        self.assertAllClose(
            s.spoints,
            np.array([
                [3, 1j, 1],
                [3, 1j, 2],
                [3, 1+1j, 1],
                [3, 1+1j, 2],
            ])
        )

    def test_set_spoints_grid_empty(self):
        s = Scanner()
        s.set_spoints_grid({})
        self.assertEqual(len(np.squeeze(s.spoints)), 0)

    def test_set_spoints_equidist(self):
        s = Scanner()
        s.imaginary_prefix = "xxx"
        s.set_spoints_equidist({
            "a": (1, 2, 2),
            "xxxa": (3, 4, 2),
            "c": (1, 1, 1)
        })
        self.assertAllClose(
            s.spoints,
            np.array([
                [1+3j, 1],
                [1+4j, 1],
                [2+3j, 1],
                [2+4j, 1]
            ])
        )

    def test_run_zero(self):
        s = Scanner()
        d = Data()
        s.set_spoints_equidist({"a": (0, 1, 2)})
        s.set_dfunction(func_zero)
        s.run(d)
        self.assertEqual(
            sorted(list(d.df.columns)),
            ["a", "bin0"]
        )
        self.assertAllClose(
            d.df.values,
            np.array([[0., 0.], [1., 0.]])
        )
        d.write(Path(self.tmpdir.name) / "test.sql")

    def test_run_identity(self):
        s = Scanner()
        d = Data()
        s.set_spoints_equidist({"a": (0, 1, 2)})
        s.set_dfunction(func_identity)
        s.run(d)
        self.assertEqual(
            sorted(list(d.df.columns)),
            ["a", "bin0"]
        )
        self.assertAllClose(
            d.df.values,
            np.array([[0., 0.], [1., 1.]])
        )
        d.write(Path(self.tmpdir.name) / "test.sql")

    def test_run_identity_singlecore(self):
        s = Scanner()
        d = Data()
        s.set_spoints_equidist({"a": (0, 1, 2)})
        s.set_dfunction(func_identity)
        s.run(d, 1)
        self.assertEqual(
            sorted(list(d.df.columns)),
            ["a", "bin0"]
        )
        self.assertAllClose(
            d.df.values,
            np.array([[0., 0.], [1., 1.]])
        )
        d.write(Path(self.tmpdir.name) / "test.sql")

    def test_run_simple_bins(self):
        s = Scanner()
        d = Data()
        s.set_spoints_equidist({"a": (0, 1, 2)})
        s.set_dfunction(func_zero_bins, binning=[0, 1, 2])
        s.run(d)
        self.assertEqual(
            sorted(list(d.df.columns)),
            ["a", "bin0", "bin1"]
        )
        self.assertAllClose(
            d.df.values,
            np.array([[0., 0., 0.], [1., 1., 1.]])
        )
        d.write(Path(self.tmpdir.name) / "test.sql")

    def test_run_simple_bins_singlecore(self):
        s = Scanner()
        d = Data()
        s.set_spoints_equidist({"a": (0, 1, 2)})
        s.set_dfunction(func_zero_bins, binning=[0, 1, 2])
        s.run(d, 1)
        self.assertEqual(
            sorted(list(d.df.columns)),
            ["a", "bin0", "bin1"]
        )
        self.assertAllClose(
            d.df.values,
            np.array([[0., 0., 0.], [1., 1., 1.]])
        )
        d.write(Path(self.tmpdir.name) / "test.sql")


if __name__ == "__main__":
    unittest.main()
