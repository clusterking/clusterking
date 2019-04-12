#!/usr/bin/env python3

# std
from pathlib import Path
import unittest

# ours
from clusterking.util.log import silence_all_logs
from clusterking.util.testing import MyTestCase
from clusterking.data.data import Data


class TestData(MyTestCase):
    def setUp(self):
        silence_all_logs()
        self.ddir = Path(__file__).parent / "data"
        self.dname = "test_scan"
        self.data = [[100, 200], [400, 500]]

    def test_norms(self):
        self.assertAllClose(
            Data(self.ddir, self.dname).norms(),
            [300, 900]
        )

    def test_data(self):
        self.assertAllClose(
            Data(self.ddir, self.dname).data(),
            self.data
        )

    def test_data_normed(self):
        self.assertAllClose(
            Data(self.ddir, self.dname).data(normalize=True),
            [[1/3, 2/3], [4/9, 5/9]]
        )

    def test_only_bpoints(self):
        d = Data(self.ddir, self.dname)
        e = d.only_bpoints()
        self.assertAllClose(e.data(), [[400, 500]])
        d.only_bpoints(inplace=True)
        self.assertAllClose(d.data(), [[400, 500]])


if __name__ == "__main__":
    unittest.main()
