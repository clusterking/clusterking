#!/usr/bin/env python3

# std
import unittest
from pathlib import Path

# ours
from bclustering.util.testing import MyTestCase
from bclustering.data.data import Data


class TestData(MyTestCase):
    def setUp(self):
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
        self.assertAllClose(
            Data(self.ddir, self.dname).data(normalize=True),
            [[1/3, 2/3], [4/9, 5/9]]
        )


if __name__ == "__main__":
    unittest.main()