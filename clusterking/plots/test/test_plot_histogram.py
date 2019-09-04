#!/usr/bin/env python3

# std
import unittest

# 3rd
import numpy as np

# ours
from clusterking.plots.plot_histogram import plot_hist_with_mean


class TestPlotHistWithMean(unittest.TestCase):
    def test(self):
        plot_hist_with_mean(np.random.normal(size=100))


if __name__ == "__main__":
    unittest.main()
