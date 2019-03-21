#!/usr/bin/env python3

# std
import unittest

# ours
from bclustering.maths.metric import *
from bclustering.util.testing import MyTestCase


class TestMetric(MyTestCase):
    def setUp(self):
        self.d_matrix = np.array([
            [0, 2, 3],
            [2, 0, 1],
            [3, 1, 0]
        ])
        self.d_matrix_condensed = np.array([
            2, 3, 1
        ])

    def test_condense_distance_matrix(self):
        self.assertAllClose(
            condense_distance_matrix(self.d_matrix),
            self.d_matrix_condensed
        )
        self.assertAllClose(
            condense_distance_matrix(self.d_matrix),
            self.d_matrix[np.triu_indices(len(self.d_matrix), k=1)]
        )

    def test_uncodense_distance_matrix(self):
        self.assertAllClose(
            uncondense_distance_matrix(self.d_matrix_condensed),
            self.d_matrix
        )


if __name__ == "__main__":
    unittest.main()
