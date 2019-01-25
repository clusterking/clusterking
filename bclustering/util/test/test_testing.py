#!/usr/bin/env python3

# std
import unittest

# ours
import bclustering.util.testing


class TestTesting(unittest.TestCase):

    def test_true(self):
        bclustering.util.testing.set_testing_mode(True)
        self.assertTrue(bclustering.util.testing.is_testing_mode())

    def test_false(self):
        bclustering.util.testing.set_testing_mode(False)
        self.assertFalse(bclustering.util.testing.is_testing_mode())

    def tearDown(self):
        bclustering.util.testing.set_testing_mode(False)

if __name__ == "__main__":
    unittest.main()
