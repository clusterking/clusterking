#!/usr/bin/env python3

# std
import pathlib
import unittest

# ours
import clusterking.util.testing


class TestTestingEnvVariable(unittest.TestCase):

    def test_true(self):
        clusterking.util.testing.set_testing_mode(True)
        self.assertTrue(clusterking.util.testing.is_testing_mode())

    def test_false(self):
        clusterking.util.testing.set_testing_mode(False)
        self.assertFalse(clusterking.util.testing.is_testing_mode())

    def tearDown(self):
        clusterking.util.testing.set_testing_mode(False)


class TestTestJupyter(unittest.TestCase):

    def setUp(self):
        this_dir = pathlib.Path(__file__).resolve().parent
        self.jupyter_dir = this_dir / "jupyter"

    def test_failing(self):
        passed = False
        try:
            clusterking.util.testing.test_jupyter_notebook(
                self.jupyter_dir / "exception.ipynb"
            )
        except Exception:
            passed = True
        if not passed:
            raise Exception("No exception raised")

    def test_passing(self):
        clusterking.util.testing.test_jupyter_notebook(
            self.jupyter_dir / "hello_world.ipynb"
        )


if __name__ == "__main__":
    unittest.main()
