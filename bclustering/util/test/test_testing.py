#!/usr/bin/env python3

# std
import pathlib
import unittest

# ours
import bclustering.util.testing


class TestTestingEnvVariable(unittest.TestCase):

    def test_true(self):
        bclustering.util.testing.set_testing_mode(True)
        self.assertTrue(bclustering.util.testing.is_testing_mode())

    def test_false(self):
        bclustering.util.testing.set_testing_mode(False)
        self.assertFalse(bclustering.util.testing.is_testing_mode())

    def tearDown(self):
        bclustering.util.testing.set_testing_mode(False)


class TestTestJupyter(unittest.TestCase):

    def setUp(self):
        this_dir = pathlib.Path(__file__).resolve().parent
        self.jupyter_dir = this_dir / "jupyter"

    def test_failing(self):
        passed = False
        try:
            bclustering.util.testing.test_jupyter_notebook(
                self.jupyter_dir / "exception.ipynb"
            )
        except Exception:
            passed = True
        if not passed:
            raise Exception("No exception raised")

    def test_passing(self):
        bclustering.util.testing.test_jupyter_notebook(
            self.jupyter_dir / "hello_world.ipynb"
        )


if __name__ == "__main__":
    unittest.main()
